//! Fast outcome labels for session lists. Never read LLM call payloads here.
use super::LocalDatabase;
use engine_core::cost_outcome::session_cost_outcome_summary;
use engine_core::error::{EngineError, EngineResult};
use engine_core::event::{SessionEvent, SessionEventType};
use engine_core::{CostOutcome, Session};
use std::collections::HashMap;
use uuid::Uuid;

// Keep this predicate identical to the partial index in migration 23.
const OUTCOME_EVENTS_SQL: &str = "SELECT session_id, seq, event_type, payload, created_at
    FROM session_events
    WHERE event_type IN ('session_completed', 'loop_intervention', 'checkpoint_redeemed')
      AND session_id IN (SELECT value FROM json_each(?1))
    ORDER BY session_id, seq";

impl LocalDatabase {
    /// One indexed read for an entire list, independent of call-log volume.
    pub async fn session_outcome_summaries(
        &self,
        sessions: &[Session],
    ) -> EngineResult<HashMap<Uuid, CostOutcome>> {
        if sessions.is_empty() {
            return Ok(HashMap::new());
        }
        let sessions = sessions.to_vec();
        let ids = serde_json::to_string(&sessions.iter().map(|s| s.id).collect::<Vec<_>>())
            .map_err(|error| EngineError::Storage(error.to_string()))?;
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.read_conn();
            let mut statement = conn
                .prepare(OUTCOME_EVENTS_SQL)
                .map_err(|error| EngineError::Storage(error.to_string()))?;
            let rows = statement
                .query_map([ids], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                })
                .map_err(|error| EngineError::Storage(error.to_string()))?;
            let mut by_session: HashMap<Uuid, Vec<SessionEvent>> = HashMap::new();
            for row in rows {
                let (id, seq, event_type, payload, created_at) =
                    row.map_err(|error| EngineError::Storage(error.to_string()))?;
                let id = Uuid::parse_str(&id)
                    .map_err(|error| EngineError::Storage(error.to_string()))?;
                let event_type = match event_type.as_str() {
                    "session_completed" => SessionEventType::SessionCompleted,
                    "loop_intervention" => SessionEventType::LoopIntervention,
                    "checkpoint_redeemed" => SessionEventType::CheckpointRedeemed,
                    _ => continue,
                };
                by_session.entry(id).or_default().push(SessionEvent {
                    session_id: id,
                    seq,
                    event_type,
                    payload: serde_json::from_str(&payload)
                        .map_err(|error| EngineError::Storage(error.to_string()))?,
                    cost_cents: None,
                    created_at: created_at.parse().map_err(|error: chrono::ParseError| {
                        EngineError::Storage(error.to_string())
                    })?,
                });
            }
            Ok(sessions
                .iter()
                .map(|session| {
                    let events = by_session
                        .get(&session.id)
                        .map(Vec::as_slice)
                        .unwrap_or_default();
                    (session.id, session_cost_outcome_summary(session, events))
                })
                .collect())
        })
        .await
        .map_err(|error| EngineError::Storage(format!("sqlite task join error: {error}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::cost_outcome::SuccessStatus;
    use engine_core::{EventStore, SessionStatus};

    #[tokio::test]
    async fn outcome_summaries_use_sparse_index_and_never_parse_call_payloads() {
        let db = LocalDatabase::open_in_memory().unwrap();
        let mut sessions = Vec::new();
        for number in 0..12 {
            let mut session = Session::new(format!("session {number}"), 1000).unwrap();
            session.status = SessionStatus::Completed;
            db.insert_session(&session).unwrap();
            sessions.push(session);
        }
        {
            let mut conn = db.conn.lock().unwrap();
            let tx = conn.transaction().unwrap();
            // Large, deliberately malformed call JSON detects accidental telemetry reads.
            // The outcome query must not parse or hydrate any of these rows.
            let payload = "unparseable call log ".repeat(1024);
            {
                let mut insert = tx.prepare("INSERT INTO session_events(session_id, seq, event_type, payload, created_at)
                    VALUES (?1, ?2, 'llm_call', ?3, '2026-09-06T00:00:00Z')").unwrap();
                for session in &sessions {
                    for seq in 1..=200 {
                        insert
                            .execute(rusqlite::params![session.id.to_string(), seq, payload])
                            .unwrap();
                    }
                }
            }
            tx.commit().unwrap();
            let ids =
                serde_json::to_string(&sessions.iter().map(|s| s.id).collect::<Vec<_>>()).unwrap();
            let plan: Vec<String> = conn
                .prepare(&format!("EXPLAIN QUERY PLAN {OUTCOME_EVENTS_SQL}"))
                .unwrap()
                .query_map([ids], |row| row.get(3))
                .unwrap()
                .map(Result::unwrap)
                .collect();
            assert!(
                plan.iter()
                    .any(|line| line.contains("idx_session_events_outcomes")),
                "{plan:?}"
            );
            assert!(
                !plan.iter().any(|line| line.contains("SCAN session_events")),
                "{plan:?}"
            );
        }
        db.append_event(
            sessions[0].id,
            SessionEventType::LoopIntervention,
            serde_json::json!({"repetition_count": 2, "injected_message": "retry"}),
            None,
        )
        .await
        .unwrap();
        db.append_event(sessions[0].id, SessionEventType::SessionCompleted,
            serde_json::json!({"outcome_report": {"source": "validator", "verdict": "success", "evidence": "tests passed"}}), None).await.unwrap();
        let start = std::time::Instant::now();
        let outcomes = db.session_outcome_summaries(&sessions).await.unwrap();
        eprintln!(
            "12 session summaries over 2,400 large call rows: {:?}",
            start.elapsed()
        );
        assert_eq!(outcomes.len(), sessions.len());
        assert_eq!(
            outcomes[&sessions[0].id].success_status,
            SuccessStatus::RecoveredSuccess
        );
        assert_eq!(
            outcomes[&sessions[1].id].success_status,
            SuccessStatus::Unknown
        );
        assert!(outcomes
            .values()
            .all(|outcome| outcome.diagnostics.is_none()));
        let scoped = db.session_outcome_summaries(&sessions[1..2]).await.unwrap();
        assert_eq!(scoped.len(), 1);
        assert!(!scoped.contains_key(&sessions[0].id));
    }
}
