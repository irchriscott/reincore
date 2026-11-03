//! Events persistence for the local engine.

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use engine_core::{
    error::{EngineError, EngineResult},
    event::{
        tool_call_links_from_event, EventStore, SessionEvent, SessionEventType, SummaryPayload,
        TaskEventProjection, ToolCallLink,
    },
    models::{FileContextAttachment, ToolCall},
};
use rusqlite::{params, OptionalExtension, TransactionBehavior};
use uuid::Uuid;

use super::LocalDatabase;

pub(super) const FILE_HISTORY_LOOKUP_SQL: &str = "SELECT h.payload,l.call_ambiguous FROM (
                SELECT * FROM (SELECT payload,session_id,tool_call_id,seq FROM session_file_history WHERE session_id=?1 AND file_key=?4 ORDER BY seq DESC,tool_call_id DESC LIMIT ?3)
                UNION ALL SELECT * FROM (SELECT payload,session_id,tool_call_id,seq FROM session_file_history INDEXED BY idx_session_file_history_legacy WHERE session_id=?1 AND path=?2 AND file_key IS NULL ORDER BY seq DESC,tool_call_id DESC LIMIT ?3)) h
                JOIN session_tool_call_links l ON l.session_id=h.session_id AND l.tool_call_id=h.tool_call_id
                ORDER BY h.seq DESC,h.tool_call_id DESC LIMIT ?3";

impl LocalDatabase {
    fn file_history_scope_on(
        conn: &rusqlite::Connection,
        session_id: Uuid,
    ) -> EngineResult<Option<engine_core::file_history::FileHistoryScope>> {
        let mut stmt = conn.prepare("SELECT project_id,root_id,local_path FROM session_file_roots WHERE session_id=?1 ORDER BY root_id,local_path")
            .map_err(|e| EngineError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map([session_id.to_string()], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|e| EngineError::Storage(e.to_string()))?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| EngineError::Storage(e.to_string()))?;
        let Some(first) = rows.first() else {
            return Ok(None);
        };
        Ok(Some(engine_core::file_history::FileHistoryScope {
            project_id: Uuid::parse_str(&first.0)
                .map_err(|e| EngineError::Storage(e.to_string()))?,
            roots: rows
                .into_iter()
                .map(|(_, id, path)| engine_core::file_history::FileRoot { id, path })
                .collect(),
        }))
    }

    pub(super) fn append_event_on(
        tx: &rusqlite::Transaction<'_>,
        session_id: Uuid,
        event_type: SessionEventType,
        mut payload: serde_json::Value,
        cost_cents: Option<i64>,
    ) -> EngineResult<SessionEvent> {
        let raw_links = tool_call_links_from_event(&SessionEvent {
            session_id,
            seq: 0,
            event_type,
            payload: payload.clone(),
            cost_cents,
            created_at: Utc::now(),
        });
        let identities = engine_core::file_history::link_identities(&raw_links);
        engine_core::privacy::sanitize_event_payload(event_type, &mut payload);
        let seq: i64 = tx
            .query_row(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM session_events WHERE session_id = ?1",
                [session_id.to_string()],
                |row| row.get(0),
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;

        let created_at = Utc::now();
        let event = SessionEvent {
            session_id,
            seq,
            event_type,
            payload,
            cost_cents,
            created_at,
        };
        tx.execute(
            "INSERT INTO session_events (
                    session_id, seq, event_type, payload, cost_cents, created_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                session_id.to_string(),
                seq,
                event_type_to_str(event_type),
                event.payload.to_string(),
                cost_cents,
                created_at.to_rfc3339(),
            ],
        )
        .map_err(|e| EngineError::Storage(e.to_string()))?;

        if let Some(entry) = engine_core::continuation::indexed_tool_result(&event) {
            tx.execute("INSERT INTO tool_state_index(embedding,session_id,seq,role,content,tool_call_id,name,created_at)
                VALUES(?1,?2,?3,?4,?5,?6,?7,?8)", params![serde_json::to_string(&entry.embedding).unwrap_or_default(),entry.session_id.to_string(),entry.seq,
                    entry.message.role,entry.message.content_text(),entry.message.tool_call_id,entry.message.name,event.created_at.to_rfc3339()])
                .map_err(|e| EngineError::Storage(e.to_string()))?;
        }
        if let Some(projection) = TaskEventProjection::from_event(&event) {
            tx.execute(
                "INSERT OR REPLACE INTO session_task_events (
                        session_id, seq, request_group_id, task_id, task_name, task_type,
                        user_prompt, response_text, response_has_tool_calls,
                        input_tokens, output_tokens, cost_cents, created_at
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                params![
                    projection.session_id.to_string(),
                    projection.seq,
                    projection.request_group_id,
                    projection.task_id,
                    projection.task_name,
                    projection.task_type,
                    projection.user_prompt,
                    projection.response_text,
                    projection.response_has_tool_calls,
                    projection.input_tokens as i64,
                    projection.output_tokens as i64,
                    projection.cost_cents,
                    projection.created_at.to_rfc3339(),
                ],
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;
        }

        for link in tool_call_links_from_event(&event) {
            tx.execute(
                    "INSERT INTO session_tool_call_links (
                        session_id, tool_call_id, event_seq, request_group_id, tool_call,
                        delegation_profile, origin_llm_call_seq, requested_at, call_identity
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?10)
                     ON CONFLICT(session_id, tool_call_id) DO UPDATE SET
                        event_seq=CASE WHEN ?9 THEN excluded.event_seq ELSE session_tool_call_links.event_seq END,
                        request_group_id=COALESCE(excluded.request_group_id, request_group_id),
                        tool_call=excluded.tool_call,
                        delegation_profile=COALESCE(excluded.delegation_profile, delegation_profile),
                        origin_llm_call_seq=COALESCE(excluded.origin_llm_call_seq, origin_llm_call_seq),
                        requested_at=excluded.requested_at,
                        call_ambiguous=CASE WHEN ?11 OR (call_identity != '' AND call_identity != excluded.call_identity) THEN 1 ELSE call_ambiguous END,
                        call_identity=excluded.call_identity",
                    params![
                        link.session_id.to_string(),
                        link.tool_call.id,
                        event.seq,
                        link.request_group_id,
                        serde_json::to_string(&link.tool_call)
                            .map_err(|e| EngineError::Storage(e.to_string()))?,
                        link.delegation_profile,
                        link.origin_llm_call_seq,
                        link.requested_at.to_rfc3339(),
                        event.event_type == SessionEventType::ToolCall,
                        identities.get(&link.tool_call.id).map(|(hash,_)| hash.as_str()).unwrap_or_default(),
                        identities.get(&link.tool_call.id).is_some_and(|(_,ambiguous)| *ambiguous),
                    ],
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
        }
        if event.event_type == SessionEventType::ToolResult {
            if let Some(tool_call_id) = event
                .payload
                .pointer("/result/tool_call_id")
                .and_then(serde_json::Value::as_str)
            {
                tx.execute(
                    "UPDATE session_tool_call_links
                         SET result_event_seq = ?3
                         WHERE session_id = ?1 AND tool_call_id = ?2",
                    params![session_id.to_string(), tool_call_id, event.seq],
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            }
        }
        Self::project_file_history_on(tx, &event, &raw_links)?;
        Ok(event)
    }

    fn save_file_history_on(
        tx: &rusqlite::Transaction<'_>,
        record: &engine_core::file_history::FileWriteRecord,
    ) -> EngineResult<()> {
        tx.execute("INSERT INTO session_file_history(session_id,tool_call_id,path,seq,payload,file_key) VALUES(?1,?2,?3,?4,?5,?6)
            ON CONFLICT(session_id,tool_call_id,path) DO UPDATE SET seq=excluded.seq,payload=excluded.payload,file_key=excluded.file_key",
            params![record.session_id.to_string(),record.tool_call_id,record.path,record.result_seq.unwrap_or(record.call_seq),serde_json::to_string(record)?,record.file_id.as_ref().map(|id| id.key())])
            .map_err(|e| EngineError::Storage(e.to_string()))?;
        Ok(())
    }

    fn project_file_history_on(
        tx: &rusqlite::Transaction<'_>,
        event: &SessionEvent,
        links: &[ToolCallLink],
    ) -> EngineResult<()> {
        let scope = if links.is_empty() {
            None
        } else {
            Self::file_history_scope_on(tx, event.session_id)?
        };
        for link in links {
            for mut record in engine_core::file_history::records_for_call(
                event.session_id,
                event.seq,
                &link.tool_call,
            ) {
                record.file_id = scope
                    .as_ref()
                    .and_then(|scope| scope.identity(&record.path));
                let previous: Option<String> = tx.query_row("SELECT payload FROM session_file_history WHERE session_id=?1 AND tool_call_id=?2 AND path=?3",
                    params![event.session_id.to_string(),record.tool_call_id,record.path], |row| row.get(0)).optional()
                    .map_err(|e| EngineError::Storage(e.to_string()))?;
                if let Some(previous) = previous {
                    let mut old = serde_json::from_str(&previous)?;
                    engine_core::file_history::merge_call(&mut old, &record, false);
                    record = old;
                }
                Self::save_file_history_on(tx, &record)?;
            }
        }
        if event.event_type == SessionEventType::ToolResult {
            if let Some(id) = event
                .payload
                .pointer("/result/tool_call_id")
                .and_then(serde_json::Value::as_str)
            {
                let mut stmt = tx.prepare("SELECT payload FROM session_file_history WHERE session_id=?1 AND tool_call_id=?2")
                    .map_err(|e| EngineError::Storage(e.to_string()))?;
                let encoded = stmt
                    .query_map(params![event.session_id.to_string(), id], |row| {
                        row.get::<_, String>(0)
                    })
                    .map_err(|e| EngineError::Storage(e.to_string()))?
                    .collect::<rusqlite::Result<Vec<_>>>()
                    .map_err(|e| EngineError::Storage(e.to_string()))?;
                for value in encoded {
                    let mut record = serde_json::from_str(&value)?;
                    engine_core::file_history::apply_result(&mut record, event);
                    Self::save_file_history_on(tx, &record)?;
                }
            }
        }
        Ok(())
    }
}

#[async_trait]
impl EventStore for LocalDatabase {
    async fn read_file_history_scope(
        &self,
        session_id: Uuid,
    ) -> EngineResult<Option<engine_core::file_history::FileHistoryScope>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            Self::file_history_scope_on(&db.read_conn(), session_id)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn bind_file_history_scope(
        &self,
        session_id: Uuid,
        scope: &engine_core::file_history::FileHistoryScope,
    ) -> EngineResult<()> {
        let db = self.clone();
        let scope = scope.clone();
        tokio::task::spawn_blocking(move || {
            // Existing immutable bindings can be checked on a reader without
            // contending with every session's serialized SQLite writer.
            if let Some(old) = Self::file_history_scope_on(&db.read_conn(), session_id)? {
                if old.merge(&scope)? == old { return Ok(()); }
            }
            let mut conn = db.conn.lock().unwrap();
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(|e| EngineError::Storage(e.to_string()))?;
            let old = Self::file_history_scope_on(&tx, session_id)?.unwrap_or(engine_core::file_history::FileHistoryScope { project_id: scope.project_id, roots: vec![] });
            let merged = old.merge(&scope)?;
            if merged == old { return Ok(()); }
            for root in &merged.roots {
                tx.execute("INSERT OR IGNORE INTO session_file_roots(session_id,project_id,root_id,local_path) VALUES(?1,?2,?3,?4)",
                    params![session_id.to_string(),merged.project_id.to_string(),root.id,root.path]).map_err(|e| EngineError::Storage(e.to_string()))?;
            }
            // Upgrade only the session's unbound projection, never its event archive.
            let mut cursor: Option<(String, String)> = None;
            loop {
                let rows = {
                    let sql = if cursor.is_some() {
                        "SELECT tool_call_id,path,payload FROM session_file_history WHERE session_id=?1 AND (tool_call_id,path)>(?2,?3) AND file_key IS NULL ORDER BY tool_call_id,path LIMIT 256"
                    } else {
                        "SELECT tool_call_id,path,payload FROM session_file_history WHERE session_id=?1 AND file_key IS NULL ORDER BY tool_call_id,path LIMIT 256"
                    };
                    let mut stmt = tx.prepare(sql).map_err(|e| EngineError::Storage(e.to_string()))?;
                    let values = if let Some((id,path)) = &cursor { vec![session_id.to_string(),id.clone(),path.clone()] } else { vec![session_id.to_string()] };
                    let rows = stmt.query_map(rusqlite::params_from_iter(values), |row| Ok((row.get::<_,String>(0)?,row.get::<_,String>(1)?,row.get::<_,String>(2)?)))
                        .map_err(|e| EngineError::Storage(e.to_string()))?.collect::<rusqlite::Result<Vec<_>>>()
                        .map_err(|e| EngineError::Storage(e.to_string()))?;
                    rows
                };
                if rows.is_empty() { break; }
                for (id,path,raw) in rows {
                    // These cursor fields never change when an identity is added.
                    cursor = Some((id,path));
                    let mut record: engine_core::file_history::FileWriteRecord = serde_json::from_str(&raw)?;
                    record.file_id = merged.identity(&record.path);
                    if record.file_id.is_some() { Self::save_file_history_on(&tx, &record)?; }
                }
            }
            tx.commit().map_err(|e| EngineError::Storage(e.to_string()))
        }).await.map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn observe_file_history_call_identities(
        &self,
        session_id: Uuid,
        calls: &[ToolCall],
        results: &[engine_core::ToolResult],
    ) -> EngineResult<()> {
        if calls.is_empty() && results.is_empty() {
            return Ok(());
        }
        let db = self.clone();
        let identities: Vec<_> = calls
            .iter()
            .take(512)
            .map(|call| {
                (
                    call.id.clone(),
                    engine_core::file_history::call_identity(call),
                )
            })
            .collect();
        let result_identities: Vec<_> = results
            .iter()
            .take(512)
            .map(|result| {
                (
                    result.tool_call_id.clone(),
                    engine_core::file_history::result_identity(result),
                )
            })
            .collect();
        tokio::task::spawn_blocking(move || {
            let mut conn = db.conn.lock().unwrap();
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(|e| EngineError::Storage(e.to_string()))?;
            {
                let mut stmt = tx.prepare("UPDATE session_tool_call_links SET call_ambiguous=1 WHERE session_id=?1 AND tool_call_id=?2 AND call_identity != '' AND call_identity != ?3 AND call_ambiguous=0")
                    .map_err(|e| EngineError::Storage(e.to_string()))?;
                for (id,hash) in identities { stmt.execute(params![session_id.to_string(),id,hash]).map_err(|e| EngineError::Storage(e.to_string()))?; }
            }
            {
                let mut stmt = tx.prepare("UPDATE session_tool_call_links SET call_ambiguous=1 WHERE session_id=?1 AND tool_call_id=?2 AND call_ambiguous=0
                    AND EXISTS(SELECT 1 FROM session_file_history h WHERE h.session_id=?1 AND h.tool_call_id=?2 AND json_extract(h.payload,'$.result_hash') IS NOT NULL AND json_extract(h.payload,'$.result_hash') != ?3)")
                    .map_err(|e| EngineError::Storage(e.to_string()))?;
                for (id,hash) in result_identities { stmt.execute(params![session_id.to_string(),id,hash]).map_err(|e| EngineError::Storage(e.to_string()))?; }
            }
            tx.commit().map_err(|e| EngineError::Storage(e.to_string()))
        }).await.map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn read_file_history(
        &self,
        session_id: Uuid,
        paths: &[String],
    ) -> EngineResult<Vec<engine_core::file_history::FileWriteRecord>> {
        if paths.is_empty() {
            return Ok(Vec::new());
        }
        let db = self.clone();
        let paths = paths.to_vec();
        tokio::task::spawn_blocking(move || {
            let conn = db.read_conn();
            let scope = Self::file_history_scope_on(&conn, session_id)?;
            let paths = engine_core::file_history::lookup_paths(&paths, scope.as_ref());
            let mut stmt = conn
                .prepare(FILE_HISTORY_LOOKUP_SQL)
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let mut result = Vec::new();
            for path in paths {
                let key = scope
                    .as_ref()
                    .and_then(|scope| scope.identity(&path))
                    .map(|id| id.key());
                let rows = stmt
                    .query_map(
                        params![
                            session_id.to_string(),
                            path,
                            engine_core::file_history::PER_FILE_LIMIT as i64,
                            key
                        ],
                        |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
                    )
                    .map_err(|e| EngineError::Storage(e.to_string()))?;
                for row in rows {
                    let (encoded, ambiguous) =
                        row.map_err(|e| EngineError::Storage(e.to_string()))?;
                    let mut record = serde_json::from_str(&encoded)?;
                    if ambiguous != 0 {
                        engine_core::file_history::invalidate(&mut record);
                    }
                    if let Some(record) = engine_core::file_history::rebase(
                        &record,
                        std::slice::from_ref(&path),
                        scope.as_ref(),
                    ) {
                        result.push(record);
                    }
                }
            }
            Ok(result)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn compare_and_append_continuation(
        &self,
        session_id: Uuid,
        task_id: &str,
        expected_seq: Option<i64>,
        payload: serde_json::Value,
    ) -> EngineResult<Option<SessionEvent>> {
        let db = self.clone();
        let task_id = task_id.to_owned();
        let result = tokio::task::spawn_blocking(move || {
            let mut conn = db.conn.lock().unwrap();
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(|e| EngineError::Storage(e.to_string()))?;
            let previous: Option<(i64,String,String)> = tx.query_row("SELECT seq,payload,created_at FROM session_events
                WHERE session_id=?1 AND event_type='continuation' AND json_extract(payload,'$.task_id')=?2 ORDER BY seq DESC LIMIT 1",
                params![session_id.to_string(),task_id], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?))).optional().map_err(|e| EngineError::Storage(e.to_string()))?;
            if previous.as_ref().map(|(seq,_,_)| *seq) != expected_seq { return Ok(None); }
            if let Some((seq,encoded,created_at)) = previous {
                let prior: serde_json::Value = serde_json::from_str(&encoded)?;
                if engine_core::continuation::same_snapshot(&prior, &payload) {
                    return Ok(Some((SessionEvent { session_id, seq, event_type:SessionEventType::Continuation,
                        payload:prior, created_at:created_at.parse().map_err(|e| EngineError::Storage(format!("invalid continuation timestamp: {e}")))?, cost_cents:None }, false)));
                }
            }
            let event = Self::append_event_on(&tx, session_id, SessionEventType::Continuation, payload, None)?;
            tx.commit().map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok::<_, EngineError>(Some((event, true)))
        }).await.map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))??;
        if let Some((event, true)) = &result {
            self.event_hub.publish(event.clone());
        }
        Ok(result.map(|(event, _)| event))
    }

    async fn read_latest_continuation(
        &self,
        session_id: Uuid,
        task_id: &str,
    ) -> EngineResult<Option<SessionEvent>> {
        let db = self.clone();
        let task_id = task_id.to_owned();
        tokio::task::spawn_blocking(move || {
            let conn = db.read_conn();
            let sql = if task_id.is_empty() {
                "SELECT seq,payload,created_at FROM session_events WHERE session_id=?1 AND event_type='continuation' AND ?2='' ORDER BY seq DESC LIMIT 1"
            } else {
                "SELECT seq,payload,created_at FROM session_events WHERE session_id=?1 AND event_type='continuation' AND json_extract(payload,'$.task_id')=?2 ORDER BY seq DESC LIMIT 1"
            };
            conn.query_row(sql, params![session_id.to_string(), task_id], |row| {
                let payload: String = row.get(1)?; let created_at: String = row.get(2)?;
                Ok(SessionEvent { session_id, seq: row.get(0)?, event_type: SessionEventType::Continuation,
                    payload: serde_json::from_str(&payload).unwrap_or_default(),
                    created_at: created_at.parse().unwrap_or_else(|_| Utc::now()), cost_cents: None })
            }).optional().map_err(|e| EngineError::Storage(e.to_string()))
        }).await.map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn append_event(
        &self,
        session_id: Uuid,
        event_type: SessionEventType,
        payload: serde_json::Value,
        cost_cents: Option<i64>,
    ) -> EngineResult<SessionEvent> {
        let db = self.clone();
        let event = tokio::task::spawn_blocking(move || {
            let mut conn = db.conn.lock().unwrap();
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let event = Self::append_event_on(&tx, session_id, event_type, payload, cost_cents)?;
            tx.commit()
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok::<SessionEvent, EngineError>(event)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))??;
        self.event_hub.publish(event.clone());
        Ok(event)
    }

    async fn read_diagnostic_events(&self, session_id: Uuid) -> EngineResult<Vec<SessionEvent>> {
        self.read_diagnostic_event_page(session_id, 1, usize::MAX)
            .await
    }

    async fn read_diagnostic_event_page(
        &self,
        session_id: Uuid,
        since_seq: i64,
        limit: usize,
    ) -> EngineResult<Vec<SessionEvent>> {
        let since_seq = Some(since_seq);
        let limit = (limit != usize::MAX).then_some(limit.min(512));
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.read_conn();
            let sql = format!(
                "SELECT session_id, seq, event_type, json_patch(json_object(), json_object('cost_recorded', json(CASE WHEN json_type(payload, '$.cost') IS NOT NULL THEN 'true' ELSE 'false' END), 'cost', json(payload -> '$.cost'), 'input_tokens', json(payload -> '$.input_tokens'), 'cached_prompt_tokens', json(payload -> '$.cached_prompt_tokens'), 'predicted_cost_low_cents', json(payload -> '$.predicted_cost_low_cents'), 'predicted_cost_high_cents', json(payload -> '$.predicted_cost_high_cents'), 'outcome_report', json(payload -> '$.outcome_report'), 'repetition_count', json(payload -> '$.repetition_count'), 'injected_message', json(payload -> '$.injected_message'), 'interventions', json(payload -> '$.interventions'), 'intervention', json(payload -> '$.intervention'), 'request_group_id', json(payload -> '$.request_group_id'))) AS payload, cost_cents, created_at
                 FROM session_events
                 WHERE event_type IN ('llm_call', 'auxiliary_call', 'session_completed', 'loop_intervention', 'checkpoint_redeemed', 'agent_health') AND session_id = ?1 {}
                 ORDER BY seq ASC {}",
                since_seq.map(|_| "AND seq >= ?2").unwrap_or(""),
                limit.map(|l| format!("LIMIT {}", l)).unwrap_or_default()
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let session_id_str = session_id.to_string();
            let since_seq_i64: Option<i64> = since_seq;
            let params: Vec<&dyn rusqlite::ToSql> = if let Some(ref s) = since_seq_i64 {
                vec![&session_id_str as &dyn rusqlite::ToSql, s]
            } else {
                vec![&session_id_str as &dyn rusqlite::ToSql]
            };
            let rows = stmt
                .query_map(params.as_slice(), |row| {
                    let id_str: String = row.get(0)?;
                    let etype: String = row.get(2)?;
                    let payload_str: String = row.get(3)?;
                    let created_str: String = row.get(5)?;
                    Ok(SessionEvent {
                        session_id: Uuid::parse_str(&id_str).unwrap_or_default(),
                        seq: row.get(1)?,
                        event_type: str_to_event_type(&etype),
                        payload: serde_json::from_str(&payload_str).unwrap_or_default(),
                        cost_cents: row.get(4)?,
                        created_at: created_str.parse().unwrap_or_else(|_| Utc::now()),
                    })
                })
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| EngineError::Storage(e.to_string()))
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn read_task_decisions(
        &self,
        session_id: Uuid,
        task_id: &str,
    ) -> EngineResult<Vec<SessionEvent>> {
        let db = self.clone();
        let task_id = task_id.to_owned();
        tokio::task::spawn_blocking(move || {
            let conn = db.read_conn();
            let sql = "SELECT session_id, seq, event_type, payload, cost_cents, created_at
                FROM session_events WHERE session_id = ?1 AND event_type = 'task_decision'
                AND json_extract(payload, '$.task_id') = ?2 ORDER BY seq ASC";
            let mut stmt = conn
                .prepare(sql)
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map(params![session_id.to_string(), task_id], |row| {
                    let id_str: String = row.get(0)?;
                    let etype: String = row.get(2)?;
                    let payload_str: String = row.get(3)?;
                    let created_str: String = row.get(5)?;
                    Ok(SessionEvent {
                        session_id: Uuid::parse_str(&id_str).unwrap_or_default(),
                        seq: row.get(1)?,
                        event_type: str_to_event_type(&etype),
                        payload: serde_json::from_str(&payload_str).unwrap_or_default(),
                        cost_cents: row.get(4)?,
                        created_at: created_str.parse().unwrap_or_else(|_| Utc::now()),
                    })
                })
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| EngineError::Storage(e.to_string()))
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn read_events(
        &self,
        session_id: Uuid,
        since_seq: Option<i64>,
        limit: Option<usize>,
    ) -> EngineResult<Vec<SessionEvent>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.read_conn();
            let sql = format!(
                "SELECT session_id, seq, event_type, payload, cost_cents, created_at
                 FROM session_events
                 WHERE session_id = ?1 {}
                 ORDER BY seq ASC {}",
                since_seq.map(|_| "AND seq >= ?2").unwrap_or(""),
                limit.map(|l| format!("LIMIT {}", l)).unwrap_or_default()
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let session_id_str = session_id.to_string();
            let since_seq_i64: Option<i64> = since_seq;
            let params: Vec<&dyn rusqlite::ToSql> = if let Some(ref s) = since_seq_i64 {
                vec![&session_id_str as &dyn rusqlite::ToSql, s]
            } else {
                vec![&session_id_str as &dyn rusqlite::ToSql]
            };
            let rows = stmt
                .query_map(params.as_slice(), |row| {
                    let id_str: String = row.get(0)?;
                    let etype: String = row.get(2)?;
                    let payload_str: String = row.get(3)?;
                    let created_str: String = row.get(5)?;
                    Ok(SessionEvent {
                        session_id: Uuid::parse_str(&id_str).unwrap_or_default(),
                        seq: row.get(1)?,
                        event_type: str_to_event_type(&etype),
                        payload: serde_json::from_str(&payload_str).unwrap_or_default(),
                        cost_cents: row.get(4)?,
                        created_at: created_str.parse().unwrap_or_else(|_| Utc::now()),
                    })
                })
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| EngineError::Storage(e.to_string()))
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn read_event_page(
        &self,
        session_id: Uuid,
        since_seq: Option<i64>,
        until_seq: Option<i64>,
        limit: usize,
    ) -> EngineResult<Vec<SessionEvent>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.read_conn();
            let mut sql = String::from(
                "SELECT session_id, seq, event_type,
                        CASE
                          WHEN event_type = 'llm_call'
                           AND json_type(payload, '$.request_messages') IS NOT NULL
                          THEN json_set(
                            json_remove(payload, '$.request_messages'),
                            '$._reincore_omitted_fields',
                            json_array('request_messages')
                          )
                          ELSE payload
                        END AS payload,
                        cost_cents, created_at
                 FROM session_events WHERE session_id = ?",
            );
            let mut values = vec![rusqlite::types::Value::Text(session_id.to_string())];
            if let Some(since) = since_seq {
                sql.push_str(" AND seq >= ?");
                values.push(rusqlite::types::Value::Integer(since));
            }
            if let Some(until) = until_seq {
                sql.push_str(" AND seq <= ?");
                values.push(rusqlite::types::Value::Integer(until));
            }
            sql.push_str(" ORDER BY seq ASC LIMIT ?");
            values.push(rusqlite::types::Value::Integer(limit as i64));

            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map(rusqlite::params_from_iter(values), session_event_from_row)
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| EngineError::Storage(e.to_string()))
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn read_timeline_events(&self, session_id: Uuid) -> EngineResult<Vec<SessionEvent>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.read_conn();
            let mut stmt = conn
                .prepare(
                    "SELECT e.session_id, e.seq, e.event_type, e.cost_cents, e.created_at,
                            COALESCE(
                              p.request_group_id,
                              json_extract(e.payload, '$.request_group_id')
                            ) AS request_group_id,
                            p.task_id, COALESCE((SELECT title FROM generated_titles WHERE title_key = 'task:' || p.session_id || ':' || p.task_id), p.task_name), p.task_type, p.user_prompt,
                            p.response_text, p.response_has_tool_calls,
                            p.input_tokens, p.output_tokens,
                            CASE WHEN e.event_type = 'llm_call'
                              THEN json_extract(e.payload, '$.model') END AS model,
                            CASE WHEN e.event_type = 'llm_call'
                              THEN json_extract(e.payload, '$.batch_submitted') END AS batch_submitted,
                            CASE WHEN e.event_type = 'loop_intervention'
                              THEN json_extract(e.payload, '$.repetition_count') END AS repetition_count,
                            CASE WHEN e.event_type = 'loop_intervention'
                              THEN json_extract(e.payload, '$.injected_message') END AS injected_message,
                            json_extract(e.payload, '$.interventions') AS interventions,
                            json_extract(e.payload, '$.intervention') AS intervention,
                            CASE WHEN e.event_type = 'tool_result'
                              THEN json_extract(e.payload, '$.result.is_error') END AS result_is_error, json_extract(e.payload, '$.cost') AS cost,
                            CASE WHEN e.event_type = 'llm_call' AND (p.user_prompt IS NULL OR length(p.user_prompt) >= 8192 OR p.user_prompt LIKE '%<%')
                              THEN json_extract(e.payload, '$.request_messages') END AS original_request_messages,
                            json_extract(e.payload, '$._reincore_privacy_redaction.pii_redaction_count') AS pii_redaction_count
                     FROM session_events AS e
                     LEFT JOIN session_task_events AS p
                       ON p.session_id = e.session_id AND p.seq = e.seq
                     WHERE e.session_id = ?1
                     ORDER BY e.seq ASC",
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map([session_id.to_string()], |row| {
                    let id: String = row.get(0)?;
                    let seq: i64 = row.get(1)?;
                    let event_type_raw: String = row.get(2)?;
                    let cost_cents: Option<i64> = row.get(3)?;
                    let created_at_raw: String = row.get(4)?;
                    let event_type = str_to_event_type(&event_type_raw);
                    let created_at = created_at_raw.parse().unwrap_or_else(|_| Utc::now());
                    let request_group_id: Option<String> = row.get(5)?;

                    let mut payload = if event_type == SessionEventType::LlmCall {
                        TaskEventProjection {
                            session_id: Uuid::parse_str(&id).unwrap_or_default(),
                            seq,
                            request_group_id: request_group_id.clone(),
                            task_id: row.get(6)?,
                            task_name: row.get(7)?,
                            task_type: row.get(8)?,
                            user_prompt: TaskEventProjection::recover_user_prompt(row.get(9)?, row.get::<_, Option<String>>("original_request_messages")?.and_then(|raw| serde_json::from_str(&raw).ok())),
                            response_text: row.get(10)?,
                            response_has_tool_calls: row
                                .get::<_, Option<bool>>(11)?
                                .unwrap_or(false),
                            input_tokens: row.get::<_, Option<i64>>(12)?.unwrap_or(0).max(0)
                                as u64,
                            output_tokens: row.get::<_, Option<i64>>(13)?.unwrap_or(0).max(0)
                                as u64,
                            cost_cents,
                            created_at,
                        }
                        .into_event()
                        .payload
                    } else {
                        serde_json::json!({})
                    };
                    if let Some(cost) = row.get::<_, Option<String>>("cost")? { payload["cost"] = serde_json::from_str(&cost).map_err(|e|rusqlite::Error::FromSqlConversionFailure(19,rusqlite::types::Type::Text,Box::new(e)))?; }
                    if let Some(group) = request_group_id {
                        payload["request_group_id"] = serde_json::Value::String(group);
                    }
                    if let Some(model) = row.get::<_, Option<String>>(14)? {
                        payload["model"] = serde_json::Value::String(model);
                    }
                    if row.get::<_, Option<bool>>(15)?.unwrap_or(false) {
                        payload["batch_submitted"] = serde_json::Value::Bool(true);
                    }
                    if let Some(count) = row.get::<_, Option<i64>>(16)? {
                        payload["repetition_count"] = serde_json::json!(count.max(0));
                    }
                    if let Some(message) = row.get::<_, Option<String>>(17)? {
                        payload["injected_message"] = serde_json::Value::String(message);
                    }
                    if let Some(records) = row.get::<_, Option<String>>(18)? {
                        payload["interventions"] = serde_json::from_str(&records).map_err(|e| rusqlite::Error::FromSqlConversionFailure(18, rusqlite::types::Type::Text, Box::new(e)))?;
                    }
                    if let Some(record) = row.get::<_, Option<String>>(19)? {
                        payload["intervention"] = serde_json::from_str(&record).map_err(|e| rusqlite::Error::FromSqlConversionFailure(19, rusqlite::types::Type::Text, Box::new(e)))?;
                    }
                    if let Some(is_error) = row.get::<_, Option<bool>>(20)? {
                        payload["result"] = serde_json::json!({"is_error": is_error});
                    }
                    if let Some(pii) =
                        row.get::<_, Option<i64>>("pii_redaction_count")?.filter(|count| *count > 0)
                    {
                        payload["_reincore_privacy_redaction"] = serde_json::json!({
                            "pii_redaction_count": pii.max(0),
                            "representation": "sanitized",
                        });
                    }
                    Ok(SessionEvent {
                        session_id: Uuid::parse_str(&id).unwrap_or_default(),
                        seq,
                        event_type,
                        payload,
                        cost_cents,
                        created_at,
                    })
                })
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| EngineError::Storage(e.to_string()))
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn latest_seq(&self, session_id: Uuid) -> EngineResult<Option<i64>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.read_conn();
            let seq: Option<i64> = conn
                .query_row(
                    "SELECT MAX(seq) FROM session_events WHERE session_id = ?1",
                    [session_id.to_string()],
                    |row| row.get::<_, Option<i64>>(0),
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(seq)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn read_recent_task_events(
        &self,
        session_id: Uuid,
        limit: usize,
    ) -> EngineResult<Vec<TaskEventProjection>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.read_conn();
            let mut stmt = conn
                .prepare(
                    "SELECT session_id, seq, request_group_id, task_id, COALESCE((SELECT title FROM generated_titles WHERE title_key = 'task:' || session_task_events.session_id || ':' || session_task_events.task_id), task_name), task_type,
                            user_prompt, response_text, response_has_tool_calls,
                            input_tokens, output_tokens, cost_cents, created_at,
                            CASE WHEN user_prompt IS NULL OR length(user_prompt) >= 8192 OR user_prompt LIKE '%<%' THEN (SELECT json_extract(e.payload, '$.request_messages') FROM session_events e WHERE e.session_id = session_task_events.session_id AND e.seq = session_task_events.seq) END AS original_request_messages
                     FROM session_task_events
                     WHERE session_id = ?1
                     ORDER BY seq DESC LIMIT ?2",
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map(params![session_id.to_string(), limit as i64], |row| {
                    let id: String = row.get(0)?;
                    let created_at: String = row.get(12)?;
                    Ok(TaskEventProjection {
                        session_id: Uuid::parse_str(&id).unwrap_or_default(),
                        seq: row.get(1)?,
                        request_group_id: row.get(2)?,
                        task_id: row.get(3)?,
                        task_name: row.get(4)?,
                        task_type: row.get(5)?,
                        user_prompt: TaskEventProjection::recover_user_prompt(row.get(6)?, row.get::<_, Option<String>>("original_request_messages")?.and_then(|raw| serde_json::from_str(&raw).ok())),
                        response_text: row.get(7)?,
                        response_has_tool_calls: row.get(8)?,
                        input_tokens: row.get::<_, i64>(9)?.max(0) as u64,
                        output_tokens: row.get::<_, i64>(10)?.max(0) as u64,
                        cost_cents: row.get(11)?,
                        created_at: created_at.parse().unwrap_or_else(|_| Utc::now()),
                    })
                })
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let mut projections = rows
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            projections.reverse();
            Ok(projections)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn read_tool_events_by_call_ids(
        &self,
        session_id: Uuid,
        tool_call_ids: &[String],
    ) -> EngineResult<Vec<SessionEvent>> {
        if tool_call_ids.is_empty() {
            return Ok(Vec::new());
        }
        let db = self.clone();
        let ids = tool_call_ids.to_vec();
        tokio::task::spawn_blocking(move || {
            let conn = db.read_conn();
            let placeholders = std::iter::repeat("?")
                .take(ids.len())
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!(
                "SELECT e.session_id, e.seq, e.event_type, e.payload, e.cost_cents, e.created_at
                 FROM session_events AS e
                 JOIN (
                   SELECT event_seq AS seq FROM session_tool_call_links
                    WHERE session_id = ? AND tool_call_id IN ({0})
                   UNION
                   SELECT result_event_seq AS seq FROM session_tool_call_links
                    WHERE session_id = ? AND tool_call_id IN ({0}) AND result_event_seq IS NOT NULL
                 ) AS wanted ON wanted.seq = e.seq
                 WHERE e.session_id = ? ORDER BY e.seq ASC",
                placeholders
            );
            let session = session_id.to_string();
            let mut values = Vec::<String>::with_capacity(ids.len() * 2 + 3);
            values.push(session.clone());
            values.extend(ids.iter().cloned());
            values.push(session.clone());
            values.extend(ids.iter().cloned());
            values.push(session);
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map(rusqlite::params_from_iter(values), session_event_from_row)
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| EngineError::Storage(e.to_string()))
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn read_tool_call_links(
        &self,
        session_id: Uuid,
        tool_call_ids: &[String],
    ) -> EngineResult<Vec<ToolCallLink>> {
        if tool_call_ids.is_empty() {
            return Ok(Vec::new());
        }
        let db = self.clone();
        let ids = tool_call_ids.to_vec();
        tokio::task::spawn_blocking(move || {
            let conn = db.read_conn();
            let placeholders = std::iter::repeat("?")
                .take(ids.len())
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!(
                "SELECT session_id, tool_call, request_group_id, origin_llm_call_seq,
                        delegation_profile, requested_at
                 FROM session_tool_call_links
                 WHERE session_id = ? AND tool_call_id IN ({placeholders})"
            );
            let session = session_id.to_string();
            let mut values = Vec::<String>::with_capacity(ids.len() + 1);
            values.push(session);
            values.extend(ids);
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map(rusqlite::params_from_iter(values), |row| {
                    let session: String = row.get(0)?;
                    let tool_call: String = row.get(1)?;
                    let requested_at: String = row.get(5)?;
                    Ok(ToolCallLink {
                        session_id: Uuid::parse_str(&session).unwrap_or_default(),
                        tool_call: serde_json::from_str(&tool_call).unwrap_or(ToolCall {
                            id: String::new(),
                            call_type: "function".to_string(),
                            function: engine_core::models::FunctionCall {
                                name: String::new(),
                                arguments: "{}".to_string(),
                            },
                        }),
                        request_group_id: row.get(2)?,
                        origin_llm_call_seq: row.get(3)?,
                        delegation_profile: row.get(4)?,
                        requested_at: requested_at.parse().unwrap_or_else(|_| Utc::now()),
                    })
                })
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| EngineError::Storage(e.to_string()))
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn count_effective_loop_interventions(&self, session_id: Uuid) -> EngineResult<u64> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM session_events
                     WHERE session_id = ?1 AND event_type = 'loop_intervention'
                       AND (
                         CAST(COALESCE(json_extract(payload, '$.repetition_count'), 0) AS INTEGER) > 0
                         OR TRIM(COALESCE(json_extract(payload, '$.injected_message'), '')) <> ''
                       )",
                    [session_id.to_string()],
                    |row| row.get(0),
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(count.max(0) as u64)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn read_latest_summary(&self, session_id: Uuid) -> EngineResult<Option<SummaryPayload>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let payload_str: Option<String> = conn
                .query_row(
                    "SELECT payload FROM session_events
                     WHERE session_id = ?1 AND event_type = 'summary'
                     ORDER BY seq DESC LIMIT 1",
                    [session_id.to_string()],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(payload_str.and_then(|s| serde_json::from_str::<SummaryPayload>(&s).ok()))
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn read_latest_files(
        &self,
        session_id: Uuid,
    ) -> EngineResult<Vec<FileContextAttachment>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.read_conn();
            // For each path, pick the row with the highest seq (i.e. the
            // latest version) among tool_result events carrying a `file`
            // attachment. `json_extract` returns NULL when the payload has
            // no `file` key, so those rows are excluded by the WHERE
            // clause. The subquery computes the winning seq per path; the
            // outer query joins back to fetch the full payload.
            let mut stmt = conn
                .prepare(
                    "WITH latest AS (
                        SELECT
                            json_extract(payload, '$.file.path') AS path,
                            MAX(seq) AS max_seq
                        FROM session_events
                        WHERE session_id = ?1
                          AND event_type = 'tool_result'
                          AND json_extract(payload, '$.file.path') IS NOT NULL
                        GROUP BY json_extract(payload, '$.file.path')
                    )
                    SELECT json_extract(e.payload, '$.file')
                    FROM session_events e
                    JOIN latest l
                      ON json_extract(e.payload, '$.file.path') = l.path
                     AND e.seq = l.max_seq
                    WHERE e.session_id = ?1
                    ORDER BY l.path ASC",
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map([session_id.to_string()], |row| {
                    let payload_str: String = row.get(0)?;
                    Ok(payload_str)
                })
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let mut files: Vec<FileContextAttachment> = Vec::new();
            for payload_str in rows {
                let payload_str = payload_str.map_err(|e| EngineError::Storage(e.to_string()))?;
                if let Ok(file) = serde_json::from_str::<FileContextAttachment>(&payload_str) {
                    files.push(file);
                }
            }
            // The join orders by `latest.path`, but re-sort here to
            // guarantee path-ascending order regardless of join quirks.
            files.sort_by(|a, b| a.path.cmp(&b.path));
            Ok(files)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn read_files_by_paths(
        &self,
        session_id: Uuid,
        paths: &[String],
    ) -> EngineResult<Vec<FileContextAttachment>> {
        if paths.is_empty() {
            return Ok(vec![]);
        }
        let db = self.clone();
        let paths: Vec<String> = paths.iter().take(24).cloned().collect();
        tokio::task::spawn_blocking(move || {
            let conn = db.read_conn();
            let sql = format!("SELECT payload->'$.file' FROM session_events WHERE session_id=?1 AND event_type='tool_result' AND json_extract(payload,'$.file.path') IS NOT NULL
                AND json_extract(payload,'$.file.path') IN ({}) AND seq IN (
                    SELECT MAX(seq) FROM session_events WHERE session_id=?1 AND event_type='tool_result' AND json_extract(payload,'$.file.path') IS NOT NULL
                    AND json_extract(payload,'$.file.path') IN ({}) GROUP BY json_extract(payload,'$.file.path'))
                ORDER BY json_extract(payload,'$.file.path')", (2..=paths.len()+1).map(|i| format!("?{i}")).collect::<Vec<_>>().join(","),
                    (2..=paths.len()+1).map(|i| format!("?{i}")).collect::<Vec<_>>().join(","));
            let mut values = vec![session_id.to_string()]; values.extend(paths);
            let mut stmt = conn.prepare(&sql).map_err(|e| EngineError::Storage(e.to_string()))?;
            let rows = stmt.query_map(rusqlite::params_from_iter(values), |row| row.get::<_,String>(0))
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            rows.map(|row| row.map_err(|e| EngineError::Storage(e.to_string())).and_then(|text|
                serde_json::from_str(&text).map_err(EngineError::Serialization))).collect()
        }).await.map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn redeem_checkpoint(
        &self,
        session_id: Uuid,
        token_fingerprint: &str,
        payload: serde_json::Value,
    ) -> EngineResult<SessionEvent> {
        let db = self.clone();
        let token_fingerprint = token_fingerprint.to_string();
        tokio::task::spawn_blocking(move || {
            let mut conn = db.conn.lock().unwrap();
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let session_id_string = session_id.to_string();
            let already_redeemed: bool = tx
                .query_row(
                    "SELECT EXISTS(
                        SELECT 1 FROM session_events
                        WHERE session_id = ?1
                          AND event_type = 'checkpoint_redeemed'
                          AND json_extract(payload, '$.token_fingerprint') = ?2
                    )",
                    params![&session_id_string, &token_fingerprint],
                    |row| row.get(0),
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            if already_redeemed {
                return Err(EngineError::CheckpointAlreadyRedeemed);
            }

            let seq: i64 = tx
                .query_row(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM session_events WHERE session_id = ?1",
                    [&session_id_string],
                    |row| row.get(0),
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let created_at = Utc::now();
            tx.execute(
                "INSERT INTO session_events (
                    session_id, seq, event_type, payload, cost_cents, created_at
                ) VALUES (?1, ?2, 'checkpoint_redeemed', ?3, NULL, ?4)",
                params![
                    &session_id_string,
                    seq,
                    payload.to_string(),
                    created_at.to_rfc3339(),
                ],
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;
            tx.commit()
                .map_err(|e| EngineError::Storage(e.to_string()))?;

            Ok(SessionEvent {
                session_id,
                seq,
                event_type: SessionEventType::CheckpointRedeemed,
                payload,
                cost_cents: None,
                created_at,
            })
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }
}

pub(super) fn session_event_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SessionEvent> {
    let id: String = row.get(0)?;
    let event_type: String = row.get(2)?;
    let payload: String = row.get(3)?;
    let created_at: String = row.get(5)?;
    Ok(SessionEvent {
        session_id: Uuid::parse_str(&id).unwrap_or_default(),
        seq: row.get(1)?,
        event_type: str_to_event_type(&event_type),
        payload: serde_json::from_str(&payload).unwrap_or_default(),
        cost_cents: row.get(4)?,
        created_at: created_at.parse().unwrap_or_else(|_| Utc::now()),
    })
}

pub(super) fn event_type_to_str(et: SessionEventType) -> &'static str {
    match et {
        SessionEventType::LlmCall => "llm_call",
        SessionEventType::AuxiliaryCall => "auxiliary_call",
        SessionEventType::ToolCall => "tool_call",
        SessionEventType::ToolResult => "tool_result",
        SessionEventType::Summary => "summary",
        SessionEventType::TierChange => "tier_change",
        SessionEventType::LoopIntervention => "loop_intervention",
        SessionEventType::Checkpoint => "checkpoint",
        SessionEventType::CheckpointRedeemed => "checkpoint_redeemed",
        SessionEventType::AgentHealth => "agent_health",
        SessionEventType::SessionRollover => "session_rollover",
        SessionEventType::SessionCompleted => "session_completed",
        SessionEventType::TitleUpdated => "title_updated",
        SessionEventType::PreWarm => "pre_warm",
        SessionEventType::SettingsApplied => "settings_applied",
        SessionEventType::Batch => "batch",
        SessionEventType::TaskDecision => "task_decision",
        SessionEventType::Continuation => "continuation",
    }
}

pub(super) fn str_to_event_type(s: &str) -> SessionEventType {
    match s {
        "auxiliary_call" => SessionEventType::AuxiliaryCall,
        "tool_call" => SessionEventType::ToolCall,
        "tool_result" => SessionEventType::ToolResult,
        "summary" => SessionEventType::Summary,
        "tier_change" => SessionEventType::TierChange,
        "loop_intervention" => SessionEventType::LoopIntervention,
        "checkpoint" => SessionEventType::Checkpoint,
        "checkpoint_redeemed" => SessionEventType::CheckpointRedeemed,
        "agent_health" => SessionEventType::AgentHealth,
        "session_rollover" => SessionEventType::SessionRollover,
        "session_completed" => SessionEventType::SessionCompleted,
        "title_updated" => SessionEventType::TitleUpdated,
        "pre_warm" => SessionEventType::PreWarm,
        "settings_applied" => SessionEventType::SettingsApplied,
        "batch" => SessionEventType::Batch,
        "task_decision" => SessionEventType::TaskDecision,
        "continuation" => SessionEventType::Continuation,
        _ => SessionEventType::LlmCall,
    }
}
