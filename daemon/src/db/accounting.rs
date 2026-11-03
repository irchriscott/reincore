use anyhow::Result;
use async_trait::async_trait;
use engine_core::{
    budget::SettingsStore,
    error::{EngineError, EngineResult},
    event::SessionEventType,
};
use rusqlite::{params, OptionalExtension, TransactionBehavior};
use uuid::Uuid;

use super::sessions::session_from_row;
use super::LocalDatabase;
use engine_core::accounting::{AccountingStore, CompletionAccounting};
fn storage(e: impl std::fmt::Display) -> EngineError {
    EngineError::Storage(e.to_string())
}

impl LocalDatabase {
    fn stage_accounting_sync(&self, record: &CompletionAccounting) -> EngineResult<()> {
        let mut record = record.clone();
        engine_core::privacy::current_policy().sanitize_persistent_value(&mut record.payload);
        record.validate()?;
        let conn = self.conn.lock().unwrap();
        let encoded = serde_json::to_string(&record)?;
        conn.execute(
            "INSERT INTO completion_accounting (id, session_id, record, settled)
            VALUES (?1, ?2, ?3, 0) ON CONFLICT(id) DO NOTHING",
            params![
                record.id.to_string(),
                record.session_id.to_string(),
                encoded
            ],
        )
        .map_err(storage)?;
        let saved: String = conn
            .query_row(
                "SELECT record FROM completion_accounting WHERE id=?1",
                [record.id.to_string()],
                |row| row.get(0),
            )
            .map_err(storage)?;
        if serde_json::from_str::<CompletionAccounting>(&saved)? != record {
            return Err(EngineError::InvalidInput(
                "accounting ID reused with different usage".into(),
            ));
        }
        Ok(())
    }
    fn settle_accounting_sync(&self, id: Uuid) -> EngineResult<Option<Uuid>> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(storage)?;
        let (encoded, settled): (String, i64) = tx
            .query_row(
                "SELECT record, settled FROM completion_accounting WHERE id=?1",
                [id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(storage)?;
        if settled == 1 {
            return Ok(None);
        }
        let record: CompletionAccounting = serde_json::from_str(&encoded)?;
        let mut session = tx.query_row("SELECT id, task_summary, project_path, project_name, project_id, client_id, external_session_id,
                        parent_session_id, root_session_id, rollover_index, inherited_spend_cents,
                        context_tokens, mode, status, budget_cents, spend_cents, tier,
                        predicted_high_cents, cost_accuracy_bps, cost_accuracy_samples,
                        cache_hit_rate_bps, cache_hit_rate_samples,
                        total_input_tokens, total_output_tokens,
                        is_subagent, subagent_parent_session_id,
                        subagent_spawn_tool_call_id, subagent_spawn_request_group_id,
                        created_at, updated_at, spend_nano_usd, inherited_spend_nano_usd, unknown_cost_calls
                 FROM sessions WHERE id = ?1", [record.session_id.to_string()], session_from_row).map_err(storage)?;
        let previous_tier = session.tier;
        record.apply(&mut session)?;
        let mut payload = record.payload.clone();
        payload["accounting_id"] = serde_json::json!(id);
        if let Some(cost) = &record.cost {
            payload["cost"] = serde_json::to_value(cost)?;
        }
        payload["purpose"] = serde_json::json!(record.purpose);
        let mut events = vec![Self::append_event_on(
            &tx,
            session.id,
            if record.purpose.is_some() {
                SessionEventType::AuxiliaryCall
            } else {
                SessionEventType::LlmCall
            },
            payload,
            record
                .amount_nano_usd()
                .map(|n| n / engine_core::pricing::NANO_USD_PER_CENT),
        )?];
        tx.execute("INSERT INTO ledger_events (session_id, amount_cents, reason, created_at, accounting_id, amount_nano_usd, cost) VALUES (?1, ?2, ?5, ?3, ?4, ?6, ?7)",
            params![session.id.to_string(), record.cost_cents, record.recorded_at.to_rfc3339(), id.to_string(), record.purpose.as_deref().unwrap_or("llm_call"), record.amount_nano_usd(), record.cost.as_ref().map(serde_json::to_string).transpose()?]).map_err(storage)?;
        tx.execute("UPDATE sessions SET spend_cents=?2, tier=?3, context_tokens=?4, total_input_tokens=?5, total_output_tokens=?6, predicted_high_cents=?7, cost_accuracy_bps=?8, cost_accuracy_samples=?9, cache_hit_rate_bps=?10, cache_hit_rate_samples=?11, updated_at=?12, spend_nano_usd=?13, unknown_cost_calls=?14 WHERE id=?1", params![session.id.to_string(), session.spend_cents, session.tier as i64, session.context_tokens as i64, session.total_input_tokens as i64, session.total_output_tokens as i64, session.predicted_high_cents, session.cost_accuracy_bps, session.cost_accuracy_samples, session.cache_hit_rate_bps, session.cache_hit_rate_samples, session.updated_at.to_rfc3339(), session.spend_nano_usd, session.unknown_cost_calls as i64]).map_err(storage)?;
        // A provider call can finish after another request has rolled over.
        // Refresh descendants' inherited spend atomically, without another
        // ledger charge or changing their own usage/lifecycle.
        let children: Vec<String> = {
            let mut stmt = tx.prepare("WITH RECURSIVE descendants(id) AS (
                SELECT id FROM sessions WHERE parent_session_id=?1 AND rollover_index>?2
                UNION SELECT s.id FROM sessions s JOIN descendants d ON s.parent_session_id=d.id WHERE s.rollover_index>?2)
                SELECT id FROM sessions WHERE id IN (SELECT id FROM descendants) ORDER BY rollover_index,id").map_err(storage)?;
            let rows = stmt
                .query_map(
                    params![session.id.to_string(), session.rollover_index],
                    |row| row.get(0),
                )
                .map_err(storage)?;
            rows.collect::<Result<Vec<_>, _>>().map_err(storage)?
        };
        let mut lineage = std::collections::HashMap::from([(session.id, session.clone())]);
        let mut target = session.id;
        for child in children {
            let child_id = Uuid::parse_str(&child).map_err(storage)?;
            let mut child = Self::get_session_on(&tx, child_id)?
                .ok_or_else(|| storage("missing descendant"))?;
            let previous_child_tier = child.tier;
            let parent = child
                .parent_session_id
                .and_then(|id| lineage.get(&id))
                .ok_or_else(|| storage("invalid continuation lineage"))?;
            child.inherited_spend_nano_usd = Some(parent.lineage_spend_nano_usd());
            child.inherited_spend_cents = parent.lineage_spend_cents();
            child.tier = engine_core::ProgressiveBudget::new(child.budget_cents)?
                .current_tier_nano_usd(child.lineage_spend_nano_usd())
                .as_u8();
            tx.execute("UPDATE sessions SET inherited_spend_cents=?2,inherited_spend_nano_usd=?3,tier=?4,updated_at=?5 WHERE id=?1",
                params![child.id.to_string(),child.inherited_spend_cents,child.inherited_spend_nano_usd,child.tier as i64,session.updated_at.to_rfc3339()]).map_err(storage)?;
            if previous_child_tier != child.tier {
                events.push(Self::append_event_on(&tx, child.id, SessionEventType::TierChange, serde_json::json!({
                    "previous_tier":previous_child_tier,"new_tier":child.tier,
                    "spend_cents":child.lineage_spend_cents(),"budget_cents":child.budget_cents,
                    "accounting_id":id,"source_session_id":session.id,"reason":"ancestor_accounting",
                }), None)?);
            }
            target = child.id;
            lineage.insert(child.id, child);
        }
        if record.purpose.is_none() {
            if let Some(task) = record.payload["task_id"].as_str() {
                for target in [session.id, target]
                    .into_iter()
                    .collect::<std::collections::BTreeSet<_>>()
                {
                    if lineage[&target].project_id != session.project_id {
                        continue;
                    }
                    let previous: Option<String> = tx.query_row("SELECT payload FROM session_events WHERE session_id=?1
                        AND event_type='continuation' AND json_extract(payload,'$.task_id')=?2 ORDER BY seq DESC LIMIT 1",
                        params![target.to_string(),task], |row| row.get(0)).optional().map_err(storage)?;
                    if let Some(previous) = previous {
                        let previous: serde_json::Value = serde_json::from_str(&previous)?;
                        let seq: i64 = tx.query_row("SELECT COALESCE(MAX(seq),0) FROM session_events WHERE session_id=?1",
                            [target.to_string()], |row| row.get(0)).map_err(storage)?;
                        if let Some(payload) = engine_core::continuation::completion_snapshot(
                            &previous,
                            &record.payload,
                            session.id,
                            seq,
                        )? {
                            if !engine_core::continuation::same_snapshot(&previous, &payload) {
                                events.push(Self::append_event_on(
                                    &tx,
                                    target,
                                    SessionEventType::Continuation,
                                    payload,
                                    None,
                                )?);
                            }
                        }
                    }
                }
            }
        }
        if previous_tier != session.tier {
            events.push(Self::append_event_on(&tx, session.id, SessionEventType::TierChange, serde_json::json!({
                "previous_tier": previous_tier, "new_tier": session.tier,
                "spend_cents": session.lineage_spend_cents(), "budget_cents": session.budget_cents,
                "accounting_id": id,
            }), None)?);
        }
        tx.execute(
            "UPDATE completion_accounting SET settled=1 WHERE id=?1",
            [id.to_string()],
        )
        .map_err(storage)?;
        tx.commit().map_err(storage)?;
        drop(conn);
        for event in events {
            self.event_hub.publish(event);
        }
        Ok((previous_tier != session.tier && session.tier >= 2).then_some(session.id))
    }
    pub(super) fn recover_accounting_sync(&self) -> EngineResult<()> {
        let ids = {
            let conn = self.conn.lock().unwrap();
            let mut stmt = conn
                .prepare("SELECT id FROM completion_accounting WHERE settled=0 ORDER BY rowid")
                .map_err(storage)?;
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(storage)?;
            rows.collect::<Result<Vec<_>, _>>().map_err(storage)?
        };
        for id in ids {
            self.settle_accounting_sync(Uuid::parse_str(&id).map_err(storage)?)?;
        }
        Ok(())
    }
}
#[async_trait]
impl AccountingStore for LocalDatabase {
    async fn stage_accounting(&self, record: &CompletionAccounting) -> EngineResult<()> {
        let db = self.clone();
        let record = record.clone();
        tokio::task::spawn_blocking(move || db.stage_accounting_sync(&record))
            .await
            .map_err(storage)?
    }
    async fn settle_accounting(&self, id: Uuid) -> EngineResult<()> {
        let db = self.clone();
        let compact = tokio::task::spawn_blocking(move || db.settle_accounting_sync(id))
            .await
            .map_err(storage)??;
        if let Some(session_id) = compact {
            if let Ok(settings) = self.get_compression_settings().await {
                if let Some(mode) = settings.mode {
                    let _ = engine_core::ContextCompressor::new()
                        .compress_session(self, session_id, mode)
                        .await;
                }
            }
        }
        Ok(())
    }
    async fn recover_accounting(&self) -> EngineResult<()> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.recover_accounting_sync())
            .await
            .map_err(storage)?
    }
}

#[cfg(test)]
mod tests {
    use engine_core::{
        budget::LedgerStore,
        event::{EventStore, SessionEventType},
        session::{Session, SessionStatus},
    };
    use uuid::Uuid;

    use super::*;
    fn record(session: Uuid, cost: i64) -> CompletionAccounting {
        CompletionAccounting {
            cost: None,
            purpose: None,
            recorded_at: chrono::Utc::now(),
            id: Uuid::new_v4(),
            session_id: session,
            cost_cents: cost,
            payload: serde_json::json!({"input_tokens": 12, "output_tokens": 3,
            "cost_accuracy_bps": 10000, "cache_hit_rate_bps": 5000,
            "predicted_cost_high_cents": cost,
            "request_messages": [], "response_message": {"role":"assistant", "content":"ok"},
            "model":"test", "model_tier":"lite", "request_group_id":"shared-group"}),
        }
    }
    fn fixture() -> (LocalDatabase, Session) {
        let db = LocalDatabase::open_in_memory().unwrap();
        let session = Session::new("accounting", 10000).unwrap();
        db.insert_session(&session).unwrap();
        (db, session)
    }

    #[tokio::test]
    async fn late_accounting_refreshes_descendant_budgets_and_durable_execution_state() {
        let (db, root) = fixture();
        let mut request: engine_core::ChatCompletionRequest =
        serde_json::from_value(serde_json::json!({
            "model":"test","messages":[{"role":"user","content":"Implement a durable continuation"}]
        }))
        .unwrap();
        let task = engine_core::resolve_task(Some(root.id), "first", &request, &[]);
        engine_core::continuation::prepare(&db, Some(root.id), &task, "first", &mut request)
            .await
            .unwrap();
        db.record_accounting(&record(root.id, 10)).await.unwrap();
        let child = db
            .rollover_session(
                db.get_session(root.id).unwrap().unwrap(),
                "test".into(),
                100,
                None,
            )
            .await
            .unwrap();
        db.record_accounting(&record(child.id, 7)).await.unwrap();
        let grandchild = db
            .rollover_session(
                db.get_session(child.id).unwrap().unwrap(),
                "test".into(),
                100,
                None,
            )
            .await
            .unwrap();
        let mut late = record(root.id, 20);
        late.payload["task_id"] = serde_json::json!(task.task_id);
        late.payload["response_message"] = serde_json::json!({"role":"assistant","tool_calls":[{
        "id":"late-write","type":"function","function":{"name":"write_file","arguments":"{\"path\":\"a.rs\"}"}}]});
        db.record_accounting(&late).await.unwrap();
        db.record_accounting(&late).await.unwrap();
        assert_eq!(db.get_session(root.id).unwrap().unwrap().spend_cents, 30);
        let saved = db.get_session(child.id).unwrap().unwrap();
        assert_eq!(saved.spend_cents, 7);
        assert_eq!(saved.inherited_spend_cents, 30);
        let saved = db.get_session(grandchild.id).unwrap().unwrap();
        assert_eq!(saved.spend_cents, 0);
        assert_eq!(saved.inherited_spend_cents, 37);
        assert_eq!(saved.inherited_spend_nano_usd, Some(370_000_000));
        assert_eq!(db.total_spend(root.id).await.unwrap(), 30);
        assert_eq!(db.total_spend(child.id).await.unwrap(), 7);
        assert_eq!(db.total_spend(grandchild.id).await.unwrap(), 0);
        // No post-accounting observer was called: settlement itself must checkpoint
        // the proposed action, both on its source and the live continuation.
        for id in [root.id, grandchild.id] {
            let state: engine_core::continuation::Continuation = serde_json::from_value(
                db.read_latest_continuation(id, &task.task_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .payload,
            )
            .unwrap();
            assert_eq!(
                state
                    .tools
                    .iter()
                    .find(|call| call.tool_call_id == "late-write")
                    .unwrap()
                    .status,
                engine_core::continuation::ToolStatus::Pending
            );
            assert_eq!(state.completion_source_session_id, Some(root.id));
        }
    }
    #[tokio::test]
    async fn concurrent_completions_and_duplicate_retries_reconcile_exactly() {
        let (db, session) = fixture();
        let mut tasks = Vec::new();
        for cost in 1..=24 {
            let record = record(session.id, cost);
            for _ in 0..3 {
                let db = db.clone();
                let record = record.clone();
                tasks.push(tokio::spawn(async move {
                    db.record_accounting(&record).await.unwrap();
                }));
            }
        }
        for task in tasks {
            task.await.unwrap();
        }
        let session = db.get_session(session.id).unwrap().unwrap();
        assert_eq!(session.spend_cents, 300);
        assert_eq!(db.total_spend(session.id).await.unwrap(), 300);
        assert_eq!(session.total_input_tokens, 24 * 12);
        assert_eq!(session.total_output_tokens, 24 * 3);
        assert_eq!(session.cost_accuracy_samples, Some(24));
        assert_eq!(session.cache_hit_rate_samples, Some(24));
        let events = db.read_events(session.id, None, None).await.unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|e| e.event_type == SessionEventType::LlmCall)
                .count(),
            24
        );
        assert_eq!(events.iter().filter_map(|e| e.cost_cents).sum::<i64>(), 300);
        let conn = db.conn.lock().unwrap();
        assert_eq!(
            conn.query_row(
                "SELECT COUNT(DISTINCT accounting_id) FROM ledger_events",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            24
        );
    }

    #[tokio::test]
    async fn late_ancestor_tier_change_notifies_descendants_without_resetting_budget() {
        let (db, root) = fixture();
        let child = db
            .rollover_session(root.clone(), "test".into(), 100, None)
            .await
            .unwrap();
        let before = db.get_session(child.id).unwrap().unwrap().updated_at;
        let mut late = record(root.id, 7000);
        late.recorded_at = before - chrono::Duration::minutes(1);
        db.record_accounting(&late).await.unwrap();
        db.record_accounting(&late).await.unwrap();
        let saved = db.get_session(child.id).unwrap().unwrap();
        assert_eq!(saved.budget_cents, 10000);
        assert_eq!(saved.spend_cents, 0);
        assert_eq!(saved.tier, 2);
        assert!(saved.updated_at >= before);
        let notifications: Vec<_> = db
            .read_events(child.id, None, None)
            .await
            .unwrap()
            .into_iter()
            .filter(|event| event.event_type == SessionEventType::TierChange)
            .collect();
        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].payload["reason"], "ancestor_accounting");
        assert_eq!(notifications[0].payload["spend_cents"], 7000);
    }
    #[tokio::test]
    async fn fractional_classification_roundtrips_and_settles_exactly_once() {
        let (db, session) = fixture();
        let mut entry = record(session.id, 20);
        // This f32 becomes an f64 in the JSON payload. Without float_roundtrip,
        // parsing its serialized value changes it and falsely reports an ID conflict.
        entry.payload["classification"] = serde_json::json!({
            "probabilities": [0.00016_f32, 0.5_f32, 0.49984_f32]
        });

        db.record_accounting(&entry).await.unwrap();
        db.record_accounting(&entry).await.unwrap();

        let mut conflict = entry.clone();
        conflict.payload["input_tokens"] = serde_json::json!(13);
        assert!(matches!(
            db.record_accounting(&conflict).await,
            Err(EngineError::InvalidInput(_))
        ));

        let saved = db.get_session(session.id).unwrap().unwrap();
        assert_eq!(saved.spend_cents, 20);
        assert_eq!(saved.total_input_tokens, 12);
        assert_eq!(saved.total_output_tokens, 3);
        assert_eq!(db.total_spend(session.id).await.unwrap(), 20);
        let events = db.read_events(session.id, None, None).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, SessionEventType::LlmCall);
        let conn = db.conn.lock().unwrap();
        let (encoded, settled): (String, i64) = conn
            .query_row(
                "SELECT record, settled FROM completion_accounting WHERE id=?1",
                [entry.id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            serde_json::from_str::<CompletionAccounting>(&encoded).unwrap(),
            entry
        );
        assert_eq!(settled, 1);
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM ledger_events", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1
        );
    }

    #[tokio::test]
    async fn conflict_is_rejected_and_stale_accounting_preserves_lifecycle_and_totals() {
        let (db, mut session) = fixture();
        let stale = session.clone();
        let record = record(session.id, 20);
        db.stage_accounting(&record).await.unwrap();
        session.status = SessionStatus::Completed;
        session.task_summary = "finished".into();
        db.insert_session(&session).unwrap();
        db.settle_accounting(record.id).await.unwrap();
        let saved = db.get_session(session.id).unwrap().unwrap();
        assert_eq!(saved.status, SessionStatus::Completed);
        assert_eq!(saved.task_summary, "finished");
        assert_eq!(saved.spend_cents, 20);
        // An unrelated stale session write cannot reset settled financial totals.
        db.insert_session(&stale).unwrap();
        assert_eq!(db.get_session(session.id).unwrap().unwrap().spend_cents, 20);
        let mut conflict = record.clone();
        conflict.cost_cents += 1;
        assert!(db.record_accounting(&conflict).await.is_err());
        assert_eq!(db.total_spend(session.id).await.unwrap(), 20);
    }
    #[tokio::test]
    async fn failed_transaction_rolls_back_events_and_ledger_then_recovery_settles_once() {
        let (db, session) = fixture();
        let record = record(session.id, 30);
        db.conn.lock().unwrap().execute_batch("CREATE TRIGGER fail_ledger BEFORE INSERT ON ledger_events BEGIN SELECT RAISE(ABORT,'injected failure'); END;").unwrap();
        assert!(db.record_accounting(&record).await.is_err());
        assert!(db
            .read_events(session.id, None, None)
            .await
            .unwrap()
            .is_empty());
        assert_eq!(db.total_spend(session.id).await.unwrap(), 0);
        assert_eq!(db.get_session(session.id).unwrap().unwrap().spend_cents, 0);
        db.conn
            .lock()
            .unwrap()
            .execute_batch("DROP TRIGGER fail_ledger")
            .unwrap();
        db.recover_accounting().await.unwrap();
        db.recover_accounting().await.unwrap();
        assert_eq!(db.total_spend(session.id).await.unwrap(), 30);
        assert_eq!(db.get_session(session.id).unwrap().unwrap().spend_cents, 30);
        assert_eq!(
            db.read_events(session.id, None, None).await.unwrap().len(),
            1
        );
    }
    #[tokio::test]
    async fn tier_change_and_zero_usage_are_atomic_and_idempotent() {
        let (db, mut session) = fixture();
        session.budget_cents = 100;
        db.insert_session(&session).unwrap();
        let entry = record(session.id, 80);
        db.record_accounting(&entry).await.unwrap();
        db.record_accounting(&entry).await.unwrap();
        let events = db.read_events(session.id, None, None).await.unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|e| e.event_type == SessionEventType::TierChange)
                .count(),
            1
        );
        let free = record(session.id, 0);
        db.record_accounting(&free).await.unwrap();
        assert_eq!(db.total_spend(session.id).await.unwrap(), 80);
        assert_eq!(
            db.get_session(session.id)
                .unwrap()
                .unwrap()
                .total_input_tokens,
            24
        );
    }
    #[test]
    fn crash_worker() {
        let Ok(path) = std::env::var("REINCORE_ACCOUNTING_CRASH_TEST") else {
            return;
        };
        let db = LocalDatabase::open(&path).unwrap();
        let record: CompletionAccounting =
            serde_json::from_str(&std::env::var("REINCORE_ACCOUNTING_TEST_RECORD").unwrap())
                .unwrap();
        db.stage_accounting_sync(&record).unwrap();
        std::fs::write(format!("{path}.ready"), "durable").unwrap();
        loop {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
    #[test]
    fn killed_process_recovers_staged_usage_on_reopen_without_double_billing() {
        let path = std::env::temp_dir().join(format!("reincore-accounting-{}.db", Uuid::new_v4()));
        let session = Session::new("crash test", 1000).unwrap();
        let mut entry = record(session.id, 42);
        entry.payload = serde_json::json!({"input_tokens":420001,"output_tokens":0});
        entry.cost = Some(engine_core::pricing::calculate(
            "fixture",
            "m",
            "v1",
            engine_core::pricing::TokenRates {
                input: 1000,
                output: 0,
                cache_read: None,
                cache_write: None,
                cache_write_1h: None,
                batch_input: None,
                batch_output: None,
                ..Default::default()
            },
            &engine_core::TokenUsage {
                prompt_tokens: 420001,
                ..Default::default()
            },
            false,
        ));
        {
            let db = LocalDatabase::open(&path).unwrap();
            db.insert_session(&session).unwrap();
        }
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "db::accounting::tests::crash_worker",
                "--nocapture",
            ])
            .env("REINCORE_ACCOUNTING_CRASH_TEST", &path)
            .env(
                "REINCORE_ACCOUNTING_TEST_RECORD",
                serde_json::to_string(&entry).unwrap(),
            )
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = command.spawn().unwrap();
        let ready = format!("{}.ready", path.display());
        for _ in 0..400 {
            if std::path::Path::new(&ready).exists() {
                break;
            }
            if child.try_wait().unwrap().is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        let staged = std::path::Path::new(&ready).exists();
        let _ = child.kill();
        child.wait().unwrap();
        assert!(staged, "child did not stage usage");
        for _ in 0..2 {
            let db = LocalDatabase::open(&path).unwrap();
            assert_eq!(db.get_session(session.id).unwrap().unwrap().spend_cents, 42);
            assert_eq!(
                db.get_session(session.id).unwrap().unwrap().spend_nano_usd,
                Some(420001000)
            );
            let conn = db.conn.lock().unwrap();
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM ledger_events", [], |r| r
                    .get::<_, i64>(0))
                    .unwrap(),
                1
            );
            assert_eq!(
                conn.query_row(
                    "SELECT COUNT(*) FROM session_events WHERE event_type='llm_call'",
                    [],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
                1
            );
        }
        for suffix in ["", "-wal", "-shm", ".ready"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }

    #[tokio::test]
    async fn tiny_calls_preserve_exact_event_ledger_session_and_timeline_totals() {
        let (db, session) = fixture();
        let mut jobs = vec![];
        for _ in 0..100 {
            let mut r = record(session.id, 0);
            r.payload["input_tokens"] = serde_json::json!(100);
            r.payload["output_tokens"] = serde_json::json!(0);
            r.cost = Some(engine_core::pricing::calculate(
                "fixture",
                "text",
                "v1",
                engine_core::pricing::TokenRates {
                    input: 1000,
                    output: 0,
                    cache_read: None,
                    cache_write: None,
                    cache_write_1h: None,
                    batch_input: None,
                    batch_output: None,
                    ..Default::default()
                },
                &engine_core::TokenUsage {
                    prompt_tokens: 100,
                    ..Default::default()
                },
                false,
            ));
            let db = db.clone();
            jobs.push(tokio::spawn(async move {
                db.record_accounting(&r).await.unwrap();
                db.record_accounting(&r).await.unwrap();
            }));
        }
        for job in jobs {
            job.await.unwrap();
        }
        let saved = db.get_session(session.id).unwrap().unwrap();
        assert_eq!(saved.spend_nano_usd, Some(10_000_000));
        assert_eq!(saved.spend_cents, 1);
        assert_eq!(db.total_spend(session.id).await.unwrap(), 1);
        let events = db.read_events(session.id, None, None).await.unwrap();
        assert_eq!(
            events
                .iter()
                .filter_map(|e| e.amount_nano_usd())
                .sum::<i64>(),
            10_000_000
        );
        let timeline = db.read_timeline_events(session.id).await.unwrap();
        assert_eq!(
            timeline
                .iter()
                .filter_map(|e| e.amount_nano_usd())
                .sum::<i64>(),
            10_000_000
        );
        let ledger: i64 = db
            .conn
            .lock()
            .unwrap()
            .query_row("SELECT SUM(amount_nano_usd) FROM ledger_events", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(ledger, 10_000_000);
        // An unrelated stale write must not erase the fractional accumulation.
        db.insert_session(&session).unwrap();
        assert_eq!(
            db.get_session(session.id).unwrap().unwrap().spend_nano_usd,
            Some(10_000_000)
        );
    }
    #[tokio::test]
    async fn unknown_auxiliary_usage_is_audited_without_changing_conversation_context() {
        let (db, mut session) = fixture();
        session.context_tokens = 900;
        db.insert_session(&session).unwrap();
        let mut r = record(session.id, 0);
        r.cost = Some(engine_core::pricing::CostEstimate::unknown(
            "p",
            "m",
            "usage missing",
        ));
        r.purpose = Some("title".into());
        r.payload = serde_json::json!({"input_tokens":0,"output_tokens":0});
        db.record_accounting(&r).await.unwrap();
        let saved = db.get_session(session.id).unwrap().unwrap();
        assert_eq!(saved.context_tokens, 900);
        assert_eq!(saved.unknown_cost_calls, 1);
        let events = db.read_events(session.id, None, None).await.unwrap();
        assert_eq!(events[0].event_type, SessionEventType::AuxiliaryCall);
        assert_eq!(events[0].cost_cents, None);
        assert_eq!(events[0].amount_nano_usd(), None);
        assert_eq!(saved.cost_accuracy_samples, None);
    }

    #[tokio::test]
    async fn fractional_spend_survives_legacy_upgrade_and_rollover() {
        let (db, mut session) = fixture();
        session.spend_cents = 7;
        session.budget_cents = 10;
        db.insert_session(&session).unwrap();
        let mut receipt = record(session.id, 0);
        receipt.payload = serde_json::json!({"input_tokens":100,"output_tokens":0});
        receipt.cost = Some(engine_core::pricing::calculate(
            "fixture",
            "m",
            "v1",
            engine_core::pricing::TokenRates {
                input: 1000,
                output: 0,
                cache_read: None,
                cache_write: None,
                cache_write_1h: None,
                batch_input: None,
                batch_output: None,
                ..Default::default()
            },
            &engine_core::TokenUsage {
                prompt_tokens: 100,
                ..Default::default()
            },
            false,
        ));
        db.record_accounting(&receipt).await.unwrap();
        let mut saved = db.get_session(session.id).unwrap().unwrap();
        assert_eq!(saved.spend_nano_usd, Some(70_100_000));
        assert_eq!(saved.tier, 2);
        let child = saved.rollover();
        db.insert_session(&child).unwrap();
        let child = db.get_session(child.id).unwrap().unwrap();
        assert_eq!(child.lineage_spend_nano_usd(), 70_100_000);
        assert!((child.spend_ratio() - 0.701).abs() < 1e-12);
    }
}
