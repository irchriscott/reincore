//! Tests persistence for the local engine.

use chrono::Utc;
use engine_core::{
    error::EngineError,
    event::{EventStore, SessionEventType},
    models::ChatMessage,
    session::Session,
    store::{CacheStore, IndexedToolState, ToolStateStore},
};
use uuid::Uuid;

use super::*;

#[tokio::test]
async fn file_history_survives_database_restart_and_compaction() {
    let path =
        std::env::temp_dir().join(format!("reincore-file-history-{}.sqlite", Uuid::new_v4()));
    let session = Session::new("file history recovery", 100).unwrap();
    {
        let db = LocalDatabase::open(&path).unwrap();
        db.insert_session(&session).unwrap();
        db.append_event(session.id,SessionEventType::ToolCall,serde_json::json!({"tool_call":{
            "id":"write-a","type":"function","function":{"name":"write_file","arguments":serde_json::json!({"path":"src/a.rs","content":"fn preserved() {}\n"}).to_string()}
        }}),None).await.unwrap();
        db.append_event(
            session.id,
            SessionEventType::ToolResult,
            serde_json::json!({"result":{
            "tool_call_id":"write-a","name":"write_file","content":"success"},"cache_hit":false}),
            None,
        )
        .await
        .unwrap();
        db.append_event(session.id,SessionEventType::Summary,serde_json::json!({"summary_text":"Compacted","summarized_seq_start":1,"summarized_seq_end":2,"mode":"low"}),None).await.unwrap();
    }
    {
        let db = LocalDatabase::open(&path).unwrap();
        let history = db
            .read_file_history(session.id, &["./src\\a.rs".into()])
            .await
            .unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(
            history[0].status,
            engine_core::continuation::ToolStatus::Succeeded
        );
        assert_eq!(history[0].excerpt.as_deref(), Some("fn preserved() {}\n"));
        assert!(db
            .read_file_history(session.id, &["other/a.rs".into()])
            .await
            .unwrap()
            .is_empty());
        let mut request: engine_core::ChatCompletionRequest = serde_json::from_value(serde_json::json!({"model":"test","messages":[
            {"role":"user","content":"Read the file again"},
            {"role":"assistant","tool_calls":[{"id":"read-a","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"src/a.rs\"}"}}]}
        ],"reincore_file_versions":[{"path":"src/a.rs","version":10,"content_hash":history[0].after_hash,"content":null}],
        "reincore_file_evidence":[{"tool_call_id":"write-a","path":"src/a.rs","version":10,"content_hash":history[0].after_hash,"content":"fn preserved() {}\n"}]})).unwrap();
        let task = engine_core::resolve_task(Some(session.id), "read-group", &request, &[]);
        engine_core::continuation::prepare(
            &db,
            Some(session.id),
            &task,
            "read-group",
            &mut request,
        )
        .await
        .unwrap();
        engine_core::continuation::attach(&mut request);
        let recovered = request
            .messages
            .iter()
            .find(|m| m.name.as_deref() == Some(engine_core::file_history::DATA_NAME))
            .unwrap()
            .full_text();
        assert!(recovered.contains("fn preserved()"));
        assert!(recovered.contains("\"matches_reported_current_hash\":true"));
        let state: engine_core::continuation::Continuation = serde_json::from_value(
            db.read_latest_continuation(session.id, &task.task_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        assert_eq!(state.files.len(), 1);
        request.extra.insert("reincore_file_versions".into(),serde_json::json!([{"path":"src/a.rs","version":11,"content_hash":"changed","content":null}]));
        engine_core::continuation::prepare(
            &db,
            Some(session.id),
            &task,
            "read-group",
            &mut request,
        )
        .await
        .unwrap();
        engine_core::continuation::attach(&mut request);
        let recovered = request
            .messages
            .iter()
            .find(|m| m.name.as_deref() == Some(engine_core::file_history::DATA_NAME))
            .unwrap()
            .full_text();
        assert!(recovered.contains("\"matches_reported_current_hash\":false"));
        let state: engine_core::continuation::Continuation = serde_json::from_value(
            db.read_latest_continuation(session.id, &task.task_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        assert!(state.files.is_empty());
        let stmt = db.conn.lock().unwrap();
        let plan = stmt.prepare("EXPLAIN QUERY PLAN SELECT payload FROM session_file_history WHERE session_id='x' AND path='src/a.rs' ORDER BY seq DESC LIMIT 4").unwrap()
            .query_map([],|row| row.get::<_,String>(3)).unwrap().collect::<Result<Vec<_>,_>>().unwrap().join(" ");
        assert!(plan.contains("idx_session_file_history_path"), "{plan}");
        assert!(!plan.contains("TEMP B-TREE"), "{plan}");
    }
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(path.with_extension("sqlite-wal"));
    let _ = std::fs::remove_file(path.with_extension("sqlite-shm"));
}

#[tokio::test]
async fn file_history_concurrent_writes_replays_failures_and_call_id_collisions() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("concurrent file changes", 100).unwrap();
    db.insert_session(&session).unwrap();
    let mut workers = vec![];
    for index in 0..8 {
        let db = db.clone();
        let id = session.id;
        workers.push(tokio::spawn(async move {
            db.append_event(id,SessionEventType::ToolCall,serde_json::json!({"tool_call":{
                "id":format!("write-{index}"),"type":"function","function":{"name":"write_file","arguments":serde_json::json!({"path":"a.rs","content":format!("body-{index}")}).to_string()}
            }}),None).await.unwrap();
            let payload = serde_json::json!({"result":{"tool_call_id":format!("write-{index}"),"name":"write_file","content":if index==7 { "failed: disk error" } else { "success" }},"cache_hit":false});
            db.append_event(id,SessionEventType::ToolResult,payload.clone(),None).await.unwrap();
            db.append_event(id,SessionEventType::ToolResult,payload,None).await.unwrap();
        }));
    }
    for worker in workers {
        worker.await.unwrap();
    }
    let history = db
        .read_file_history(session.id, &["a.rs".into()])
        .await
        .unwrap();
    assert_eq!(history.len(), engine_core::file_history::PER_FILE_LIMIT);
    assert!(history
        .windows(2)
        .all(|pair| pair[0].result_seq >= pair[1].result_seq));
    assert!(history.iter().all(|r| !r.ambiguous));
    let first = history[0].tool_call_id.clone();
    db.append_event(session.id,SessionEventType::ToolCall,serde_json::json!({"tool_call":{
        "id":first,"type":"function","function":{"name":"read_file","arguments":"{\"path\":\"b.rs\"}"}
    }}),None).await.unwrap();
    let history = db
        .read_file_history(session.id, &["a.rs".into()])
        .await
        .unwrap();
    let collided = history.iter().find(|r| r.tool_call_id == first).unwrap();
    assert!(collided.ambiguous);
    assert_eq!(
        collided.status,
        engine_core::continuation::ToolStatus::Unknown
    );
    assert!(collided.after_hash.is_none());
    assert!(db
        .read_file_history(Uuid::new_v4(), &["a.rs".into()])
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn file_history_replayed_conflicting_identity_is_permanently_disqualified() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("replayed file mutation", 100).unwrap();
    db.insert_session(&session).unwrap();
    let mut messages: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
        {"role":"assistant","tool_calls":[{"id":"write","type":"function","function":{"name":"write_file","arguments":"{\"path\":\"a.rs\",\"content\":\"saved\"}"}}]},
        {"role":"tool","tool_call_id":"write","content":"success"}
    ])).unwrap();
    engine_core::external_tools::record_external_tool_executions_with_file_evidence(
        &db,
        session.id,
        &messages,
        "group",
        None,
        &[],
    )
    .await
    .unwrap();
    assert_eq!(
        db.read_file_history(session.id, &["a.rs".into()])
            .await
            .unwrap()[0]
            .status,
        engine_core::continuation::ToolStatus::Succeeded
    );
    messages[0].tool_calls.as_mut().unwrap()[0]
        .function
        .arguments = "{\"path\":\"b.rs\",\"content\":\"different\"}".into();
    engine_core::external_tools::record_external_tool_executions_with_file_evidence(
        &db,
        session.id,
        &messages,
        "group",
        None,
        &[],
    )
    .await
    .unwrap();
    assert!(
        db.read_file_history(session.id, &["a.rs".into()])
            .await
            .unwrap()[0]
            .ambiguous
    );
    assert!(db
        .read_file_history(session.id, &["b.rs".into()])
        .await
        .unwrap()
        .is_empty());
    messages[0].tool_calls.as_mut().unwrap()[0]
        .function
        .arguments = "{\"path\":\"a.rs\",\"content\":\"saved\"}".into();
    engine_core::external_tools::record_external_tool_executions_with_file_evidence(
        &db,
        session.id,
        &messages,
        "group",
        None,
        &[],
    )
    .await
    .unwrap();
    assert!(
        db.read_file_history(session.id, &["a.rs".into()])
            .await
            .unwrap()[0]
            .ambiguous
    );
}

#[tokio::test]
async fn file_history_conflicting_replayed_result_stays_unknown() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("conflicting file result", 100).unwrap();
    db.insert_session(&session).unwrap();
    let mut messages: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
        {"role":"assistant","tool_calls":[{"id":"write","type":"function","function":{"name":"write_file","arguments":"{\"path\":\"a.rs\",\"content\":\"saved\"}"}}]},
        {"role":"tool","tool_call_id":"write","content":"success"}
    ])).unwrap();
    for _ in 0..2 {
        engine_core::external_tools::record_external_tool_executions_with_file_evidence(
            &db,
            session.id,
            &messages,
            "group",
            None,
            &[],
        )
        .await
        .unwrap();
    }
    assert!(
        !db.read_file_history(session.id, &["a.rs".into()])
            .await
            .unwrap()[0]
            .ambiguous
    );
    messages[1].content = Some(engine_core::MessageContent::text(
        "failed: contradictory result",
    ));
    engine_core::external_tools::record_external_tool_executions_with_file_evidence(
        &db,
        session.id,
        &messages,
        "group",
        None,
        &[],
    )
    .await
    .unwrap();
    assert_eq!(
        db.read_file_history(session.id, &["a.rs".into()])
            .await
            .unwrap()[0]
            .status,
        engine_core::continuation::ToolStatus::Unknown
    );
}

#[tokio::test]
#[ignore = "opt-in local continuation database performance measurement"]
async fn benchmark_continuation_database_lookups() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("lookup benchmark", 100).unwrap();
    db.insert_session(&session).unwrap();
    {
        let mut conn = db.conn.lock().unwrap();
        let tx = conn.transaction().unwrap();
        let mut stmt = tx.prepare("INSERT INTO session_events(session_id,seq,event_type,payload,created_at) VALUES(?1,?2,?3,?4,?5)").unwrap();
        for seq in 1..=25000 {
            let (kind, payload) = if seq == 1 {
                ("continuation", serde_json::json!({"task_id":"task"}))
            } else {
                (
                    "tool_result",
                    serde_json::json!({"file":{"path":format!("{}.rs",seq%2500),"version":seq,"content_hash":"hash","content":"known evidence"}}),
                )
            };
            stmt.execute(rusqlite::params![
                session.id.to_string(),
                seq,
                kind,
                payload.to_string(),
                "2026-10-01T00:00:00Z"
            ])
            .unwrap();
        }
        drop(stmt);
        tx.commit().unwrap();
    }
    let mut samples = Vec::new();
    for _ in 0..100 {
        let start = std::time::Instant::now();
        assert_eq!(
            db.read_latest_continuation(session.id, "task")
                .await
                .unwrap()
                .unwrap()
                .seq,
            1
        );
        assert_eq!(
            db.read_files_by_paths(session.id, &["7.rs".into(), "8.rs".into()])
                .await
                .unwrap()
                .len(),
            2
        );
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(f64::total_cmp);
    eprintln!(
        "SQLite lookup benchmark events=25000 p50_ms={:.3} p95_ms={:.3} max_ms={:.3}",
        samples[50], samples[95], samples[99]
    );
}

#[tokio::test]
#[ignore = "opt-in portable file history scalability measurement"]
async fn benchmark_portable_file_history_lookups() {
    use engine_core::file_history::{FileHistoryScope, FileRoot};
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("file history benchmark", 100).unwrap();
    db.insert_session(&session).unwrap();
    let scope = FileHistoryScope {
        project_id: session.id,
        roots: vec![FileRoot {
            id: "project".into(),
            path: "/repo".into(),
        }],
    };
    db.bind_file_history_scope(session.id, &scope)
        .await
        .unwrap();
    let paths: Vec<String> = (0..16).map(|i| format!("/repo/{i}.rs")).collect();
    let mut counts = Vec::new();
    for target in [1000, 10000, 100000] {
        {
            let mut conn = db.conn.lock().unwrap();
            let tx = conn.transaction().unwrap();
            let mut links = tx.prepare("INSERT INTO session_tool_call_links(session_id,tool_call_id,event_seq,tool_call,requested_at) VALUES(?1,?2,?3,'{}','2026-10-01T00:00:00Z')").unwrap();
            let mut rows = tx.prepare("INSERT INTO session_file_history(session_id,tool_call_id,path,seq,payload,file_key) VALUES(?1,?2,?3,?4,?5,?6)").unwrap();
            let start = counts.last().copied().unwrap_or(0);
            for i in start..target {
                let path = &paths[i % 16];
                let id = format!("write-{i:06}");
                let call:engine_core::ToolCall = serde_json::from_value(serde_json::json!({"id":id,"type":"function","function":{"name":"write_file","arguments":serde_json::json!({"path":path,"content":"benchmark data"}).to_string()}})).unwrap();
                let mut record =
                    engine_core::file_history::records_for_call(session.id, i as i64 + 1, &call)
                        .remove(0);
                record.file_id = scope.identity(path);
                links
                    .execute(rusqlite::params![session.id.to_string(), id, i as i64 + 1])
                    .unwrap();
                rows.execute(rusqlite::params![
                    session.id.to_string(),
                    id,
                    path,
                    i as i64 + 1,
                    serde_json::to_string(&record).unwrap(),
                    record.file_id.as_ref().unwrap().key()
                ])
                .unwrap();
            }
            drop(links);
            drop(rows);
            tx.commit().unwrap();
        }
        counts.push(target);
        {
            let conn = db.read_conn();
            let key = scope.identity(&paths[0]).unwrap().key();
            let mut statement = conn
                .prepare(super::events::FILE_HISTORY_LOOKUP_SQL)
                .unwrap();
            let rows = statement
                .query_map(
                    rusqlite::params![session.id.to_string(), paths[0], 4, key],
                    |row| row.get::<_, String>(0),
                )
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(rows.len(), 4);
            eprintln!(
                "Lookup VM steps rows={target}: {}",
                statement.get_status(rusqlite::StatementStatus::VmStep)
            );
            assert!(
                statement.get_status(rusqlite::StatementStatus::VmStep) < 1000,
                "history lookup work must remain bounded"
            );
            if target == 100000 {
                let mut plan = conn
                    .prepare(&format!(
                        "EXPLAIN QUERY PLAN {}",
                        super::events::FILE_HISTORY_LOOKUP_SQL
                    ))
                    .unwrap();
                let detail = plan
                    .query_map(
                        rusqlite::params![session.id.to_string(), paths[0], 4, key],
                        |row| row.get::<_, String>(3),
                    )
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap()
                    .join("\n");
                eprintln!("Lookup query plan:\n{detail}");
            }
        }
        let mut samples = Vec::new();
        let mut bind_samples = Vec::new();
        for run in 0..110 {
            let started = std::time::Instant::now();
            assert_eq!(
                db.read_file_history(session.id, &paths)
                    .await
                    .unwrap()
                    .len(),
                64
            );
            if run >= 10 {
                samples.push(started.elapsed().as_secs_f64() * 1000.0);
            }
            let started = std::time::Instant::now();
            db.bind_file_history_scope(session.id, &scope)
                .await
                .unwrap();
            if run >= 10 {
                bind_samples.push(started.elapsed().as_secs_f64() * 1000.0);
            }
        }
        samples.sort_by(f64::total_cmp);
        bind_samples.sort_by(f64::total_cmp);
        eprintln!("File history SQLite benchmark rows={target} paths=16 returned=64 lookup_p50_ms={:.3} lookup_p95_ms={:.3} bind_p95_ms={:.3}",samples[50],samples[95],bind_samples[95]);
    }
}

#[tokio::test]
async fn file_history_upgrade_and_batched_backfill_keep_lookup_work_bounded() {
    use engine_core::file_history::{FileHistoryScope, FileRoot};
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("history upgrade", 100).unwrap();
    db.insert_session(&session).unwrap();
    {
        let mut conn = db.conn.lock().unwrap();
        // Reproduce the populated schema immediately before the index upgrade.
        conn.execute_batch("DROP INDEX idx_session_file_history_legacy; DROP INDEX idx_session_file_history_identity; CREATE INDEX idx_session_file_history_identity ON session_file_history(session_id,file_key,seq DESC); DELETE FROM schema_migrations WHERE version=4;").unwrap();
        let tx = conn.transaction().unwrap();
        for i in 0..2048 {
            let path = if i % 2 == 0 {
                "/repo/a.rs"
            } else {
                "/other/a.rs"
            };
            let id = format!("write-{i:06}");
            let call:engine_core::ToolCall = serde_json::from_value(serde_json::json!({"id":id,"type":"function","function":{"name":"write_file","arguments":serde_json::json!({"path":path,"content":"data"}).to_string()}})).unwrap();
            let record =
                engine_core::file_history::records_for_call(session.id, i + 1, &call).remove(0);
            tx.execute("INSERT INTO session_tool_call_links(session_id,tool_call_id,event_seq,tool_call,requested_at) VALUES(?1,?2,?3,'{}','2026-10-01T00:00:00Z')",rusqlite::params![session.id.to_string(),id,i+1]).unwrap();
            tx.execute("INSERT INTO session_file_history(session_id,tool_call_id,path,seq,payload) VALUES(?1,?2,?3,?4,?5)",rusqlite::params![session.id.to_string(),id,path,i+1,serde_json::to_string(&record).unwrap()]).unwrap();
        }
        tx.commit().unwrap();
    }
    db.migrate().unwrap();
    db.migrate().unwrap();
    let scope = FileHistoryScope {
        project_id: session.id,
        roots: vec![FileRoot {
            id: "project".into(),
            path: "/repo".into(),
        }],
    };
    db.bind_file_history_scope(session.id, &scope)
        .await
        .unwrap();
    let conn = db.read_conn();
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*),COUNT(file_key) FROM session_file_history WHERE session_id=?1",
            [session.id.to_string()],
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
        )
        .unwrap(),
        (2048, 1024)
    );
    for path in ["/repo/a.rs", "/other/a.rs"] {
        let key = scope.identity(path).map(|id| id.key());
        let mut statement = conn
            .prepare(super::events::FILE_HISTORY_LOOKUP_SQL)
            .unwrap();
        let rows = statement
            .query_map(
                rusqlite::params![session.id.to_string(), path, 4, key],
                |r| r.get::<_, String>(0),
            )
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(rows.len(), 4);
        assert!(
            statement.get_status(rusqlite::StatementStatus::VmStep) < 1000,
            "lookup must not scan older rows for either identity type"
        );
        let newest: engine_core::file_history::FileWriteRecord =
            serde_json::from_str(&rows[0]).unwrap();
        assert_eq!(
            newest.tool_call_id,
            if path.starts_with("/repo/") {
                "write-002046"
            } else {
                "write-002047"
            }
        );
    }
}

#[tokio::test]
async fn continuation_cas_rejects_stale_writers_and_concurrent_prepares_merge() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("concurrent continuation", 100).unwrap();
    db.insert_session(&session).unwrap();
    let mut initial: engine_core::ChatCompletionRequest =
        serde_json::from_value(serde_json::json!({
            "model":"test","messages":[{"role":"user","content":"Implement reliable continuation"}]
        }))
        .unwrap();
    let task = engine_core::resolve_task(Some(session.id), "group", &initial, &[]);
    engine_core::continuation::prepare(&db, Some(session.id), &task, "group", &mut initial)
        .await
        .unwrap();
    let previous = db
        .read_latest_continuation(session.id, &task.task_id)
        .await
        .unwrap()
        .unwrap();
    let mut changed = previous.payload.clone();
    changed["latest_request"] = serde_json::json!("new evidence");
    assert!(db
        .compare_and_append_continuation(session.id, &task.task_id, Some(previous.seq), changed)
        .await
        .unwrap()
        .is_some());
    assert!(db
        .compare_and_append_continuation(
            session.id,
            &task.task_id,
            Some(previous.seq),
            previous.payload
        )
        .await
        .unwrap()
        .is_none());
    let mut workers = vec![];
    for index in 0..4 {
        let db = db.clone();
        let task = task.clone();
        let mut request = initial.clone();
        request.messages.push(serde_json::from_value(serde_json::json!({"role":"assistant","tool_calls":[{
            "id":format!("write-{index}"),"type":"function","function":{"name":"write_file","arguments":serde_json::json!({"path":format!("{index}.rs")}).to_string()}}]})).unwrap());
        workers.push(tokio::spawn(async move {
            engine_core::continuation::prepare(&db, Some(session.id), &task, "group", &mut request)
                .await
                .unwrap();
        }));
    }
    for worker in workers {
        worker.await.unwrap();
    }
    let state: engine_core::continuation::Continuation = serde_json::from_value(
        db.read_latest_continuation(session.id, &task.task_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(state.tools.len(), 4);
    assert!(state
        .tools
        .iter()
        .all(|call| call.status == engine_core::continuation::ToolStatus::Pending));
}

#[tokio::test]
async fn bounded_evidence_queries_use_indexes_and_only_return_requested_paths() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("targeted lookup", 100).unwrap();
    db.insert_session(&session).unwrap();
    for (path, version) in [("a.rs", 1), ("b.rs", 1), ("a.rs", 2)] {
        db.append_event(session.id,SessionEventType::ToolResult,serde_json::json!({"file":{
            "path":path,"version":version,"content_hash":format!("hash-{version}"),"content":"bounded evidence"}}),None).await.unwrap();
    }
    let files = db
        .read_files_by_paths(session.id, &["a.rs".into()])
        .await
        .unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].version, 2);
    assert!(db
        .read_files_by_paths(session.id, &[])
        .await
        .unwrap()
        .is_empty());
    let conn = db.conn.lock().unwrap();
    for (sql,index) in [
        ("EXPLAIN QUERY PLAN SELECT seq FROM session_events WHERE session_id='x' AND event_type='continuation' AND json_extract(payload,'$.task_id')='task' ORDER BY seq DESC LIMIT 1", "idx_continuation_task"),
        ("EXPLAIN QUERY PLAN SELECT seq FROM session_events WHERE session_id='x' AND event_type='continuation' ORDER BY seq DESC LIMIT 1", "idx_continuation_latest"),
        ("EXPLAIN QUERY PLAN SELECT seq FROM session_events WHERE session_id='x' AND event_type='tool_result' AND json_extract(payload,'$.file.path') IS NOT NULL AND json_extract(payload,'$.file.path')='a.rs' ORDER BY seq DESC LIMIT 1", "idx_latest_file_path"),
    ] {
        let mut stmt = conn.prepare(sql).unwrap();
        let plan = stmt.query_map([],|row| row.get::<_,String>(3)).unwrap().collect::<Result<Vec<_>,_>>().unwrap().join(" ");
        assert!(plan.contains(index),"{plan}"); assert!(!plan.contains("TEMP B-TREE"),"{plan}");
    }
}

#[tokio::test]
async fn rollover_is_atomic_idempotent_and_carries_memory_and_budget() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let mut session = Session::new("implement continuation", 100).unwrap();
    session.spend_cents = 25;
    session.spend_nano_usd = Some(250_000_000);
    db.insert_session(&session).unwrap();
    db.append_event(
        session.id,
        SessionEventType::Continuation,
        serde_json::json!({"task_id":"task","schema_version":1,"goal":"finish verification"}),
        None,
    )
    .await
    .unwrap();
    db.append_event(
        session.id,
        SessionEventType::TaskDecision,
        serde_json::json!({"task_id":"task","entries":[]}),
        None,
    )
    .await
    .unwrap();
    let (first, retry) = tokio::join!(
        db.rollover_session(session.clone(), "context_pressure".into(), 100, None),
        db.rollover_session(session.clone(), "context_pressure".into(), 100, None)
    );
    let first = first.unwrap();
    assert_eq!(first.id, retry.unwrap().id);
    assert_eq!(first.parent_session_id, Some(session.id));
    assert_eq!(first.inherited_spend_cents, 25);
    assert_eq!(
        db.get_session(session.id).unwrap().unwrap().status,
        engine_core::SessionStatus::Completed
    );
    assert!(db
        .read_latest_continuation(first.id, "task")
        .await
        .unwrap()
        .is_some());
    assert_eq!(
        db.read_task_decisions(first.id, "task")
            .await
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        db.read_events(session.id, None, None)
            .await
            .unwrap()
            .iter()
            .filter(|e| e.event_type == SessionEventType::SessionRollover)
            .count(),
        1
    );
}

#[tokio::test]
async fn failed_rollover_rolls_back_all_changes() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("rollback", 100).unwrap();
    db.insert_session(&session).unwrap();
    db.conn.lock().unwrap().execute_batch("CREATE TRIGGER reject_rollover BEFORE INSERT ON session_events WHEN NEW.event_type='session_rollover' BEGIN SELECT RAISE(ABORT,'injected failure'); END;").unwrap();
    assert!(db
        .rollover_session(session.clone(), "test".into(), 100, None)
        .await
        .is_err());
    assert_eq!(
        db.get_session(session.id).unwrap().unwrap().status,
        engine_core::SessionStatus::Active
    );
    let children: i64 = db
        .conn
        .lock()
        .unwrap()
        .query_row(
            "SELECT COUNT(*) FROM sessions WHERE parent_session_id=?1",
            [session.id.to_string()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(children, 0);
    assert!(db
        .read_events(session.id, None, None)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn tool_state_retrieval_cannot_cross_sessions() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let allowed = Uuid::new_v4();
    let other = Uuid::new_v4();
    let embedding = engine_core::embed_text("identical topic").unwrap();
    db.index_tool_result(IndexedToolState {
        session_id: other,
        seq: 1,
        embedding: embedding.clone(),
        message: serde_json::from_value(
            serde_json::json!({"role":"tool","content":"another session's evidence"}),
        )
        .unwrap(),
    })
    .await
    .unwrap();
    assert!(db
        .retrieve_tool_states(allowed, &embedding, 8, 0.0)
        .await
        .unwrap()
        .is_empty());
    assert_eq!(
        db.retrieve_tool_states(other, &embedding, 8, 0.0)
            .await
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn persistent_boundaries_redact_events_and_skip_sensitive_cache_rows() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("privacy boundary", 100).unwrap();
    db.insert_session(&session).unwrap();
    let event = db
        .append_event(
            session.id,
            SessionEventType::LlmCall,
            serde_json::json!({
                "request_messages":[{"role":"user","content":"api_key=abcdefghijk"}],
                "response_message":{
                    "role":"assistant",
                    "content":"done",
                    "provider_state":{"payload":{"signature":"exact-signed-state"}}
                },
                "reasoning_content":"raw reasoning"
            }),
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        event.payload["request_messages"][0]["content"],
        "api_key=[REDACTED]"
    );
    assert_eq!(
        event.payload["_reincore_privacy_redaction"]["representation"],
        "sanitized"
    );
    assert!(event.payload.get("reasoning_content").is_none());
    assert!(event.payload["response_message"]
        .get("provider_state")
        .is_none());

    db.update_session_summary(session.id, "password=abcdefghijk")
        .unwrap();
    assert_eq!(
        db.get_session(session.id).unwrap().unwrap().task_summary,
        "password=[REDACTED]"
    );

    let key = engine_core::CacheKey("privacy-cache-key".into());
    db.put_memoized(
        key.clone(),
        engine_core::cache::CacheEntry {
            tool_name: "lookup".into(),
            result_payload: serde_json::json!({"content":"token=abcdefghijk"}),
            content_hash: "unused".into(),
            created_at: Utc::now(),
        },
    )
    .await
    .unwrap();
    assert!(db.get_memoized(&key).await.unwrap().is_none());
    let state_key = engine_core::CacheKey("privacy-state-cache-key".into());
    db.put_memoized(
        state_key.clone(),
        engine_core::cache::CacheEntry {
            tool_name: "lookup".into(),
            result_payload: serde_json::json!({
                "content":"safe",
                "provider_state":{"signature":"opaque-signed-state"}
            }),
            content_hash: "unused".into(),
            created_at: Utc::now(),
        },
    )
    .await
    .unwrap();
    assert!(db.get_memoized(&state_key).await.unwrap().is_none());

    db.index_tool_result(IndexedToolState {
        session_id: session.id,
        seq: 9,
        message: serde_json::from_value(serde_json::json!({
            "role":"tool",
            "content":"access_token=abcdefghijk",
            "tool_call_id":"call-privacy",
            "name":"lookup"
        }))
        .unwrap(),
        embedding: vec![0.0; engine_core::EMBEDDING_DIMENSIONS],
    })
    .await
    .unwrap();
    let indexed: i64 = db
        .read_conn()
        .query_row(
            "SELECT COUNT(*) FROM tool_state_index WHERE session_id = ?1 AND seq = 9",
            [session.id.to_string()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(indexed, 0);
}

#[tokio::test]
async fn compact_projections_preserve_intervention_metrics() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("intervention projection", 100).unwrap();
    db.insert_session(&session).unwrap();
    db.append_event(
        session.id,
        SessionEventType::LlmCall,
        serde_json::json!({
            "request_group_id": "group-1",
            "task_id": "task-1",
            "task_name": "Review intervention metrics",
            "task_type": "coding",
            "request_messages": [{"role":"user", "content":"review metrics"}],
            "response_message": {"role":"assistant", "content":"done"},
            "input_tokens": 10,
            "output_tokens": 2,
        }),
        None,
    )
    .await
    .unwrap();
    let finding = engine_core::AgentHealthFinding {
        kind: engine_core::AgentHealthKind::PromptInjection,
        severity: engine_core::AgentHealthSeverity::Warning,
        message: "suspicious retrieved instruction".into(),
        tool_name: None,
    };
    let record = engine_core::InterventionRecord::from_finding(
        &finding,
        &engine_core::InterventionPolicy::default(),
        engine_core::InterventionAction::Warned,
    );
    db.append_event(
        session.id,
        SessionEventType::AgentHealth,
        serde_json::json!({
            "request_group_id": "group-1",
            "intervention": record,
        }),
        None,
    )
    .await
    .unwrap();

    let diagnostic = db.read_diagnostic_events(session.id).await.unwrap();
    let session_metrics = engine_core::InterventionMetrics::from_events(&diagnostic, true, 0, 0);
    assert_eq!(session_metrics.detections, 1);
    assert_eq!(session_metrics.warnings, 1);

    let timeline = db.read_timeline_events(session.id).await.unwrap();
    let tasks = engine_core::session_tasks(&timeline);
    assert_eq!(tasks[0].intervention_metrics.detections, 1);
    assert_eq!(tasks[0].request_groups[0].intervention_metrics.warnings, 1);
}

#[tokio::test]
async fn semantic_response_cache_enforces_scope_ttl_and_invalidation() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let now = Utc::now();
    let scope = engine_core::SemanticCacheScope {
        workspace_id: "local".to_string(),
        project_id: Uuid::new_v4().to_string(),
    };
    let fingerprint = engine_core::SemanticCacheFingerprint {
        provider: "test".to_string(),
        resolved_model: "model-a".to_string(),
        system_prefix_hash: "prefix-a".to_string(),
        tool_schema_hash: "tools-a".to_string(),
        generation_params_hash: "params-a".to_string(),
        application_version: "app-a".to_string(),
        config_version: "config-a".to_string(),
        source_data_version: "source-a".to_string(),
        immutable_workflow: None,
    };
    let mut embedding = vec![0.0; engine_core::EMBEDDING_DIMENSIONS];
    embedding[0] = 1.0;
    let policy = engine_core::SemanticCachePolicy {
        safe_to_reuse: true,
        ttl_seconds: 600,
        immutable_workflow: None,
    };
    db.put_semantic(engine_core::SemanticCacheEntry {
        embedding: embedding.clone(),
        prompt: "cached prompt".to_string(),
        response: serde_json::json!({"answer": "scoped"}),
        scope: scope.clone(),
        fingerprint: fingerprint.clone(),
        policy,
        created_at: now,
        expires_at: now + chrono::Duration::minutes(10),
    })
    .await
    .unwrap();

    let query = engine_core::SemanticCacheQuery {
        embedding,
        scope: scope.clone(),
        fingerprint_hash: fingerprint.digest().unwrap(),
        answer_threshold: 0.97,
        candidate_threshold: 0.80,
        candidate_limit: 3,
        now,
    };
    let hit = db.get_semantic(&query).await.unwrap();
    assert_eq!(
        hit.reusable_answer.unwrap().response,
        serde_json::json!({"answer": "scoped"})
    );

    let isolated = db
        .get_semantic(&engine_core::SemanticCacheQuery {
            scope: engine_core::SemanticCacheScope {
                workspace_id: "local".to_string(),
                project_id: Uuid::new_v4().to_string(),
            },
            ..query.clone()
        })
        .await
        .unwrap();
    assert!(isolated.reusable_answer.is_none());

    assert_eq!(db.invalidate_semantic(&scope, None).await.unwrap(), 1);
    assert!(db
        .get_semantic(&query)
        .await
        .unwrap()
        .reusable_answer
        .is_none());
}

#[test]
fn conversation_binding_insert_does_not_leave_losing_session_rows() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let first = Session::new("first", 0).unwrap();
    let losing = Session::new("losing duplicate", 0).unwrap();
    let binding_key =
        engine_core::conversation_binding_key("vscode", "path:/workspace/test", "chat-123");

    let winner = db
        .resolve_or_insert_conversation_session(
            &binding_key,
            "vscode",
            "path:/workspace/test",
            &engine_core::conversation_id_hash("chat-123"),
            &first,
            false,
            None,
            None,
        )
        .unwrap();
    let repeated = db
        .resolve_or_insert_conversation_session(
            &binding_key,
            "vscode",
            "path:/workspace/test",
            &engine_core::conversation_id_hash("chat-123"),
            &losing,
            false,
            None,
            None,
        )
        .unwrap();
    let session_count: i64 = db
        .conn
        .lock()
        .unwrap()
        .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
        .unwrap();

    assert_eq!(winner, first.id);
    assert_eq!(repeated, first.id);
    assert_eq!(session_count, 1);
    assert!(db.get_session(losing.id).unwrap().is_none());
}

#[tokio::test]
async fn generated_titles_preserve_request_prompts_and_recover_legacy_previews() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let prompt = "Order local engine models by descending rank";
    let wrapped = format!(
        "<context>{}</context><userRequest>{prompt}</userRequest>",
        "old context ".repeat(1500)
    );
    let request: engine_core::ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model":"test", "messages":[{"role":"user", "content":wrapped}]
    }))
    .unwrap();
    let session = Session::new(engine_core::derive_task_summary(&request), 0).unwrap();
    db.insert_session(&session).unwrap();
    let task = engine_core::resolve_task(Some(session.id), "group", &request, &[]);
    let event = db
        .append_event(
            session.id,
            SessionEventType::LlmCall,
            serde_json::json!({
                "task_id": task.task_id, "task_name": task.task_name, "task_type":"operations",
                "request_group_id":"group", "request_messages":request.messages,
                "response_message":{"role":"assistant", "content":"Done"}
            }),
            None,
        )
        .await
        .unwrap();
    let check = async {
        let timeline =
            engine_core::session_timeline(&db.read_timeline_events(session.id).await.unwrap());
        assert_eq!(timeline.request_groups[0].prompt.as_deref(), Some(prompt));
    };
    check.await;
    db.save_generated_title(
        session.clone(),
        task,
        "Sort engine models by descending rank".into(),
    )
    .await
    .unwrap();
    // Reproduce an already-persisted preview cut off inside the context tag.
    db.conn
        .lock()
        .unwrap()
        .execute(
            "UPDATE session_task_events SET user_prompt = ?1 WHERE session_id = ?2 AND seq = ?3",
            rusqlite::params![
                wrapped.chars().take(8192).collect::<String>(),
                session.id.to_string(),
                event.seq
            ],
        )
        .unwrap();
    let timeline =
        engine_core::session_timeline(&db.read_timeline_events(session.id).await.unwrap());
    assert_eq!(
        timeline.tasks[0].name,
        "Sort engine models by descending rank"
    );
    assert_eq!(timeline.request_groups[0].prompt.as_deref(), Some(prompt));
    assert_eq!(timeline.tasks[0].first_prompt, prompt);
    assert_eq!(
        db.read_recent_task_events(session.id, 10).await.unwrap()[0]
            .user_prompt
            .as_deref(),
        Some(prompt)
    );
}

#[tokio::test]
async fn generated_titles_survive_event_timing_and_stale_session_writes() {
    for title_first in [true, false] {
        let db = LocalDatabase::open_in_memory().unwrap();
        let request: engine_core::ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model":"test", "messages":[{"role":"user", "content":"Fix navigation scroll position"}]
        })).unwrap();
        let session = Session::new(engine_core::derive_task_summary(&request), 0).unwrap();
        db.insert_session(&session).unwrap();
        let task = engine_core::resolve_task(Some(session.id), "group", &request, &[]);
        let title = "Reset scroll position on navigation";
        let payload = serde_json::json!({
            "task_id": task.task_id, "task_name": task.task_name, "task_type":"coding",
            "request_group_id":"group", "request_messages": request.messages,
            "response_message":{"role":"assistant", "content":"Done"}
        });
        if title_first {
            db.save_generated_title(session.clone(), task.clone(), title.into())
                .await
                .unwrap();
        }
        db.append_event(session.id, SessionEventType::LlmCall, payload, None)
            .await
            .unwrap();
        if !title_first {
            db.save_generated_title(session.clone(), task.clone(), title.into())
                .await
                .unwrap();
        }
        // Simulate completion accounting writing its original session snapshot.
        db.insert_session(&session).unwrap();
        assert_eq!(
            db.get_session(session.id).unwrap().unwrap().task_summary,
            title
        );
        let events = db.read_timeline_events(session.id).await.unwrap();
        assert_eq!(engine_core::session_tasks(&events)[0].name, title);
        let recent = db.read_recent_task_events(session.id, 10).await.unwrap();
        assert_eq!(recent[0].task_name.as_deref(), Some(title));
        // Explicit rename wins over the asynchronous title.
        db.update_session_summary(session.id, "My manually chosen session title")
            .unwrap();
        db.save_generated_title(
            session.clone(),
            task,
            "A different generated session title".into(),
        )
        .await
        .unwrap();
        db.insert_session(&session).unwrap();
        assert_eq!(
            db.get_session(session.id).unwrap().unwrap().task_summary,
            "My manually chosen session title"
        );
    }
}

#[tokio::test]
async fn latest_seq_is_none_for_session_without_events() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("empty event log", 1_000).unwrap();
    db.insert_session(&session).unwrap();

    assert_eq!(db.latest_seq(session.id).await.unwrap(), None);

    db.append_event(
        session.id,
        SessionEventType::LlmCall,
        serde_json::json!({}),
        None,
    )
    .await
    .unwrap();
    assert_eq!(db.latest_seq(session.id).await.unwrap(), Some(1));
}

#[tokio::test]
async fn event_projections_support_bounded_task_and_tool_lookups() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("projection lookup", 1_000).unwrap();
    db.insert_session(&session).unwrap();

    db.append_event(
        session.id,
        SessionEventType::LlmCall,
        serde_json::json!({
            "request_group_id": "group-1",
            "task_id": "task-1",
            "task_name": "Inspect projections",
            "task_type": "coding",
            "request_messages": [{"role": "user", "content": "inspect the database"}],
            "response_message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"db.rs\"}"}
                }]
            },
            "input_tokens": 120,
            "output_tokens": 15
        }),
        Some(25),
    )
    .await
    .unwrap();
    db.append_event(
        session.id,
        SessionEventType::ToolCall,
        serde_json::json!({
            "tool_call": {
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\":\"db.rs\"}"}
            },
            "request_group_id": "group-1",
            "origin_llm_call_seq": 1
        }),
        None,
    )
    .await
    .unwrap();
    db.append_event(
        session.id,
        SessionEventType::ToolResult,
        serde_json::json!({
            "result": {
                "tool_call_id": "call-1",
                "role": "tool",
                "name": "read_file",
                "content": "contents"
            },
            "request_group_id": "group-1",
            "origin_llm_call_seq": 1
        }),
        None,
    )
    .await
    .unwrap();

    let task_events = db.read_recent_task_events(session.id, 10).await.unwrap();
    assert_eq!(task_events.len(), 1);
    assert_eq!(task_events[0].task_id.as_deref(), Some("task-1"));
    assert_eq!(
        task_events[0].user_prompt.as_deref(),
        Some("inspect the database")
    );
    assert!(task_events[0].response_has_tool_calls);

    let call_ids = vec!["call-1".to_string()];
    let links = db
        .read_tool_call_links(session.id, &call_ids)
        .await
        .unwrap();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].origin_llm_call_seq, Some(1));
    let tool_events = db
        .read_tool_events_by_call_ids(session.id, &call_ids)
        .await
        .unwrap();
    assert_eq!(tool_events.len(), 2);
    assert_eq!(tool_events[0].event_type, SessionEventType::ToolCall);
    assert_eq!(tool_events[1].event_type, SessionEventType::ToolResult);

    // Replaying the same provider response must not move the link back
    // from its canonical ToolCall event to the newer LLM audit event.
    db.append_event(
        session.id,
        SessionEventType::LlmCall,
        serde_json::json!({
            "request_group_id": "group-1",
            "request_messages": [{"role": "user", "content": "inspect the database"}],
            "response_message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"db.rs\"}"}
                }]
            }
        }),
        None,
    )
    .await
    .unwrap();
    let tool_events = db
        .read_tool_events_by_call_ids(session.id, &call_ids)
        .await
        .unwrap();
    assert_eq!(tool_events.len(), 2);
    assert_eq!(tool_events[0].event_type, SessionEventType::ToolCall);
}

#[tokio::test]
async fn endpoint_reads_are_bounded_and_preserve_timeline_results() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("compact endpoint reads", 1_000).unwrap();
    db.insert_session(&session).unwrap();

    db.append_event(
        session.id,
        SessionEventType::LlmCall,
        serde_json::json!({
            "request_group_id": "group-1",
            "task_id": "task-1",
            "task_name": "Compact endpoint reads",
            "task_type": "coding",
            "model": "test-model",
            "request_messages": [
                {"role": "system", "content": "large historical context"},
                {"role": "user", "content": "read compactly"}
            ],
            "response_message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"}
                }]
            },
            "input_tokens": 50,
            "output_tokens": 10
        }),
        Some(4),
    )
    .await
    .unwrap();
    db.append_event(
        session.id,
        SessionEventType::ToolCall,
        serde_json::json!({"request_group_id": "group-1"}),
        Some(1),
    )
    .await
    .unwrap();
    db.append_event(
        session.id,
        SessionEventType::ToolResult,
        serde_json::json!({
            "request_group_id": "group-1",
            "result": {"is_error": true}
        }),
        Some(2),
    )
    .await
    .unwrap();
    db.append_event(
        session.id,
        SessionEventType::LlmCall,
        serde_json::json!({
            "request_group_id": "group-1",
            "task_id": "task-1",
            "task_name": "Compact endpoint reads",
            "task_type": "coding",
            "model": "test-model",
            "request_messages": [{"role": "user", "content": "continue"}],
            "response_message": {"role": "assistant", "content": "done"},
            "input_tokens": 25,
            "output_tokens": 5
        }),
        Some(3),
    )
    .await
    .unwrap();

    let page = db
        .read_event_page(session.id, Some(1), Some(2), 10)
        .await
        .unwrap();
    assert_eq!(page.len(), 2);
    assert!(page[0].payload.get("request_messages").is_none());
    assert_eq!(
        page[0].payload["_reincore_omitted_fields"],
        serde_json::json!(["request_messages"])
    );

    let raw = db.read_events(session.id, None, None).await.unwrap();
    assert!(raw[0].payload.get("request_messages").is_some());
    let compact = db.read_timeline_events(session.id).await.unwrap();
    assert_eq!(
        engine_core::session_timeline(&compact),
        engine_core::session_timeline(&raw)
    );
}

#[tokio::test]
async fn file_database_read_pool_observes_migrated_projection_tables() {
    let path = std::env::temp_dir().join(format!("reincore-read-pool-{}.db", Uuid::new_v4()));
    {
        let db = LocalDatabase::open(&path).unwrap();
        let session = Session::new("read pool", 1_000).unwrap();
        db.insert_session(&session).unwrap();
        db.append_event(
            session.id,
            SessionEventType::LlmCall,
            serde_json::json!({
                "request_group_id": "group-pool",
                "task_id": "task-pool",
                "request_messages": [{"role": "user", "content": "read concurrently"}],
                "response_message": {"role": "assistant", "content": "done"}
            }),
            None,
        )
        .await
        .unwrap();

        let (events, projections) = tokio::join!(
            db.read_events(session.id, None, Some(10)),
            db.read_recent_task_events(session.id, 10)
        );
        assert_eq!(events.unwrap().len(), 1);
        assert_eq!(
            projections.unwrap()[0].task_id.as_deref(),
            Some("task-pool")
        );
    }
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{}-wal", path.display()));
    let _ = std::fs::remove_file(format!("{}-shm", path.display()));
}

#[tokio::test]
async fn checkpoint_redemption_is_persisted_and_unique() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let manager = engine_core::CheckpointManager::new("test-secret", 60);
    let mut session = Session::new("checkpoint test", 1_000).unwrap();
    db.append_event(
        session.id,
        SessionEventType::LlmCall,
        serde_json::json!({}),
        None,
    )
    .await
    .unwrap();
    let token = manager.create(&mut session, 1, "test").unwrap();

    manager.resume(&token, &mut session, &db).await.unwrap();
    let error = manager.resume(&token, &mut session, &db).await.unwrap_err();

    assert!(matches!(error, EngineError::CheckpointAlreadyRedeemed));
    let events = db.read_events(session.id, None, None).await.unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| event.event_type == SessionEventType::CheckpointRedeemed)
            .count(),
        1
    );
}

#[tokio::test]
async fn checkpoint_redemption_aggregate_counts_sessions_and_events() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let empty = db.aggregate_checkpoint_redemptions().unwrap();
    assert_eq!(empty.resumed_session_count, 0);
    assert_eq!(empty.redemption_count, 0);
    assert!(empty.latest_redeemed_at.is_none());

    let first = Session::new("first recovered task", 1_000).unwrap();
    let second = Session::new("second recovered task", 1_000).unwrap();
    let mut subagent = Session::new("recovered child", 1_000).unwrap();
    subagent.is_subagent = true;
    for session in [&first, &second, &subagent] {
        db.insert_session(session).unwrap();
    }

    for session_id in [first.id, first.id, second.id, subagent.id] {
        db.append_event(
            session_id,
            SessionEventType::CheckpointRedeemed,
            serde_json::json!({"token_fingerprint": Uuid::new_v4().to_string()}),
            None,
        )
        .await
        .unwrap();
    }
    db.append_event(
        first.id,
        SessionEventType::LlmCall,
        serde_json::json!({}),
        None,
    )
    .await
    .unwrap();

    let aggregate = db.aggregate_checkpoint_redemptions_async().await.unwrap();
    assert_eq!(aggregate.scope, "all");
    assert_eq!(aggregate.resumed_session_count, 2);
    assert_eq!(aggregate.redemption_count, 3);
    assert!(aggregate.latest_redeemed_at.is_some());
}

#[test]
fn tracking_migration_round_trips_session_lineage_and_project_usage() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let mut parent = Session::new_tracked(
        "tracked task",
        1_000,
        Some("demo"),
        Some("/workspace/demo"),
        Some("client"),
        Some("external-1"),
    )
    .unwrap();
    parent.spend_cents = 25;
    parent.context_tokens = 8_000;
    db.insert_session(&parent).unwrap();

    let loaded = db.get_session(parent.id).unwrap().unwrap();
    assert_eq!(loaded.project_name.as_deref(), Some("demo"));
    assert_eq!(loaded.external_session_id.as_deref(), Some("external-1"));

    let child = parent.rollover();
    db.insert_session(&parent).unwrap();
    db.insert_session(&child).unwrap();
    let active = db
        .get_active_session_by_external_id("client", None, "/workspace/demo", "external-1")
        .unwrap()
        .unwrap();
    assert_eq!(active.id, child.id);
    assert_eq!(active.parent_session_id, Some(parent.id));

    let usage = db.list_project_usage().unwrap();
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0].project_name, "demo");
    assert_eq!(usage[0].total_spend_cents, 25);
    assert_eq!(usage[0].session_count, 2);
}

#[tokio::test]
async fn local_projects_resolve_multiple_directories_and_attribute_sessions() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let root = std::env::temp_dir().join(format!("reincore-project-test-{}", Uuid::new_v4()));
    let primary = root.join("primary");
    let worktree = root.join("worktree");
    std::fs::create_dir_all(&primary).unwrap();
    std::fs::create_dir_all(&worktree).unwrap();

    let project = db
        .create_local_project("Demo", primary.to_str().unwrap())
        .unwrap();
    assert_eq!(project.name, "Demo");
    assert_eq!(project.directories.len(), 1);
    assert_eq!(
        db.get_local_project_by_path(primary.to_str().unwrap())
            .unwrap()
            .unwrap()
            .id,
        project.id
    );

    let session = Session::new_tracked(
        "worktree task",
        1_000,
        Some("worktree"),
        Some(worktree.to_string_lossy().into_owned()),
        Some("vscode"),
        Some("conversation"),
    )
    .unwrap();
    db.insert_session(&session).unwrap();

    let updated = db
        .assign_local_project_directory(project.id, worktree.to_str().unwrap())
        .unwrap();
    assert_eq!(updated.directories.len(), 2);
    let attributed = db.get_session(session.id).unwrap().unwrap();
    assert_eq!(attributed.project_id, Some(project.id));
    assert_eq!(attributed.project_name.as_deref(), Some("Demo"));
    let active = db
        .get_active_session_by_external_id(
            "vscode",
            Some(project.id),
            worktree.to_str().unwrap(),
            "conversation",
        )
        .unwrap()
        .unwrap();
    assert_eq!(active.id, session.id);

    let mut primary_session = Session::new_tracked(
        "primary task",
        1_000,
        Some("Demo"),
        Some(primary.to_string_lossy().into_owned()),
        Some("vscode"),
        Some("other-conversation"),
    )
    .unwrap();
    primary_session.project_id = Some(project.id);
    db.insert_session(&primary_session).unwrap();

    for session_id in [session.id, primary_session.id] {
        db.append_event(
            session_id,
            SessionEventType::LlmCall,
            serde_json::json!({}),
            None,
        )
        .await
        .unwrap();
    }

    let usage = db.list_project_usage().unwrap();
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0].project_id, Some(project.id));
    assert_eq!(usage[0].session_count, 2);

    let project_id = project.id.to_string();
    let filtered = db
        .list_sessions(&SessionFilter {
            project_id: Some(&project_id),
            ..SessionFilter::default()
        })
        .unwrap();
    assert_eq!(filtered.len(), 2);
    assert!(filtered.iter().any(|item| item.id == session.id));
    assert!(filtered.iter().any(|item| item.id == primary_session.id));

    let duplicate = db.create_local_project("Other", primary.to_str().unwrap());
    assert!(duplicate.is_err());

    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn sqlite_tool_state_store_indexes_and_retrieves_similar_results() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session_id = Uuid::new_v4();

    // Index a tool result about "authentication helpers".
    let embedding_a =
        engine_core::embed_text("search the codebase for authentication helpers\ngrep_search")
            .unwrap();
    db.index_tool_result(IndexedToolState {
        session_id,
        seq: 1,
        embedding: embedding_a,
        message: ChatMessage {
            provider_state: None,
            role: "tool".to_string(),
            content: Some(engine_core::MessageContent::text(
                "found auth.rs, oauth.rs".to_string(),
            )),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
            name: Some("grep_search".to_string()),
        },
    })
    .await
    .unwrap();

    // Query with similar text - should retrieve the indexed result.
    let query =
        engine_core::embed_text("search the codebase for authentication utilities again").unwrap();
    let results = db
        .retrieve_tool_states(session_id, &query, 4, 0.35)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].role, "tool");
    assert_eq!(results[0].name.as_deref(), Some("grep_search"));
    assert_eq!(results[0].content_str(), Some("found auth.rs, oauth.rs"));
}

#[tokio::test]
async fn sqlite_tool_state_store_upsert_is_idempotent() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session_id = Uuid::new_v4();
    let embedding =
        engine_core::embed_text("find the database connection helper\ngrep_search").unwrap();

    // Index the same (session_id, seq) twice - should not duplicate.
    for _ in 0..2 {
        db.index_tool_result(IndexedToolState {
            session_id,
            seq: 1,
            embedding: embedding.clone(),
            message: ChatMessage {
                provider_state: None,
                role: "tool".to_string(),
                content: Some(engine_core::MessageContent::text("found db.rs".to_string())),
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                name: Some("grep_search".to_string()),
            },
        })
        .await
        .unwrap();
    }

    let query = engine_core::embed_text("find the database connection helper again").unwrap();
    let results = db
        .retrieve_tool_states(session_id, &query, 8, 0.0)
        .await
        .unwrap();
    // Even with min_similarity=0, re-indexing the same event must not
    // produce duplicate rows.
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn sqlite_tool_state_store_rejects_wrong_dimension() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let result = db
        .index_tool_result(IndexedToolState {
            session_id: Uuid::new_v4(),
            seq: 1,
            embedding: vec![0.0; 128], // wrong dimension
            message: ChatMessage {
                provider_state: None,
                role: "tool".to_string(),
                content: Some(engine_core::MessageContent::text("x".to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn sqlite_tool_state_store_end_to_end_with_memory_manager() {
    // End-to-end: the MemoryManager assembles context with the SQLite
    // ToolStateStore, indexing ToolResult events via the event log and
    // retrieving them for a similar current turn.
    use engine_core::{MemoryManager, ToolStateRetrievalConfig};
    use std::sync::Arc;

    let db = Arc::new(LocalDatabase::open_in_memory().unwrap());
    let session_id = Uuid::new_v4();

    // Earlier turn: user asks to search for auth helpers, tool returns.
    db.append_event(
        session_id,
        SessionEventType::LlmCall,
        serde_json::json!({
            "request_messages": [{"role": "user", "content": "search the codebase for authentication helpers"}],
            "response_message": {"role": "assistant", "content": "let me search"}
        }),
        None,
    )
    .await
    .unwrap();
    db.append_event(
        session_id,
        SessionEventType::ToolResult,
        serde_json::json!({
            "result": {
                "content": "found auth.rs, oauth.rs",
                "tool_call_id": "call_1",
                "name": "grep_search"
            }
        }),
        None,
    )
    .await
    .unwrap();

    // Later turn: similar question. The RAG tier should surface the
    // earlier grep_search result.
    db.append_event(
        session_id,
        SessionEventType::LlmCall,
        serde_json::json!({
            "request_messages": [{"role": "user", "content": "search the codebase for authentication utilities again"}],
            "response_message": {"role": "assistant", "content": "ok"}
        }),
        None,
    )
    .await
    .unwrap();

    let manager = MemoryManager::default_config().with_tool_state_store(
        db.clone() as Arc<dyn engine_core::ToolStateStore>,
        ToolStateRetrievalConfig::default(),
    );
    let context = manager
        .assemble(db.as_ref(), session_id, None)
        .await
        .unwrap();

    assert!(
        !context.retrieved_tool_states.is_empty(),
        "expected RAG-retrieved tool states from the SQLite store"
    );
    let retrieved = &context.retrieved_tool_states[0];
    assert_eq!(retrieved.name.as_deref(), Some("grep_search"));
    assert_eq!(retrieved.content_str(), Some("found auth.rs, oauth.rs"));
}

#[tokio::test]
async fn sqlite_read_latest_summary_returns_most_recent_summary() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session_id = Uuid::new_v4();

    // No summary yet.
    assert!(db.read_latest_summary(session_id).await.unwrap().is_none());

    // Two summaries; the later one should win.
    db.append_event(
        session_id,
        SessionEventType::Summary,
        serde_json::json!({
            "summary_text": "first summary",
            "summarized_seq_start": 0,
            "summarized_seq_end": 5,
        }),
        None,
    )
    .await
    .unwrap();
    db.append_event(
        session_id,
        SessionEventType::LlmCall,
        serde_json::json!({ "response_message": { "role": "user", "content": "hi" } }),
        None,
    )
    .await
    .unwrap();
    db.append_event(
        session_id,
        SessionEventType::Summary,
        serde_json::json!({
            "summary_text": "second summary",
            "summarized_seq_start": 0,
            "summarized_seq_end": 7,
        }),
        None,
    )
    .await
    .unwrap();

    let summary = db
        .read_latest_summary(session_id)
        .await
        .unwrap()
        .expect("expected a summary");
    assert_eq!(summary.summary_text, "second summary");
    assert_eq!(summary.summarized_seq_end, 7);
}

#[tokio::test]
async fn sqlite_read_latest_files_deduplicates_by_path_to_latest_version() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session_id = Uuid::new_v4();

    // v1 of src/main.rs, then v2 (should win), plus an unrelated file.
    db.append_event(
        session_id,
        SessionEventType::ToolResult,
        serde_json::json!({
            "result": { "content": "v1", "tool_call_id": "c1", "name": "read_file" },
            "file": { "path": "src/main.rs", "version": 1 }
        }),
        None,
    )
    .await
    .unwrap();
    db.append_event(
        session_id,
        SessionEventType::ToolResult,
        serde_json::json!({
            "result": { "content": "v2", "tool_call_id": "c2", "name": "read_file" },
            "file": { "path": "src/main.rs", "version": 2 }
        }),
        None,
    )
    .await
    .unwrap();
    db.append_event(
        session_id,
        SessionEventType::ToolResult,
        serde_json::json!({
            "result": { "content": "lib", "tool_call_id": "c3", "name": "read_file" },
            "file": { "path": "src/lib.rs", "version": 1 }
        }),
        None,
    )
    .await
    .unwrap();
    // A tool_result with no file attachment should be ignored.
    db.append_event(
        session_id,
        SessionEventType::ToolResult,
        serde_json::json!({
            "result": { "content": "no file", "tool_call_id": "c4", "name": "run" }
        }),
        None,
    )
    .await
    .unwrap();

    let files = db.read_latest_files(session_id).await.unwrap();
    assert_eq!(files.len(), 2);
    assert_eq!(files[0].path, "src/lib.rs");
    assert_eq!(files[0].version, 1);
    assert_eq!(files[1].path, "src/main.rs");
    assert_eq!(files[1].version, 2);
}

#[tokio::test]
async fn committed_events_are_published_to_live_subscribers() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session_id = Uuid::new_v4();
    let mut subscriber = db.event_hub().subscribe();

    let committed = db
        .append_event(
            session_id,
            SessionEventType::ToolCall,
            serde_json::json!({"tool_call": {"id": "call-live"}}),
            None,
        )
        .await
        .unwrap();

    let delivered = subscriber.recv().await.unwrap();
    assert_eq!(delivered.session_id, session_id);
    assert_eq!(delivered.seq, committed.seq);
    assert_eq!(delivered.event_type, SessionEventType::ToolCall);
}

#[tokio::test]
async fn session_keyset_pages_handle_ties_and_sparse_events() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let timestamp = Utc::now();
    let mut expected = Vec::new();
    let mut conn = db.conn.lock().unwrap();
    let tx = conn.transaction().unwrap();
    for i in 0..10_000 {
        let mut session = Session::new_tracked(
            format!("page {i}"),
            100,
            None::<String>,
            None::<String>,
            None::<String>,
            None::<String>,
        )
        .unwrap();
        session.updated_at = timestamp;
        LocalDatabase::insert_session_on(&tx, &session).unwrap();
        if i % 7 != 0 {
            LocalDatabase::append_event_on(
                &tx,
                session.id,
                SessionEventType::LlmCall,
                serde_json::json!({}),
                None,
            )
            .unwrap();
            expected.push(session.id);
        }
    }
    tx.commit().unwrap();
    drop(conn);
    expected.sort_by(|a, b| b.cmp(a));
    let mut found = Vec::new();
    let mut before = None;
    loop {
        let page = db
            .list_sessions(&SessionFilter {
                limit: Some(17),
                before: before.as_deref(),
                ..Default::default()
            })
            .unwrap();
        assert!(page.len() <= 17);
        if page.is_empty() {
            break;
        }
        let last = page.last().unwrap();
        before = Some(format!("{}|{}", last.updated_at.to_rfc3339(), last.id));
        found.extend(page.iter().map(|s| s.id));
    }
    assert_eq!(found, expected);
    assert!(matches!(
        db.list_sessions(&SessionFilter {
            before: Some("broken"),
            ..Default::default()
        }),
        Err(EngineError::InvalidInput(_))
    ));
    let conn = db.read_conn();
    let mut statement = conn.prepare("EXPLAIN QUERY PLAN SELECT id FROM sessions WHERE is_subagent = 0 AND (updated_at, id) < (?1, ?2) ORDER BY updated_at DESC, id DESC LIMIT 17").unwrap();
    let plan = statement
        .query_map(
            rusqlite::params![timestamp.to_rfc3339(), Uuid::nil().to_string()],
            |row| row.get::<_, String>(3),
        )
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .join(" ");
    assert!(plan.contains("idx_sessions_page"), "{plan}");
    assert!(!plan.contains("TEMP B-TREE"), "{plan}");
}

#[tokio::test]
async fn diagnostic_pages_preserve_exact_aggregates() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let id = Uuid::new_v4();
    for i in 0..1100 {
        db.append_event(
            id,
            SessionEventType::LlmCall,
            serde_json::json!({
                "input_tokens": 100, "cached_prompt_tokens": i % 100,
                "predicted_cost_low_cents": 1, "predicted_cost_high_cents": 3,
                "messages": "ignored payload", "cost": {"amount_nano_usd": i * 100000}
            }),
            None,
        )
        .await
        .unwrap();
    }
    let page = db.read_diagnostic_event_page(id, 510, 512).await.unwrap();
    assert_eq!(page.len(), 512);
    assert_eq!(page[0].seq, 510);
    assert!(page[0].payload.get("messages").is_none());
    let expected = engine_core::diagnostics::CallDiagnostics::from_events(
        &db.read_diagnostic_events(id).await.unwrap(),
    );
    let mut accumulator = engine_core::diagnostics::DiagnosticAccumulator::default();
    accumulator.append_session(&db, id).await.unwrap();
    assert_eq!(accumulator.finish(), expected);
}

#[tokio::test]
async fn session_filters_normalize_uuid_and_keep_fractional_end_of_day() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let project = Uuid::new_v4();
    let mut session = Session::new("date boundary", 100).unwrap();
    session.project_id = Some(project);
    session.updated_at = "2026-09-06T20:59:59.999999999Z".parse().unwrap();
    db.insert_session(&session).unwrap();
    db.append_event(
        session.id,
        SessionEventType::LlmCall,
        serde_json::json!({}),
        None,
    )
    .await
    .unwrap();
    let project_filter = project.to_string().to_uppercase();
    let rows = db
        .list_sessions(&SessionFilter {
            project_id: Some(&project_filter),
            date_from: Some("2026-09-06T00:00:00+03:00"),
            date_to: Some("2026-09-06T23:59:59.999999999+03:00"),
            limit: Some(10),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].id, session.id);
    assert!(db
        .list_sessions(&SessionFilter {
            date_to: Some("2026-09-06T23:59:59+03:00"),
            ..Default::default()
        })
        .unwrap()
        .is_empty());
}

#[test]
fn project_scoped_listing_includes_in_flight_session_without_events() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let project = Uuid::new_v4();
    // Membership is fixed at creation, but the first durable event only lands
    // after the first provider response settles: the running loop below has
    // no events and no tokens yet.
    let mut linked = Session::new("running loop", 100).unwrap();
    linked.project_id = Some(project);
    linked.project_path = Some("/workspace/demo".to_string());
    db.insert_session(&linked).unwrap();
    let other = Session::new("idle draft", 100).unwrap();
    db.insert_session(&other).unwrap();

    // Unscoped listings still hide event-less sessions; the project page must
    // already show its in-flight session with live 0-token totals.
    assert!(db.list_sessions(&SessionFilter::default()).unwrap().is_empty());
    let rows = db
        .list_sessions(&SessionFilter {
            project_id: Some(&project.to_string()),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].id, linked.id);
}

#[test]
fn external_id_lookup_survives_mid_loop_project_registration() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let project = Uuid::new_v4();
    // The loop started before its working directory was registered as a
    // project, so the row carries the path but no project id yet.
    let mut session = Session::new("loop", 100).unwrap();
    session.client_id = Some("local".to_string());
    session.external_session_id = Some("ext-1".to_string());
    session.project_path = Some("/workspace/demo".to_string());
    db.insert_session(&session).unwrap();

    // The loop's next request resolves with the project id set and the same
    // path: it must rejoin the existing session, not fork a second one.
    let found = db
        .get_active_session_by_external_id("local", Some(project), "/workspace/demo", "ext-1")
        .unwrap()
        .expect("loop session must stay reachable after registration");
    assert_eq!(found.id, session.id);
    let latest = db
        .get_latest_session_by_external_id("local", Some(project), "/workspace/demo", "ext-1")
        .unwrap()
        .expect("latest lookup must agree");
    assert_eq!(latest.id, session.id);
}

#[tokio::test]
async fn task_decision_sqlite_survives_reopen_and_filters_session_and_task() {
    let path = std::env::temp_dir().join(format!("reincore-decisions-{}.sqlite", Uuid::new_v4()));
    let session = Session::new("Task decisions", 100).unwrap();
    {
        let db = LocalDatabase::open(&path).unwrap();
        db.insert_session(&session).unwrap();
        let mut request:engine_core::ChatCompletionRequest=serde_json::from_value(serde_json::json!({
            "model":"test","messages":[{"role":"user","content":"Use PostgreSQL"}],"reincore_task_id":"task-one"
        })).unwrap();
        let task = engine_core::resolve_task(Some(session.id), "first", &request, &[]);
        engine_core::task_decisions::prepare(
            &db,
            Some(session.id),
            &task,
            "first",
            "project",
            &mut request,
            true,
        )
        .await
        .unwrap();
        db.append_event(
            session.id,
            SessionEventType::TaskDecision,
            serde_json::json!({"task_id":"other-task","entries":[]}),
            None,
        )
        .await
        .unwrap();
        db.append_event(
            session.id,
            SessionEventType::LlmCall,
            serde_json::json!({"task_id":"task-one","content":"unrelated audit output"}),
            None,
        )
        .await
        .unwrap();
    }
    {
        let db = LocalDatabase::open(&path).unwrap();
        let record = engine_core::task_decisions::load(&db, session.id, "task-one")
            .await
            .unwrap();
        assert_eq!(
            record
                .lookup(
                    "Which database should we use?",
                    "project::task",
                    "second",
                    Utc::now()
                )
                .unwrap()
                .value,
            "Use PostgreSQL"
        );
        assert_eq!(
            db.read_task_decisions(session.id, "task-one")
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(db
            .read_task_decisions(Uuid::new_v4(), "task-one")
            .await
            .unwrap()
            .is_empty());
        assert!(db
            .read_task_decisions(session.id, "missing-task")
            .await
            .unwrap()
            .is_empty());
    }
    std::fs::remove_file(path).unwrap();
}

#[tokio::test]
async fn continuation_sqlite_survives_reopen_and_recovers_task_without_llm_events() {
    let path =
        std::env::temp_dir().join(format!("reincore-continuation-{}.sqlite", Uuid::new_v4()));
    let session = Session::new("Continue verified work", 100).unwrap();
    let task_id;
    {
        let db = LocalDatabase::open(&path).unwrap();
        db.insert_session(&session).unwrap();
        let mut request: engine_core::ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model":"test", "messages":[
                {"role":"user","content":"Implement a durable session continuation"},
                {"role":"assistant","tool_calls":[{"id":"todo","type":"function","function":{
                    "name":"manage_todo_list","arguments":"{\"todos\":[{\"title\":\"Run recovery checks\",\"status\":\"in-progress\"}]}"}}]},
                {"role":"tool","tool_call_id":"todo","content":"ok"}
            ]
        })).unwrap();
        let task = engine_core::resolve_task(Some(session.id), "first", &request, &[]);
        task_id = task.task_id.clone();
        engine_core::continuation::prepare(&db, Some(session.id), &task, "first", &mut request)
            .await
            .unwrap();
    }
    {
        let db = LocalDatabase::open(&path).unwrap();
        let event = db
            .read_latest_continuation(session.id, "")
            .await
            .unwrap()
            .unwrap();
        let seed = engine_core::continuation::task_resolution_seed(event).unwrap();
        let mut request: engine_core::ChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model":"test","messages":[{"role":"user","content":"Please continue"}]
            }))
            .unwrap();
        let task = engine_core::resolve_task(Some(session.id), "second", &request, &[seed]);
        assert_eq!(task.task_id, task_id);
        assert!(!task.is_new);
        engine_core::continuation::prepare(&db, Some(session.id), &task, "second", &mut request)
            .await
            .unwrap();
        let state: engine_core::continuation::Continuation = serde_json::from_value(
            db.read_latest_continuation(session.id, &task_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        assert_eq!(state.next_step.as_deref(), Some("Run recovery checks"));
        assert!(db
            .read_latest_continuation(session.id, "another-task")
            .await
            .unwrap()
            .is_none());
        engine_core::continuation::attach(&mut request);
        assert!(request
            .messages
            .iter()
            .any(|m| m.name.as_deref() == Some(engine_core::continuation::DATA_NAME)));
    }
    std::fs::remove_file(path).unwrap();
}

#[tokio::test]
async fn portable_file_history_sqlite_survives_device_move_and_preserves_session_scope() {
    use engine_core::file_history::{FileHistoryScope, FileRoot};
    let path = std::env::temp_dir().join(format!(
        "reincore-portable-history-{}.sqlite",
        Uuid::new_v4()
    ));
    let mut session = Session::new("Move task between devices", 100).unwrap();
    let project = Uuid::new_v4();
    session.project_id = Some(project);
    let mapping = |id: &str, path: &str| FileHistoryScope {
        project_id: project,
        roots: vec![FileRoot {
            id: id.into(),
            path: path.into(),
        }],
    };
    {
        let db = LocalDatabase::open(&path).unwrap();
        db.insert_session(&session).unwrap();
        // A pre-mapping record, including its old JSON representation, upgrades safely.
        db.append_event(session.id, SessionEventType::ToolCall, serde_json::json!({"tool_call": {
            "id":"windows-write", "type":"function", "function": {"name":"write_file",
            "arguments":serde_json::json!({"path":"C:/work/repo/src/main.rs","content":"first version"}).to_string()}
        }}), None).await.unwrap();
        db.append_event(session.id, SessionEventType::ToolResult, serde_json::json!({"result": {
            "tool_call_id":"windows-write","name":"write_file","content":"success"},"cache_hit":false
        }), None).await.unwrap();
        db.bind_file_history_scope(session.id, &mapping("project", "C:/work/repo"))
            .await
            .unwrap();
    }
    {
        let db = LocalDatabase::open(&path).unwrap();
        assert_eq!(
            db.read_file_history_scope(session.id)
                .await
                .unwrap()
                .unwrap()
                .roots[0]
                .path,
            "C:/work/repo"
        );
        let linux = mapping("project", "/home/chris/repo");
        let other = mapping("other", "/different/repo");
        let (one, two) = tokio::join!(
            db.bind_file_history_scope(session.id, &linux),
            db.bind_file_history_scope(session.id, &other)
        );
        one.unwrap();
        two.unwrap();
        let history = db
            .read_file_history(session.id, &["/home/chris/repo/src/main.rs".into()])
            .await
            .unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(
            history[0].observed_path.as_deref(),
            Some("C:/work/repo/src/main.rs")
        );
        assert_eq!(history[0].file_id.as_ref().unwrap().project_id, project);
        assert_eq!(
            history[0].status,
            engine_core::continuation::ToolStatus::Succeeded
        );
        assert!(db
            .read_file_history(session.id, &["/different/repo/src/main.rs".into()])
            .await
            .unwrap()
            .is_empty());
        assert!(db
            .read_file_history(session.id, &["src/main.rs".into()])
            .await
            .unwrap()
            .is_empty());
        assert!(db
            .read_file_history(Uuid::new_v4(), &["/home/chris/repo/src/main.rs".into()])
            .await
            .unwrap()
            .is_empty());
        // A rejected transaction must not leave its first, valid mapping behind.
        let bad = FileHistoryScope {
            project_id: project,
            roots: vec![
                FileRoot {
                    id: "new".into(),
                    path: "/new/repo".into(),
                },
                FileRoot {
                    id: "wrong".into(),
                    path: "/home/chris/repo".into(),
                },
            ],
        };
        assert!(db.bind_file_history_scope(session.id, &bad).await.is_err());
        assert_eq!(
            db.read_file_history_scope(session.id)
                .await
                .unwrap()
                .unwrap()
                .roots
                .len(),
            3
        );
        for index in 0..6 {
            let id = format!("linux-write-{index}");
            db.append_event(session.id,SessionEventType::ToolCall,serde_json::json!({"tool_call": {
                "id":id,"type":"function","function":{"name":"write_file","arguments":serde_json::json!({"path":"/home/chris/repo/src/main.rs","content":format!("version {index}")}).to_string()}
            }}),None).await.unwrap();
            db.append_event(session.id,SessionEventType::ToolResult,serde_json::json!({"result":{"tool_call_id":id,"name":"write_file","content":"success"}}),None).await.unwrap();
        }
        let newest = db
            .read_file_history(session.id, &["C:/work/repo/src/main.rs".into()])
            .await
            .unwrap();
        assert_eq!(newest.len(), 4);
        assert_eq!(newest[0].tool_call_id, "linux-write-5");
        let aliases = db
            .read_file_history(
                session.id,
                &[
                    "C:/work/repo/src/main.rs".into(),
                    "/home/chris/repo/src/main.rs".into(),
                ],
            )
            .await
            .unwrap();
        assert_eq!(aliases.len(), 4);
        assert!(newest
            .iter()
            .all(|record| record.path == "C:/work/repo/src/main.rs"
                && record.observed_path.as_deref() == Some("/home/chris/repo/src/main.rs")));
        let key = history[0].file_id.as_ref().unwrap().key();
        let conn = db.read_conn();
        let mut stmt = conn.prepare("EXPLAIN QUERY PLAN SELECT payload FROM session_file_history WHERE session_id=?1 AND file_key=?2 ORDER BY seq DESC LIMIT 4").unwrap();
        let plan = stmt
            .query_map(rusqlite::params![session.id.to_string(), key], |row| {
                row.get::<_, String>(3)
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .join(" ");
        assert!(plan.contains("idx_session_file_history_identity"), "{plan}");
        assert!(!plan.contains("TEMP B-TREE"), "{plan}");
    }
    std::fs::remove_file(path).unwrap();
}
