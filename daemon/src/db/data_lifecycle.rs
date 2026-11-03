//! Maintenance is explicit and offline for deletion; backups use SQLite's WAL-aware API.
use super::LocalDatabase;
use anyhow::{Context, Result};
use rusqlite::{
    backup::{Backup, StepResult},
    Connection, OpenFlags, TransactionBehavior,
};
use std::{
    path::Path,
    time::{Duration, Instant},
};

fn existing(path: &Path, writable: bool) -> Result<Connection> {
    LocalDatabase::register_vec_auto_extension();
    anyhow::ensure!(path.is_file(), "database must already exist");
    let flags = if writable {
        OpenFlags::SQLITE_OPEN_READ_WRITE
    } else {
        OpenFlags::SQLITE_OPEN_READ_ONLY
    };
    let conn = Connection::open_with_flags(path, flags)?;
    conn.busy_timeout(Duration::from_secs(2))?;
    conn.execute_batch("PRAGMA foreign_keys=ON;")?;
    Ok(conn)
}
pub fn verify(path: &Path) -> Result<()> {
    verify_connection(&existing(path, false)?)
}
fn verify_connection(conn: &Connection) -> Result<()> {
    let integrity: String = conn.query_row("PRAGMA quick_check", [], |r| r.get(0))?;
    anyhow::ensure!(integrity == "ok", "SQLite integrity check failed");
    anyhow::ensure!(
        !conn.prepare("PRAGMA foreign_key_check")?.exists([])?,
        "foreign key check failed"
    );
    let version: i64 = conn.query_row("SELECT MAX(version) FROM schema_migrations", [], |r| {
        r.get(0)
    })?;
    let supported = crate::migrations::all_migrations().last().unwrap().version;
    anyhow::ensure!(
        version >= 0 && version <= supported,
        "unsupported database schema version"
    );
    conn.prepare("SELECT id, status, updated_at FROM sessions LIMIT 0")?;
    conn.prepare("SELECT session_id, seq, payload FROM session_events LIMIT 0")?;
    Ok(())
}
/// Snapshot to a new file, including transactions still present in the source WAL.
pub fn backup(source: &Path, destination: &Path) -> Result<()> {
    for suffix in ["-wal", "-shm", "-journal"] {
        let mut sidecar = destination.as_os_str().to_os_string();
        sidecar.push(suffix);
        anyhow::ensure!(
            !Path::new(&sidecar).try_exists()?,
            "backup destination has existing SQLite sidecars"
        );
    }
    let source = existing(source, false)?;
    verify_connection(&source)?;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options
        .open(destination)
        .context("backup destination must be a new file")?;
    drop(file);
    let result = (|| {
        let mut target = Connection::open(destination)?;
        {
            let backup = Backup::new(&source, &mut target)?;
            let deadline = Instant::now() + Duration::from_secs(60);
            loop {
                anyhow::ensure!(
                    Instant::now() < deadline,
                    "backup exceeded 60-second deadline"
                );
                match backup.step(128)? {
                    StepResult::Done => break,
                    StepResult::More => {}
                    StepResult::Busy | StepResult::Locked => {
                        std::thread::sleep(Duration::from_millis(10))
                    }
                    _ => anyhow::bail!("unexpected backup status"),
                }
            }
        }
        verify_connection(&target)?;
        target.close().map_err(|(_, e)| e)?;
        std::fs::OpenOptions::new()
            .write(true)
            .open(destination)?
            .sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(destination);
    }
    result
}
/// Restore to a fresh path; never overwrite a running or existing database.
pub fn restore(source: &Path, destination: &Path) -> Result<()> {
    backup(source, destination)
}

pub fn retain(
    path: &Path,
    before: chrono::DateTime<chrono::Utc>,
    limit: usize,
    apply: bool,
) -> Result<Vec<String>> {
    anyhow::ensure!((1..=100).contains(&limit), "limit must be 1 through 100");
    anyhow::ensure!(
        before < chrono::Utc::now(),
        "retention cutoff must be in the past"
    );
    let mut conn = existing(path, true)?;
    let version: i64 = conn.query_row("SELECT MAX(version) FROM schema_migrations", [], |r| {
        r.get(0)
    })?;
    anyhow::ensure!(
        version == crate::migrations::all_migrations().last().unwrap().version,
        "migrate database before retention"
    );
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let sql = format!("SELECT s.id FROM sessions s WHERE {} AND s.updated_at < ?1 ORDER BY s.updated_at, s.id LIMIT ?2", engine_core::data_lifecycle::eligibility(false));
    let ids = tx
        .prepare(&sql)?
        .query_map(rusqlite::params![before.to_rfc3339(), limit as i64], |r| {
            r.get::<_, String>(0)
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for id in &ids {
        anyhow::ensure!(
            uuid::Uuid::parse_str(id)?.to_string() == *id,
            "noncanonical stored session ID"
        );
    }
    if apply {
        for id in &ids {
            for table in engine_core::data_lifecycle::SESSION_TABLES {
                tx.execute(&format!("DELETE FROM {table} WHERE session_id = ?1"), [id])?;
            }
            tx.execute("DELETE FROM generated_titles WHERE title_key = 'session:' || ?1 OR title_key LIKE 'task:' || ?1 || ':%'", [id])?;
            tx.execute("DELETE FROM sessions WHERE id = ?1", [id])?;
        }
        tx.commit()?;
    } else {
        tx.rollback()?;
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{
        accounting::{AccountingStore, CompletionAccounting},
        EventStore, Session, SessionEventType, SessionStatus,
    };
    #[tokio::test]
    async fn wal_backup_restore_and_retention_preserve_recovery_and_accounting() {
        let dir = std::env::temp_dir().join(format!("reincore-recovery-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        let source = dir.join("source.db");
        let snapshot = dir.join("snapshot.db");
        let restored = dir.join("restored.db");
        let db = LocalDatabase::open(&source).unwrap();
        let orphan = dir.join("unsafe.db-wal");
        std::fs::write(&orphan, b"preserve orphan").unwrap();
        assert!(backup(&source, &dir.join("unsafe.db")).is_err());
        assert!(!dir.join("unsafe.db").exists());
        assert_eq!(std::fs::read(&orphan).unwrap(), b"preserve orphan");
        let mut old = Session::new("old transcript", 100).unwrap();
        old.status = SessionStatus::Completed;
        old.updated_at = chrono::Utc::now() - chrono::Duration::days(100);
        db.insert_session(&old).unwrap();
        db.append_event(
            old.id,
            SessionEventType::LlmCall,
            serde_json::json!({"response":"recover this text"}),
            Some(1),
        )
        .await
        .unwrap();
        db.conn.lock().unwrap().execute("INSERT INTO tool_state_index(embedding,session_id,seq,role,content,created_at) VALUES (?1,?2,1,'assistant','indexed transcript',?3)", rusqlite::params![serde_json::to_string(&vec![1.0f32;384]).unwrap(),old.id.to_string(),chrono::Utc::now().to_rfc3339()]).unwrap();
        db.conn.lock().unwrap().execute("INSERT INTO checkpoints(token,session_id,last_event_seq,expires_at,created_at) VALUES ('expired',?1,1,?2,?2)", rusqlite::params![old.id.to_string(),old.updated_at.to_rfc3339()]).unwrap();
        let active = Session::new("active", 100).unwrap();
        db.insert_session(&active).unwrap();
        db.conn.lock().unwrap().execute("INSERT INTO ledger_events(session_id,amount_cents,reason,created_at) VALUES (?1,1,'usage',?2)", rusqlite::params![old.id.to_string(), chrono::Utc::now().to_rfc3339()]).unwrap();
        db.stage_accounting(&CompletionAccounting {
            id: uuid::Uuid::new_v4(), session_id: active.id, cost_cents: 3, cost: None, purpose: None,
            recorded_at: chrono::Utc::now(), payload: serde_json::json!({"input_tokens": 12, "output_tokens": 3,
              "predicted_cost_high_cents": 3, "request_messages": [], "response_message": {"role":"assistant","content":"ok"},
              "model":"test", "model_tier":"lite", "request_group_id":"recovery-test"})
        }).await.unwrap();
        let cutoff = chrono::Utc::now() - chrono::Duration::days(90);
        assert_eq!(
            retain(&source, cutoff, 10, false).unwrap(),
            vec![old.id.to_string()]
        );
        // Keep source connections open: the snapshot must include uncheckpointed WAL data.
        backup(&source, &snapshot).unwrap();
        assert!(backup(&source, &snapshot).is_err());
        drop(db);
        assert_eq!(retain(&source, cutoff, 10, true).unwrap().len(), 1);
        assert!(retain(&source, cutoff, 10, true).unwrap().is_empty());
        let db = LocalDatabase::open(&source).unwrap();
        assert!(db.get_session(old.id).unwrap().is_none());
        assert_eq!(
            db.conn
                .lock()
                .unwrap()
                .query_row("SELECT COUNT(*) FROM tool_state_index", [], |r| r
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert!(db.get_session(active.id).unwrap().is_some());
        assert!(db.read_events(old.id, None, None).await.unwrap().is_empty());
        assert_eq!(
            db.conn
                .lock()
                .unwrap()
                .query_row(
                    "SELECT COUNT(*) FROM ledger_events WHERE session_id = ?1",
                    [old.id.to_string()],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
            1
        );
        restore(&snapshot, &restored).unwrap();
        assert!(restore(&snapshot, &source).is_err());
        let restored_db = LocalDatabase::open(&restored).unwrap();
        let events = restored_db.read_events(old.id, None, None).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].payload["response"], "recover this text");
        assert_eq!(
            restored_db.get_session(old.id).unwrap().unwrap().status,
            SessionStatus::Completed
        );
        assert_eq!(
            restored_db
                .get_session(active.id)
                .unwrap()
                .unwrap()
                .spend_cents,
            3
        );
        drop(restored_db);
        let reopened = LocalDatabase::open(&restored).unwrap();
        assert_eq!(
            reopened
                .get_session(active.id)
                .unwrap()
                .unwrap()
                .spend_cents,
            3
        );
        assert_eq!(
            reopened
                .conn
                .lock()
                .unwrap()
                .query_row("SELECT COUNT(*) FROM ledger_events", [], |r| r
                    .get::<_, i64>(0))
                .unwrap(),
            2
        );
        drop(reopened);
        drop(db);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[tokio::test]
    async fn retention_protects_work_and_rolls_back_partial_deletions() {
        let dir = std::env::temp_dir().join(format!("reincore-retention-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("db.sqlite");
        let db = LocalDatabase::open(&path).unwrap();
        let mut sessions = Vec::new();
        for i in 0..6 {
            let mut session = Session::new(format!("protected {i}"), 100).unwrap();
            session.status = SessionStatus::Completed;
            session.updated_at = chrono::Utc::now() - chrono::Duration::days(100);
            if i == 0 {
                session.status = SessionStatus::Active;
            }
            if i == 4 {
                session.parent_session_id = Some(sessions[3]);
            }
            db.insert_session(&session).unwrap();
            db.append_event(
                session.id,
                SessionEventType::LlmCall,
                serde_json::json!({}),
                None,
            )
            .await
            .unwrap();
            sessions.push(session.id);
        }
        let conn = db.conn.lock().unwrap();
        let plan_sql = format!(
            "EXPLAIN QUERY PLAN SELECT s.id FROM sessions s WHERE {}",
            engine_core::data_lifecycle::eligibility(false)
        );
        let plan = conn
            .prepare(&plan_sql)
            .unwrap()
            .query_map([], |row| row.get::<_, String>(3))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
            .join("\n");
        assert!(
            !plan.contains("SCAN child"),
            "lineage must use indexes: {plan}"
        );
        conn.execute("INSERT INTO completion_accounting(id,session_id,record,settled) VALUES ('pending',?1,'{}',0)", [sessions[1].to_string()]).unwrap();
        conn.execute(
            "INSERT INTO batch_jobs(id,session_id,revision,active,record) VALUES (?1,?2,0,1,'{}')",
            rusqlite::params![uuid::Uuid::new_v4().to_string(), sessions[2].to_string()],
        )
        .unwrap();
        conn.execute_batch("CREATE TRIGGER fail_retention BEFORE DELETE ON sessions BEGIN SELECT RAISE(ABORT, 'test failure'); END;").unwrap();
        drop(conn);
        let cutoff = chrono::Utc::now() - chrono::Duration::days(90);
        assert_eq!(
            retain(&path, cutoff, 1, false).unwrap(),
            vec![sessions[5].to_string()]
        );
        assert!(retain(&path, cutoff, 1, true).is_err());
        assert_eq!(
            db.read_events(sessions[5], None, None).await.unwrap().len(),
            1
        );
        assert!(db.get_session(sessions[5]).unwrap().is_some());
        db.conn
            .lock()
            .unwrap()
            .execute_batch("DROP TRIGGER fail_retention")
            .unwrap();
        assert_eq!(
            retain(&path, cutoff, 1, true).unwrap(),
            vec![sessions[5].to_string()]
        );
        drop(db);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
