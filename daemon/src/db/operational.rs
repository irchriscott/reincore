use super::LocalDatabase;
use engine_core::operational::{database_observation, Metric, Timer};

impl LocalDatabase {
    pub async fn ready(&self) -> bool {
        static PROBE: std::sync::OnceLock<std::sync::Arc<tokio::sync::Semaphore>> =
            std::sync::OnceLock::new();

        let Ok(permit) = PROBE
            .get_or_init(|| std::sync::Arc::new(tokio::sync::Semaphore::new(1)))
            .clone()
            .try_acquire_owned()
        else {
            return false;
        };

        let db = self.clone();
        matches!(
            tokio::time::timeout(
                std::time::Duration::from_millis(750),
                tokio::task::spawn_blocking(move || {
                    let _permit = permit;
                    let Ok(conn) = db.conn.try_lock() else {
                        return false;
                    };
                    // Execute the read: preparing alone can succeed against a cached schema
                    // while another connection holds an exclusive database lock.
                    let ready = conn
                        .query_row(
                            "SELECT COUNT(*) FROM (SELECT id FROM sessions LIMIT 1)",
                            [],
                            |row| row.get::<_, i64>(0),
                        )
                        .is_ok();
                    ready
                })
            )
            .await,
            Ok(Ok(true))
        )
    }

    pub async fn observe_dependencies(&self) {
        let db = self.clone();
        let timer = Timer::new(Metric::DatabaseProbe);
        // Never queue behind normal SQLite work just to collect a metric.
        let result = tokio::task::spawn_blocking(move || {
            let conn = db.conn.try_lock().ok()?;
            Some(
                conn.query_row("SELECT 1", [], |row| row.get::<_, i64>(0))
                    .is_ok(),
            )
        })
        .await
        .ok()
        .flatten();
        drop(timer);
        database_observation(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn busy_sqlite_is_unknown_and_recovers_without_queued_probes() {
        let db = LocalDatabase::open_in_memory().unwrap();
        let lock = db.conn.lock().unwrap();
        db.observe_dependencies().await;
        assert!(!engine_core::operational::render().contains("reincore_database_up "));
        drop(lock);
        db.observe_dependencies().await;
        assert!(engine_core::operational::render().contains("reincore_database_up 1\n"));
    }
}

#[cfg(test)]
mod readiness_tests {
    use super::*;
    #[tokio::test]
    async fn readiness_rejects_contention_and_missing_schema_then_recovers() {
        let db = LocalDatabase::open_in_memory().unwrap();
        assert!(db.ready().await);
        let lock = db.conn.lock().unwrap();
        assert!(!db.ready().await);
        drop(lock);
        assert!(db.ready().await);
        db.conn
            .lock()
            .unwrap()
            .execute_batch("ALTER TABLE sessions RENAME TO unavailable_sessions")
            .unwrap();
        assert!(!db.ready().await);
        db.conn
            .lock()
            .unwrap()
            .execute_batch("ALTER TABLE unavailable_sessions RENAME TO sessions")
            .unwrap();
        assert!(db.ready().await);
    }
}
