//! Schema persistence for the local engine.

use crate::migrations::all_migrations;
use anyhow::Result;
use engine_core::migration::{validate_migrations, Dialect, MigrationRunner};

use super::LocalDatabase;

impl LocalDatabase {
    pub(super) fn migrate(&self) -> Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let migrations = all_migrations();
        validate_migrations(&migrations)
            .map_err(|e| anyhow::anyhow!("invalid migrations: {}", e))?;

        let runner = MigrationRunner::new(Dialect::SQLite, &migrations);

        let tx = conn.transaction()?;
        tx.execute_batch(&runner.migration_table_sql())?;

        let current_version: i64 = tx
            .query_row(
                "SELECT COALESCE(MAX(version), -1) FROM schema_migrations",
                [],
                |row| row.get(0),
            )
            .unwrap_or(-1);

        for (version, description, stmts) in runner.pending_sql(current_version) {
            for stmt in stmts {
                tx.execute_batch(&stmt)?;
            }
            tx.execute_batch(&runner.record_migration_sql(version, &description))?;
        }

        tx.commit()?;
        drop(conn);
        self.backfill_local_projects()?;
        self.seed_provider_models()?;
        Ok(())
    }

    /// Run migrations and return a human-readable summary.
    pub fn run_migrations(&self) -> Result<String> {
        let mut conn = self.conn.lock().unwrap();
        let migrations = all_migrations();
        validate_migrations(&migrations)
            .map_err(|e| anyhow::anyhow!("invalid migrations: {}", e))?;

        let runner = MigrationRunner::new(Dialect::SQLite, &migrations);

        let tx = conn.transaction()?;
        tx.execute_batch(&runner.migration_table_sql())?;

        let current_version: i64 = tx
            .query_row(
                "SELECT COALESCE(MAX(version), -1) FROM schema_migrations",
                [],
                |row| row.get(0),
            )
            .unwrap_or(-1);

        let mut applied = 0;
        for (version, description, stmts) in runner.pending_sql(current_version) {
            for stmt in stmts {
                tx.execute_batch(&stmt)?;
            }
            tx.execute_batch(&runner.record_migration_sql(version, &description))?;
            applied += 1;
        }

        tx.commit()?;
        drop(conn);
        self.backfill_local_projects()?;
        self.seed_provider_models()?;
        Ok(format!(
            "SQLite migrations complete. Database is at version {} ({} migration(s) applied).",
            migrations
                .last()
                .map(|m| m.version)
                .unwrap_or(current_version),
            applied
        ))
    }
}
