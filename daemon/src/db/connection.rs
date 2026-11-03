//! Connection persistence for the local engine.

use anyhow::Result;
use rusqlite::Connection;
use sqlite_vec::sqlite3_vec_init;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::LocalDatabase;

impl LocalDatabase {
    /// Subscribe to events after their storage transaction has committed.
    pub fn event_hub(&self) -> engine_core::SessionEventHub {
        self.event_hub.clone()
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::register_vec_auto_extension();
        let path = path.as_ref().to_path_buf();
        let conn = Connection::open(&path)?;
        Self::verify_vec_extension(&conn)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA foreign_keys = ON;
             PRAGMA busy_timeout = 5000;",
        )?;
        let mut readers = Vec::with_capacity(4);
        for _ in 0..4 {
            let reader = Connection::open(&path)?;
            Self::verify_vec_extension(&reader)?;
            reader.execute_batch(
                "PRAGMA foreign_keys = ON;
                 PRAGMA busy_timeout = 5000;
                 PRAGMA query_only = ON;",
            )?;
            readers.push(Mutex::new(reader));
        }
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
            readers: Arc::new(readers),
            next_reader: Arc::new(AtomicUsize::new(0)),
            event_hub: engine_core::SessionEventHub::default(),
        };
        db.migrate()?;
        db.recover_accounting_sync()?;
        Ok(db)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::register_vec_auto_extension();
        let conn = Connection::open_in_memory()?;
        Self::verify_vec_extension(&conn)?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
            readers: Arc::new(Vec::new()),
            next_reader: Arc::new(AtomicUsize::new(0)),
            event_hub: engine_core::SessionEventHub::default(),
        };
        db.migrate()?;
        db.recover_accounting_sync()?;
        Ok(db)
    }

    pub(super) fn register_vec_auto_extension() {
        // sqlite-vec exposes a C entrypoint with SQLite's extension signature;
        // rusqlite requires it through sqlite3_auto_extension's callback type.
        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite3_vec_init as *const (),
            )));
        }
    }

    pub(super) fn read_conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        if self.readers.is_empty() {
            return self.conn.lock().unwrap();
        }
        let index = self.next_reader.fetch_add(1, Ordering::Relaxed) % self.readers.len();
        self.readers[index].lock().unwrap()
    }

    pub(super) fn verify_vec_extension(conn: &Connection) -> Result<()> {
        let _version: String = conn.query_row("SELECT vec_version()", [], |row| row.get(0))?;
        Ok(())
    }
}
