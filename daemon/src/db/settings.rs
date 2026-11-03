//! Settings persistence for the local engine.

use async_trait::async_trait;
use engine_core::{
    budget::SettingsStore,
    circuit::LoopSettings,
    compression::CompressionSettings,
    error::{EngineError, EngineResult},
};
use rusqlite::{params, OptionalExtension};

use super::LocalDatabase;

#[async_trait]
impl SettingsStore for LocalDatabase {
    async fn get_compression_settings(&self) -> EngineResult<CompressionSettings> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let mut stmt = conn
                .prepare("SELECT value FROM engine_settings WHERE key = ?1")
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let row = stmt
                .query_row(["compression"], |row| {
                    let value: String = row.get(0)?;
                    Ok(value)
                })
                .optional()
                .map_err(|e| EngineError::Storage(e.to_string()))?;

            match row {
                Some(value) => serde_json::from_str(&value).map_err(EngineError::Serialization),
                None => Ok(CompressionSettings::disabled()),
            }
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn set_compression_settings(&self, settings: CompressionSettings) -> EngineResult<()> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let value = serde_json::to_string(&settings)?;
            conn.execute(
                "INSERT INTO engine_settings (key, value)
                 VALUES (?1, ?2)
                 ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value",
                params!["compression", value],
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn get_loop_settings(&self) -> EngineResult<LoopSettings> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let mut stmt = conn
                .prepare("SELECT value FROM engine_settings WHERE key = ?1")
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let row = stmt
                .query_row(["loop"], |row| {
                    let value: String = row.get(0)?;
                    Ok(value)
                })
                .optional()
                .map_err(|e| EngineError::Storage(e.to_string()))?;

            match row {
                Some(value) => serde_json::from_str(&value).map_err(EngineError::Serialization),
                // Absent means "no overrides", which resolves to the defaults.
                None => Ok(LoopSettings::default()),
            }
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn set_loop_settings(&self, settings: LoopSettings) -> EngineResult<()> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let value = serde_json::to_string(&settings)?;
            conn.execute(
                "INSERT INTO engine_settings (key, value)
                 VALUES (?1, ?2)
                 ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value",
                params!["loop", value],
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }
}
