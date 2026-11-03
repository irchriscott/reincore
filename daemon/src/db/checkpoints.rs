//! Checkpoints persistence for the local engine.

use chrono::Utc;
use engine_core::{
    checkpoint::ResumableToken,
    error::{EngineError, EngineResult},
};
use rusqlite::{params, OptionalExtension};
use uuid::Uuid;

use super::LocalDatabase;

impl LocalDatabase {
    pub fn get_latest_checkpoint_token(&self, session_id: Uuid) -> EngineResult<Option<String>> {
        let conn = self.conn.lock().unwrap();
        let token: Option<String> = conn
            .query_row(
                "SELECT token FROM checkpoints
                 WHERE session_id = ?1 AND expires_at > ?2
                 ORDER BY created_at DESC LIMIT 1",
                params![session_id.to_string(), Utc::now().to_rfc3339()],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| EngineError::Storage(e.to_string()))?;
        Ok(token)
    }

    pub async fn get_latest_checkpoint_token_async(
        &self,
        session_id: Uuid,
    ) -> EngineResult<Option<String>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.get_latest_checkpoint_token(session_id))
            .await
            .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    pub fn insert_checkpoint(&self, token: &ResumableToken) -> EngineResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO checkpoints (token, session_id, last_event_seq, expires_at, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(token) DO UPDATE SET
               session_id = EXCLUDED.session_id,
               last_event_seq = EXCLUDED.last_event_seq,
               expires_at = EXCLUDED.expires_at",
            params![
                &token.token,
                token.session_id.to_string(),
                token.last_event_seq,
                token.expires_at.to_rfc3339(),
                Utc::now().to_rfc3339(),
            ],
        )
        .map_err(|e| EngineError::Storage(e.to_string()))?;
        Ok(())
    }

    pub async fn insert_checkpoint_async(&self, token: ResumableToken) -> EngineResult<()> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.insert_checkpoint(&token))
            .await
            .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }
}
