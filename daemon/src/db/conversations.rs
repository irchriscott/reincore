//! Conversations persistence for the local engine.

use engine_core::{
    error::{EngineError, EngineResult},
    session::Session,
};
use rusqlite::{params, OptionalExtension, TransactionBehavior};
use uuid::Uuid;

use super::LocalDatabase;

impl LocalDatabase {
    /// Atomically resolve or create the session owned by a client conversation.
    /// The immediate transaction prevents parallel first turns from allocating
    /// multiple visible sessions for the same durable binding.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve_or_insert_conversation_session(
        &self,
        binding_key: &str,
        client_id: &str,
        scope_key: &str,
        conversation_id_hash: &str,
        proposed: &Session,
        is_subagent: bool,
        parent_binding_key: Option<&str>,
        spawn_tool_call_id: Option<&str>,
    ) -> EngineResult<Uuid> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| EngineError::Storage(error.to_string()))?;
        let existing: Option<String> = tx
            .query_row(
                "SELECT session_id FROM session_conversation_bindings WHERE binding_key = ?1",
                params![binding_key],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| EngineError::Storage(error.to_string()))?;
        if let Some(existing) = existing {
            tx.commit()
                .map_err(|error| EngineError::Storage(error.to_string()))?;
            return Uuid::parse_str(&existing)
                .map_err(|error| EngineError::Storage(error.to_string()));
        }

        Self::insert_session_on(&tx, proposed)?;
        let now = chrono::Utc::now().to_rfc3339();
        tx.execute(
            "INSERT INTO session_conversation_bindings (
                binding_key, client_id, scope_key, conversation_id_hash,
                session_id, is_subagent, parent_binding_key,
                spawn_tool_call_id, created_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?9)",
            params![
                binding_key,
                client_id,
                scope_key,
                conversation_id_hash,
                proposed.id.to_string(),
                is_subagent as i64,
                parent_binding_key,
                spawn_tool_call_id,
                now,
            ],
        )
        .map_err(|error| EngineError::Storage(error.to_string()))?;
        tx.commit()
            .map_err(|error| EngineError::Storage(error.to_string()))?;
        Ok(proposed.id)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn resolve_or_insert_conversation_session_async(
        &self,
        binding_key: String,
        client_id: String,
        scope_key: String,
        conversation_id_hash: String,
        proposed: Session,
        is_subagent: bool,
        parent_binding_key: Option<String>,
        spawn_tool_call_id: Option<String>,
    ) -> EngineResult<Uuid> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            db.resolve_or_insert_conversation_session(
                &binding_key,
                &client_id,
                &scope_key,
                &conversation_id_hash,
                &proposed,
                is_subagent,
                parent_binding_key.as_deref(),
                spawn_tool_call_id.as_deref(),
            )
        })
        .await
        .map_err(|error| EngineError::Storage(format!("sqlite task join error: {error}")))?
    }

    pub fn get_conversation_binding_session_id(
        &self,
        binding_key: &str,
    ) -> EngineResult<Option<Uuid>> {
        let conn = self.read_conn();
        let value: Option<String> = conn
            .query_row(
                "SELECT session_id FROM session_conversation_bindings WHERE binding_key = ?1",
                params![binding_key],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| EngineError::Storage(error.to_string()))?;
        value
            .map(|value| {
                Uuid::parse_str(&value).map_err(|error| EngineError::Storage(error.to_string()))
            })
            .transpose()
    }

    pub async fn get_conversation_binding_session_id_async(
        &self,
        binding_key: String,
    ) -> EngineResult<Option<Uuid>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.get_conversation_binding_session_id(&binding_key))
            .await
            .map_err(|error| EngineError::Storage(format!("sqlite task join error: {error}")))?
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn bind_conversation_to_session_async(
        &self,
        binding_key: String,
        client_id: String,
        scope_key: String,
        conversation_id_hash: String,
        session_id: Uuid,
        is_subagent: bool,
        parent_binding_key: Option<String>,
        spawn_tool_call_id: Option<String>,
    ) -> EngineResult<Uuid> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = db.conn.lock().unwrap();
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(|error| EngineError::Storage(error.to_string()))?;
            if let Some(existing) = tx
                .query_row(
                    "SELECT session_id FROM session_conversation_bindings WHERE binding_key = ?1",
                    params![binding_key],
                    |row| row.get::<_, String>(0),
                )
                .optional()
                .map_err(|error| EngineError::Storage(error.to_string()))?
            {
                tx.commit()
                    .map_err(|error| EngineError::Storage(error.to_string()))?;
                return Uuid::parse_str(&existing)
                    .map_err(|error| EngineError::Storage(error.to_string()));
            }
            let now = chrono::Utc::now().to_rfc3339();
            tx.execute(
                "INSERT INTO session_conversation_bindings (
                    binding_key, client_id, scope_key, conversation_id_hash,
                    session_id, is_subagent, parent_binding_key,
                    spawn_tool_call_id, created_at, updated_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?9)",
                params![
                    binding_key,
                    client_id,
                    scope_key,
                    conversation_id_hash,
                    session_id.to_string(),
                    is_subagent as i64,
                    parent_binding_key,
                    spawn_tool_call_id,
                    now,
                ],
            )
            .map_err(|error| EngineError::Storage(error.to_string()))?;
            tx.commit()
                .map_err(|error| EngineError::Storage(error.to_string()))?;
            Ok(session_id)
        })
        .await
        .map_err(|error| EngineError::Storage(format!("sqlite task join error: {error}")))?
    }
}
