//! Tool state persistence for the local engine.

use async_trait::async_trait;
use chrono::Utc;
use engine_core::{
    error::{EngineError, EngineResult},
    models::ChatMessage,
    store::{IndexedToolState, ToolStateStore},
};
use rusqlite::{params, TransactionBehavior};

use super::LocalDatabase;

#[async_trait]
impl ToolStateStore for LocalDatabase {
    async fn index_tool_result(&self, entry: IndexedToolState) -> EngineResult<()> {
        if entry.embedding.len() != 384 {
            return Err(EngineError::Internal(format!(
                "tool_state_index expects 384-dimensional embeddings, got {}",
                entry.embedding.len()
            )));
        }
        let privacy = engine_core::privacy::current_policy();
        if privacy.value_contains_secret(&serde_json::to_value(&entry.message)?) {
            let db = self.clone();
            let session_id = entry.session_id.to_string();
            let seq = entry.seq;
            return tokio::task::spawn_blocking(move || {
                let conn = db.conn.lock().unwrap();
                conn.execute(
                    "DELETE FROM tool_state_index WHERE session_id = ?1 AND seq = ?2",
                    params![session_id, seq],
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
                Ok(())
            })
            .await
            .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?;
        }
        let embedding_json = format!(
            "[{}]",
            entry
                .embedding
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let session_id = entry.session_id.to_string();
        let seq = entry.seq;
        let role = entry.message.role.clone();
        let content = entry.message.content_text();
        let tool_call_id = entry.message.tool_call_id.clone();
        let name = entry.message.name.clone();
        let created_at = Utc::now().to_rfc3339();

        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = db.conn.lock().unwrap();
            // vec0 virtual tables don't support ON CONFLICT upserts, so
            // delete any existing row for this (session_id, seq) pair
            // before inserting. Wrapped in a transaction so a crash between
            // the delete and insert can't lose the prior entry without
            // installing the new one.
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            tx.execute(
                "DELETE FROM tool_state_index WHERE session_id = ?1 AND seq = ?2",
                params![session_id, seq],
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;
            tx.execute(
                "INSERT INTO tool_state_index (
                    embedding, session_id, seq, role, content,
                    tool_call_id, name, created_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    embedding_json,
                    session_id,
                    seq,
                    role,
                    content,
                    tool_call_id,
                    name,
                    created_at,
                ],
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;
            tx.commit()
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn retrieve_tool_states(
        &self,
        session_id: uuid::Uuid,
        query_embedding: &[f32],
        k: usize,
        min_similarity: f32,
    ) -> EngineResult<Vec<ChatMessage>> {
        if query_embedding.len() != 384 {
            return Err(EngineError::Internal(format!(
                "tool_state_index expects 384-dimensional embeddings, got {}",
                query_embedding.len()
            )));
        }
        // cosine distance = 1 - cosine_similarity; min_similarity is a
        // similarity floor, so the matching distance ceiling is
        // 1 - min_similarity.
        let max_distance = (1.0 - min_similarity).max(0.0) as f64;
        let query_json = format!(
            "[{}]",
            query_embedding
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        // Cap k internally to bound context growth, matching the trait doc.
        let effective_k = k.min(8) as i64;

        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.read_conn();
            let mut stmt = conn
                .prepare(
                    "SELECT role, content, tool_call_id, name, vec_distance_cosine(embedding, ?1) AS distance
                     FROM tool_state_index
                     WHERE session_id = ?4 AND vec_distance_cosine(embedding, ?1) <= ?3
                     ORDER BY distance ASC LIMIT ?2",
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map(params![query_json, effective_k, max_distance, session_id.to_string()], |row| {
                    let role: String = row.get(0)?;
                    let content: Option<String> = row.get(1)?;
                    let content = content.map(|s| engine_core::MessageContent::text(s));
                    let tool_call_id: Option<String> = row.get(2)?;
                    let name: Option<String> = row.get(3)?;
                    Ok(ChatMessage {
                        provider_state: None,
                        role,
                        content,
                        tool_calls: None,
                        tool_call_id,
                        name,
                    })
                })
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let mut messages = Vec::new();
            for row in rows {
                messages.push(row.map_err(|e| EngineError::Storage(e.to_string()))?);
            }
            Ok(messages)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }
}
