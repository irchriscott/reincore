//! Cache persistence for the local engine.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use engine_core::{
    cache::{
        CacheEntry, CacheKey, MemoizationCache, SemanticCacheCandidate, SemanticCacheEntry,
        SemanticCacheHit, SemanticCacheLookup, SemanticCacheProvenance, SemanticCacheQuery,
        SemanticCacheScope, ToolDefinitionForCache,
    },
    error::{EngineError, EngineResult},
    models::{FileContextAttachment, ToolCall, ToolResult},
    store::CacheStore,
};
use rusqlite::{params, OptionalExtension};
use sha2::Digest;
use uuid::Uuid;

use super::LocalDatabase;

#[async_trait]
impl MemoizationCache for LocalDatabase {
    // This legacy contract carries no session/repository/credential scope.
    // Keep compatibility without permitting unscoped cache reads or writes.
    async fn lookup(
        &self,
        _tool: &ToolDefinitionForCache,
        _call: &ToolCall,
    ) -> EngineResult<Option<ToolResult>> {
        Ok(None)
    }

    async fn store(
        &self,
        _tool: &ToolDefinitionForCache,
        _call: &ToolCall,
        _result: &ToolResult,
    ) -> EngineResult<()> {
        Ok(())
    }

    async fn invalidate_for_file_update(&self, file: &FileContextAttachment) -> EngineResult<u64> {
        let db = self.clone();
        let file_path = file.path.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let affected = conn
                .execute(
                    "DELETE FROM tool_cache WHERE EXISTS (
                        SELECT 1 FROM json_tree(tool_cache.result_payload)
                        WHERE type = 'text' AND instr(atom, ?1) > 0
                    )",
                    params![file_path],
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(affected as u64)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }
}

#[async_trait]
impl CacheStore for LocalDatabase {
    async fn get_memoized(&self, key: &CacheKey) -> EngineResult<Option<CacheEntry>> {
        let db = self.clone();
        let cache_key = key.0.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT tool_name, result_payload, content_hash, created_at
                     FROM tool_cache WHERE cache_key = ?1",
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let row = stmt
                .query_row([cache_key], |row| {
                    let payload_str: String = row.get(1)?;
                    let created_str: String = row.get(3)?;
                    Ok(CacheEntry {
                        tool_name: row.get(0)?,
                        result_payload: serde_json::from_str(&payload_str).unwrap_or_default(),
                        content_hash: row.get(2)?,
                        created_at: created_str
                            .parse()
                            .unwrap_or(chrono::DateTime::<Utc>::UNIX_EPOCH),
                    })
                })
                .optional()
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(row)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn put_memoized(&self, key: CacheKey, entry: CacheEntry) -> EngineResult<()> {
        // A cache is durable storage, not a temporary execution buffer. Skip
        // sensitive results instead of changing integrity-bound cache envelopes.
        if engine_core::privacy::current_policy()
            .should_skip_persistent_value(&entry.result_payload)
        {
            return Ok(());
        }
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO tool_cache (
                    cache_key, tool_name, result_payload, content_hash, created_at
                ) VALUES (?1, ?2, ?3, ?4, ?5)
                ON CONFLICT(cache_key) DO UPDATE SET
                   result_payload=excluded.result_payload,
                   content_hash=excluded.content_hash,
                   created_at=excluded.created_at",
                params![
                    key.0,
                    entry.tool_name,
                    entry.result_payload.to_string(),
                    entry.content_hash,
                    entry.created_at.to_rfc3339(),
                ],
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn get_semantic(&self, query: &SemanticCacheQuery) -> EngineResult<SemanticCacheLookup> {
        if query.embedding.len() != 384 {
            return Err(EngineError::Internal(format!(
                "semantic cache expects 384-dimensional embeddings, got {}",
                query.embedding.len()
            )));
        }
        let max_distance = (1.0 - query.candidate_threshold).max(0.0) as f64;
        let query_json = format!(
            "[{}]",
            query
                .embedding
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let scope = query.scope.clone();
        let fingerprint_hash = query.fingerprint_hash.clone();
        let now_timestamp = query.now.timestamp();
        let answer_threshold = query.answer_threshold;
        let candidate_limit = query.candidate_limit.max(1) as i64;

        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT entry_id, prompt, response, fingerprint_json,
                            created_at, expires_at, distance
                     FROM semantic_response_cache
                     WHERE embedding MATCH ?1 AND k = ?2 AND distance <= ?3
                       AND workspace_id = ?4 AND project_id = ?5
                       AND fingerprint_hash = ?6 AND safe_to_reuse = 1
                       AND expires_at > ?7
                     ORDER BY distance ASC",
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map(
                    params![
                        query_json,
                        candidate_limit,
                        max_distance,
                        scope.workspace_id,
                        scope.project_id,
                        fingerprint_hash,
                        now_timestamp
                    ],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                            row.get::<_, String>(4)?,
                            row.get::<_, i64>(5)?,
                            row.get::<_, f64>(6)?,
                        ))
                    },
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let mut lookup = SemanticCacheLookup::default();
            for row in rows {
                let (entry_id, prompt, response, fingerprint, created_at, expires_at, distance) =
                    row.map_err(|e| EngineError::Storage(e.to_string()))?;
                let fingerprint: engine_core::SemanticCacheFingerprint =
                    serde_json::from_str(&fingerprint)?;
                let response = serde_json::from_str(&response)?;
                let similarity = (1.0 - distance) as f32;
                let provenance = SemanticCacheProvenance {
                    entry_id,
                    prompt_hash: hex::encode(sha2::Sha256::digest(prompt.as_bytes())),
                    scope: scope.clone(),
                    fingerprint_hash: fingerprint_hash.clone(),
                    similarity,
                    created_at: DateTime::parse_from_rfc3339(&created_at)
                        .map_err(|e| EngineError::Storage(e.to_string()))?
                        .with_timezone(&Utc),
                    expires_at: DateTime::from_timestamp(expires_at, 0).ok_or_else(|| {
                        EngineError::Storage("invalid semantic cache expiry timestamp".to_string())
                    })?,
                    provider: fingerprint.provider,
                    resolved_model: fingerprint.resolved_model,
                    source_data_version: fingerprint.source_data_version,
                    immutable_workflow: fingerprint.immutable_workflow,
                };
                if similarity >= answer_threshold && lookup.reusable_answer.is_none() {
                    lookup.reusable_answer = Some(SemanticCacheHit {
                        response,
                        provenance,
                    });
                } else if similarity < answer_threshold {
                    lookup.candidates.push(SemanticCacheCandidate {
                        response,
                        provenance,
                    });
                }
            }
            Ok(lookup)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn put_semantic(&self, entry: SemanticCacheEntry) -> EngineResult<()> {
        if entry.embedding.len() != 384 {
            return Err(EngineError::Internal(format!(
                "semantic cache expects 384-dimensional embeddings, got {}",
                entry.embedding.len()
            )));
        }
        if !entry.policy.safe_to_reuse || entry.expires_at <= entry.created_at {
            return Ok(());
        }
        let privacy = engine_core::privacy::current_policy();
        if privacy.contains_secret(&entry.prompt)
            || privacy.should_skip_persistent_value(&entry.response)
        {
            return Ok(());
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
        let response_str = serde_json::to_string(&entry.response)?;
        let fingerprint_hash = entry.fingerprint.digest()?;
        let fingerprint_json = serde_json::to_string(&entry.fingerprint)?;

        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            conn.execute(
                "DELETE FROM semantic_response_cache WHERE expires_at <= ?1",
                params![Utc::now().timestamp()],
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;
            conn.execute(
                "INSERT INTO semantic_response_cache(
                    entry_id, workspace_id, project_id, fingerprint_hash,
                    safe_to_reuse, expires_at, embedding, fingerprint_json,
                    prompt, response, created_at
                 ) VALUES (?1, ?2, ?3, ?4, 1, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    Uuid::new_v4().to_string(),
                    entry.scope.workspace_id,
                    entry.scope.project_id,
                    fingerprint_hash,
                    entry.expires_at.timestamp(),
                    embedding_json,
                    fingerprint_json,
                    entry.prompt,
                    response_str,
                    entry.created_at.to_rfc3339(),
                ],
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn invalidate_semantic(
        &self,
        scope: &SemanticCacheScope,
        fingerprint_hash: Option<&str>,
    ) -> EngineResult<u64> {
        let db = self.clone();
        let scope = scope.clone();
        let fingerprint_hash = fingerprint_hash.map(str::to_string);
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let affected = if let Some(fingerprint_hash) = fingerprint_hash {
                conn.execute(
                    "DELETE FROM semantic_response_cache
                     WHERE workspace_id = ?1 AND project_id = ?2 AND fingerprint_hash = ?3",
                    params![scope.workspace_id, scope.project_id, fingerprint_hash],
                )
            } else {
                conn.execute(
                    "DELETE FROM semantic_response_cache
                     WHERE workspace_id = ?1 AND project_id = ?2",
                    params![scope.workspace_id, scope.project_id],
                )
            }
            .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(affected as u64)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn invalidate(
        &self,
        tool_name: Option<&str>,
        key_pattern: Option<&str>,
    ) -> EngineResult<u64> {
        let db = self.clone();
        let tool_name = tool_name.map(|value| value.to_string());
        let key_pattern = key_pattern.map(|value| value.to_string());
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let mut sql = "DELETE FROM tool_cache WHERE 1=1".to_string();
            if tool_name.is_some() {
                sql.push_str(" AND tool_name = ?1");
            }
            if key_pattern.is_some() {
                sql.push_str(" AND cache_key LIKE ?2");
            }
            let affected = conn
                .execute(
                    &sql,
                    params![tool_name, key_pattern.map(|p| format!("%{}%", p))],
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(affected as u64)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }
}
