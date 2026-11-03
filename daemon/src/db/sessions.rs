//! Sessions persistence for the local engine.

use anyhow::Result;
use chrono::Utc;
use engine_core::{
    error::{EngineError, EngineResult},
    event::{EventStore, SessionEventType},
    session::{Session, SessionMode, SessionStatus},
};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use uuid::Uuid;

use super::{LocalDatabase, OwnedSessionFilter, SessionFilter};

impl LocalDatabase {
    pub(super) fn insert_session_on(conn: &Connection, session: &Session) -> EngineResult<()> {
        let task_summary =
            engine_core::privacy::current_policy().sanitize_persistent_text(&session.task_summary);
        // Accounting can finish with a snapshot captured before title generation.
        let title: Option<String> = conn
            .query_row(
                "SELECT title FROM generated_titles WHERE title_key = ?1 AND original_title = ?2",
                params![
                    format!("session:{}", session.root_session_id.unwrap_or(session.id)),
                    task_summary
                ],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| EngineError::Storage(e.to_string()))?;
        conn.execute(
            "INSERT INTO sessions (
                id, task_summary, project_path, project_name, project_id, client_id, external_session_id,
                parent_session_id, root_session_id, rollover_index, inherited_spend_cents,
                context_tokens, mode, status, budget_cents, spend_cents, tier,
                predicted_high_cents, cost_accuracy_bps, cost_accuracy_samples,
                cache_hit_rate_bps, cache_hit_rate_samples,
                total_input_tokens, total_output_tokens,
                is_subagent, subagent_parent_session_id,
                subagent_spawn_tool_call_id, subagent_spawn_request_group_id,
                created_at, updated_at, spend_nano_usd, inherited_spend_nano_usd, unknown_cost_calls
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32, ?33)
            ON CONFLICT(id) DO UPDATE SET
                task_summary=CASE WHEN EXISTS (
                    SELECT 1 FROM generated_titles
                    WHERE title_key = 'session:' || COALESCE(sessions.root_session_id, sessions.id)
                      AND sessions.task_summary != original_title
                ) THEN sessions.task_summary ELSE excluded.task_summary END,
                project_path=excluded.project_path,
                project_name=excluded.project_name,
                project_id=excluded.project_id,
                client_id=excluded.client_id,
                external_session_id=excluded.external_session_id,
                parent_session_id=excluded.parent_session_id,
                root_session_id=excluded.root_session_id,
                rollover_index=excluded.rollover_index,
                inherited_spend_cents=excluded.inherited_spend_cents,
                context_tokens=CASE WHEN EXISTS (SELECT 1 FROM completion_accounting WHERE session_id=sessions.id AND settled=1) THEN sessions.context_tokens ELSE excluded.context_tokens END,
                mode=excluded.mode,
                status=excluded.status,
                budget_cents=excluded.budget_cents,
                spend_cents=CASE WHEN EXISTS (SELECT 1 FROM completion_accounting WHERE session_id=sessions.id AND settled=1) THEN sessions.spend_cents ELSE excluded.spend_cents END,
                inherited_spend_nano_usd=excluded.inherited_spend_nano_usd,
                spend_nano_usd=CASE WHEN EXISTS (SELECT 1 FROM completion_accounting WHERE session_id=sessions.id AND settled=1) THEN sessions.spend_nano_usd ELSE excluded.spend_nano_usd END,
                unknown_cost_calls=CASE WHEN EXISTS (SELECT 1 FROM completion_accounting WHERE session_id=sessions.id AND settled=1) THEN sessions.unknown_cost_calls ELSE excluded.unknown_cost_calls END,
                tier=CASE WHEN EXISTS (SELECT 1 FROM completion_accounting WHERE session_id=sessions.id AND settled=1) THEN sessions.tier ELSE excluded.tier END,
                predicted_high_cents=CASE WHEN EXISTS (SELECT 1 FROM completion_accounting WHERE session_id=sessions.id AND settled=1) THEN sessions.predicted_high_cents ELSE excluded.predicted_high_cents END,
                cost_accuracy_bps=CASE WHEN EXISTS (SELECT 1 FROM completion_accounting WHERE session_id=sessions.id AND settled=1) THEN sessions.cost_accuracy_bps ELSE excluded.cost_accuracy_bps END,
                cost_accuracy_samples=CASE WHEN EXISTS (SELECT 1 FROM completion_accounting WHERE session_id=sessions.id AND settled=1) THEN sessions.cost_accuracy_samples ELSE excluded.cost_accuracy_samples END,
                cache_hit_rate_bps=CASE WHEN EXISTS (SELECT 1 FROM completion_accounting WHERE session_id=sessions.id AND settled=1) THEN sessions.cache_hit_rate_bps ELSE excluded.cache_hit_rate_bps END,
                cache_hit_rate_samples=CASE WHEN EXISTS (SELECT 1 FROM completion_accounting WHERE session_id=sessions.id AND settled=1) THEN sessions.cache_hit_rate_samples ELSE excluded.cache_hit_rate_samples END,
                total_input_tokens=CASE WHEN EXISTS (SELECT 1 FROM completion_accounting WHERE session_id=sessions.id AND settled=1) THEN sessions.total_input_tokens ELSE excluded.total_input_tokens END,
                total_output_tokens=CASE WHEN EXISTS (SELECT 1 FROM completion_accounting WHERE session_id=sessions.id AND settled=1) THEN sessions.total_output_tokens ELSE excluded.total_output_tokens END,
                is_subagent=excluded.is_subagent,
                subagent_parent_session_id=excluded.subagent_parent_session_id,
                subagent_spawn_tool_call_id=excluded.subagent_spawn_tool_call_id,
                subagent_spawn_request_group_id=excluded.subagent_spawn_request_group_id,
                updated_at=excluded.updated_at",
            params![
                session.id.to_string(),
                title.as_deref().unwrap_or(&task_summary),
                session.project_path.as_ref(),
                session.project_name.as_ref(),
                session.project_id.map(|id| id.to_string()),
                session.client_id.as_ref(),
                session.external_session_id.as_ref(),
                session.parent_session_id.map(|id| id.to_string()),
                session.root_session_id.map(|id| id.to_string()),
                session.rollover_index,
                session.inherited_spend_cents,
                session.context_tokens,
                session.mode.as_str(),
                status_to_str(session.status),
                session.budget_cents,
                session.spend_cents,
                session.tier,
                session.predicted_high_cents,
                session.cost_accuracy_bps,
                session.cost_accuracy_samples,
                session.cache_hit_rate_bps,
                session.cache_hit_rate_samples,
                session.total_input_tokens,
                session.total_output_tokens,
                session.is_subagent as i64,
                session.subagent_parent_session_id.as_ref(),
                session.subagent_spawn_tool_call_id.as_ref(),
                session.subagent_spawn_request_group_id.as_ref(),
                session.created_at.to_rfc3339(),
                session.updated_at.to_rfc3339(),
                session.spend_nano_usd, session.inherited_spend_nano_usd, session.unknown_cost_calls as i64,
            ],
        )
        .map_err(|e| EngineError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn insert_session(&self, session: &Session) -> EngineResult<()> {
        let conn = self.conn.lock().unwrap();
        Self::insert_session_on(&conn, session)
    }

    /// Display overrides are separate from immutable completion audit payloads.
    /// They may arrive before or after the first completion event is persisted.
    pub async fn save_generated_title(
        &self,
        session: Session,
        task: engine_core::TaskAssignment,
        title: String,
    ) -> EngineResult<()> {
        let privacy = engine_core::privacy::current_policy();
        let stored_title = privacy.sanitize_persistent_text(&title);
        let stored_task_name = privacy.sanitize_persistent_text(&task.task_name);
        let stored_session_summary = privacy.sanitize_persistent_text(&session.task_summary);
        let db = self.clone();
        let session_id = session.id;
        let notification =
            serde_json::json!({"task_id": task.task_id, "title": stored_title.clone()});
        let saved = tokio::task::spawn_blocking(move || -> EngineResult<bool> {
            let mut conn = db.conn.lock().unwrap();
            let tx = conn.transaction().map_err(|e| EngineError::Storage(e.to_string()))?;
            let inserted = tx.execute("INSERT OR IGNORE INTO generated_titles (title_key, title, original_title) VALUES (?1, ?2, ?3)",
                params![format!("task:{}:{}", session.id, task.task_id), stored_title, stored_task_name])
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            if inserted == 0 { return Ok(false); }
            // A new task in an existing session must not rename the session.
            if session.task_summary == task.task_name {
                let root = session.root_session_id.unwrap_or(session.id);
                tx.execute("INSERT OR IGNORE INTO generated_titles (title_key, title, original_title) VALUES (?1, ?2, ?3)",
                    params![format!("session:{root}"), stored_title, stored_session_summary])
                    .map_err(|e| EngineError::Storage(e.to_string()))?;
                tx.execute("UPDATE sessions SET task_summary = (SELECT title FROM generated_titles WHERE title_key = 'session:' || ?1) WHERE (id = ?1 OR root_session_id = ?1) AND task_summary = ?3",
                    params![root.to_string(), stored_title, stored_session_summary])
                    .map_err(|e| EngineError::Storage(e.to_string()))?;
            }
            tx.commit().map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(true)
        }).await.map_err(|e| EngineError::Storage(format!("title task join error: {e}")))??;
        if saved {
            self.append_event(
                session_id,
                SessionEventType::TitleUpdated,
                notification,
                None,
            )
            .await?;
        }
        Ok(())
    }

    pub async fn insert_session_async(&self, session: Session) -> EngineResult<()> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.insert_session(&session))
            .await
            .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    pub fn get_session(&self, id: Uuid) -> EngineResult<Option<Session>> {
        let conn = self.read_conn();
        Self::get_session_on(&conn, id)
    }

    pub(super) fn get_session_on(conn: &Connection, id: Uuid) -> EngineResult<Option<Session>> {
        let mut stmt = conn
            .prepare(
                "SELECT id, task_summary, project_path, project_name, project_id, client_id, external_session_id,
                        parent_session_id, root_session_id, rollover_index, inherited_spend_cents,
                        context_tokens, mode, status, budget_cents, spend_cents, tier,
                        predicted_high_cents, cost_accuracy_bps, cost_accuracy_samples,
                        cache_hit_rate_bps, cache_hit_rate_samples,
                        total_input_tokens, total_output_tokens,
                        is_subagent, subagent_parent_session_id,
                        subagent_spawn_tool_call_id, subagent_spawn_request_group_id,
                        created_at, updated_at, spend_nano_usd, inherited_spend_nano_usd, unknown_cost_calls
                 FROM sessions WHERE id = ?1",
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;
        let row = stmt
            .query_row([id.to_string()], session_from_row)
            .optional()
            .map_err(|e| EngineError::Storage(e.to_string()))?;
        Ok(row)
    }

    pub async fn rollover_session(
        &self,
        previous: Session,
        reason: String,
        context_tokens_after: u64,
        summary: Option<String>,
    ) -> EngineResult<Session> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = db.conn.lock().unwrap();
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(|e| EngineError::Storage(e.to_string()))?;
            let existing: Option<String> = tx.query_row("SELECT id FROM sessions WHERE parent_session_id=?1 AND rollover_index>0 ORDER BY rollover_index LIMIT 1",
                [previous.id.to_string()], |row| row.get(0)).optional().map_err(|e| EngineError::Storage(e.to_string()))?;
            if let Some(id) = existing {
                return Self::get_session_on(&tx, Uuid::parse_str(&id).map_err(|e| EngineError::Storage(e.to_string()))?)?
                    .ok_or_else(|| EngineError::Storage("missing rollover successor".into()));
            }
            let mut current = Self::get_session_on(&tx, previous.id)?.ok_or_else(|| EngineError::Storage("missing rollover predecessor".into()))?;
            if current.status != SessionStatus::Active { return Err(EngineError::InvalidInput("only an active session can roll over".into())); }
            current.context_tokens = previous.context_tokens;
            let before = current.context_tokens;
            let mut next = current.rollover(); next.context_tokens = context_tokens_after;
            if let Some(summary) = summary { next.task_summary = summary; }
            Self::insert_session_on(&tx, &current)?; Self::insert_session_on(&tx, &next)?;
            let payload = serde_json::json!({"previous_session_id":current.id,"next_session_id":next.id,
                "root_session_id":next.root_session_id.unwrap_or(current.id),"reason":reason,
                "context_tokens_before":before,"context_tokens_after":context_tokens_after});
            let mut notifications = vec![
                Self::append_event_on(&tx, current.id, SessionEventType::SessionRollover, payload.clone(), None)?,
                Self::append_event_on(&tx, next.id, SessionEventType::SessionRollover, payload, None)?,
            ];
            let carried: Vec<(String,String)> = {
                let mut stmt = tx.prepare("SELECT event_type,payload FROM session_events WHERE session_id=?1 AND
                    (event_type='task_decision' OR (event_type='continuation' AND seq IN
                        (SELECT MAX(seq) FROM session_events WHERE session_id=?1 AND event_type='continuation' GROUP BY json_extract(payload,'$.task_id')))) ORDER BY seq")
                    .map_err(|e| EngineError::Storage(e.to_string()))?;
                let rows = stmt.query_map([current.id.to_string()], |row| Ok((row.get(0)?,row.get(1)?))).map_err(|e| EngineError::Storage(e.to_string()))?;
                rows.collect::<Result<Vec<_>, _>>().map_err(|e| EngineError::Storage(e.to_string()))?
            };
            for (kind,payload) in carried {
                let mut payload: serde_json::Value = serde_json::from_str(&payload).map_err(|e| EngineError::Storage(e.to_string()))?;
                payload["previous_session_id"] = serde_json::json!(current.id);
                notifications.push(Self::append_event_on(&tx, next.id, super::events::str_to_event_type(&kind), payload, None)?);
            }
            tx.execute("UPDATE session_conversation_bindings SET session_id=?2 WHERE session_id=?1", params![current.id.to_string(), next.id.to_string()])
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            tx.commit().map_err(|e| EngineError::Storage(e.to_string()))?;
            for event in notifications { db.event_hub.publish(event); }
            Ok(next)
        }).await.map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    pub async fn get_session_async(&self, id: Uuid) -> EngineResult<Option<Session>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.get_session(id))
            .await
            .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    pub fn list_sessions(&self, filter: &SessionFilter<'_>) -> EngineResult<Vec<Session>> {
        let conn = self.read_conn();
        // Exclude subagent sessions - they are surfaced inside their
        // parent session's detail view, not as standalone list entries.
        // Only show sessions that have at least one event (i.e. were actually
        // used). The EXISTS subquery uses idx_session_events_session_id and
        // short-circuits after the first matching row.
        let mut where_clauses = vec!["is_subagent = 0".to_string()];
        // Project membership is fixed at session creation, but the first
        // durable event only lands after the first provider response settles.
        // Keep event-less (in-flight) sessions visible on project-scoped
        // listings so a running loop shows up with live 0-token totals
        // instead of vanishing until its first call completes.
        if filter.project_id.is_none() {
            where_clauses.push(
                "EXISTS (SELECT 1 FROM session_events WHERE session_id = sessions.id)".to_string(),
            );
        }
        let mut params: Vec<String> = Vec::new();

        if let Some(path) = filter.path {
            where_clauses.push(format!("project_path = ?{}", params.len() + 1));
            params.push(path.to_string());
        }
        if let Some(status) = filter.status {
            where_clauses.push(format!("status = ?{}", params.len() + 1));
            params.push(status.to_string());
        }
        if let Some(mode) = filter.mode {
            where_clauses.push(format!("mode = ?{}", params.len() + 1));
            params.push(mode.to_string());
        }
        if let Some(project_id) = filter.project_id {
            where_clauses.push(format!("project_id = ?{}", params.len() + 1));
            params.push(
                Uuid::parse_str(project_id)
                    .map_err(|_| EngineError::InvalidInput("invalid project_id".into()))?
                    .to_string(),
            );
        }
        if let Some(date_from) = filter.date_from {
            where_clauses.push(format!("updated_at >= ?{}", params.len() + 1));
            params.push(
                chrono::DateTime::parse_from_rfc3339(date_from)
                    .map_err(|_| EngineError::InvalidInput("invalid date filter".into()))?
                    .with_timezone(&Utc)
                    .to_rfc3339(),
            );
        }
        if let Some(date_to) = filter.date_to {
            where_clauses.push(format!("updated_at <= ?{}", params.len() + 1));
            params.push(
                chrono::DateTime::parse_from_rfc3339(date_to)
                    .map_err(|_| EngineError::InvalidInput("invalid date filter".into()))?
                    .with_timezone(&Utc)
                    .to_rfc3339(),
            );
        }

        if let Some(cursor) = filter.before {
            let (date, id) = cursor
                .split_once('|')
                .ok_or_else(|| EngineError::InvalidInput("invalid session cursor".into()))?;
            let date = chrono::DateTime::parse_from_rfc3339(date)
                .map_err(|_| EngineError::InvalidInput("invalid session cursor".into()))?
                .with_timezone(&Utc)
                .to_rfc3339();
            let id = Uuid::parse_str(id)
                .map_err(|_| EngineError::InvalidInput("invalid session cursor".into()))?;
            where_clauses.push(format!(
                "(updated_at, id) < (?{}, ?{})",
                params.len() + 1,
                params.len() + 2
            ));
            params.push(date);
            params.push(id.to_string());
        }
        let sql = format!(
            "SELECT id, task_summary, project_path, project_name, project_id, client_id, external_session_id,
                    parent_session_id, root_session_id, rollover_index, inherited_spend_cents,
                    context_tokens, mode, status, budget_cents, spend_cents, tier,
                    predicted_high_cents, cost_accuracy_bps, cost_accuracy_samples,
                    cache_hit_rate_bps, cache_hit_rate_samples,
                    total_input_tokens, total_output_tokens,
                    is_subagent, subagent_parent_session_id,
                    subagent_spawn_tool_call_id, subagent_spawn_request_group_id,
                    created_at, updated_at, spend_nano_usd, inherited_spend_nano_usd, unknown_cost_calls
             FROM sessions
             WHERE {}
             ORDER BY updated_at DESC, id DESC {}",
            where_clauses.join(" AND "),
            filter.limit.map(|n| format!("LIMIT {}", n.clamp(1, 1001))).unwrap_or_default()
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| EngineError::Storage(e.to_string()))?;
        let param_refs: Vec<&dyn rusqlite::ToSql> =
            params.iter().map(|p| p as &dyn rusqlite::ToSql).collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), session_from_row)
            .map_err(|e| EngineError::Storage(e.to_string()))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| EngineError::Storage(e.to_string()))
    }

    /// Same as `list_sessions` but takes an owned filter (for spawn_blocking).
    pub(super) fn list_sessions_owned(
        &self,
        filter: &OwnedSessionFilter,
    ) -> EngineResult<Vec<Session>> {
        let borrowed = SessionFilter {
            limit: filter.limit,
            before: filter.before.as_deref(),
            path: filter.path.as_deref(),
            status: filter.status.as_deref(),
            mode: filter.mode.as_deref(),
            project_id: filter.project_id.as_deref(),
            date_from: filter.date_from.as_deref(),
            date_to: filter.date_to.as_deref(),
        };
        self.list_sessions(&borrowed)
    }

    pub async fn list_sessions_async(
        &self,
        filter: SessionFilter<'_>,
    ) -> EngineResult<Vec<Session>> {
        // Convert borrowed filter to owned so it can move into spawn_blocking
        let owned = OwnedSessionFilter {
            limit: filter.limit,
            before: filter.before.map(str::to_owned),
            path: filter.path.map(|s| s.to_string()),
            status: filter.status.map(|s| s.to_string()),
            mode: filter.mode.map(|s| s.to_string()),
            project_id: filter.project_id.map(|s| s.to_string()),
            date_from: filter.date_from.map(|s| s.to_string()),
            date_to: filter.date_to.map(|s| s.to_string()),
        };
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.list_sessions_owned(&owned))
            .await
            .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    /// List subagent sessions whose `subagent_parent_session_id` matches the
    /// given external session ID. These are the child sessions spawned by a
    /// delegation tool call during the parent conversation.
    pub fn list_subagent_sessions(
        &self,
        parent_external_session_id: &str,
    ) -> EngineResult<Vec<Session>> {
        let conn = self.read_conn();
        let mut stmt = conn
            .prepare(
                "SELECT id, task_summary, project_path, project_name, project_id, client_id, external_session_id,
                    parent_session_id, root_session_id, rollover_index, inherited_spend_cents,
                    context_tokens, mode, status, budget_cents, spend_cents, tier,
                    predicted_high_cents, cost_accuracy_bps, cost_accuracy_samples,
                    cache_hit_rate_bps, cache_hit_rate_samples,
                    total_input_tokens, total_output_tokens,
                    is_subagent, subagent_parent_session_id,
                    subagent_spawn_tool_call_id, subagent_spawn_request_group_id,
                    created_at, updated_at, spend_nano_usd, inherited_spend_nano_usd, unknown_cost_calls
                 FROM sessions
                 WHERE is_subagent = 1 AND subagent_parent_session_id = ?1
                 ORDER BY created_at ASC",
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(
                rusqlite::params![parent_external_session_id],
                session_from_row,
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| EngineError::Storage(e.to_string()))
    }

    pub async fn list_subagent_sessions_async(
        &self,
        parent_external_session_id: String,
    ) -> EngineResult<Vec<Session>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.list_subagent_sessions(&parent_external_session_id))
            .await
            .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    pub fn get_active_session_by_external_id(
        &self,
        client_id: &str,
        project_id: Option<Uuid>,
        project_path: &str,
        external_session_id: &str,
    ) -> EngineResult<Option<Session>> {
        let conn = self.read_conn();
        conn.query_row(
            "SELECT id, task_summary, project_path, project_name, project_id, client_id, external_session_id,
                    parent_session_id, root_session_id, rollover_index, inherited_spend_cents,
                    context_tokens, mode, status, budget_cents, spend_cents, tier,
                    predicted_high_cents, cost_accuracy_bps, cost_accuracy_samples,
                    cache_hit_rate_bps, cache_hit_rate_samples,
                    total_input_tokens, total_output_tokens,
                    is_subagent, subagent_parent_session_id,
                    subagent_spawn_tool_call_id, subagent_spawn_request_group_id,
                    created_at, updated_at, spend_nano_usd, inherited_spend_nano_usd, unknown_cost_calls
             FROM sessions
             WHERE client_id = ?1 AND external_session_id = ?2 AND status = 'active'
               -- A project registered mid-loop (launch working-dir registration
               -- racing the first request) must not fork the loop into a second
               -- session: a row filed under the same path with no project yet
               -- belongs to the same loop.
               AND ((?3 IS NOT NULL AND (project_id = ?3 OR (project_id IS NULL AND project_path = ?4)))
                    OR (?3 IS NULL AND project_id IS NULL AND project_path = ?4))
             ORDER BY rollover_index DESC, updated_at DESC LIMIT 1",
            params![
                client_id,
                external_session_id,
                project_id.map(|id| id.to_string()),
                project_path
            ],
            session_from_row,
        )
        .optional()
        .map_err(|e| EngineError::Storage(e.to_string()))
    }

    pub async fn get_active_session_by_external_id_async(
        &self,
        client_id: String,
        project_id: Option<Uuid>,
        project_path: String,
        external_session_id: String,
    ) -> EngineResult<Option<Session>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            db.get_active_session_by_external_id(
                &client_id,
                project_id,
                &project_path,
                &external_session_id,
            )
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    pub fn get_latest_session_by_external_id(
        &self,
        client_id: &str,
        project_id: Option<Uuid>,
        project_path: &str,
        external_session_id: &str,
    ) -> EngineResult<Option<Session>> {
        let conn = self.read_conn();
        conn.query_row(
            "SELECT id, task_summary, project_path, project_name, project_id, client_id, external_session_id,
                    parent_session_id, root_session_id, rollover_index, inherited_spend_cents,
                    context_tokens, mode, status, budget_cents, spend_cents, tier,
                    predicted_high_cents, cost_accuracy_bps, cost_accuracy_samples,
                    cache_hit_rate_bps, cache_hit_rate_samples,
                    total_input_tokens, total_output_tokens,
                    is_subagent, subagent_parent_session_id,
                    subagent_spawn_tool_call_id, subagent_spawn_request_group_id,
                    created_at, updated_at, spend_nano_usd, inherited_spend_nano_usd, unknown_cost_calls
             FROM sessions
             WHERE client_id = ?1 AND external_session_id = ?2
               -- Same mid-loop registration tolerance as the active lookup above.
               AND ((?3 IS NOT NULL AND (project_id = ?3 OR (project_id IS NULL AND project_path = ?4)))
                    OR (?3 IS NULL AND project_id IS NULL AND project_path = ?4))
             ORDER BY rollover_index DESC, updated_at DESC LIMIT 1",
            params![
                client_id,
                external_session_id,
                project_id.map(|id| id.to_string()),
                project_path
            ],
            session_from_row,
        )
        .optional()
        .map_err(|e| EngineError::Storage(e.to_string()))
    }

    pub async fn get_latest_session_by_external_id_async(
        &self,
        client_id: String,
        project_id: Option<Uuid>,
        project_path: String,
        external_session_id: String,
    ) -> EngineResult<Option<Session>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            db.get_latest_session_by_external_id(
                &client_id,
                project_id,
                &project_path,
                &external_session_id,
            )
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    pub fn get_active_session_in_lineage(&self, id: Uuid) -> EngineResult<Option<Session>> {
        let root = self
            .get_session(id)?
            .map(|session| session.root_session_id.unwrap_or(session.id));
        let Some(root) = root else {
            return Ok(None);
        };
        let conn = self.read_conn();
        conn.query_row(
            "SELECT id, task_summary, project_path, project_name, project_id, client_id, external_session_id,
                    parent_session_id, root_session_id, rollover_index, inherited_spend_cents,
                    context_tokens, mode, status, budget_cents, spend_cents, tier,
                    predicted_high_cents, cost_accuracy_bps, cost_accuracy_samples,
                    cache_hit_rate_bps, cache_hit_rate_samples,
                    total_input_tokens, total_output_tokens,
                    is_subagent, subagent_parent_session_id,
                    subagent_spawn_tool_call_id, subagent_spawn_request_group_id,
                    created_at, updated_at, spend_nano_usd, inherited_spend_nano_usd, unknown_cost_calls
             FROM sessions
             WHERE status = 'active' AND (id = ?1 OR root_session_id = ?1)
             ORDER BY rollover_index DESC LIMIT 1",
            [root.to_string()],
            session_from_row,
        )
        .optional()
        .map_err(|e| EngineError::Storage(e.to_string()))
    }

    pub async fn get_active_session_in_lineage_async(
        &self,
        id: Uuid,
    ) -> EngineResult<Option<Session>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.get_active_session_in_lineage(id))
            .await
            .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    /// Rename every physical session in a rollover lineage. Narrower than
    /// `insert_session` so a rename can never roll back concurrent accounting.
    pub fn update_session_summary(&self, root_id: Uuid, task_summary: &str) -> EngineResult<()> {
        let task_summary =
            engine_core::privacy::current_policy().sanitize_persistent_text(task_summary);
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE sessions SET task_summary = ?2, updated_at = ?3
             WHERE id = ?1 OR root_session_id = ?1",
            params![
                root_id.to_string(),
                task_summary,
                chrono::Utc::now().to_rfc3339()
            ],
        )
        .map_err(|e| EngineError::Storage(e.to_string()))?;
        Ok(())
    }

    pub async fn update_session_summary_async(
        &self,
        root_id: Uuid,
        task_summary: String,
    ) -> EngineResult<()> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.update_session_summary(root_id, &task_summary))
            .await
            .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }
}

pub(super) fn session_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Session> {
    let id: String = row.get(0)?;
    let project_id: Option<String> = row.get(4)?;
    let parent: Option<String> = row.get(7)?;
    let root: Option<String> = row.get(8)?;
    let mode: String = row.get(12)?;
    let status: String = row.get(13)?;
    Ok(Session {
        id: Uuid::parse_str(&id).unwrap_or_default(),
        task_summary: row.get(1)?,
        project_path: row.get(2)?,
        project_name: row.get(3)?,
        workspace_id: None,
        project_id: project_id.and_then(|id| Uuid::parse_str(&id).ok()),
        user_id: None,
        client_id: row.get(5)?,
        external_session_id: row.get(6)?,
        parent_session_id: parent.and_then(|id| Uuid::parse_str(&id).ok()),
        root_session_id: root.and_then(|id| Uuid::parse_str(&id).ok()),
        rollover_index: row.get(9)?,
        inherited_spend_cents: row.get(10)?,
        context_tokens: row.get(11)?,
        mode: str_to_mode(&mode),
        status: str_to_status(&status),
        budget_cents: row.get(14)?,
        spend_cents: row.get(15)?,
        spend_nano_usd: row.get("spend_nano_usd")?,
        inherited_spend_nano_usd: row.get("inherited_spend_nano_usd")?,
        unknown_cost_calls: row.get::<_, i64>("unknown_cost_calls")? as u64,
        tier: row.get(16)?,
        // Columns 16-18 were added by the cost-calibration migration, and
        // columns 19-20 by the cache-hit-rate migration. Use unwrap_or(None)
        // so a row from a database that hasn't run the migration yet (or a
        // column the SELECT didn't include) degrades gracefully to an
        // uncalibrated/unsampled session instead of panicking.
        predicted_high_cents: row.get(17).unwrap_or(None),
        cost_accuracy_bps: row.get(18).unwrap_or(None),
        cost_accuracy_samples: row.get(19).unwrap_or(None),
        cache_hit_rate_bps: row.get(20).unwrap_or(None),
        cache_hit_rate_samples: row.get(21).unwrap_or(None),
        // Columns 22-23 were added by the cumulative-tokens migration. Use
        // unwrap_or(0) so a row from a database that hasn't run the
        // migration yet (or a column the SELECT didn't include) degrades
        // gracefully to zero instead of panicking.
        total_input_tokens: row.get::<_, i64>(22).unwrap_or(0) as u64,
        total_output_tokens: row.get::<_, i64>(23).unwrap_or(0) as u64,
        cost_outcome: None,
        // Columns 24-25 were added by the subagent-tracking migration.
        // Use unwrap_or defaults so rows from a pre-migration database (or
        // a SELECT that didn't include these columns) degrade gracefully.
        is_subagent: row.get::<_, i64>(24).unwrap_or(0) != 0,
        subagent_parent_session_id: row.get(25).unwrap_or(None),
        subagent_spawn_tool_call_id: row.get(26).unwrap_or(None),
        subagent_spawn_request_group_id: row.get(27).unwrap_or(None),
        created_at: row
            .get::<_, String>(28)?
            .parse()
            .unwrap_or_else(|_| Utc::now()),
        updated_at: row
            .get::<_, String>(29)?
            .parse()
            .unwrap_or_else(|_| Utc::now()),
    })
}

pub(super) fn str_to_mode(s: &str) -> SessionMode {
    s.parse().unwrap_or(SessionMode::Interactive)
}

pub(super) fn status_to_str(status: SessionStatus) -> &'static str {
    match status {
        SessionStatus::Active => "active",
        SessionStatus::Paused => "paused",
        SessionStatus::Completed => "completed",
        SessionStatus::Checkpointed => "checkpointed",
    }
}

pub(super) fn str_to_status(s: &str) -> SessionStatus {
    match s {
        "paused" => SessionStatus::Paused,
        "completed" => SessionStatus::Completed,
        "checkpointed" => SessionStatus::Checkpointed,
        _ => SessionStatus::Active,
    }
}
