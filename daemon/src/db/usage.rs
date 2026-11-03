//! Usage persistence for the local engine.

use anyhow::Result;
use chrono::{DateTime, Utc};
use engine_core::{
    error::{EngineError, EngineResult},
    session::{CheckpointRedemptionBreakdown, ProjectUsage},
};
use rusqlite::params;
use uuid::Uuid;

use super::LocalDatabase;

impl LocalDatabase {
    /// Aggregate successful checkpoint redemptions across top-level local
    /// sessions. The partial event index added in migration 17 keeps this
    /// dashboard query proportional to redemption rows, not all events.
    pub fn aggregate_checkpoint_redemptions(&self) -> EngineResult<CheckpointRedemptionBreakdown> {
        let conn = self.read_conn();
        let (resumed_session_count, redemption_count, latest_redeemed_at): (
            i64,
            i64,
            Option<String>,
        ) = conn
            .query_row(
                "SELECT COUNT(DISTINCT events.session_id),
                        COUNT(*),
                        MAX(events.created_at)
                 FROM session_events AS events
                 INNER JOIN sessions ON sessions.id = events.session_id
                 WHERE events.event_type = 'checkpoint_redeemed'
                   AND sessions.is_subagent = 0",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .map_err(|error| EngineError::Storage(error.to_string()))?;
        let latest_redeemed_at = latest_redeemed_at
            .map(|value| {
                DateTime::parse_from_rfc3339(&value)
                    .map(|timestamp| timestamp.with_timezone(&Utc))
                    .map_err(|error| EngineError::Storage(error.to_string()))
            })
            .transpose()?;
        Ok(CheckpointRedemptionBreakdown {
            scope: "all".to_string(),
            resumed_session_count: resumed_session_count.max(0) as u64,
            redemption_count: redemption_count.max(0) as u64,
            latest_redeemed_at,
        })
    }

    pub async fn aggregate_checkpoint_redemptions_async(
        &self,
    ) -> EngineResult<CheckpointRedemptionBreakdown> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.aggregate_checkpoint_redemptions())
            .await
            .map_err(|error| EngineError::Storage(format!("sqlite task join error: {error}")))?
    }

    pub fn list_project_usage(&self) -> EngineResult<Vec<ProjectUsage>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT COALESCE(p.name, s.project_name, '(unassigned)'), MIN(s.project_path),
                        MAX(s.project_id),
                        COUNT(*), SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END),
                        (COALESCE(SUM(COALESCE(spend_nano_usd, spend_cents * 10000000)), 0) / 10000000),
                        COALESCE(SUM(CASE WHEN status = 'active' THEN context_tokens ELSE 0 END), 0),
                        COALESCE(SUM(context_tokens), 0),
                        COALESCE(SUM(s.total_input_tokens), 0),
                        COALESCE(SUM(s.total_output_tokens), 0),
                        COALESCE(SUM(COALESCE(s.spend_nano_usd, s.spend_cents * 10000000)), 0),
                        COALESCE(SUM(s.unknown_cost_calls), 0)
                 FROM sessions s
                 LEFT JOIN local_projects p ON p.id = s.project_id
                 WHERE s.is_subagent = 0
                 GROUP BY CASE
                     WHEN s.project_id IS NOT NULL THEN 'project:' || s.project_id
                     ELSE 'legacy:' || COALESCE(s.project_name, '') || ':' || COALESCE(s.project_path, '')
                 END
                 ORDER BY (COALESCE(SUM(COALESCE(spend_nano_usd, spend_cents * 10000000)), 0) / 10000000) DESC",
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map([], |row| {
                let project_id: Option<String> = row.get(2)?;
                Ok(ProjectUsage {
                    project_name: row.get(0)?,
                    project_path: row.get(1)?,
                    project_id: project_id.and_then(|id| Uuid::parse_str(&id).ok()),
                    session_count: row.get(3)?,
                    active_session_count: row.get(4)?,
                    total_spend_cents: row.get(5)?,
                    total_spend_nano_usd: row.get(10)?,
                    unknown_cost_calls: row.get::<_, i64>(11)? as u64,
                    active_context_tokens: row.get(6)?,
                    recorded_context_tokens: row.get(7)?,
                    total_input_tokens: row.get(8)?,
                    total_output_tokens: row.get(9)?,
                })
            })
            .map_err(|e| EngineError::Storage(e.to_string()))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| EngineError::Storage(e.to_string()))
    }

    pub async fn list_project_usage_async(&self) -> EngineResult<Vec<ProjectUsage>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.list_project_usage())
            .await
            .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    // Mirrors the flat cost_estimates row and keeps call sites explicit.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_cost_estimate(
        &self,
        session_id: Option<Uuid>,
        request_group_id: Option<&str>,
        request_type: &str,
        model_tier: &str,
        input_tokens: u64,
        output_tokens: u64,
        low_cents: i64,
        high_cents: i64,
    ) -> EngineResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO cost_estimates (
                session_id, request_group_id, request_type, model_tier, input_tokens, output_tokens, low_cents, high_cents, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                session_id.map(|id| id.to_string()),
                request_group_id,
                request_type,
                model_tier,
                input_tokens as i64,
                output_tokens as i64,
                low_cents,
                high_cents,
                Utc::now().to_rfc3339(),
            ],
        )
        .map_err(|e| EngineError::Storage(e.to_string()))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn insert_cost_estimate_async(
        &self,
        session_id: Option<Uuid>,
        request_group_id: Option<String>,
        request_type: String,
        model_tier: String,
        input_tokens: u64,
        output_tokens: u64,
        low_cents: i64,
        high_cents: i64,
    ) -> EngineResult<()> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            db.insert_cost_estimate(
                session_id,
                request_group_id.as_deref(),
                &request_type,
                &model_tier,
                input_tokens,
                output_tokens,
                low_cents,
                high_cents,
            )
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }
}
