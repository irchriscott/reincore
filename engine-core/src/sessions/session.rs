//! Session lifecycle and metadata.
//!
//! Defines the [`Session`] struct - the core state object tracking a
//! ReinCore agent conversation: its ID, task summary, project/workspace
//! binding, budget, spend lineage, and lifecycle status (`Active`, `Paused`,
//! `Completed`, `Checkpointed`). Sessions can run in `Interactive` or
//! `Background` mode and maintain a parent/child lineage so budget and spend
//! are inherited across handoffs.

use crate::error::{EngineError, EngineResult};
use crate::CostOutcome;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Lifecycle status of a ReinCore session.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    #[default]
    Active,
    Paused,
    Completed,
    Checkpointed,
}

/// Execution mode for a session.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SessionMode {
    #[default]
    Interactive,
    Background,
}

impl SessionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            SessionMode::Interactive => "interactive",
            SessionMode::Background => "background",
        }
    }
}

impl std::str::FromStr for SessionMode {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "interactive" => Ok(SessionMode::Interactive),
            "background" => Ok(SessionMode::Background),
            _ => Err(()),
        }
    }
}

/// A ReinCore agent session.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Session {
    pub id: Uuid,
    pub task_summary: String,
    /// Project or workspace path this session belongs to (especially useful
    /// for the local edition so sessions can be traced to a folder).
    pub project_path: Option<String>,
    /// Human-friendly project name used for grouping usage and context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_name: Option<String>,
    /// Control-plane workspace this session belongs to. Cloud authorization
    /// uses this boundary before applying project or user-level access rules.
    /// It is `None` for local sessions and cloud sessions created before the
    /// workspace attribution migration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_id: Option<Uuid>,
    /// Control-plane project this session is attributed to. It is signed into
    /// the engine JWT at mint time; the control plane is authoritative and may
    /// reassign it, so the engine sets it only when the session is first
    /// created and never overwrites it afterwards.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<Uuid>,
    /// Authenticated ReinCore user who created this session (the engine JWT
    /// `sub`). It records who made the call - most useful for CLI/`reincore
    /// auth` logins. Set once at creation and never overwritten; `None` for
    /// local/unauthenticated sessions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<Uuid>,
    /// Stable caller/application identity. Cloud deployments should use the
    /// authenticated user id; local deployments default to `local`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
    /// Session identifier supplied by an external application. It remains
    /// stable when ReinCore rolls the physical session over.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_session_id: Option<String>,
    /// Previous physical session in this context-window lineage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<Uuid>,
    /// First physical session in this context-window lineage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_session_id: Option<Uuid>,
    /// Number of automatic context-window rollovers from the root session.
    #[serde(default)]
    pub rollover_index: u32,
    /// Spend from completed ancestors. This preserves budget degradation
    /// across rollover without duplicating spend in project aggregates.
    #[serde(default)]
    pub inherited_spend_cents: i64,
    /// Last observed provider input-context size for this physical session.
    #[serde(default)]
    pub context_tokens: u64,
    /// Session interaction mode. Background sessions are eligible for
    /// non-latency-sensitive/batch routing.
    #[serde(default)]
    pub mode: SessionMode,
    pub status: SessionStatus,
    /// Pre-flight estimate / approved ceiling, in cents of the billing currency.
    pub budget_cents: i64,
    /// Running total spend, denormalized for fast reads.
    pub spend_cents: i64,
    /// Estimated USD in billionths of a dollar; None identifies legacy cent-only data.
    #[serde(default)]
    pub spend_nano_usd: Option<i64>,
    #[serde(default)]
    pub inherited_spend_nano_usd: Option<i64>,
    #[serde(default)]
    pub unknown_cost_calls: u64,
    /// Current budget degradation tier (1-4).
    pub tier: u8,
    /// Most recent pre-flight cost prediction's high bound, in cents. Set
    /// before each provider call so a feedback loop can compare it against
    /// the actual billed cost after the call completes. `None` for sessions
    /// created before the cost-calibration migration or for free providers
    /// whose prediction is always zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted_high_cents: Option<i64>,
    /// Legacy rounded running call mean of actual / predicted high * 10,000.
    /// This ratio may exceed 100%; it is not prediction error or answer accuracy.
    /// Exact event-derived metrics are available in cost_outcome.diagnostics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(
        rename = "actual_to_predicted_high_ratio_bps",
        alias = "cost_accuracy_bps"
    )]
    pub cost_accuracy_bps: Option<i64>,
    /// Number of LLM calls that have contributed to `cost_accuracy_bps`.
    /// Incremented each time a post-completion accuracy ratio is recorded,
    /// so aggregations can weight sessions by sample size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_accuracy_samples: Option<i64>,
    /// Running sample-weighted mean of the provider-side cache-prefix hit
    /// rate, in basis points (e.g. `7500` = 75 % of prompt tokens were
    /// served from the provider's prefix cache). `None` until at least one
    /// call with a non-zero `prompt_tokens` has been recorded. See
    /// [`crate::cost::cache_hit_rate_bps`] for the per-call definition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(
        rename = "legacy_mean_call_cache_hit_rate_bps",
        alias = "cache_hit_rate_bps"
    )]
    pub cache_hit_rate_bps: Option<i64>,
    /// Number of LLM calls that have contributed to `cache_hit_rate_bps`.
    /// Incremented each time a post-completion cache-hit ratio is recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_hit_rate_samples: Option<i64>,
    /// Cumulative prompt (input) tokens across all LLM calls in this
    /// physical session. Unlike `context_tokens` (which is overwritten
    /// each turn with the latest prompt size), this is a running total
    /// that grows monotonically so the session detail view can surface
    /// the true token consumption for the entire session.
    #[serde(default)]
    pub total_input_tokens: u64,
    /// Cumulative completion (output) tokens across all LLM calls in
    /// this physical session. Mirrors `total_input_tokens` for the
    /// provider's response tokens.
    #[serde(default)]
    pub total_output_tokens: u64,
    /// Event-log-derived lifecycle, reported outcome and call diagnostics. This is a
    /// read projection and is not persisted in the sessions table.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_outcome: Option<CostOutcome>,
    /// True when this session was spawned through a client delegation tool
    /// rather than a direct user conversation.
    /// Set at creation from the `reincore_is_subagent` request extra. See
    /// [`Self::subagent_parent_session_id`] for the parent linkage.
    #[serde(default)]
    pub is_subagent: bool,
    /// The external session identifier of the parent conversation that
    /// spawned this subagent. Resolved from the
    /// `reincore_subagent_parent_session_id` request extra at session-creation
    /// time. This is distinct from [`Self::parent_session_id`],
    /// which links physical sessions in a context-window rollover lineage.
    /// Stored as a String because it is the client-provided external session
    /// identifier (a SHA-256 hash), not a server-generated UUID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent_parent_session_id: Option<String>,
    /// Tool-call id of the delegation invocation that created this child.
    /// This is stable across replay and is used for endpoint/UI correlation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent_spawn_tool_call_id: Option<String>,
    /// Parent request group in which the delegation tool was invoked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent_spawn_request_group_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Aggregate usage for project dashboards and external clients.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProjectUsage {
    pub project_name: String,
    pub project_path: Option<String>,
    /// Stable local-project identifier when the aggregate represents a
    /// registered project. Legacy and cloud aggregates may not have one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<Uuid>,
    pub session_count: u64,
    pub active_session_count: u64,
    pub total_spend_cents: i64,
    #[serde(default)]
    pub total_spend_nano_usd: i64,
    #[serde(default)]
    pub unknown_cost_calls: u64,
    pub active_context_tokens: u64,
    pub recorded_context_tokens: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
}

/// Event-derived cost calibration; legacy route name retained.
pub type CostAccuracyBreakdown = crate::diagnostics::PredictionBreakdown;
/// Event-derived token-weighted cache utilization.
pub type CacheHitRateBreakdown = crate::diagnostics::CacheBreakdown;

/// Successful checkpoint resumptions for a dashboard scope.
///
/// `resumed_session_count` counts distinct top-level sessions, while
/// `redemption_count` counts every successfully consumed checkpoint token.
/// A session can therefore contribute more than one redemption over its
/// lifetime without inflating the number of sessions recovered.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckpointRedemptionBreakdown {
    /// Human-readable label for the scope this breakdown covers.
    pub scope: String,
    /// Number of distinct sessions with at least one successful redemption.
    pub resumed_session_count: u64,
    /// Total number of successfully redeemed checkpoint tokens.
    pub redemption_count: u64,
    /// Most recent successful redemption, or `None` when there are none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latest_redeemed_at: Option<DateTime<Utc>>,
}

impl Session {
    pub fn new(task_summary: impl Into<String>, budget_cents: i64) -> EngineResult<Self> {
        Self::new_with_path(task_summary, budget_cents, None::<String>)
    }

    pub fn new_with_path(
        task_summary: impl Into<String>,
        budget_cents: i64,
        project_path: Option<impl Into<String>>,
    ) -> EngineResult<Self> {
        Self::new_tracked(
            task_summary,
            budget_cents,
            None::<String>,
            project_path,
            None::<String>,
            None::<String>,
        )
    }

    pub fn new_tracked(
        task_summary: impl Into<String>,
        budget_cents: i64,
        project_name: Option<impl Into<String>>,
        project_path: Option<impl Into<String>>,
        client_id: Option<impl Into<String>>,
        external_session_id: Option<impl Into<String>>,
    ) -> EngineResult<Self> {
        if budget_cents < 0 {
            return Err(EngineError::InvalidBudget(
                "budget cannot be negative".to_string(),
            ));
        }
        let now = Utc::now();
        Ok(Self {
            id: Uuid::new_v4(),
            task_summary: task_summary.into(),
            project_path: project_path.map(Into::into),
            project_name: project_name.map(Into::into),
            workspace_id: None,
            project_id: None,
            user_id: None,
            client_id: client_id.map(Into::into),
            external_session_id: external_session_id.map(Into::into),
            parent_session_id: None,
            root_session_id: None,
            rollover_index: 0,
            inherited_spend_cents: 0,
            context_tokens: 0,
            mode: SessionMode::Interactive,
            status: SessionStatus::Active,
            budget_cents,
            spend_cents: 0,
            spend_nano_usd: None,
            inherited_spend_nano_usd: None,
            unknown_cost_calls: 0,
            tier: 1,
            predicted_high_cents: None,
            cost_accuracy_bps: None,
            cost_accuracy_samples: None,
            cache_hit_rate_bps: None,
            cache_hit_rate_samples: None,
            total_input_tokens: 0,
            total_output_tokens: 0,
            cost_outcome: None,
            is_subagent: false,
            subagent_parent_session_id: None,
            subagent_spawn_tool_call_id: None,
            subagent_spawn_request_group_id: None,
            created_at: now,
            updated_at: now,
        })
    }

    pub fn spend_ratio(&self) -> f64 {
        if self.budget_cents == 0 {
            return 0.0;
        }
        (self.lineage_spend_nano_usd() as f64)
            / (self.budget_cents as f64 * crate::pricing::NANO_USD_PER_CENT as f64)
    }

    /// Spend used for budget decisions across a chain of physical sessions.
    pub fn lineage_spend_cents(&self) -> i64 {
        self.lineage_spend_nano_usd() / crate::pricing::NANO_USD_PER_CENT
    }

    pub fn lineage_spend_nano_usd(&self) -> i64 {
        self.spend_nano_usd
            .unwrap_or_else(|| {
                self.spend_cents
                    .saturating_mul(crate::pricing::NANO_USD_PER_CENT)
            })
            .saturating_add(self.inherited_spend_nano_usd.unwrap_or_else(|| {
                self.inherited_spend_cents
                    .saturating_mul(crate::pricing::NANO_USD_PER_CENT)
            }))
    }

    /// Complete this physical session and create its active successor.
    pub fn rollover(&mut self) -> Self {
        let now = Utc::now();
        self.status = SessionStatus::Completed;
        self.updated_at = now;
        let root_id = self.root_session_id.unwrap_or(self.id);
        Self {
            id: Uuid::new_v4(),
            task_summary: self.task_summary.clone(),
            project_path: self.project_path.clone(),
            project_name: self.project_name.clone(),
            workspace_id: self.workspace_id,
            project_id: self.project_id,
            user_id: self.user_id,
            client_id: self.client_id.clone(),
            external_session_id: self.external_session_id.clone(),
            parent_session_id: Some(self.id),
            root_session_id: Some(root_id),
            rollover_index: self.rollover_index.saturating_add(1),
            inherited_spend_cents: self.lineage_spend_cents(),
            inherited_spend_nano_usd: Some(self.lineage_spend_nano_usd()),
            context_tokens: 0,
            mode: self.mode,
            status: SessionStatus::Active,
            budget_cents: self.budget_cents,
            spend_cents: 0,
            spend_nano_usd: None,
            unknown_cost_calls: 0,
            tier: self.tier,
            // Calibration is per-physical-session: the successor gets a fresh
            // prediction on its first call, so it starts uncalibrated rather
            // than inheriting stale numbers from the predecessor.
            predicted_high_cents: None,
            cost_accuracy_bps: None,
            cost_accuracy_samples: None,
            cache_hit_rate_bps: None,
            cache_hit_rate_samples: None,
            // Cumulative tokens are per-physical-session: the successor
            // starts fresh since it has a new context window.
            total_input_tokens: 0,
            total_output_tokens: 0,
            cost_outcome: None,
            // Subagent identity is preserved across rollover: if this
            // physical session was a subagent, its successor continues
            // to be one (same parent linkage).
            is_subagent: self.is_subagent,
            subagent_parent_session_id: self.subagent_parent_session_id.clone(),
            subagent_spawn_tool_call_id: self.subagent_spawn_tool_call_id.clone(),
            subagent_spawn_request_group_id: self.subagent_spawn_request_group_id.clone(),
            created_at: now,
            updated_at: now,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_spend_ratio() {
        let session = Session::new("test task", 1000).unwrap();
        assert_eq!(session.spend_ratio(), 0.0);

        let mut spent = session.clone();
        spent.spend_cents = 750;
        assert!((spent.spend_ratio() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn rollover_links_sessions_and_preserves_budget_lineage() {
        let mut session = Session::new_tracked(
            "task",
            1_000,
            Some("project"),
            Some("/workspace/project"),
            Some("client"),
            Some("external"),
        )
        .unwrap();
        session.spend_cents = 400;
        session.context_tokens = 100_000;

        let child = session.rollover();
        assert_eq!(session.status, SessionStatus::Completed);
        assert_eq!(child.parent_session_id, Some(session.id));
        assert_eq!(child.root_session_id, Some(session.id));
        assert_eq!(child.external_session_id.as_deref(), Some("external"));
        assert_eq!(child.spend_cents, 0);
        assert_eq!(child.lineage_spend_cents(), 400);
        assert_eq!(child.context_tokens, 0);
    }

    #[test]
    fn rollover_does_not_inherit_cost_calibration() {
        let mut session = Session::new("task", 1_000).unwrap();
        session.predicted_high_cents = Some(50);
        session.cost_accuracy_bps = Some(750);
        session.cost_accuracy_samples = Some(3);

        let child = session.rollover();
        assert_eq!(child.predicted_high_cents, None);
        assert_eq!(child.cost_accuracy_bps, None);
        assert_eq!(child.cost_accuracy_samples, None);
    }

    #[test]
    fn rollover_does_not_inherit_cache_hit_rate() {
        let mut session = Session::new("task", 1_000).unwrap();
        session.cache_hit_rate_bps = Some(6000);
        session.cache_hit_rate_samples = Some(5);

        let child = session.rollover();
        assert_eq!(child.cache_hit_rate_bps, None);
        assert_eq!(child.cache_hit_rate_samples, None);
    }
}
