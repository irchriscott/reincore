//! Progressive budget degradation and spend ledger.
//!
//! Tracks per-session spend against a user-configured budget and drives a
//! four-tier state machine (Tier 1 → Tier 4) as the spend ratio rises:
//!
//! - **Tier 1** (0–70 %): normal operation, standard model.
//! - **Tier 2** (70–90 %): routine steps routed to the cheaper Lite model.
//! - **Tier 3** (90–99 %): aggressive context compaction; costly external
//!   tools blocked.
//! - **Tier 4** (100 %+): pause the session and emit a resumable checkpoint.
//!
//! All spend events are written to an append-only [`LedgerEvent`] log so the
//! full cost lineage of a session (and its successors) is reconstructable.

use crate::compression::{CompressionSettings, ContextCompressor};
use crate::error::{EngineError, EngineResult};
use crate::event::{EventStore, SessionEventType, TierChangePayload};
use crate::session::Session;
use crate::ModelTier;
use async_trait::async_trait;
use chrono::{DateTime, Datelike, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A single append-only spend event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LedgerEvent {
    pub session_id: Uuid,
    pub amount_cents: i64,
    pub reason: LedgerReason,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LedgerReason {
    LlmCall,
    ToolCall,
    Completion,
    EstimateLock,
}

impl LedgerReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            LedgerReason::LlmCall => "llm_call",
            LedgerReason::ToolCall => "tool_call",
            LedgerReason::Completion => "completion",
            LedgerReason::EstimateLock => "estimate_lock",
        }
    }
}

/// The four progressive budget degradation tiers.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd)]
pub enum BudgetTier {
    /// 0-70% of budget: normal operation.
    Tier1,
    /// 70-90%: route routine steps to lite model.
    Tier2,
    /// 90-99%: aggressive context compaction; block costly external tools.
    Tier3,
    /// 100%+: pause with a resumable checkpoint.
    Tier4,
}

impl BudgetTier {
    pub fn from_ratio(ratio: f64) -> Self {
        match ratio {
            r if r < 0.70 => BudgetTier::Tier1,
            r if r < 0.90 => BudgetTier::Tier2,
            r if r < 1.00 => BudgetTier::Tier3,
            _ => BudgetTier::Tier4,
        }
    }

    pub fn as_u8(&self) -> u8 {
        match self {
            BudgetTier::Tier1 => 1,
            BudgetTier::Tier2 => 2,
            BudgetTier::Tier3 => 3,
            BudgetTier::Tier4 => 4,
        }
    }

    /// Apply the quality ceiling associated with this budget tier.
    /// Tier 4 has no eligible model because the session must checkpoint.
    pub fn cap_model_tier(self, selected: ModelTier) -> Option<ModelTier> {
        match self {
            BudgetTier::Tier1 => Some(selected),
            BudgetTier::Tier2 => Some(match selected {
                ModelTier::Ultra => ModelTier::Pro,
                other => other,
            }),
            BudgetTier::Tier3 => Some(ModelTier::Lite),
            BudgetTier::Tier4 => None,
        }
    }
}

/// Scope at which a budget is enforced.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum BudgetScope {
    /// Budget applies to a single session.
    Session,
    /// Budget applies to all sessions for a named project or workspace.
    Project,
    /// Budget applies to all sessions on a single calendar day.
    Day,
    /// Budget applies to all sessions in a calendar month.
    Month,
}

impl BudgetScope {
    pub fn as_str(&self) -> &'static str {
        match self {
            BudgetScope::Session => "session",
            BudgetScope::Project => "project",
            BudgetScope::Day => "day",
            BudgetScope::Month => "month",
        }
    }
}

/// A budget policy record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BudgetPolicy {
    pub id: Uuid,
    pub scope: BudgetScope,
    /// For project-scoped budgets, the project or workspace name. For
    /// session/day/month, this is empty or a derived key.
    pub scope_key: String,
    pub budget_cents: i64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A historical record of a budget update.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BudgetHistoryEntry {
    pub policy_id: Uuid,
    pub scope: BudgetScope,
    pub scope_key: String,
    pub previous_budget_cents: i64,
    pub new_budget_cents: i64,
    pub updated_at: DateTime<Utc>,
}

/// Storage contract for budget policies and their history.
#[async_trait]
pub trait BudgetStore: Send + Sync {
    /// Set or update a budget policy. Returns the policy and a history entry.
    async fn set_budget(
        &self,
        scope: BudgetScope,
        scope_key: String,
        budget_cents: i64,
    ) -> EngineResult<(BudgetPolicy, BudgetHistoryEntry)>;

    /// Get the current budget policy for a scope.
    async fn get_budget(
        &self,
        scope: BudgetScope,
        scope_key: &str,
    ) -> EngineResult<Option<BudgetPolicy>>;

    /// List history entries for a scope, newest first.
    async fn list_budget_history(
        &self,
        scope: BudgetScope,
        scope_key: Option<&str>,
        limit: Option<usize>,
    ) -> EngineResult<Vec<BudgetHistoryEntry>>;
}

/// Storage contract for global engine settings such as context compaction.
#[async_trait]
pub trait SettingsStore: Send + Sync {
    async fn get_compression_settings(&self) -> EngineResult<CompressionSettings>;
    async fn set_compression_settings(&self, settings: CompressionSettings) -> EngineResult<()>;

    /// Stored loop-detection overrides. Unset fields keep their default, so an
    /// empty value is "use the defaults" rather than "disable everything".
    async fn get_loop_settings(&self) -> EngineResult<crate::circuit::LoopSettings>;
    async fn set_loop_settings(&self, settings: crate::circuit::LoopSettings) -> EngineResult<()>;
}

/// Storage contract for the spend ledger.
#[async_trait]
pub trait LedgerStore: Send + Sync {
    async fn record_spend(
        &self,
        session_id: Uuid,
        amount_cents: i64,
        reason: LedgerReason,
    ) -> EngineResult<LedgerEvent>;

    async fn total_spend(&self, session_id: Uuid) -> EngineResult<i64>;

    /// Total spend in a calendar day (UTC).
    async fn total_spend_day(&self, date: chrono::NaiveDate) -> EngineResult<i64>;

    /// Total spend in a calendar month (UTC).
    async fn total_spend_month(&self, year: i32, month: u32) -> EngineResult<i64>;

    /// Total spend for a project or workspace across all sessions.
    async fn total_spend_project(&self, project: &str) -> EngineResult<i64>;

    /// Total spend across all sessions.
    async fn total_spend_all(&self) -> EngineResult<i64>;
}

/// Resolve the effective budget for a session/task/day/month combination.
pub struct BudgetResolver;

impl BudgetResolver {
    pub async fn resolve_session_budget<B: BudgetStore>(
        budget_store: &B,
        session: &Session,
        project: Option<&str>,
    ) -> EngineResult<i64> {
        // 1. Session-scoped budget wins if set (specific session override).
        if let Some(policy) = budget_store
            .get_budget(BudgetScope::Session, &session.id.to_string())
            .await?
        {
            return Ok(policy.budget_cents);
        }

        // 2. General session budget (empty key = applies to all sessions
        //    that don't have their own specific override).
        if let Some(policy) = budget_store.get_budget(BudgetScope::Session, "").await? {
            return Ok(policy.budget_cents);
        }

        // 3. Project-scoped budget.
        if let Some(project_name) = project {
            if let Some(policy) = budget_store
                .get_budget(BudgetScope::Project, project_name)
                .await?
            {
                return Ok(policy.budget_cents);
            }
        }

        // 4. Day-scoped budget.
        let today = Utc::now().date_naive();
        if let Some(policy) = budget_store
            .get_budget(BudgetScope::Day, &today.to_string())
            .await?
        {
            return Ok(policy.budget_cents);
        }

        // 5. Month-scoped budget.
        let month_key = format!("{}-{}", today.year(), today.month());
        if let Some(policy) = budget_store
            .get_budget(BudgetScope::Month, &month_key)
            .await?
        {
            return Ok(policy.budget_cents);
        }

        // 6. Fallback to the session's own approved budget.
        Ok(session.budget_cents)
    }
}

/// Progressive budget degradation logic.
#[derive(Clone, Debug)]
pub struct ProgressiveBudget {
    budget_cents: i64,
}

impl ProgressiveBudget {
    pub fn new(budget_cents: i64) -> EngineResult<Self> {
        if budget_cents < 0 {
            return Err(EngineError::InvalidBudget(
                "budget cannot be negative".to_string(),
            ));
        }
        Ok(Self { budget_cents })
    }

    pub fn current_tier_nano_usd(&self, spend: i64) -> BudgetTier {
        if self.budget_cents == 0 {
            return BudgetTier::Tier1;
        }
        let budget = i128::from(self.budget_cents) * i128::from(crate::pricing::NANO_USD_PER_CENT);
        let spent = i128::from(spend) * 100;
        if spent >= budget * 100 {
            BudgetTier::Tier4
        } else if spent >= budget * 90 {
            BudgetTier::Tier3
        } else if spent >= budget * 70 {
            BudgetTier::Tier2
        } else {
            BudgetTier::Tier1
        }
    }

    pub fn current_tier(&self, spend_cents: i64) -> BudgetTier {
        let ratio = if self.budget_cents == 0 {
            0.0
        } else {
            (spend_cents as f64) / (self.budget_cents as f64)
        };
        BudgetTier::from_ratio(ratio)
    }

    /// Resolve the tier a request would enter using the conservative high end
    /// of its pre-flight estimate. This allows degradation before the call is
    /// billed instead of one request too late.
    pub fn projected_tier(&self, spend_cents: i64, predicted_high_cents: i64) -> BudgetTier {
        self.current_tier(spend_cents.saturating_add(predicted_high_cents.max(0)))
    }

    /// Apply a spend and return the resulting budget tier. If the tier
    /// crosses into Tier2 (or higher) and compaction is enabled globally,
    /// the session's context is compacted automatically.
    pub async fn apply_spend<E: EventStore, S: SettingsStore>(
        &self,
        session: &mut Session,
        event_store: &E,
        ledger: &dyn LedgerStore,
        settings_store: &S,
        amount_cents: i64,
        reason: LedgerReason,
    ) -> EngineResult<BudgetTier> {
        if amount_cents < 0 {
            return Err(EngineError::InvalidBudget(
                "spend amount cannot be negative".to_string(),
            ));
        }

        let previous_tier = session.tier;
        ledger
            .record_spend(session.id, amount_cents, reason)
            .await?;
        let current = session.spend_nano_usd.unwrap_or_else(|| {
            session
                .spend_cents
                .saturating_mul(crate::pricing::NANO_USD_PER_CENT)
        });
        session.spend_nano_usd = Some(
            current.saturating_add(amount_cents.saturating_mul(crate::pricing::NANO_USD_PER_CENT)),
        );
        session.spend_cents = session.spend_nano_usd.unwrap() / crate::pricing::NANO_USD_PER_CENT;
        let lineage_spend = session.lineage_spend_cents();
        let new_tier = self.current_tier_nano_usd(session.lineage_spend_nano_usd());
        let new_tier_u8 = new_tier.as_u8();
        session.tier = new_tier_u8;
        session.updated_at = Utc::now();

        if new_tier_u8 != previous_tier {
            let payload = TierChangePayload {
                previous_tier,
                new_tier: new_tier_u8,
                spend_cents: lineage_spend,
                budget_cents: self.budget_cents,
            };
            event_store
                .append_event(
                    session.id,
                    SessionEventType::TierChange,
                    serde_json::to_value(payload)?,
                    None,
                )
                .await?;

            // Trigger automatic context compaction when we enter Tier2
            // and the user has configured a compaction mode.
            if new_tier >= BudgetTier::Tier2 {
                if let Ok(settings) = settings_store.get_compression_settings().await {
                    if let Some(mode) = settings.mode {
                        let compressor = ContextCompressor::new();
                        let _ = compressor
                            .compress_session(event_store, session.id, mode)
                            .await;
                    }
                }
            }
        }

        Ok(new_tier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subcent_budget_thresholds_are_exact() {
        let budget = ProgressiveBudget::new(1).unwrap();
        assert_eq!(budget.current_tier_nano_usd(6_999_999).as_u8(), 1);
        assert_eq!(budget.current_tier_nano_usd(7_000_000).as_u8(), 2);
        assert_eq!(budget.current_tier_nano_usd(8_999_999).as_u8(), 2);
        assert_eq!(budget.current_tier_nano_usd(9_000_000).as_u8(), 3);
        assert_eq!(budget.current_tier_nano_usd(10_000_000).as_u8(), 4);
    }
    #[test]
    fn budget_tier_from_ratio() {
        assert!(matches!(BudgetTier::from_ratio(0.0), BudgetTier::Tier1));
        assert!(matches!(BudgetTier::from_ratio(0.69), BudgetTier::Tier1));
        assert!(matches!(BudgetTier::from_ratio(0.75), BudgetTier::Tier2));
        assert!(matches!(BudgetTier::from_ratio(0.95), BudgetTier::Tier3));
        assert!(matches!(BudgetTier::from_ratio(1.05), BudgetTier::Tier4));
    }

    #[test]
    fn progressive_budget_tiers() {
        let budget = ProgressiveBudget::new(1000).unwrap();
        assert_eq!(budget.current_tier(0).as_u8(), 1);
        assert_eq!(budget.current_tier(699).as_u8(), 1);
        assert_eq!(budget.current_tier(700).as_u8(), 2);
        assert_eq!(budget.current_tier(899).as_u8(), 2);
        assert_eq!(budget.current_tier(900).as_u8(), 3);
        assert_eq!(budget.current_tier(1000).as_u8(), 4);
    }

    #[test]
    fn projected_tier_uses_predicted_spend_and_caps_models() {
        let budget = ProgressiveBudget::new(100).unwrap();
        let tier = budget.projected_tier(65, 10);
        assert_eq!(tier, BudgetTier::Tier2);
        assert_eq!(tier.cap_model_tier(ModelTier::Ultra), Some(ModelTier::Pro));
        assert_eq!(
            BudgetTier::Tier3.cap_model_tier(ModelTier::Pro),
            Some(ModelTier::Lite)
        );
        assert_eq!(BudgetTier::Tier4.cap_model_tier(ModelTier::Lite), None);
    }
}
