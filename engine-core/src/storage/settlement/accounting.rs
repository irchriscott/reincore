//! Durable completion accounting. One immutable ID per provider invocation;
//! retry settlement with the same record, never a newly generated ID.
use crate::{EngineError, EngineResult, Session};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CompletionAccounting {
    pub id: Uuid,
    pub session_id: Uuid,
    pub recorded_at: chrono::DateTime<chrono::Utc>,
    pub payload: serde_json::Value,
    pub cost_cents: i64,
    #[serde(default)]
    pub cost: Option<crate::pricing::CostEstimate>,
    #[serde(default)]
    pub purpose: Option<String>,
}

impl CompletionAccounting {
    pub fn auxiliary(
        session_id: Uuid,
        provider: &dyn crate::Provider,
        model: &str,
        usage: Option<&crate::TokenUsage>,
        purpose: &str,
    ) -> Self {
        let cost = usage
            .map(|u| {
                crate::pricing::estimate(provider.label(), model, u, provider.is_free(), false)
            })
            .unwrap_or_else(|| {
                crate::pricing::CostEstimate::unknown(
                    provider.label(),
                    model,
                    "provider did not return usage",
                )
            });
        let counts = usage.cloned().unwrap_or_default();
        Self {
            id: Uuid::new_v4(),
            session_id,
            recorded_at: chrono::Utc::now(),
            cost_cents: cost.display_cents(),
            cost: Some(cost),
            purpose: Some(purpose.into()),
            payload: serde_json::json!({
                "model": model,
                "input_tokens": counts.prompt_tokens,
                "output_tokens": counts.completion_tokens,
                "usage": usage,
            }),
        }
    }

    pub fn amount_nano_usd(&self) -> Option<i64> {
        match &self.cost {
            Some(cost) => cost.amount_nano_usd,
            None => self
                .cost_cents
                .checked_mul(crate::pricing::NANO_USD_PER_CENT),
        }
    }

    pub fn validate(&self) -> EngineResult<()> {
        if self.amount_nano_usd().is_some_and(|n| n < 0)
            || self.cost_cents < 0
            || self.id.is_nil()
            || !self.payload.is_object()
        {
            return Err(EngineError::InvalidInput(
                "invalid accounting record".into(),
            ));
        }

        if let Some(cost) = &self.cost {
            if (cost.status == crate::pricing::CostStatus::Unknown)
                != cost.amount_nano_usd.is_none()
                || cost.display_cents() != self.cost_cents
                || cost.currency != "USD"
                || cost.usage.as_ref().is_some_and(|u| {
                    self.payload["input_tokens"].as_u64() != Some(u.prompt_tokens)
                        || self.payload["output_tokens"].as_u64() != Some(u.completion_tokens)
                })
            {
                return Err(EngineError::InvalidInput(
                    "inconsistent cost estimate".into(),
                ));
            }
        } else if self.amount_nano_usd().is_none() {
            return Err(EngineError::InvalidInput("legacy spend overflow".into()));
        }

        for field in ["input_tokens", "output_tokens"] {
            if self
                .payload
                .get(field)
                .and_then(|v| v.as_u64())
                .map_or(true, |v| v > i64::MAX as u64)
            {
                return Err(EngineError::InvalidInput(
                    "invalid accounting token count".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn apply(&self, session: &mut Session) -> EngineResult<()> {
        self.validate()?;
        let amount = self.amount_nano_usd();
        let previous = match session.spend_nano_usd {
            Some(amount) => amount,
            None => session
                .spend_cents
                .checked_mul(crate::pricing::NANO_USD_PER_CENT)
                .ok_or_else(|| EngineError::InvalidInput("legacy spend overflow".into()))?,
        };

        session.spend_nano_usd = Some(
            previous
                .checked_add(amount.unwrap_or(0))
                .ok_or_else(|| EngineError::InvalidInput("spend overflow".into()))?,
        );

        session.spend_cents = session.spend_nano_usd.unwrap() / crate::pricing::NANO_USD_PER_CENT;
        if amount.is_none() {
            session.unknown_cost_calls = session.unknown_cost_calls.saturating_add(1);
        }

        let input = self.payload["input_tokens"].as_u64().unwrap();
        let output = self.payload["output_tokens"].as_u64().unwrap();
        session.total_input_tokens = session
            .total_input_tokens
            .checked_add(input)
            .filter(|v| *v <= i64::MAX as u64)
            .ok_or_else(|| EngineError::InvalidInput("token overflow".into()))?;

        session.total_output_tokens = session
            .total_output_tokens
            .checked_add(output)
            .filter(|v| *v <= i64::MAX as u64)
            .ok_or_else(|| EngineError::InvalidInput("token overflow".into()))?;
        if self.purpose.is_none() {
            session.context_tokens = input;
        }

        if let Some(high) = self
            .payload
            .get("predicted_cost_high_cents")
            .and_then(|v| v.as_i64())
        {
            session.predicted_high_cents = Some(high);
        }

        fn mean(value: Option<i64>, samples: Option<i64>, next: i64) -> (Option<i64>, Option<i64>) {
            let n = samples.unwrap_or(0).max(0);
            (
                Some(
                    ((value.unwrap_or(0) as f64 * n as f64 + next as f64) / (n + 1) as f64).round()
                        as i64,
                ),
                Some(n + 1),
            )
        }

        if let Some(value) = self
            .payload
            .get("actual_to_predicted_high_ratio_bps")
            .or_else(|| self.payload.get("cost_accuracy_bps"))
            .and_then(|v| v.as_i64())
        {
            (session.cost_accuracy_bps, session.cost_accuracy_samples) = mean(
                session.cost_accuracy_bps,
                session.cost_accuracy_samples,
                value,
            );
        }

        if let Some(value) = self
            .payload
            .get("cache_hit_rate_bps")
            .and_then(|v| v.as_i64())
        {
            (session.cache_hit_rate_bps, session.cache_hit_rate_samples) = mean(
                session.cache_hit_rate_bps,
                session.cache_hit_rate_samples,
                value,
            );
        }
        session.tier = crate::ProgressiveBudget::new(session.budget_cents)?
            .current_tier_nano_usd(session.lineage_spend_nano_usd())
            .as_u8();
        session.updated_at = chrono::Utc::now();
        Ok(())
    }
}

/// A durable journal entry must be committed before settlement. Settlement is
/// atomic and idempotent: event/projections, ledger, totals, and journal state.
#[async_trait::async_trait]
pub trait AccountingStore: Send + Sync {
    async fn stage_accounting(&self, record: &CompletionAccounting) -> EngineResult<()>;

    async fn settle_accounting(&self, id: Uuid) -> EngineResult<()>;

    async fn recover_accounting(&self) -> EngineResult<()>;

    async fn record_accounting(&self, record: &CompletionAccounting) -> EngineResult<()> {
        self.stage_accounting(record).await?;
        self.settle_accounting(record.id).await
    }
}
