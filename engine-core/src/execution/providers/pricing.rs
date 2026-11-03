//! Fixed-point USD estimates. Catalog-derived costs are estimates, never invoices.
use crate::provider::TokenUsage;
use serde::{Deserialize, Serialize};
pub const NANO_USD_PER_CENT: i64 = 10_000_000;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CostStatus {
    Estimated,
    Unknown,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TokenRates {
    /// Rate values are nano-USD per token multiplied by this positive scale.
    /// Old receipts omit it and retain scale 1; catalog uses 1000 (pico-USD).
    #[serde(default = "unit_scale")]
    pub rate_scale: i64,
    /// Ordinary input and output rates in scaled units.
    pub input: i64,
    pub output: i64,
    pub cache_read: Option<i64>,
    pub cache_write: Option<i64>,
    pub cache_write_1h: Option<i64>,
    /// Separate batch rates: discounts are not inferred from request eligibility.
    pub batch_input: Option<i64>,
    pub batch_output: Option<i64>,
    #[serde(default)]
    pub batch_cache_read: Option<i64>,
    #[serde(default)]
    pub batch_cache_write: Option<i64>,
    #[serde(default)]
    pub batch_cache_write_1h: Option<i64>,
}
fn unit_scale() -> i64 {
    1
}
impl Default for TokenRates {
    fn default() -> Self {
        Self {
            rate_scale: 1,
            input: 0,
            output: 0,
            cache_read: None,
            cache_write: None,
            cache_write_1h: None,
            batch_input: None,
            batch_output: None,
            batch_cache_read: None,
            batch_cache_write: None,
            batch_cache_write_1h: None,
        }
    }
}
impl TokenRates {
    pub fn is_valid(&self) -> bool {
        self.rate_scale > 0
            && self.input >= 0
            && self.output >= 0
            && [
                self.cache_read,
                self.cache_write,
                self.cache_write_1h,
                self.batch_input,
                self.batch_output,
                self.batch_cache_read,
                self.batch_cache_write,
                self.batch_cache_write_1h,
            ]
            .iter()
            .all(|r| r.is_none_or(|n| n >= 0))
            && self.batch_input.is_some() == self.batch_output.is_some()
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CostEstimate {
    pub amount_nano_usd: Option<i64>,
    #[serde(default)]
    pub batch_submitted: bool,
    #[serde(default)]
    pub usage: Option<TokenUsage>,
    pub status: CostStatus,
    pub provider: String,
    pub model: String,
    pub currency: String,
    pub pricing_version: String,
    pub rates: Option<TokenRates>,
    pub reason: Option<String>,
}
impl CostEstimate {
    pub fn unknown(provider: &str, model: &str, reason: &str) -> Self {
        Self {
            usage: None,
            amount_nano_usd: None,
            batch_submitted: false,
            status: CostStatus::Unknown,
            provider: provider.into(),
            model: model.into(),
            currency: "USD".into(),
            pricing_version: "unavailable".into(),
            rates: None,
            reason: Some(reason.into()),
        }
    }
    pub fn cents(&self) -> Option<f64> {
        self.amount_nano_usd
            .map(|n| n as f64 / NANO_USD_PER_CENT as f64)
    }
    pub fn accuracy_bps(&self, predicted_cents: i64) -> Option<i64> {
        self.amount_nano_usd.and_then(|n| {
            predicted_cents
                .checked_mul(NANO_USD_PER_CENT)
                .and_then(|p| crate::cost_accuracy_bps(n, p))
        })
    }
    /// Compatibility display field only. Never sum these rounded values.
    pub fn display_cents(&self) -> i64 {
        self.amount_nano_usd.unwrap_or(0) / NANO_USD_PER_CENT
    }
}

pub fn calculate(
    provider: &str,
    model: &str,
    version: &str,
    rates: TokenRates,
    usage: &TokenUsage,
    batch: bool,
) -> CostEstimate {
    let unknown = |why| {
        let mut cost = CostEstimate::unknown(provider, model, why);
        cost.usage = Some(*usage);
        cost.batch_submitted = batch;
        cost
    };
    if !rates.is_valid() {
        return unknown("invalid pricing rates or scale");
    }
    let Some(ordinary) = usage
        .prompt_tokens
        .checked_sub(usage.cached_prompt_tokens)
        .and_then(|n| n.checked_sub(usage.cache_creation_input_tokens))
    else {
        return unknown("inconsistent input/cache token counts");
    };
    let Some(short_writes) = usage
        .cache_creation_input_tokens
        .checked_sub(usage.cache_creation_1h_tokens)
    else {
        return unknown("inconsistent cache write durations");
    };
    if usage.reasoning_tokens > usage.completion_tokens {
        return unknown("reasoning exceeds total output tokens");
    }
    let (input, output) = if batch {
        match (rates.batch_input, rates.batch_output) {
            (Some(a), Some(b)) => (a, b),
            _ => return unknown("batch pricing not configured"),
        }
    } else {
        (rates.input, rates.output)
    };
    let components = [
        (ordinary, Some(input)),
        (usage.completion_tokens, Some(output)),
        (
            usage.cached_prompt_tokens,
            if batch {
                rates.batch_cache_read
            } else {
                rates.cache_read
            },
        ),
        (
            short_writes,
            if batch {
                rates.batch_cache_write
            } else {
                rates.cache_write
            },
        ),
        (
            usage.cache_creation_1h_tokens,
            if batch {
                rates.batch_cache_write_1h
            } else {
                rates.cache_write_1h
            },
        ),
    ];
    let mut total = 0i128;
    for (tokens, rate) in components {
        if tokens == 0 {
            continue;
        }
        let Some(rate) = rate.filter(|v| *v >= 0) else {
            return unknown("rate unavailable for reported token category");
        };
        let Some(next) = total.checked_add(i128::from(tokens) * i128::from(rate)) else {
            return unknown("cost overflow");
        };
        total = next;
    }
    // The only rounding is at the ledger's one nano-dollar precision boundary.
    let total = total / i128::from(rates.rate_scale)
        + i128::from(total % i128::from(rates.rate_scale) != 0);
    let Ok(total) = i64::try_from(total) else {
        return unknown("cost overflow");
    };
    CostEstimate {
        usage: Some(usage.clone()),
        amount_nano_usd: Some(total),
        batch_submitted: batch,
        status: CostStatus::Estimated,
        provider: provider.into(),
        model: model.into(),
        currency: "USD".into(),
        pricing_version: version.into(),
        rates: Some(rates),
        reason: None,
    }
}

pub fn estimate(
    provider: &str,
    model: &str,
    usage: &TokenUsage,
    free: bool,
    batch: bool,
) -> CostEstimate {
    if free {
        return calculate(
            provider,
            model,
            "explicit-free-provider-v1",
            TokenRates {
                input: 0,
                output: 0,
                cache_read: Some(0),
                cache_write: Some(0),
                cache_write_1h: Some(0),
                batch_input: Some(0),
                batch_output: Some(0),
                batch_cache_read: Some(0),
                batch_cache_write: Some(0),
                batch_cache_write_1h: Some(0),
                ..Default::default()
            },
            usage,
            batch,
        );
    }
    estimate_from_catalog(
        crate::model_library::model_catalog(),
        provider,
        model,
        usage,
        batch,
        chrono::Utc::now(),
    )
}

/// Explicit time/catalog inputs support reproducible historical estimates and fixtures.
/// Persisted receipts keep the selected rates and fingerprint; replay never reprices them.
pub fn estimate_from_catalog(
    catalog: &[crate::model_library::ProviderModelRecord],
    provider: &str,
    model: &str,
    usage: &TokenUsage,
    batch: bool,
    at: chrono::DateTime<chrono::Utc>,
) -> CostEstimate {
    let unknown = |why| {
        let mut cost = CostEstimate::unknown(provider, model, why);
        cost.usage = Some(*usage);
        cost.batch_submitted = batch;
        cost
    };
    let Some(record) = catalog.iter().find(|r| r.matches(provider, model)) else {
        return unknown("model absent from pricing catalog; no tier fallback");
    };
    let Some(schedule) = &record.pricing else {
        return unknown("model has no published/configured pricing");
    };
    let Some(rates) = schedule.rates_at(usage.prompt_tokens, at) else {
        return unknown("no pricing for this date, UTC time or input context length");
    };
    let version = crate::CacheKey::for_tool_call(
        "catalog-text-pricing-v2",
        &serde_json::json!({
            "provider": record.provider,
            "model": record.model_id,
            "schedule": schedule,
            "rates": rates
        }),
    )
    .expect("serializable pricing schedule")
    .0;
    let mut cost = calculate(provider, model, &version, rates.clone(), usage, batch);
    if cost.status == CostStatus::Estimated {
        cost.reason =
            Some("catalog estimate for standard text requests; not externally reconciled".into());
    }
    cost
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rates() -> TokenRates {
        TokenRates {
            input: 1000,
            output: 2000,
            cache_read: Some(100),
            cache_write: Some(1250),
            cache_write_1h: Some(2000),
            batch_input: Some(500),
            batch_output: Some(1000),
            ..Default::default()
        }
    }
    #[test]
    fn tiny_calls_accumulate_without_per_call_rounding() {
        let usage = TokenUsage {
            prompt_tokens: 100,
            ..Default::default()
        };
        let estimate = calculate("fixture", "text", "v1", rates(), &usage, false);
        assert_eq!(estimate.amount_nano_usd, Some(100_000));
        assert_eq!(estimate.cents(), Some(0.01));
        assert_eq!(estimate.amount_nano_usd.unwrap() * 100, NANO_USD_PER_CENT);
        assert_eq!(estimate.accuracy_bps(1), Some(100));
        assert_eq!(estimate.status, CostStatus::Estimated);
    }
    #[test]
    fn mixed_cache_writes_reads_and_reasoning_follow_billing_formula() {
        let usage = TokenUsage {
            prompt_tokens: 1000,
            cached_prompt_tokens: 600,
            cache_creation_input_tokens: 200,
            cache_creation_1h_tokens: 80,
            completion_tokens: 100,
            reasoning_tokens: 60,
            ..Default::default()
        };
        // 200 ordinary + 600 reads + 120 five-minute writes + 80 one-hour writes.
        // Reasoning is already included in the 100 output tokens.
        assert_eq!(
            calculate("fixture", "text", "v1", rates(), &usage, false).amount_nano_usd,
            Some(200 * 1000 + 600 * 100 + 120 * 1250 + 80 * 2000 + 100 * 2000)
        );
    }
    #[test]
    fn batch_uses_explicit_rates_only_when_submitted() {
        let usage = TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 20,
            ..Default::default()
        };
        assert_eq!(
            calculate("p", "m", "v1", rates(), &usage, false).amount_nano_usd,
            Some(140_000)
        );
        assert_eq!(
            calculate("p", "m", "v1", rates(), &usage, true).amount_nano_usd,
            Some(70_000)
        );
        let mut missing = rates();
        missing.batch_output = None;
        assert_eq!(
            calculate("p", "m", "v1", missing, &usage, true).status,
            CostStatus::Unknown
        );
    }
    #[test]
    fn unknown_is_not_free_and_missing_cache_rates_are_not_ordinary_rates() {
        let usage = TokenUsage {
            prompt_tokens: 10,
            cached_prompt_tokens: 10,
            ..Default::default()
        };
        assert_eq!(
            estimate("missing", "missing", &usage, false, false).amount_nano_usd,
            None
        );
        assert_eq!(
            estimate("local", "local", &usage, true, false).amount_nano_usd,
            Some(0)
        );
        let mut missing = rates();
        missing.cache_read = None;
        assert_eq!(
            calculate("p", "m", "v1", missing, &usage, false).amount_nano_usd,
            None
        );
    }
    #[test]
    fn inconsistent_counts_and_overflow_are_unknown() {
        for usage in [
            TokenUsage {
                prompt_tokens: 1,
                cached_prompt_tokens: 2,
                ..Default::default()
            },
            TokenUsage {
                prompt_tokens: u64::MAX,
                ..Default::default()
            },
            TokenUsage {
                completion_tokens: 1,
                reasoning_tokens: 2,
                ..Default::default()
            },
        ] {
            assert_eq!(
                calculate("p", "m", "v1", rates(), &usage, false).status,
                CostStatus::Unknown
            );
        }
    }
    #[test]
    fn catalog_version_is_stable_and_provider_specific() {
        let usage = TokenUsage {
            prompt_tokens: 100,
            ..Default::default()
        };
        let a = estimate("openai", "gpt-5.4-2026-03-05", &usage, false, false);
        assert_eq!(a.status, CostStatus::Estimated);
        assert_eq!(
            a,
            estimate("openai", "gpt-5.4-2026-03-05", &usage, false, false)
        );
        assert_eq!(
            estimate("other", "gpt-5.4-2026-03-05", &usage, false, false).status,
            CostStatus::Unknown
        );
    }
}
