//! Provider model capabilities backed by a versioned JSON catalog.
//! The local engine loads and seeds the catalog at startup. Operators can add or
//! replace records using REINCORE_MODEL_CATALOG without rebuilding the engine.
//! Rank/reasoning/speed labels are curated routing judgments, not benchmarks.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use uuid::Uuid;

/// Supported data modalities for a model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
}

/// Reasoning capability level for a model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Reasoning {
    None,
    Average,
    Moderate,
    High,
    Higher,
    Highest,
}

impl Reasoning {
    pub fn as_str(&self) -> &'static str {
        match self {
            Reasoning::None => "none",
            Reasoning::Average => "average",
            Reasoning::Moderate => "moderate",
            Reasoning::High => "high",
            Reasoning::Higher => "higher",
            Reasoning::Highest => "highest",
        }
    }
}

impl std::fmt::Display for Reasoning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Speed label for a model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Speed {
    Slowest,
    Slow,
    Medium,
    Fast,
    Fastest,
}

impl Speed {
    pub fn as_str(&self) -> &'static str {
        match self {
            Speed::Slowest => "slowest",
            Speed::Slow => "slow",
            Speed::Medium => "medium",
            Speed::Fast => "fast",
            Speed::Fastest => "fastest",
        }
    }
}

impl std::fmt::Display for Speed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A curated provider-model record.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderModelRecord {
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,
    pub provider: String,
    pub model_name: String,
    pub model_id: String,
    /// Input price in USD per 1M tokens.
    pub input_price: f64,
    /// Output price in USD per 1M tokens.
    pub output_price: f64,
    /// Rank within the provider. Higher rank means newer/preferred.
    pub rank: i64,
    pub max_input_tokens: i64,
    pub max_output_tokens: i64,
    pub reasoning: Reasoning,
    pub speed: Speed,
    pub supported_input_data: Vec<Modality>,
    pub supported_output_data: Vec<Modality>,
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
    #[serde(default = "Utc::now")]
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub pricing: Option<crate::model_catalog::PricingSchedule>,
    #[serde(default = "default_auto_select")]
    pub auto_select: bool,
    #[serde(default)]
    pub notes: String,
}

fn default_auto_select() -> bool {
    true
}

impl ProviderModelRecord {
    /// Current standard short-context prices for preflight/routing, in USD/M tokens.
    /// Legacy scalar prices are a preflight heuristic only when no schedule applies.
    pub fn current_text_prices(&self) -> (f64, f64) {
        self.pricing
            .as_ref()
            .and_then(|p| p.rates_at(0, Utc::now()))
            .map(|r| {
                (
                    r.input as f64 / r.rate_scale as f64 / 1000.0,
                    r.output as f64 / r.rate_scale as f64 / 1000.0,
                )
            })
            .unwrap_or((self.input_price, self.output_price))
    }

    pub fn matches(&self, provider: &str, model: &str) -> bool {
        self.provider.eq_ignore_ascii_case(provider)
            && (self.model_id == model || self.aliases.iter().any(|a| a == model))
    }
    // Catalog entries are intentionally table-shaped; keeping construction in
    // one call makes provider records auditable against upstream pricing pages.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider: impl Into<String>,
        model_name: impl Into<String>,
        model_id: impl Into<String>,
        input_price: f64,
        output_price: f64,
        rank: i64,
        max_input_tokens: i64,
        max_output_tokens: i64,
        reasoning: Reasoning,
        speed: Speed,
        supported_input_data: Vec<Modality>,
        supported_output_data: Vec<Modality>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            aliases: Vec::new(),
            pricing: None,
            auto_select: true,
            notes: String::new(),
            provider: provider.into(),
            model_name: model_name.into(),
            model_id: model_id.into(),
            input_price,
            output_price,
            rank,
            max_input_tokens,
            max_output_tokens,
            reasoning,
            speed,
            supported_input_data,
            supported_output_data,
            created_at: now,
            updated_at: now,
        }
    }
}

/// Return the full curated model catalog.
///
/// The catalog is built once per process. Previously every routing and cost
/// prediction call rebuilt all records, timestamps, UUIDs, strings, and
/// modality vectors.
pub fn model_catalog() -> &'static [ProviderModelRecord] {
    static CATALOG: OnceLock<Vec<ProviderModelRecord>> = OnceLock::new();
    CATALOG.get_or_init(|| {
        let path = std::env::var_os("REINCORE_MODEL_CATALOG");
        crate::model_catalog::load_catalog(path.as_deref().map(std::path::Path::new))
            .unwrap_or_else(|e| panic!("invalid provider model catalog: {e}"))
    })
}

/// Look up a model record by provider and model id.
pub fn find_model(provider: &str, model_id: &str) -> Option<ProviderModelRecord> {
    model_catalog()
        .iter()
        .find(|m| m.matches(provider, model_id))
        .cloned()
}

/// Derive a concrete model for `tier` from a configured standard model.
///
/// Automatic selection is deliberately conservative. It only considers
/// models exposed through the same provider adapter as the standard model.
/// Only `Lite` is automatically degraded: candidates must be no more
/// expensive, highly ranked, or reasoning-capable than the standard model,
/// may not be slower, and the cheapest eligible candidate wins. `Pro` and `Ultra` keep the
/// standard model as the user-selected quality ceiling unless an explicit
/// tier override is configured by the caller. A Lite candidate must also
/// preserve every input/output modality supported by the standard model. If
/// the catalog cannot prove a safe relationship, the standard model is
/// returned unchanged.
pub fn derive_model_from_standard(
    provider: &str,
    standard_model: &str,
    tier: crate::models::ModelTier,
) -> Option<String> {
    use crate::models::ModelTier;

    let standard = model_catalog()
        .iter()
        .find(|model| model.matches(provider, standard_model))?;

    if tier != ModelTier::Lite {
        return Some(standard.model_id.clone());
    }

    let candidates = model_catalog().iter().filter(|candidate| {
        candidate.auto_select
            && candidate.provider.eq_ignore_ascii_case(provider)
            && candidate.model_id != standard.model_id
            && preserves_modalities(candidate, standard)
    });

    let selected = candidates
        .filter(|candidate| {
            candidate.current_text_prices().0 <= standard.current_text_prices().0
                && candidate.current_text_prices().1 <= standard.current_text_prices().1
                && candidate.rank <= standard.rank
                && reasoning_score(&candidate.reasoning) <= reasoning_score(&standard.reasoning)
                && speed_score(&candidate.speed) >= speed_score(&standard.speed)
        })
        .min_by(|left, right| {
            total_price(left)
                .total_cmp(&total_price(right))
                .then_with(|| speed_score(&right.speed).cmp(&speed_score(&left.speed)))
                .then_with(|| right.rank.cmp(&left.rank))
        });

    Some(selected.unwrap_or(standard).model_id.clone())
}

fn reasoning_score(reasoning: &Reasoning) -> u8 {
    match reasoning {
        Reasoning::None => 0,
        Reasoning::Average => 1,
        Reasoning::Moderate => 2,
        Reasoning::High => 3,
        Reasoning::Higher => 4,
        Reasoning::Highest => 5,
    }
}

fn speed_score(speed: &Speed) -> u8 {
    match speed {
        Speed::Slowest => 0,
        Speed::Slow => 1,
        Speed::Medium => 2,
        Speed::Fast => 3,
        Speed::Fastest => 4,
    }
}

fn total_price(model: &ProviderModelRecord) -> f64 {
    {
        let (input, output) = model.current_text_prices();
        input + output
    }
}

fn preserves_modalities(candidate: &ProviderModelRecord, standard: &ProviderModelRecord) -> bool {
    standard
        .supported_input_data
        .iter()
        .all(|modality| candidate.supported_input_data.contains(modality))
        && standard
            .supported_output_data
            .iter()
            .all(|modality| candidate.supported_output_data.contains(modality))
}

#[cfg(test)]
mod hybrid_policy_tests {
    use super::*;
    use crate::models::ModelTier;

    #[test]
    fn lite_derivation_stays_within_provider_and_anchor_ceiling() {
        let standard = "gpt-5.4-2026-03-05";
        let lite = derive_model_from_standard("openai", standard, ModelTier::Lite).unwrap();
        let standard_record = find_model("openai", standard).unwrap();
        let lite_record = find_model("openai", &lite).unwrap();

        assert_eq!(lite_record.provider, standard_record.provider);
        assert!(lite_record.input_price <= standard_record.input_price);
        assert!(lite_record.output_price <= standard_record.output_price);
        assert!(lite_record.rank <= standard_record.rank);
        assert!(
            reasoning_score(&lite_record.reasoning) <= reasoning_score(&standard_record.reasoning)
        );
        assert!(speed_score(&lite_record.speed) >= speed_score(&standard_record.speed));
    }

    #[test]
    fn pro_and_ultra_do_not_upgrade_above_standard() {
        let standard = "gpt-5.4-2026-03-05";

        assert_eq!(
            derive_model_from_standard("openai", standard, ModelTier::Pro).as_deref(),
            Some(standard)
        );
        assert_eq!(
            derive_model_from_standard("openai", standard, ModelTier::Ultra).as_deref(),
            Some(standard)
        );
    }

    #[test]
    fn unknown_catalog_anchor_is_not_guessed() {
        assert_eq!(
            derive_model_from_standard("custom", "frontier-v2", ModelTier::Lite),
            None
        );
    }

    #[test]
    fn provider_wide_derivation_supports_cross_generation_degradation() {
        assert_eq!(
            derive_model_from_standard("anthropic", "claude-sonnet-5", ModelTier::Lite).as_deref(),
            Some("claude-haiku-4-5")
        );
    }
}

/// Return the cheapest model id for a given provider and tier intent.
///
/// Tier intent is interpreted from the catalog: models with `reasoning=true`
/// and high rank are preferred for Ultra/Pro; fast, low-price models are
/// preferred for Lite.  When no exact match exists, falls back to the
/// highest-ranked model for the provider.
pub fn default_model_for_provider_and_tier(provider: &str, tier: &str) -> Option<String> {
    let models: Vec<&ProviderModelRecord> = model_catalog()
        .iter()
        .filter(|m| m.auto_select && m.provider.eq_ignore_ascii_case(provider))
        .collect();
    if models.is_empty() {
        return None;
    }

    let tier_lower = tier.to_lowercase();
    let reasoning_score = |m: &ProviderModelRecord| match m.reasoning {
        Reasoning::None => 0,
        Reasoning::Average => 1,
        Reasoning::Moderate => 2,
        Reasoning::High => 3,
        Reasoning::Higher => 4,
        Reasoning::Highest => 5,
    };
    let speed_score = |m: &ProviderModelRecord| match m.speed {
        Speed::Slowest => 0,
        Speed::Slow => 1,
        Speed::Medium => 2,
        Speed::Fast => 3,
        Speed::Fastest => 4,
    };

    let preferred = match tier_lower.as_str() {
        "lite" => models
            .iter()
            .filter(|m| speed_score(m) >= 3)
            .min_by(|a, b| {
                a.current_text_prices()
                    .0
                    .partial_cmp(&b.current_text_prices().0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.rank.cmp(&b.rank).reverse())
            }),
        "pro" => models
            .iter()
            .filter(|m| reasoning_score(m) >= 3)
            .max_by(|a, b| {
                a.rank.cmp(&b.rank).then_with(|| {
                    a.current_text_prices()
                        .0
                        .partial_cmp(&b.current_text_prices().0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            }),
        "ultra" => models
            .iter()
            .filter(|m| reasoning_score(m) >= 4)
            .max_by(|a, b| {
                a.rank.cmp(&b.rank).then_with(|| {
                    a.current_text_prices()
                        .0
                        .partial_cmp(&b.current_text_prices().0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            }),
        _ => None,
    };

    preferred
        .or_else(|| models.iter().max_by_key(|m| m.rank))
        .map(|m| m.model_id.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_unique_ids() {
        let catalog = model_catalog();
        let mut ids = std::collections::HashSet::new();
        for m in catalog {
            assert!(ids.insert(m.id), "duplicate model id in catalog: {}", m.id);
        }
    }

    #[test]
    fn catalog_has_unique_provider_model_id_pairs() {
        let catalog = model_catalog();
        let mut keys = std::collections::HashSet::new();
        for m in catalog {
            let key = format!("{}:{}", m.provider, m.model_id);
            assert!(
                keys.insert(key.clone()),
                "duplicate provider/model_id: {}",
                key
            );
        }
    }
}
