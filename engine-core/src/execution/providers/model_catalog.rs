//! Versioned, operator-loadable model and pricing data for the local engine.
use crate::{model_library::ProviderModelRecord, pricing::TokenRates};
use chrono::{DateTime, Datelike, NaiveDate, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::Path};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        pricing::{estimate_from_catalog, CostStatus},
        provider::TokenUsage,
    };

    fn cost(
        provider: &str,
        model: &str,
        usage: TokenUsage,
        batch: bool,
        at: &str,
    ) -> crate::pricing::CostEstimate {
        estimate_from_catalog(
            &parse_catalog(BUILTIN_CATALOG).unwrap(),
            provider,
            model,
            &usage,
            batch,
            at.parse().unwrap(),
        )
    }
    const NOW: &str = "2026-09-05T12:00:00Z";
    const REFRESHED: &str = "2026-10-07T12:00:00Z";

    #[test]
    fn provider_case_override_keeps_one_canonical_storage_identity() {
        let mut models = parse_catalog(BUILTIN_CATALOG).unwrap();
        let original_count = models.len();
        let mut replacement = models[0].clone();
        replacement.provider = "OPENAI".into();
        replacement.notes = "operator override".into();
        let json = serde_json::to_string(&CatalogDocument {
            schema_version: 1,
            models: vec![replacement.clone()],
        })
        .unwrap();
        let parsed = parse_catalog(&json).unwrap();
        assert_eq!(parsed[0].provider, "openai");
        // Direct merges must have the same normalization as file loading.
        merge_catalog(&mut models, vec![replacement]).unwrap();
        assert_eq!(models.len(), original_count);
        assert_eq!(models[0].provider, "openai");
        assert_eq!(models[0].notes, "operator override");
    }

    #[test]
    fn invalid_provenance_and_whitespace_aliases_are_rejected() {
        let original: serde_json::Value = serde_json::from_str(BUILTIN_CATALOG).unwrap();
        for source in ["https://", "https:// bad host", "http://example.com/prices"] {
            let mut doc = original.clone();
            doc["models"][0]["pricing"]["sources"] = serde_json::json!([source]);
            assert!(parse_catalog(&doc.to_string()).is_err(), "{source}");
        }
        let mut doc = original;
        doc["models"][0]["aliases"] = serde_json::json!([" gpt-alias "]);
        assert!(parse_catalog(&doc.to_string()).is_err());
    }

    #[test]
    fn price_version_ignores_storage_identity_but_tracks_rate_changes() {
        let mut models = parse_catalog(BUILTIN_CATALOG).unwrap();
        let usage = TokenUsage {
            prompt_tokens: 100,
            ..Default::default()
        };
        let at = NOW.parse().unwrap();
        let before = estimate_from_catalog(&models, "openai", "gpt-5.6-sol", &usage, false, at);
        models[0].id = uuid::Uuid::new_v4();
        let reloaded = estimate_from_catalog(&models, "OPENAI", "gpt-5.6", &usage, false, at);
        assert_eq!(before.pricing_version, reloaded.pricing_version);
        models[0].pricing.as_mut().unwrap().periods[0].bands[0]
            .rates
            .input *= 2;
        let updated = estimate_from_catalog(&models, "openai", "gpt-5.6-sol", &usage, false, at);
        assert_ne!(before.pricing_version, updated.pricing_version);
        assert_eq!(
            updated.amount_nano_usd,
            before.amount_nano_usd.map(|n| n * 2)
        );
        assert_eq!(before.amount_nano_usd, Some(400_000));
    }

    #[test]
    fn latest_models_have_versioned_pricing_and_provenance() {
        let models = parse_catalog(BUILTIN_CATALOG).unwrap();
        for id in [
            "gpt-6-astra",
            "claude-fable-5-1",
            "claude-opus-5",
            "gemini-3.6-flash",
            "gemini-3.7-flash",
            "gemini-3.8-flash",
            "grok-4.6",
            "deepseek-v4-flash-vision-exp",
        ] {
            let model = models.iter().find(|m| m.model_id == id).unwrap();
            let schedule = model.pricing.as_ref().unwrap();
            assert!(!schedule.sources.is_empty());
            assert!(schedule.rates_at(100, NOW.parse().unwrap()).is_some());
        }
    }

    #[test]
    fn october_releases_have_current_pricing_and_selection_guards() {
        let models = parse_catalog(BUILTIN_CATALOG).unwrap();
        for (provider, id, auto_select) in [
            ("openai", "gpt-6.1-sol", false),
            ("openai", "gpt-6-sol", false),
            ("openai", "gpt-6-luna", false),
            ("openai", "gpt-5.6-cyber", false),
            ("anthropic", "claude-opus-5-5", true),
            ("anthropic", "claude-sonnet-5-5", true),
            ("gemini", "gemini-robotics-er-2-preview", false),
            ("xai", "grok-4.7", true),
            ("xai", "grok-build-0.1", true),
            ("xai", "grok-4.20-multi-agent-0309", false),
            ("deepseek", "deepseek-flash", true),
            ("groq", "meta-llama/llama-prompt-guard-2-22m", false),
            ("groq", "meta-llama/llama-prompt-guard-2-86m", false),
            ("groq", "openai/gpt-oss-safeguard-20b", false),
        ] {
            let model = models.iter().find(|m| m.matches(provider, id)).unwrap();
            let pricing = model.pricing.as_ref().unwrap();
            assert_eq!(pricing.verified_at.to_string(), "2026-10-07", "{id}");
            assert!(
                pricing.rates_at(100, REFRESHED.parse().unwrap()).is_some(),
                "{id}"
            );
            assert_eq!(model.auto_select, auto_select, "{id}");
        }
        let enterprise = models
            .iter()
            .find(|m| m.matches("groq", "minimaxai/minimax-m2.7"))
            .unwrap();
        assert!(!enterprise.auto_select);
        assert!(enterprise.pricing.is_none());
        assert_eq!(
            cost(
                "groq",
                &enterprise.model_id,
                TokenUsage::default(),
                false,
                REFRESHED
            )
            .status,
            CostStatus::Unknown
        );
        for id in [
            "deepseek-v4-flash",
            "deepseek-v4-flash-vision-exp",
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ] {
            assert!(
                !models
                    .iter()
                    .find(|m| m.model_id == id)
                    .unwrap()
                    .auto_select,
                "{id}"
            );
        }
        assert_eq!(
            cost(
                "groq",
                "openai/gpt-oss-safeguard-20b",
                TokenUsage::default(),
                true,
                REFRESHED
            )
            .status,
            CostStatus::Unknown
        );
    }

    #[test]
    fn sol_cache_rates_and_luna_fractional_batch_writes_are_explicit() {
        let usage = TokenUsage {
            prompt_tokens: 272_000,
            cached_prompt_tokens: 272_000,
            completion_tokens: 100,
            ..Default::default()
        };
        assert_eq!(
            cost("openai", "gpt-6.1-sol", usage, false, REFRESHED).amount_nano_usd,
            Some(28_200_000)
        );
        assert_eq!(
            cost("openai", "gpt-6-sol", usage, false, REFRESHED).amount_nano_usd,
            Some(55_400_000)
        );
        let long = TokenUsage {
            prompt_tokens: 272_001,
            cached_prompt_tokens: 272_001,
            ..usage
        };
        assert_eq!(
            cost("openai", "gpt-6.1-sol", long, false, REFRESHED).amount_nano_usd,
            Some(55_900_200)
        );
        assert_eq!(
            cost("openai", "gpt-6.1-sol", long, true, REFRESHED).amount_nano_usd,
            Some(27_950_100)
        );
        let writes = TokenUsage {
            prompt_tokens: 100,
            cache_creation_input_tokens: 100,
            ..Default::default()
        };
        assert_eq!(
            cost("openai", "gpt-6-luna", writes, true, REFRESHED).amount_nano_usd,
            Some(6_250)
        );
    }

    #[test]
    fn opus_55_cache_writes_reads_and_batch_discounts_stack() {
        let usage = TokenUsage {
            prompt_tokens: 1000,
            cached_prompt_tokens: 600,
            cache_creation_input_tokens: 200,
            cache_creation_1h_tokens: 80,
            completion_tokens: 100,
            ..Default::default()
        };
        let expected = 200 * 4_000 + 600 * 200 + 120 * 5_000 + 80 * 8_000 + 100 * 20_000;
        assert_eq!(
            cost("anthropic", "claude-opus-5-5", usage, false, REFRESHED).amount_nano_usd,
            Some(expected)
        );
        assert_eq!(
            cost("anthropic", "claude-opus-5-5", usage, true, REFRESHED).amount_nano_usd,
            Some(expected / 2)
        );
    }

    #[test]
    fn grok_47_long_context_boundary_and_build_alias_are_priced() {
        for (tokens, expected) in [(199_999, 399_998_000), (200_000, 800_000_000)] {
            let usage = TokenUsage {
                prompt_tokens: tokens,
                ..Default::default()
            };
            assert_eq!(
                cost("xai", "grok-4.7", usage, false, REFRESHED).amount_nano_usd,
                Some(expected)
            );
        }
        assert_eq!(
            cost("xai", "grok-4.7", TokenUsage::default(), true, REFRESHED).status,
            CostStatus::Unknown
        );
        let usage = TokenUsage {
            prompt_tokens: 1000,
            ..Default::default()
        };
        assert_eq!(
            cost("xai", "grok-code-fast-1", usage, false, REFRESHED).amount_nano_usd,
            Some(1_000_000)
        );
    }

    #[test]
    fn deepseek_refresh_preserves_history_and_uses_weekdays_and_holidays() {
        let usage = TokenUsage {
            prompt_tokens: 1000,
            ..Default::default()
        };
        for model in [
            "deepseek-flash",
            "deepseek-v4-flash",
            "deepseek-v4-flash-vision-exp",
        ] {
            for (at, expected) in [
                ("2026-10-07T02:00:00Z", 150_000), // National Day holiday, Wednesday.
                ("2026-10-08T00:59:59Z", 150_000),
                ("2026-10-08T01:00:00Z", 300_000),
                ("2026-10-08T04:00:00Z", 150_000),
                ("2026-10-08T06:00:00Z", 300_000),
                ("2026-10-08T10:00:00Z", 150_000),
                ("2026-10-09T02:00:00Z", 300_000), // Friday peak.
                ("2026-10-10T02:00:00Z", 150_000), // Saturday off-peak.
                ("2026-10-11T07:00:00Z", 150_000), // Sunday off-peak.
                ("2026-10-12T02:00:00Z", 300_000), // Monday peak.
            ] {
                assert_eq!(
                    cost("deepseek", model, usage, false, at).amount_nano_usd,
                    Some(expected),
                    "{model} at {at}"
                );
            }
            // A new holiday calendar must be verified before 2027 rates are estimated.
            assert_eq!(
                cost("deepseek", model, usage, false, "2027-01-01T00:00:00Z").status,
                CostStatus::Unknown
            );
        }
        assert_eq!(
            cost("deepseek", "deepseek-v4-flash", usage, false, NOW).amount_nano_usd,
            Some(220_000)
        );
        assert_eq!(
            cost(
                "deepseek",
                "deepseek-v4-pro",
                usage,
                false,
                "2026-10-09T02:00:00Z"
            )
            .amount_nano_usd,
            Some(1_320_000)
        );
        assert_eq!(
            cost(
                "deepseek",
                "deepseek-v4-pro",
                usage,
                false,
                "2026-10-10T02:00:00Z"
            )
            .amount_nano_usd,
            Some(660_000)
        );
        let cached = TokenUsage {
            prompt_tokens: 1000,
            cached_prompt_tokens: 1000,
            ..Default::default()
        };
        assert_eq!(
            cost(
                "deepseek",
                "deepseek-flash",
                cached,
                false,
                "2026-10-09T02:00:00Z"
            )
            .amount_nano_usd,
            Some(6_000)
        );
    }

    #[test]
    fn weekday_filters_validate_overlap_and_preserve_legacy_serialization() {
        let models = parse_catalog(BUILTIN_CATALOG).unwrap();
        let mut pricing = models[0].pricing.clone().unwrap();
        let mut weekday = pricing.periods[0].bands[0].clone();
        let serialized = serde_json::to_value(&weekday).unwrap();
        assert!(serialized.get("utc_weekdays").is_none());
        weekday.utc_weekdays = vec![0, 1, 2, 3, 4];
        let mut weekend = weekday.clone();
        weekend.utc_weekdays = vec![5, 6];
        pricing.periods[0].bands = vec![weekday, weekend];
        assert!(pricing.validate().is_ok());
        pricing.periods[0].bands[1].utc_weekdays = vec![4, 5, 6];
        assert!(pricing.validate().unwrap_err().contains("overlapping"));
        pricing.periods[0].bands.pop();
        for invalid in [vec![7], vec![0, 0]] {
            pricing.periods[0].bands[0].utc_weekdays = invalid;
            assert!(pricing.validate().is_err());
        }
        pricing.periods[0].bands[0].utc_weekdays = vec![0];
        assert!(pricing
            .rates_at(100, "2026-10-12T12:00:00Z".parse().unwrap())
            .is_some());
        assert!(pricing
            .rates_at(100, "2026-10-11T12:00:00Z".parse().unwrap())
            .is_none());
        pricing.periods[0].bands[0].utc_weekdays.clear();
        assert!(pricing
            .rates_at(100, "2026-10-11T12:00:00Z".parse().unwrap())
            .is_some());
    }

    #[test]
    fn robotics_promotion_expires_and_qwen_context_is_not_truncated() {
        let usage = TokenUsage {
            prompt_tokens: 1000,
            ..Default::default()
        };
        assert_eq!(
            cost(
                "gemini",
                "gemini-robotics-er-2-preview",
                usage,
                false,
                "2026-12-31T23:59:59Z"
            )
            .amount_nano_usd,
            Some(1_000_000)
        );
        assert_eq!(
            cost(
                "gemini",
                "gemini-robotics-er-2-preview",
                usage,
                false,
                "2027-01-01T00:00:00Z"
            )
            .amount_nano_usd,
            Some(2_000_000)
        );
        let full_context = TokenUsage {
            prompt_tokens: 131_072,
            ..Default::default()
        };
        assert_eq!(
            cost("groq", "qwen/qwen3.8-27b", full_context, false, REFRESHED).amount_nano_usd,
            Some(104_857_600)
        );
    }

    #[test]
    fn arbitrary_model_from_json_is_priced_without_model_name_rules() {
        let mut models = parse_catalog(BUILTIN_CATALOG).unwrap();
        let mut new = models[0].clone();
        new.id = uuid::Uuid::new_v4();
        new.model_id = "unreleased-model-with-arbitrary-name".into();
        new.aliases = vec!["operator-alias".into()];
        new.provider = "custom-provider".into();
        let json = serde_json::to_string(&CatalogDocument {
            schema_version: 1,
            models: vec![new],
        })
        .unwrap();
        merge_catalog(&mut models, parse_catalog(&json).unwrap()).unwrap();
        let a = estimate_from_catalog(
            &models,
            "CUSTOM-PROVIDER",
            "operator-alias",
            &TokenUsage {
                prompt_tokens: 100,
                ..Default::default()
            },
            false,
            NOW.parse().unwrap(),
        );
        assert_eq!(a.amount_nano_usd, Some(400_000));
        assert_eq!(a.status, CostStatus::Estimated);
    }

    #[test]
    fn malformed_or_ambiguous_catalog_is_rejected() {
        let base: serde_json::Value = serde_json::from_str(BUILTIN_CATALOG).unwrap();
        let paths = [
            ("/schema_version", serde_json::json!(99)),
            ("/models/0/pricing/currency", serde_json::json!("EUR")),
            (
                "/models/0/pricing/periods/0/bands/0/rates/input",
                serde_json::json!(-1),
            ),
            (
                "/models/0/pricing/periods/0/bands/0/rates/rate_scale",
                serde_json::json!(0),
            ),
            (
                "/models/0/pricing/periods/0/bands/0/utc_end_minute",
                serde_json::json!(1441),
            ),
        ];
        for (path, value) in paths {
            let mut invalid = base.clone();
            *invalid.pointer_mut(path).unwrap() = value;
            assert!(parse_catalog(&invalid.to_string()).is_err(), "{path}");
        }
        let mut invalid = base.clone();
        invalid["models"][0]["pricing"]["periods"][0]["bands"][0]["rates"]["cache_reed"] =
            serde_json::json!(1);
        assert!(parse_catalog(&invalid.to_string()).is_err());
        let mut invalid = base;
        invalid["models"][0]["aliases"] = serde_json::json!(["gpt-5.6-terra"]);
        assert!(parse_catalog(&invalid.to_string()).is_err());
    }

    #[test]
    fn overlapping_periods_and_bands_are_rejected_but_gaps_return_unknown() {
        let mut models = parse_catalog(BUILTIN_CATALOG).unwrap();
        let pricing = models[0].pricing.as_mut().unwrap();
        let p = pricing.periods[0].clone();
        pricing.periods.push(p);
        assert!(validate_models(&models).is_err());
        let pricing = models[0].pricing.as_mut().unwrap();
        pricing.periods.pop();
        let b = pricing.periods[0].bands[0].clone();
        pricing.periods[0].bands.push(b);
        assert!(validate_models(&models).is_err());
        assert_eq!(
            cost(
                "openai",
                "gpt-6-astra",
                TokenUsage::default(),
                false,
                "2020-01-01T00:00:00Z"
            )
            .status,
            CostStatus::Unknown
        );
    }

    #[test]
    fn fable_cache_discount_and_batch_writes_use_explicit_rates() {
        let usage = TokenUsage {
            prompt_tokens: 1_000,
            cached_prompt_tokens: 600,
            cache_creation_input_tokens: 200,
            cache_creation_1h_tokens: 80,
            completion_tokens: 100,
            reasoning_tokens: 60,
            ..Default::default()
        };
        // 200 normal + 600 reads + 120 short writes + 80 one-hour writes + 100 outputs.
        let expected = 200 * 10_000 + 600 * 250 + 120 * 12_500 + 80 * 20_000 + 100 * 50_000;
        assert_eq!(
            cost("anthropic", "claude-fable-5-1", usage, false, NOW).amount_nano_usd,
            Some(expected)
        );
        assert_eq!(
            cost("anthropic", "claude-fable-5-1", usage, true, NOW).amount_nano_usd,
            Some(expected / 2)
        );
    }

    #[test]
    fn gemini_promotion_expires_and_fractional_nano_rates_are_preserved() {
        let usage = TokenUsage {
            prompt_tokens: 100,
            cached_prompt_tokens: 100,
            ..Default::default()
        };
        for model in ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"] {
            let before = cost("gemini", model, usage, true, "2026-12-31T23:59:59Z");
            let after = cost("gemini", model, usage, true, "2027-01-01T00:00:00Z");
            assert_eq!(before.amount_nano_usd, Some(3750)); // 100 * 37.5 nano-USD
            assert_eq!(after.amount_nano_usd, Some(7500));
            assert_ne!(before.pricing_version, after.pricing_version);
        }
    }

    #[test]
    fn context_price_boundary_counts_cached_input_and_reasoning_only_once() {
        let usage = TokenUsage {
            prompt_tokens: 272_000,
            cached_prompt_tokens: 272_000,
            completion_tokens: 100,
            reasoning_tokens: 50,
            ..Default::default()
        };
        assert_eq!(
            cost("openai", "gpt-6-astra", usage, false, NOW).amount_nano_usd,
            Some(277_000_000)
        );
        let long = TokenUsage {
            prompt_tokens: 272_001,
            cached_prompt_tokens: 272_001,
            ..usage
        };
        assert_eq!(
            cost("openai", "gpt-6-astra", long, false, NOW).amount_nano_usd,
            Some(551_502_000)
        );
        assert_eq!(
            cost(
                "openai",
                "gpt-6-astra",
                TokenUsage {
                    prompt_tokens: 1_050_001,
                    ..Default::default()
                },
                false,
                NOW
            )
            .status,
            CostStatus::Unknown
        );
    }

    #[test]
    fn deepseek_utc_windows_and_unknown_sales_prices() {
        let usage = TokenUsage {
            prompt_tokens: 1000,
            ..Default::default()
        };
        for (at, expected) in [
            ("2026-09-05T00:59:59Z", 220_000),
            ("2026-09-05T01:00:00Z", 440_000),
            ("2026-09-05T04:00:00Z", 220_000),
            ("2026-09-05T06:00:00Z", 440_000),
            ("2026-09-05T10:00:00Z", 220_000),
        ] {
            assert_eq!(
                cost("deepseek", "deepseek-v4-flash", usage, false, at).amount_nano_usd,
                Some(expected)
            );
        }
        assert_eq!(
            cost("groq", "llama-3.3-70b-versatile", usage, false, NOW).status,
            CostStatus::Unknown
        );
    }

    #[test]
    fn old_receipts_deserialize_with_unit_scale_and_do_not_reprice() {
        let rates: TokenRates =
            serde_json::from_value(serde_json::json!({"input":1000,"output":2000})).unwrap();
        assert_eq!(rates.rate_scale, 1);
        assert_eq!(
            crate::pricing::calculate(
                "p",
                "m",
                "old-v1",
                rates,
                &TokenUsage {
                    prompt_tokens: 100,
                    ..Default::default()
                },
                false
            )
            .amount_nano_usd,
            Some(100_000)
        );
    }

    #[test]
    fn batch_discounts_are_provider_data_and_do_not_always_stack() {
        let usage = TokenUsage {
            prompt_tokens: 1000,
            cached_prompt_tokens: 1000,
            ..Default::default()
        };
        // Groq bills cached batch input at ordinary batch input, not half its cache price.
        assert_eq!(
            cost("groq", "openai/gpt-oss-120b", usage, true, NOW).amount_nano_usd,
            Some(75_000)
        );
        assert_eq!(
            cost("groq", "openai/gpt-oss-20b", usage, true, NOW).amount_nano_usd,
            Some(37_500)
        );
        // xAI's eligible models have a 20% batch discount, not a universal 50%.
        assert_eq!(
            cost("xai", "grok-4.3", usage, true, NOW).amount_nano_usd,
            Some(160_000)
        );
        assert_eq!(
            cost("xai", "grok-4.6", usage, true, NOW).status,
            CostStatus::Unknown
        );
    }

    #[test]
    fn file_override_adds_models_and_invalid_file_does_not_fall_back() {
        let path =
            std::env::temp_dir().join(format!("reincore-models-{}.json", uuid::Uuid::new_v4()));
        let mut model = parse_catalog(BUILTIN_CATALOG).unwrap().remove(0);
        model.model_id = "deployment-model".into();
        model.aliases.clear();
        std::fs::write(
            &path,
            serde_json::to_string(&CatalogDocument {
                schema_version: 1,
                models: vec![model],
            })
            .unwrap(),
        )
        .unwrap();
        assert!(load_catalog(Some(&path))
            .unwrap()
            .iter()
            .any(|m| m.model_id == "deployment-model"));
        std::fs::write(&path, "{broken}").unwrap();
        assert!(load_catalog(Some(&path)).is_err());
        std::fs::remove_file(path).unwrap();
    }
}

pub const BUILTIN_CATALOG: &str = include_str!("../../../catalog/models.json");

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PricingSchedule {
    pub currency: String,
    pub version: String,
    pub verified_at: NaiveDate,
    pub sources: Vec<String>,
    pub periods: Vec<PricingPeriod>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PricingPeriod {
    pub effective_from: DateTime<Utc>,
    pub effective_until: Option<DateTime<Utc>>,
    pub bands: Vec<PricingBand>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PricingBand {
    /// Inclusive bounds on total input, including cached and cache-write tokens.
    pub min_input_tokens: u64,
    pub max_input_tokens: u64,
    /// UTC minute of day, start inclusive / end exclusive; split midnight ranges.
    pub utc_start_minute: u32,
    pub utc_end_minute: u32,
    /// Optional UTC weekday filter: 0 = Monday, 6 = Sunday. Empty means every day.
    /// Omit empty filters so existing schedule fingerprints remain unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub utc_weekdays: Vec<u32>,
    pub rates: TokenRates,
}

impl PricingBand {
    fn applies_on_weekday(&self, weekday: u32) -> bool {
        self.utc_weekdays.is_empty() || self.utc_weekdays.contains(&weekday)
    }
}

impl PricingSchedule {
    pub fn rates_at(&self, tokens: u64, at: DateTime<Utc>) -> Option<&TokenRates> {
        let minute = at.hour() * 60 + at.minute();
        let weekday = at.weekday().num_days_from_monday();
        self.periods
            .iter()
            .find(|p| at >= p.effective_from && p.effective_until.is_none_or(|end| at < end))?
            .bands
            .iter()
            .find(|b| {
                tokens >= b.min_input_tokens
                    && tokens <= b.max_input_tokens
                    && minute >= b.utc_start_minute
                    && minute < b.utc_end_minute
                    && b.applies_on_weekday(weekday)
            })
            .map(|b| &b.rates)
    }

    fn validate(&self) -> Result<(), String> {
        if self.currency != "USD"
            || self.version.trim().is_empty()
            || self.sources.is_empty()
            || self.sources.iter().any(|s| {
                url::Url::parse(s).map_or(true, |url| {
                    url.scheme() != "https" || url.host_str().is_none()
                })
            })
            || self.periods.is_empty()
        {
            return Err("pricing requires USD, a version, HTTPS sources and periods".into());
        }
        for (i, p) in self.periods.iter().enumerate() {
            if p.effective_until.is_some_and(|end| end <= p.effective_from) || p.bands.is_empty() {
                return Err("invalid pricing period".into());
            }
            for q in &self.periods[..i] {
                if q.effective_until.is_none_or(|end| p.effective_from < end)
                    && p.effective_until.is_none_or(|end| q.effective_from < end)
                {
                    return Err("overlapping pricing periods".into());
                }
            }
            for (j, b) in p.bands.iter().enumerate() {
                if b.min_input_tokens > b.max_input_tokens
                    || b.utc_start_minute >= b.utc_end_minute
                    || b.utc_end_minute > 1440
                    || b.utc_weekdays.iter().any(|day| *day > 6)
                    || b.utc_weekdays.iter().collect::<HashSet<_>>().len() != b.utc_weekdays.len()
                    || !b.rates.is_valid()
                {
                    return Err("invalid pricing band or negative/invalid rates".into());
                }
                for c in &p.bands[..j] {
                    if b.min_input_tokens <= c.max_input_tokens
                        && c.min_input_tokens <= b.max_input_tokens
                        && b.utc_start_minute < c.utc_end_minute
                        && c.utc_start_minute < b.utc_end_minute
                        && (0..7).any(|day| b.applies_on_weekday(day) && c.applies_on_weekday(day))
                    {
                        return Err("overlapping context/time pricing bands".into());
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogDocument {
    pub schema_version: u32,
    pub models: Vec<ProviderModelRecord>,
}

pub fn parse_catalog(json: &str) -> Result<Vec<ProviderModelRecord>, String> {
    let mut document: CatalogDocument = serde_json::from_str(json).map_err(|e| e.to_string())?;
    if document.schema_version != 1 {
        return Err("unsupported catalog schema_version".into());
    }
    // SQL uniqueness is case-sensitive, while provider lookup is not. Keep
    // a canonical spelling so overrides update the same stored model row.
    for model in &mut document.models {
        model.provider.make_ascii_lowercase();
    }
    validate_models(&document.models)?;
    Ok(document.models)
}

fn validate_models(models: &[ProviderModelRecord]) -> Result<(), String> {
    let mut names = HashSet::new();
    let mut ids = HashSet::new();
    for m in models {
        if !ids.insert(m.id)
            || m.provider.trim() != m.provider
            || m.model_id.trim() != m.model_id
            || m.provider.trim().is_empty()
            || m.model_id.trim().is_empty()
            || m.model_name.trim().is_empty()
            || m.max_input_tokens <= 0
            || m.max_output_tokens <= 0
            || !m.input_price.is_finite()
            || m.input_price < 0.0
            || !m.output_price.is_finite()
            || m.output_price < 0.0
        {
            return Err(format!("invalid model {}:{}", m.provider, m.model_id));
        }
        for name in std::iter::once(&m.model_id).chain(m.aliases.iter()) {
            if name.trim().is_empty()
                || name.trim() != name
                || !names.insert((m.provider.to_lowercase(), name))
            {
                return Err(format!(
                    "duplicate/empty model or alias {}:{name}",
                    m.provider
                ));
            }
        }
        if let Some(pricing) = &m.pricing {
            pricing
                .validate()
                .map_err(|e| format!("{}:{}: {e}", m.provider, m.model_id))?;
            if pricing
                .periods
                .iter()
                .flat_map(|p| &p.bands)
                .any(|b| b.max_input_tokens > m.max_input_tokens as u64)
            {
                return Err(format!("pricing exceeds model context: {}", m.model_id));
            }
        }
    }
    Ok(())
}

/// An optional JSON document adds/replaces complete records by provider + model ID.
/// A malformed override fails startup; it never silently falls back to old prices.
pub fn load_catalog(override_path: Option<&Path>) -> Result<Vec<ProviderModelRecord>, String> {
    let mut models = parse_catalog(BUILTIN_CATALOG)?;
    if let Some(path) = override_path {
        let json = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        merge_catalog(&mut models, parse_catalog(&json)?)?;
    }
    Ok(models)
}

pub fn merge_catalog(
    models: &mut Vec<ProviderModelRecord>,
    mut overrides: Vec<ProviderModelRecord>,
) -> Result<(), String> {
    for model in &mut overrides {
        model.provider.make_ascii_lowercase();
    }
    validate_models(&overrides)?;
    let mut merged = models.clone();
    for model in overrides {
        if let Some(old) = merged.iter_mut().find(|m| {
            m.provider.eq_ignore_ascii_case(&model.provider) && m.model_id == model.model_id
        }) {
            *old = model;
        } else {
            merged.push(model);
        }
    }
    validate_models(&merged)?;
    *models = merged;
    Ok(())
}
