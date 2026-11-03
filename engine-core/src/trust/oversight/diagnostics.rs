//! Call-level diagnostics from recorded usage, never task-quality rewards.
use crate::event::{SessionEvent, SessionEventType};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct CallDiagnostics {
    pub prediction: PredictionBreakdown,
    pub cache: CacheBreakdown,
    pub reincore_cache: ReinCoreCacheBreakdown,
}

/// Hit rates for the caches ReinCore owns, as distinct from the upstream
/// prompt cache measured by [`CacheBreakdown`].
///
/// The two answer different questions and must not be collapsed. A provider
/// that never reports cached prompt tokens - a local Ollama model, say - says
/// nothing about whether these are working, and a response-cache hit skips the
/// provider entirely, so it contributes no prompt tokens to [`CacheBreakdown`]
/// at all. Measured on its own, effective ReinCore caching would otherwise look
/// like an idle cache.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ReinCoreCacheBreakdown {
    /// LLM calls seen, whether answered by a provider or by the response cache.
    pub llm_call_count: u64,
    /// Calls answered from the response cache with no provider request.
    pub response_cache_hit_count: u64,
    /// 10,000 * response_cache_hit_count / llm_call_count.
    pub response_hit_rate_bps: Option<f64>,
    /// Tool results seen, whether executed or replayed from memoization.
    pub tool_result_count: u64,
    /// Tool results replayed from memoization without re-executing the tool.
    pub tool_cache_hit_count: u64,
    /// 10,000 * tool_cache_hit_count / tool_result_count.
    pub tool_hit_rate_bps: Option<f64>,
    /// Median similarity of the entries that actually answered from cache.
    /// Clustered near the reuse threshold means reuse is marginal; clustered
    /// near 10,000 means it is mostly catching near-exact repeats.
    pub hit_similarity_median_bps: Option<f64>,
    /// Calls that went to the provider while a candidate had been retrieved
    /// below the reuse threshold.
    pub near_miss_count: u64,
    pub near_miss_median_bps: Option<f64>,
    pub near_miss_p90_bps: Option<f64>,
    /// Near misses a lower reuse threshold would have converted into hits.
    ///
    /// This is the counterfactual, measured on real traffic without changing
    /// what was served: it says what lowering the threshold would buy before
    /// anyone lowers it. It says nothing about whether those reuses would have
    /// been *correct* - that needs a labelled eval, not counters.
    pub would_reuse_at_9500_bps: u64,
    pub would_reuse_at_9000_bps: u64,
    pub would_reuse_at_8500_bps: u64,
    /// Calls whose declared tool schema differed from the previous call in the
    /// same session.
    ///
    /// Each one invalidated the *entire* cached prefix rather than just the
    /// conversation tail, because tools render before system and messages. For a
    /// gateway aggregating MCP servers - where a server connecting or dropping
    /// rewrites the tool list - this is the most likely silent cause of a
    /// collapsed cache-read rate. Meaningful only when events are supplied in
    /// per-session sequence order, which is how the stores return them.
    pub tool_schema_changes: u64,
    /// Calls that declared a tool schema at all: the denominator for the above.
    pub tool_schema_observations: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct PredictionBreakdown {
    pub scope: String,
    pub calibrated_session_count: u64,
    /// Known costs with nonnegative ordered prediction bounds, including zero.
    pub sample_count: u64,
    pub ratio_sample_count: u64,
    /// Arithmetic mean of actual / high * 10,000 for positive high bounds.
    pub mean_actual_to_predicted_high_ratio_bps: Option<f64>,
    pub median_actual_to_predicted_high_ratio_bps: Option<f64>,
    pub p90_actual_to_predicted_high_ratio_bps: Option<f64>,
    /// 10,000 * count(low <= actual <= high) / sample_count.
    pub interval_coverage_bps: Option<f64>,
    /// Mean abs(actual - midpoint(low, high)), in nano USD.
    pub mean_absolute_midpoint_error_nano_usd: Option<f64>,
    /// Mean distance outside [low, high], zero for covered calls, in nano USD.
    pub mean_interval_miss_nano_usd: Option<f64>,
    pub total_predicted_high_cents: i64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct CacheBreakdown {
    pub scope: String,
    pub sampled_session_count: u64,
    /// Calls with explicit valid cached and positive prompt token counts.
    pub sample_count: u64,
    /// 10,000 * sum(cached_prompt_tokens) / sum(input_tokens).
    pub token_weighted_utilization_bps: Option<f64>,
    /// Compatibility field: same token-weighted formula; zero if no samples.
    pub mean_hit_rate_bps: f64,
    /// Percentiles of individual call ratios, using linear interpolation.
    pub median_hit_rate_bps: f64,
    pub p10_hit_rate_bps: f64,
    pub p90_hit_rate_bps: f64,
    pub total_cached_prompt_tokens: i64,
    pub total_prompt_tokens: i64,
}

// Input is sorted once for all requested percentiles.
fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let rank = p * (values.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    Some(values[lo] + (values[hi] - values[lo]) * (rank - lo as f64))
}

#[derive(Default)]
pub struct DiagnosticAccumulator {
    result: CallDiagnostics,
    prediction_sessions: HashSet<uuid::Uuid>,
    cache_sessions: HashSet<uuid::Uuid>,
    ratios: Vec<f64>,
    cache_ratios: Vec<f64>,
    hit_similarities: Vec<f64>,
    near_misses: Vec<f64>,
    last_tool_schema: HashMap<uuid::Uuid, String>,
    covered: u64,
    absolute_error: f64,
    interval_miss: f64,
    seen: HashSet<(uuid::Uuid, i64)>,
}

impl CallDiagnostics {
    pub fn from_events(events: &[SessionEvent]) -> Self {
        let mut accumulator = DiagnosticAccumulator::default();
        accumulator.extend(events);
        accumulator.finish()
    }
}

impl DiagnosticAccumulator {
    pub fn extend(&mut self, events: &[SessionEvent]) {
        let Self {
            result,
            prediction_sessions,
            cache_sessions,
            ratios,
            cache_ratios,
            hit_similarities,
            near_misses,
            last_tool_schema,
            covered,
            absolute_error,
            interval_miss,
            seen,
        } = self;
        for event in events {
            if !matches!(
                event.event_type,
                SessionEventType::LlmCall
                    | SessionEventType::AuxiliaryCall
                    | SessionEventType::ToolResult
            ) || !seen.insert((event.session_id, event.seq))
            {
                continue;
            }
            let payload = &event.payload;
            // Tool memoization replays a stored result instead of executing the
            // tool again. It is provider independent, so it is counted whatever
            // the upstream prompt cache reports.
            if matches!(event.event_type, SessionEventType::ToolResult) {
                result.reincore_cache.tool_result_count += 1;
                if payload.get("cache_hit").and_then(|v| v.as_bool()) == Some(true) {
                    result.reincore_cache.tool_cache_hit_count += 1;
                }
                continue;
            }
            // A response-cache hit is recorded as an ordinary LLM call carrying
            // the synthetic `semantic_cache` tier, since no provider ran.
            result.reincore_cache.llm_call_count += 1;
            if payload.get("model_tier").and_then(|v| v.as_str()) == Some("semantic_cache") {
                result.reincore_cache.response_cache_hit_count += 1;
            }
            if let Some(similarity) = payload
                .get("semantic_cache_similarity_bps")
                .and_then(|v| v.as_i64())
            {
                hit_similarities.push(similarity as f64);
            }
            // A candidate was retrieved but did not clear the reuse threshold,
            // so the provider answered instead. Counting how far below it landed
            // is what makes a later threshold change arithmetic rather than a
            // guess - and costs nothing, because the lookup already ran.
            if let Some(best) = payload
                .get("semantic_best_candidate_bps")
                .and_then(|v| v.as_i64())
            {
                let cache = &mut result.reincore_cache;
                cache.near_miss_count += 1;
                if best >= 9_500 {
                    cache.would_reuse_at_9500_bps += 1;
                }
                if best >= 9_000 {
                    cache.would_reuse_at_9000_bps += 1;
                }
                if best >= 8_500 {
                    cache.would_reuse_at_8500_bps += 1;
                }
                near_misses.push(best as f64);
            }
            // Tools render first, so a change here costs the whole prefix rather
            // than the tail. Counted per session against the previous call.
            if let Some(hash) = payload.get("tool_schema_hash").and_then(|v| v.as_str()) {
                result.reincore_cache.tool_schema_observations += 1;
                if let Some(previous) = last_tool_schema.insert(event.session_id, hash.to_string())
                {
                    if previous != hash {
                        result.reincore_cache.tool_schema_changes += 1;
                    }
                }
            }
            if let (Some(input), Some(cached)) = (
                payload.get("input_tokens").and_then(|v| v.as_i64()),
                payload.get("cached_prompt_tokens").and_then(|v| v.as_i64()),
            ) {
                if input > 0 && cached >= 0 && cached <= input {
                    // Skip counts that cannot be represented in the public totals.
                    if let (Some(total), Some(hits)) = (
                        result.cache.total_prompt_tokens.checked_add(input),
                        result.cache.total_cached_prompt_tokens.checked_add(cached),
                    ) {
                        result.cache.total_prompt_tokens = total;
                        result.cache.total_cached_prompt_tokens = hits;
                        cache_sessions.insert(event.session_id);
                        cache_ratios.push(cached as f64 / input as f64 * 10_000.0);
                    }
                }
            }
            let actual = if payload.get("cost").is_some()
                || payload.get("cost_recorded").and_then(|v| v.as_bool()) == Some(true)
            {
                payload
                    .get("cost")
                    .and_then(|cost| cost.get("amount_nano_usd"))
                    .and_then(|v| v.as_i64())
            } else {
                event.cost_cents.and_then(|v| v.checked_mul(10_000_000))
            };
            let bounds = payload
                .get("predicted_cost_low_cents")
                .and_then(|v| v.as_i64())
                .zip(
                    payload
                        .get("predicted_cost_high_cents")
                        .and_then(|v| v.as_i64()),
                );
            if let (Some(actual), Some((low, high))) = (actual, bounds) {
                if actual < 0 || low < 0 || high < low {
                    continue;
                }
                let Some(total_high) = result
                    .prediction
                    .total_predicted_high_cents
                    .checked_add(high)
                else {
                    continue;
                };
                let (Some(low_nano), Some(high_nano)) =
                    (low.checked_mul(10_000_000), high.checked_mul(10_000_000))
                else {
                    continue;
                };
                result.prediction.total_predicted_high_cents = total_high;
                result.prediction.sample_count += 1;
                prediction_sessions.insert(event.session_id);
                *covered += u64::from(low_nano <= actual && actual <= high_nano);
                *absolute_error +=
                    (actual as f64 - (low_nano as f64 / 2.0 + high_nano as f64 / 2.0)).abs();
                *interval_miss += (low_nano as f64 - actual as f64).max(0.0)
                    + (actual as f64 - high_nano as f64).max(0.0);
                if high_nano > 0 {
                    ratios.push(actual as f64 / high_nano as f64 * 10_000.0);
                }
            }
        }
    }

    /// Fixed-size event pages with a per-session high-water mark. Exact
    /// percentiles retain numeric samples, never historical payloads.
    pub async fn append_session<E: crate::EventStore + ?Sized>(
        &mut self,
        store: &E,
        session_id: uuid::Uuid,
    ) -> crate::EngineResult<()> {
        let end = store.latest_seq(session_id).await?.unwrap_or(0);
        let mut next = 1;
        while next <= end {
            let mut page = store
                .read_diagnostic_event_page(session_id, next, 512)
                .await?;
            page.retain(|e| e.seq <= end);
            let Some(last) = page.last() else {
                break;
            };
            let last_seq = last.seq;
            self.extend(&page);
            let Some(cursor) = last_seq.checked_add(1) else {
                break;
            };
            if cursor <= next {
                return Err(crate::EngineError::Storage(
                    "diagnostic cursor did not advance".into(),
                ));
            }
            next = cursor;
        }
        Ok(())
    }

    pub fn finish(self) -> CallDiagnostics {
        let Self {
            mut result,
            prediction_sessions,
            cache_sessions,
            mut ratios,
            mut cache_ratios,
            mut hit_similarities,
            mut near_misses,
            covered,
            absolute_error,
            interval_miss,
            ..
        } = self;
        let prediction = &mut result.prediction;
        prediction.calibrated_session_count = prediction_sessions.len() as u64;
        prediction.ratio_sample_count = ratios.len() as u64;
        if prediction.sample_count > 0 {
            let n = prediction.sample_count as f64;
            prediction.interval_coverage_bps = Some(covered as f64 / n * 10_000.0);
            prediction.mean_absolute_midpoint_error_nano_usd = Some(absolute_error / n);
            prediction.mean_interval_miss_nano_usd = Some(interval_miss / n);
        }
        if !ratios.is_empty() {
            prediction.mean_actual_to_predicted_high_ratio_bps =
                Some(ratios.iter().sum::<f64>() / ratios.len() as f64);
        }
        ratios.sort_by(f64::total_cmp);
        prediction.median_actual_to_predicted_high_ratio_bps = percentile(&ratios, 0.5);
        prediction.p90_actual_to_predicted_high_ratio_bps = percentile(&ratios, 0.9);
        let reincore_cache = &mut result.reincore_cache;
        if reincore_cache.llm_call_count > 0 {
            reincore_cache.response_hit_rate_bps = Some(
                reincore_cache.response_cache_hit_count as f64
                    / reincore_cache.llm_call_count as f64
                    * 10_000.0,
            );
        }
        if reincore_cache.tool_result_count > 0 {
            reincore_cache.tool_hit_rate_bps = Some(
                reincore_cache.tool_cache_hit_count as f64
                    / reincore_cache.tool_result_count as f64
                    * 10_000.0,
            );
        }
        hit_similarities.sort_by(f64::total_cmp);
        reincore_cache.hit_similarity_median_bps = percentile(&hit_similarities, 0.5);
        near_misses.sort_by(f64::total_cmp);
        reincore_cache.near_miss_median_bps = percentile(&near_misses, 0.5);
        reincore_cache.near_miss_p90_bps = percentile(&near_misses, 0.9);
        let cache = &mut result.cache;
        cache.sampled_session_count = cache_sessions.len() as u64;
        cache.sample_count = cache_ratios.len() as u64;
        if cache.total_prompt_tokens > 0 {
            cache.token_weighted_utilization_bps = Some(
                cache.total_cached_prompt_tokens as f64 / cache.total_prompt_tokens as f64
                    * 10_000.0,
            );
        }
        cache.mean_hit_rate_bps = cache.token_weighted_utilization_bps.unwrap_or(0.0);
        cache_ratios.sort_by(f64::total_cmp);
        cache.median_hit_rate_bps = percentile(&cache_ratios, 0.5).unwrap_or(0.0);
        cache.p10_hit_rate_bps = percentile(&cache_ratios, 0.1).unwrap_or(0.0);
        cache.p90_hit_rate_bps = percentile(&cache_ratios, 0.9).unwrap_or(0.0);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use uuid::Uuid;

    fn call(seq: i64, input: i64, cached: i64, actual: i64, low: i64, high: i64) -> SessionEvent {
        SessionEvent {
            session_id: Uuid::nil(),
            seq,
            event_type: SessionEventType::LlmCall,
            payload: json!({"input_tokens": input, "cached_prompt_tokens": cached,
                "cost": {"amount_nano_usd": actual},
                "predicted_cost_low_cents": low, "predicted_cost_high_cents": high}),
            cost_cents: Some(0),
            created_at: chrono::Utc::now(),
        }
    }

    fn tool_result(seq: i64, cache_hit: bool) -> SessionEvent {
        SessionEvent {
            session_id: Uuid::nil(),
            seq,
            event_type: SessionEventType::ToolResult,
            payload: json!({ "cache_hit": cache_hit }),
            cost_cents: None,
            created_at: chrono::Utc::now(),
        }
    }

    fn response_cache_hit(seq: i64) -> SessionEvent {
        SessionEvent {
            session_id: Uuid::nil(),
            seq,
            event_type: SessionEventType::LlmCall,
            payload: json!({ "model_tier": "semantic_cache" }),
            cost_cents: None,
            created_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn reincore_own_cache_hits_are_counted_apart_from_the_provider_cache() {
        let stats = CallDiagnostics::from_events(&[
            call(1, 100, 0, 10_000_000, 0, 2),
            response_cache_hit(2),
            tool_result(3, true),
            tool_result(4, false),
            tool_result(5, true),
        ]);
        assert_eq!(stats.reincore_cache.llm_call_count, 2);
        assert_eq!(stats.reincore_cache.response_cache_hit_count, 1);
        assert_eq!(stats.reincore_cache.response_hit_rate_bps, Some(5000.0));
        assert_eq!(stats.reincore_cache.tool_result_count, 3);
        assert_eq!(stats.reincore_cache.tool_cache_hit_count, 2);
        assert_eq!(
            stats.reincore_cache.tool_hit_rate_bps,
            Some(2.0 / 3.0 * 10_000.0)
        );
        // The provider-side gauge stays flat throughout: the one real provider
        // call reported no cached prompt tokens. This is the Ollama shape, and
        // the reason the two must be reported separately.
        assert_eq!(stats.cache.token_weighted_utilization_bps, Some(0.0));
    }

    fn near_miss(seq: i64, best_bps: i64) -> SessionEvent {
        SessionEvent {
            session_id: Uuid::nil(),
            seq,
            event_type: SessionEventType::LlmCall,
            payload: json!({ "semantic_best_candidate_bps": best_bps }),
            cost_cents: None,
            created_at: chrono::Utc::now(),
        }
    }

    fn call_with_tools(seq: i64, schema: &str) -> SessionEvent {
        SessionEvent {
            session_id: Uuid::nil(),
            seq,
            event_type: SessionEventType::LlmCall,
            payload: json!({ "tool_schema_hash": schema }),
            cost_cents: None,
            created_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn tool_schema_churn_is_counted_per_session_transition() {
        let stats = CallDiagnostics::from_events(&[
            call_with_tools(1, "aaa"),
            call_with_tools(2, "aaa"),
            call_with_tools(3, "bbb"),
            call_with_tools(4, "aaa"),
        ]);
        assert_eq!(stats.reincore_cache.tool_schema_observations, 4);
        // aaa->bbb and bbb->aaa. The first call has no predecessor, so it is
        // never a change - a session is not churning just by starting.
        assert_eq!(stats.reincore_cache.tool_schema_changes, 2);
    }

    #[test]
    fn a_stable_tool_schema_reports_no_churn() {
        let stats =
            CallDiagnostics::from_events(&[call_with_tools(1, "aaa"), call_with_tools(2, "aaa")]);
        assert_eq!(stats.reincore_cache.tool_schema_changes, 0);
        assert_eq!(stats.reincore_cache.tool_schema_observations, 2);
    }

    #[test]
    fn near_misses_measure_what_a_lower_threshold_would_have_bought() {
        let stats = CallDiagnostics::from_events(&[
            near_miss(1, 9_800),
            near_miss(2, 9_200),
            near_miss(3, 8_700),
            near_miss(4, 6_000),
        ]);
        assert_eq!(stats.reincore_cache.near_miss_count, 4);
        assert_eq!(stats.reincore_cache.would_reuse_at_9500_bps, 1);
        assert_eq!(stats.reincore_cache.would_reuse_at_9000_bps, 2);
        assert_eq!(stats.reincore_cache.would_reuse_at_8500_bps, 3);
        assert_eq!(stats.reincore_cache.near_miss_median_bps, Some(8_950.0));
        // A near miss is a provider call, never a hit.
        assert_eq!(stats.reincore_cache.response_cache_hit_count, 0);
    }

    #[test]
    fn hit_similarity_is_recorded_apart_from_near_misses() {
        let mut hit = response_cache_hit(1);
        hit.payload = json!({
            "model_tier": "semantic_cache",
            "semantic_cache_similarity_bps": 9_900
        });
        let stats = CallDiagnostics::from_events(&[hit, near_miss(2, 9_000)]);
        assert_eq!(
            stats.reincore_cache.hit_similarity_median_bps,
            Some(9_900.0)
        );
        assert_eq!(stats.reincore_cache.response_cache_hit_count, 1);
        assert_eq!(stats.reincore_cache.near_miss_count, 1);
        assert_eq!(stats.reincore_cache.would_reuse_at_9000_bps, 1);
    }

    #[test]
    fn unequal_call_volumes_use_exact_token_sums_and_call_percentiles() {
        let a = call(1, 10, 10, 10_000_000, 0, 2);
        let mut b = call(2, 90, 0, 40_000_000, 1, 2);
        b.session_id = Uuid::new_v4();
        let stats = CallDiagnostics::from_events(&[a.clone(), b, a]);
        assert_eq!(stats.cache.sample_count, 2);
        assert_eq!(stats.cache.sampled_session_count, 2);
        assert_eq!(stats.cache.total_prompt_tokens, 100);
        assert_eq!(stats.cache.total_cached_prompt_tokens, 10);
        assert_eq!(stats.cache.token_weighted_utilization_bps, Some(1000.0));
        assert_eq!(stats.cache.median_hit_rate_bps, 5000.0);
        assert_eq!(stats.prediction.interval_coverage_bps, Some(5000.0));
        assert_eq!(
            stats.prediction.mean_actual_to_predicted_high_ratio_bps,
            Some(12500.0)
        );
        assert_eq!(
            stats.prediction.mean_absolute_midpoint_error_nano_usd,
            Some(12_500_000.0)
        );
        assert_eq!(
            stats.prediction.mean_interval_miss_nano_usd,
            Some(10_000_000.0)
        );
        assert_eq!(stats.prediction.total_predicted_high_cents, 4);
    }

    #[test]
    fn unknown_invalid_and_zero_denominators_are_explicit() {
        let free = call(1, 0, 0, 0, 0, 0);
        let mut unknown = call(2, 10, 11, 0, 1, 2);
        unknown.payload["cost"]["amount_nano_usd"] = serde_json::Value::Null;
        let mut absent = call(3, 10, 0, 0, 1, 2);
        absent
            .payload
            .as_object_mut()
            .unwrap()
            .remove("cached_prompt_tokens");
        absent.payload.as_object_mut().unwrap().remove("cost");
        absent.cost_cents = None;
        let invalid = call(4, -1, 0, 1, 3, 2);
        let stats = CallDiagnostics::from_events(&[free, unknown, absent, invalid]);
        assert_eq!(stats.prediction.sample_count, 1);
        assert_eq!(stats.prediction.ratio_sample_count, 0);
        assert_eq!(
            stats.prediction.mean_actual_to_predicted_high_ratio_bps,
            None
        );
        assert_eq!(stats.prediction.interval_coverage_bps, Some(10_000.0));
        assert_eq!(stats.cache.token_weighted_utilization_bps, None);
        assert_eq!(stats.cache.sample_count, 0);
        assert_eq!(
            CallDiagnostics::default().prediction.interval_coverage_bps,
            None
        );
    }

    #[test]
    fn fractional_cents_and_legacy_costs_are_supported() {
        let a = call(1, 1, 0, 100_000, 0, 1);
        let mut b = call(2, 1, 1, 0, 0, 1);
        b.payload.as_object_mut().unwrap().remove("cost");
        b.cost_cents = Some(1);
        let stats = CallDiagnostics::from_events(&[a, b]);
        assert_eq!(
            stats.prediction.mean_actual_to_predicted_high_ratio_bps,
            Some(5050.0)
        );
        assert_eq!(stats.prediction.interval_coverage_bps, Some(10_000.0));
    }
}
