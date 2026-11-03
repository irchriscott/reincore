//! Cascading model routing and escalation classifier.
//!
//! Decides which model tier (Lite, Standard, Pro, Ultra) should handle a
//! given turn. A lightweight heuristic classifier inspects the request
//! complexity and produces a [`RoutingDecision`] with a confidence score.
//! Three [`RoutingMode`]s govern the strategy: `Auto` (derive a cheaper Lite
//! sibling from the catalog), `Custom` (user explicitly maps each tier), or
//! `Locked` (never switch tiers without asking). The classifier output feeds
//! into the budget router, which may override the tier based on spend.

use crate::error::{EngineError, EngineResult};
use crate::event::{EventStore, SessionEvent, SessionEventType};
use crate::models::{ChatCompletionRequest, ModelTier};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Decision produced by the escalation classifier.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub model_tier: ModelTier,
    pub confidence: f32,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classification: Option<ClassificationTelemetry>,
}

/// How ReinCore picks the model tier around the user's standard model.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RoutingMode {
    /// User selects a standard model; ReinCore derives a compatible cheaper
    /// Lite sibling from the catalog and keeps Pro/Ultra at the standard
    /// unless explicitly overridden. Recommended default.
    #[default]
    Auto,
    /// User explicitly configures LITE, PRO, and ULTRA. ReinCore never
    /// derives a model - an unmapped tier falls back to the standard model
    /// (or the legacy Pro mapping) instead of catalog derivation.
    Custom,
    /// Always use the standard model. Budget monitoring still runs and may
    /// checkpoint an exhausted session, but ReinCore never switches tiers
    /// without stopping or asking.
    Locked,
}

impl RoutingMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            RoutingMode::Auto => "auto",
            RoutingMode::Custom => "custom",
            RoutingMode::Locked => "locked",
        }
    }
}

impl std::str::FromStr for RoutingMode {
    type Err = EngineError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "custom" => Ok(Self::Custom),
            "locked" => Ok(Self::Locked),
            other => Err(EngineError::InvalidInput(format!(
                "unknown routing mode '{other}' (expected auto, custom, or locked)"
            ))),
        }
    }
}

/// Provider selection strategy when deriving tier models. Cross-provider
/// routing is opt-in because providers differ in privacy, authentication,
/// tool support, prompt caching, regional availability, and response
/// behavior. Only same-provider derivation is currently supported.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderStrategy {
    /// Only derive candidate models from the same provider as the standard
    /// model. This is the safe default.
    #[default]
    SameProvider,
}

/// Minimum quality floor expressed as a model tier. Requests may not be
/// routed below this tier regardless of classification or budget pressure.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MinimumQuality {
    /// Allow the cheapest eligible model (Lite). Default.
    #[default]
    Economy,
    /// Never route below Pro.
    Balanced,
    /// Always use the strongest available tier (Ultra, falling back to Pro).
    High,
}

impl MinimumQuality {
    /// Convert the quality floor into a minimum model tier.
    pub fn min_tier(&self) -> ModelTier {
        match self {
            MinimumQuality::Economy => ModelTier::Lite,
            MinimumQuality::Balanced => ModelTier::Pro,
            MinimumQuality::High => ModelTier::Ultra,
        }
    }
}

/// Per-provider routing policy. Attached to `ProviderConfig` so the same
/// policy flows through local TOML configuration and cloud control-plane
/// JWT claims. All fields default to the recommended `auto` behavior, so
/// payloads that omit the policy keep working unchanged.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoutingPolicy {
    /// How tiers are selected around the standard model.
    #[serde(default)]
    pub mode: RoutingMode,
    /// Strategy for deriving candidate models from the catalog.
    #[serde(default)]
    pub provider_strategy: ProviderStrategy,
    /// Quality floor. ReinCore never routes below this tier.
    #[serde(default)]
    pub minimum_quality: MinimumQuality,
    /// Allow routing to a stronger tier than the standard model when an
    /// explicit Ultra override is configured. Disabled by default so
    /// automatic routing never upgrades above the chosen ceiling.
    #[serde(default)]
    pub allow_upgrade: bool,
    /// Allow budget pressure to degrade the selected tier. When `false`,
    /// budget monitoring still runs and may checkpoint an exhausted
    /// session, but the tier is never downgraded.
    #[serde(default = "default_allow_degradation")]
    pub allow_degradation: bool,
}

impl Default for RoutingPolicy {
    /// The recommended defaults: `auto` mode that may degrade under budget
    /// pressure but never upgrades above the classified tier.
    fn default() -> Self {
        Self {
            mode: RoutingMode::Auto,
            provider_strategy: ProviderStrategy::SameProvider,
            minimum_quality: MinimumQuality::Economy,
            allow_upgrade: false,
            allow_degradation: default_allow_degradation(),
        }
    }
}

fn default_allow_degradation() -> bool {
    true
}

impl RoutingPolicy {
    /// Build the policy from a TOML/JSON value, accepting aliases used by
    /// the control plane. Unknown or malformed fields fall back to the
    /// default `auto` policy rather than failing the whole provider.
    pub fn from_raw(value: Option<&serde_json::Value>) -> Self {
        let Some(value) = value else {
            return Self::default();
        };
        serde_json::from_value(value.clone()).unwrap_or_default()
    }
}

/// Per-request escape hatches a client can send in the request `extra` map.
/// These override the provider's configured policy for a single call.
#[derive(Clone, Copy, Debug, Default)]
pub struct RequestRoutingOverrides {
    pub mode: Option<RoutingMode>,
    pub minimum_tier: Option<ModelTier>,
    pub maximum_tier: Option<ModelTier>,
}

impl RequestRoutingOverrides {
    /// Read escape hatches from a request's `extra` map. Unknown values are
    /// ignored so a malformed override never breaks a request.
    pub fn from_extra(extra: &std::collections::HashMap<String, serde_json::Value>) -> Self {
        Self {
            mode: extra
                .get("reincore_routing_mode")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse().ok()),
            minimum_tier: extra
                .get("reincore_minimum_tier")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse().ok()),
            maximum_tier: extra
                .get("reincore_maximum_tier")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse().ok()),
        }
    }
}

/// Reason a routing-tier decision was adjusted, surfaced in response metadata.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RoutingAdjustment {
    /// No adjustment; the classified/budget tier was honored.
    None,
    /// `locked` mode pinned the request to the standard model.
    LockedToStandard,
    /// `custom` mode refused catalog derivation; an unmapped tier fell back
    /// to the standard model.
    CustomFallback,
    /// A per-request or configured minimum tier raised the selection.
    MinimumTierFloor,
    /// A per-request or configured maximum tier lowered the selection.
    MaximumTierCeiling,
    /// Degradation was disabled; the budget tier was ignored.
    DegradationDisabled,
    /// Upgrades above the standard model are disabled.
    UpgradeDisabled,
}

/// The final tier decision after applying the routing policy and per-request
/// escape hatches to the classified + budget-degraded tier.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ResolvedTier {
    pub tier: ModelTier,
    pub adjustment: RoutingAdjustment,
}

/// Compare tiers by quality. Lite < Pro < Ultra.
fn tier_rank(tier: ModelTier) -> u8 {
    match tier {
        ModelTier::Lite => 0,
        ModelTier::Pro => 1,
        ModelTier::Ultra => 2,
    }
}

/// Apply the routing policy and per-request overrides to the tier selected
/// by classification + budget degradation.
///
/// `classified_tier` is what the escalation classifier chose.
/// `budget_tier` is the tier after budget capping (already degradation-aware).
/// The result is the concrete tier the provider should run, plus the reason
/// it was adjusted (if any).
pub fn resolve_routing_tier(
    policy: &RoutingPolicy,
    overrides: &RequestRoutingOverrides,
    classified_tier: ModelTier,
    budget_tier: ModelTier,
) -> EngineResult<ResolvedTier> {
    let mode = overrides.mode.unwrap_or(policy.mode);
    let allow_degradation = policy.allow_degradation;
    let allow_upgrade = policy.allow_upgrade;

    // Budget degradation is the only thing that lowers the tier below the
    // classified value. If degradation is disabled, ignore the budget tier.
    let mut selected = budget_tier;
    let mut adjustment = RoutingAdjustment::None;
    if !allow_degradation && budget_tier != classified_tier {
        selected = classified_tier;
        adjustment = RoutingAdjustment::DegradationDisabled;
    }

    // `locked` always uses the standard model, which is Pro in the tier
    // ladder. The provider's `model_for_tier(Pro)` resolves to the standard
    // model (or its explicit Pro override). A locked-mode pin is an explicit
    // user choice and is not treated as a forbidden upgrade.
    if mode == RoutingMode::Locked && selected != ModelTier::Pro {
        selected = ModelTier::Pro;
        adjustment = RoutingAdjustment::LockedToStandard;
    }

    // `custom` forbids catalog derivation. Upgrades above the standard model
    // are also gated by `allow_upgrade`/`allow_degradation` in the provider's
    // `model_for_tier`, but the tier selection here only needs to ensure we
    // do not ask for a tier that would trigger derivation. The provider's
    // `model_for_tier` is responsible for honoring custom semantics.

    // Upgrades above the classified tier require explicit opt-in. This
    // prevents auto/custom routing from escalating routine work to Ultra.
    // Locked mode is exempt because it is an explicit user choice to pin to
    // the standard model.
    if mode != RoutingMode::Locked
        && tier_rank(selected) > tier_rank(classified_tier)
        && !allow_upgrade
    {
        selected = classified_tier;
        adjustment = RoutingAdjustment::UpgradeDisabled;
    }

    // Apply the configured quality floor. The floor never lowers a locked
    // pin below Pro; it can only raise a non-locked selection.
    let floor = overrides
        .minimum_tier
        .unwrap_or_else(|| policy.minimum_quality.min_tier());
    if tier_rank(selected) < tier_rank(floor) {
        selected = floor;
        adjustment = RoutingAdjustment::MinimumTierFloor;
    }

    // Apply the per-request maximum tier ceiling.
    if let Some(ceiling) = overrides.maximum_tier {
        if tier_rank(selected) > tier_rank(ceiling) {
            selected = ceiling;
            adjustment = RoutingAdjustment::MaximumTierCeiling;
        }
    }

    Ok(ResolvedTier {
        tier: selected,
        adjustment,
    })
}

const CLASSIFIER_MODEL_VERSION: u32 = 1;
const CLASSIFIER_FEATURE_COUNT: usize = 5;
const CLASSIFIER_TIER_COUNT: usize = 3;
const MIN_TRAINING_EXAMPLES: usize = 8;
const CLASSIFIER_TRAINING_EVENT_LIMIT: usize = 512;
const CLASSIFIER_CACHE_LIMIT: usize = 4096;
const CLASSIFIER_CACHE_TTL: Duration = Duration::from_secs(30 * 60);

/// Cheap, bounded request features used by the routing classifier.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ClassificationFeatures {
    pub token_count: u32,
    pub tool_count: u32,
    pub code_fence_density: f32,
    pub question_depth: u32,
}

impl ClassificationFeatures {
    pub fn from_request(request: &ChatCompletionRequest) -> Self {
        // Routing needs a complexity signal, not billing-grade tokenization.
        // Bound the scan to the most recent 64 KiB and use the same four-byte
        // heuristic as the fallback predictor. This removes a full transcript
        // allocation plus BPE pass from every completion.
        const MAX_CLASSIFIER_BYTES: u64 = 64 * 1024;
        let mut approximate_bytes = 0u64;
        'messages: for message in request.messages.iter().rev() {
            if let Some(content) = message.content_str() {
                let content = if message.role.eq_ignore_ascii_case("user") {
                    crate::user_content::session_user_request(content).unwrap_or_default()
                } else {
                    content
                };
                approximate_bytes = approximate_bytes
                    .saturating_add(content.len() as u64)
                    .min(MAX_CLASSIFIER_BYTES);
            } else {
                // Multipart messages: include text parts and attachment
                // data URL sizes so classification sees real complexity.
                let text = message.full_text();
                let content = if message.role.eq_ignore_ascii_case("user") {
                    crate::user_content::session_user_request(&text)
                        .unwrap_or_default()
                        .to_string()
                } else {
                    text
                };
                let total_len = message.content_len();
                approximate_bytes = approximate_bytes
                    .saturating_add(content.len() as u64)
                    .saturating_add(total_len.saturating_sub(content.len()) as u64 / 4)
                    .min(MAX_CLASSIFIER_BYTES);
            }
            for call in message.tool_calls.as_deref().unwrap_or_default() {
                approximate_bytes = approximate_bytes
                    .saturating_add(call.function.name.len() as u64)
                    .saturating_add(call.function.arguments.len() as u64)
                    .min(MAX_CLASSIFIER_BYTES);
                if approximate_bytes == MAX_CLASSIFIER_BYTES {
                    break 'messages;
                }
            }
            if approximate_bytes == MAX_CLASSIFIER_BYTES {
                break;
            }
        }
        let last_user_text = request
            .messages
            .iter()
            .rev()
            .filter(|message| message.role.eq_ignore_ascii_case("user"))
            .filter_map(|message| {
                // Handle both text-only and multipart user messages.
                if let Some(content) = message.content_str() {
                    crate::user_content::session_user_request(content).map(|s| s.to_string())
                } else {
                    let text = message.full_text();
                    if text.is_empty() {
                        None
                    } else {
                        crate::user_content::session_user_request(&text).map(|s| s.to_string())
                    }
                }
            })
            .next()
            .unwrap_or_default();
        let historical_tool_calls = request
            .messages
            .iter()
            .map(|message| message.tool_calls.as_deref().unwrap_or_default().len())
            .sum::<usize>();

        Self {
            token_count: approximate_bytes.div_ceil(4).min(u32::MAX as u64) as u32,
            tool_count: request
                .tools
                .as_deref()
                .unwrap_or_default()
                .len()
                .saturating_add(historical_tool_calls)
                .min(u32::MAX as usize) as u32,
            code_fence_density: code_fence_density(&last_user_text),
            question_depth: last_user_text
                .chars()
                .filter(|character| *character == '?')
                .count()
                .min(u32::MAX as usize) as u32,
        }
    }

    fn normalized(&self) -> [f32; CLASSIFIER_FEATURE_COUNT] {
        [
            1.0,
            (self.token_count as f32).ln_1p() / 4096.0_f32.ln_1p(),
            (self.tool_count.min(32) as f32) / 32.0,
            self.code_fence_density.clamp(0.0, 1.0),
            (self.question_depth.min(8) as f32) / 8.0,
        ]
    }
}

/// Serializable multinomial logistic model. Rows correspond to Lite, Pro,
/// and Ultra; columns are intercept, token count, tool count, code density,
/// and question depth.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LogisticRoutingModel {
    pub weights: [[f32; CLASSIFIER_FEATURE_COUNT]; CLASSIFIER_TIER_COUNT],
}

impl Default for LogisticRoutingModel {
    fn default() -> Self {
        Self {
            weights: [
                [2.0, -1.5, -0.75, -1.5, -1.0],
                [-0.5, 2.5, 1.0, 2.0, 2.0],
                [-2.0, 3.5, 1.75, 2.5, 2.5],
            ],
        }
    }
}

impl LogisticRoutingModel {
    pub fn probabilities(&self, features: &ClassificationFeatures) -> [f32; 3] {
        self.probabilities_from_normalized(features.normalized())
    }

    fn probabilities_from_normalized(
        &self,
        inputs: [f32; CLASSIFIER_FEATURE_COUNT],
    ) -> [f32; CLASSIFIER_TIER_COUNT] {
        let mut logits = [0.0_f32; CLASSIFIER_TIER_COUNT];

        for (tier, weights) in self.weights.iter().enumerate() {
            logits[tier] = weights
                .iter()
                .zip(inputs)
                .map(|(weight, input)| weight * input)
                .sum();
        }

        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = 0.0;

        for logit in &mut logits {
            *logit = (*logit - max_logit).exp();
            denominator += *logit;
        }

        if !denominator.is_finite() || denominator <= f32::EPSILON {
            return [1.0, 0.0, 0.0];
        }
        logits.map(|value| value / denominator)
    }

    fn fit(&mut self, examples: &[(ClassificationFeatures, ModelTier)]) {
        if examples.len() < MIN_TRAINING_EXAMPLES {
            return;
        }

        const EPOCHS: usize = 80;
        const LEARNING_RATE: f32 = 0.08;
        const L2_PENALTY: f32 = 0.01;

        for _ in 0..EPOCHS {
            let mut gradient = [[0.0_f32; CLASSIFIER_FEATURE_COUNT]; CLASSIFIER_TIER_COUNT];

            for (features, target) in examples {
                let inputs = features.normalized();
                let probabilities = self.probabilities(features);
                let target_index = model_tier_index(*target);
                for tier in 0..CLASSIFIER_TIER_COUNT {
                    let error = probabilities[tier] - f32::from(tier == target_index);
                    for feature in 0..CLASSIFIER_FEATURE_COUNT {
                        gradient[tier][feature] += error * inputs[feature];
                    }
                }
            }

            let scale = LEARNING_RATE / examples.len() as f32;

            for (tier, weights) in self.weights.iter_mut().enumerate() {
                for (feature, weight) in weights.iter_mut().enumerate() {
                    let regularization = if feature == 0 {
                        0.0
                    } else {
                        L2_PENALTY * *weight
                    };
                    *weight = (*weight - scale * (gradient[tier][feature] + regularization))
                        .clamp(-8.0, 8.0);
                }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ClassificationTelemetry {
    pub model_version: u32,
    pub features: ClassificationFeatures,
    pub probabilities: [f32; 3],
    pub predicted_tier: ModelTier,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation_signal: Option<String>,
}

/// Feature-based escalation classifier. A request is scored locally with a
/// multinomial logistic model; recognized validation feedback can safely
/// raise the current decision and trains future session-local predictions.
#[derive(Clone, Debug)]
pub struct EscalationClassifier {
    lite_model: String,
    pro_model: String,
    ultra_model: Option<String>,
    confidence_threshold: f32,
    model: LogisticRoutingModel,
    training_cache: Arc<Mutex<HashMap<Uuid, TrainingCacheEntry>>>,
}

#[derive(Clone, Debug)]
struct TrainingCacheEntry {
    latest_seq: i64,
    examples: Vec<(i64, ClassificationFeatures, ModelTier)>,
    model: LogisticRoutingModel,
    last_seen: Instant,
}

impl EscalationClassifier {
    pub fn new(
        lite_model: impl Into<String>,
        pro_model: impl Into<String>,
        ultra_model: Option<impl Into<String>>,
    ) -> Self {
        Self {
            lite_model: lite_model.into(),
            pro_model: pro_model.into(),
            ultra_model: ultra_model.map(Into::into),
            confidence_threshold: 0.85,
            model: LogisticRoutingModel::default(),
            training_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = if threshold.is_finite() {
            threshold.clamp(0.0, 1.0)
        } else {
            0.85
        };
        self
    }

    pub fn with_model(mut self, model: LogisticRoutingModel) -> Self {
        self.model = model;
        self.training_cache = Arc::new(Mutex::new(HashMap::new()));
        self
    }

    /// Derive a classifier from observations in one session's event log.
    /// Starting from the configured baseline on every call makes training
    /// deterministic and prevents one tenant's feedback from leaking into
    /// another tenant's routing model.
    pub fn trained_from_events(&self, events: &[SessionEvent]) -> Self {
        let examples: Vec<(ClassificationFeatures, ModelTier)> = events
            .iter()
            .filter_map(training_example)
            .map(|(_, features, target)| (features, target))
            .collect();
        let mut trained = self.clone();
        trained.model.fit(&examples);
        trained
    }

    pub async fn trained_for_session<E: EventStore>(
        &self,
        event_store: &E,
        session_id: Uuid,
    ) -> EngineResult<Self> {
        let latest = event_store.latest_seq(session_id).await?.unwrap_or(0);
        if latest == 0 {
            return Ok(self.clone());
        }
        let now = Instant::now();
        let cached = self
            .training_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&session_id)
            .filter(|entry| now.duration_since(entry.last_seen) <= CLASSIFIER_CACHE_TTL)
            .cloned();
        if let Some(entry) = &cached {
            if entry.latest_seq == latest {
                let mut trained = self.clone();
                trained.model = entry.model.clone();
                return Ok(trained);
            }
        }

        let window_start = latest
            .saturating_sub(CLASSIFIER_TRAINING_EVENT_LIMIT.saturating_sub(1) as i64)
            .max(1);
        let can_increment = cached
            .as_ref()
            .is_some_and(|entry| entry.latest_seq < latest && entry.latest_seq >= window_start - 1);
        let mut examples = if can_increment {
            cached
                .as_ref()
                .map(|entry| entry.examples.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let before_len = examples.len();
        examples.retain(|(seq, _, _)| *seq >= window_start);
        let read_since = if can_increment {
            cached
                .as_ref()
                .map(|entry| entry.latest_seq.saturating_add(1))
                .unwrap_or(window_start)
                .max(window_start)
        } else {
            window_start
        };
        let read_limit = latest
            .saturating_sub(read_since)
            .saturating_add(1)
            .min(CLASSIFIER_TRAINING_EVENT_LIMIT as i64) as usize;
        let events = event_store
            .read_events(session_id, Some(read_since), Some(read_limit))
            .await?;
        let new_examples: Vec<_> = events.iter().filter_map(training_example).collect();
        let training_changed =
            cached.is_none() || !new_examples.is_empty() || examples.len() != before_len;
        examples.extend(new_examples);

        let model = if training_changed {
            let mut model = self.model.clone();
            let fit_examples: Vec<_> = examples
                .iter()
                .map(|(_, features, target)| (features.clone(), *target))
                .collect();
            model.fit(&fit_examples);
            model
        } else {
            cached
                .as_ref()
                .map(|entry| entry.model.clone())
                .unwrap_or_else(|| self.model.clone())
        };

        let mut cache = self
            .training_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache.retain(|_, entry| now.duration_since(entry.last_seen) <= CLASSIFIER_CACHE_TTL);
        if cache.len() >= CLASSIFIER_CACHE_LIMIT {
            if let Some(oldest) = cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_seen)
                .map(|(session_id, _)| *session_id)
            {
                cache.remove(&oldest);
            }
        }
        cache.insert(
            session_id,
            TrainingCacheEntry {
                latest_seq: latest,
                examples,
                model: model.clone(),
                last_seen: now,
            },
        );

        let mut trained = self.clone();
        trained.model = model;
        Ok(trained)
    }

    /// Classify a request into a model tier.
    pub fn classify(
        &self,
        request: &ChatCompletionRequest,
        validation_signal: Option<&str>,
    ) -> EngineResult<RoutingDecision> {
        let last_user_text = request
            .messages
            .iter()
            .rev()
            .filter(|message| message.role.eq_ignore_ascii_case("user"))
            .filter_map(|message| {
                // Handle both text-only and multipart user messages.
                if let Some(content) = message.content_str() {
                    crate::user_content::session_user_request(content).map(|s| s.to_string())
                } else {
                    let text = message.full_text();
                    if text.is_empty() {
                        None
                    } else {
                        crate::user_content::session_user_request(&text).map(|s| s.to_string())
                    }
                }
            })
            .next()
            .unwrap_or_default();

        let features = ClassificationFeatures::from_request(request);
        let probabilities = self.model.probabilities(&features);

        let predicted_index = probabilities
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .unwrap_or(0);

        let predicted_tier = model_tier_from_index(predicted_index);
        let mut selected_tier = if predicted_tier == ModelTier::Ultra
            && probabilities[predicted_index] < self.confidence_threshold
        {
            ModelTier::Pro
        } else {
            predicted_tier
        };

        let mut reason = format!(
            "logistic classifier selected {} (p={:.3}; tokens={}, tools={}, code_fence_density={:.3}, question_depth={})",
            selected_tier.as_str(),
            probabilities[model_tier_index(selected_tier)],
            features.token_count,
            features.tool_count,
            features.code_fence_density,
            features.question_depth,
        );

        let explicitly_requests_ultra = last_user_text
            .split(|character: char| !character.is_alphanumeric())
            .any(|word| word.eq_ignore_ascii_case("ultra"));

        if explicitly_requests_ultra {
            selected_tier = ModelTier::Ultra;
            reason = "explicit ultra request".to_string();
        } else if let Some(target) =
            validation_signal.and_then(|signal| validation_target_tier(predicted_tier, signal))
        {
            if model_tier_index(target) > model_tier_index(selected_tier) {
                selected_tier = target;
                reason = format!(
                    "validation feedback raised logistic prediction from {} to {}",
                    predicted_tier.as_str(),
                    target.as_str()
                );
            }
        }

        let confidence = if explicitly_requests_ultra {
            1.0
        } else {
            probabilities[model_tier_index(selected_tier)]
        };

        Ok(RoutingDecision {
            model_tier: selected_tier,
            confidence,
            reason,
            classification: Some(ClassificationTelemetry {
                model_version: CLASSIFIER_MODEL_VERSION,
                features,
                probabilities,
                predicted_tier,
                validation_signal: validation_signal.map(sanitize_validation_signal),
            }),
        })
    }

    pub fn resolve_model_name(&self, decision: &RoutingDecision) -> EngineResult<String> {
        let name = match decision.model_tier {
            ModelTier::Lite => self.lite_model.clone(),
            ModelTier::Pro => self.pro_model.clone(),
            ModelTier::Ultra => self
                .ultra_model
                .clone()
                .unwrap_or_else(|| self.pro_model.clone()),
        };
        Ok(name)
    }
}

fn code_fence_density(text: &str) -> f32 {
    let total = text.chars().count();
    if total == 0 {
        return 0.0;
    }
    let mut inside_fence = false;
    let mut fenced_characters = 0_usize;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            inside_fence = !inside_fence;
        } else if inside_fence {
            fenced_characters = fenced_characters.saturating_add(line.chars().count() + 1);
        }
    }
    (fenced_characters as f32 / total as f32).clamp(0.0, 1.0)
}

fn sanitize_validation_signal(signal: &str) -> String {
    signal.trim().chars().take(256).collect()
}

fn training_example(event: &SessionEvent) -> Option<(i64, ClassificationFeatures, ModelTier)> {
    if event.event_type != SessionEventType::LlmCall {
        return None;
    }
    let telemetry = serde_json::from_value::<ClassificationTelemetry>(
        event.payload.get("classification")?.clone(),
    )
    .ok()?;
    if telemetry.model_version != CLASSIFIER_MODEL_VERSION {
        return None;
    }
    let target = validation_target_tier(
        telemetry.predicted_tier,
        telemetry.validation_signal.as_deref()?,
    )?;
    Some((event.seq, telemetry.features, target))
}

fn validation_target_tier(predicted_tier: ModelTier, signal: &str) -> Option<ModelTier> {
    let normalized = signal.trim().to_ascii_lowercase();
    for tier in [ModelTier::Lite, ModelTier::Pro, ModelTier::Ultra] {
        let tier_name = tier.as_str();
        if normalized.contains(&format!("target_tier={tier_name}"))
            || normalized.contains(&format!("target_tier:{tier_name}"))
            || normalized.contains(&format!("target_tier\":\"{tier_name}"))
        {
            return Some(tier);
        }
    }
    if [
        "low_confidence",
        "malformed",
        "contradiction",
        "validation_failed",
        "retry_required",
    ]
    .iter()
    .any(|marker| validation_has_marker(&normalized, marker))
    {
        return Some(match predicted_tier {
            ModelTier::Lite => ModelTier::Pro,
            ModelTier::Pro | ModelTier::Ultra => ModelTier::Ultra,
        });
    }
    if ["validated", "high_confidence", "correct", "success"]
        .iter()
        .any(|marker| validation_has_marker(&normalized, marker))
    {
        return Some(predicted_tier);
    }
    None
}

fn validation_has_marker(signal: &str, marker: &str) -> bool {
    signal
        .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .any(|token| token == marker)
}

fn model_tier_index(tier: ModelTier) -> usize {
    match tier {
        ModelTier::Lite => 0,
        ModelTier::Pro => 1,
        ModelTier::Ultra => 2,
    }
}

fn model_tier_from_index(index: usize) -> ModelTier {
    match index {
        1 => ModelTier::Pro,
        2 => ModelTier::Ultra,
        _ => ModelTier::Lite,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventStore;
    use crate::in_memory_stores::InMemoryEventStore;
    use crate::models::{ChatCompletionRequest, ChatMessage};
    use crate::MessageContent;

    #[test]
    fn classifier_feature_scan_is_bounded() {
        let request = ChatCompletionRequest {
            model: "router".to_string(),
            messages: vec![ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(crate::MessageContent::text("x".repeat(128 * 1024))),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        };
        assert_eq!(
            ClassificationFeatures::from_request(&request).token_count,
            16_384
        );
    }

    #[test]
    fn classifier_features_use_only_the_copilot_user_request() {
        let make_request = |content: &str| ChatCompletionRequest {
            model: "router".to_string(),
            messages: vec![ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(MessageContent::text(content)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        };
        let plain = make_request("What failed? Why?\n```rust\npanic!()\n```");
        let wrapped = make_request(
            "<context>lots of volatile editor context</context><reminderInstructions>many operational words here</reminderInstructions><userRequest>What failed? Why?\n```rust\npanic!()\n```</userRequest>",
        );

        assert_eq!(
            ClassificationFeatures::from_request(&plain),
            ClassificationFeatures::from_request(&wrapped)
        );
    }

    #[test]
    fn non_finite_confidence_threshold_uses_the_default() {
        let classifier = EscalationClassifier::new("lite", "pro", Some("ultra"))
            .with_confidence_threshold(f32::NAN);
        assert_eq!(classifier.confidence_threshold, 0.85);
    }

    #[tokio::test]
    async fn session_training_cache_advances_incrementally() {
        let classifier = EscalationClassifier::new("lite", "pro", Some("ultra"));
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        for expected_seq in 1..=2 {
            store
                .append_event(
                    session_id,
                    SessionEventType::LlmCall,
                    serde_json::json!({}),
                    None,
                )
                .await
                .unwrap();
            classifier
                .trained_for_session(&store, session_id)
                .await
                .unwrap();
            let cache = classifier
                .training_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let entry = cache.get(&session_id).expect("cached classifier state");
            assert_eq!(entry.latest_seq, expected_seq);
            assert!(entry.examples.is_empty());
        }
    }

    #[test]
    fn escalation_classifier_routes_routine_to_lite() {
        let classifier = EscalationClassifier::new("haiku", "sonnet", Some::<&str>("opus"));
        let req = ChatCompletionRequest {
            model: "reincore-router".to_string(),
            messages: vec![ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(crate::MessageContent::text("list files".to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        };

        let decision = classifier.classify(&req, None).unwrap();
        assert_eq!(decision.model_tier, ModelTier::Lite);
    }

    #[test]
    fn escalation_classifier_does_not_depend_on_complex_keywords() {
        let classifier = EscalationClassifier::new("haiku", "sonnet", None::<&str>);
        let req = ChatCompletionRequest {
            model: "reincore-router".to_string(),
            messages: vec![ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(crate::MessageContent::text(
                    "debug this failure".to_string(),
                )),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        };

        let decision = classifier.classify(&req, None).unwrap();
        assert_eq!(decision.model_tier, ModelTier::Lite);
        assert!(decision.reason.contains("logistic classifier"));
    }

    #[test]
    fn escalation_classifier_uses_structural_complexity_features() {
        let classifier = EscalationClassifier::new("haiku", "sonnet", None::<&str>);
        let fenced_code = "x += 1;\n".repeat(80);
        let req = ChatCompletionRequest {
            model: "reincore-router".to_string(),
            messages: vec![ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(MessageContent::text(format!(
                    "What failed? Why? Which invariant broke? How should it be fixed?\n```rust\n{fenced_code}```"
                ))),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        };

        let decision = classifier.classify(&req, None).unwrap();
        assert_ne!(decision.model_tier, ModelTier::Lite);
        let telemetry = decision.classification.unwrap();
        assert!(telemetry.features.code_fence_density > 0.5);
        assert_eq!(telemetry.features.question_depth, 4);
        assert!((telemetry.probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn validation_feedback_trains_future_predictions_upward() {
        let classifier = EscalationClassifier::new("haiku", "sonnet", None::<&str>);
        let req = ChatCompletionRequest {
            model: "reincore-router".to_string(),
            messages: vec![ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(crate::MessageContent::text(
                    "summarize this module".to_string(),
                )),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        };
        let baseline = classifier.classify(&req, None).unwrap();
        let telemetry = classifier
            .classify(&req, Some("low_confidence"))
            .unwrap()
            .classification
            .unwrap();
        let events: Vec<SessionEvent> = (1..=MIN_TRAINING_EXAMPLES as i64)
            .map(|seq| SessionEvent {
                session_id: uuid::Uuid::nil(),
                seq,
                event_type: SessionEventType::LlmCall,
                payload: serde_json::json!({"classification": telemetry}),
                cost_cents: None,
                created_at: chrono::Utc::now(),
            })
            .collect();

        let trained = classifier.trained_from_events(&events);
        let trained_decision = trained.classify(&req, None).unwrap();
        let baseline_pro = baseline.classification.unwrap().probabilities[1];
        let trained_pro = trained_decision.classification.unwrap().probabilities[1];

        assert!(trained_pro > baseline_pro);
        assert!(trained_pro.is_finite());
    }

    #[test]
    #[ignore = "manual microbenchmark; run with --release --ignored --nocapture"]
    fn benchmark_fit_normalize_once() {
        use std::hint::black_box;
        use std::time::Instant;

        fn fit_pre_normalized(
            model: &mut LogisticRoutingModel,
            examples: &[(ClassificationFeatures, ModelTier)],
        ) {
            let normalized: Vec<([f32; CLASSIFIER_FEATURE_COUNT], ModelTier)> = examples
                .iter()
                .map(|(features, target)| (features.normalized(), *target))
                .collect();

            for _ in 0..80 {
                let mut gradient = [[0.0_f32; CLASSIFIER_FEATURE_COUNT]; CLASSIFIER_TIER_COUNT];
                for (inputs, target) in &normalized {
                    let probabilities = model.probabilities_from_normalized(*inputs);
                    let target_index = model_tier_index(*target);
                    for tier in 0..CLASSIFIER_TIER_COUNT {
                        let error = probabilities[tier] - f32::from(tier == target_index);
                        for feature in 0..CLASSIFIER_FEATURE_COUNT {
                            gradient[tier][feature] += error * inputs[feature];
                        }
                    }
                }

                let scale = 0.08 / normalized.len() as f32;
                for (tier, weights) in model.weights.iter_mut().enumerate() {
                    for (feature, weight) in weights.iter_mut().enumerate() {
                        let regularization = if feature == 0 { 0.0 } else { 0.01 * *weight };
                        *weight = (*weight - scale * (gradient[tier][feature] + regularization))
                            .clamp(-8.0, 8.0);
                    }
                }
            }
        }

        let examples: Vec<(ClassificationFeatures, ModelTier)> = (0
            ..CLASSIFIER_TRAINING_EVENT_LIMIT)
            .map(|index| {
                (
                    ClassificationFeatures {
                        token_count: 32 + (index % 4096) as u32,
                        tool_count: (index % 12) as u32,
                        code_fence_density: (index % 10) as f32 / 10.0,
                        question_depth: (index % 8) as u32,
                    },
                    model_tier_from_index(index % CLASSIFIER_TIER_COUNT),
                )
            })
            .collect();

        let mut current = LogisticRoutingModel::default();
        let mut pre_normalized = LogisticRoutingModel::default();
        current.fit(&examples);
        fit_pre_normalized(&mut pre_normalized, &examples);
        assert_eq!(current.weights, pre_normalized.weights);

        let iterations = if cfg!(debug_assertions) { 10 } else { 100 };
        let started = Instant::now();
        for _ in 0..iterations {
            let mut model = LogisticRoutingModel::default();
            model.fit(black_box(&examples));
            black_box(model.weights);
        }
        let current_duration = started.elapsed();

        let started = Instant::now();
        for _ in 0..iterations {
            let mut model = LogisticRoutingModel::default();
            fit_pre_normalized(&mut model, black_box(&examples));
            black_box(model.weights);
        }
        let pre_normalized_duration = started.elapsed();

        println!(
            "fit over {} examples, {iterations} iterations: current={current_duration:?}, normalize_once={pre_normalized_duration:?}, speedup={:.2}x",
            examples.len(),
            current_duration.as_secs_f64() / pre_normalized_duration.as_secs_f64()
        );
    }

    #[test]
    fn training_ignores_malformed_and_unknown_version_observations() {
        let classifier = EscalationClassifier::new("haiku", "sonnet", None::<&str>);
        let events = vec![
            SessionEvent {
                session_id: uuid::Uuid::nil(),
                seq: 1,
                event_type: SessionEventType::LlmCall,
                payload: serde_json::json!({"classification": "not-an-object"}),
                cost_cents: None,
                created_at: chrono::Utc::now(),
            },
            SessionEvent {
                session_id: uuid::Uuid::nil(),
                seq: 2,
                event_type: SessionEventType::LlmCall,
                payload: serde_json::json!({
                    "classification": {
                        "model_version": 999,
                        "features": {
                            "token_count": 10,
                            "tool_count": 0,
                            "code_fence_density": 0.0,
                            "question_depth": 0
                        },
                        "probabilities": [1.0, 0.0, 0.0],
                        "predicted_tier": "lite",
                        "validation_signal": "low_confidence"
                    }
                }),
                cost_cents: None,
                created_at: chrono::Utc::now(),
            },
        ];

        assert_eq!(
            classifier.trained_from_events(&events).model,
            classifier.model
        );
    }

    #[tokio::test]
    async fn classifier_trains_from_persisted_session_events() {
        let classifier = EscalationClassifier::new("haiku", "sonnet", None::<&str>);
        let store = crate::in_memory_stores::InMemoryEventStore::default();
        let session_id = uuid::Uuid::new_v4();
        let req = ChatCompletionRequest {
            model: "reincore-router".to_string(),
            messages: vec![ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(crate::MessageContent::text(
                    "summarize this module".to_string(),
                )),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        };
        let feedback = classifier
            .classify(&req, Some("low_confidence"))
            .unwrap()
            .classification
            .unwrap();
        for _ in 0..MIN_TRAINING_EXAMPLES {
            store
                .append_event(
                    session_id,
                    SessionEventType::LlmCall,
                    serde_json::json!({"classification": feedback}),
                    None,
                )
                .await
                .unwrap();
        }

        let baseline = classifier.classify(&req, None).unwrap();
        let trained = classifier
            .trained_for_session(&store, session_id)
            .await
            .unwrap()
            .classify(&req, None)
            .unwrap();

        assert!(
            trained.classification.unwrap().probabilities[1]
                > baseline.classification.unwrap().probabilities[1]
        );
    }

    #[test]
    fn validation_markers_do_not_match_substrings() {
        assert_eq!(
            validation_target_tier(ModelTier::Lite, "unsuccessful"),
            None
        );
        assert_eq!(
            validation_target_tier(ModelTier::Lite, "status=success"),
            Some(ModelTier::Lite)
        );
    }

    #[test]
    fn explicit_ultra_classification_does_not_require_global_ultra_model() {
        let classifier = EscalationClassifier::new("haiku", "sonnet", None::<&str>);
        let req = ChatCompletionRequest {
            model: "reincore-router".to_string(),
            messages: vec![ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(crate::MessageContent::text(
                    "use ultra quality for this synthesis".to_string(),
                )),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        };

        let decision = classifier.classify(&req, None).unwrap();
        assert_eq!(decision.model_tier, ModelTier::Ultra);
        // The legacy global resolver remains backward-compatible by falling
        // back to Pro when no global Ultra name was supplied.
        assert_eq!(classifier.resolve_model_name(&decision).unwrap(), "sonnet");
    }

    fn policy(mode: RoutingMode) -> RoutingPolicy {
        RoutingPolicy {
            mode,
            ..RoutingPolicy::default()
        }
    }

    #[test]
    fn auto_mode_honors_budget_degradation() {
        let resolved = resolve_routing_tier(
            &policy(RoutingMode::Auto),
            &RequestRoutingOverrides::default(),
            ModelTier::Pro,
            ModelTier::Lite,
        )
        .unwrap();
        assert_eq!(resolved.tier, ModelTier::Lite);
        assert_eq!(resolved.adjustment, RoutingAdjustment::None);
    }

    #[test]
    fn locked_mode_pins_to_standard_tier() {
        let resolved = resolve_routing_tier(
            &policy(RoutingMode::Locked),
            &RequestRoutingOverrides::default(),
            ModelTier::Lite,
            ModelTier::Lite,
        )
        .unwrap();
        assert_eq!(resolved.tier, ModelTier::Pro);
        assert_eq!(resolved.adjustment, RoutingAdjustment::LockedToStandard);
    }

    #[test]
    fn locked_mode_keeps_pro_without_reporting_adjustment() {
        let resolved = resolve_routing_tier(
            &policy(RoutingMode::Locked),
            &RequestRoutingOverrides::default(),
            ModelTier::Pro,
            ModelTier::Pro,
        )
        .unwrap();
        assert_eq!(resolved.tier, ModelTier::Pro);
        assert_eq!(resolved.adjustment, RoutingAdjustment::None);
    }

    #[test]
    fn disable_degradation_ignores_budget_tier() {
        let mut p = policy(RoutingMode::Auto);
        p.allow_degradation = false;
        let resolved = resolve_routing_tier(
            &p,
            &RequestRoutingOverrides::default(),
            ModelTier::Pro,
            ModelTier::Lite,
        )
        .unwrap();
        assert_eq!(resolved.tier, ModelTier::Pro);
        assert_eq!(resolved.adjustment, RoutingAdjustment::DegradationDisabled);
    }

    #[test]
    fn upgrades_above_classified_tier_require_opt_in() {
        // Budget never upgrades, so simulate an upgrade by classifying Lite
        // but requesting Ultra via the configured minimum-quality floor.
        let mut p = policy(RoutingMode::Auto);
        p.minimum_quality = MinimumQuality::High;
        let resolved = resolve_routing_tier(
            &p,
            &RequestRoutingOverrides::default(),
            ModelTier::Lite,
            ModelTier::Lite,
        )
        .unwrap();
        // The floor raises Lite to Ultra.
        assert_eq!(resolved.tier, ModelTier::Ultra);
        assert_eq!(resolved.adjustment, RoutingAdjustment::MinimumTierFloor);
    }

    #[test]
    fn request_override_mode_wins_over_configured_mode() {
        let overrides = RequestRoutingOverrides {
            mode: Some(RoutingMode::Locked),
            minimum_tier: None,
            maximum_tier: None,
        };
        let resolved = resolve_routing_tier(
            &policy(RoutingMode::Auto),
            &overrides,
            ModelTier::Ultra,
            ModelTier::Ultra,
        )
        .unwrap();
        assert_eq!(resolved.tier, ModelTier::Pro);
        assert_eq!(resolved.adjustment, RoutingAdjustment::LockedToStandard);
    }

    #[test]
    fn request_minimum_tier_raises_selection() {
        let overrides = RequestRoutingOverrides {
            mode: None,
            minimum_tier: Some(ModelTier::Pro),
            maximum_tier: None,
        };
        let resolved = resolve_routing_tier(
            &policy(RoutingMode::Auto),
            &overrides,
            ModelTier::Lite,
            ModelTier::Lite,
        )
        .unwrap();
        assert_eq!(resolved.tier, ModelTier::Pro);
        assert_eq!(resolved.adjustment, RoutingAdjustment::MinimumTierFloor);
    }

    #[test]
    fn request_maximum_tier_lowers_selection() {
        let overrides = RequestRoutingOverrides {
            mode: None,
            minimum_tier: None,
            maximum_tier: Some(ModelTier::Pro),
        };
        let resolved = resolve_routing_tier(
            &policy(RoutingMode::Auto),
            &overrides,
            ModelTier::Ultra,
            ModelTier::Ultra,
        )
        .unwrap();
        assert_eq!(resolved.tier, ModelTier::Pro);
        assert_eq!(resolved.adjustment, RoutingAdjustment::MaximumTierCeiling);
    }

    #[test]
    fn request_overrides_round_trip_through_extra_map() {
        let mut extra = std::collections::HashMap::new();
        extra.insert(
            "reincore_routing_mode".to_string(),
            serde_json::json!("locked"),
        );
        extra.insert(
            "reincore_minimum_tier".to_string(),
            serde_json::json!("pro"),
        );
        extra.insert(
            "reincore_maximum_tier".to_string(),
            serde_json::json!("ultra"),
        );
        let overrides = RequestRoutingOverrides::from_extra(&extra);
        assert_eq!(overrides.mode, Some(RoutingMode::Locked));
        assert_eq!(overrides.minimum_tier, Some(ModelTier::Pro));
        assert_eq!(overrides.maximum_tier, Some(ModelTier::Ultra));
    }

    #[test]
    fn unknown_routing_mode_override_is_ignored() {
        let mut extra = std::collections::HashMap::new();
        extra.insert(
            "reincore_routing_mode".to_string(),
            serde_json::json!("turbo"),
        );
        let overrides = RequestRoutingOverrides::from_extra(&extra);
        assert_eq!(overrides.mode, None);
    }

    #[test]
    fn routing_mode_parses_known_strings() {
        assert_eq!("auto".parse::<RoutingMode>().unwrap(), RoutingMode::Auto);
        assert_eq!(
            "CUSTOM".parse::<RoutingMode>().unwrap(),
            RoutingMode::Custom
        );
        assert_eq!(
            "Locked".parse::<RoutingMode>().unwrap(),
            RoutingMode::Locked
        );
        assert!("fast".parse::<RoutingMode>().is_err());
    }

    #[test]
    fn routing_policy_from_raw_falls_back_on_malformed_json() {
        let policy = RoutingPolicy::from_raw(Some(&serde_json::json!({"mode": "nonsense"})));
        assert_eq!(policy.mode, RoutingMode::Auto);
        let none_policy = RoutingPolicy::from_raw(None);
        assert_eq!(none_policy, RoutingPolicy::default());
    }
}
