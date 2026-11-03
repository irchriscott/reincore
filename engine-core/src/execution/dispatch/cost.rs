//! Pre-flight cost prediction for LLM requests.
//!
//! [`CostPredictor`] estimates the dollar cost of a chat-completion request
//! *before* it is sent, using per-million-token pricing from the
//! [`model_library`](crate::model_library) catalog and a shared cl100k BPE
//! tokenizer for accurate token counts. The result is a low–high cost range
//! (in cents) that the budget tier state machine consumes to decide whether
//! the session should be degraded or checkpointed. A standalone
//! [`count_text_tokens`] helper is available for ad-hoc token counting.

use crate::error::EngineResult;
use crate::model_library::{
    default_model_for_provider_and_tier, model_catalog, ProviderModelRecord,
};
use crate::models::{ChatCompletionRequest, ChatMessage, ModelTier, ToolDefinition};
use crate::provider::TokenUsage;
use crate::user_content::session_user_request;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, OnceLock};
use tiktoken_rs::CoreBPE;

/// Process-wide shared cl100k tokenizer. Building a BPE encoder is
/// expensive (vocabulary parsing), so it is constructed exactly once and
/// shared by every `CostPredictor` and by [`count_text_tokens`].
fn shared_tokenizer() -> Option<Arc<CoreBPE>> {
    static TOKENIZER: OnceLock<Option<Arc<CoreBPE>>> = OnceLock::new();
    TOKENIZER
        .get_or_init(|| tiktoken_rs::cl100k_base().ok().map(Arc::new))
        .clone()
}

const TOKEN_COUNT_CACHE_ENTRIES: usize = 4096;

/// Counts only: never retain text or token IDs from prompts, files or secrets.
/// Each cache belongs to one encoding. Release its lock before tokenization so
/// unrelated sessions do not wait behind a large cache miss.
#[derive(Default)]
struct TokenCountCache {
    counts: HashMap<[u8; 32], u64>,
    insertion_order: VecDeque<[u8; 32]>,
}

fn shared_token_count_cache() -> Arc<Mutex<TokenCountCache>> {
    static CACHE: OnceLock<Arc<Mutex<TokenCountCache>>> = OnceLock::new();
    CACHE
        .get_or_init(|| Arc::new(Mutex::new(TokenCountCache::default())))
        .clone()
}

fn cached_bpe_count(cache: &Mutex<TokenCountCache>, bpe: &CoreBPE, text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    let key: [u8; 32] = Sha256::digest(text.as_bytes()).into();
    let existing = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .counts
        .get(&key)
        .copied();
    if let Some(count) = existing {
        return count;
    }
    let count = bpe.encode_ordinary(text).len() as u64;
    let mut cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !cache.counts.contains_key(&key) {
        if cache.counts.len() >= TOKEN_COUNT_CACHE_ENTRIES {
            if let Some(oldest) = cache.insertion_order.pop_front() {
                cache.counts.remove(&oldest);
            }
        }
        cache.counts.insert(key, count);
        cache.insertion_order.push_back(key);
    }
    count
}

/// Count tokens in a piece of text without constructing a `CostPredictor`.
/// Uses the shared BPE tokenizer when available and a 4-chars-per-token
/// heuristic otherwise.
pub fn count_text_tokens(text: &str) -> u64 {
    match shared_tokenizer() {
        Some(bpe) => cached_bpe_count(&shared_token_count_cache(), &bpe, text),
        None => (text.len() as u64).div_ceil(4),
    }
}

/// Minimum input tokens assumed for any request, even a trivially short one.
/// Covers system-prompt / formatting overhead that isn't in the raw char count.
const MIN_INPUT_TOKENS: u64 = 100;

/// Flat per-tool overhead in tokens, representing tool-call formatting
/// (name framing, function-call wrapper tokens) *beyond* the raw JSON
/// schema, which is tokenized separately as part of the request text.
/// If you ever see this producing double-counted estimates, this is
/// intentional additive overhead, not a duplicate of the schema tokens.
const DEFAULT_TOOL_OVERHEAD_TOKENS: u64 = 50;

/// Flat cost estimate (in cents) for a single heavy-modal (e.g. image)
/// generation, used because those aren't priced per output token the
/// way text completions are.
const HEAVY_MODAL_FLAT_CENTS: i64 = 4;

/// A per-request cost prediction expressed as a range.
///
/// The low bound assumes minimal output and no retries; the high bound
/// assumes a longer response and one retry/escalation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CostPrediction {
    /// Lower bound of the predicted cost, in cents.
    pub low_cents: i64,
    /// Upper bound of the predicted cost, in cents.
    pub high_cents: i64,
    /// Estimated input tokens for this request.
    pub input_tokens: u64,
    /// Estimated output tokens (range midpoint).
    pub output_tokens: u64,
    /// Model tier used for the prediction.
    pub model_tier: ModelTier,
    /// Reasoning behind the prediction, for transparency.
    pub reasoning: String,
}

impl CostPrediction {
    /// Format the range as a human-readable dollar string.
    pub fn range_dollars(&self) -> String {
        format!(
            "${:.2} - ${:.2}",
            self.low_cents as f64 / 100.0,
            self.high_cents as f64 / 100.0
        )
    }
}

/// Scale factor used by [`cost_accuracy_bps`] - one hundredth of a percent,
/// the conventional granularity for ratios in finance and routing metrics.
pub const COST_ACCURACY_BPS_SCALE: i64 = 10_000;

/// Scale for cache-hit-rate basis points: `10_000` = 100%.
pub const CACHE_HIT_RATE_BPS_SCALE: i64 = 10_000;

/// Compute the per-call cache-prefix hit rate in basis points.
///
/// `cached_prompt_tokens / prompt_tokens * 10_000`, rounded to the nearest
/// basis point. Returns `None` when `prompt_tokens` is zero (no input to
/// cache, e.g. a degenerate request) so the aggregation loop skips
/// unmeasurable calls.
pub fn cache_hit_rate_bps(cached_prompt_tokens: u64, prompt_tokens: u64) -> Option<i64> {
    if prompt_tokens == 0 {
        return None;
    }
    let ratio =
        (cached_prompt_tokens as f64 / prompt_tokens as f64) * CACHE_HIT_RATE_BPS_SCALE as f64;
    if ratio.is_finite() {
        Some(ratio.round().clamp(0.0, CACHE_HIT_RATE_BPS_SCALE as f64) as i64)
    } else {
        None
    }
}

/// Compute the post-completion calibration ratio between an estimated actual
/// cost and the pre-flight prediction's high bound, expressed in basis
/// points (`10_000` = prediction matched exactly; `5_000` = actual was half
/// the prediction; `20_000` = actual was double the prediction).
///
/// Returns `None` when the prediction was zero or absent, which happens for
/// free providers (e.g. local Ollama) and for sessions created before the
/// cost-calibration migration. Callers should treat `None` as "not
/// calibratable" and skip the sample rather than recording a misleading
/// infinity / NaN.
pub fn actual_to_predicted_high_ratio_bps(
    actual_cents: i64,
    predicted_high_cents: i64,
) -> Option<i64> {
    if predicted_high_cents <= 0 || actual_cents < 0 {
        return None;
    }
    // actual / predicted * 10_000, rounded to the nearest basis point.
    // Invalid negative amounts are excluded above.
    let ratio =
        (actual_cents as f64 / predicted_high_cents as f64) * COST_ACCURACY_BPS_SCALE as f64;
    if ratio.is_finite() {
        Some(ratio.round() as i64)
    } else {
        None
    }
}

/// Legacy function name. Returns a ratio, not accuracy or error magnitude.
pub fn cost_accuracy_bps(actual_cents: i64, predicted_high_cents: i64) -> Option<i64> {
    actual_to_predicted_high_ratio_bps(actual_cents, predicted_high_cents)
}

/// Per-model pricing in cents per 1M tokens.
#[derive(Clone, Debug)]
pub struct ModelPricing {
    pub input_cents_per_1m: f64,
    pub output_cents_per_1m: f64,
}

impl ModelPricing {
    /// Build pricing from a catalog record. Prices in the catalog are USD per
    /// 1M tokens; convert to cents.
    pub fn from_record(record: &ProviderModelRecord) -> Self {
        let (input, output) = record.current_text_prices();
        Self {
            input_cents_per_1m: input * 100.0,
            output_cents_per_1m: output * 100.0,
        }
    }

    /// Fallback pricing used when no provider/model mapping is available.
    pub fn lite() -> Self {
        Self {
            input_cents_per_1m: 30.0,
            output_cents_per_1m: 60.0,
        }
    }

    pub fn pro() -> Self {
        Self {
            input_cents_per_1m: 300.0,
            output_cents_per_1m: 1_500.0,
        }
    }

    pub fn ultra() -> Self {
        Self {
            input_cents_per_1m: 1_500.0,
            output_cents_per_1m: 7_500.0,
        }
    }

    pub fn for_tier(tier: ModelTier) -> Self {
        match tier {
            ModelTier::Lite => Self::lite(),
            ModelTier::Pro => Self::pro(),
            ModelTier::Ultra => Self::ultra(),
        }
    }
}

/// What kind of work the request is expected to perform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskCategory {
    /// Read-only tool selection / classification (lite, deterministic).
    ToolSelection,
    /// Code generation, refactoring, or math.
    Code,
    /// A substantial non-code deliverable such as a report, proposal,
    /// presentation, campaign, lesson plan, contract, or spreadsheet.
    Artifact,
    /// Long-form reasoning, planning, analysis.
    Reasoning,
    /// General chat / text generation.
    Text,
    /// Image generation or other heavy modal output.
    HeavyModal,
}

/// Configurable cost predictor.
///
/// Holds pricing tables and heuristics so callers can override defaults
/// without changing global state.
#[derive(Clone)]
pub struct CostPredictor {
    lite_pricing: ModelPricing,
    pro_pricing: ModelPricing,
    ultra_pricing: ModelPricing,
    /// Optional provider label used to resolve catalog pricing. When `None`
    /// the predictor falls back to tier averages.
    provider: Option<String>,
    /// Concrete model selected by the provider's hybrid policy. When set,
    /// this takes precedence over provider-wide tier defaults for catalog
    /// pricing and transparent prediction labels.
    model: Option<String>,
    /// When `true`, the provider is treated as free (no per-token pricing
    /// available, e.g. local Ollama). `predict()` returns a zero-cost
    /// prediction and `pricing_for_tier` returns zero pricing, so the
    /// budget-degradation state machine has nothing to degrade through.
    free_provider: bool,
    /// Default chars-per-token heuristic, used for any category without a
    /// more specific override (see `chars_per_token_for`).
    chars_per_token: usize,
    /// Chars-per-token heuristic specifically for code content, which tends
    /// to tokenize denser than prose due to punctuation/symbols.
    /// Used only as a fallback when `tokenizer` is unavailable.
    code_chars_per_token: usize,
    /// Real BPE tokenizer, used for accurate input-token counts instead of
    /// the chars-per-token heuristic whenever it's available. `None` means
    /// the encoder failed to build (should be rare) and we fall back to the
    /// char heuristic rather than panicking on a cost-estimation path.
    tokenizer: Option<Arc<CoreBPE>>,
    token_count_cache: Arc<Mutex<TokenCountCache>>,
    tool_overhead_tokens: u64,
    heavy_modal_flat_cents: i64,
    read_only_tools: Vec<String>,
    code_signals: Vec<String>,
    artifact_signals: Vec<String>,
    reasoning_signals: Vec<String>,
    modal_signals: Vec<String>,
}

impl Default for CostPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl CostPredictor {
    /// Create a predictor with built-in default pricing and heuristics.
    pub fn new() -> Self {
        Self {
            lite_pricing: ModelPricing::lite(),
            pro_pricing: ModelPricing::pro(),
            ultra_pricing: ModelPricing::ultra(),
            provider: None,
            model: None,
            free_provider: false,
            chars_per_token: 4,
            code_chars_per_token: 3,
            // cl100k_base is the encoding used by GPT-4-class / most modern
            // chat models; close enough across providers for cost *estimation*
            // purposes even when the exact target model uses a different
            // encoding. Shared process-wide so it is built exactly once.
            tokenizer: shared_tokenizer(),
            token_count_cache: shared_token_count_cache(),
            tool_overhead_tokens: DEFAULT_TOOL_OVERHEAD_TOKENS,
            heavy_modal_flat_cents: HEAVY_MODAL_FLAT_CENTS,
            read_only_tools: vec![
                "read_file".to_string(),
                "list_dir".to_string(),
                "grep_search".to_string(),
                "fetch_webpage".to_string(),
                "search".to_string(),
                "lookup".to_string(),
                "query".to_string(),
                "retrieve".to_string(),
            ],
            code_signals: vec![
                "code".to_string(),
                "function".to_string(),
                "refactor".to_string(),
                "implement".to_string(),
                "bug".to_string(),
                "test".to_string(),
                "algorithm".to_string(),
                "math".to_string(),
                "equation".to_string(),
            ],
            artifact_signals: vec![
                "brief".to_string(),
                "campaign".to_string(),
                "contract".to_string(),
                "curriculum".to_string(),
                "dashboard".to_string(),
                "email".to_string(),
                "forecast".to_string(),
                "invoice".to_string(),
                "itinerary".to_string(),
                "lesson plan".to_string(),
                "marketing copy".to_string(),
                "memo".to_string(),
                "policy".to_string(),
                "presentation".to_string(),
                "proposal".to_string(),
                "report".to_string(),
                "spreadsheet".to_string(),
                "survey".to_string(),
                "transcript".to_string(),
                "workflow".to_string(),
            ],
            reasoning_signals: vec![
                "plan".to_string(),
                "reason".to_string(),
                "analyze".to_string(),
                "analyse".to_string(),
                "assess".to_string(),
                "audit".to_string(),
                "compare".to_string(),
                "decide".to_string(),
                "evaluate".to_string(),
                "forecast".to_string(),
                "investigate".to_string(),
                "recommend".to_string(),
                "research".to_string(),
                "synthesize".to_string(),
                "debug".to_string(),
                "architecture".to_string(),
                "design".to_string(),
            ],
            modal_signals: vec![
                "draw".to_string(),
                "illustrate".to_string(),
                "generate image".to_string(),
                "create image".to_string(),
                "create illustration".to_string(),
                "generate video".to_string(),
                "create video".to_string(),
                "generate audio".to_string(),
                "create audio".to_string(),
                "compose music".to_string(),
                "text to speech".to_string(),
                "voiceover".to_string(),
            ],
        }
    }

    /// Override pricing for a given tier.
    pub fn with_pricing(mut self, tier: ModelTier, pricing: ModelPricing) -> Self {
        match tier {
            ModelTier::Lite => self.lite_pricing = pricing,
            ModelTier::Pro => self.pro_pricing = pricing,
            ModelTier::Ultra => self.ultra_pricing = pricing,
        }
        self
    }

    /// Set the provider label so the predictor can look up catalog pricing
    /// for the concrete model selected for each tier.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Set the concrete model selected by routing.
    ///
    /// Call this with the result of `ProviderConfig::model_for_tier` so cost
    /// prediction and actual-cost accounting use the same standard model or
    /// explicit tier override that will be sent to the provider.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Mark this predictor as backing a free provider (e.g. local Ollama)
    /// whose models have no known per-token pricing. `predict()` will
    /// return a zero-cost prediction so the budget-degradation state
    /// machine has nothing to degrade through.
    pub fn with_free_provider(mut self, free: bool) -> Self {
        self.free_provider = free;
        self
    }

    /// Return whether this predictor is configured for a free provider.
    pub fn is_free(&self) -> bool {
        self.free_provider
    }

    /// Return the currently configured provider label, if any.
    pub fn provider(&self) -> Option<&str> {
        self.provider.as_deref()
    }

    /// Return the concrete model override, if one was configured.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// Override the default characters-per-token heuristic (non-code content).
    pub fn with_chars_per_token(mut self, chars_per_token: usize) -> Self {
        self.chars_per_token = chars_per_token.max(1);
        self
    }

    /// Override the code-specific characters-per-token heuristic.
    pub fn with_code_chars_per_token(mut self, chars_per_token: usize) -> Self {
        self.code_chars_per_token = chars_per_token.max(1);
        self
    }

    /// Override the per-tool overhead in tokens.
    pub fn with_tool_overhead(mut self, overhead: u64) -> Self {
        self.tool_overhead_tokens = overhead;
        self
    }

    /// Inject a specific tokenizer (e.g. a non-cl100k encoding for a
    /// provider whose models tokenize differently). Pass `None` to force
    /// the char-heuristic fallback, which is mainly useful for testing.
    pub fn with_tokenizer(mut self, tokenizer: Option<Arc<CoreBPE>>) -> Self {
        self.tokenizer = tokenizer;
        // Never reuse a count obtained with a different/custom encoding.
        self.token_count_cache = Arc::new(Mutex::new(TokenCountCache::default()));
        self
    }

    /// Override the flat cost (in cents) assumed for heavy-modal requests.
    pub fn with_heavy_modal_flat_cents(mut self, cents: i64) -> Self {
        self.heavy_modal_flat_cents = cents;
        self
    }

    /// Predict the cost of a single chat completion request.
    ///
    /// The prediction is intentionally a range: the low bound uses the
    /// requested `max_tokens` (or a default short output) and no retries.
    /// The high bound represents a worst-plausible-case scenario - the
    /// longest output the request could actually produce, with a full
    /// extra pass applied if the category is prone to retry/escalation.
    ///
    /// When `provider` is supplied, pricing is resolved from the catalog for
    /// that provider's default model at the requested tier.
    pub fn predict(
        &self,
        request: &ChatCompletionRequest,
        tier: ModelTier,
    ) -> EngineResult<CostPrediction> {
        let category = self.categorize(request);
        let input_tokens = self.estimate_input_tokens(request, category);
        Ok(self.finish_prediction(request, tier, category, input_tokens))
    }

    /// Build a cost prediction from an input-token count already computed by
    /// the context-window enforcer. This avoids re-tokenizing the same prompt
    /// during final completion preflight.
    pub fn predict_with_input_tokens(
        &self,
        request: &ChatCompletionRequest,
        tier: ModelTier,
        input_tokens: u64,
    ) -> EngineResult<CostPrediction> {
        let category = self.categorize(request);
        Ok(self.finish_prediction(request, tier, category, input_tokens))
    }

    fn finish_prediction(
        &self,
        request: &ChatCompletionRequest,
        tier: ModelTier,
        category: TaskCategory,
        input_tokens: u64,
    ) -> CostPrediction {
        let (low_output, high_output) = self.estimate_output_range(request, category);

        // Free providers (e.g. local Ollama) have no known per-token pricing,
        // so report a zero-cost prediction. The budget-degradation state
        // machine keys off spend; with zero spend it never advances tiers.
        if self.free_provider {
            let model_name = self.model_name_for_tier(tier);
            let reasoning = format!(
                "category={:?}, input_tokens={}, output_range={}-{}, model={}, tier={}, free_provider=true (no per-token pricing; degradation disabled)",
                category, input_tokens, low_output, high_output, model_name, tier.as_str()
            );
            return CostPrediction {
                low_cents: 0,
                high_cents: 0,
                input_tokens,
                output_tokens: (low_output + high_output) / 2,
                model_tier: tier,
                reasoning,
            };
        }

        let pricing = self.pricing_for_tier(tier);
        let retry_multiplier = self.retry_multiplier(category);

        let (low_cents, high_cents) = if category == TaskCategory::HeavyModal {
            let low = self.heavy_modal_flat_cents
                + tokens_to_cents(input_tokens, pricing.input_cents_per_1m);
            let high = ((low as f64) * retry_multiplier).ceil() as i64;
            (low, high.max(low))
        } else {
            let low_cents = tokens_to_cents(input_tokens, pricing.input_cents_per_1m)
                + tokens_to_cents(low_output, pricing.output_cents_per_1m);

            // High bound: the same request, but with the longer plausible
            // output. Retries are modeled as a multiplier over the *entire*
            // cost (input + output), since a retry resends the full input
            // and generates a full new output - not just extra input tokens.
            let base_high_cents = tokens_to_cents(input_tokens, pricing.input_cents_per_1m)
                + tokens_to_cents(high_output, pricing.output_cents_per_1m);
            let high_cents = ((base_high_cents as f64) * retry_multiplier).ceil() as i64;

            (low_cents, high_cents.max(low_cents))
        };

        let model_name = self.model_name_for_tier(tier);
        let reasoning = format!(
            "category={:?}, input_tokens={}, output_range={}-{}, model={}, tier={}, retry_multiplier={:.1}",
            category,
            input_tokens,
            low_output,
            high_output,
            model_name,
            tier.as_str(),
            retry_multiplier
        );

        CostPrediction {
            low_cents,
            high_cents,
            input_tokens,
            output_tokens: (low_output + high_output) / 2,
            model_tier: tier,
            reasoning,
        }
    }

    /// Estimate the token cost of a single chat message (content, tool calls,
    /// and replayed provider state). Used by the context-window enforcer to remove
    /// messages incrementally without re-tokenizing the whole request after
    /// every removal.
    pub fn count_message_tokens(&self, message: &ChatMessage) -> u64 {
        // Use content_str() for text-only messages (avoids allocation).
        // For multipart messages, use full_text() for text token counting
        // and add a rough estimate for file/image data URL payloads.
        let mut tokens = if let Some(content) = message.content_str() {
            match &self.tokenizer {
                Some(bpe) => cached_bpe_count(&self.token_count_cache, bpe, content),
                None => (content.len() as u64).div_ceil(self.chars_per_token.max(1) as u64),
            }
        } else {
            // Multipart or no content: count text parts + attachment size.
            let text = message.full_text();
            let text_tokens = if !text.is_empty() {
                match &self.tokenizer {
                    Some(bpe) => cached_bpe_count(&self.token_count_cache, bpe, &text),
                    None => (text.len() as u64).div_ceil(self.chars_per_token.max(1) as u64),
                }
            } else {
                0
            };
            // Estimate tokens for image/file data URLs. Base64 data is
            // roughly 4 chars per 3 bytes; providers typically process
            // attachments differently from text, but counting the data URL
            // length gives a reasonable upper-bound estimate for context
            // window enforcement.
            let attachment_len = message.content_len().saturating_sub(text.len());
            let attachment_tokens =
                (attachment_len as u64).div_ceil(self.chars_per_token.max(1) as u64);
            text_tokens + attachment_tokens
        };
        if let Some(calls) = message.tool_calls.as_deref() {
            for call in calls {
                let mut call_text =
                    String::with_capacity(call.function.name.len() + call.function.arguments.len());
                call_text.push_str(&call.function.name);
                call_text.push_str(&call.function.arguments);
                tokens += match &self.tokenizer {
                    Some(bpe) => cached_bpe_count(&self.token_count_cache, bpe, &call_text),
                    None => (call_text.len() as u64).div_ceil(self.chars_per_token.max(1) as u64),
                };
            }
        }
        let replay = crate::provider::replay::estimation_text(message);
        tokens += match &self.tokenizer {
            Some(bpe) => cached_bpe_count(&self.token_count_cache, bpe, &replay),
            None => (replay.len() as u64).div_ceil(self.chars_per_token.max(1) as u64),
        };
        tokens
    }

    /// Estimated token charge with explicit pricing provenance. No tier fallback.
    pub fn estimate_usage(&self, usage: &TokenUsage) -> crate::pricing::CostEstimate {
        crate::pricing::estimate(
            self.provider.as_deref().unwrap_or("unknown"),
            self.model.as_deref().unwrap_or("unknown"),
            usage,
            self.free_provider,
            false,
        )
    }

    /// Legacy whole-cent display only. Accounting must use estimate_usage.
    pub fn actual_cost_cents(&self, _tier: ModelTier, usage: &TokenUsage) -> i64 {
        self.estimate_usage(usage).display_cents()
    }

    fn pricing_for_tier(&self, tier: ModelTier) -> ModelPricing {
        if let Some(provider) = &self.provider {
            if let Some(model_id) = self.model.as_deref() {
                if let Some(record) = model_catalog()
                    .iter()
                    .find(|m| m.matches(provider, &model_id))
                {
                    return ModelPricing::from_record(record);
                }
            } else if let Some(model_id) =
                default_model_for_provider_and_tier(provider, tier.as_str())
            {
                if let Some(record) = model_catalog()
                    .iter()
                    .find(|m| m.matches(provider, &model_id))
                {
                    return ModelPricing::from_record(record);
                }
            }
        }
        match tier {
            ModelTier::Lite => self.lite_pricing.clone(),
            ModelTier::Pro => self.pro_pricing.clone(),
            ModelTier::Ultra => self.ultra_pricing.clone(),
        }
    }

    fn model_name_for_tier(&self, tier: ModelTier) -> String {
        self.model.clone().unwrap_or_else(|| {
            self.provider
                .as_ref()
                .and_then(|provider| default_model_for_provider_and_tier(provider, tier.as_str()))
                .unwrap_or_else(|| tier.as_str().to_string())
        })
    }

    /// Chars-per-token heuristic to use for a given category. Code tokenizes
    /// denser than prose (more punctuation/symbols), so it gets its own ratio.
    fn chars_per_token_for(&self, category: TaskCategory) -> usize {
        match category {
            TaskCategory::Code => self.code_chars_per_token,
            _ => self.chars_per_token,
        }
    }

    fn estimate_input_tokens(
        &self,
        request: &ChatCompletionRequest,
        category: TaskCategory,
    ) -> u64 {
        if let Some(bpe) = &self.tokenizer {
            // Chat messages have separate framing: tokenize their components
            // independently, not one synthetic multi-megabyte joined string.
            // Unchanged tool results now reuse exact component counts when a
            // new message arrives. Include framing, attachments and replay;
            // never accept token counts supplied by a client.
            let message_tokens = request.messages.iter().fold(0u64, |total, message| {
                total
                    .saturating_add(self.count_message_tokens(message))
                    .saturating_add(4)
            });
            let tool_tokens =
                request
                    .tools
                    .as_deref()
                    .unwrap_or_default()
                    .iter()
                    .fold(0u64, |total, tool| {
                        total.saturating_add(cached_bpe_count(
                            &self.token_count_cache,
                            bpe,
                            &tool.parameters.to_string(),
                        ))
                    });
            let tool_overhead = (request.tools.as_deref().unwrap_or_default().len() as u64)
                .saturating_mul(self.tool_overhead_tokens);
            return message_tokens
                .saturating_add(tool_tokens)
                .max(MIN_INPUT_TOKENS)
                .saturating_add(tool_overhead);
        }
        let mut input = String::new();
        for message in &request.messages {
            // Use content_str() for text-only messages. For multipart
            // messages, include the full text content plus a rough estimate
            // of attachment data URL sizes so context isn't underestimated.
            if let Some(content) = message.content_str() {
                input.push_str(content);
                input.push('\n');
            } else {
                let text = message.full_text();
                if !text.is_empty() {
                    input.push_str(&text);
                    input.push('\n');
                }
                // Include attachment data URL length for size estimation.
                // The actual token count for images/files depends on the
                // provider, but including the data URL length prevents
                // underestimating context usage for large attachments.
                let attachment_len = message.content_len().saturating_sub(text.len());
                if attachment_len > 0 {
                    // Reserve space without inlining base64 (which would
                    // bloat the string). Add a proportional placeholder.
                    for _ in 0..(attachment_len / 4) {
                        input.push('x');
                    }
                    input.push('\n');
                }
            }
            // Tool-loop transcripts can carry substantial arguments even when
            // assistant content is empty. Omitting them underestimated both
            // context usage and cost exactly where long agent runs slow down.
            for call in message.tool_calls.as_deref().unwrap_or_default() {
                input.push_str(&call.function.name);
                input.push_str(&call.function.arguments);
                input.push('\n');
            }
        }
        for message in &request.messages {
            let replay = crate::provider::replay::estimation_text(message);
            if !replay.is_empty() {
                input.push_str(&replay);
                input.push('\n');
            }
        }
        for tool in request.tools.as_deref().unwrap_or_default() {
            let _ = writeln!(&mut input, "{}", tool.parameters);
        }

        let base_input_tokens = match &self.tokenizer {
            Some(bpe) => (bpe.encode_ordinary(&input).len() as u64).max(MIN_INPUT_TOKENS),
            None => {
                // Fallback: char-heuristic, only used if the BPE encoder
                // failed to build (should be rare in practice).
                let chars_per_token = self.chars_per_token_for(category);
                ((input.len() / chars_per_token).max(MIN_INPUT_TOKENS as usize)) as u64
            }
        };

        let tool_count = request.tools.as_ref().map(|t| t.len()).unwrap_or(0) as u64;
        let tool_overhead_tokens = tool_count * self.tool_overhead_tokens;

        base_input_tokens + tool_overhead_tokens
    }

    /// Concatenated lowercase text used for categorization: any system
    /// message content plus the last user message. Including system messages
    /// lets a domain persona disambiguate generic follow-ups without assuming
    /// that the agent is a software assistant.
    fn relevant_text(&self, request: &ChatCompletionRequest) -> String {
        let mut text = String::new();
        for message in request
            .messages
            .iter()
            .filter(|message| message.role.eq_ignore_ascii_case("system"))
        {
            // Handle both text and multipart system messages.
            if let Some(content) = message.content_str() {
                text.push_str(content);
                text.push(' ');
            } else {
                let part_text = message.full_text();
                if !part_text.is_empty() {
                    text.push_str(&part_text);
                    text.push(' ');
                }
            }
        }
        // Find the last user message with usable text for categorization.
        // For multipart messages, use the concatenated text parts.
        if let Some(last_user_text) = request
            .messages
            .iter()
            .rev()
            .filter(|message| message.role.eq_ignore_ascii_case("user"))
            .filter_map(|message| {
                if let Some(content) = message.content_str() {
                    session_user_request(content).map(|s| s.to_string())
                } else {
                    let text = message.full_text();
                    if text.is_empty() {
                        None
                    } else {
                        session_user_request(&text).map(|s| s.to_string())
                    }
                }
            })
            .next()
        {
            text.push_str(&last_user_text);
        }
        text.make_ascii_lowercase();
        text
    }

    /// Determine read-only intent from explicit tool metadata first, then a
    /// domain-neutral naming convention. This supports tools such as
    /// `search_crm`, `query_ledger`, or `get_calendar` in addition to file
    /// tools, while rejecting names that also carry a mutation verb.
    fn is_read_only_tool(&self, tool: &ToolDefinition) -> bool {
        if tool.safe_to_memoize
            || self
                .read_only_tools
                .iter()
                .any(|name| tool.name.eq_ignore_ascii_case(name))
        {
            return true;
        }
        let terms = crate::lexical::words(&tool.name);
        let Some(first) = terms.first().map(String::as_str) else {
            return false;
        };
        const READ_PREFIXES: &[&str] = &[
            "fetch", "find", "get", "inspect", "list", "lookup", "query", "read", "retrieve",
            "search", "view",
        ];
        const MUTATION_TERMS: &[&str] = &[
            "approve", "book", "cancel", "create", "delete", "edit", "execute", "pay", "post",
            "publish", "purchase", "run", "save", "send", "submit", "transfer", "update", "write",
        ];
        READ_PREFIXES.contains(&first)
            && !terms
                .iter()
                .any(|term| MUTATION_TERMS.contains(&term.as_str()))
    }

    fn categorize(&self, request: &ChatCompletionRequest) -> TaskCategory {
        let tools = request.tools.as_deref();

        let all_read_only = tools.map(|tools| {
            !tools.is_empty() && tools.iter().all(|tool| self.is_read_only_tool(tool))
        });

        if all_read_only == Some(true) {
            return TaskCategory::ToolSelection;
        }

        let text = self.relevant_text(request);

        if self
            .modal_signals
            .iter()
            .any(|signal| crate::lexical::contains_signal(&text, signal))
        {
            return TaskCategory::HeavyModal;
        }

        // Score rather than first-match-wins, so overlapping signals
        // (e.g. "debug" and "architecture" appear in both lists) don't get
        // silently decided by list ordering.
        let code_score = self
            .code_signals
            .iter()
            .filter(|signal| crate::lexical::contains_signal(&text, signal))
            .count();
        let artifact_score = self
            .artifact_signals
            .iter()
            .filter(|signal| crate::lexical::contains_signal(&text, signal))
            .count();
        let reasoning_score = self
            .reasoning_signals
            .iter()
            .filter(|signal| crate::lexical::contains_signal(&text, signal))
            .count();

        let best_score = code_score.max(artifact_score).max(reasoning_score);
        if best_score == 0 {
            TaskCategory::Text
        } else if reasoning_score == best_score && reasoning_score > code_score {
            TaskCategory::Reasoning
        } else if artifact_score == best_score && artifact_score > code_score {
            TaskCategory::Artifact
        } else {
            TaskCategory::Code
        }
    }

    fn estimate_output_range(
        &self,
        request: &ChatCompletionRequest,
        category: TaskCategory,
    ) -> (u64, u64) {
        let (default_low, default_high) = match category {
            TaskCategory::ToolSelection => (50, 150),
            TaskCategory::Code => (200, 800),
            TaskCategory::Artifact => (250, 900),
            TaskCategory::Reasoning => (300, 1_200),
            TaskCategory::Text => (150, 500),
            TaskCategory::HeavyModal => (0, 0),
        };

        match request.max_tokens.map(|m| m as u64) {
            // `max_tokens` is a hard ceiling the provider will not exceed,
            // so the high bound can never be larger than it. Low bound is
            // a fraction of the cap, floored at a sane minimum.
            Some(requested) => {
                let low = requested.min(default_low.max(requested / 4)).max(1);
                let high = requested.max(low);
                (low, high)
            }
            None => (default_low, default_high),
        }
    }

    fn retry_multiplier(&self, category: TaskCategory) -> f64 {
        match category {
            TaskCategory::Reasoning => 1.5,
            TaskCategory::Code => 1.3,
            TaskCategory::Artifact => 1.2,
            TaskCategory::HeavyModal => 2.0,
            _ => 1.0,
        }
    }
}

fn tokens_to_cents(tokens: u64, cents_per_1m: f64) -> i64 {
    ((tokens as f64 * cents_per_1m) / 1_000_000.0).ceil() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ChatMessage, FunctionCall, ToolCall, ToolDefinition};
    use crate::MessageContent;

    #[test]
    fn token_cache_is_bounded_and_changed_text_gets_a_new_exact_count() {
        let bpe = shared_tokenizer().unwrap();
        let cache = Mutex::new(TokenCountCache::default());
        for index in 0..TOKEN_COUNT_CACHE_ENTRIES + 5 {
            let text = format!("cache regression unique content {index}");
            assert_eq!(
                cached_bpe_count(&cache, &bpe, &text),
                bpe.encode_ordinary(&text).len() as u64
            );
        }
        let saved = cache.lock().unwrap();
        assert_eq!(saved.counts.len(), TOKEN_COUNT_CACHE_ENTRIES);
        assert_eq!(saved.insertion_order.len(), TOKEN_COUNT_CACHE_ENTRIES);
        drop(saved);
        for text in [
            "original content",
            "changed content with a substantially longer response",
        ] {
            let expected = bpe.encode_ordinary(text).len() as u64;
            assert_eq!(cached_bpe_count(&cache, &bpe, text), expected);
            assert_eq!(cached_bpe_count(&cache, &bpe, text), expected);
        }
    }

    #[test]
    fn tokenizer_override_has_a_separate_cache_and_message_framing_is_counted() {
        let predictor = CostPredictor::new();
        let custom = predictor.clone().with_tokenizer(shared_tokenizer());
        assert!(!Arc::ptr_eq(
            &predictor.token_count_cache,
            &custom.token_count_cache
        ));
        let request = base_request(&"long content ".repeat(200), None);
        let expected: u64 = request
            .messages
            .iter()
            .map(|message| predictor.count_message_tokens(message) + 4)
            .sum();
        assert_eq!(
            predictor.estimate_input_tokens(&request, TaskCategory::Text),
            expected
        );
        assert_eq!(
            custom.estimate_input_tokens(&request, TaskCategory::Text),
            expected
        );
    }

    fn user_msg(content: &str) -> ChatMessage {
        ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(MessageContent::text(content)),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn base_request(content: &str, max_tokens: Option<u32>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "reincore-router".to_string(),
            messages: vec![user_msg(content)],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens,
            stream: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn replay_is_counted_by_both_token_estimation_paths() {
        for predictor in [
            CostPredictor::new(),
            CostPredictor::new().with_tokenizer(None),
        ] {
            let mut req = base_request(&"input ".repeat(300), None);
            let mut assistant = user_msg("Checking");
            assistant.role = "assistant".into();
            let base_message = predictor.count_message_tokens(&assistant);
            req.messages.push(assistant.clone());
            let base_request = predictor.estimate_input_tokens(&req, TaskCategory::Text);
            assistant.provider_state = crate::provider::replay::capture(
                "scope".into(),
                &serde_json::json!({"thinking": "reasoning ".repeat(500)}),
                false,
            )
            .map(|state| state.bind(&assistant).for_request_group("group"));
            req.messages[1] = assistant.clone();
            assert!(predictor.count_message_tokens(&assistant) > base_message + 100);
            assert!(predictor.estimate_input_tokens(&req, TaskCategory::Text) > base_request + 100);
            let with_replay = predictor.count_message_tokens(&assistant);
            let state = assistant.provider_state.as_mut().unwrap();
            state.scope = "internal hash must not count ".repeat(100);
            state.request_group_id = Some("internal group must not count ".repeat(100));
            assert_eq!(predictor.count_message_tokens(&assistant), with_replay);
            crate::provider::retain_request_group_state(&mut req.messages, "next-group");
            assert_eq!(
                predictor.estimate_input_tokens(&req, TaskCategory::Text),
                base_request
            );
        }
    }

    #[test]
    fn replay_estimate_excludes_duplicate_claude_content_and_edited_state() {
        let predictor = CostPredictor::new();
        let mut message = user_msg(&"visible answer ".repeat(200));
        message.role = "assistant".into();
        let visible_tokens = predictor.count_message_tokens(&message);
        let thinking = serde_json::json!({"type":"thinking","thinking":"reasoning","signature":"opaque-signature"});
        let raw = serde_json::json!({"content":[
            thinking,
            {"type":"text","text":message.full_text()},
            {"type":"tool_use","id":"call","name":"read_file","input":{"path":"a.rs"}}
        ]});
        message.provider_state = crate::provider::replay::capture("scope".into(), &raw, true)
            .map(|state| state.bind(&message));
        let replay_text = crate::provider::replay::estimation_text(&message);
        assert_eq!(replay_text, thinking.to_string());
        assert_eq!(
            predictor.count_message_tokens(&message),
            visible_tokens
                + predictor
                    .tokenizer
                    .as_ref()
                    .unwrap()
                    .encode_ordinary(&replay_text)
                    .len() as u64
        );
        message.content = Some(MessageContent::text("edited"));
        let edited_tokens = predictor.count_message_tokens(&message);
        message.provider_state = None;
        assert_eq!(predictor.count_message_tokens(&message), edited_tokens);
    }

    #[test]
    fn prediction_is_a_range() {
        let req = base_request("hello world", Some(100));
        let pred = CostPredictor::new().predict(&req, ModelTier::Lite).unwrap();
        assert!(pred.low_cents > 0);
        assert!(pred.high_cents >= pred.low_cents);
    }

    #[test]
    fn ultra_is_more_expensive_than_lite() {
        let req = base_request("explain quantum computing", Some(200));
        let lite = CostPredictor::new().predict(&req, ModelTier::Lite).unwrap();
        let ultra = CostPredictor::new()
            .predict(&req, ModelTier::Ultra)
            .unwrap();
        assert!(ultra.low_cents > lite.high_cents);
    }

    #[test]
    fn tool_schemas_increase_input_tokens() {
        let mut req = base_request("hi", Some(100));
        req.tools = Some(vec![ToolDefinition {
            action_contract: None,
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                }
            }),
            safe_to_memoize: true,
            memoization_policy: None,
        }]);

        let predictor = CostPredictor::new();
        let with_tools = predictor.predict(&req, ModelTier::Lite).unwrap();
        let without_tools = predictor
            .predict(
                &ChatCompletionRequest {
                    tools: None,
                    ..req.clone()
                },
                ModelTier::Lite,
            )
            .unwrap();
        assert!(with_tools.input_tokens >= without_tools.input_tokens);
    }

    #[test]
    fn historical_tool_arguments_are_counted() {
        let predictor = CostPredictor::new();
        let mut request = base_request("inspect the result", Some(100));
        let baseline = predictor
            .predict(&request, ModelTier::Lite)
            .unwrap()
            .input_tokens;
        request.messages.push(ChatMessage {
            provider_state: None,
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call-1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "write_file".to_string(),
                    arguments: format!(r#"{{"content":"{}"}}"#, "x".repeat(2_000)),
                },
            }]),
            tool_call_id: None,
            name: None,
        });

        let with_call = predictor
            .predict(&request, ModelTier::Lite)
            .unwrap()
            .input_tokens;
        assert!(with_call > baseline);
    }

    #[test]
    fn prediction_can_reuse_an_existing_token_count() {
        let request = base_request("hello", Some(100));
        let prediction = CostPredictor::new()
            .predict_with_input_tokens(&request, ModelTier::Lite, 12_345)
            .unwrap();
        assert_eq!(prediction.input_tokens, 12_345);
    }

    #[test]
    fn high_bound_never_exceeds_requested_max_tokens() {
        // Regression test: previously `high` could be `max_tokens * 2`,
        // which is impossible since the provider enforces max_tokens as a hard cap.
        let req = base_request("write me a long story please", Some(50));
        let predictor = CostPredictor::new();
        let (_, high_output) = predictor.estimate_output_range(&req, TaskCategory::Text);
        assert!(high_output <= 50);
    }

    #[test]
    fn categorize_prefers_higher_scoring_category_over_list_order() {
        // Multiple reasoning signals should resolve to Reasoning rather than
        // silently falling to Code because code signals are scored first.
        let req = base_request(
            "let's analyze and design the architecture, then debug it",
            None,
        );
        let predictor = CostPredictor::new();
        assert_eq!(predictor.categorize(&req), TaskCategory::Reasoning);
    }

    #[test]
    fn categorize_picks_code_when_code_signal_dominates() {
        let req = base_request(
            "refactor this function and fix the bug in the algorithm",
            None,
        );
        let predictor = CostPredictor::new();
        assert_eq!(predictor.categorize(&req), TaskCategory::Code);
    }

    #[test]
    fn categorize_picks_artifact_for_business_deliverables() {
        let req = base_request(
            "Draft a renewal proposal and executive report for Acme procurement",
            None,
        );
        let predictor = CostPredictor::new();
        assert_eq!(predictor.categorize(&req), TaskCategory::Artifact);
    }

    #[test]
    fn categorize_picks_reasoning_for_non_coding_research() {
        let req = base_request(
            "Research and compare regional suppliers, then recommend the best option",
            None,
        );
        let predictor = CostPredictor::new();
        assert_eq!(predictor.categorize(&req), TaskCategory::Reasoning);
    }

    #[test]
    fn categorize_recognizes_non_image_modal_work() {
        let req = base_request("Create audio for the campaign voiceover", None);
        let predictor = CostPredictor::new();
        assert_eq!(predictor.categorize(&req), TaskCategory::HeavyModal);
    }

    #[test]
    fn read_only_tool_detection_is_domain_neutral() {
        let predictor = CostPredictor::new();
        for name in [
            "search_crm_accounts",
            "query_ledger_entries",
            "get_calendar_events",
        ] {
            let mut req = base_request("Find the relevant records", None);
            req.tools = Some(vec![ToolDefinition {
                action_contract: None,
                name: name.to_string(),
                description: "Look up records".to_string(),
                parameters: serde_json::json!({"type": "object"}),
                safe_to_memoize: false,
                memoization_policy: None,
            }]);
            assert_eq!(
                predictor.categorize(&req),
                TaskCategory::ToolSelection,
                "{name} should be recognized as read-only"
            );
        }
    }

    #[test]
    fn read_only_prefix_does_not_hide_a_mutation() {
        let mut req = base_request("Find and send the customer email", None);
        req.tools = Some(vec![ToolDefinition {
            action_contract: None,
            name: "get_and_send_customer_email".to_string(),
            description: "Send an email".to_string(),
            parameters: serde_json::json!({"type": "object"}),
            safe_to_memoize: false,
            memoization_policy: None,
        }]);
        assert_ne!(
            CostPredictor::new().categorize(&req),
            TaskCategory::ToolSelection
        );
    }

    #[test]
    fn categorize_uses_system_prompt_context() {
        let mut req = base_request("please fix this", None);
        req.messages.insert(
            0,
            ChatMessage {
                provider_state: None,
                role: "system".to_string(),
                content: Some(crate::MessageContent::text(
                    "You are a senior software engineer helping refactor code.".to_string(),
                )),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        );
        let predictor = CostPredictor::new();
        assert_eq!(predictor.categorize(&req), TaskCategory::Code);
    }

    #[test]
    fn tokenizer_gives_exact_count_for_known_string() {
        // "hello world" is 2 tokens under cl100k_base. Pin this so a future
        // tiktoken-rs upgrade that changes encoding behavior gets caught.
        let req = base_request("hello world", None);
        let predictor = CostPredictor::new();
        let tokens = predictor.estimate_input_tokens(&req, TaskCategory::Text);
        // Below MIN_INPUT_TOKENS floor, so the real count is masked by the
        // floor here - assert the floor kicks in rather than the raw count.
        assert_eq!(tokens, MIN_INPUT_TOKENS);
    }

    #[test]
    fn tokenizer_produces_higher_fidelity_count_than_heuristic_for_long_text() {
        let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(50);
        let req = base_request(&long_text, None);

        let with_tokenizer = CostPredictor::new();
        let without_tokenizer = CostPredictor::new().with_tokenizer(None);

        let real_count = with_tokenizer.estimate_input_tokens(&req, TaskCategory::Text);
        let heuristic_count = without_tokenizer.estimate_input_tokens(&req, TaskCategory::Text);

        // They shouldn't be wildly different, but they also shouldn't be
        // identical - confirms the tokenizer path is actually being used
        // rather than silently falling through to the heuristic.
        assert_ne!(real_count, heuristic_count);
    }

    #[test]
    fn missing_tokenizer_falls_back_to_char_heuristic_without_panicking() {
        let req = base_request("some reasonably long piece of text to estimate", None);
        let predictor = CostPredictor::new().with_tokenizer(None);
        let tokens = predictor.estimate_input_tokens(&req, TaskCategory::Text);
        assert!(tokens >= MIN_INPUT_TOKENS);
    }

    #[test]
    fn heavy_modal_uses_flat_cost_not_zero() {
        let req = base_request("please draw a cat for me", None);
        let predictor = CostPredictor::new();
        let pred = predictor.predict(&req, ModelTier::Lite).unwrap();
        assert!(pred.low_cents > 0);
    }

    #[test]
    fn retry_multiplier_scales_full_cost_not_just_input() {
        let req = base_request("plan and analyze this architecture design", Some(300));
        let predictor = CostPredictor::new();
        let pred = predictor.predict(&req, ModelTier::Lite).unwrap();
        // Reasoning has a 1.5x retry multiplier; high bound should reflect
        // that applied to the full (input+output) cost, not just input.
        assert!(pred.high_cents as f64 >= pred.low_cents as f64 * 1.4);
    }

    #[test]
    fn actual_cost_never_uses_unattributed_tier_pricing() {
        let predictor = CostPredictor::new().with_pricing(
            ModelTier::Lite,
            ModelPricing {
                input_cents_per_1m: 100.0,
                output_cents_per_1m: 200.0,
            },
        );
        let usage = TokenUsage {
            prompt_tokens: 1_000,
            completion_tokens: 1_000,
            total_tokens: 2_000,
            cached_prompt_tokens: 0,
            cache_creation_input_tokens: 0,
            ..Default::default()
        };
        assert_eq!(
            predictor.estimate_usage(&usage).status,
            crate::pricing::CostStatus::Unknown
        );
        assert_eq!(predictor.estimate_usage(&usage).amount_nano_usd, None);
        assert_eq!(
            predictor
                .clone()
                .with_free_provider(true)
                .actual_cost_cents(ModelTier::Lite, &usage),
            0
        );
    }

    #[test]
    fn concrete_standard_model_pricing_overrides_lite_tier_default() {
        let model = "gpt-5.4-2026-03-05";
        let record = model_catalog()
            .iter()
            .find(|record| record.provider == "openai" && record.model_id == model)
            .unwrap();
        let predictor = CostPredictor::new()
            .with_provider("openai")
            .with_model(model);
        let usage = TokenUsage {
            prompt_tokens: 100_000,
            completion_tokens: 100_000,
            total_tokens: 200_000,
            cached_prompt_tokens: 0,
            cache_creation_input_tokens: 0,
            ..Default::default()
        };

        assert_eq!(predictor.model(), Some(model));
        assert_eq!(
            predictor.actual_cost_cents(ModelTier::Lite, &usage),
            ((record.input_price + record.output_price) * 10.0).floor() as i64
        );
        assert_ne!(
            predictor.actual_cost_cents(ModelTier::Lite, &usage),
            CostPredictor::new()
                .with_provider("openai")
                .actual_cost_cents(ModelTier::Lite, &usage)
        );
    }

    #[test]
    fn concrete_explicit_override_pricing_and_label_beat_ultra_default() {
        let model = "gpt-5-nano-2025-08-07";
        let record = model_catalog()
            .iter()
            .find(|record| record.provider == "openai" && record.model_id == model)
            .unwrap();
        let predictor = CostPredictor::new()
            .with_provider("openai")
            .with_model(model);
        let usage = TokenUsage {
            prompt_tokens: 100_000,
            completion_tokens: 100_000,
            total_tokens: 200_000,
            cached_prompt_tokens: 0,
            cache_creation_input_tokens: 0,
            ..Default::default()
        };
        let prediction = predictor
            .predict(&base_request("hello", Some(100)), ModelTier::Ultra)
            .unwrap();

        assert_eq!(
            predictor.actual_cost_cents(ModelTier::Ultra, &usage),
            ((record.input_price + record.output_price) * 10.0).floor() as i64
        );
        assert!(prediction.reasoning.contains(&format!("model={model}")));
    }

    #[test]
    fn free_provider_reasoning_uses_concrete_model_label() {
        let prediction = CostPredictor::new()
            .with_provider("ollama")
            .with_model("qwen3:8b")
            .with_free_provider(true)
            .predict(&base_request("hello", None), ModelTier::Lite)
            .unwrap();

        assert_eq!(prediction.low_cents, 0);
        assert_eq!(prediction.high_cents, 0);
        assert!(prediction.reasoning.contains("model=qwen3:8b"));
    }

    #[test]
    fn cost_accuracy_bps_matches_prediction_exactly() {
        // actual == predicted_high → 10_000 bps (a perfect prediction).
        assert_eq!(cost_accuracy_bps(50, 50), Some(10_000));
    }

    #[test]
    fn cost_accuracy_bps_reports_prediction_was_too_high() {
        // actual = 25, predicted_high = 50 → 5_000 bps (prediction was 2× too high).
        assert_eq!(cost_accuracy_bps(25, 50), Some(5_000));
    }

    #[test]
    fn cost_accuracy_bps_reports_prediction_was_too_low() {
        // actual = 100, predicted_high = 50 → 20_000 bps (actual was 2× the prediction).
        assert_eq!(cost_accuracy_bps(100, 50), Some(20_000));
    }

    #[test]
    fn cost_accuracy_bps_rounds_to_nearest_basis_point() {
        // actual = 1, predicted_high = 3 → 0.3333... → 3333 bps.
        assert_eq!(cost_accuracy_bps(1, 3), Some(3333));
    }

    #[test]
    fn cost_accuracy_bps_returns_none_for_zero_prediction() {
        // Free providers (Ollama) and sessions without a prediction have
        // nothing to calibrate against - never record a misleading infinity.
        assert_eq!(cost_accuracy_bps(0, 0), None);
        assert_eq!(cost_accuracy_bps(10, 0), None);
        assert_eq!(cost_accuracy_bps(10, -1), None);
    }

    #[test]
    fn cost_accuracy_bps_handles_zero_actual() {
        // A free completion against a non-zero prediction is a 0% ratio.
        assert_eq!(cost_accuracy_bps(0, 50), Some(0));
    }

    #[test]
    fn cache_hit_rate_bps_full_cache_hit() {
        // All prompt tokens served from cache → 10_000 bps (100%).
        assert_eq!(cache_hit_rate_bps(500, 500), Some(10_000));
    }

    #[test]
    fn cache_hit_rate_bps_partial_cache_hit() {
        // 750 of 1000 prompt tokens cached → 7500 bps (75%).
        assert_eq!(cache_hit_rate_bps(750, 1000), Some(7500));
    }

    #[test]
    fn cache_hit_rate_bps_no_cache_hit() {
        // Zero cached tokens → 0 bps.
        assert_eq!(cache_hit_rate_bps(0, 1000), Some(0));
    }

    #[test]
    fn cache_hit_rate_bps_returns_none_for_zero_prompt_tokens() {
        // Degenerate request with no input - can't compute a ratio.
        assert_eq!(cache_hit_rate_bps(0, 0), None);
        assert_eq!(cache_hit_rate_bps(100, 0), None);
    }

    #[test]
    fn cache_hit_rate_bps_clamps_to_100_percent() {
        // Some providers report more cached tokens than prompt tokens
        // (e.g. when the cache includes system prompt overhead). Clamp to
        // 10_000 rather than reporting a misleading >100% rate.
        assert_eq!(cache_hit_rate_bps(2000, 1000), Some(10_000));
    }

    #[test]
    fn cache_hit_rate_bps_rounds_to_nearest_basis_point() {
        // 1 of 3 → 0.3333... → 3333 bps.
        assert_eq!(cache_hit_rate_bps(1, 3), Some(3333));
    }
}
