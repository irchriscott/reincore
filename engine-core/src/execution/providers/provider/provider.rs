//! Provider abstraction for outbound LLM calls.
//!
//! The gateway is provider-agnostic: it classifies a request, picks a tier,
//! and asks a `Provider` implementation to produce a `ChatCompletionResponse`.
//! Concrete adapters (OpenAI, Anthropic, etc.) live in sibling modules.

use crate::error::EngineResult;
use crate::models::{ChatCompletionRequest, ChatMessage, ModelTier, ToolCall, ToolDefinition};
use crate::routing::{RoutingMode, RoutingPolicy};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::pin::Pin;

/// Maximum number of transparent provider retries after a stream completes
/// without text or tool calls. Keep this bounded to avoid runaway billing.
pub const EMPTY_PROVIDER_RESPONSE_RETRY_LIMIT: usize = 1;

/// HTTP response headers returned by an upstream provider.
///
/// Header names are normalized to lowercase. Repeated values are joined with
/// a comma so the metadata remains serializable and easy to inspect without
/// retaining the provider's live HTTP response body.
pub type ProviderResponseHeaders = BTreeMap<String, String>;

/// Provider-neutral view of the rate-limit capacity left after a request.
///
/// Providers do not agree on header names or token buckets. The raw headers
/// remain available on [`ChatCompletionResponse::response_headers`]; this
/// type covers the common request and token dimensions exposed by OpenAI,
/// Azure OpenAI, Groq, and Anthropic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateLimitRemaining {
    pub requests: Option<u64>,
    pub tokens: Option<u64>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}

/// Outcome of a chat-completion call, normalized across providers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Provider-generated completion id, if any.
    pub id: String,
    /// Concrete model name returned by the provider.
    pub model: String,
    /// The assistant message. May contain `tool_calls`.
    pub message: ChatMessage,
    /// Provider-reported usage, when available.
    pub usage: Option<TokenUsage>,
    /// Raw provider response, useful for debugging/auditing.
    pub raw: serde_json::Value,
    /// Raw HTTP response headers from the provider.
    ///
    /// This is empty for synthetic/cached responses and providers that do not
    /// expose HTTP metadata. Use [`ChatCompletionResponse::header`] for exact
    /// provider fields or [`ChatCompletionResponse::rate_limit_remaining`] for
    /// a normalized view of the common remaining-capacity headers.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub response_headers: ProviderResponseHeaders,
}

impl ChatCompletionResponse {
    /// Read one upstream response header case-insensitively.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.response_headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }

    /// Return the provider's common remaining request/token capacity.
    ///
    /// Missing values are expected: Gemini, DeepSeek, Ollama, and other
    /// providers may not publish remaining capacity, while Anthropic exposes
    /// separate input/output buckets in addition to a combined token bucket.
    pub fn rate_limit_remaining(&self) -> RateLimitRemaining {
        RateLimitRemaining {
            requests: parse_first_u64_header(
                &self.response_headers,
                &[
                    "x-ratelimit-remaining-requests",
                    "x-ms-ratelimit-remaining-requests",
                    "anthropic-ratelimit-requests-remaining",
                    "x-ratelimit-remaining",
                    "ratelimit-remaining",
                ],
            ),
            tokens: parse_first_u64_header(
                &self.response_headers,
                &[
                    "x-ratelimit-remaining-tokens",
                    "x-ms-ratelimit-remaining-tokens",
                    "anthropic-ratelimit-tokens-remaining",
                ],
            ),
            input_tokens: parse_first_u64_header(
                &self.response_headers,
                &["anthropic-ratelimit-input-tokens-remaining"],
            ),
            output_tokens: parse_first_u64_header(
                &self.response_headers,
                &["anthropic-ratelimit-output-tokens-remaining"],
            ),
        }
    }
}

fn parse_first_u64_header(headers: &ProviderResponseHeaders, names: &[&str]) -> Option<u64> {
    names
        .iter()
        .find_map(|name| headers.get(*name)?.trim().parse().ok())
}

/// Copy serializable upstream HTTP headers before the response body is read.
pub(crate) fn capture_response_headers(
    headers: &reqwest::header::HeaderMap,
) -> ProviderResponseHeaders {
    let mut captured = ProviderResponseHeaders::new();
    for name in headers.keys() {
        let values = headers
            .get_all(name)
            .iter()
            .filter_map(|value| value.to_str().ok())
            .collect::<Vec<_>>()
            .join(", ");
        if !values.is_empty() {
            captured.insert(name.as_str().to_ascii_lowercase(), values);
        }
    }
    captured
}

/// Whether an upstream header is safe and useful to expose through ReinCore's
/// HTTP proxy response.
pub fn is_rate_limit_response_header(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name == "retry-after"
        || name == "ratelimit-limit"
        || name == "ratelimit-remaining"
        || name == "ratelimit-reset"
        || name.starts_with("x-ratelimit-")
        || name.starts_with("x-ms-ratelimit-")
        || name.starts_with("anthropic-ratelimit-")
        || name.starts_with("anthropic-priority-")
        || name.starts_with("anthropic-fast-")
}

/// Provider-acknowledged queued batch request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchSubmission {
    pub batch_id: String,
    pub status: String,
}

/// Token usage reported by a provider.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    /// Subset of completion_tokens; never add these a second time for billing.
    #[serde(default)]
    pub reasoning_tokens: u64,
    /// Subset of cache_creation_input_tokens written with a one-hour TTL.
    #[serde(default)]
    pub cache_creation_1h_tokens: u64,
    /// Total input including cache reads and writes (normalized across providers).
    pub prompt_tokens: u64,
    /// All billed output, including reasoning tokens.
    pub completion_tokens: u64,
    pub total_tokens: u64,
    /// Provider-side prompt cache tokens read from prior cached prefixes.
    pub cached_prompt_tokens: u64,
    /// Provider-side prompt cache tokens written/created this call.
    pub cache_creation_input_tokens: u64,
}

impl TokenUsage {
    /// Combine usage from multiple provider attempts without overflowing.
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            reasoning_tokens: self.reasoning_tokens.saturating_add(other.reasoning_tokens),
            cache_creation_1h_tokens: self
                .cache_creation_1h_tokens
                .saturating_add(other.cache_creation_1h_tokens),
            prompt_tokens: self.prompt_tokens.saturating_add(other.prompt_tokens),
            completion_tokens: self
                .completion_tokens
                .saturating_add(other.completion_tokens),
            total_tokens: self.total_tokens.saturating_add(other.total_tokens),
            cached_prompt_tokens: self
                .cached_prompt_tokens
                .saturating_add(other.cached_prompt_tokens),
            cache_creation_input_tokens: self
                .cache_creation_input_tokens
                .saturating_add(other.cache_creation_input_tokens),
        }
    }
}

/// A configured model mapping for a single provider.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderModelMapping {
    pub tier: ModelTier,
    /// Concrete model name/id the provider understands.
    pub provider_model: String,
}

/// A model advertised by a provider's model-list endpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderModel {
    /// Provider-specific model id.
    pub id: String,
    /// Provider label that owns this model.
    pub provider: String,
    /// Human-readable name, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// When the model was created, if the provider exposes it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Raw provider metadata (context window, pricing, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Credentials and endpoint configuration for a provider.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Human-readable label, e.g. "openai" or "anthropic".
    pub label: String,
    /// Base URL for the provider API.
    pub base_url: String,
    /// API key. In production this should come from a secret store.
    pub api_key: String,
    /// Standard/base model for hybrid routing. Pro requests use this model;
    /// Lite requests are derived conservatively from the curated provider
    /// catalog when no explicit mapping exists. Pro and Ultra retain this
    /// user-selected quality ceiling unless explicitly overridden.
    ///
    /// The aliases keep control-plane payloads free to call this either a
    /// base or default model. Old payloads omit the field and continue using
    /// the legacy tier mappings.
    #[serde(default, alias = "base_model", alias = "default_model")]
    pub standard_model: Option<String>,
    /// Optional explicit model mappings per tier. A mapping always takes
    /// precedence over automatic derivation from `standard_model`.
    #[serde(default)]
    pub models: Vec<ProviderModelMapping>,
    /// Routing policy controlling how tiers are selected around the standard
    /// model. Defaults to `auto`. Omitted by legacy payloads, which keep the
    /// auto behavior.
    #[serde(default)]
    pub routing: RoutingPolicy,
    /// Default timeout for a single request.
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,
    /// Optional organization id (OpenAI) or similar.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub organization_id: Option<String>,
}

impl Default for ProviderConfig {
    /// Defaults produce an empty, inert config. Real configs are built with
    /// explicit label/base_url/api_key and a default `auto` routing policy.
    fn default() -> Self {
        Self {
            label: String::new(),
            base_url: String::new(),
            api_key: String::new(),
            standard_model: None,
            models: Vec::new(),
            routing: RoutingPolicy::default(),
            timeout_seconds: default_timeout_seconds(),
            organization_id: None,
        }
    }
}

fn default_timeout_seconds() -> u64 {
    300
}

impl ProviderConfig {
    /// Title generation can explicitly choose the standard model even when
    /// the normal Pro tier has a separate override. Other requests keep routing.
    pub fn model_for_request(
        &self,
        request: &ChatCompletionRequest,
        tier: ModelTier,
    ) -> EngineResult<String> {
        if request
            .extra
            .get("reincore_title_standard_model")
            .and_then(|v| v.as_bool())
            == Some(true)
        {
            if let Some(model) = &self.standard_model {
                return Ok(model.clone());
            }
        }
        self.model_for_tier(tier)
    }

    /// Resolve the concrete provider model name for a tier using the hybrid
    /// policy. Resolution order:
    ///
    /// 1. **Explicit override** - a `ProviderModelMapping` for this tier
    ///    always wins, in every mode.
    /// 2. **`locked` mode** - always return the standard model (or legacy
    ///    Pro mapping) regardless of tier. Budget may still checkpoint.
    /// 3. **`custom` mode** - fall back to the standard model (or legacy
    ///    Pro mapping) for unmapped tiers. No catalog derivation.
    /// 4. **`auto` mode** - derive a safe cheaper Lite from the catalog;
    ///    keep Pro/Ultra at the standard model unless overridden.
    pub fn model_for_tier(&self, tier: ModelTier) -> EngineResult<String> {
        if let Some(explicit) = self.models.iter().find(|m| m.tier == tier) {
            return Ok(explicit.provider_model.clone());
        }

        let standard = self.standard_model.as_deref().or_else(|| {
            self.models
                .iter()
                .find(|mapping| mapping.tier == ModelTier::Pro)
                .map(|mapping| mapping.provider_model.as_str())
        });

        let Some(standard) = standard else {
            return Err(crate::error::EngineError::InvalidModelTier(format!(
                "no model mapping or standard model for tier {} in provider {}",
                tier.as_str(),
                self.label
            )));
        };

        // `locked` and `custom` never derive from the catalog. `locked`
        // pins every tier to the standard model; `custom` falls back to
        // the standard for unmapped tiers instead of guessing.
        if matches!(self.routing.mode, RoutingMode::Locked | RoutingMode::Custom) {
            return Ok(standard.to_string());
        }

        Ok(
            crate::model_library::derive_model_from_standard(&self.label, standard, tier)
                .unwrap_or_else(|| standard.to_string()),
        )
    }

    /// Return a reference to the provider's routing policy.
    pub fn routing_policy(&self) -> &RoutingPolicy {
        &self.routing
    }
}

/// Provider-agnostic interface for chat completions.
///
/// Implementations are responsible for:
/// - Translating `ChatCompletionRequest` into provider-specific JSON.
/// - Authenticating with the provider.
/// - Parsing the response into `ChatCompletionResponse`.
/// - Surfacing provider errors as `EngineError::Provider`.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Return the provider label.
    fn label(&self) -> &str;

    /// Return the current configuration.
    fn config(&self) -> &ProviderConfig;

    /// Send a non-streaming chat completion request.
    async fn complete(
        &self,
        request: &ChatCompletionRequest,
    ) -> EngineResult<ChatCompletionResponse>;

    /// Native batch lifecycle client; unsupported adapters fail explicitly.
    fn batch_client(&self) -> EngineResult<super::batch::BatchClient> {
        Err(crate::error::EngineError::Provider(
            "batch lifecycle is not supported".into(),
        ))
    }

    fn prepare_batch_body(
        &self,
        _request: &ChatCompletionRequest,
    ) -> EngineResult<serde_json::Value> {
        Err(crate::error::EngineError::Provider(
            "batch lifecycle is not supported".into(),
        ))
    }

    /// Submit a non-latency-sensitive request to the provider's batch tier.
    ///
    /// Providers that do not support batch submission should return an error.
    async fn submit_batch(
        &self,
        _request: &ChatCompletionRequest,
    ) -> EngineResult<BatchSubmission> {
        Err(crate::error::EngineError::Provider(
            "batch submission not implemented for this provider".to_string(),
        ))
    }

    /// Send a streaming chat completion request.
    ///
    /// The returned stream yields normalized chunks. Not all providers support
    /// streaming; a default implementation returns an error.
    async fn complete_stream(
        &self,
        _request: &ChatCompletionRequest,
    ) -> EngineResult<ProviderStream> {
        Err(crate::error::EngineError::Provider(
            "streaming not implemented for this provider".to_string(),
        ))
    }

    /// Whether this adapter can forward provider tokens as they arrive.
    /// Callers use this to avoid making a speculative streaming request to
    /// providers that only implement the non-streaming fallback.
    fn supports_streaming(&self) -> bool {
        false
    }

    /// List models available from this provider.
    ///
    /// Providers that do not support a model-list endpoint should return an
    /// empty vector rather than an error.
    async fn list_models(&self) -> EngineResult<Vec<ProviderModel>>;

    /// Check the subscription tier for providers that have tiered access
    /// (e.g. GitHub Copilot free vs paid). Returns `None` for providers
    /// that don't have tiers. Returns `Some(true)` for free tier,
    /// `Some(false)` for paid tier.
    async fn check_tier(&self) -> EngineResult<Option<bool>> {
        Ok(None)
    }

    /// Get the provider's model family identifier (e.g. "openai" or
    /// "anthropic" for Copilot). Returns `None` for providers that
    /// don't have model families.
    fn model_family(&self) -> Option<&str> {
        None
    }

    /// Whether this provider's models are free or have no known per-token
    /// pricing. When `true`, the cost predictor reports zero-cost
    /// predictions and the budget-degradation state machine is skipped
    /// (there is nothing to degrade). Local Ollama is the canonical
    /// example: model pricing varies across community-published models
    /// and isn't tracked in the catalog, so we treat it as free.
    fn is_free(&self) -> bool {
        false
    }

    /// Report the true input context window (in tokens) for `model` when the
    /// provider can determine it out-of-band (e.g. Ollama's `/api/show`).
    /// Returns `None` when unknown, so callers fall back to the model catalog.
    async fn context_window(&self, _model: &str) -> Option<u64> {
        None
    }

    /// Pre-warm the provider connection for a cheap (Lite) model.
    ///
    /// Called by the [`crate::prewarm::PreWarmManager`] when a session
    /// enters a degradation tier (≥ Tier 2). The default implementation
    /// issues a minimal 1-token completion request via [`Provider::complete`],
    /// which populates the HTTP connection pool and (for providers with
    /// prompt caching) seeds the cache prefix. Providers with a cheaper
    /// warm-up path (e.g. a dedicated health endpoint) can override this.
    ///
    /// The request carries `reincore_prewarm=true` in its extra map so
    /// implementations can distinguish warm-up traffic from real calls
    /// in their billing / logging.
    ///
    /// Errors are swallowed by the caller - pre-warming is best-effort.
    async fn pre_warm(
        &self,
        request: &ChatCompletionRequest,
    ) -> EngineResult<ChatCompletionResponse> {
        // Default: issue the actual completion. The request is already
        // minimal (1 token, trivial prompt) so the cost is negligible.
        self.complete(request).await
    }
}

/// Result of retrying a non-streaming provider response that contained no
/// user-visible text or tool calls. Prior usage is retained for accounting.
pub struct ProviderResponseAttempts {
    pub response: ChatCompletionResponse,
    pub prior_usage: TokenUsage,
    pub all_usage_reported: bool,
    pub retry_count: usize,
}

pub fn chat_message_has_visible_output(message: &ChatMessage) -> bool {
    !message.full_text().is_empty()
        || message.tool_calls.as_ref().is_some_and(|calls| {
            calls
                .iter()
                .any(|call| !crate::is_terminal_notification_tool(&call.function.name))
        })
}

pub fn provider_response_has_visible_output(response: &ChatCompletionResponse) -> bool {
    chat_message_has_visible_output(&response.message)
}

/// Retry a completed but empty provider response without altering the request.
/// The bounded retry happens before gateway accounting and cache write-back.
pub async fn retry_empty_provider_response(
    provider: &dyn Provider,
    request: &ChatCompletionRequest,
    mut response: ChatCompletionResponse,
) -> EngineResult<ProviderResponseAttempts> {
    let mut prior_usage = TokenUsage::default();
    let mut all_usage_reported = true;
    let mut retry_count = 0;
    while !provider_response_has_visible_output(&response)
        && retry_count < EMPTY_PROVIDER_RESPONSE_RETRY_LIMIT
    {
        match response.usage {
            Some(usage) => prior_usage = prior_usage.saturating_add(usage),
            None => all_usage_reported = false,
        }
        retry_count += 1;
        response = provider.complete(request).await?;
    }
    Ok(ProviderResponseAttempts {
        response,
        prior_usage,
        all_usage_reported,
        retry_count,
    })
}

/// A single normalized chunk from a streaming completion.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChatCompletionStreamChunk {
    /// Complete opaque replay snapshot, emitted at the end of a provider stream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_state: Option<super::replay::ProviderReplay>,
    /// Delta content, if any.
    pub delta_content: Option<String>,
    /// Provider-returned reasoning/thinking delta, if the provider exposes it.
    pub delta_reasoning_content: Option<String>,
    /// Delta tool calls, if any.
    pub delta_tool_calls: Option<Vec<ToolCallDelta>>,
    /// Finish reason, present on the final chunk.
    pub finish_reason: Option<String>,
    /// Usage, usually present only on the final chunk.
    pub usage: Option<TokenUsage>,
}

/// A provider-neutral streaming tool update, correlated by its stable call index.
/// OpenAI/Anthropic send fragments; native Ollama calls carry complete snapshots.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallDelta {
    /// Native object-based providers send complete fields, not append-only fragments.
    #[serde(default)]
    pub snapshot: bool,
    pub index: usize,
    pub id: Option<String>,
    pub call_type: Option<String>,
    pub function_name: Option<String>,
    pub function_arguments: Option<String>,
}

impl ToolCallDelta {
    pub fn merge_function_fields(&self, name: &mut String, arguments: &mut String) {
        if self.snapshot {
            // An incomplete snapshot must not borrow fields from an older valid call.
            *name = self.function_name.clone().unwrap_or_default();
            *arguments = self.function_arguments.clone().unwrap_or_default();
            return;
        }
        for (target, incoming) in [
            (name, &self.function_name),
            (arguments, &self.function_arguments),
        ] {
            if let Some(value) = incoming {
                target.push_str(value);
            }
        }
    }
}

/// A streaming provider response with its initial HTTP response headers.
///
/// Rate-limit headers arrive before the SSE body, so retaining them here lets
/// the gateway expose the same metadata for streaming and non-streaming calls.
pub struct ProviderStream {
    pub response_headers: ProviderResponseHeaders,
    pub chunks:
        Pin<Box<dyn futures::Stream<Item = EngineResult<ChatCompletionStreamChunk>> + Send>>,
}

impl ProviderStream {
    /// Build a streaming response from normalized chunks and upstream headers.
    pub fn new<S>(stream: S, response_headers: ProviderResponseHeaders) -> Self
    where
        S: futures::Stream<Item = EngineResult<ChatCompletionStreamChunk>> + Send + 'static,
    {
        Self {
            response_headers,
            chunks: Box::pin(stream),
        }
    }

    /// Build a synthetic stream that has no upstream HTTP response metadata.
    pub fn without_headers<S>(stream: S) -> Self
    where
        S: futures::Stream<Item = EngineResult<ChatCompletionStreamChunk>> + Send + 'static,
    {
        Self::new(stream, ProviderResponseHeaders::new())
    }
}

impl futures::Stream for ProviderStream {
    type Item = EngineResult<ChatCompletionStreamChunk>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.get_mut().chunks.as_mut().poll_next(cx)
    }
}

/// Extract human-readable reasoning that a provider explicitly included in
/// its response. Provider APIs use several incompatible field names and
/// shapes; signatures and opaque/encrypted reasoning blobs are deliberately
/// ignored because they are not a readable model explanation.
pub fn extract_provider_reasoning(raw: &serde_json::Value) -> Option<String> {
    let direct_paths = [
        "/choices/0/message/reasoning_content",
        "/choices/0/message/reasoning",
        "/choices/0/message/reasoning_details",
        "/choices/0/message/thinking",
        "/message/reasoning_content",
        "/message/reasoning",
        "/message/thinking",
        "/reasoning",
    ];
    for path in direct_paths {
        if let Some(text) = readable_reasoning_text(raw.pointer(path)) {
            return Some(text);
        }
    }

    // Anthropic returns thinking as content blocks.
    if let Some(blocks) = raw.get("content").and_then(serde_json::Value::as_array) {
        let text = blocks
            .iter()
            .filter(|block| {
                block.get("type").and_then(serde_json::Value::as_str) == Some("thinking")
            })
            .filter_map(|block| block.get("thinking").and_then(serde_json::Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        if !text.trim().is_empty() {
            return Some(text);
        }
    }

    // Native Gemini responses mark thought parts explicitly.
    if let Some(parts) = raw
        .pointer("/candidates/0/content/parts")
        .and_then(serde_json::Value::as_array)
    {
        let text = parts
            .iter()
            .filter(|part| part.get("thought").and_then(serde_json::Value::as_bool) == Some(true))
            .filter_map(|part| part.get("text").and_then(serde_json::Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        if !text.trim().is_empty() {
            return Some(text);
        }
    }

    None
}

fn readable_reasoning_text(value: Option<&serde_json::Value>) -> Option<String> {
    let value = value?;
    let text = match value {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .or_else(|| {
                        item.get("text")
                            .and_then(serde_json::Value::as_str)
                            .map(str::to_string)
                    })
                    .or_else(|| {
                        item.get("content")
                            .and_then(serde_json::Value::as_str)
                            .map(str::to_string)
                    })
                    .or_else(|| {
                        item.get("summary")
                            .and_then(serde_json::Value::as_str)
                            .map(str::to_string)
                    })
            })
            .collect::<Vec<_>>()
            .join("\n"),
        serde_json::Value::Object(fields) => ["text", "content", "summary"]
            .iter()
            .find_map(|field| fields.get(*field).and_then(serde_json::Value::as_str))
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    };
    (!text.trim().is_empty()).then_some(text)
}

/// Convert an OpenAI-compatible SSE response into normalized streaming
/// chunks. This parser keeps its own byte buffer because HTTP chunks do not
/// align with SSE line boundaries.
pub(crate) async fn openai_compatible_stream(
    response: reqwest::Response,
    provider_name: &'static str,
    replay_scope: Option<String>,
) -> EngineResult<ProviderStream> {
    use futures::StreamExt;

    let status = response.status();
    if !status.is_success() {
        let body = response
            .text()
            .await
            .unwrap_or_else(|error| format!("<failed to read error body: {error}>"));
        return Err(crate::error::EngineError::Provider(format!(
            "{provider_name} returned {status}: {body}"
        )));
    }

    let response_headers = capture_response_headers(response.headers());
    let mut bytes = response.bytes_stream();
    let stream = async_stream::stream! {
        let mut buffer = Vec::<u8>::new();
        let mut done = false;
        let mut replay = super::replay::ReplayAccumulator::default();

        while let Some(next) = bytes.next().await {
            let chunk = match next {
                Ok(chunk) => chunk,
                Err(error) => {
                    yield Err(crate::error::EngineError::from_request_error(error));
                    break;
                }
            };
            buffer.extend_from_slice(&chunk);

            while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                let mut line = buffer.drain(..=newline).collect::<Vec<_>>();
                line.pop();
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                let Ok(line) = std::str::from_utf8(&line) else {
                    yield Err(crate::error::EngineError::Provider(
                        "provider returned non-UTF-8 SSE data".to_string(),
                    ));
                    done = true;
                    break;
                };
                let Some(data) = line.strip_prefix("data:").map(str::trim) else {
                    continue;
                };
                if data == "[DONE]" {
                    done = true;
                    break;
                }
                if data.is_empty() {
                    continue;
                }
                match parse_openai_stream_event(data) {
                    Ok(Some(mut chunk)) => {
                        if replay_scope.is_some() { if let Ok(raw) = serde_json::from_str(data) { replay.compatible(&raw); } }
                        if chunk.finish_reason.is_some() { chunk.provider_state = replay_scope.as_ref().and_then(|scope| replay.snapshot(scope.clone(), false)); }
                        yield Ok(chunk);
                    },
                    Ok(None) => { if replay_scope.is_some() { if let Ok(raw) = serde_json::from_str(data) { replay.compatible(&raw); } } }
                    Err(error) => {
                        yield Err(error);
                        done = true;
                        break;
                    }
                }
            }
            if done {
                break;
            }
        }

        // A few compatible servers omit the trailing newline. Process the
        // final buffered data event instead of silently dropping its usage or
        // finish reason.
        if !done && !buffer.is_empty() {
            let line = String::from_utf8_lossy(&buffer);
            if let Some(data) = line.trim().strip_prefix("data:").map(str::trim) {
                if data != "[DONE]" && !data.is_empty() {
                    match parse_openai_stream_event(data) {
                        Ok(Some(mut chunk)) => {
                        if replay_scope.is_some() { if let Ok(raw) = serde_json::from_str(data) { replay.compatible(&raw); } }
                        if chunk.finish_reason.is_some() { chunk.provider_state = replay_scope.as_ref().and_then(|scope| replay.snapshot(scope.clone(), false)); }
                        yield Ok(chunk);
                    },
                        Ok(None) => { if replay_scope.is_some() { if let Ok(raw) = serde_json::from_str(data) { replay.compatible(&raw); } } }
                        Err(error) => yield Err(error),
                    }
                }
            }
        }
    };
    Ok(ProviderStream::new(stream, response_headers))
}

fn parse_openai_stream_event(data: &str) -> EngineResult<Option<ChatCompletionStreamChunk>> {
    let value: serde_json::Value = serde_json::from_str(data).map_err(|error| {
        crate::error::EngineError::Provider(format!("invalid provider SSE JSON: {error}"))
    })?;
    if let Some(error) = value.get("error") {
        return Err(crate::error::EngineError::Provider(format!(
            "provider stream returned an error: {error}"
        )));
    }

    let choice = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first());
    let delta = choice.and_then(|choice| choice.get("delta"));
    let delta_content = delta
        .and_then(|delta| delta.get("content"))
        .and_then(|content| content.as_str())
        .map(str::to_string);
    let delta_reasoning_content = delta.and_then(|delta| {
        [
            "reasoning_content",
            "reasoning",
            "reasoning_details",
            "thinking",
        ]
        .iter()
        .find_map(|field| {
            delta
                .get(*field)
                .and_then(|value| readable_reasoning_text(Some(value)))
        })
    });
    let delta_tool_calls = delta
        .and_then(|delta| delta.get("tool_calls"))
        .and_then(|calls| calls.as_array())
        .map(|calls| {
            calls
                .iter()
                .map(|call| ToolCallDelta {
                    snapshot: false,
                    index: call
                        .get("index")
                        .and_then(|index| index.as_u64())
                        .unwrap_or(0) as usize,
                    id: call
                        .get("id")
                        .and_then(|id| id.as_str())
                        .map(str::to_string),
                    call_type: call
                        .get("type")
                        .and_then(|kind| kind.as_str())
                        .map(str::to_string),
                    function_name: call
                        .get("function")
                        .and_then(|function| function.get("name"))
                        .and_then(|name| name.as_str())
                        .map(str::to_string),
                    function_arguments: call
                        .get("function")
                        .and_then(|function| function.get("arguments"))
                        .and_then(|arguments| arguments.as_str())
                        .map(str::to_string),
                })
                .collect::<Vec<_>>()
        })
        .filter(|calls| !calls.is_empty());
    let finish_reason = choice
        .and_then(|choice| choice.get("finish_reason"))
        .and_then(|reason| reason.as_str())
        .map(str::to_string);
    let usage = value.get("usage").and_then(parse_openai_stream_usage);

    if delta_content.is_none()
        && delta_reasoning_content.is_none()
        && delta_tool_calls.is_none()
        && finish_reason.is_none()
        && usage.is_none()
    {
        return Ok(None);
    }
    Ok(Some(ChatCompletionStreamChunk {
        provider_state: None,
        delta_content,
        delta_reasoning_content,
        delta_tool_calls,
        finish_reason,
        usage,
    }))
}

/// Whether the caller asked for the provider's longer cache retention window.
///
/// Providers spell the longer window differently - Anthropic `ttl: "1h"`,
/// OpenAI `prompt_cache_retention: "24h"` - so the knob is written in terms of
/// intent and each request builder maps it onto its own wire form. `1h` stays
/// accepted as the original Anthropic-shaped spelling.
///
/// It is opt-in on every provider for the same reason: the longer window costs
/// more per write (Anthropic bills 2x base input against 1.25x for its
/// default), while a cache read refreshes the entry's timer under either, so
/// traffic that re-hits a prefix inside the default window gains nothing but
/// the higher write price. Whether the gaps are long enough to repay it is a
/// property of the caller's traffic that this layer cannot observe.
pub(super) fn extended_cache_retention_requested(request: &ChatCompletionRequest) -> bool {
    let requested = request
        .extra
        .get("reincore_cache_ttl")
        .and_then(|value| value.as_str())
        .map(str::to_owned)
        .or_else(|| std::env::var("REINCORE_CACHE_TTL").ok());
    matches!(
        requested.as_deref().map(str::trim),
        Some("extended") | Some("1h")
    )
}

/// Narrow an OpenAI `allowed_tools` choice down to a plain mode.
///
/// `allowed_tools` restricts which of the declared tools may be called while
/// leaving the tools array byte-identical - the supported way to vary
/// availability without resetting the prompt cache, since tools render first.
/// Only OpenAI and Azure implement it. The other OpenAI-compatible providers
/// reject the unknown shape, so the subset is dropped here and only the mode
/// survives: every tool stays declared, and a call to one that is currently
/// unavailable is refused by the action gate instead.
///
/// Anthropic needs the same treatment but a different output shape, so it maps
/// this in its own builder rather than sharing this helper.
///
/// A `custom` endpoint runs through [`OpenAiProvider`](super::OpenAiProvider)
/// and so keeps the caller's field as sent: ReinCore cannot know what an
/// arbitrary compatible server supports, and the caller chose to send it.
pub(super) fn tool_choice_without_allowed_tools(choice: &serde_json::Value) -> serde_json::Value {
    let Some(map) = choice.as_object() else {
        return choice.clone();
    };
    if map.get("type").and_then(|kind| kind.as_str()) != Some("allowed_tools") {
        return choice.clone();
    }
    match map.get("mode").and_then(|mode| mode.as_str()) {
        Some("required") => serde_json::json!("required"),
        _ => serde_json::json!("auto"),
    }
}

/// Apply OpenAI-family prompt-cache controls to an outgoing request body.
///
/// Only genuine OpenAI and Azure OpenAI deployments implement these fields. The
/// same [`OpenAiProvider`](super::OpenAiProvider) also serves `custom`
/// endpoints, which may be any OpenAI-compatible server, and a strict one
/// rejects unknown body fields - so nothing is sent unless the configured label
/// says the target really is OpenAI or Azure.
pub(super) fn apply_openai_cache_controls(
    body: &mut serde_json::Value,
    config: &ProviderConfig,
    request: &ChatCompletionRequest,
) {
    if !matches!(
        config.label.trim().to_ascii_lowercase().as_str(),
        "openai" | "azure"
    ) {
        return;
    }
    // OpenAI's automatic prompt cache is routing dependent: with no stable key,
    // requests sharing a prefix can land on different backends and miss
    // entirely. ReinCore already derives exactly that prefix identity for its
    // own bookkeeping, so it is reused here rather than recomputed.
    if let Some(prefix_hash) = request
        .extra
        .get("reincore_prefix_hash")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
    {
        body["prompt_cache_key"] = serde_json::json!(prefix_hash);
    }
    if extended_cache_retention_requested(request) {
        body["prompt_cache_retention"] = serde_json::json!("24h");
    }
}

pub(super) fn parse_openai_stream_usage(value: &serde_json::Value) -> Option<TokenUsage> {
    let prompt_tokens = value.get("prompt_tokens")?.as_u64()?;
    let completion_tokens = value.get("completion_tokens")?.as_u64()?;
    let total_tokens = value
        .get("total_tokens")
        .and_then(|tokens| tokens.as_u64())
        .unwrap_or(prompt_tokens.saturating_add(completion_tokens));
    let cached_prompt_tokens = value
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .and_then(|tokens| tokens.as_u64())
        .or_else(|| {
            value
                .get("prompt_cache_hit_tokens")
                .and_then(|v| v.as_u64())
        })
        .unwrap_or(0);
    Some(TokenUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cached_prompt_tokens,
        cache_creation_input_tokens: 0,
        reasoning_tokens: value
            .pointer("/completion_tokens_details/reasoning_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        ..Default::default()
    })
}

#[cfg(test)]
mod stream_tests {
    use super::*;

    fn cache_control_body(
        label: &str,
        extra: std::collections::HashMap<String, serde_json::Value>,
    ) -> serde_json::Value {
        let config = ProviderConfig {
            label: label.to_string(),
            ..Default::default()
        };
        let request = ChatCompletionRequest {
            model: "m".to_string(),
            messages: Vec::new(),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra,
        };
        let mut body = serde_json::json!({});
        apply_openai_cache_controls(&mut body, &config, &request);
        body
    }

    fn extra_with(pairs: &[(&str, &str)]) -> std::collections::HashMap<String, serde_json::Value> {
        pairs
            .iter()
            .map(|(key, value)| (key.to_string(), serde_json::json!(value)))
            .collect()
    }

    #[test]
    fn allowed_tools_is_narrowed_for_providers_that_do_not_implement_it() {
        let choice = serde_json::json!({
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [{"type": "function", "function": {"name": "read_file"}}]
        });
        assert_eq!(
            tool_choice_without_allowed_tools(&choice),
            serde_json::json!("auto")
        );
        let required = serde_json::json!({
            "type": "allowed_tools", "mode": "required", "tools": []
        });
        assert_eq!(
            tool_choice_without_allowed_tools(&required),
            serde_json::json!("required")
        );
    }

    #[test]
    fn ordinary_tool_choices_pass_through_untouched() {
        for choice in [
            serde_json::json!("auto"),
            serde_json::json!("required"),
            serde_json::json!("none"),
            serde_json::json!({"type": "function", "function": {"name": "read_file"}}),
        ] {
            assert_eq!(tool_choice_without_allowed_tools(&choice), choice);
        }
    }

    #[test]
    fn openai_receives_the_prefix_hash_as_a_cache_routing_key() {
        let body = cache_control_body("openai", extra_with(&[("reincore_prefix_hash", "abc123")]));
        assert_eq!(body["prompt_cache_key"], "abc123");
        // The longer retention window is not implied by the routing key.
        assert!(body.get("prompt_cache_retention").is_none());
    }

    #[test]
    fn azure_receives_the_same_controls_as_openai() {
        let body = cache_control_body("azure", extra_with(&[("reincore_prefix_hash", "abc123")]));
        assert_eq!(body["prompt_cache_key"], "abc123");
    }

    #[test]
    fn openai_compatible_endpoints_are_not_sent_openai_only_fields() {
        // `custom` runs through the same provider but may be any compatible
        // server, and a strict one rejects unknown body fields.
        for label in ["custom", "groq", "ollama", ""] {
            let body = cache_control_body(label, extra_with(&[("reincore_prefix_hash", "abc123")]));
            assert!(
                body.get("prompt_cache_key").is_none(),
                "{label} must not receive OpenAI-only cache fields"
            );
        }
    }

    #[test]
    fn extended_retention_is_opt_in_and_maps_to_the_openai_spelling() {
        for spelling in ["extended", "1h"] {
            let body = cache_control_body(
                "openai",
                extra_with(&[
                    ("reincore_prefix_hash", "abc123"),
                    ("reincore_cache_ttl", spelling),
                ]),
            );
            assert_eq!(
                body["prompt_cache_retention"], "24h",
                "{spelling} should request the longer window"
            );
        }
        let unrecognized =
            cache_control_body("openai", extra_with(&[("reincore_cache_ttl", "7d")]));
        assert!(unrecognized.get("prompt_cache_retention").is_none());
    }
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    };

    struct ResponseProvider {
        config: ProviderConfig,
        response: Mutex<Option<ChatCompletionResponse>>,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl Provider for ResponseProvider {
        fn label(&self) -> &str {
            "test"
        }
        fn config(&self) -> &ProviderConfig {
            &self.config
        }
        async fn list_models(&self) -> EngineResult<Vec<ProviderModel>> {
            Ok(vec![])
        }
        async fn complete(
            &self,
            _: &ChatCompletionRequest,
        ) -> EngineResult<ChatCompletionResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.response
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| crate::EngineError::Provider("missing test response".into()))
        }
    }

    fn test_response(content: Option<&str>, prompt_tokens: u64) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: "test".into(),
            model: "test".into(),
            message: serde_json::from_value(
                serde_json::json!({"role":"assistant","content":content}),
            )
            .unwrap(),
            usage: Some(TokenUsage {
                prompt_tokens,
                total_tokens: prompt_tokens,
                ..Default::default()
            }),
            raw: serde_json::json!({}),
            response_headers: Default::default(),
        }
    }

    #[tokio::test]
    async fn retries_one_empty_complete_response_and_retains_prior_usage() {
        let provider = ResponseProvider {
            config: ProviderConfig::default(),
            response: Mutex::new(Some(test_response(Some("Recovered"), 7))),
            calls: AtomicUsize::new(0),
        };
        let request = serde_json::from_value(serde_json::json!({
            "model":"test","messages":[{"role":"user","content":"hello"}]
        }))
        .unwrap();
        let attempts = retry_empty_provider_response(&provider, &request, test_response(None, 5))
            .await
            .unwrap();
        assert_eq!(attempts.retry_count, 1);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(attempts.response.message.full_text(), "Recovered");
        assert_eq!(attempts.prior_usage.prompt_tokens, 5);
        assert!(attempts.all_usage_reported);
    }

    #[test]
    fn internal_terminal_notification_is_not_visible_provider_output() {
        let message = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "terminal-notification",
                "type": "function",
                "function": {
                    "name": crate::TERMINAL_NOTIFICATION_TOOL,
                    "arguments": "{}"
                }
            }]
        }))
        .unwrap();

        assert!(!chat_message_has_visible_output(&message));
    }

    #[test]
    fn parses_content_finish_and_usage_events() {
        let content = parse_openai_stream_event(
            r#"{"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}"#,
        )
        .unwrap()
        .unwrap();
        assert_eq!(content.delta_content.as_deref(), Some("hello"));

        let final_event = parse_openai_stream_event(
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":4}}}"#,
        )
        .unwrap()
        .unwrap();
        assert_eq!(final_event.finish_reason.as_deref(), Some("stop"));
        let usage = final_event.usage.unwrap();
        assert_eq!(usage.total_tokens, 12);
        assert_eq!(usage.cached_prompt_tokens, 4);
    }

    #[test]
    fn parses_reasoning_content_delta() {
        let chunk = parse_openai_stream_event(
            r#"{"choices":[{"delta":{"reasoning_content":"checking constraints"},"finish_reason":null}]}"#,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            chunk.delta_reasoning_content.as_deref(),
            Some("checking constraints")
        );
    }

    #[test]
    fn extracts_only_readable_provider_reasoning() {
        let openai = serde_json::json!({
            "choices": [{"message": {"reasoning_content": "step summary"}}]
        });
        assert_eq!(
            extract_provider_reasoning(&openai).as_deref(),
            Some("step summary")
        );

        let structured = serde_json::json!({
            "choices": [{"message": {"reasoning_details": [{"summary": "checked constraints"}]}}]
        });
        assert_eq!(
            extract_provider_reasoning(&structured).as_deref(),
            Some("checked constraints")
        );

        let anthropic = serde_json::json!({
            "content": [
                {"type": "thinking", "thinking": "considering tools"},
                {"type": "text", "text": "answer"}
            ]
        });
        assert_eq!(
            extract_provider_reasoning(&anthropic).as_deref(),
            Some("considering tools")
        );

        let encrypted_only = serde_json::json!({
            "content": [{"type": "redacted_thinking", "data": "opaque-signature"}]
        });
        assert!(extract_provider_reasoning(&encrypted_only).is_none());
    }

    #[test]
    fn parses_partial_tool_call_by_index() {
        let event = parse_openai_stream_event(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_1","type":"function","function":{"name":"read_","arguments":"{\"pa"}}]},"finish_reason":null}]}"#,
        )
        .unwrap()
        .unwrap();
        let delta = &event.delta_tool_calls.unwrap()[0];
        assert_eq!(delta.index, 1);
        assert_eq!(delta.id.as_deref(), Some("call_1"));
        assert_eq!(delta.function_name.as_deref(), Some("read_"));
        assert_eq!(delta.function_arguments.as_deref(), Some("{\"pa"));
    }

    #[test]
    fn surfaces_stream_error_payloads() {
        let error = parse_openai_stream_event(r#"{"error":{"message":"overloaded"}}"#).unwrap_err();
        assert!(error.to_string().contains("overloaded"));
    }

    #[test]
    fn normalizes_openai_and_anthropic_remaining_headers() {
        let mut openai: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "id": "response-1",
            "model": "test-model",
            "message": {"role": "assistant", "content": "ok"},
            "usage": null,
            "raw": {}
        }))
        .unwrap();
        assert!(openai.response_headers.is_empty());

        openai.response_headers.insert(
            "x-ratelimit-remaining-requests".to_string(),
            "42".to_string(),
        );
        openai.response_headers.insert(
            "x-ratelimit-remaining-tokens".to_string(),
            "8192".to_string(),
        );
        assert_eq!(
            openai.rate_limit_remaining(),
            RateLimitRemaining {
                requests: Some(42),
                tokens: Some(8192),
                input_tokens: None,
                output_tokens: None,
            }
        );
        assert_eq!(openai.header("X-RateLimit-Remaining-Requests"), Some("42"));

        let mut anthropic = openai;
        anthropic.response_headers.clear();
        anthropic.response_headers.insert(
            "anthropic-ratelimit-requests-remaining".to_string(),
            "7".to_string(),
        );
        anthropic.response_headers.insert(
            "anthropic-ratelimit-tokens-remaining".to_string(),
            "12000".to_string(),
        );
        anthropic.response_headers.insert(
            "anthropic-ratelimit-input-tokens-remaining".to_string(),
            "9000".to_string(),
        );
        anthropic.response_headers.insert(
            "anthropic-ratelimit-output-tokens-remaining".to_string(),
            "3000".to_string(),
        );
        assert_eq!(
            anthropic.rate_limit_remaining(),
            RateLimitRemaining {
                requests: Some(7),
                tokens: Some(12000),
                input_tokens: Some(9000),
                output_tokens: Some(3000),
            }
        );
    }

    #[test]
    fn captures_repeated_headers_and_filters_proxy_metadata() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.append(
            "x-ratelimit-remaining-requests",
            reqwest::header::HeaderValue::from_static("10"),
        );
        headers.append(
            "x-ratelimit-remaining-requests",
            reqwest::header::HeaderValue::from_static("9"),
        );
        headers.insert(
            "set-cookie",
            reqwest::header::HeaderValue::from_static("secret=value"),
        );

        let captured = capture_response_headers(&headers);
        assert_eq!(
            captured.get("x-ratelimit-remaining-requests"),
            Some(&"10, 9".to_string())
        );
        assert!(is_rate_limit_response_header(
            "x-ratelimit-remaining-requests"
        ));
        assert!(is_rate_limit_response_header("retry-after"));
        assert!(!is_rate_limit_response_header("set-cookie"));
    }
}

/// A tool executor abstracts how tool calls are fulfilled.
///
/// Implementations include in-process function maps and the configured MCP
/// Streamable HTTP runtime in `crate::mcp`.
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Authorize caching using effective server-side tool/data/credential state.
    /// Never trust caller safe_to_memoize or version strings as authorization.
    /// Return None for mutations, user-sensitive results without proper scope,
    /// unversioned live data, or any tool whose reuse safety is unknown.
    async fn cache_policy(
        &self,
        _tool_call: &ToolCall,
        _context: &crate::tool_cache::ToolCacheContext,
    ) -> EngineResult<Option<crate::tool_cache::ToolCachePolicy>> {
        Ok(None)
    }

    /// Local reuse requires explicit executor approval. data_namespace must bind
    /// the effective repository/working directory; versions and credential IDs
    /// must describe execution, never caller claims. None disables reuse.
    async fn local_cache_policy(
        &self,
        _tool_call: &ToolCall,
        _context: &crate::tool_cache::LocalToolCacheContext,
    ) -> EngineResult<Option<crate::tool_cache::ToolCachePolicy>> {
        Ok(None)
    }

    /// Execute using the same local context as policy resolution. Executors that
    /// opt into local caching must honor this context and their policy namespace.
    async fn execute_local_in_context(
        &self,
        tool_call: &ToolCall,
        _context: &crate::tool_cache::LocalToolCacheContext,
    ) -> EngineResult<crate::models::ToolResult> {
        self.execute(tool_call).await
    }

    /// List tools this executor can fulfill.
    async fn list_tools(&self) -> EngineResult<Vec<ToolDefinition>>;

    async fn list_tools_in_context(
        &self,
        _context: &crate::tool_cache::ToolCacheContext,
    ) -> EngineResult<Vec<ToolDefinition>> {
        self.list_tools().await
    }

    async fn list_local_tools_in_context(
        &self,
        _context: &crate::tool_cache::LocalToolCacheContext,
    ) -> EngineResult<Vec<ToolDefinition>> {
        self.list_tools().await
    }

    /// Execute with the same trusted identity used to resolve cache policy.
    /// Override for tenant-specific credentials/data; never store that context
    /// in mutable executor-global state between policy and execution calls.
    async fn execute_in_context(
        &self,
        tool_call: &ToolCall,
        _context: &crate::tool_cache::ToolCacheContext,
    ) -> EngineResult<crate::models::ToolResult> {
        self.execute(tool_call).await
    }

    /// Execute a single tool call and return its result.
    async fn execute(&self, tool_call: &ToolCall) -> EngineResult<crate::models::ToolResult>;
}

/// A registry of tool executors. The gateway queries each executor to build the
/// tool list exposed to the LLM, then routes returned `tool_calls` to the
/// executor that owns the matching tool name.
#[derive(Clone, Default)]
pub struct ToolExecutorRegistry {
    executors: Vec<std::sync::Arc<dyn ToolExecutor>>,
}

impl ToolExecutorRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(mut self, executor: std::sync::Arc<dyn ToolExecutor>) -> Self {
        self.executors.push(executor);
        self
    }

    /// Aggregate tool definitions from all executors.
    pub async fn list_all_tools(&self) -> EngineResult<Vec<ToolDefinition>> {
        let mut all = Vec::new();
        for executor in &self.executors {
            let mut tools = executor.list_tools().await?;
            all.append(&mut tools);
        }
        Ok(all)
    }

    pub async fn list_tools_in_context(
        &self,
        context: &crate::tool_cache::ToolCacheContext,
    ) -> EngineResult<Vec<ToolDefinition>> {
        let mut all = Vec::new();
        for executor in &self.executors {
            all.extend(executor.list_tools_in_context(context).await?);
        }
        Ok(all)
    }

    pub async fn list_local_tools_in_context(
        &self,
        context: &crate::tool_cache::LocalToolCacheContext,
    ) -> EngineResult<Vec<ToolDefinition>> {
        let mut all = Vec::new();
        for executor in &self.executors {
            all.extend(executor.list_local_tools_in_context(context).await?);
        }
        Ok(all)
    }

    /// Resolve once so policy and execution use the same registered executor.
    /// Without context, duplicate names are ambiguous and must never select the first.
    pub async fn resolve(
        &self,
        tool_call: &ToolCall,
    ) -> EngineResult<std::sync::Arc<dyn ToolExecutor>> {
        self.resolve_scoped(tool_call, None).await
    }

    /// Resolve through each executor's authorized listing before cache policy or execution.
    pub async fn resolve_in_context(
        &self,
        tool_call: &ToolCall,
        context: &crate::tool_cache::ToolCacheContext,
    ) -> EngineResult<std::sync::Arc<dyn ToolExecutor>> {
        self.resolve_scoped(tool_call, Some(context)).await
    }

    pub async fn resolve_local_in_context(
        &self,
        tool_call: &ToolCall,
        context: &crate::tool_cache::LocalToolCacheContext,
    ) -> EngineResult<std::sync::Arc<dyn ToolExecutor>> {
        let mut selected = None;
        for executor in &self.executors {
            let tools = executor.list_local_tools_in_context(context).await?;
            if tools
                .iter()
                .any(|tool| tool.name == tool_call.function.name)
            {
                crate::action_gate::validate_call(tool_call, &tools)?;
                if selected.is_some() {
                    return Err(crate::EngineError::ToolExecution(
                        "ambiguous tool name in execution scope".into(),
                    ));
                }
                selected = Some(executor.clone());
            }
        }
        selected.ok_or_else(|| {
            crate::EngineError::ToolExecution("tool not authorized in local execution scope".into())
        })
    }

    async fn resolve_scoped(
        &self,
        tool_call: &ToolCall,
        context: Option<&crate::tool_cache::ToolCacheContext>,
    ) -> EngineResult<std::sync::Arc<dyn ToolExecutor>> {
        let mut selected = None;
        for executor in &self.executors {
            let tools = match context {
                Some(context) => executor.list_tools_in_context(context).await?,
                None => executor.list_tools().await?,
            };
            if tools.iter().any(|t| t.name == tool_call.function.name) {
                crate::action_gate::validate_call(tool_call, &tools)?;
                if selected.is_some() {
                    return Err(crate::error::EngineError::ToolExecution(
                        "ambiguous tool name in execution scope".into(),
                    ));
                }
                selected = Some(executor.clone());
            }
        }
        selected.ok_or_else(|| {
            crate::error::EngineError::Provider(format!(
                "no executor registered for tool {}",
                tool_call.function.name
            ))
        })
    }

    pub async fn execute(&self, tool_call: &ToolCall) -> EngineResult<crate::models::ToolResult> {
        self.resolve(tool_call).await?.execute(tool_call).await
    }
}

/// Check whether `model` is one of the models a provider owns: its standard
/// model or any explicit tier mapping (case-insensitive).
fn owns_model(config: &ProviderConfig, model: &str) -> bool {
    config
        .standard_model
        .as_deref()
        .is_some_and(|owned| owned.eq_ignore_ascii_case(model))
        || config
            .models
            .iter()
            .any(|mapping| mapping.provider_model.eq_ignore_ascii_case(model))
}

/// A registry that selects a provider based on request hints.
#[derive(Clone, Default)]
pub struct ProviderRegistry {
    providers: Vec<std::sync::Arc<dyn Provider>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(mut self, provider: std::sync::Arc<dyn Provider>) -> Self {
        self.providers.push(provider);
        self
    }

    /// Return the registered providers in routing order, optionally moving a
    /// configured primary provider to the front. Control-plane JWTs preserve
    /// project priority order, so callers can use this list as an explicit
    /// cross-provider failover chain without exposing the registry internals.
    pub fn ordered_providers(
        &self,
        primary_label: Option<&str>,
    ) -> EngineResult<Vec<std::sync::Arc<dyn Provider>>> {
        let mut providers = self.providers.clone();
        if let Some(label) = primary_label {
            let Some(index) = providers
                .iter()
                .position(|provider| provider.label().eq_ignore_ascii_case(label))
            else {
                return Err(crate::error::EngineError::Provider(format!(
                    "provider '{}' not registered for this user",
                    label
                )));
            };
            let primary = providers.remove(index);
            providers.insert(0, primary);
        }
        if providers.is_empty() {
            return Err(crate::error::EngineError::Provider(
                "no provider registered".to_string(),
            ));
        }
        Ok(providers)
    }

    /// Find the registered provider owning `model` (standard model or tier
    /// mapping), if any. Used to default provider selection for clients
    /// that send model ids (e.g. Muse Code) without an explicit header.
    pub fn provider_owning_model(&self, model: &str) -> Option<std::sync::Arc<dyn Provider>> {
        self.providers
            .iter()
            .find(|p| owns_model(p.config(), model))
            .cloned()
    }

    /// Pick a provider. The request `model` field can hint a provider label
    /// (e.g. "openai/gpt-4o"). A request naming a model owned by a provider
    /// (e.g. Muse Code sending `muse-spark-1.3`) routes to that provider, so
    /// Muse traffic defaults to Muse models once Meta is configured.
    /// Otherwise the first registered provider wins.
    pub fn select(
        &self,
        request: &ChatCompletionRequest,
    ) -> EngineResult<std::sync::Arc<dyn Provider>> {
        let hint = request.model.split('/').next().unwrap_or(&request.model);
        let provider = self
            .providers
            .iter()
            .find(|p| p.label().eq_ignore_ascii_case(hint))
            .cloned()
            .or_else(|| self.provider_owning_model(&request.model))
            .or_else(|| self.providers.first().cloned())
            .ok_or_else(|| {
                crate::error::EngineError::Provider("no provider registered".to_string())
            })?;
        Ok(provider)
    }

    /// Pick a provider with an explicit label override. If `provider_label`
    /// is `Some`, the registry looks up that provider directly - useful for
    /// the `X-ReinCore-Provider` header which forces a specific provider
    /// regardless of the model field. If the override is `None` or doesn't
    /// match a registered provider, falls back to `select()` behavior.
    pub fn select_with_override(
        &self,
        request: &ChatCompletionRequest,
        provider_label: Option<&str>,
    ) -> EngineResult<std::sync::Arc<dyn Provider>> {
        if let Some(label) = provider_label {
            if let Some(provider) = self.get_by_label(label) {
                return Ok(provider);
            }
            // Override was specified but not found - that's an error, not a
            // fallback, so the caller knows their header was invalid.
            return Err(crate::error::EngineError::Provider(format!(
                "provider '{}' not registered for this user",
                label
            )));
        }
        self.select(request)
    }

    /// Aggregate models from all registered providers.
    pub async fn list_all_models(&self) -> EngineResult<Vec<ProviderModel>> {
        let mut all = Vec::new();
        for provider in &self.providers {
            let mut models = provider.list_models().await?;
            all.append(&mut models);
        }
        Ok(all)
    }

    /// Return true if no providers are registered.
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Return the number of registered providers.
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Return whether callers must disambiguate between multiple providers.
    pub fn is_multi_provider(&self) -> bool {
        self.providers.len() > 1
    }

    /// Iterate over registered providers.
    pub fn iter(&self) -> impl Iterator<Item = &std::sync::Arc<dyn Provider>> {
        self.providers.iter()
    }

    /// Find a provider by label without removing it.
    pub fn get_by_label(&self, label: &str) -> Option<std::sync::Arc<dyn Provider>> {
        self.providers
            .iter()
            .find(|p| p.label().eq_ignore_ascii_case(label))
            .cloned()
    }

    /// Remove and return a provider by label, giving sole ownership to the caller.
    pub fn take_by_label(&mut self, label: &str) -> Option<std::sync::Arc<dyn Provider>> {
        let pos = self
            .providers
            .iter()
            .position(|p| p.label().eq_ignore_ascii_case(label))?;
        Some(self.providers.remove(pos))
    }
}

#[cfg(test)]
mod hybrid_config_tests {
    use super::*;

    struct StubProvider {
        config: ProviderConfig,
    }

    #[async_trait]
    impl Provider for StubProvider {
        fn label(&self) -> &str {
            &self.config.label
        }

        fn config(&self) -> &ProviderConfig {
            &self.config
        }

        async fn complete(
            &self,
            _request: &ChatCompletionRequest,
        ) -> EngineResult<ChatCompletionResponse> {
            Err(crate::error::EngineError::Provider(
                "stub provider does not complete requests".to_string(),
            ))
        }

        async fn list_models(&self) -> EngineResult<Vec<ProviderModel>> {
            Ok(Vec::new())
        }
    }

    fn config(standard_model: Option<&str>, models: Vec<ProviderModelMapping>) -> ProviderConfig {
        ProviderConfig {
            label: "openai".to_string(),
            base_url: "https://example.invalid".to_string(),
            api_key: "test".to_string(),
            standard_model: standard_model.map(str::to_string),
            models,
            ..Default::default()
        }
    }

    #[test]
    fn explicit_tier_override_has_highest_precedence() {
        let config = config(
            Some("gpt-5.4-2026-03-05"),
            vec![
                ProviderModelMapping {
                    tier: ModelTier::Lite,
                    provider_model: "my-lite-fine-tune".to_string(),
                },
                ProviderModelMapping {
                    tier: ModelTier::Pro,
                    provider_model: "my-pro-fine-tune".to_string(),
                },
                ProviderModelMapping {
                    tier: ModelTier::Ultra,
                    provider_model: "my-ultra-fine-tune".to_string(),
                },
            ],
        );

        assert_eq!(
            config.model_for_tier(ModelTier::Lite).unwrap(),
            "my-lite-fine-tune"
        );
        assert_eq!(
            config.model_for_tier(ModelTier::Pro).unwrap(),
            "my-pro-fine-tune"
        );
        assert_eq!(
            config.model_for_tier(ModelTier::Ultra).unwrap(),
            "my-ultra-fine-tune"
        );
    }

    #[test]
    fn standard_model_automatically_degrades_only_lite() {
        let config = config(Some("gpt-5.4-2026-03-05"), vec![]);

        assert_eq!(
            config.model_for_tier(ModelTier::Lite).unwrap(),
            "gpt-5-nano-2025-08-07"
        );
        assert_eq!(
            config.model_for_tier(ModelTier::Ultra).unwrap(),
            "gpt-5.4-2026-03-05"
        );
    }

    #[test]
    fn legacy_pro_mapping_can_anchor_missing_tiers() {
        let config = config(
            None,
            vec![ProviderModelMapping {
                tier: ModelTier::Pro,
                provider_model: "gpt-5.4-2026-03-05".to_string(),
            }],
        );

        assert_eq!(
            config.model_for_tier(ModelTier::Lite).unwrap(),
            "gpt-5-nano-2025-08-07"
        );
        assert_eq!(
            config.model_for_tier(ModelTier::Ultra).unwrap(),
            "gpt-5.4-2026-03-05"
        );
    }

    #[test]
    fn old_serialized_config_without_standard_model_remains_valid() {
        let config: ProviderConfig = serde_json::from_value(serde_json::json!({
            "label": "openai",
            "base_url": "https://example.invalid",
            "api_key": "test",
            "models": [{"tier": "lite", "provider_model": "legacy-lite"}]
        }))
        .unwrap();

        assert_eq!(config.standard_model, None);
        assert_eq!(config.timeout_seconds, 300);
        assert_eq!(
            config.model_for_tier(ModelTier::Lite).unwrap(),
            "legacy-lite"
        );
    }

    #[test]
    fn base_model_alias_is_accepted_from_control_plane_payloads() {
        let config: ProviderConfig = serde_json::from_value(serde_json::json!({
            "label": "custom",
            "base_url": "https://example.invalid",
            "api_key": "test",
            "base_model": "frontier-v2"
        }))
        .unwrap();

        assert_eq!(config.standard_model.as_deref(), Some("frontier-v2"));
        assert_eq!(
            config.model_for_tier(ModelTier::Lite).unwrap(),
            "frontier-v2"
        );
    }

    #[test]
    fn registry_reports_when_provider_header_disambiguation_is_needed() {
        let registry = ProviderRegistry::new()
            .register(std::sync::Arc::new(StubProvider {
                config: config(Some("gpt-5.4-2026-03-05"), vec![]),
            }))
            .register(std::sync::Arc::new(StubProvider {
                config: ProviderConfig {
                    label: "anthropic".to_string(),
                    ..config(Some("claude-sonnet-5"), vec![])
                },
            }));

        assert_eq!(registry.len(), 2);
        assert!(registry.is_multi_provider());
    }

    fn select_request(model: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: model.to_string(),
            messages: Vec::new(),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn select_routes_owned_model_to_its_provider() {
        fn stub(label: &str, standard: &str) -> StubProvider {
            StubProvider {
                config: ProviderConfig {
                    label: label.to_string(),
                    standard_model: Some(standard.to_string()),
                    ..config(None, vec![])
                },
            }
        }
        let registry = ProviderRegistry::new()
            .register(std::sync::Arc::new(stub("ollama", "kimi-k3:cloud")))
            .register(std::sync::Arc::new(stub("meta", "muse-spark-1.3")));

        // A request naming a Muse model defaults to the Meta provider even
        // though Ollama is registered first.
        assert_eq!(
            registry
                .select(&select_request("muse-spark-1.3"))
                .unwrap()
                .label(),
            "meta"
        );
        // Tier mappings count as owned models too.
        assert_eq!(
            registry
                .select(&select_request("kimi-k3:cloud"))
                .unwrap()
                .label(),
            "ollama"
        );
        // Unknown models still fall back to the first registered provider.
        assert_eq!(
            registry
                .select(&select_request("something-else"))
                .unwrap()
                .label(),
            "ollama"
        );
    }

    fn routing_config(
        standard_model: Option<&str>,
        models: Vec<ProviderModelMapping>,
        mode: crate::routing::RoutingMode,
    ) -> ProviderConfig {
        ProviderConfig {
            label: "openai".to_string(),
            base_url: "https://example.invalid".to_string(),
            api_key: "test".to_string(),
            standard_model: standard_model.map(str::to_string),
            models,
            routing: crate::routing::RoutingPolicy {
                mode,
                ..crate::routing::RoutingPolicy::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn custom_mode_never_derives_lite_from_catalog() {
        let config = routing_config(
            Some("gpt-5.4-2026-03-05"),
            vec![],
            crate::routing::RoutingMode::Custom,
        );

        // In auto mode Lite would derive to gpt-5-nano-2025-08-07; in custom
        // mode the unmapped tier falls back to the standard model.
        assert_eq!(
            config.model_for_tier(ModelTier::Lite).unwrap(),
            "gpt-5.4-2026-03-05"
        );
        assert_eq!(
            config.model_for_tier(ModelTier::Ultra).unwrap(),
            "gpt-5.4-2026-03-05"
        );
    }

    #[test]
    fn custom_mode_still_honors_explicit_tier_overrides() {
        let config = routing_config(
            Some("gpt-5.4-2026-03-05"),
            vec![ProviderModelMapping {
                tier: ModelTier::Lite,
                provider_model: "pinned-lite".to_string(),
            }],
            crate::routing::RoutingMode::Custom,
        );

        assert_eq!(
            config.model_for_tier(ModelTier::Lite).unwrap(),
            "pinned-lite"
        );
        // Unmapped Ultra still falls back to the standard, not the catalog.
        assert_eq!(
            config.model_for_tier(ModelTier::Ultra).unwrap(),
            "gpt-5.4-2026-03-05"
        );
    }

    #[test]
    fn locked_mode_pins_every_tier_to_standard_model() {
        let config = routing_config(
            Some("gpt-5.4-2026-03-05"),
            vec![],
            crate::routing::RoutingMode::Locked,
        );

        for tier in [ModelTier::Lite, ModelTier::Pro, ModelTier::Ultra] {
            assert_eq!(
                config.model_for_tier(tier).unwrap(),
                "gpt-5.4-2026-03-05",
                "locked mode should resolve {tier:?} to the standard model"
            );
        }
    }

    #[test]
    fn locked_mode_still_honors_explicit_pro_override() {
        // An explicit Pro mapping is the highest-precedence override and wins
        // even in locked mode, so a user can pin "locked" to a specific model.
        let config = routing_config(
            None,
            vec![ProviderModelMapping {
                tier: ModelTier::Pro,
                provider_model: "pinned-standard".to_string(),
            }],
            crate::routing::RoutingMode::Locked,
        );

        assert_eq!(
            config.model_for_tier(ModelTier::Lite).unwrap(),
            "pinned-standard"
        );
        assert_eq!(
            config.model_for_tier(ModelTier::Ultra).unwrap(),
            "pinned-standard"
        );
    }
}
