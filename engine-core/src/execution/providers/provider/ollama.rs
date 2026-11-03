//! Ollama provider adapter.
//!
//! Talks to a local (or remote) Ollama server using its native API at
//! `<base_url>/api/chat` for chat completions and `<base_url>/api/tags` for
//! listing installed models.
//!
//! Two deployment modes are supported:
//!
//! - **Local** (`OllamaMode::Local`): talks to a self-hosted Ollama server
//!   (default `http://localhost:11434`). No authentication; the `api_key`
//!   field is ignored. The user picks a concrete model from the list of
//!   installed models (`/api/tags`).
//!
//! - **Cloud** (`OllamaMode::Cloud`): talks to a hosted Ollama endpoint. The
//!   user supplies an API key and a default cloud model name, which is
//!   validated against the endpoint's `/api/tags` listing before the
//!   provider is constructed.
//!
//! For OpenAI-compatible deployments of Ollama you can also use the
//! `OpenAiProvider` with `base_url` set to `<ollama_url>/v1`.

use crate::error::{EngineError, EngineResult};
use crate::models::{
    ChatCompletionRequest, ChatMessage, MessageContent, ModelTier, ToolCall, ToolDefinition,
};
use crate::provider::{
    capture_response_headers, ChatCompletionResponse, ChatCompletionStreamChunk, Provider,
    ProviderConfig, ProviderModel, ProviderStream, TokenUsage, ToolCallDelta,
};
use async_trait::async_trait;
use futures::StreamExt;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use tracing::debug;

/// A model advertised by an Ollama endpoint, with the display metadata the
/// setup UIs surface (the parameter size guides tier selection).
#[derive(Clone, Debug)]
pub struct OllamaModelInfo {
    /// The model tag used on the wire (e.g. `llama3.1:8b`).
    pub name: String,
    /// Human-readable parameter size (e.g. `8.0B`), when advertised.
    pub parameter_size: Option<String>,
}

/// Derive a size label (e.g. `120B`) from a model tag such as
/// `gpt-oss:120b-cloud`, for Ollama Cloud models whose `/api/tags` details
/// are empty. Returns `None` when the tag carries no size.
fn size_from_name(name: &str) -> Option<String> {
    let tag = name.rsplit_once(':')?.1;
    let bytes = tag.as_bytes();
    let mut i = 0;
    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
        i += 1;
    }
    if i == 0 || i >= bytes.len() {
        return None;
    }
    let unit = bytes[i].to_ascii_lowercase();
    if unit != b'b' && unit != b't' {
        return None;
    }
    Some(format!(
        "{}{}",
        &tag[..i],
        (unit as char).to_ascii_uppercase()
    ))
}

/// Model metadata returned by Ollama's `/api/show` endpoint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct OllamaModelMetadata {
    context_length: Option<u64>,
    /// `None` means the endpoint did not expose enough information to decide.
    supports_vision: Option<bool>,
}

/// Process-wide cache of `/api/show` metadata, keyed by `base_url\u{1f}model`.
/// Negative lookups are cached too, so an endpoint that doesn't expose
/// `/api/show` is queried at most once per model.
fn model_metadata_cache() -> &'static Mutex<HashMap<String, OllamaModelMetadata>> {
    static CACHE: OnceLock<Mutex<HashMap<String, OllamaModelMetadata>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Deployment mode for the Ollama provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OllamaMode {
    /// Self-hosted Ollama server (no auth, model picked from installed list).
    Local,
    /// Hosted Ollama endpoint (API key required, default model validated).
    Cloud,
}

impl OllamaMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            OllamaMode::Local => "local",
            OllamaMode::Cloud => "cloud",
        }
    }
}

/// Ollama provider implementation using the native Ollama API.
#[derive(Debug, Clone)]
pub struct OllamaProvider {
    config: ProviderConfig,
    client: reqwest::Client,
    /// Local vs. cloud deployment mode.
    mode: OllamaMode,
}

impl OllamaProvider {
    /// Create a new Ollama provider from configuration.
    ///
    /// `base_url` should point at the Ollama server root, e.g.
    /// `http://localhost:11434` for local mode or the hosted endpoint URL
    /// for cloud mode. `api_key` is only used in cloud mode.
    pub fn new(config: ProviderConfig, mode: OllamaMode) -> EngineResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            // Separate from the overall timeout: a refused or unreachable
            // endpoint should fail in seconds, not after a generation-length wait.
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| EngineError::Provider(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            config,
            client,
            mode,
        })
    }

    /// Return the deployment mode (local or cloud).
    pub fn mode(&self) -> OllamaMode {
        self.mode
    }

    /// Fetch the list of models advertised by the Ollama endpoint
    /// (`/api/tags`) without constructing a full provider. Used by the
    /// CLI to validate a cloud model name and to present the local
    /// installed-models picker.
    pub async fn fetch_models(base_url: &str, api_key: Option<&str>) -> EngineResult<Vec<String>> {
        crate::privacy::current_policy().check_provider("ollama")?;
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| EngineError::Provider(format!("failed to build HTTP client: {e}")))?;
        let mut req = client
            .get(format!("{}/api/tags", base_url.trim_end_matches('/')))
            .header("Content-Type", "application/json");
        if let Some(key) = api_key {
            if !key.is_empty() {
                req = req.header("Authorization", format!("Bearer {}", key));
            }
        }
        let response = req.send().await.map_err(EngineError::from_request_error)?;
        let status = response.status();
        let raw = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| EngineError::Provider(format!("failed to read response: {e}")))?;
        if !status.is_success() {
            return Err(EngineError::Provider(format!(
                "Ollama returned {status} on /api/tags: {}",
                crate::privacy::current_policy().sanitize_for_logging(&raw.to_string())
            )));
        }
        let models = raw
            .get("models")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.get("name").and_then(|n| n.as_str()).map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Ok(models)
    }

    /// Like [`fetch_models`](Self::fetch_models) but also returns each model's
    /// advertised `parameter_size`, so setup UIs can show model capability
    /// when the user picks per-tier models.
    pub async fn fetch_models_detailed(
        base_url: &str,
        api_key: Option<&str>,
    ) -> EngineResult<Vec<OllamaModelInfo>> {
        crate::privacy::current_policy().check_provider("ollama")?;
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| EngineError::Provider(format!("failed to build HTTP client: {e}")))?;
        let mut req = client
            .get(format!("{}/api/tags", base_url.trim_end_matches('/')))
            .header("Content-Type", "application/json");
        if let Some(key) = api_key {
            if !key.is_empty() {
                req = req.header("Authorization", format!("Bearer {}", key));
            }
        }
        let response = req.send().await.map_err(EngineError::from_request_error)?;
        let status = response.status();
        let raw = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| EngineError::Provider(format!("failed to read response: {e}")))?;
        if !status.is_success() {
            return Err(EngineError::Provider(format!(
                "Ollama returned {status} on /api/tags: {}",
                crate::privacy::current_policy().sanitize_for_logging(&raw.to_string())
            )));
        }
        let models = raw
            .get("models")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        let name = v.get("name").and_then(|n| n.as_str())?.to_string();
                        let parameter_size = v
                            .get("details")
                            .and_then(|d| d.get("parameter_size"))
                            .and_then(|p| p.as_str())
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(String::from)
                            // Ollama Cloud omits details for remote models; the
                            // size lives in the tag (e.g. "gpt-oss:120b").
                            .or_else(|| size_from_name(&name));
                        Some(OllamaModelInfo {
                            name,
                            parameter_size,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Ok(models)
    }

    /// Resolve `/api/show` metadata for `model`, caching the result so context
    /// window and vision checks share a single endpoint request.
    async fn resolve_model_metadata(&self, model: &str) -> OllamaModelMetadata {
        let key = format!("{}\u{1f}{}", self.config.base_url, model);
        if let Ok(cache) = model_metadata_cache().lock() {
            if let Some(cached) = cache.get(&key) {
                return *cached;
            }
        }
        let value = self.fetch_model_metadata(model).await.unwrap_or_default();
        if let Ok(mut cache) = model_metadata_cache().lock() {
            cache.insert(key, value);
        }
        value
    }

    /// Query `/api/show` and extract both the context window and image-input
    /// capability. Older Ollama versions may omit `capabilities`; in that
    /// case legacy projector/model-info markers can prove vision support, but
    /// their absence is treated as unknown rather than unsupported.
    async fn fetch_model_metadata(&self, model: &str) -> Option<OllamaModelMetadata> {
        let mut req = self
            .client
            .post(format!("{}/api/show", self.config.base_url))
            .json(&serde_json::json!({ "model": model, "name": model }));
        if self.mode == OllamaMode::Cloud && !self.config.api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", self.config.api_key));
        }
        let response = req.send().await.ok()?;
        if !response.status().is_success() {
            return None;
        }
        let raw = response.json::<serde_json::Value>().await.ok()?;
        Some(parse_ollama_model_metadata(&raw))
    }

    async fn ensure_image_support(
        &self,
        request: &ChatCompletionRequest,
        model: &str,
    ) -> EngineResult<()> {
        if !request_contains_images(request) {
            return Ok(());
        }

        match self.resolve_model_metadata(model).await.supports_vision {
            Some(true) => Ok(()),
            Some(false) => Err(unsupported_vision_error(model)),
            None => {
                debug!(
                    "ollama: could not determine whether model '{}' supports images; allowing Ollama to validate the request",
                    model
                );
                Ok(())
            }
        }
    }

    /// Validate that `model` is present in the endpoint's `/api/tags`
    /// listing. Used by the CLI when the user supplies a default cloud
    /// model - fail fast at config time rather than on the first request.
    pub async fn validate_model(
        base_url: &str,
        api_key: Option<&str>,
        model: &str,
    ) -> EngineResult<()> {
        let models = Self::fetch_models(base_url, api_key).await?;
        // Ollama model names sometimes include a trailing `:latest` tag; match
        // case-insensitively on the base name (before any `:`) so a user
        // typing `llama3` still validates against `llama3:latest`.
        let normalize = |s: &str| s.split(':').next().unwrap_or(s).to_lowercase();
        let want = normalize(model);
        if models.iter().any(|m| normalize(m) == want) {
            Ok(())
        } else {
            Err(EngineError::Provider(format!(
                "model '{}' is not available at {} (available: {})",
                model,
                base_url,
                models.join(", ")
            )))
        }
    }

    fn build_body(
        &self,
        request: &ChatCompletionRequest,
        provider_model: &str,
    ) -> EngineResult<serde_json::Value> {
        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(|message| {
                let mut value = ollama_message_from_chat_message(message)?;
                if super::replay::enabled(request) {
                    super::replay::apply(
                        message,
                        &mut value,
                        &super::replay::scope(&self.config, provider_model),
                    );
                }
                if message.role == "tool" {
                    // OpenAI results carry an invocation ID; Ollama needs the
                    // function name. Resolve each result independently so parallel
                    // calls and out-of-order results retain their association.
                    let name = message
                        .tool_call_id
                        .as_ref()
                        .and_then(|id| {
                            request
                                .messages
                                .iter()
                                .rev()
                                .filter_map(|message| message.tool_calls.as_ref())
                                .flatten()
                                .find(|call| &call.id == id)
                                .map(|call| call.function.name.as_str())
                        })
                        .or(message.name.as_deref());
                    if let Some(name) = name {
                        value["tool_name"] = serde_json::json!(name);
                    }
                }
                Ok(value)
            })
            .collect::<EngineResult<_>>()?;

        let mut body = serde_json::json!({
            "model": provider_model,
            "messages": messages,
            "stream": false,
        });

        // Thinking-capable Ollama models enable their reasoning trace by
        // default. A title does not need that trace, and a short num_predict
        // budget can otherwise be consumed before message.content starts.
        if request
            .extra
            .get(crate::title_generation::TITLE_REQUEST_EXTRA_KEY)
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        {
            body["think"] = serde_json::json!(false);
        }

        if let Some(tools) = &request.tools {
            let tool_specs: Vec<serde_json::Value> =
                tools.iter().map(ollama_tool_definition).collect();
            body["tools"] = serde_json::Value::Array(tool_specs);
        }

        if let Some(temperature) = request.temperature {
            body["options"]["temperature"] = serde_json::json!(temperature);
        }

        if let Some(max_tokens) = request.max_tokens {
            // Ollama exposes this as `num_predict` under the `options` object.
            body["options"]["num_predict"] = serde_json::json!(max_tokens);
        }

        Ok(body)
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn label(&self) -> &str {
        &self.config.label
    }

    fn config(&self) -> &ProviderConfig {
        &self.config
    }

    async fn complete(
        &self,
        request: &ChatCompletionRequest,
    ) -> EngineResult<ChatCompletionResponse> {
        let prepared_request = crate::privacy::prepare_provider_request(request, self.label())?;
        let request = &prepared_request;
        // Ollama does not have tier-based model degradation. All tiers map to
        // the same user-selected model (the only one configured in
        // `ProviderConfig.models`). We still honor `reincore_model_tier` for
        // logging/auditing, but always resolve to the single configured model.
        let tier = request
            .extra
            .get("reincore_model_tier")
            .and_then(|v| v.as_str())
            .and_then(|value| value.parse().ok())
            .unwrap_or(ModelTier::Pro);

        let provider_model = self.config.model_for_request(request, tier)?;
        self.ensure_image_support(request, &provider_model).await?;
        let sent_images = request_contains_images(request);
        let body = self.build_body(request, &provider_model)?;

        debug!(
            "Ollama request to {}/api/chat: {}",
            self.config.base_url,
            crate::privacy::current_policy()
                .sanitize_for_logging(&serde_json::to_string(&body).unwrap_or_default())
        );

        let mut req = self
            .client
            .post(format!("{}/api/chat", self.config.base_url))
            .header("Content-Type", "application/json")
            .json(&body);

        // Cloud deployments require an API key; local Ollama ignores it.
        if self.mode == OllamaMode::Cloud && !self.config.api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", self.config.api_key));
        }

        let response = req.send().await.map_err(EngineError::from_request_error)?;

        let status = response.status();
        let response_headers = capture_response_headers(response.headers());
        let raw = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| EngineError::Provider(format!("failed to read response: {e}")))?;

        if !status.is_success() {
            return Err(ollama_response_error(
                status,
                &raw.to_string(),
                &provider_model,
                sent_images,
            ));
        }

        let mut parsed = parse_completion_response(raw)?;
        super::replay::capture_response(
            &mut parsed,
            super::replay::request_scope(
                &self.config,
                body["model"].as_str().unwrap_or_default(),
                request,
            ),
            false,
        );
        parsed.response_headers = response_headers;
        Ok(parsed)
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn complete_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> EngineResult<ProviderStream> {
        let prepared_request = crate::privacy::prepare_provider_request(request, self.label())?;
        let request = &prepared_request;
        let tier = request
            .extra
            .get("reincore_model_tier")
            .and_then(|value| value.as_str())
            .and_then(|value| value.parse().ok())
            .unwrap_or(ModelTier::Pro);
        let provider_model = self.config.model_for_request(request, tier)?;
        self.ensure_image_support(request, &provider_model).await?;
        let sent_images = request_contains_images(request);
        let mut body = self.build_body(request, &provider_model)?;
        body["stream"] = serde_json::json!(true);

        let mut builder = self
            .client
            .post(format!("{}/api/chat", self.config.base_url))
            .header("Content-Type", "application/json")
            .json(&body);
        if self.mode == OllamaMode::Cloud && !self.config.api_key.is_empty() {
            builder = builder.header("Authorization", format!("Bearer {}", self.config.api_key));
        }
        let response = builder
            .send()
            .await
            .map_err(EngineError::from_request_error)?;
        ollama_stream(
            response,
            &provider_model,
            sent_images,
            super::replay::request_scope(&self.config, &provider_model, request),
        )
        .await
    }

    async fn list_models(&self) -> EngineResult<Vec<ProviderModel>> {
        crate::privacy::current_policy().check_provider(self.label())?;
        let mut req = self
            .client
            .get(format!("{}/api/tags", self.config.base_url))
            .header("Content-Type", "application/json");

        if self.mode == OllamaMode::Cloud && !self.config.api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", self.config.api_key));
        }

        let response = req.send().await.map_err(EngineError::from_request_error)?;

        let status = response.status();
        let raw = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| EngineError::Provider(format!("failed to read response: {e}")))?;

        if !status.is_success() {
            return Err(EngineError::Provider(format!(
                "Ollama returned {status} on /api/tags: {}",
                crate::privacy::current_policy().sanitize_for_logging(&raw.to_string())
            )));
        }

        let models = raw
            .get("models")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        let id = v.get("name").and_then(|v| v.as_str())?;
                        let display_name = v
                            .get("details")
                            .and_then(|d| d.get("family"))
                            .and_then(|v| v.as_str())
                            .map(String::from);
                        let created_at = v
                            .get("modified_at")
                            .and_then(|v| v.as_str())
                            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                            .map(|dt| dt.to_utc());
                        Some(ProviderModel {
                            id: id.to_string(),
                            provider: self.label().to_string(),
                            display_name,
                            created_at,
                            metadata: Some(v.clone()),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(models)
    }

    /// Ollama (both local and cloud) is treated as a free provider for
    /// cost-prediction and budget-degradation purposes. Per-token pricing
    /// for community Ollama models isn't tracked in the model catalog, so
    /// there is no meaningful budget tier to degrade through.
    fn is_free(&self) -> bool {
        true
    }

    async fn context_window(&self, model: &str) -> Option<u64> {
        self.resolve_model_metadata(model).await.context_length
    }
}

fn parse_ollama_model_metadata(raw: &serde_json::Value) -> OllamaModelMetadata {
    let model_info = raw.get("model_info").and_then(|value| value.as_object());
    let context_length = model_info.and_then(|info| {
        info.iter()
            .find(|(key, _)| key.ends_with(".context_length"))
            .and_then(|(_, value)| value.as_u64())
    });

    let advertised_vision = raw
        .get("capabilities")
        .and_then(|value| value.as_array())
        .map(|capabilities| {
            capabilities.iter().any(|capability| {
                capability
                    .as_str()
                    .is_some_and(|value| value.eq_ignore_ascii_case("vision"))
            })
        });
    let legacy_vision = raw
        .get("projector_info")
        .and_then(|value| value.as_object())
        .is_some_and(|info| !info.is_empty())
        || model_info.is_some_and(|info| info.keys().any(|key| key.contains(".vision.")));

    OllamaModelMetadata {
        context_length,
        supports_vision: advertised_vision.or(legacy_vision.then_some(true)),
    }
}

fn request_contains_images(request: &ChatCompletionRequest) -> bool {
    request.messages.iter().any(|message| {
        message
            .content
            .as_ref()
            .and_then(|content| content.parts())
            .is_some_and(|parts| parts.iter().any(|part| part.image_url_ref().is_some()))
    })
}

fn unsupported_vision_error(model: &str) -> EngineError {
    EngineError::UnsupportedCapability {
        provider: "Ollama".to_string(),
        model: model.to_string(),
        capability: "image input; choose a vision-capable model or remove the image".to_string(),
    }
}

fn ollama_response_error(
    status: reqwest::StatusCode,
    body: &str,
    model: &str,
    sent_images: bool,
) -> EngineError {
    let lower = body.to_ascii_lowercase();
    let vision_rejected = lower.contains("does not support vision")
        || lower.contains("doesn't support vision")
        || lower.contains("does not support image")
        || lower.contains("image input is not supported")
        || lower.contains("images are not supported");
    if sent_images && vision_rejected {
        unsupported_vision_error(model)
    } else {
        EngineError::Provider(format!("Ollama returned {status}: {body}"))
    }
}

fn ollama_message_from_chat_message(msg: &ChatMessage) -> EngineResult<serde_json::Value> {
    // Ollama uses `assistant` and `user`/`system` roles. Tool calls are
    // represented via the `tool_calls` field on assistant messages and the
    // `role: tool` with `tool_call_id` for tool results.
    let role = msg.role.clone();

    if role == "tool" {
        // The request builder resolves tool_name from the matching invocation.
        let content = msg.content_text().unwrap_or_default();
        return Ok(serde_json::json!({
            "role": "tool",
            "content": content,
        }));
    }

    // Ollama's `/api/chat` endpoint expects `content` as a plain string and
    // images as a separate top-level `images` array of raw base64 strings
    // (no `data:` prefix). This is different from OpenAI's multi-part format
    // where images are `image_url` objects inside a content array. Serializing
    // `MessageContent::Parts` directly would produce an array, which Ollama
    // rejects with: "cannot unmarshal array into Go struct field
    // ChatRequest.messages.content of type string".
    let text = msg
        .content
        .as_ref()
        .map(|c| c.text_content())
        .unwrap_or_default();

    let images: Vec<String> = msg
        .content
        .as_ref()
        .and_then(|c| c.parts())
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p.image_url_ref())
                .filter_map(|image_url| strip_data_url_prefix(&image_url.url))
                .collect()
        })
        .unwrap_or_default();

    // Ollama does not support file attachments (e.g. PDFs via `file_url`), so
    // they are dropped here and logged for observability. Image support is
    // validated against the selected model before this message is sent.
    if let Some(parts) = msg.content.as_ref().and_then(|c| c.parts()) {
        let file_count = parts
            .iter()
            .filter(|p| matches!(p, crate::models::ContentPart::File { .. }))
            .count();
        if file_count > 0 {
            debug!(
                "ollama: dropping {} file part(s) - Ollama/models like GLM do not support file inputs",
                file_count
            );
        }
    }

    let mut value = serde_json::json!({
        "role": role,
        "content": text,
    });

    if !images.is_empty() {
        value["images"] =
            serde_json::Value::Array(images.into_iter().map(serde_json::Value::String).collect());
    }

    if let Some(tool_calls) = &msg.tool_calls {
        let calls: Vec<serde_json::Value> = tool_calls.iter().map(ollama_tool_call).collect();
        value["tool_calls"] = serde_json::Value::Array(calls);
    }

    Ok(value)
}

/// Extract the raw base64 payload from a data URL (e.g.
/// `data:image/png;base64,iVBOR...` → `iVBOR...`). Returns the input
/// unchanged if it is not a data URL (already-raw base64).
fn strip_data_url_prefix(url: &str) -> Option<String> {
    url.split_once(",")
        .filter(|(prefix, _)| prefix.starts_with("data:"))
        .map(|(_, payload)| payload.to_string())
        .or_else(|| {
            // Not a data URL - return the whole string as-is (may already
            // be raw base64 from a client that strips the prefix itself).
            Some(url.to_string())
        })
}

fn ollama_tool_definition(tool: &ToolDefinition) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    })
}

fn ollama_tool_call(call: &ToolCall) -> serde_json::Value {
    // Ollama's Go JSON decoder expects `arguments` to be a raw JSON object,
    // not a JSON-encoded string. If we embed the string directly the `json!`
    // macro produces `"arguments": "{\"key\":\"val\"}"`, and Ollama sees the
    // leading `{` inside the string, tries to parse it as an object, and
    // fails with "Value looks like object, but can't find closing '}'
    // symbol". Parse the string into a `Value` first so it serializes as a
    // proper object; fall back to an empty object if the string isn't valid
    // JSON.
    let arguments = serde_json::from_str::<serde_json::Value>(&call.function.arguments)
        .unwrap_or(serde_json::json!({}));
    serde_json::json!({
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.function.name,
            "arguments": arguments,
        },
    })
}

fn parse_completion_response(raw: serde_json::Value) -> EngineResult<ChatCompletionResponse> {
    let model = raw
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Ollama's non-streaming chat response includes the assistant content in
    // `message.content` (and `message.tool_calls` when tools were invoked).
    let message = raw
        .get("message")
        .ok_or_else(|| EngineError::Provider("missing message in Ollama response".to_string()))?;

    let chat_message = chat_message_from_ollama(message)?;

    let usage = parse_ollama_usage(&raw, &chat_message);

    Ok(ChatCompletionResponse {
        id: format!("ollama-{}", uuid::Uuid::new_v4()),
        model,
        message: chat_message,
        usage,
        raw,
        response_headers: Default::default(),
    })
}

fn chat_message_from_ollama(value: &serde_json::Value) -> EngineResult<ChatMessage> {
    let role = value
        .get("role")
        .and_then(|v| v.as_str())
        .unwrap_or("assistant")
        .to_string();
    let content = value
        .get("content")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| MessageContent::text(s.to_string()));

    let tool_calls = value
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(parse_ollama_tool_call)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;

    Ok(ChatMessage {
        provider_state: None,
        role,
        content,
        tool_calls,
        tool_call_id: None,
        name: None,
    })
}

fn parse_ollama_tool_call(value: &serde_json::Value) -> EngineResult<ToolCall> {
    let id = value
        .get("id")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| {
            // Ollama does not always include an id; synthesize one so the
            // downstream tool-result round-trip has a stable reference.
            format!("call_{}", uuid::Uuid::new_v4())
        });
    let function = value
        .get("function")
        .ok_or_else(|| EngineError::Provider("missing function in tool_call".to_string()))?;
    let name = function
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let arguments = function
        .get("arguments")
        .map(|v| {
            // Ollama sometimes returns arguments as a JSON object rather than
            // a JSON string. Normalize to a string for compatibility with the
            // `FunctionCall.arguments` field used elsewhere in the engine.
            match v {
                serde_json::Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_default(),
            }
        })
        .unwrap_or_default();

    Ok(ToolCall {
        id,
        call_type: "function".to_string(),
        function: crate::models::FunctionCall { name, arguments },
    })
}

fn parse_ollama_usage(raw: &serde_json::Value, chat_message: &ChatMessage) -> Option<TokenUsage> {
    let prompt_tokens = raw
        .get("prompt_eval_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let completion_tokens = raw
        .get("eval_count")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| {
            // Fallback: estimate from the content length if eval_count is missing.
            chat_message
                .content_text()
                .map(|c| c.len() as u64)
                .unwrap_or(0)
        });
    let total_tokens = prompt_tokens + completion_tokens;

    if prompt_tokens == 0 && completion_tokens == 0 {
        return None;
    }

    Some(TokenUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cached_prompt_tokens: 0,
        cache_creation_input_tokens: 0,
        ..Default::default()
    })
}

async fn ollama_stream(
    response: reqwest::Response,
    model: &str,
    sent_images: bool,
    replay_scope: Option<String>,
) -> EngineResult<ProviderStream> {
    let status = response.status();
    if !status.is_success() {
        let body = response
            .text()
            .await
            .unwrap_or_else(|error| format!("<failed to read error body: {error}>"));
        return Err(ollama_response_error(status, &body, model, sent_images));
    }

    let response_headers = capture_response_headers(response.headers());
    let mut bytes = response.bytes_stream();
    let stream = async_stream::stream! {
        let mut buffer = Vec::<u8>::new();
        let mut tool_indices = OllamaToolIndices::default();
        let mut replay = super::replay::ReplayAccumulator::default();
        while let Some(next) = bytes.next().await {
            let chunk = match next {
                Ok(chunk) => chunk,
                Err(error) => {
                    yield Err(EngineError::from_request_error(error));
                    break;
                }
            };
            buffer.extend_from_slice(&chunk);
            let mut done = false;
            while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = buffer.drain(..=newline).collect::<Vec<_>>();
                let line = String::from_utf8_lossy(&line);
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if replay_scope.is_some() { if let Ok(raw) = serde_json::from_str(line) { replay.compatible(&raw); } }
                match parse_ollama_stream_line_with_indices(line, &mut tool_indices) {
                    Ok(Some(mut parsed)) => {
                        if parsed.finish_reason.is_some() { parsed.provider_state = replay_scope.as_ref().and_then(|scope| replay.snapshot(scope.clone(), false)); }
                        done = parsed.finish_reason.is_some();
                        yield Ok(parsed);
                    }
                    Ok(None) => {}
                    Err(error) => {
                        yield Err(error);
                        done = true;
                    }
                }
                if done {
                    break;
                }
            }
            if done {
                break;
            }
        }
        if !buffer.is_empty() {
            let line = String::from_utf8_lossy(&buffer);
            if replay_scope.is_some() { if let Ok(raw) = serde_json::from_str(line.trim()) { replay.compatible(&raw); } }
            if let Ok(Some(mut parsed)) = parse_ollama_stream_line_with_indices(line.trim(), &mut tool_indices) {
                if parsed.finish_reason.is_some() { parsed.provider_state = replay_scope.as_ref().and_then(|scope| replay.snapshot(scope.clone(), false)); }
                yield Ok(parsed);
            }
        }
    };
    Ok(ProviderStream::new(stream, response_headers))
}

// Native calls without a stable ID/index are separate calls, even across NDJSON lines.
// Explicit indices (including function.index) and IDs identify repeated snapshots.
#[derive(Default)]
struct OllamaToolIndices {
    next: usize,
    keys: std::collections::HashMap<String, usize>,
}
impl OllamaToolIndices {
    fn resolve(&mut self, call: &serde_json::Value) -> usize {
        let key = call
            .get("id")
            .and_then(serde_json::Value::as_str)
            .filter(|id| !id.is_empty())
            .map(|id| format!("id:{id}"))
            .or_else(|| {
                call.get("index")
                    .or_else(|| call.get("function").and_then(|f| f.get("index")))
                    .and_then(serde_json::Value::as_u64)
                    .map(|index| format!("index:{index}"))
            });
        if let Some(index) = key.as_ref().and_then(|key| self.keys.get(key)) {
            return *index;
        }
        let index = self.next;
        self.next = self.next.saturating_add(1);
        if let Some(key) = key {
            self.keys.insert(key, index);
        }
        index
    }
}

#[cfg(test)]
fn parse_ollama_stream_line(line: &str) -> EngineResult<Option<ChatCompletionStreamChunk>> {
    parse_ollama_stream_line_with_indices(line, &mut OllamaToolIndices::default())
}

fn parse_ollama_stream_line_with_indices(
    line: &str,
    tool_indices: &mut OllamaToolIndices,
) -> EngineResult<Option<ChatCompletionStreamChunk>> {
    let value: serde_json::Value = serde_json::from_str(line)
        .map_err(|error| EngineError::Provider(format!("invalid Ollama stream JSON: {error}")))?;
    if let Some(error) = value.get("error") {
        return Err(EngineError::Provider(format!(
            "Ollama stream returned an error: {error}"
        )));
    }
    let message = value.get("message");
    let delta_content = message
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .filter(|content| !content.is_empty())
        .map(str::to_string);
    let delta_reasoning_content = message
        .and_then(|message| message.get("thinking"))
        .and_then(|thinking| thinking.as_str())
        .filter(|thinking| !thinking.is_empty())
        .map(str::to_string);
    let delta_tool_calls = message
        .and_then(|message| message.get("tool_calls"))
        .and_then(|calls| calls.as_array())
        .map(|calls| {
            calls
                .iter()
                .map(|call| {
                    let function = call.get("function");
                    let arguments =
                        function
                            .and_then(|function| function.get("arguments"))
                            .map(|arguments| match arguments {
                                serde_json::Value::String(arguments) => arguments.clone(),
                                value => serde_json::to_string(value).unwrap_or_default(),
                            });
                    ToolCallDelta {
                        snapshot: true,
                        index: tool_indices.resolve(call),
                        id: call
                            .get("id")
                            .and_then(|id| id.as_str())
                            .map(str::to_string),
                        call_type: Some("function".to_string()),
                        function_name: function
                            .and_then(|function| function.get("name"))
                            .and_then(|name| name.as_str())
                            .map(str::to_string),
                        function_arguments: arguments,
                    }
                })
                .collect::<Vec<_>>()
        })
        .filter(|calls| !calls.is_empty());
    let done = value.get("done").and_then(|done| done.as_bool()) == Some(true);
    let finish_reason = done.then(|| {
        value
            .get("done_reason")
            .and_then(|reason| reason.as_str())
            .unwrap_or_else(|| {
                if delta_tool_calls.is_some() {
                    "tool_calls"
                } else {
                    "stop"
                }
            })
            .to_string()
    });
    let usage = done.then(|| {
        let prompt_tokens = value
            .get("prompt_eval_count")
            .and_then(|tokens| tokens.as_u64())
            .unwrap_or(0);
        let completion_tokens = value
            .get("eval_count")
            .and_then(|tokens| tokens.as_u64())
            .unwrap_or(0);
        TokenUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            cached_prompt_tokens: 0,
            cache_creation_input_tokens: 0,
            ..Default::default()
        }
    });
    if delta_content.is_none()
        && delta_reasoning_content.is_none()
        && delta_tool_calls.is_none()
        && finish_reason.is_none()
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

#[cfg(test)]
mod stream_tests {
    use super::*;

    #[test]
    fn native_tool_snapshots_keep_identity_without_concatenating_names_or_json() {
        let mut indices = OllamaToolIndices::default();
        let mut name = String::new();
        let mut args = String::new();
        for query in ["first", "second", "second"] {
            let raw = serde_json::json!({"message":{"tool_calls":[{"function":{"index":7,"name":"file_search","arguments":{"query":query}}}]},"done":false});
            let chunk = parse_ollama_stream_line_with_indices(&raw.to_string(), &mut indices)
                .unwrap()
                .unwrap();
            let call = &chunk.delta_tool_calls.unwrap()[0];
            assert!(call.snapshot);
            assert_eq!(call.index, 0);
            call.merge_function_fields(&mut name, &mut args);
        }
        assert_eq!(name, "file_search");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&args).unwrap(),
            serde_json::json!({"query":"second"})
        );
        for expected in [1, 2] {
            let raw = r#"{"message":{"tool_calls":[{"function":{"name":"file_search","arguments":{"query":"other"}}}]}}"#;
            let chunk = parse_ollama_stream_line_with_indices(raw, &mut indices)
                .unwrap()
                .unwrap();
            assert_eq!(chunk.delta_tool_calls.unwrap()[0].index, expected);
        }
        for id in ["a", "b", "a"] {
            let raw = serde_json::json!({"message":{"tool_calls":[{"id":id,"function":{"name":"file_search","arguments":{}}}]}});
            let chunk = parse_ollama_stream_line_with_indices(&raw.to_string(), &mut indices)
                .unwrap()
                .unwrap();
            assert_eq!(
                chunk.delta_tool_calls.unwrap()[0].index,
                if id == "a" { 3 } else { 4 }
            );
        }
    }

    #[test]
    fn tool_results_resolve_function_names_for_parallel_calls() {
        let provider = OllamaProvider::new(ProviderConfig::default(), OllamaMode::Local).unwrap();
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "assistant", "tool_calls": [
                    {"id":"a", "type":"function", "function":{"name":"read_file","arguments":"{}"}},
                    {"id":"b", "type":"function", "function":{"name":"grep_search","arguments":"{}"}}
                ]},
                {"role":"tool", "tool_call_id":"b", "content":"matches"},
                {"role":"tool", "tool_call_id":"a", "content":"file"},
                {"role":"tool", "tool_call_id":"missing", "name":"fallback", "content":"result"},
                {"role":"tool", "tool_call_id":"unknown", "content":"orphan"}
            ]
        })).unwrap();
        let body = provider.build_body(&request, "test").unwrap();
        assert_eq!(body["messages"][1]["tool_name"], "grep_search");
        assert_eq!(body["messages"][2]["tool_name"], "read_file");
        assert_eq!(body["messages"][3]["tool_name"], "fallback");
        assert!(body["messages"][4].get("tool_name").is_none());
        for message in body["messages"].as_array().unwrap().iter().skip(1) {
            assert!(message.get("name").is_none());
        }
        assert_eq!(body["messages"][1]["content"], "matches");
    }

    #[test]
    fn title_requests_disable_ollama_thinking() {
        let provider = OllamaProvider::new(ProviderConfig::default(), OllamaMode::Local).unwrap();
        let mut request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "kimi-k2.7-code:cloud",
            "messages": [{"role": "user", "content": "Fix navigation scroll position"}],
            "max_tokens": 128
        }))
        .unwrap();

        let normal = provider
            .build_body(&request, "kimi-k2.7-code:cloud")
            .unwrap();
        assert!(normal.get("think").is_none());

        request.extra.insert(
            crate::title_generation::TITLE_REQUEST_EXTRA_KEY.to_string(),
            serde_json::json!(true),
        );
        let title = provider
            .build_body(&request, "kimi-k2.7-code:cloud")
            .unwrap();
        assert_eq!(title["think"], serde_json::json!(false));
        assert_eq!(title["options"]["num_predict"], serde_json::json!(128));
    }

    #[test]
    fn parses_advertised_and_legacy_vision_capabilities() {
        let vision = parse_ollama_model_metadata(&serde_json::json!({
            "capabilities": ["completion", "vision"],
            "model_info": { "gemma3.context_length": 131_072 }
        }));
        assert_eq!(vision.supports_vision, Some(true));
        assert_eq!(vision.context_length, Some(131_072));

        let text_only = parse_ollama_model_metadata(&serde_json::json!({
            "capabilities": ["completion", "tools"]
        }));
        assert_eq!(text_only.supports_vision, Some(false));

        let legacy_vision = parse_ollama_model_metadata(&serde_json::json!({
            "projector_info": { "clip.embedding_length": 1024 }
        }));
        assert_eq!(legacy_vision.supports_vision, Some(true));

        let unknown = parse_ollama_model_metadata(&serde_json::json!({
            "model_info": { "llama.context_length": 8192 }
        }));
        assert_eq!(unknown.supports_vision, None);
    }

    #[test]
    fn recognizes_image_parts_and_normalizes_ollama_rejections() {
        let request = ChatCompletionRequest {
            model: "glm".to_string(),
            messages: vec![ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(MessageContent::Parts(vec![
                    crate::models::ContentPart::Text {
                        text: "describe this".to_string(),
                    },
                    crate::models::ContentPart::ImageUrl {
                        image_url: crate::models::ImageUrl {
                            url: "data:image/png;base64,AAAA".to_string(),
                            detail: None,
                        },
                    },
                ])),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: HashMap::new(),
        };
        assert!(request_contains_images(&request));

        let error = ollama_response_error(
            reqwest::StatusCode::BAD_REQUEST,
            r#"{"error":"model does not support vision"}"#,
            "glm",
            true,
        );
        assert!(matches!(error, EngineError::UnsupportedCapability { .. }));
        assert!(!error.is_retryable());
        assert!(error.to_string().contains("vision-capable model"));
    }

    #[test]
    fn parses_content_and_terminal_usage() {
        let content = parse_ollama_stream_line(
            r#"{"model":"llama","message":{"role":"assistant","content":"hi"},"done":false}"#,
        )
        .unwrap()
        .unwrap();
        assert_eq!(content.delta_content.as_deref(), Some("hi"));
        assert!(content.finish_reason.is_none());

        let final_chunk = parse_ollama_stream_line(
            r#"{"model":"llama","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":8,"eval_count":2}"#,
        )
        .unwrap()
        .unwrap();
        assert_eq!(final_chunk.finish_reason.as_deref(), Some("stop"));
        assert_eq!(final_chunk.usage.unwrap().total_tokens, 10);
    }

    #[test]
    fn parses_thinking_delta() {
        let chunk = parse_ollama_stream_line(
            r#"{"model":"qwen","message":{"role":"assistant","content":"","thinking":"checking"},"done":false}"#,
        )
        .unwrap()
        .unwrap();
        assert_eq!(chunk.delta_reasoning_content.as_deref(), Some("checking"));
    }

    #[test]
    fn normalizes_streamed_tool_arguments() {
        let chunk = parse_ollama_stream_line(
            r#"{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"read_file","arguments":{"path":"a.rs"}}}]},"done":true}"#,
        )
        .unwrap()
        .unwrap();
        let tool = &chunk.delta_tool_calls.unwrap()[0];
        assert_eq!(tool.function_name.as_deref(), Some("read_file"));
        assert_eq!(
            tool.function_arguments.as_deref(),
            Some(r#"{"path":"a.rs"}"#)
        );
        assert_eq!(chunk.finish_reason.as_deref(), Some("tool_calls"));
    }
}
