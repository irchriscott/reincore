//! Anthropic Messages API provider adapter.
//!
//! Converts the normalized `ChatCompletionRequest`/`ChatMessage` types into
//! Anthropic's Messages API format and converts the response back.

use crate::error::{EngineError, EngineResult};
use crate::models::{
    ChatCompletionRequest, ChatMessage, MessageContent, ModelTier, ToolCall, ToolDefinition,
};
use crate::provider::{
    capture_response_headers, BatchSubmission, ChatCompletionResponse, ChatCompletionStreamChunk,
    Provider, ProviderConfig, ProviderModel, ProviderStream, TokenUsage, ToolCallDelta,
};
use async_trait::async_trait;
use futures::StreamExt;
use tracing::debug;

/// Anthropic provider implementation.
#[derive(Clone, Debug)]
pub struct AnthropicProvider {
    config: ProviderConfig,
    client: reqwest::Client,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider from configuration.
    pub fn new(config: ProviderConfig) -> EngineResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            // Separate from the overall timeout: an unreachable endpoint should
            // fail in seconds, not after a generation-length wait.
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| EngineError::Provider(format!("failed to build HTTP client: {e}")))?;
        Ok(Self { config, client })
    }

    fn auth_headers(&self) -> EngineResult<std::collections::HashMap<String, String>> {
        let mut headers = std::collections::HashMap::new();
        headers.insert("x-api-key".to_string(), self.config.api_key.clone());
        headers.insert("anthropic-version".to_string(), "2023-06-01".to_string());
        Ok(headers)
    }

    fn build_body(
        &self,
        request: &ChatCompletionRequest,
        provider_model: String,
    ) -> EngineResult<serde_json::Value> {
        let mut static_system_parts = Vec::new();
        let mut dynamic_system_parts = Vec::new();
        let mut messages = Vec::new();

        for m in &request.messages {
            if m.role == "system" {
                // Anthropic separates system prompt from conversation messages.
                // Preserve every system message: ReinCore may add a context
                // summary or loop intervention after the caller's anchor.
                //
                // Those injected messages are rewritten between turns, so they
                // are split out here and emitted after the cache breakpoint.
                // Folding them into the cached prefix would invalidate it on
                // every call that compacts context or nudges the loop.
                if let Some(content) = m.content_text().filter(|text| !text.trim().is_empty()) {
                    if crate::normalizer::is_dynamic_system_message(m.name.as_deref()) {
                        dynamic_system_parts.push(content);
                    } else {
                        static_system_parts.push(content);
                    }
                }
                continue;
            }
            let mut wire = anthropic_message_from_chat_message(m)?;
            if super::replay::enabled(request) {
                super::replay::apply(
                    m,
                    &mut wire,
                    &super::replay::scope(&self.config, &provider_model),
                );
            }
            messages.push(wire);
        }

        // Multi-turn agent loops resend the whole prior conversation, so a
        // breakpoint on the tail gives the next request a point to read it back
        // from. Single-shot requests are skipped on purpose: their tail is
        // unique per call, so the entry would be written and never read - the
        // cache write premium with none of the benefit.
        if request.messages.iter().any(|m| m.role == "assistant") {
            if let Some(block) = messages
                .last_mut()
                .and_then(|message| message.get_mut("content"))
                .and_then(|content| content.as_array_mut())
                .and_then(|blocks| blocks.last_mut())
                .and_then(|block| block.as_object_mut())
            {
                block.insert(
                    "cache_control".to_string(),
                    serde_json::json!({"type": "ephemeral"}),
                );
            }
            mark_intermediate_breakpoints(&mut messages);
        }

        let mut body = serde_json::json!({
            "model": provider_model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(1024),
        });

        // Anthropic renders tools -> system -> messages, so a breakpoint on the
        // last static system block caches the tool schemas and the frozen system
        // prompt together. Dynamic system content follows in its own block past
        // the breakpoint, where rewriting it costs nothing.
        // The static prefix may use a longer TTL than the conversation tail.
        // Anthropic requires longer-lived entries to appear before shorter
        // ones, and tools -> system -> messages renders this block first, so
        // that ordering constraint holds by construction.
        let mut system_cache_control = serde_json::json!({"type": "ephemeral"});
        if let Some(ttl) = static_prefix_cache_ttl(request) {
            system_cache_control["ttl"] = serde_json::json!(ttl);
        }
        let mut system_blocks = Vec::new();
        if !static_system_parts.is_empty() {
            system_blocks.push(serde_json::json!({
                "type": "text",
                "text": static_system_parts.join("\n\n"),
                "cache_control": system_cache_control,
            }));
        }
        if !dynamic_system_parts.is_empty() {
            system_blocks.push(serde_json::json!({
                "type": "text",
                "text": dynamic_system_parts.join("\n\n"),
            }));
        }
        if !system_blocks.is_empty() {
            body["system"] = serde_json::Value::Array(system_blocks);
        }

        if let Some(tools) = &request.tools {
            let tool_specs: Vec<serde_json::Value> =
                tools.iter().map(anthropic_tool_definition).collect();
            body["tools"] = serde_json::Value::Array(tool_specs);
        }

        if let Some(choice) = request.tool_choice.as_ref() {
            body["tool_choice"] = match choice {
                serde_json::Value::String(value) if value == "required" => {
                    serde_json::json!({"type": "any"})
                }
                serde_json::Value::String(value) => serde_json::json!({"type": value}),
                // OpenAI's `allowed_tools` narrows the callable subset without
                // touching the tools array, which is how a caller varies
                // availability while keeping the cached prefix byte-identical.
                // Anthropic has no equivalent, so only the mode survives: the
                // tools stay declared, and a call to one that is currently
                // unavailable is refused by the action gate instead. Forwarding
                // the field verbatim - which is what the object arm below would
                // do, since it carries no `function` key - is a 400.
                serde_json::Value::Object(map)
                    if map.get("type").and_then(|kind| kind.as_str()) == Some("allowed_tools") =>
                {
                    match map.get("mode").and_then(|mode| mode.as_str()) {
                        Some("required") => serde_json::json!({"type": "any"}),
                        _ => serde_json::json!({"type": "auto"}),
                    }
                }
                serde_json::Value::Object(map) => map
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(|name| name.as_str())
                    .map(|name| serde_json::json!({"type": "tool", "name": name}))
                    .unwrap_or_else(|| choice.clone()),
                _ => choice.clone(),
            };
        }

        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }

        Ok(body)
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn batch_client(&self) -> EngineResult<super::batch::BatchClient> {
        crate::privacy::current_policy().check_provider(self.label())?;
        Ok(super::batch::BatchClient::new(
            self.client.clone(),
            self.config.base_url.clone(),
            self.auth_headers()?,
            super::batch::BatchProtocol::Anthropic,
            self.config.label.clone(),
        ))
    }

    fn prepare_batch_body(
        &self,
        request: &ChatCompletionRequest,
    ) -> EngineResult<serde_json::Value> {
        let prepared_request = crate::privacy::prepare_provider_request(request, self.label())?;
        let request = &prepared_request;
        let model = self.config.model_for_request(request, ModelTier::Pro)?;
        self.build_body(request, model)
    }

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
        let tier = request
            .extra
            .get("reincore_model_tier")
            .and_then(|v| v.as_str())
            .and_then(|value| value.parse().ok())
            .unwrap_or(ModelTier::Pro);

        let provider_model = self.config.model_for_request(request, tier)?;
        let body = self.build_body(request, provider_model)?;

        debug!(
            "Anthropic request to {}: {}",
            self.config.base_url,
            crate::privacy::current_policy()
                .sanitize_for_logging(&serde_json::to_string(&body).unwrap_or_default())
        );

        let mut req_builder = self
            .client
            .post(format!("{}/v1/messages", self.config.base_url))
            .header("Content-Type", "application/json");

        for (key, value) in self.auth_headers()? {
            req_builder = req_builder.header(key, value);
        }

        let response = req_builder
            .json(&body)
            .send()
            .await
            .map_err(EngineError::from_request_error)?;

        let status = response.status();
        let response_headers = capture_response_headers(response.headers());
        let raw = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| EngineError::Provider(format!("failed to read response: {e}")))?;

        if !status.is_success() {
            return Err(EngineError::Provider(format!(
                "Anthropic returned {status}: {}",
                crate::privacy::current_policy().sanitize_for_logging(&raw.to_string())
            )));
        }

        let mut parsed = parse_messages_response(raw)?;
        super::replay::capture_response(
            &mut parsed,
            super::replay::request_scope(
                &self.config,
                body["model"].as_str().unwrap_or_default(),
                request,
            ),
            true,
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
        let mut body = self.build_body(request, provider_model)?;
        body["stream"] = serde_json::json!(true);
        let mut builder = self
            .client
            .post(format!("{}/v1/messages", self.config.base_url))
            .header("Content-Type", "application/json");
        for (key, value) in self.auth_headers()? {
            builder = builder.header(key, value);
        }
        let response = builder
            .json(&body)
            .send()
            .await
            .map_err(EngineError::from_request_error)?;
        anthropic_stream(
            response,
            super::replay::request_scope(
                &self.config,
                body["model"].as_str().unwrap_or_default(),
                request,
            ),
        )
        .await
    }

    async fn submit_batch(&self, request: &ChatCompletionRequest) -> EngineResult<BatchSubmission> {
        let prepared_request = crate::privacy::prepare_provider_request(request, self.label())?;
        let request = &prepared_request;
        let tier = request
            .extra
            .get("reincore_model_tier")
            .and_then(|v| v.as_str())
            .and_then(|value| value.parse().ok())
            .unwrap_or(ModelTier::Pro);

        let provider_model = self.config.model_for_request(request, tier)?;
        let params = self.build_body(request, provider_model)?;
        let custom_id = format!("reincore-{}", uuid::Uuid::new_v4());
        let body = serde_json::json!({
            "requests": [{
                "custom_id": custom_id,
                "params": params,
            }]
        });

        let mut req_builder = self
            .client
            .post(format!("{}/v1/messages/batches", self.config.base_url))
            .header("Content-Type", "application/json");

        for (key, value) in self.auth_headers()? {
            req_builder = req_builder.header(key, value);
        }

        let response = req_builder
            .json(&body)
            .send()
            .await
            .map_err(EngineError::from_request_error)?;

        let status = response.status();
        let raw = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| EngineError::Provider(format!("failed to read response: {e}")))?;

        if !status.is_success() {
            return Err(EngineError::Provider(format!(
                "Anthropic returned {status} on /v1/messages/batches: {}",
                crate::privacy::current_policy().sanitize_for_logging(&raw.to_string())
            )));
        }

        let batch_id = raw
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| EngineError::Provider("missing batch id in response".to_string()))?
            .to_string();
        let status = raw
            .get("processing_status")
            .and_then(|v| v.as_str())
            .unwrap_or("submitted")
            .to_string();

        Ok(BatchSubmission { batch_id, status })
    }

    async fn list_models(&self) -> EngineResult<Vec<ProviderModel>> {
        crate::privacy::current_policy().check_provider(self.label())?;
        let mut req_builder = self
            .client
            .get(format!("{}/v1/models", self.config.base_url))
            .header("Content-Type", "application/json");

        for (key, value) in self.auth_headers()? {
            req_builder = req_builder.header(key, value);
        }

        let response = req_builder
            .send()
            .await
            .map_err(EngineError::from_request_error)?;

        let status = response.status();
        let raw = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| EngineError::Provider(format!("failed to read response: {e}")))?;

        if !status.is_success() {
            return Err(EngineError::Provider(format!(
                "Anthropic returned {status} on /v1/models: {}",
                crate::privacy::current_policy().sanitize_for_logging(&raw.to_string())
            )));
        }

        let data = raw
            .get("data")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let models = data
            .iter()
            .filter_map(|v| {
                let id = v.get("id").and_then(|v| v.as_str())?;
                let display_name = v
                    .get("display_name")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let created = v.get("created_at").and_then(|v| v.as_str()).and_then(|s| {
                    chrono::DateTime::parse_from_rfc3339(s)
                        .ok()
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                });
                Some(ProviderModel {
                    id: id.to_string(),
                    provider: self.label().to_string(),
                    display_name,
                    created_at: created,
                    metadata: Some(v.clone()),
                })
            })
            .collect();

        Ok(models)
    }
}

/// Parse a data URL (e.g. `data:image/png;base64,iVBOR...`) into its
/// media type and base64-encoded data. Returns `None` if the URL is not
/// a valid data URL.
fn parse_data_url(url: &str) -> Option<(String, String)> {
    let url = url.strip_prefix("data:")?;
    let (metadata, data) = url.split_once(',')?;
    let (mime, is_base64) = if let Some((m, _)) = metadata.split_once(';') {
        (m.to_string(), metadata.contains("base64"))
    } else {
        (metadata.to_string(), false)
    };
    if !is_base64 {
        return None;
    }
    Some((mime, data.to_string()))
}

fn anthropic_message_from_chat_message(msg: &ChatMessage) -> EngineResult<serde_json::Value> {
    let role = if msg.role == "tool" {
        "user"
    } else {
        &msg.role
    };

    let mut content = Vec::new();

    // Convert multi-part content to Anthropic's content block format.
    // Text parts become {"type": "text", "text": ...} blocks.
    // Image parts become {"type": "image", "source": {"type": "base64", ...}} blocks.
    if let Some(msg_content) = &msg.content {
        match msg_content {
            crate::MessageContent::Text(text) => {
                content.push(serde_json::json!({
                    "type": "text",
                    "text": text,
                }));
            }
            crate::MessageContent::Parts(parts) => {
                for part in parts {
                    match part {
                        crate::ContentPart::Text { text } => {
                            content.push(serde_json::json!({
                                "type": "text",
                                "text": text,
                            }));
                        }
                        crate::ContentPart::ImageUrl { image_url } => {
                            // Parse the data URL to extract media type and base64 data.
                            if let Some((media_type, data)) = parse_data_url(&image_url.url) {
                                content.push(serde_json::json!({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data,
                                    }
                                }));
                            }
                        }
                        crate::ContentPart::File { file_url } => {
                            // Parse the data URL to extract media type and base64 data.
                            // Anthropic supports PDFs as "document" content blocks.
                            if let Some((media_type, data)) = parse_data_url(&file_url.url) {
                                if media_type == "application/pdf" {
                                    content.push(serde_json::json!({
                                        "type": "document",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "application/pdf",
                                            "data": data,
                                        }
                                    }));
                                } else {
                                    // Non-PDF files: send as document with the
                                    // detected media type (Anthropic may reject
                                    // unsupported types, but we forward anyway).
                                    content.push(serde_json::json!({
                                        "type": "document",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": data,
                                        }
                                    }));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Tool results are represented as tool_result content blocks in Anthropic.
    if let Some(tool_call_id) = &msg.tool_call_id {
        content.push(serde_json::json!({
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": msg.content_str().unwrap_or(""),
            "is_error": false,
        }));
    }

    // Assistant tool_calls are represented as tool_use content blocks.
    if let Some(tool_calls) = &msg.tool_calls {
        for call in tool_calls {
            content.push(serde_json::json!({
                "type": "tool_use",
                "id": call.id,
                "name": call.function.name,
                "input": serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                    .unwrap_or(serde_json::json!({})),
            }));
        }
    }

    Ok(serde_json::json!({
        "role": role,
        "content": content,
    }))
}

/// Resolve the cache TTL for the static prefix breakpoint, or `None` to leave
/// the 5-minute default in place.
///
/// A 1-hour cache write bills at 2x base input against 1.25x for the 5-minute
/// default, and a read refreshes the entry's timer under either. Traffic that
/// keeps hitting the same prefix inside 5 minutes therefore keeps the cheap
/// entry alive indefinitely and gains nothing from the longer window - it just
/// pays the doubled write. The 1-hour TTL only repays itself across gaps of
/// roughly 5 to 60 minutes between requests sharing a prefix, which is a
/// property of the caller's traffic and not something this layer can observe,
/// so it stays opt-in: `reincore_cache_ttl` per request, or `REINCORE_CACHE_TTL`
/// process-wide. The knob is shared with the OpenAI family, which maps the same
/// intent onto its own retention field.
fn static_prefix_cache_ttl(request: &ChatCompletionRequest) -> Option<&'static str> {
    super::provider::extended_cache_retention_requested(request).then_some("1h")
}

/// Stride between trailing breakpoints, kept comfortably inside the lookback
/// window so an entry is still reachable after a long turn.
const CACHE_BREAKPOINT_STRIDE: usize = 15;

/// Anthropic allows four breakpoints per request. The static system block and
/// the conversation tail take one each, leaving two for the trailing markers.
const MAX_INTERMEDIATE_BREAKPOINTS: usize = 2;

/// Lookback distance Anthropic searches back from a breakpoint for a prior
/// cache entry.
const CACHE_LOOKBACK_POSITIONS: usize = 20;

/// Position index of every content block, collapsing runs the way Anthropic's
/// lookback counts them: consecutive `tool_use` blocks occupy one position, and
/// so do consecutive `tool_result` blocks. That is why a turn full of *parallel*
/// tool calls is harmless while a long *sequential* loop is not.
fn block_positions(messages: &[serde_json::Value]) -> Vec<(usize, usize, usize)> {
    let mut positions = Vec::new();
    let mut position = 0usize;
    let mut previous_run: Option<&str> = None;
    for (message_index, message) in messages.iter().enumerate() {
        let Some(blocks) = message.get("content").and_then(|c| c.as_array()) else {
            continue;
        };
        for (block_index, block) in blocks.iter().enumerate() {
            let kind = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
            let run = matches!(kind, "tool_use" | "tool_result").then_some(kind);
            if !(run.is_some() && run == previous_run) {
                position += 1;
            }
            previous_run = run;
            positions.push((message_index, block_index, position));
        }
    }
    positions
}

/// Mark trailing intermediate breakpoints in a long conversation.
///
/// A breakpoint walks back at most [`CACHE_LOOKBACK_POSITIONS`] to find a prior
/// entry. A turn that appends more than that - a long sequential tool loop,
/// which is the shape an agent gateway sees most - pushes the previous request's
/// entry out of the window, and every later request then rewrites the whole
/// conversation with byte-identical content while reads collapse to zero. The
/// symptom is `cache_creation_input_tokens` near the full conversation size on
/// every call.
///
/// Markers placed here are read back by the *next* request, not this one, and
/// two extra anchors cannot guarantee a hit after an arbitrarily long turn -
/// they bound the damage. The next request reads back to the nearest surviving
/// anchor and rewrites only what follows it, instead of rewriting everything.
/// In a normal tool loop each request appends about two positions and the tail
/// marker alone suffices; these earn their place when a client submits many
/// blocks at once, such as a replayed history or a restructuring compaction.
fn mark_intermediate_breakpoints(messages: &mut [serde_json::Value]) {
    let positions = block_positions(messages);
    let Some(&(_, _, total)) = positions.last() else {
        return;
    };
    if total <= CACHE_LOOKBACK_POSITIONS {
        // The tail marker is already within reach of the previous entry.
        return;
    }
    for step in 1..=MAX_INTERMEDIATE_BREAKPOINTS {
        let Some(target) = total.checked_sub(CACHE_BREAKPOINT_STRIDE * step) else {
            break;
        };
        if target == 0 {
            break;
        }
        let Some(&(message_index, block_index, _)) = positions
            .iter()
            .filter(|(_, _, position)| *position <= target)
            .next_back()
        else {
            continue;
        };
        if let Some(block) = messages
            .get_mut(message_index)
            .and_then(|message| message.get_mut("content"))
            .and_then(|content| content.as_array_mut())
            .and_then(|blocks| blocks.get_mut(block_index))
            .and_then(|block| block.as_object_mut())
        {
            // Never disturb a marker already placed - the tail keeps its own.
            block
                .entry("cache_control")
                .or_insert_with(|| serde_json::json!({"type": "ephemeral"}));
        }
    }
}

fn anthropic_tool_definition(tool: &ToolDefinition) -> serde_json::Value {
    serde_json::json!({
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    })
}

fn parse_messages_response(raw: serde_json::Value) -> EngineResult<ChatCompletionResponse> {
    let id = raw
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let model = raw
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let content_blocks = raw
        .get("content")
        .and_then(|v| v.as_array())
        .ok_or_else(|| EngineError::Provider("missing content in response".to_string()))?;

    let mut content_text = String::new();
    let mut tool_calls = Vec::new();

    for block in content_blocks {
        match block.get("type").and_then(|v| v.as_str()) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                    content_text.push_str(text);
                }
            }
            Some("tool_use") => {
                let id = block
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let input = block.get("input").cloned().unwrap_or(serde_json::json!({}));
                tool_calls.push(ToolCall {
                    id,
                    call_type: "function".to_string(),
                    function: crate::models::FunctionCall {
                        name,
                        arguments: input.to_string(),
                    },
                });
            }
            _ => {}
        }
    }

    let message = ChatMessage {
        provider_state: None,
        role: "assistant".to_string(),
        content: if content_text.is_empty() && !tool_calls.is_empty() {
            None
        } else {
            Some(MessageContent::text(content_text))
        },
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        tool_call_id: None,
        name: None,
    };

    let usage = raw.get("usage").and_then(parse_usage);

    Ok(ChatCompletionResponse {
        id,
        model,
        message,
        usage,
        raw,
        response_headers: Default::default(),
    })
}

pub(super) fn parse_usage(value: &serde_json::Value) -> Option<TokenUsage> {
    let prompt_tokens = value.get("input_tokens")?.as_u64()?;
    let completion_tokens = value.get("output_tokens")?.as_u64()?;
    let cached_prompt_tokens = value
        .get("cache_read_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cache_creation_input_tokens = value
        .get("cache_creation_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let prompt_tokens = prompt_tokens + cached_prompt_tokens + cache_creation_input_tokens;
    Some(TokenUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        cached_prompt_tokens,
        cache_creation_input_tokens,
        cache_creation_1h_tokens: value
            .pointer("/cache_creation/ephemeral_1h_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        ..Default::default()
    })
}

#[derive(Default)]
struct AnthropicStreamUsage {
    input_tokens: u64,
    cache_read_input_tokens: u64,
    cache_creation_input_tokens: u64,
    cache_creation_1h_tokens: u64,
}

async fn anthropic_stream(
    response: reqwest::Response,
    replay_scope: Option<String>,
) -> EngineResult<ProviderStream> {
    let status = response.status();
    if !status.is_success() {
        let body = response
            .text()
            .await
            .unwrap_or_else(|error| format!("<failed to read error body: {error}>"));
        return Err(EngineError::Provider(format!(
            "Anthropic returned {status}: {body}"
        )));
    }

    let response_headers = capture_response_headers(response.headers());
    let mut bytes = response.bytes_stream();
    let stream = async_stream::stream! {
        let mut buffer = Vec::<u8>::new();
        let mut usage = AnthropicStreamUsage::default();
        let mut replay = super::replay::ReplayAccumulator::default();
        let mut stopped = false;
        while let Some(next) = bytes.next().await {
            let chunk = match next {
                Ok(chunk) => chunk,
                Err(error) => {
                    yield Err(EngineError::from_request_error(error));
                    break;
                }
            };
            buffer.extend_from_slice(&chunk);
            while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = buffer.drain(..=newline).collect::<Vec<_>>();
                let line = String::from_utf8_lossy(&line);
                let Some(data) = line.trim().strip_prefix("data:").map(str::trim) else {
                    continue;
                };
                if data.is_empty() {
                    continue;
                }
                if replay_scope.is_some() {
                    if let Ok(raw) = serde_json::from_str(data) {
                        if let Err(error) = replay.anthropic(&raw) {
                            yield Err(error); stopped = true; break;
                        }
                    }
                }
                match parse_anthropic_stream_event(data, &mut usage) {
                    Ok(Some(mut parsed)) => {
                        if parsed.finish_reason.is_some() { parsed.provider_state = replay_scope.as_ref().and_then(|scope| replay.snapshot(scope.clone(), true)); }
                        yield Ok(parsed);
                    },
                    Ok(None) => {}
                    Err(error) => {
                        yield Err(error);
                        stopped = true;
                        break;
                    }
                }
                if data.contains("\"type\":\"message_stop\"") {
                    stopped = true;
                    break;
                }
            }
            if stopped {
                break;
            }
        }
    };
    Ok(ProviderStream::new(stream, response_headers))
}

fn parse_anthropic_stream_event(
    data: &str,
    usage_state: &mut AnthropicStreamUsage,
) -> EngineResult<Option<ChatCompletionStreamChunk>> {
    let value: serde_json::Value = serde_json::from_str(data).map_err(|error| {
        EngineError::Provider(format!("invalid Anthropic stream JSON: {error}"))
    })?;
    let event_type = value
        .get("type")
        .and_then(|kind| kind.as_str())
        .unwrap_or("");
    if event_type == "error" {
        return Err(EngineError::Provider(format!(
            "Anthropic stream returned an error: {}",
            value.get("error").unwrap_or(&value)
        )));
    }

    match event_type {
        "message_start" => {
            if let Some(usage) = value
                .get("message")
                .and_then(|message| message.get("usage"))
            {
                usage_state.input_tokens = usage
                    .get("input_tokens")
                    .and_then(|tokens| tokens.as_u64())
                    .unwrap_or(0);
                usage_state.cache_read_input_tokens = usage
                    .get("cache_read_input_tokens")
                    .and_then(|tokens| tokens.as_u64())
                    .unwrap_or(0);
                usage_state.cache_creation_1h_tokens = usage
                    .pointer("/cache_creation/ephemeral_1h_input_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                usage_state.cache_creation_input_tokens = usage
                    .get("cache_creation_input_tokens")
                    .and_then(|tokens| tokens.as_u64())
                    .unwrap_or(0);
            }
            Ok(None)
        }
        "content_block_start" => {
            let block = value.get("content_block");
            if block
                .and_then(|block| block.get("type"))
                .and_then(|kind| kind.as_str())
                != Some("tool_use")
            {
                return Ok(None);
            }
            let index = value
                .get("index")
                .and_then(|index| index.as_u64())
                .unwrap_or(0) as usize;
            Ok(Some(ChatCompletionStreamChunk {
                provider_state: None,
                delta_content: None,
                delta_reasoning_content: None,
                delta_tool_calls: Some(vec![ToolCallDelta {
                    snapshot: false,
                    index,
                    id: block
                        .and_then(|block| block.get("id"))
                        .and_then(|id| id.as_str())
                        .map(str::to_string),
                    call_type: Some("function".to_string()),
                    function_name: block
                        .and_then(|block| block.get("name"))
                        .and_then(|name| name.as_str())
                        .map(str::to_string),
                    function_arguments: None,
                }]),
                finish_reason: None,
                usage: None,
            }))
        }
        "content_block_delta" => {
            let delta = value.get("delta");
            let delta_type = delta
                .and_then(|delta| delta.get("type"))
                .and_then(|kind| kind.as_str());
            let index = value
                .get("index")
                .and_then(|index| index.as_u64())
                .unwrap_or(0) as usize;
            match delta_type {
                Some("text_delta") => Ok(Some(ChatCompletionStreamChunk {
                    provider_state: None,
                    delta_content: delta
                        .and_then(|delta| delta.get("text"))
                        .and_then(|text| text.as_str())
                        .map(str::to_string),
                    ..Default::default()
                })),
                Some("thinking_delta") => Ok(Some(ChatCompletionStreamChunk {
                    provider_state: None,
                    delta_reasoning_content: delta
                        .and_then(|delta| delta.get("thinking"))
                        .and_then(|thinking| thinking.as_str())
                        .map(str::to_string),
                    ..Default::default()
                })),
                Some("input_json_delta") => Ok(Some(ChatCompletionStreamChunk {
                    provider_state: None,
                    delta_tool_calls: Some(vec![ToolCallDelta {
                        snapshot: false,
                        index,
                        function_arguments: delta
                            .and_then(|delta| delta.get("partial_json"))
                            .and_then(|json| json.as_str())
                            .map(str::to_string),
                        ..Default::default()
                    }]),
                    ..Default::default()
                })),
                _ => Ok(None),
            }
        }
        "message_delta" => {
            let output_tokens = value
                .get("usage")
                .and_then(|usage| usage.get("output_tokens"))
                .and_then(|tokens| tokens.as_u64())
                .unwrap_or(0);
            let finish_reason = value
                .get("delta")
                .and_then(|delta| delta.get("stop_reason"))
                .and_then(|reason| reason.as_str())
                .map(|reason| match reason {
                    "tool_use" => "tool_calls",
                    "max_tokens" => "length",
                    _ => "stop",
                })
                .map(str::to_string);
            Ok(Some(ChatCompletionStreamChunk {
                finish_reason,
                usage: Some(TokenUsage {
                    prompt_tokens: usage_state.input_tokens
                        + usage_state.cache_read_input_tokens
                        + usage_state.cache_creation_input_tokens,
                    completion_tokens: output_tokens,
                    total_tokens: usage_state
                        .input_tokens
                        .saturating_add(output_tokens)
                        .saturating_add(usage_state.cache_read_input_tokens)
                        .saturating_add(usage_state.cache_creation_input_tokens),
                    cached_prompt_tokens: usage_state.cache_read_input_tokens,
                    cache_creation_input_tokens: usage_state.cache_creation_input_tokens,
                    cache_creation_1h_tokens: usage_state.cache_creation_1h_tokens,
                    ..Default::default()
                }),
                ..Default::default()
            }))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_input_counts_are_normalized_for_billing() {
        let usage=parse_usage(&serde_json::json!({"input_tokens":200,"output_tokens":100,"cache_read_input_tokens":600,"cache_creation_input_tokens":200,"cache_creation":{"ephemeral_1h_input_tokens":80}})).unwrap();
        assert_eq!(usage.prompt_tokens, 1000);
        assert_eq!(usage.total_tokens, 1100);
        assert_eq!(usage.cache_creation_1h_tokens, 80);
    }
    #[test]
    fn build_body_preserves_all_system_messages_in_order() {
        let provider = AnthropicProvider::new(ProviderConfig {
            label: "anthropic".to_string(),
            base_url: "http://localhost:0".to_string(),
            api_key: "test-key".to_string(),
            standard_model: Some("claude-test".to_string()),
            ..Default::default()
        })
        .unwrap();
        let request = ChatCompletionRequest {
            model: "claude-test".to_string(),
            messages: vec![
                ChatMessage {
                    provider_state: None,
                    role: "system".to_string(),
                    content: Some(MessageContent::text("original instructions")),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                ChatMessage {
                    provider_state: None,
                    role: "system".to_string(),
                    content: Some(MessageContent::text("compacted context")),
                    tool_calls: None,
                    tool_call_id: None,
                    name: Some("context_summary".to_string()),
                },
                ChatMessage {
                    provider_state: None,
                    role: "user".to_string(),
                    content: Some(MessageContent::text("continue")),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
            ],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        };

        let body = provider
            .build_body(&request, "claude-test".to_string())
            .unwrap();
        assert_eq!(body["system"][0]["text"], "original instructions");
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(body["system"][1]["text"], "compacted context");
        assert!(
            body["system"][1].get("cache_control").is_none(),
            "per-turn context must stay past the cache breakpoint"
        );
        assert_eq!(body["messages"].as_array().map(Vec::len), Some(1));
    }

    fn cache_test_body(messages: Vec<ChatMessage>) -> serde_json::Value {
        cache_test_body_with_extra(messages, Default::default())
    }

    fn cache_test_body_with_extra(
        messages: Vec<ChatMessage>,
        extra: std::collections::HashMap<String, serde_json::Value>,
    ) -> serde_json::Value {
        let provider = AnthropicProvider::new(ProviderConfig {
            label: "anthropic".to_string(),
            base_url: "http://localhost:0".to_string(),
            api_key: "test-key".to_string(),
            standard_model: Some("claude-test".to_string()),
            ..Default::default()
        })
        .unwrap();
        provider
            .build_body(
                &ChatCompletionRequest {
                    model: "claude-test".to_string(),
                    messages,
                    tools: None,
                    tool_choice: None,
                    temperature: None,
                    max_tokens: None,
                    stream: None,
                    extra,
                },
                "claude-test".to_string(),
            )
            .unwrap()
    }

    #[test]
    fn static_prefix_keeps_the_cheaper_five_minute_ttl_by_default() {
        let body = cache_test_body(vec![
            plain_message("system", "instructions"),
            plain_message("user", "question"),
        ]);
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert!(
            body["system"][0]["cache_control"].get("ttl").is_none(),
            "the doubled write price of a 1h entry must be opted into"
        );
    }

    #[test]
    fn one_hour_ttl_covers_the_static_prefix_and_never_the_tail() {
        let mut extra = std::collections::HashMap::new();
        extra.insert("reincore_cache_ttl".to_string(), serde_json::json!("1h"));
        let body = cache_test_body_with_extra(
            vec![
                plain_message("system", "instructions"),
                plain_message("user", "first question"),
                plain_message("assistant", "first answer"),
                plain_message("user", "second question"),
            ],
            extra,
        );
        assert_eq!(body["system"][0]["cache_control"]["ttl"], "1h");
        // Anthropic requires longer-lived entries to precede shorter-lived
        // ones. The tail renders after system, so it must stay on the default.
        assert!(tail_cache_control(&body)
            .expect("multi-turn tail is marked")
            .get("ttl")
            .is_none());
    }

    fn message_breakpoints(body: &serde_json::Value) -> usize {
        body["messages"]
            .as_array()
            .map(|messages| {
                messages
                    .iter()
                    .filter_map(|message| message.get("content")?.as_array())
                    .flatten()
                    .filter(|block| block.get("cache_control").is_some())
                    .count()
            })
            .unwrap_or(0)
    }

    fn conversation(turns: usize) -> Vec<ChatMessage> {
        let mut messages = vec![plain_message("system", "instructions")];
        for turn in 0..turns {
            messages.push(plain_message("user", &format!("question {turn}")));
            messages.push(plain_message("assistant", &format!("answer {turn}")));
        }
        messages
    }

    fn tool_call_message(calls: usize) -> ChatMessage {
        ChatMessage {
            provider_state: None,
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(
                (0..calls)
                    .map(|index| crate::models::ToolCall {
                        id: format!("call_{index}"),
                        call_type: "function".to_string(),
                        function: crate::models::FunctionCall {
                            name: "search".to_string(),
                            arguments: "{}".to_string(),
                        },
                    })
                    .collect(),
            ),
            tool_call_id: None,
            name: None,
        }
    }

    #[test]
    fn long_conversations_get_trailing_breakpoints_inside_the_lookback_window() {
        // 50 message positions: both trailing anchors land at 35 and 20.
        let body = cache_test_body(conversation(25));
        assert_eq!(
            message_breakpoints(&body),
            3,
            "tail plus two trailing anchors"
        );
        // Plus the static system block: exactly Anthropic's four-marker limit,
        // which is why the anchor count is capped rather than per-15-positions.
        assert_eq!(body["system"].as_array().map(Vec::len), Some(1));
    }

    #[test]
    fn anchor_placement_never_walks_off_the_front_of_the_conversation() {
        // 30 positions: the first anchor lands at 15, the second would land at
        // position 0, which does not exist and must not be marked.
        let body = cache_test_body(conversation(15));
        assert_eq!(message_breakpoints(&body), 2);
    }

    fn tool_choice_body(choice: serde_json::Value) -> serde_json::Value {
        let provider = AnthropicProvider::new(ProviderConfig {
            label: "anthropic".to_string(),
            base_url: "http://localhost:0".to_string(),
            api_key: "test-key".to_string(),
            standard_model: Some("claude-test".to_string()),
            ..Default::default()
        })
        .unwrap();
        provider
            .build_body(
                &ChatCompletionRequest {
                    model: "claude-test".to_string(),
                    messages: vec![plain_message("user", "question")],
                    tools: None,
                    tool_choice: Some(choice),
                    temperature: None,
                    max_tokens: None,
                    stream: None,
                    extra: Default::default(),
                },
                "claude-test".to_string(),
            )
            .unwrap()
    }

    #[test]
    fn allowed_tools_degrades_to_a_mode_anthropic_understands() {
        // Anthropic cannot express "only these five", so the subset is dropped
        // rather than forwarded - the field verbatim would be a 400.
        let body = tool_choice_body(serde_json::json!({
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [{"type": "function", "function": {"name": "read_file"}}]
        }));
        assert_eq!(body["tool_choice"], serde_json::json!({"type": "auto"}));

        let required = tool_choice_body(serde_json::json!({
            "type": "allowed_tools",
            "mode": "required",
            "tools": []
        }));
        assert_eq!(required["tool_choice"], serde_json::json!({"type": "any"}));
    }

    #[test]
    fn a_named_function_choice_still_maps_to_anthropic_tool_choice() {
        let body = tool_choice_body(serde_json::json!({
            "type": "function",
            "function": {"name": "read_file"}
        }));
        assert_eq!(
            body["tool_choice"],
            serde_json::json!({"type": "tool", "name": "read_file"})
        );
    }

    #[test]
    fn short_conversations_do_not_spend_extra_breakpoints() {
        let body = cache_test_body(conversation(2));
        assert_eq!(
            message_breakpoints(&body),
            1,
            "the tail marker is already within the lookback window"
        );
    }

    #[test]
    fn parallel_tool_calls_collapse_into_a_single_position() {
        // A burst of parallel calls must not be mistaken for a long turn: a run
        // of consecutive tool_use blocks occupies one position. Counting them
        // individually would put this at 27 positions and waste two markers.
        let mut messages = conversation(3);
        messages.push(tool_call_message(20));
        messages.push(plain_message("user", "continue"));
        let body = cache_test_body(messages);
        assert_eq!(message_breakpoints(&body), 1);
    }

    #[test]
    fn an_unrecognized_cache_ttl_falls_back_to_the_default() {
        let mut extra = std::collections::HashMap::new();
        extra.insert("reincore_cache_ttl".to_string(), serde_json::json!("7d"));
        let body = cache_test_body_with_extra(
            vec![
                plain_message("system", "instructions"),
                plain_message("user", "question"),
            ],
            extra,
        );
        assert!(body["system"][0]["cache_control"].get("ttl").is_none());
    }

    fn plain_message(role: &str, text: &str) -> ChatMessage {
        ChatMessage {
            provider_state: None,
            role: role.to_string(),
            content: Some(MessageContent::text(text)),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn tail_cache_control(body: &serde_json::Value) -> Option<&serde_json::Value> {
        body["messages"]
            .as_array()?
            .last()?
            .get("content")?
            .as_array()?
            .last()?
            .get("cache_control")
    }

    #[test]
    fn multi_turn_conversation_marks_the_tail_for_cache_reuse() {
        let body = cache_test_body(vec![
            plain_message("system", "instructions"),
            plain_message("user", "first question"),
            plain_message("assistant", "first answer"),
            plain_message("user", "second question"),
        ]);
        assert_eq!(
            tail_cache_control(&body).and_then(|value| value.get("type")),
            Some(&serde_json::json!("ephemeral"))
        );
    }

    #[test]
    fn single_shot_request_does_not_pay_the_tail_write_premium() {
        let body = cache_test_body(vec![
            plain_message("system", "instructions"),
            plain_message("user", "one off question"),
        ]);
        assert!(
            tail_cache_control(&body).is_none(),
            "a unique tail would be written to cache and never read back"
        );
    }

    #[test]
    fn parse_usage_reads_anthropic_cache_tokens() {
        let usage = parse_usage(&serde_json::json!({
            "input_tokens": 200,
            "output_tokens": 40,
            "cache_read_input_tokens": 150,
            "cache_creation_input_tokens": 32
        }))
        .unwrap();
        assert_eq!(usage.cached_prompt_tokens, 150);
        assert_eq!(usage.cache_creation_input_tokens, 32);
    }

    #[test]
    fn parses_streamed_text_tool_fragments_and_usage() {
        let mut usage = AnthropicStreamUsage::default();
        assert!(parse_anthropic_stream_event(
            r#"{"type":"message_start","message":{"usage":{"input_tokens":20,"cache_read_input_tokens":8}}}"#,
            &mut usage,
        )
        .unwrap()
        .is_none());
        let text = parse_anthropic_stream_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
            &mut usage,
        )
        .unwrap()
        .unwrap();
        assert_eq!(text.delta_content.as_deref(), Some("hi"));
        let tool = parse_anthropic_stream_event(
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}"#,
            &mut usage,
        )
        .unwrap()
        .unwrap();
        assert_eq!(tool.delta_tool_calls.unwrap()[0].index, 1);
        let final_chunk = parse_anthropic_stream_event(
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}"#,
            &mut usage,
        )
        .unwrap()
        .unwrap();
        assert_eq!(final_chunk.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(final_chunk.usage.unwrap().total_tokens, 32);
        assert_eq!(final_chunk.usage.unwrap().prompt_tokens, 28);
    }

    // Multi-part content tests

    #[test]
    fn parse_data_url_extracts_png_base64() {
        let (mime, data) = parse_data_url("data:image/png;base64,iVBORw0KGgo==").unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(data, "iVBORw0KGgo==");
    }

    #[test]
    fn parse_data_url_extracts_jpeg_base64() {
        let (mime, data) = parse_data_url("data:image/jpeg;base64,/9j/4AAQ==").unwrap();
        assert_eq!(mime, "image/jpeg");
        assert_eq!(data, "/9j/4AAQ==");
    }

    #[test]
    fn parse_data_url_returns_none_for_non_data_url() {
        assert!(parse_data_url("https://example.com/image.png").is_none());
    }

    #[test]
    fn parse_data_url_returns_none_for_non_base64() {
        assert!(parse_data_url("data:image/png,notbase64").is_none());
    }

    #[test]
    fn anthropic_message_converts_text_content_to_text_block() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(crate::MessageContent::text("Hello!")),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let value = anthropic_message_from_chat_message(&msg).unwrap();
        assert!(value["content"].is_array());
        assert_eq!(value["content"][0]["type"], "text");
        assert_eq!(value["content"][0]["text"], "Hello!");
    }

    #[test]
    fn anthropic_message_converts_multipart_to_text_and_image_blocks() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(crate::MessageContent::Parts(vec![
                crate::ContentPart::Text {
                    text: "Analyze this image".to_string(),
                },
                crate::ContentPart::ImageUrl {
                    image_url: crate::ImageUrl {
                        url: "data:image/png;base64,iVBORw0KGgo==".to_string(),
                        detail: None,
                    },
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let value = anthropic_message_from_chat_message(&msg).unwrap();
        assert!(value["content"].is_array());
        assert_eq!(value["content"][0]["type"], "text");
        assert_eq!(value["content"][0]["text"], "Analyze this image");
        assert_eq!(value["content"][1]["type"], "image");
        assert_eq!(value["content"][1]["source"]["type"], "base64");
        assert_eq!(value["content"][1]["source"]["media_type"], "image/png");
        assert_eq!(value["content"][1]["source"]["data"], "iVBORw0KGgo==");
    }

    #[test]
    fn anthropic_message_skips_non_data_url_images() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(crate::MessageContent::Parts(vec![
                crate::ContentPart::Text {
                    text: "Look at this".to_string(),
                },
                crate::ContentPart::ImageUrl {
                    image_url: crate::ImageUrl {
                        url: "https://example.com/image.png".to_string(),
                        detail: None,
                    },
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let value = anthropic_message_from_chat_message(&msg).unwrap();
        // The text block is included, but the non-data-URL image is skipped.
        assert_eq!(value["content"].as_array().unwrap().len(), 1);
        assert_eq!(value["content"][0]["type"], "text");
    }

    #[test]
    fn anthropic_message_handles_jpeg_data_url() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(crate::MessageContent::Parts(vec![
                crate::ContentPart::ImageUrl {
                    image_url: crate::ImageUrl {
                        url: "data:image/jpeg;base64,/9j/4AAQ==".to_string(),
                        detail: Some("high".to_string()),
                    },
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let value = anthropic_message_from_chat_message(&msg).unwrap();
        assert_eq!(value["content"][0]["type"], "image");
        assert_eq!(value["content"][0]["source"]["media_type"], "image/jpeg");
        assert_eq!(value["content"][0]["source"]["data"], "/9j/4AAQ==");
    }

    // PDF / document content tests

    #[test]
    fn anthropic_message_converts_pdf_file_part_to_document_block() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(crate::MessageContent::Parts(vec![
                crate::ContentPart::Text {
                    text: "Summarize this PDF".to_string(),
                },
                crate::ContentPart::File {
                    file_url: crate::FileUrl {
                        url: "data:application/pdf;base64,JVBERi0xLjQK".to_string(),
                    },
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let value = anthropic_message_from_chat_message(&msg).unwrap();
        // First block: text
        assert_eq!(value["content"][0]["type"], "text");
        assert_eq!(value["content"][0]["text"], "Summarize this PDF");
        // Second block: document
        assert_eq!(value["content"][1]["type"], "document");
        assert_eq!(value["content"][1]["source"]["type"], "base64");
        assert_eq!(
            value["content"][1]["source"]["media_type"],
            "application/pdf"
        );
        assert_eq!(value["content"][1]["source"]["data"], "JVBERi0xLjQK");
    }

    #[test]
    fn anthropic_message_converts_non_pdf_file_part_to_document_block() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(crate::MessageContent::Parts(vec![
                crate::ContentPart::File {
                    file_url: crate::FileUrl {
                        url: "data:text/csv;base64,QSxCLEMK".to_string(),
                    },
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let value = anthropic_message_from_chat_message(&msg).unwrap();
        assert_eq!(value["content"][0]["type"], "document");
        assert_eq!(value["content"][0]["source"]["media_type"], "text/csv");
        assert_eq!(value["content"][0]["source"]["data"], "QSxCLEMK");
    }

    #[test]
    fn anthropic_message_skips_file_part_with_non_data_url() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(crate::MessageContent::Parts(vec![
                crate::ContentPart::Text {
                    text: "Check this".to_string(),
                },
                crate::ContentPart::File {
                    file_url: crate::FileUrl {
                        url: "https://example.com/doc.pdf".to_string(),
                    },
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let value = anthropic_message_from_chat_message(&msg).unwrap();
        // Only the text block should be present; the non-data-URL file is skipped.
        assert_eq!(value["content"].as_array().unwrap().len(), 1);
        assert_eq!(value["content"][0]["type"], "text");
    }

    #[test]
    fn anthropic_message_handles_mixed_text_image_and_file() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(crate::MessageContent::Parts(vec![
                crate::ContentPart::Text {
                    text: "Analyze these".to_string(),
                },
                crate::ContentPart::ImageUrl {
                    image_url: crate::ImageUrl {
                        url: "data:image/png;base64,iVBOR=".to_string(),
                        detail: None,
                    },
                },
                crate::ContentPart::File {
                    file_url: crate::FileUrl {
                        url: "data:application/pdf;base64,JVBERi0=".to_string(),
                    },
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let value = anthropic_message_from_chat_message(&msg).unwrap();
        assert_eq!(value["content"].as_array().unwrap().len(), 3);
        assert_eq!(value["content"][0]["type"], "text");
        assert_eq!(value["content"][1]["type"], "image");
        assert_eq!(value["content"][2]["type"], "document");
    }
}
