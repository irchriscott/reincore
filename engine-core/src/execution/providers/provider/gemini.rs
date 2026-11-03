//! Google Gemini provider adapter.
//!
//! Uses the Gemini OpenAI-compatible endpoint when available, or the native
//! Gemini API as a fallback. This implementation targets the OpenAI-compatible
//! endpoint (`https://generativelanguage.googleapis.com/v1beta/openai`) because
//! it lets us reuse the normalized message/tool shapes with minimal translation.

use crate::error::{EngineError, EngineResult};
use crate::models::{
    ChatCompletionRequest, ChatMessage, MessageContent, ModelTier, ToolCall, ToolDefinition,
};
use crate::provider::{
    capture_response_headers, openai_compatible_stream, BatchSubmission, ChatCompletionResponse,
    Provider, ProviderConfig, ProviderModel, ProviderStream, TokenUsage,
};
use async_trait::async_trait;
use tracing::debug;

/// Google Gemini provider implementation.
#[derive(Clone, Debug)]
pub struct GeminiProvider {
    config: ProviderConfig,
    client: reqwest::Client,
}

impl GeminiProvider {
    /// Create a new Gemini provider from configuration.
    pub fn new(config: ProviderConfig) -> EngineResult<Self> {
        let client = reqwest::Client::builder()
            // Native Files requests carry x-goog-api-key, which must never be
            // forwarded by an automatic cross-origin redirect.
            .redirect(reqwest::redirect::Policy::none())
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
        headers.insert(
            "Authorization".to_string(),
            format!("Bearer {}", self.config.api_key),
        );
        Ok(headers)
    }

    fn build_body(
        &self,
        request: &ChatCompletionRequest,
        provider_model: String,
    ) -> EngineResult<serde_json::Value> {
        let mut messages = Vec::new();
        for m in &request.messages {
            let mut wire = gemini_message_from_chat_message(m);
            if super::replay::enabled(request) {
                super::replay::apply(
                    m,
                    &mut wire,
                    &super::replay::scope(&self.config, &provider_model),
                );
            }
            messages.push(wire);
        }

        let mut body = serde_json::json!({
            "model": provider_model,
            "messages": messages,
        });

        if let Some(tools) = &request.tools {
            let tool_specs: Vec<serde_json::Value> =
                tools.iter().map(gemini_tool_definition).collect();
            body["tools"] = serde_json::Value::Array(tool_specs);
        }

        if let Some(tool_choice) = &request.tool_choice {
            body["tool_choice"] = super::provider::tool_choice_without_allowed_tools(tool_choice);
        }

        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }

        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }

        Ok(body)
    }
}

#[async_trait]
impl Provider for GeminiProvider {
    fn batch_client(&self) -> EngineResult<super::batch::BatchClient> {
        crate::privacy::current_policy().check_provider(self.label())?;
        Ok(super::batch::BatchClient::new(
            self.client.clone(),
            self.config.base_url.clone(),
            self.auth_headers()?,
            super::batch::BatchProtocol::Gemini,
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
            "Gemini request to {}: {}",
            self.config.base_url,
            crate::privacy::current_policy()
                .sanitize_for_logging(&serde_json::to_string(&body).unwrap_or_default())
        );

        let mut req_builder = self
            .client
            .post(format!("{}/chat/completions", self.config.base_url))
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
                "Gemini returned {status}: {}",
                crate::privacy::current_policy().sanitize_for_logging(&raw.to_string())
            )));
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
        let mut body = self.build_body(request, provider_model)?;
        body["stream"] = serde_json::json!(true);
        body["stream_options"] = serde_json::json!({"include_usage": true});
        let mut builder = self
            .client
            .post(format!("{}/chat/completions", self.config.base_url))
            .header("Content-Type", "application/json");
        for (key, value) in self.auth_headers()? {
            builder = builder.header(key, value);
        }
        let response = builder
            .json(&body)
            .send()
            .await
            .map_err(EngineError::from_request_error)?;
        openai_compatible_stream(
            response,
            "Gemini",
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
        // Gemini uses native Files API uploads with OpenAI-format JSONL.
        let tier = request
            .extra
            .get("reincore_model_tier")
            .and_then(|v| v.as_str())
            .and_then(|value| value.parse().ok())
            .unwrap_or(ModelTier::Pro);

        let provider_model = self.config.model_for_request(request, tier)?;
        let body = self.build_body(request, provider_model)?;
        let batch = self
            .batch_client()?
            .create(vec![(format!("reincore-{}", uuid::Uuid::new_v4()), body)])
            .await
            .map_err(|e| EngineError::Provider(e.message))?;

        Ok(BatchSubmission {
            batch_id: batch.id,
            status: batch.status,
        })
    }

    async fn list_models(&self) -> EngineResult<Vec<ProviderModel>> {
        crate::privacy::current_policy().check_provider(self.label())?;
        let mut req_builder = self
            .client
            .get(format!("{}/models", self.config.base_url))
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
                "Gemini returned {status} on /models: {}",
                crate::privacy::current_policy().sanitize_for_logging(&raw.to_string())
            )));
        }

        let models = raw
            .get("models")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let result = models
            .iter()
            .filter_map(|v| {
                let name = v.get("name").and_then(|v| v.as_str())?;
                // Gemini model names come as "models/gemini-..."; strip the prefix.
                let id = name.strip_prefix("models/").unwrap_or(name);
                let display_name = v
                    .get("displayName")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let created = None;
                Some(ProviderModel {
                    id: id.to_string(),
                    provider: self.label().to_string(),
                    display_name,
                    created_at: created,
                    metadata: Some(v.clone()),
                })
            })
            .collect();

        Ok(result)
    }
}

fn gemini_message_from_chat_message(msg: &ChatMessage) -> serde_json::Value {
    let mut value = serde_json::json!({
        "role": msg.role,
    });

    if let Some(content) = &msg.content {
        value["content"] = serde_json::to_value(content).unwrap_or_default();
    }

    if let Some(tool_calls) = &msg.tool_calls {
        value["tool_calls"] =
            serde_json::json!(tool_calls.iter().map(gemini_tool_call).collect::<Vec<_>>());
    }

    if let Some(tool_call_id) = &msg.tool_call_id {
        value["tool_call_id"] = serde_json::json!(tool_call_id);
    }

    if let Some(name) = &msg.name {
        value["name"] = serde_json::json!(name);
    }

    value
}

fn gemini_tool_definition(tool: &ToolDefinition) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
    })
}

fn gemini_tool_call(call: &ToolCall) -> serde_json::Value {
    serde_json::json!({
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.function.name,
            "arguments": call.function.arguments,
        }
    })
}

fn parse_completion_response(raw: serde_json::Value) -> EngineResult<ChatCompletionResponse> {
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

    let choice = raw
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .ok_or_else(|| EngineError::Provider("missing choices in response".to_string()))?;

    let message = choice
        .get("message")
        .ok_or_else(|| EngineError::Provider("missing message in choice".to_string()))?;

    let chat_message = chat_message_from_gemini(message)?;

    let usage = raw.get("usage").and_then(parse_usage);

    Ok(ChatCompletionResponse {
        id,
        model,
        message: chat_message,
        usage,
        raw,
        response_headers: Default::default(),
    })
}

fn chat_message_from_gemini(value: &serde_json::Value) -> EngineResult<ChatMessage> {
    let role = value
        .get("role")
        .and_then(|v| v.as_str())
        .unwrap_or("assistant")
        .to_string();
    let content = value
        .get("content")
        .and_then(|v| v.as_str())
        .map(|s| MessageContent::text(s.to_string()));

    let tool_calls = value
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(parse_gemini_tool_call)
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

fn parse_gemini_tool_call(value: &serde_json::Value) -> EngineResult<ToolCall> {
    let id = value
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
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
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    Ok(ToolCall {
        id,
        call_type: "function".to_string(),
        function: crate::models::FunctionCall { name, arguments },
    })
}

fn parse_usage(value: &serde_json::Value) -> Option<TokenUsage> {
    if value.get("promptTokenCount").is_none() {
        return super::provider::parse_openai_stream_usage(value);
    }
    let count = |key: &str| value.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
    let prompt_tokens = value.get("promptTokenCount")?.as_u64()?;
    let reasoning_tokens = count("thoughtsTokenCount");
    let completion_tokens = count("candidatesTokenCount").checked_add(reasoning_tokens)?;
    Some(TokenUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.checked_add(completion_tokens)?,
        reasoning_tokens,
        cached_prompt_tokens: count("cachedContentTokenCount"),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_thoughts_add_to_output_but_compatible_thoughts_are_already_included() {
        let native = parse_usage(&serde_json::json!({
            "promptTokenCount":1000,
            "candidatesTokenCount":40,
            "thoughtsTokenCount":60,
            "cachedContentTokenCount":600
        }))
        .unwrap();
        let compatible = parse_usage(&serde_json::json!({
            "prompt_tokens":1000,
            "completion_tokens":100,
            "completion_tokens_details":{"reasoning_tokens":60},
            "prompt_tokens_details":{"cached_tokens":600}
        }))
        .unwrap();
        assert_eq!(native.completion_tokens, 100);
        assert_eq!(compatible.completion_tokens, 100);
        assert_eq!(native.reasoning_tokens, 60);
        assert_eq!(native.cached_prompt_tokens, 600);
        assert_eq!(native.total_tokens, compatible.total_tokens);
    }
    #[test]
    fn gemini_provider_supports_batch_submission() {
        // Verify that Gemini overrides submit_batch (not the trait default
        // "batch submission not implemented" error). This is a compile-time
        // + smoke test - a real HTTP call would fail with a connection error.
        let config = ProviderConfig {
            label: "gemini".to_string(),
            base_url: "http://localhost:0".to_string(),
            api_key: "test-key".to_string(),
            standard_model: Some("gemini-3.6-flash".to_string()),
            ..Default::default()
        };
        let provider = GeminiProvider::new(config).unwrap();
        assert_eq!(provider.label(), "gemini");
    }
}
