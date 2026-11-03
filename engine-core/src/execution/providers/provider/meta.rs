//! Meta Model API provider adapter (Meta Muse).
//!
//! Meta exposes an OpenAI-compatible API at `https://api.meta.ai/v1`. This
//! adapter reuses the OpenAI message/response translation with Meta's base
//! URL and bearer-token auth (`META_API_KEY`). It serves Muse models such as
//! `muse-spark-1.1` through `/chat/completions`, with tool calling and
//! streaming supported by the API.

use crate::error::{EngineError, EngineResult};
use crate::models::{
    ChatCompletionRequest, ChatMessage, MessageContent, ModelTier, ToolCall, ToolDefinition,
};
use crate::provider::{
    capture_response_headers, openai_compatible_stream, ChatCompletionResponse, Provider,
    ProviderConfig, ProviderModel, ProviderStream, TokenUsage,
};
use async_trait::async_trait;
use tracing::debug;

/// Meta Muse provider implementation.
#[derive(Clone, Debug)]
pub struct MetaProvider {
    config: ProviderConfig,
    client: reqwest::Client,
}

impl MetaProvider {
    /// Create a new Meta provider from configuration. The base URL defaults
    /// to the Meta Model API; override it with `META_BASE_URL` for a proxy.
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
            let mut wire = meta_message_from_chat_message(m);
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
                tools.iter().map(meta_tool_definition).collect();
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
impl Provider for MetaProvider {
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
            "Meta request to {}: {}",
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
                "Meta returned {status}: {}",
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
            .and_then(|v| v.as_str())
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
            "Meta",
            super::replay::request_scope(
                &self.config,
                body["model"].as_str().unwrap_or_default(),
                request,
            ),
        )
        .await
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
                "Meta returned {status} on /models: {}",
                crate::privacy::current_policy().sanitize_for_logging(&raw.to_string())
            )));
        }

        let models = raw
            .get("data")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let result = models
            .iter()
            .filter_map(|v| {
                let id = v.get("id").and_then(|v| v.as_str())?;
                Some(ProviderModel {
                    id: id.to_string(),
                    provider: self.label().to_string(),
                    display_name: None,
                    created_at: None,
                    metadata: Some(v.clone()),
                })
            })
            .collect();

        Ok(result)
    }
}

fn meta_message_from_chat_message(msg: &ChatMessage) -> serde_json::Value {
    let mut value = serde_json::json!({
        "role": msg.role,
    });

    if let Some(content) = &msg.content {
        value["content"] = serde_json::to_value(content).unwrap_or_default();
    }

    if let Some(tool_calls) = &msg.tool_calls {
        value["tool_calls"] =
            serde_json::json!(tool_calls.iter().map(meta_tool_call).collect::<Vec<_>>());
    }

    if let Some(tool_call_id) = &msg.tool_call_id {
        value["tool_call_id"] = serde_json::json!(tool_call_id);
    }

    if let Some(name) = &msg.name {
        value["name"] = serde_json::json!(name);
    }

    value
}

fn meta_tool_definition(tool: &ToolDefinition) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
    })
}

fn meta_tool_call(call: &ToolCall) -> serde_json::Value {
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

    let role = message
        .get("role")
        .and_then(|v| v.as_str())
        .unwrap_or("assistant")
        .to_string();
    let content = message
        .get("content")
        .and_then(|v| v.as_str())
        .map(|s| MessageContent::text(s.to_string()));

    let tool_calls = message
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|tc| {
                    let id = tc.get("id").and_then(|v| v.as_str())?.to_string();
                    let function = tc.get("function")?;
                    let name = function.get("name").and_then(|v| v.as_str())?.to_string();
                    let arguments = function
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}")
                        .to_string();
                    Some(ToolCall {
                        id,
                        call_type: "function".to_string(),
                        function: crate::models::FunctionCall { name, arguments },
                    })
                })
                .collect()
        });

    let usage = raw.get("usage").and_then(parse_usage);

    Ok(ChatCompletionResponse {
        id,
        model,
        message: ChatMessage {
            provider_state: None,
            role,
            content,
            tool_calls,
            tool_call_id: None,
            name: None,
        },
        usage,
        raw,
        response_headers: Default::default(),
    })
}

fn parse_usage(value: &serde_json::Value) -> Option<TokenUsage> {
    super::provider::parse_openai_stream_usage(value)
}

#[cfg(test)]
mod fixture_tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Minimal OpenAI-compatible fixture: answers one `/chat/completions`
    /// POST and one `/models` GET, then stops.
    async fn fixture_server() -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let handle = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 16384];
                let n = socket.read(&mut buf).await.unwrap();
                let head = String::from_utf8_lossy(&buf[..n]);
                let body = if head.starts_with("POST") {
                    assert!(
                        head.contains("authorization: Bearer test-key")
                            || head.contains("Authorization: Bearer test-key"),
                        "fixture expected the Meta bearer token, got: {head}"
                    );
                    serde_json::json!({
                        "id": "chatcmpl-1",
                        "model": "muse-spark-1.1",
                        "choices": [{
                            "message": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 5,
                            "completion_tokens": 3,
                            "total_tokens": 8
                        }
                    })
                    .to_string()
                } else {
                    serde_json::json!({"data": [{"id": "muse-spark-1.1"}]}).to_string()
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });
        (url, handle)
    }

    fn test_config(base_url: String) -> ProviderConfig {
        ProviderConfig {
            label: "meta".to_string(),
            base_url,
            api_key: "test-key".to_string(),
            standard_model: Some("muse-spark-1.1".to_string()),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn meta_complete_and_list_models_against_fixture() {
        let (url, handle) = fixture_server().await;
        let provider = MetaProvider::new(test_config(url)).unwrap();

        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "muse-spark-1.1",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let response = provider.complete(&request).await.unwrap();
        assert_eq!(response.model, "muse-spark-1.1");
        assert_eq!(response.message.role, "assistant");
        assert_eq!(
            response.message.content,
            Some(MessageContent::Text("hello".to_string()))
        );
        let usage = response.usage.expect("fixture usage parses");
        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.completion_tokens, 3);

        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "muse-spark-1.1");
        assert_eq!(models[0].provider, "meta");

        handle.await.unwrap();
    }
}
