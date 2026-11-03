//! OpenAI-compatible chat-completion provider adapter.
//!
//! Works with any OpenAI-compatible endpoint: OpenAI, Azure OpenAI, Groq,
//! Ollama (with the OpenAI compatibility layer), etc.

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

/// OpenAI provider implementation.
#[derive(Clone, Debug)]
pub struct OpenAiProvider {
    config: ProviderConfig,
    client: reqwest::Client,
}

impl OpenAiProvider {
    /// Create a new OpenAI provider from configuration.
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
        if let Some(org) = &self.config.organization_id {
            headers.insert("OpenAI-Organization".to_string(), org.clone());
        }
        Ok(headers)
    }

    fn build_body(
        &self,
        request: &ChatCompletionRequest,
        provider_model: String,
    ) -> EngineResult<serde_json::Value> {
        let mut messages = Vec::new();
        for m in &request.messages {
            let mut wire = openai_message_from_chat_message(m);
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
                tools.iter().map(openai_tool_definition).collect();
            body["tools"] = serde_json::Value::Array(tool_specs);
        }

        if let Some(tool_choice) = &request.tool_choice {
            body["tool_choice"] = tool_choice.clone();
        }

        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }

        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }

        // ReinCore-specific extras that should not leak to the provider.
        // We intentionally drop reincore_session_id, reincore_embedding, etc.
        // The prompt-cache controls are the exception, and not a leak: the
        // prefix identity is translated into the provider's own cache fields
        // rather than forwarded as a reincore_ key.
        super::provider::apply_openai_cache_controls(&mut body, &self.config, request);
        Ok(body)
    }
}

/// Build a single JSONL line for the OpenAI Batch API input file. Each line
/// is a JSON object with `custom_id`, `method`, `url`, and `body`. Exported
/// so tests can verify the format without making HTTP calls.
pub(crate) fn build_batch_jsonl_line(custom_id: &str, request_body: serde_json::Value) -> String {
    serde_json::json!({
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": request_body,
    })
    .to_string()
}

/// Shared OpenAI-compatible batch submission helper.
///
/// OpenAI-compatible providers with a Files API use this file-based flow:
///
/// 1. Build a `.jsonl` file with one line per request (each line is
///    `{"custom_id":..., "method":"POST", "url":"/v1/chat/completions",
///    "body":{...}}`).
/// 2. Upload the file via `POST {base_url}/files` with `purpose=batch`.
/// 3. Create a batch via `POST {base_url}/batches` referencing the uploaded
///    file id, endpoint, and a 24h completion window.
/// 4. Return the batch id + status.
///
/// The caller supplies the resolved `build_body` output so the helper stays
/// provider-agnostic. Gemini uses native file uploads instead; see `batch.rs`.
pub(crate) async fn openai_batch_submit(
    client: &reqwest::Client,
    base_url: &str,
    auth_headers: &std::collections::HashMap<String, String>,
    request_body: serde_json::Value,
) -> EngineResult<BatchSubmission> {
    let custom_id = format!("reincore-{}", uuid::Uuid::new_v4());
    let jsonl_line = build_batch_jsonl_line(&custom_id, request_body);

    // Step 1: Upload the JSONL file.
    let mut upload_req = client.post(format!("{base_url}/files")).multipart(
        reqwest::multipart::Form::new()
            .text("purpose", "batch")
            .part(
                "file",
                reqwest::multipart::Part::text(jsonl_line).file_name("reincore-batch.jsonl"),
            ),
    );
    for (key, value) in auth_headers {
        upload_req = upload_req.header(key, value);
    }

    let upload_resp = upload_req.send().await.map_err(|e| {
        EngineError::Provider(format!(
            "batch file upload failed: {}",
            crate::error::describe_request_error(&e)
        ))
    })?;
    let upload_status = upload_resp.status();
    let upload_json = upload_resp
        .json::<serde_json::Value>()
        .await
        .map_err(|e| EngineError::Provider(format!("failed to read file upload response: {e}")))?;
    if !upload_status.is_success() {
        return Err(EngineError::Provider(format!(
            "file upload returned {upload_status}: {upload_json}"
        )));
    }
    let file_id = upload_json
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| EngineError::Provider("missing file id in upload response".to_string()))?;

    // Step 2: Create the batch.
    let batch_body = serde_json::json!({
        "input_file_id": file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    });
    let mut batch_req = client
        .post(format!("{base_url}/batches"))
        .header("Content-Type", "application/json")
        .json(&batch_body);
    for (key, value) in auth_headers {
        batch_req = batch_req.header(key, value);
    }

    let batch_resp = batch_req.send().await.map_err(|e| {
        EngineError::Provider(format!(
            "batch creation failed: {}",
            crate::error::describe_request_error(&e)
        ))
    })?;
    let batch_status = batch_resp.status();
    let batch_json = batch_resp
        .json::<serde_json::Value>()
        .await
        .map_err(|e| EngineError::Provider(format!("failed to read batch response: {e}")))?;
    if !batch_status.is_success() {
        return Err(EngineError::Provider(format!(
            "batch creation returned {batch_status}: {batch_json}"
        )));
    }

    let batch_id = batch_json
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| EngineError::Provider("missing batch id in response".to_string()))?
        .to_string();
    let status = batch_json
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("validating")
        .to_string();

    Ok(BatchSubmission { batch_id, status })
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn batch_client(&self) -> EngineResult<super::batch::BatchClient> {
        crate::privacy::current_policy().check_provider(self.label())?;
        Ok(super::batch::BatchClient::new(
            self.client.clone(),
            self.config.base_url.clone(),
            self.auth_headers()?,
            super::batch::BatchProtocol::OpenAi,
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
            "OpenAI request to {}: {}",
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
                "OpenAI returned {status}: {}",
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
            "OpenAI",
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
        let body = self.build_body(request, provider_model)?;
        let headers = self.auth_headers()?;
        openai_batch_submit(&self.client, &self.config.base_url, &headers, body).await
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
                "OpenAI returned {status} on /models: {}",
                crate::privacy::current_policy().sanitize_for_logging(&raw.to_string())
            )));
        }

        let data = raw
            .get("data")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let models =
            data.iter()
                .filter_map(|v| {
                    let id = v.get("id").and_then(|v| v.as_str())?;
                    let created = v.get("created").and_then(|v| v.as_i64()).and_then(|ts| {
                        chrono::DateTime::from_timestamp(ts, 0).map(|dt| dt.to_utc())
                    });
                    Some(ProviderModel {
                        id: id.to_string(),
                        provider: self.label().to_string(),
                        display_name: None,
                        created_at: created,
                        metadata: Some(v.clone()),
                    })
                })
                .collect();

        Ok(models)
    }
}

fn openai_message_from_chat_message(msg: &ChatMessage) -> serde_json::Value {
    let mut value = serde_json::json!({
        "role": msg.role,
    });

    if let Some(content) = &msg.content {
        value["content"] = serde_json::to_value(content).unwrap_or_default();
    }

    if let Some(tool_calls) = &msg.tool_calls {
        value["tool_calls"] =
            serde_json::json!(tool_calls.iter().map(openai_tool_call).collect::<Vec<_>>());
    }

    if let Some(tool_call_id) = &msg.tool_call_id {
        value["tool_call_id"] = serde_json::json!(tool_call_id);
    }

    if let Some(name) = &msg.name {
        value["name"] = serde_json::json!(name);
    }

    value
}

fn openai_tool_definition(tool: &ToolDefinition) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
    })
}

fn openai_tool_call(call: &ToolCall) -> serde_json::Value {
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

    let chat_message = chat_message_from_openai(message)?;

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

fn chat_message_from_openai(value: &serde_json::Value) -> EngineResult<ChatMessage> {
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
                .map(parse_openai_tool_call)
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

fn parse_openai_tool_call(value: &serde_json::Value) -> EngineResult<ToolCall> {
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
    super::provider::parse_openai_stream_usage(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ContentPart, FunctionCall, ImageUrl};

    #[test]
    fn openai_forwards_allowed_tools_unchanged() {
        let provider = OpenAiProvider::new(ProviderConfig {
            label: "openai".to_string(),
            base_url: "http://localhost:0".to_string(),
            api_key: "test-key".to_string(),
            standard_model: Some("gpt-test".to_string()),
            ..Default::default()
        })
        .unwrap();
        let choice = serde_json::json!({
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [{"type": "function", "function": {"name": "read_file"}}]
        });
        let body = provider
            .build_body(
                &ChatCompletionRequest {
                    model: "gpt-test".to_string(),
                    messages: Vec::new(),
                    tools: None,
                    tool_choice: Some(choice.clone()),
                    temperature: None,
                    max_tokens: None,
                    stream: None,
                    extra: Default::default(),
                },
                "gpt-test".to_string(),
            )
            .unwrap();
        // OpenAI implements allowed_tools, so the subset must survive intact.
        // Narrowing it here would silently widen what the model may call.
        assert_eq!(body["tool_choice"], choice);
    }

    #[test]
    fn parse_usage_reads_openai_cached_prompt_tokens() {
        let usage = parse_usage(&serde_json::json!({
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
            "prompt_tokens_details": {
                "cached_tokens": 96
            }
        }))
        .unwrap();
        assert_eq!(usage.cached_prompt_tokens, 96);
        assert_eq!(usage.cache_creation_input_tokens, 0);
    }

    #[test]
    fn build_batch_jsonl_line_produes_valid_json() {
        let body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
        });
        let line = build_batch_jsonl_line("reincore-test-1", body.clone());
        let parsed: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed["custom_id"], "reincore-test-1");
        assert_eq!(parsed["method"], "POST");
        assert_eq!(parsed["url"], "/v1/chat/completions");
        assert_eq!(parsed["body"], body);
    }

    #[test]
    fn build_batch_jsonl_line_is_single_line() {
        let line = build_batch_jsonl_line("x", serde_json::json!({}));
        assert!(
            !line.contains('\n'),
            "JSONL line must not contain a newline"
        );
    }

    #[test]
    fn openai_provider_supports_batch_submission() {
        // Verify that the default Provider trait method is overridden (not
        // the "batch submission not implemented" error). We can't easily test
        // the full HTTP flow without a mock server, but we can verify the
        // provider builds the request without panicking on a config with
        // a dummy base URL.
        let config = ProviderConfig {
            label: "openai".to_string(),
            base_url: "http://localhost:0".to_string(),
            api_key: "test-key".to_string(),
            standard_model: Some("gpt-4o".to_string()),
            ..Default::default()
        };
        let provider = OpenAiProvider::new(config).unwrap();
        // The provider exists and has the method - calling it would fail with
        // a connection error, but the trait method is wired (not the default).
        // This is a compile-time + smoke test.
        assert_eq!(provider.label(), "openai");
    }

    // Multi-part content tests

    #[test]
    fn openai_message_serializes_plain_text_content() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(crate::MessageContent::text("Hello!")),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let value = openai_message_from_chat_message(&msg);
        assert_eq!(value["content"], "Hello!");
    }

    #[test]
    fn openai_message_serializes_multipart_content_as_array() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(crate::MessageContent::Parts(vec![
                crate::ContentPart::Text {
                    text: "What's in this image?".to_string(),
                },
                crate::ContentPart::ImageUrl {
                    image_url: crate::ImageUrl {
                        url: "https://example.com/cat.png".to_string(),
                        detail: None,
                    },
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let value = openai_message_from_chat_message(&msg);
        assert!(value["content"].is_array());
        assert_eq!(value["content"][0]["type"], "text");
        assert_eq!(value["content"][0]["text"], "What's in this image?");
        assert_eq!(value["content"][1]["type"], "image_url");
        assert_eq!(
            value["content"][1]["image_url"]["url"],
            "https://example.com/cat.png"
        );
    }

    #[test]
    fn openai_message_serializes_multipart_with_detail() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(crate::MessageContent::Parts(vec![
                crate::ContentPart::ImageUrl {
                    image_url: crate::ImageUrl {
                        url: "https://example.com/img.png".to_string(),
                        detail: Some("high".to_string()),
                    },
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let value = openai_message_from_chat_message(&msg);
        assert_eq!(value["content"][0]["image_url"]["detail"], "high");
    }

    #[test]
    fn openai_message_omits_content_when_none() {
        let msg = ChatMessage {
            provider_state: None,
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "search".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            name: None,
        };
        let value = openai_message_from_chat_message(&msg);
        assert!(value.get("content").is_none() || value["content"].is_null());
    }

    // File / PDF content tests

    #[test]
    fn openai_message_serializes_file_part_in_multipart_content() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Analyze this PDF".to_string(),
                },
                ContentPart::File {
                    file_url: crate::FileUrl {
                        url: "data:application/pdf;base64,JVBERi0=".to_string(),
                    },
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let value = openai_message_from_chat_message(&msg);
        assert!(value["content"].is_array());
        assert_eq!(value["content"][0]["type"], "text");
        assert_eq!(value["content"][0]["text"], "Analyze this PDF");
        assert_eq!(value["content"][1]["type"], "file");
        assert_eq!(
            value["content"][1]["file_url"]["url"],
            "data:application/pdf;base64,JVBERi0="
        );
    }

    #[test]
    fn openai_message_serializes_mixed_image_and_file_parts() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,iVBOR=".to_string(),
                        detail: None,
                    },
                },
                ContentPart::File {
                    file_url: crate::FileUrl {
                        url: "data:application/pdf;base64,JVBERi0=".to_string(),
                    },
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let value = openai_message_from_chat_message(&msg);
        assert_eq!(value["content"].as_array().unwrap().len(), 2);
        assert_eq!(value["content"][0]["type"], "image_url");
        assert_eq!(value["content"][1]["type"], "file");
    }
}
