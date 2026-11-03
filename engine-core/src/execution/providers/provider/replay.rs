//! Lossless, opaque provider state for continuing a reasoning/tool turn.
//!
//! This is protocol state, not display text. It must not be summarized or sent
//! to a different provider/model, or applied to an edited assistant message.

use super::{ChatCompletionResponse, ProviderConfig};
use crate::{ChatMessage, EngineResult};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const ENABLED_KEY: &str = "reincore_provider_replay_enabled";

pub(crate) fn enabled(request: &crate::ChatCompletionRequest) -> bool {
    request
        .extra
        .get(ENABLED_KEY)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

pub(crate) fn request_scope(
    config: &ProviderConfig,
    model: &str,
    request: &crate::ChatCompletionRequest,
) -> Option<String> {
    enabled(request).then(|| scope(config, model))
}

/// Snapshot the operator's setting for the entire call (including retries),
/// overriding client extras. Strip disabled state before costing or dispatch.
pub fn prepare_request_replay(
    request: &mut crate::ChatCompletionRequest,
    group: &str,
    enabled: bool,
) {
    request
        .extra
        .insert(ENABLED_KEY.into(), Value::Bool(enabled));
    if enabled {
        retain_request_group_state(&mut request.messages, group);
    } else {
        for message in &mut request.messages {
            message.provider_state = None;
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderReplay {
    pub scope: String,
    pub message_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_group_id: Option<String>,
    pub payload: Value,
}

pub(crate) fn scope(config: &ProviderConfig, model: &str) -> String {
    hex::encode(Sha256::digest(
        json!([
            config.label,
            config.base_url,
            model,
            config.organization_id,
            config.api_key
        ])
        .to_string(),
    ))
}

fn message_hash(message: &ChatMessage) -> String {
    // Canonicalize arguments: VS Code parses and reserializes them. Exclude
    // only that insignificant formatting, never call IDs, ordering or content.
    let calls = message.tool_calls.as_ref().map(|calls| {
        calls
            .iter()
            .map(|call| {
                json!({
                    "id": call.id,
                    "name": call.function.name,
                    "arguments": serde_json::from_str::<Value>(&call.function.arguments)
                        .unwrap_or_else(|_| Value::String(call.function.arguments.clone()))
                })
            })
            .collect::<Vec<_>>()
    });
    hex::encode(Sha256::digest(
        json!({
            "role": message.role,
            "content": message.full_text(),
            "calls": calls.filter(|calls| !calls.is_empty())
        })
        .to_string(),
    ))
}

impl ProviderReplay {
    pub fn for_request_group(mut self, request_group_id: &str) -> Self {
        self.request_group_id = Some(request_group_id.to_owned());
        self
    }

    pub fn bind(mut self, message: &ChatMessage) -> Self {
        self.message_hash = message_hash(message);
        self
    }
}

/// Drop completed-group and legacy unscoped replay before estimation or dispatch.
/// Conversation history remains intact; only opaque provider state is removed.
pub fn retain_request_group_state(messages: &mut [ChatMessage], request_group_id: &str) {
    let start = messages
        .iter()
        .rposition(|message| {
            message.role.eq_ignore_ascii_case("user")
                && message.tool_call_id.is_none()
                && (crate::user_content::session_user_request(&message.full_text()).is_some()
                    || message.content_len() > message.full_text().len())
        })
        .map_or(0, |index| index + 1);
    for (index, message) in messages.iter_mut().enumerate() {
        let keep = message.provider_state.as_ref().is_some_and(|state| {
            index >= start
                && state.request_group_id.as_deref() == Some(request_group_id)
                && message.role == "assistant"
                && state.message_hash == message_hash(message)
        });
        if !keep {
            message.provider_state = None;
        }
    }
}

/// Additional protocol content for token estimation, excluding the internal
/// envelope and visible text/tool blocks already counted by the caller.
/// Opaque signatures use a conservative text estimate; provider usage remains
/// authoritative because providers tokenize protocol fields differently.
pub(crate) fn estimation_text(message: &ChatMessage) -> String {
    let Some(state) = message
        .provider_state
        .as_ref()
        .filter(|state| message.role == "assistant" && state.message_hash == message_hash(message))
    else {
        return String::new();
    };
    let mut parts = Vec::new();
    for key in [
        "thinking",
        "reasoning_content",
        "reasoning",
        "reasoning_details",
        "extra_content",
        "tool_extras",
    ] {
        if let Some(value) = state.payload.get(key).filter(|value| !value.is_null()) {
            parts.push(
                value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string()),
            );
        }
    }
    if let Some(blocks) = state.payload.get("content").and_then(Value::as_array) {
        for block in blocks {
            if matches!(
                block["type"].as_str(),
                Some("thinking" | "redacted_thinking")
            ) {
                parts.push(block.to_string());
            }
        }
    }
    parts.join("\n")
}

/// Keep only protocol fields. Arbitrary raw response fields are never replayed.
pub(crate) fn capture(scope: String, raw: &Value, anthropic: bool) -> Option<ProviderReplay> {
    let message = raw
        .pointer("/choices/0/message")
        .or_else(|| raw.get("message"))
        .unwrap_or(raw);
    let mut payload = json!({});
    if anthropic {
        let blocks = message.get("content")?.as_array()?;
        if blocks
            .iter()
            .any(|b| matches!(b["type"].as_str(), Some("thinking" | "redacted_thinking")))
        {
            payload["content"] = Value::Array(blocks.clone());
        }
    } else {
        for key in [
            "thinking",
            "reasoning_content",
            "reasoning",
            "reasoning_details",
            "extra_content",
        ] {
            if let Some(value) = message.get(key).filter(|v| !v.is_null()) {
                payload[key] = value.clone();
            }
        }
        if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
            let extras: Vec<_> = calls
                .iter()
                .map(|call| call.get("extra_content").cloned().unwrap_or(Value::Null))
                .collect();
            if extras.iter().any(|v| !v.is_null()) {
                payload["tool_extras"] = json!(extras);
            }
        }
    }
    (!payload.as_object()?.is_empty()).then_some(ProviderReplay {
        scope,
        message_hash: String::new(),
        request_group_id: None,
        payload,
    })
}

pub(crate) fn capture_response(
    response: &mut ChatCompletionResponse,
    scope: Option<String>,
    anthropic: bool,
) {
    response.message.provider_state = scope
        .and_then(|scope| capture(scope, &response.raw, anthropic))
        .map(|state| state.bind(&response.message));
}

pub(crate) fn apply(message: &ChatMessage, wire: &mut Value, scope: &str) {
    let Some(state) = message.provider_state.as_ref().filter(|state| {
        message.role == "assistant"
            && state.scope == scope
            && state.message_hash == message_hash(message)
    }) else {
        return;
    };
    for key in [
        "thinking",
        "reasoning_content",
        "reasoning",
        "reasoning_details",
        "extra_content",
        "content",
    ] {
        if let Some(value) = state.payload.get(key) {
            wire[key] = value.clone();
        }
    }
    if let (Some(extras), Some(calls)) = (
        state.payload.get("tool_extras").and_then(Value::as_array),
        wire.get_mut("tool_calls").and_then(Value::as_array_mut),
    ) {
        for (call, extra) in calls.iter_mut().zip(extras) {
            if !extra.is_null() {
                call["extra_content"] = extra.clone();
            }
        }
    }
}

/// Assemble provider deltas before emitting a single replay snapshot. Indexed
/// structured reasoning/signatures retain their order and exact string bytes.
#[derive(Default)]
pub(crate) struct ReplayAccumulator {
    message: Value,
    anthropic_inputs: std::collections::BTreeMap<usize, String>,
}

fn merge(target: &mut Value, delta: &Value) {
    match delta {
        Value::Object(fields) => {
            if !target.is_object() {
                *target = json!({});
            }
            for (key, value) in fields {
                if key == "index" {
                    continue;
                }
                if matches!(key.as_str(), "id" | "type" | "role") {
                    target[key] = value.clone();
                } else {
                    merge(&mut target[key], value);
                }
            }
        }
        Value::Array(items) => {
            if !target.is_array() {
                *target = json!([]);
            }
            let target = target.as_array_mut().unwrap();
            for item in items {
                if let Some(index) = item
                    .get("index")
                    .and_then(Value::as_u64)
                    .filter(|i| *i < 4096)
                {
                    let index = index as usize;
                    target.resize(target.len().max(index + 1), Value::Null);
                    merge(&mut target[index], item);
                } else {
                    target.push(item.clone());
                }
            }
        }
        Value::String(fragment) => {
            if let Value::String(text) = target {
                text.push_str(fragment);
            } else {
                *target = delta.clone();
            }
        }
        Value::Null => {}
        _ => *target = delta.clone(),
    }
}

impl ReplayAccumulator {
    pub(crate) fn compatible(&mut self, raw: &Value) {
        if let Some(delta) = raw
            .pointer("/choices/0/delta")
            .or_else(|| raw.get("message"))
        {
            merge(&mut self.message, delta);
        }
    }

    pub(crate) fn anthropic(&mut self, raw: &Value) -> EngineResult<()> {
        let index = raw["index"].as_u64().unwrap_or(0) as usize;
        if index >= 4096 {
            return Err(crate::EngineError::Provider(
                "invalid content block index".into(),
            ));
        }
        match raw["type"].as_str() {
            Some("content_block_start") => {
                if !self.message["content"].is_array() {
                    self.message = json!({"content":[]});
                }
                let blocks = self.message["content"].as_array_mut().unwrap();
                blocks.resize(blocks.len().max(index + 1), Value::Null);
                blocks[index] = raw["content_block"].clone();
            }
            Some("content_block_delta") => {
                if self
                    .message
                    .get("content")
                    .and_then(Value::as_array)
                    .and_then(|blocks| blocks.get(index))
                    .map_or(true, Value::is_null)
                {
                    return Err(crate::EngineError::Provider(
                        "delta without a content block start".into(),
                    ));
                }
                let delta = &raw["delta"];
                if let Some(partial) = delta["partial_json"].as_str() {
                    self.anthropic_inputs
                        .entry(index)
                        .or_default()
                        .push_str(partial);
                } else {
                    for key in ["thinking", "signature", "text"] {
                        if let Some(value) = delta.get(key) {
                            merge(&mut self.message["content"][index][key], value);
                        }
                    }
                }
            }
            Some("content_block_stop") => {
                if let Some(input) = self.anthropic_inputs.remove(&index) {
                    self.message["content"][index]["input"] = serde_json::from_str(&input)
                        .map_err(|e| {
                            crate::EngineError::Provider(format!(
                                "invalid streamed tool input: {e}"
                            ))
                        })?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn snapshot(&self, scope: String, anthropic: bool) -> Option<ProviderReplay> {
        capture(scope, &self.message, anthropic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_replay_clears_cached_state_before_costing_and_overrides_client() {
        let mut message = assistant();
        message.provider_state = capture(
            "scope".into(),
            &json!({"thinking":"reasoning ".repeat(500)}),
            false,
        )
        .map(|state| state.bind(&message).for_request_group("active"));
        let mut request: crate::ChatCompletionRequest = serde_json::from_value(json!({
            "model":"test", "messages":[{"role":"user","content":"investigate"}]
        }))
        .unwrap();
        request.messages.push(message);
        let predictor = crate::CostPredictor::new();
        prepare_request_replay(&mut request, "active", true);
        let with_replay = predictor
            .predict(&request, crate::ModelTier::Pro)
            .unwrap()
            .input_tokens;
        prepare_request_replay(&mut request, "active", false);
        assert!(!enabled(&request));
        assert!(request
            .messages
            .iter()
            .all(|message| message.provider_state.is_none()));
        let without_replay = predictor
            .predict(&request, crate::ModelTier::Pro)
            .unwrap()
            .input_tokens;
        assert!(with_replay > without_replay + 100);
        prepare_request_replay(&mut request, "active", true);
        assert!(enabled(&request));
        assert_eq!(
            predictor
                .predict(&request, crate::ModelTier::Pro)
                .unwrap()
                .input_tokens,
            without_replay
        );
    }

    fn assistant() -> ChatMessage {
        serde_json::from_value(json!({
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{\"path\":\"a.rs\",\"line\":1}"
                    }
                }
            ]
        }))
        .unwrap()
    }

    #[test]
    fn replay_is_limited_to_the_active_request_group() {
        let mut previous = assistant();
        previous.provider_state = capture(
            "ollama/model".into(),
            &json!({"thinking":"previous"}),
            false,
        )
        .map(|state| state.bind(&previous).for_request_group("previous"));
        let mut current = assistant();
        current.provider_state =
            capture("ollama/model".into(), &json!({"thinking":"current"}), false)
                .map(|state| state.bind(&current).for_request_group("current"));
        let prompt: ChatMessage =
            serde_json::from_value(json!({"role":"user","content":"next task"})).unwrap();
        let scaffold: ChatMessage = serde_json::from_value(
            json!({"role":"user","content":"<environment_info>editor</environment_info>"}),
        )
        .unwrap();
        let mut legacy = current.clone();
        legacy.provider_state.as_mut().unwrap().request_group_id = None;
        // Identical historical replies must not receive current-group state.
        let mut messages = vec![
            previous.clone(),
            current.clone(),
            prompt,
            current.clone(),
            scaffold,
            previous,
            legacy,
        ];
        retain_request_group_state(&mut messages, "current");
        assert!(messages[..3]
            .iter()
            .all(|message| message.provider_state.is_none()));
        assert_eq!(messages[3].provider_state, current.provider_state);
        assert!(messages[4..]
            .iter()
            .all(|message| message.provider_state.is_none()));
        let mut wire = json!({});
        apply(&messages[3], &mut wire, "ollama/model");
        assert_eq!(wire["thinking"], "current");
        assert!(wire.get("request_group_id").is_none());
        // A group change also drops replay when client compaction removed the prompt.
        retain_request_group_state(&mut messages, "next");
        assert!(messages
            .iter()
            .all(|message| message.provider_state.is_none()));
    }

    #[test]
    fn reasoning_survives_serialization_but_not_message_or_model_changes() {
        let mut message = assistant();
        let raw =
            json!({"message":{"thinking":"exact bytes\n  α","extra_content":{"opaque":"sig"}}});
        message.provider_state =
            capture("ollama/model".into(), &raw, false).map(|s| s.bind(&message));
        let mut message: ChatMessage =
            serde_json::from_value(serde_json::to_value(message).unwrap()).unwrap();
        message.tool_calls.as_mut().unwrap()[0].function.arguments =
            "{ \"line\": 1, \"path\": \"a.rs\" }".into();
        let mut wire = json!({});
        apply(&message, &mut wire, "ollama/model");
        assert_eq!(wire["thinking"], "exact bytes\n  α");
        let mut other = json!({});
        apply(&message, &mut other, "another/model");
        assert_eq!(other, json!({}));
        message.content = Some(crate::MessageContent::text("summary"));
        apply(&message, &mut other, "ollama/model");
        assert_eq!(other, json!({}));
    }

    #[test]
    fn compatible_stream_preserves_structured_reasoning_and_tool_signatures() {
        let mut state = ReplayAccumulator::default();
        for delta in [
            json!({"reasoning_content":"first\n", "reasoning_details":[{"index":0,"type":"reasoning.encrypted","data":"opaque-"}]}),
            json!({"reasoning_content":"second", "reasoning_details":[{"index":0,"data":"bytes"}]}),
            json!({"tool_calls":[{"index":1,"extra_content":{"google":{"thought_signature":"sig-b"}}}]}),
            json!({"tool_calls":[{"index":0,"extra_content":{"google":{"thought_signature":"sig-a"}}}]}),
        ] {
            state.compatible(&json!({"choices":[{"delta":delta}]}));
        }
        let snapshot = state.snapshot("model".into(), false).unwrap();
        assert_eq!(snapshot.payload["reasoning_content"], "first\nsecond");
        assert_eq!(
            snapshot.payload["reasoning_details"][0]["data"],
            "opaque-bytes"
        );
        assert_eq!(
            snapshot.payload["tool_extras"][0]["google"]["thought_signature"],
            "sig-a"
        );
        assert_eq!(
            snapshot.payload["tool_extras"][1]["google"]["thought_signature"],
            "sig-b"
        );
    }

    #[test]
    fn anthropic_stream_preserves_block_order_signatures_and_redaction() {
        let mut state = ReplayAccumulator::default();
        for event in [
            json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"check"}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed-"}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"bytes"}}),
            json!({"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"opaque"}}),
            json!({"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"call-1","name":"read_file","input":{}}}),
            json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}),
            json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"a.rs\",\"line\":1}"}}),
            json!({"type":"content_block_stop","index":2}),
        ] {
            state.anthropic(&event).unwrap();
        }
        let snapshot = state.snapshot("claude".into(), true).unwrap();
        let mut message = assistant();
        message.provider_state = Some(snapshot.bind(&message));
        let mut wire = json!({"role":"assistant","content":[]});
        apply(&message, &mut wire, "claude");
        assert_eq!(
            wire["content"],
            json!([
                {"type":"thinking","thinking":"check","signature":"signed-bytes"},
                {"type":"redacted_thinking","data":"opaque"},
                {"type":"tool_use","id":"call-1","name":"read_file","input":{"path":"a.rs","line":1}}
            ])
        );
    }

    #[test]
    fn plain_responses_do_not_create_replay_state() {
        assert!(capture(
            "m".into(),
            &json!({"choices":[{"message":{"content":"answer"}}]}),
            false
        )
        .is_none());
        assert!(capture(
            "m".into(),
            &json!({"content":[{"type":"text","text":"answer"}]}),
            true
        )
        .is_none());
    }

    // Exercise real request builders and response parsers against a local
    // upstream, including streaming. No provider credentials/network are used.
    #[tokio::test]
    async fn adapters_round_trip_state_through_http() {
        use crate::provider::*;
        use axum::{routing::post, Json, Router};
        use futures::StreamExt;
        use std::sync::{Arc, Mutex};
        for kind in [
            "ollama",
            "deepseek",
            "gemini",
            "anthropic",
            "openai",
            "azure",
            "groq",
            "xai",
        ] {
            let recorded = Arc::new(Mutex::new(Vec::<Value>::new()));
            let seen = recorded.clone();
            let app = Router::new().fallback(post(move |Json(body): Json<Value>| {
                seen.lock().unwrap().push(body.clone());
                async move {
                    let message = json!({"role":"assistant","content":"checking","reasoning_content":"reason\nbytes",
                        "tool_calls":[{"id":"call-1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.rs\",\"line\":1}"},
                            "extra_content":{"google":{"thought_signature":"signature"}}}]});
                    let native = json!({"role":"assistant","content":"checking","thinking":"reason\nbytes","tool_calls":[
                        {"id":"call-1","function":{"name":"read_file","arguments":{"path":"a.rs","line":1}}}
                    ]});
                    let blocks = json!([
                        {"type":"thinking","thinking":"reason\nbytes","signature":"signature"},
                        {"type":"redacted_thinking","data":"opaque"},
                        {"type":"text","text":"checking"},
                        {"type":"tool_use","id":"call-1","name":"read_file","input":{"path":"a.rs","line":1}}
                    ]);
                    if body["stream"] == true {
                        let data = if kind == "ollama" {
                            format!("{}\n{}", json!({"message":native,"done":false}), json!({"done":true,"done_reason":"stop"}))
                        } else if kind == "anthropic" {
                            let mut events = Vec::new();
                            for (index, block) in blocks.as_array().unwrap().iter().enumerate() {
                                let mut initial = block.clone();
                                if block["type"] == "text" { initial["text"] = json!(""); }
                                events.push(json!({"type":"content_block_start","index":index,"content_block":initial}));
                                if block["type"] == "text" { events.push(json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":"checking"}})); }
                                if block["type"] == "tool_use" { events.push(json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":block["input"].to_string()}})); }
                                events.push(json!({"type":"content_block_stop","index":index}));
                            }
                            events.push(json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":10}}));
                            events.into_iter().map(|event| format!("data: {event}\n\n")).collect()
                        } else {
                            let mut delta = message.clone();
                            delta["tool_calls"][0]["index"] = json!(0);
                            format!("data: {}\n\ndata: {}", json!({"choices":[{"delta":delta}]}), json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}]}))
                        };
                        axum::response::IntoResponse::into_response(data)
                    } else {
                        let raw = match kind {
                            "ollama" => json!({"model":"test","message":native}),
                            "anthropic" => json!({"id":"r","model":"test","content":blocks}),
                            _ => json!({"id":"r","model":"test","choices":[{"message":message}]})
                        };
                        axum::response::IntoResponse::into_response(Json(raw))
                    }
                }
            }));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            let config = ProviderConfig {
                label: kind.into(),
                base_url: format!("http://{address}"),
                standard_model: Some("test".into()),
                ..Default::default()
            };
            let provider: Box<dyn Provider> = match kind {
                "ollama" => Box::new(OllamaProvider::new(config, OllamaMode::Local).unwrap()),
                "anthropic" => Box::new(AnthropicProvider::new(config).unwrap()),
                "deepseek" => Box::new(DeepSeekProvider::new(config).unwrap()),
                "gemini" => Box::new(GeminiProvider::new(config).unwrap()),
                "azure" => Box::new(AzureOpenAiProvider::new(config).unwrap()),
                "groq" => Box::new(GroqProvider::new(config).unwrap()),
                "xai" => Box::new(XaiProvider::new(config).unwrap()),
                _ => Box::new(OpenAiProvider::new(config).unwrap()),
            };
            let mut initial: crate::ChatCompletionRequest = serde_json::from_value(
                json!({"model":"test","messages":[{"role":"user","content":"fix it"}]}),
            )
            .unwrap();
            initial.extra.insert(ENABLED_KEY.into(), json!(true));
            for streaming in [false, true] {
                let mut response = if streaming {
                    let mut stream = provider.complete_stream(&initial).await.unwrap();
                    let mut state = None;
                    while let Some(chunk) = stream.next().await {
                        let chunk = chunk.unwrap();
                        if chunk.provider_state.is_some() {
                            state = chunk.provider_state;
                        }
                    }
                    let mut message = assistant();
                    message.provider_state = state.map(|state| state.bind(&message));
                    message
                } else {
                    provider.complete(&initial).await.unwrap().message
                };
                assert!(
                    response.provider_state.is_some(),
                    "{kind}, streaming={streaming}"
                );
                response.provider_state = response
                    .provider_state
                    .take()
                    .map(|state| state.for_request_group("active"));
                let mut next = initial.clone();
                next.messages
                    .push(serde_json::from_value(serde_json::to_value(response).unwrap()).unwrap());
                next.messages.push(
                    serde_json::from_value(
                        json!({"role":"tool","tool_call_id":"call-1","content":"file contents"}),
                    )
                    .unwrap(),
                );
                retain_request_group_state(&mut next.messages, "active");
                provider.complete(&next).await.unwrap();
                let requests = recorded.lock().unwrap();
                let replayed = &requests.last().unwrap()["messages"][1];
                match kind {
                    "ollama" => assert_eq!(replayed["thinking"], "reason\nbytes"),
                    "anthropic" => {
                        assert_eq!(replayed["content"][0]["signature"], "signature");
                        assert_eq!(replayed["content"][1]["data"], "opaque");
                    }
                    "gemini" => assert_eq!(
                        replayed["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
                        "signature"
                    ),
                    _ => assert_eq!(replayed["reasoning_content"], "reason\nbytes"),
                }
                assert!(
                    replayed.get("provider_state").is_none(),
                    "internal envelope must never leak upstream"
                );
                assert!(replayed.get("request_group_id").is_none());
                drop(requests);
                // Also exercise a disabled request carrying stale client state:
                // builders must omit it and parsers must not collect new replay.
                let mut disabled = next.clone();
                disabled.extra.insert(ENABLED_KEY.into(), json!(false));
                if streaming {
                    let mut stream = provider.complete_stream(&disabled).await.unwrap();
                    while let Some(chunk) = stream.next().await {
                        assert!(chunk.unwrap().provider_state.is_none(), "{kind}");
                    }
                } else {
                    assert!(
                        provider
                            .complete(&disabled)
                            .await
                            .unwrap()
                            .message
                            .provider_state
                            .is_none(),
                        "{kind}"
                    );
                }
                {
                    let requests = recorded.lock().unwrap();
                    let wire = requests.last().unwrap();
                    assert!(wire.get(ENABLED_KEY).is_none());
                    let history = &wire["messages"][1];
                    assert!(history.get("thinking").is_none(), "{kind}");
                    assert!(history.get("reasoning_content").is_none(), "{kind}");
                    assert!(
                        history["tool_calls"][0].get("extra_content").is_none(),
                        "{kind}"
                    );
                    if kind == "anthropic" {
                        assert!(history["content"].as_array().unwrap().iter().all(|block| {
                            !matches!(
                                block["type"].as_str(),
                                Some("thinking" | "redacted_thinking")
                            )
                        }));
                    }
                }
                retain_request_group_state(&mut next.messages, "new-group");
                provider.complete(&next).await.unwrap();
                let requests = recorded.lock().unwrap();
                let historical = &requests.last().unwrap()["messages"][1];
                assert!(historical.get("thinking").is_none());
                assert!(historical.get("reasoning_content").is_none());
                assert!(historical["tool_calls"][0].get("extra_content").is_none());
                if kind == "anthropic" {
                    assert!(historical["content"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|block| {
                            !matches!(
                                block["type"].as_str(),
                                Some("thinking" | "redacted_thinking")
                            )
                        }));
                }
            }
            server.abort();
        }
    }
}
