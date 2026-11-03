use super::chat::{
    chat_completions, chat_completions_single_provider, extract_json_response,
    with_provider_rate_limit_headers,
};
use super::*;

/// Request body for the OpenAI Responses API (`POST /v1/responses`).
///
/// This is a subset of the full Responses API spec - enough to handle what
/// agentic coding clients (Codex, etc.) actually send. The `input` field can
/// be a simple string or an array of content items; `instructions` maps to
/// the system message.
#[derive(Deserialize, Debug)]
pub struct ResponsesApiRequest {
    pub model: String,
    pub instructions: Option<String>,
    pub input: serde_json::Value,
    #[serde(default)]
    pub stream: Option<bool>,
    pub temperature: Option<f64>,
    pub max_output_tokens: Option<u64>,
    #[allow(dead_code)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[allow(dead_code)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

pub(super) fn api_tool_definitions(
    tools: Option<&[serde_json::Value]>,
) -> Option<Vec<engine_core::ToolDefinition>> {
    let definitions = tools?
        .iter()
        .flat_map(|tool| {
            // Muse Code groups its tools under `{type: "namespace", tools:
            // [...]}`. Expand namespaces so the nested function definitions
            // reach the provider instead of a bogus single tool.
            if tool.get("type").and_then(|value| value.as_str()) == Some("namespace") {
                return tool
                    .get("tools")
                    .and_then(|value| value.as_array())
                    .map(|nested| api_tool_definitions(Some(nested)).unwrap_or_default())
                    .unwrap_or_default();
            }
            tool_definition_from_value(tool)
                .into_iter()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    (!definitions.is_empty()).then_some(definitions)
}

/// Build one tool definition from a Responses function-tool value.
fn tool_definition_from_value(tool: &serde_json::Value) -> Option<engine_core::ToolDefinition> {
    let function = tool.get("function").unwrap_or(tool);
    let name = function.get("name")?.as_str()?.to_string();
    // Memoization is server granted: a caller flag can opt out, never
    // opt in. When the caller says nothing, fall back to ReinCore's own
    // read-only classification, so standard clients - which never send
    // this non-standard field - still get tool reuse.
    let safe_to_memoize = function
        .get("safe_to_memoize")
        .and_then(|value| value.as_bool())
        .unwrap_or_else(|| engine_core::is_exploration_tool(&name));
    Some(engine_core::ToolDefinition {
        action_contract: function
            .get("action_contract")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok()),
        name,
        description: function
            .get("description")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string(),
        parameters: function
            .get("parameters")
            .or_else(|| function.get("input_schema"))
            .cloned()
            .unwrap_or_else(|| json!({"type": "object"})),
        safe_to_memoize,
        memoization_policy: function
            .get("memoization_policy")
            .and_then(|value| value.as_str())
            .map(str::to_string),
    })
}

fn normalize_api_tool_choice(choice: Option<&serde_json::Value>) -> Option<serde_json::Value> {
    let choice = choice?;
    let Some(kind) = choice.get("type").and_then(|value| value.as_str()) else {
        return Some(choice.clone());
    };
    Some(match kind {
        "auto" => json!("auto"),
        "any" => json!("required"),
        "none" => json!("none"),
        "tool" => json!({
            "type": "function",
            "function": {"name": choice.get("name").and_then(|value| value.as_str()).unwrap_or("")}
        }),
        _ => choice.clone(),
    })
}

fn extra_with_delegation_profile(
    extra: &std::collections::HashMap<String, serde_json::Value>,
    profile: &str,
) -> std::collections::HashMap<String, serde_json::Value> {
    let mut extra = extra.clone();
    extra
        .entry("reincore_delegation_profile".to_string())
        .or_insert_with(|| json!(profile));
    extra
}

/// Convert a Responses API request into the internal `ChatCompletionRequest`
/// that the existing provider pipeline understands.
pub(super) fn responses_to_chat_request(req: &ResponsesApiRequest) -> ChatCompletionRequest {
    let mut messages = Vec::new();

    // `instructions` → system message.
    if let Some(instructions) = &req.instructions {
        if !instructions.is_empty() {
            messages.push(engine_core::ChatMessage {
                provider_state: None,
                role: "system".to_string(),
                content: Some(engine_core::MessageContent::text(instructions.clone())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
    }

    // `input` can be a string or an array of input items.
    match &req.input {
        serde_json::Value::String(s) => {
            messages.push(engine_core::ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(engine_core::MessageContent::text(s.clone())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
        serde_json::Value::Array(items) => {
            for item in items {
                if let Some(obj) = item.as_object() {
                    match obj.get("type").and_then(|value| value.as_str()) {
                        Some("function_call") => {
                            messages.push(engine_core::ChatMessage {
                                provider_state: None,
                                role: "assistant".to_string(),
                                content: None,
                                tool_calls: Some(vec![engine_core::ToolCall {
                                    id: obj
                                        .get("call_id")
                                        .or_else(|| obj.get("id"))
                                        .and_then(|value| value.as_str())
                                        .unwrap_or("call_reincore")
                                        .to_string(),
                                    call_type: "function".to_string(),
                                    function: engine_core::FunctionCall {
                                        name: obj
                                            .get("name")
                                            .and_then(|value| value.as_str())
                                            .unwrap_or("")
                                            .to_string(),
                                        arguments: obj
                                            .get("arguments")
                                            .and_then(|value| value.as_str())
                                            .unwrap_or("{}")
                                            .to_string(),
                                    },
                                }]),
                                tool_call_id: None,
                                name: None,
                            });
                            continue;
                        }
                        Some("function_call_output") => {
                            messages.push(engine_core::ChatMessage {
                                provider_state: None,
                                role: "tool".to_string(),
                                content: obj.get("output").map(|value| {
                                    engine_core::MessageContent::text(
                                        value
                                            .as_str()
                                            .map(str::to_string)
                                            .unwrap_or_else(|| value.to_string()),
                                    )
                                }),
                                tool_calls: None,
                                tool_call_id: obj
                                    .get("call_id")
                                    .and_then(|value| value.as_str())
                                    .map(str::to_string),
                                name: None,
                            });
                            continue;
                        }
                        _ => {}
                    }
                    let role = obj
                        .get("role")
                        .and_then(|v| v.as_str())
                        .unwrap_or("user")
                        .to_string();
                    let content = obj
                        .get("content")
                        .map(|c| match c {
                            serde_json::Value::String(s) => s.clone(),
                            serde_json::Value::Array(parts) => {
                                // Extract text from content parts.
                                parts
                                    .iter()
                                    .filter_map(|p| {
                                        p.get("text")
                                            .and_then(|t| t.as_str())
                                            .or_else(|| p.as_str())
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            }
                            _ => c.to_string(),
                        })
                        .or_else(|| {
                            obj.get("text")
                                .and_then(|t| t.as_str())
                                .map(|s| s.to_string())
                        })
                        .map(engine_core::MessageContent::text);

                    messages.push(engine_core::ChatMessage {
                        provider_state: None,
                        role,
                        content,
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    });
                } else if let Some(s) = item.as_str() {
                    messages.push(engine_core::ChatMessage {
                        provider_state: None,
                        role: "user".to_string(),
                        content: Some(engine_core::MessageContent::text(s.to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    });
                }
            }
        }
        _ => {
            messages.push(engine_core::ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(engine_core::MessageContent::text(req.input.to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
    }

    ChatCompletionRequest {
        model: req.model.clone(),
        messages,
        tools: api_tool_definitions(req.tools.as_deref()),
        tool_choice: normalize_api_tool_choice(req.tool_choice.as_ref()),
        temperature: req.temperature,
        max_tokens: req
            .max_output_tokens
            .map(|tokens| u32::try_from(tokens).unwrap_or(u32::MAX)),
        stream: req.stream,
        extra: extra_with_delegation_profile(&req.extra, "codex"),
    }
}

/// Normalize a chat-style usage object to the Responses API shape.
///
/// The chat pipeline reports `prompt_tokens`/`completion_tokens`; Responses
/// clients (Muse Code included) read `input_tokens`/`output_tokens`. Keep
/// both namings so strict clients find the fields they require.
fn normalize_responses_usage(usage: &serde_json::Value) -> serde_json::Value {
    let mut normalized = usage.clone();
    if normalized.get("input_tokens").is_none() {
        if let Some(tokens) = usage.get("prompt_tokens").and_then(|value| value.as_u64()) {
            normalized["input_tokens"] = json!(tokens);
        }
    }
    if normalized.get("output_tokens").is_none() {
        if let Some(tokens) = usage
            .get("completion_tokens")
            .and_then(|value| value.as_u64())
        {
            normalized["output_tokens"] = json!(tokens);
        }
    }
    if normalized.get("input_tokens_details").is_none() {
        let cached = usage
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        normalized["input_tokens_details"] = json!({"cached_tokens": cached});
    }
    if normalized.get("output_tokens_details").is_none() {
        let reasoning = usage
            .get("reasoning_tokens")
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        normalized["output_tokens_details"] = json!({"reasoning_tokens": reasoning});
    }
    normalized
}

/// Convert a chat completion response into the Responses API output format.
pub(super) fn chat_response_to_responses(
    chat_response: serde_json::Value,
    model: &str,
) -> serde_json::Value {
    // Extract the assistant message from the chat completion response.
    let message = chat_response
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .cloned()
        .unwrap_or(json!({}));

    let content = message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    let usage = chat_response
        .get("usage")
        .map(normalize_responses_usage)
        .unwrap_or(json!({}));

    // Responses clients (Muse Code included) correlate stream deltas to
    // output items by id, so every item gets a stable id even when the
    // chat pipeline produced none.
    let mut output = vec![json!({
        "type": "message",
        "id": format!("msg_{}", Uuid::new_v4()),
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": content, "annotations": []}],
    })];
    if let Some(calls) = message.get("tool_calls").and_then(|value| value.as_array()) {
        for call in calls {
            let function = call.get("function").unwrap_or(&serde_json::Value::Null);
            let call_id = call
                .get("id")
                .and_then(|value| value.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| format!("fc_{}", Uuid::new_v4()));
            output.push(json!({
                "type": "function_call",
                "id": call_id,
                "call_id": call_id,
                "name": function.get("name"),
                "arguments": function.get("arguments").and_then(|value| value.as_str()).unwrap_or("{}"),
                "status": "completed",
            }));
        }
    }

    // Emit only the standard Responses fields. ReinCore diagnostics stay in
    // the chat pipeline payload; strict clients reject unknown top-level
    // fields as malformed responses.
    json!({
        "id": chat_response.get("id").cloned().unwrap_or(json!(Uuid::new_v4().to_string())),
        "object": "response",
        "created_at": chat_response.get("created").cloned().unwrap_or(json!(chrono::Utc::now().timestamp())),
        "model": model,
        "output": output,
        "status": "completed",
        "usage": usage,
    })
}

/// Stamp a monotonically increasing `sequence_number` onto a Responses SSE
/// frame. Muse Code's Meta provider requires it and rejects streams without
/// one as malformed; other Responses clients ignore the extra field.
fn stamp_sequence_number(
    mut payload: serde_json::Value,
    sequence_number: &mut u64,
) -> serde_json::Value {
    payload["sequence_number"] = json!(*sequence_number);
    *sequence_number += 1;
    payload
}

pub(super) fn responses_json_to_sse_stream(
    response: serde_json::Value,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    async_stream::stream! {
        let mut sequence_number: u64 = 0;
        let created = stamp_sequence_number(
            json!({"type": "response.created", "response": response}),
            &mut sequence_number,
        );
        yield Ok(Event::default().event("response.created").data(created.to_string()));
        let in_progress = stamp_sequence_number(
            json!({"type": "response.in_progress", "response": response}),
            &mut sequence_number,
        );
        yield Ok(Event::default().event("response.in_progress").data(in_progress.to_string()));
        if let Some(items) = response.get("output").and_then(|value| value.as_array()) {
            for (output_index, item) in items.iter().enumerate() {
                let item_id = item.get("id").or_else(|| response.get("id")).cloned();
                let added = stamp_sequence_number(
                    json!({
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": item
                    }),
                    &mut sequence_number,
                );
                yield Ok(Event::default().event("response.output_item.added").data(added.to_string()));
                if let Some(text) = item.pointer("/content/0/text").and_then(|value| value.as_str()) {
                    if !text.is_empty() {
                        let part_added = stamp_sequence_number(
                            json!({
                                "type": "response.content_part.added",
                                "item_id": item_id,
                                "output_index": output_index,
                                "content_index": 0,
                                "part": {"type": "output_text", "text": "", "annotations": []},
                            }),
                            &mut sequence_number,
                        );
                        yield Ok(Event::default().event("response.content_part.added").data(part_added.to_string()));
                        let delta = stamp_sequence_number(
                            json!({
                                "type": "response.output_text.delta",
                                "item_id": item_id,
                                "output_index": output_index,
                                "content_index": 0,
                                "delta": text,
                            }),
                            &mut sequence_number,
                        );
                        yield Ok(Event::default().event("response.output_text.delta").data(delta.to_string()));
                        let text_done = stamp_sequence_number(
                            json!({
                                "type": "response.output_text.done",
                                "item_id": item_id,
                                "output_index": output_index,
                                "content_index": 0,
                                "text": text,
                            }),
                            &mut sequence_number,
                        );
                        yield Ok(Event::default().event("response.output_text.done").data(text_done.to_string()));
                        let part_done = stamp_sequence_number(
                            json!({
                                "type": "response.content_part.done",
                                "item_id": item_id,
                                "output_index": output_index,
                                "content_index": 0,
                                "part": {"type": "output_text", "text": text, "annotations": []},
                            }),
                            &mut sequence_number,
                        );
                        yield Ok(Event::default().event("response.content_part.done").data(part_done.to_string()));
                    }
                }
                let item_done = stamp_sequence_number(
                    json!({
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "item": item
                    }),
                    &mut sequence_number,
                );
                yield Ok(Event::default().event("response.output_item.done").data(item_done.to_string()));
            }
        }
        let completed = stamp_sequence_number(
            json!({"type": "response.completed", "response": response}),
            &mut sequence_number,
        );
        yield Ok(Event::default().event("response.completed").data(completed.to_string()));
        yield Ok(Event::default().data("[DONE]"));
    }
}

/// `POST /v1/responses` - Responses API endpoint for agentic clients.
///
/// Translates the Responses API request into the internal ChatCompletionRequest,
/// runs it through the same provider pipeline (classification, cost prediction,
/// budget tracking), and converts the response back to the Responses API format.
pub async fn responses_api(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ResponsesApiRequest>,
) -> Result<axum::response::Response, StatusCode> {
    let wants_stream = req.stream.unwrap_or(false);
    let mut chat_req = responses_to_chat_request(&req);
    chat_req.stream = Some(false);
    let model = chat_req.model.clone();

    // Delegate to the existing chat completions handler logic.
    let chat_response = chat_completions(State(state), headers, Json(chat_req)).await?;
    let (chat_json, provider_response_headers) = match extract_json_response(chat_response).await {
        Ok(response) => response,
        Err(refusal) => return Ok(refusal),
    };

    let response = chat_response_to_responses(chat_json, &model);
    let response = if wants_stream {
        Sse::new(responses_json_to_sse_stream(response))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(response).into_response()
    };
    Ok(with_provider_rate_limit_headers(
        response,
        &provider_response_headers,
    ))
}

/// `POST /v1/responses` - single-provider variant.
pub async fn responses_api_single_provider(
    State(state): State<SingleProviderAppState>,
    headers: HeaderMap,
    Json(req): Json<ResponsesApiRequest>,
) -> Result<axum::response::Response, StatusCode> {
    let wants_stream = req.stream.unwrap_or(false);
    let mut chat_req = responses_to_chat_request(&req);
    chat_req.stream = Some(false);
    let model = chat_req.model.clone();

    let chat_response =
        chat_completions_single_provider(State(state), headers, Json(chat_req)).await?;
    let (chat_json, provider_response_headers) = match extract_json_response(chat_response).await {
        Ok(response) => response,
        Err(refusal) => return Ok(refusal),
    };

    let response = chat_response_to_responses(chat_json, &model);
    let response = if wants_stream {
        Sse::new(responses_json_to_sse_stream(response))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(response).into_response()
    };
    Ok(with_provider_rate_limit_headers(
        response,
        &provider_response_headers,
    ))
}

/// Request body for the Messages API (`POST /v1/messages`).
///
/// Clients that use the Anthropic Messages format (e.g. Claude Code with
/// `ANTHROPIC_BASE_URL` set) send this shape. We translate it to the
/// internal `ChatCompletionRequest`, run it through the same provider
/// pipeline, and convert the response back to the Messages format.
#[derive(Deserialize, Debug)]
pub struct MessagesApiRequest {
    pub model: String,
    /// The `system` field can be a string or an array of content blocks.
    /// We flatten it to a single string for the internal representation.
    pub system: Option<serde_json::Value>,
    pub messages: Vec<MessagesApiMessage>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    #[allow(dead_code)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    #[allow(dead_code)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Deserialize, Debug)]
pub struct MessagesApiMessage {
    pub role: String,
    /// Content can be a simple string or an array of content blocks.
    pub content: serde_json::Value,
}

/// Extract a plain-text string from a Messages API content field, which
/// can be either a string or an array of content blocks like
/// `{"type": "text", "text": "..."}`.
fn messages_content_to_string(content: &serde_json::Value) -> String {
    if let Some(s) = content.as_str() {
        return s.to_string();
    }
    if let Some(arr) = content.as_array() {
        let mut parts = Vec::new();
        for block in arr {
            if let Some(block_type) = block.get("type").and_then(|t| t.as_str()) {
                if block_type == "text" {
                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                        parts.push(text.to_string());
                    }
                } else if block_type == "tool_use" {
                    // Represent tool_use blocks as a JSON marker so the
                    // downstream model can see them. The proxy's routing
                    // classifier doesn't need the full structure.
                    if let Some(name) = block.get("name").and_then(|n| n.as_str()) {
                        parts.push(format!("[tool_use: {}]", name));
                    }
                } else if block_type == "tool_result" {
                    if let Some(content) = block.get("content") {
                        parts.push(messages_content_to_string(content));
                    }
                }
            }
        }
        return parts.join("\n");
    }
    String::new()
}

/// Convert a Messages API request to the internal ChatCompletionRequest.
pub(super) fn messages_to_chat_request(req: &MessagesApiRequest) -> ChatCompletionRequest {
    let mut messages = Vec::new();

    // The Messages API puts the system prompt in a separate top-level
    // `system` field, not as a message. Convert it to a system message.
    if let Some(system) = &req.system {
        let system_text = messages_content_to_string(system);
        if !system_text.is_empty() {
            messages.push(ChatMessage {
                provider_state: None,
                role: "system".to_string(),
                content: Some(engine_core::MessageContent::text(system_text)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
    }

    // Convert Messages API messages to OpenAI-format messages while
    // preserving structured tool-use and tool-result blocks.
    for msg in &req.messages {
        let Some(blocks) = msg.content.as_array() else {
            messages.push(ChatMessage {
                provider_state: None,
                role: msg.role.clone(),
                content: msg
                    .content
                    .as_str()
                    .map(|s| engine_core::MessageContent::text(s.to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
            continue;
        };
        let mut text = Vec::new();
        let mut calls = Vec::new();
        let mut results = Vec::new();
        for block in blocks {
            match block.get("type").and_then(|value| value.as_str()) {
                Some("text") => {
                    if let Some(value) = block.get("text").and_then(|value| value.as_str()) {
                        text.push(value.to_string());
                    }
                }
                Some("tool_use") => calls.push(engine_core::ToolCall {
                    id: block
                        .get("id")
                        .and_then(|value| value.as_str())
                        .unwrap_or("call_reincore")
                        .to_string(),
                    call_type: "function".to_string(),
                    function: engine_core::FunctionCall {
                        name: block
                            .get("name")
                            .and_then(|value| value.as_str())
                            .unwrap_or("")
                            .to_string(),
                        arguments: block
                            .get("input")
                            .map(|value| value.to_string())
                            .unwrap_or_else(|| "{}".to_string()),
                    },
                }),
                Some("tool_result") => results.push(ChatMessage {
                    provider_state: None,
                    role: "tool".to_string(),
                    content: block
                        .get("content")
                        .map(|c| engine_core::MessageContent::text(messages_content_to_string(c))),
                    tool_calls: None,
                    tool_call_id: block
                        .get("tool_use_id")
                        .and_then(|value| value.as_str())
                        .map(str::to_string),
                    name: None,
                }),
                _ => {}
            }
        }
        if !text.is_empty() || !calls.is_empty() || results.is_empty() {
            messages.push(ChatMessage {
                provider_state: None,
                role: msg.role.clone(),
                content: (!text.is_empty())
                    .then(|| engine_core::MessageContent::text(text.join("\n"))),
                tool_calls: (!calls.is_empty()).then_some(calls),
                tool_call_id: None,
                name: None,
            });
        }
        messages.extend(results);
    }

    ChatCompletionRequest {
        model: req.model.clone(),
        messages,
        tools: api_tool_definitions(req.tools.as_deref()),
        tool_choice: normalize_api_tool_choice(req.tool_choice.as_ref()),
        temperature: req.temperature,
        max_tokens: req.max_tokens,
        stream: req.stream,
        extra: extra_with_delegation_profile(&req.extra, "claude"),
    }
}

/// Convert a chat completion response (OpenAI format) back to the Messages
/// API response format.
fn chat_response_to_messages_api(
    chat_response: serde_json::Value,
    model: &str,
) -> serde_json::Value {
    let id = chat_response
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("msg_reincore")
        .to_string();

    // Extract the assistant message content from the OpenAI response.
    let choice = chat_response
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first());
    let message = choice.and_then(|c| c.get("message"));
    let content_text = message
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    // Extract tool calls if present.
    let tool_calls = message.and_then(|m| m.get("tool_calls"));
    let mut content_blocks = vec![json!({
        "type": "text",
        "text": content_text,
    })];

    if let Some(tc_arr) = tool_calls.and_then(|t| t.as_array()) {
        for tc in tc_arr {
            if let Some(function) = tc.get("function") {
                let name = function
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("unknown");
                let arguments = function
                    .get("arguments")
                    .and_then(|a| a.as_str())
                    .unwrap_or("{}");
                let tc_id = tc
                    .get("id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("tool_call_reincore");
                content_blocks.push(json!({
                    "type": "tool_use",
                    "id": tc_id,
                    "name": name,
                    "input": serde_json::from_str::<serde_json::Value>(arguments)
                        .unwrap_or(json!({})),
                }));
            }
        }
    }

    let finish_reason = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|f| f.as_str())
        .unwrap_or("stop");

    let stop_reason = match finish_reason {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        _ => "end_turn",
    };

    let usage = chat_response.get("usage").cloned().unwrap_or(json!({}));
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(|t| t.as_u64())
        .unwrap_or(0);
    let output_tokens = usage
        .get("completion_tokens")
        .and_then(|t| t.as_u64())
        .unwrap_or(0);
    // Anthropic reports cache tokens in dedicated fields and defines
    // `input_tokens` as the uncached remainder only, so on the wire
    // `total = input_tokens + cache_creation_input_tokens + cache_read_input_tokens`.
    // ReinCore's internal `prompt_tokens` is already that full total, so the
    // cache categories are subtracted back out instead of being double counted.
    let cache_read_input_tokens = chat_response
        .get("reincore_cached_prompt_tokens")
        .and_then(|t| t.as_u64())
        .or_else(|| {
            usage
                .pointer("/prompt_tokens_details/cached_tokens")
                .and_then(|t| t.as_u64())
        })
        .unwrap_or(0);
    let cache_creation_input_tokens = chat_response
        .get("reincore_cache_creation_input_tokens")
        .and_then(|t| t.as_u64())
        .unwrap_or(0);
    let input_tokens = prompt_tokens
        .saturating_sub(cache_read_input_tokens)
        .saturating_sub(cache_creation_input_tokens);

    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
        },
        "reincore_request_group_id": chat_response.get("reincore_request_group_id").cloned(),
        "reincore_task_id": chat_response.get("reincore_task_id").cloned(),
        "reincore_task_name": chat_response.get("reincore_task_name").cloned(),
        "reincore_task_type": chat_response.get("reincore_task_type").cloned(),
        "reincore_task_is_new": chat_response.get("reincore_task_is_new").cloned(),
        "reincore_task_context_messages_removed": chat_response.get("reincore_task_context_messages_removed").cloned(),
        "reincore_request_group_step_count": chat_response.get("reincore_request_group_step_count").cloned(),
        "reincore_predicted_cost_high_cents": chat_response.get("reincore_predicted_cost_high_cents").cloned(),
        "reincore_cost": chat_response.get("reincore_cost").cloned(),
        "reincore_request_group_unknown_cost_calls": chat_response.get("reincore_request_group_unknown_cost_calls").cloned(),
        "reincore_request_group_total_actual_cost_nano_usd": chat_response.get("reincore_request_group_total_actual_cost_nano_usd").cloned(),
        "reincore_actual_to_predicted_high_ratio_bps": chat_response.get("reincore_actual_to_predicted_high_ratio_bps").cloned(),
        "reincore_cache_hit_rate_bps": chat_response.get("reincore_cache_hit_rate_bps").cloned(),
        "reincore_cache_hit": chat_response.get("reincore_cache_hit").cloned(),
        "reincore_semantic_cache_provenance": chat_response.get("reincore_semantic_cache_provenance").cloned(),
        "reincore_semantic_retrieval_candidates": chat_response.get("reincore_semantic_retrieval_candidates").cloned(),
        "reincore_request_group_total_actual_cost_cents": chat_response
            .get("reincore_request_group_total_actual_cost_cents")
            .cloned(),
        "reincore_request_group_total_tokens": chat_response
            .get("reincore_request_group_total_tokens")
            .cloned(),
    })
}

pub(super) fn messages_json_to_sse_stream(
    message: serde_json::Value,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    async_stream::stream! {
        let start = json!({
            "type": "message_start",
            "message": {
                "id": message.get("id"),
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": message.get("model"),
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": message.pointer("/usage/input_tokens").and_then(|value| value.as_u64()).unwrap_or(0)},
                "reincore_cache_hit": message.get("reincore_cache_hit"),
                "reincore_semantic_cache_provenance": message.get("reincore_semantic_cache_provenance"),
                "reincore_semantic_retrieval_candidates": message.get("reincore_semantic_retrieval_candidates")
            }
        });
        yield Ok(Event::default().event("message_start").data(start.to_string()));
        if let Some(blocks) = message.get("content").and_then(|value| value.as_array()) {
            for (index, block) in blocks.iter().enumerate() {
                if block.get("type").and_then(|value| value.as_str()) == Some("tool_use") {
                    yield Ok(Event::default().event("content_block_start").data(json!({
                        "type": "content_block_start", "index": index, "content_block": block
                    }).to_string()));
                } else {
                    yield Ok(Event::default().event("content_block_start").data(json!({
                        "type": "content_block_start", "index": index,
                        "content_block": {"type": "text", "text": ""}
                    }).to_string()));
                    let text = block.get("text").and_then(|value| value.as_str()).unwrap_or("");
                    if !text.is_empty() {
                        yield Ok(Event::default().event("content_block_delta").data(json!({
                            "type": "content_block_delta", "index": index,
                            "delta": {"type": "text_delta", "text": text}
                        }).to_string()));
                    }
                }
                yield Ok(Event::default().event("content_block_stop").data(json!({
                    "type": "content_block_stop", "index": index
                }).to_string()));
            }
        }
        yield Ok(Event::default().event("message_delta").data(json!({
            "type": "message_delta",
            "delta": {"stop_reason": message.get("stop_reason").and_then(|value| value.as_str()).unwrap_or("end_turn"), "stop_sequence": null},
            "usage": {"output_tokens": message.pointer("/usage/output_tokens").and_then(|value| value.as_u64()).unwrap_or(0)}
        }).to_string()));
        yield Ok(Event::default().event("message_stop").data(json!({"type": "message_stop"}).to_string()));
    }
}

/// `POST /v1/messages` - Messages API endpoint (multi-provider).
///
/// Clients that use the Anthropic Messages format (e.g. Claude Code with
/// `ANTHROPIC_BASE_URL` set) send this format. Translates to the internal
/// ChatCompletionRequest, runs it through the same provider pipeline
/// (classification, cost prediction, budget tracking), and converts the
/// response back to the Messages API format.
pub async fn messages_api(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<MessagesApiRequest>,
) -> Result<axum::response::Response, StatusCode> {
    let wants_stream = req.stream.unwrap_or(false);
    let mut chat_req = messages_to_chat_request(&req);
    chat_req.stream = Some(false);
    let model = chat_req.model.clone();

    let chat_response = chat_completions(State(state), headers, Json(chat_req)).await?;
    let (chat_json, provider_response_headers) = match extract_json_response(chat_response).await {
        Ok(response) => response,
        Err(refusal) => return Ok(refusal),
    };

    let response = chat_response_to_messages_api(chat_json, &model);
    let response = if wants_stream {
        Sse::new(messages_json_to_sse_stream(response))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(response).into_response()
    };
    Ok(with_provider_rate_limit_headers(
        response,
        &provider_response_headers,
    ))
}

/// `POST /v1/messages` - Messages API endpoint (single-provider variant).
pub async fn messages_api_single_provider(
    State(state): State<SingleProviderAppState>,
    headers: HeaderMap,
    Json(req): Json<MessagesApiRequest>,
) -> Result<axum::response::Response, StatusCode> {
    let wants_stream = req.stream.unwrap_or(false);
    let mut chat_req = messages_to_chat_request(&req);
    chat_req.stream = Some(false);
    let model = chat_req.model.clone();

    let chat_response =
        chat_completions_single_provider(State(state), headers, Json(chat_req)).await?;
    let (chat_json, provider_response_headers) = match extract_json_response(chat_response).await {
        Ok(response) => response,
        Err(refusal) => return Ok(refusal),
    };

    let response = chat_response_to_messages_api(chat_json, &model);
    let response = if wants_stream {
        Sse::new(messages_json_to_sse_stream(response))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(response).into_response()
    };
    Ok(with_provider_rate_limit_headers(
        response,
        &provider_response_headers,
    ))
}

/// `POST /v1/messages/count_tokens` - Messages API token counting endpoint.
///
/// Returns an estimate of token count. Clients use this optionally;
/// when absent they estimate locally.
pub async fn messages_count_tokens(
    Json(req): Json<MessagesApiRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let chat_req = messages_to_chat_request(&req);
    let combined: String = chat_req
        .messages
        .iter()
        .filter_map(|m| m.content_str())
        .collect::<Vec<_>>()
        .join("\n");
    let estimated_tokens = engine_core::count_text_tokens(&combined);

    Ok(Json(json!({
        "input_tokens": estimated_tokens,
    })))
}

/// `POST /v1/messages/count_tokens` - single-provider variant.
pub async fn messages_count_tokens_single_provider(
    Json(req): Json<MessagesApiRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    messages_count_tokens(Json(req)).await
}

// Gemini native API translation (for Gemini CLI / google-generative-ai SDK)
//
// The Gemini CLI speaks the *native* Google Gemini API, not the
// OpenAI-compatible shim. Its endpoints are:
//
//   POST /v1beta/models/{model}:generateContent
//   POST /v1beta/models/{model}:streamGenerateContent
//   GET  /v1beta/models
//   POST /v1beta/models/{model}:countTokens
//
// We translate each to the internal `ChatCompletionRequest`, run it through
// the same provider pipeline as the OpenAI/Messages/Responses paths, and
// convert the response back to the Gemini `GenerateContentResponse` shape.
// This mirrors how the Copilot integration reuses the core pipeline.

/// Request body for `:generateContent` (the `contents` array + config).
#[derive(Deserialize, Debug)]
pub struct GeminiGenerateContentRequest {
    /// The conversation turns. Each entry has a `role` ("user" or "model")
    /// and a `parts` array of content parts (text, functionCall, etc.).
    #[serde(default)]
    pub contents: Vec<GeminiContent>,
    /// Optional system instruction; the SDK sends `systemInstruction` (camel
    /// case) - the proxy normalizes both camel- and snake-case.
    #[serde(default)]
    pub system_instruction: Option<GeminiContent>,
    #[serde(default, alias = "systemInstruction")]
    pub system_instruction_camel: Option<GeminiContent>,
    /// Generation config (temperature, maxOutputTokens, etc.).
    #[serde(default)]
    pub generation_config: Option<serde_json::Value>,
    #[serde(default, alias = "generationConfig")]
    pub generation_config_camel: Option<serde_json::Value>,
    /// Tool declarations (Google's function-calling schema).
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// A single `contents` entry: a role + an array of parts.
#[derive(Deserialize, Debug)]
pub struct GeminiContent {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub parts: Vec<serde_json::Value>,
}

/// Extract the plain-text payload from a Gemini `parts` array. The native
/// API uses `{"text": "..."}` blocks for text and `{"functionCall": {...}}`
/// / `{"functionResponse": {...}}` blocks for tool I/O. We surface text and
/// render tool I/O as inline markers so the routing classifier sees them.
fn gemini_parts_to_text(parts: &[serde_json::Value]) -> String {
    let mut out = Vec::new();
    for part in parts {
        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
            out.push(text.to_string());
        } else if let Some(fc) = part.get("functionCall") {
            let name = fc.get("name").and_then(|n| n.as_str()).unwrap_or("");
            out.push(format!("[functionCall: {}]", name));
        } else if let Some(fr) = part.get("functionResponse") {
            let name = fr.get("name").and_then(|n| n.as_str()).unwrap_or("");
            out.push(format!("[functionResponse: {}]", name));
        } else if let Some(_inline) = part.get("inlineData") {
            out.push("[inlineData]".to_string());
        }
    }
    out.join("\n")
}

/// Map a Gemini `role` to the OpenAI role used internally. "model" →
/// "assistant"; anything else passes through (Gemini only emits "user" and
/// "model" for `contents`).
fn gemini_role_to_openai(role: &str) -> &str {
    match role {
        "model" => "assistant",
        _ => "user",
    }
}

/// Convert a native Gemini `generateContent` request into the internal
/// `ChatCompletionRequest`.
pub(super) fn gemini_to_chat_request(
    req: &GeminiGenerateContentRequest,
    model: &str,
    stream: bool,
) -> ChatCompletionRequest {
    let mut messages = Vec::new();
    let mut pending_tool_calls =
        std::collections::HashMap::<String, std::collections::VecDeque<String>>::new();

    // System instruction → system message.
    let sys_instr = req
        .system_instruction
        .as_ref()
        .or(req.system_instruction_camel.as_ref());
    if let Some(sys) = sys_instr {
        let text = gemini_parts_to_text(&sys.parts);
        if !text.is_empty() {
            messages.push(ChatMessage {
                provider_state: None,
                role: "system".to_string(),
                content: Some(engine_core::MessageContent::text(text)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
    }

    for (content_index, content) in req.contents.iter().enumerate() {
        let role = content
            .role
            .as_deref()
            .map(gemini_role_to_openai)
            .unwrap_or("user");
        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();
        let mut tool_results = Vec::new();

        for (part_index, part) in content.parts.iter().enumerate() {
            if let Some(text) = part.get("text").and_then(|value| value.as_str()) {
                text_parts.push(text.to_string());
                continue;
            }
            if let Some(call) = part
                .get("functionCall")
                .or_else(|| part.get("function_call"))
            {
                let Some(name) = call
                    .get("name")
                    .and_then(|value| value.as_str())
                    .filter(|name| !name.is_empty())
                else {
                    continue;
                };
                let id = call
                    .get("id")
                    .and_then(|value| value.as_str())
                    .filter(|id| !id.is_empty())
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("gemini-call-{content_index}-{part_index}"));
                let arguments = call
                    .get("args")
                    .or_else(|| call.get("arguments"))
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                pending_tool_calls
                    .entry(name.to_string())
                    .or_default()
                    .push_back(id.clone());
                tool_calls.push(engine_core::ToolCall {
                    id,
                    call_type: "function".to_string(),
                    function: engine_core::FunctionCall {
                        name: name.to_string(),
                        arguments: serde_json::to_string(&arguments)
                            .unwrap_or_else(|_| "{}".to_string()),
                    },
                });
                continue;
            }
            if let Some(result) = part
                .get("functionResponse")
                .or_else(|| part.get("function_response"))
            {
                let Some(name) = result
                    .get("name")
                    .and_then(|value| value.as_str())
                    .filter(|name| !name.is_empty())
                else {
                    continue;
                };
                let id = result
                    .get("id")
                    .and_then(|value| value.as_str())
                    .filter(|id| !id.is_empty())
                    .map(str::to_string)
                    .or_else(|| {
                        pending_tool_calls
                            .get_mut(name)
                            .and_then(std::collections::VecDeque::pop_front)
                    })
                    .unwrap_or_else(|| format!("gemini-result-{content_index}-{part_index}"));
                let response = result
                    .get("response")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let response = response
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| response.to_string());
                tool_results.push(ChatMessage {
                    provider_state: None,
                    role: "tool".to_string(),
                    content: Some(engine_core::MessageContent::text(response)),
                    tool_calls: None,
                    tool_call_id: Some(id),
                    name: Some(name.to_string()),
                });
                continue;
            }
            if part.get("inlineData").is_some() || part.get("inline_data").is_some() {
                text_parts.push("[inlineData]".to_string());
            }
        }

        let text = text_parts.join("\n");
        if !text.is_empty() || !tool_calls.is_empty() {
            messages.push(ChatMessage {
                provider_state: None,
                role: role.to_string(),
                content: if text.is_empty() {
                    None
                } else {
                    Some(engine_core::MessageContent::text(text))
                },
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                tool_call_id: None,
                name: None,
            });
        }
        messages.extend(tool_results);
    }

    let gen_config = req
        .generation_config
        .as_ref()
        .or(req.generation_config_camel.as_ref());
    let temperature = gen_config
        .and_then(|c| c.get("temperature"))
        .and_then(|t| t.as_f64());
    let max_tokens = gen_config
        .and_then(|c| {
            c.get("maxOutputTokens")
                .or_else(|| c.get("max_output_tokens"))
        })
        .and_then(|t| t.as_u64())
        .map(|t| t as u32);

    ChatCompletionRequest {
        model: model.to_string(),
        messages,
        tools: req.tools.as_deref().and_then(|tools| {
            let declarations = tools
                .iter()
                .flat_map(|tool| {
                    tool.get("functionDeclarations")
                        .or_else(|| tool.get("function_declarations"))
                        .and_then(|value| value.as_array())
                        .into_iter()
                        .flatten()
                })
                .filter_map(|declaration| {
                    let name = declaration.get("name")?.as_str()?.to_string();
                    Some(engine_core::ToolDefinition {
                        action_contract: declaration
                            .get("action_contract")
                            .cloned()
                            .and_then(|value| serde_json::from_value(value).ok()),
                        name,
                        description: declaration
                            .get("description")
                            .and_then(|value| value.as_str())
                            .unwrap_or("")
                            .to_string(),
                        parameters: declaration
                            .get("parameters")
                            .or_else(|| declaration.get("parametersJsonSchema"))
                            .or_else(|| declaration.get("parameters_json_schema"))
                            .cloned()
                            .unwrap_or_else(|| json!({"type": "object"})),
                        safe_to_memoize: false,
                        memoization_policy: None,
                    })
                })
                .collect::<Vec<_>>();
            (!declarations.is_empty()).then_some(declarations)
        }),
        tool_choice: None,
        temperature,
        max_tokens,
        stream: Some(stream),
        extra: extra_with_delegation_profile(&req.extra, "gemini"),
    }
}

/// Convert an OpenAI-shape chat completion response into the Gemini
/// `GenerateContentResponse` format.
fn chat_response_to_gemini(chat_response: serde_json::Value, model: &str) -> serde_json::Value {
    let message = chat_response
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
        .and_then(|c| c.get("message"))
        .cloned()
        .unwrap_or(json!({}));

    let content_text = message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    let usage = chat_response.get("usage").cloned().unwrap_or(json!({}));
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(|t| t.as_u64())
        .unwrap_or(0);
    let completion_tokens = usage
        .get("completion_tokens")
        .and_then(|t| t.as_u64())
        .unwrap_or(0);

    // Function-call parts, if any.
    let mut parts = vec![json!({ "text": content_text })];
    if let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in tool_calls {
            let function = tc.get("function").cloned().unwrap_or(json!({}));
            let name = function.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let args_str = function
                .get("arguments")
                .and_then(|a| a.as_str())
                .unwrap_or("{}");
            let args: serde_json::Value = serde_json::from_str(args_str).unwrap_or(json!({}));
            parts.push(json!({ "functionCall": { "name": name, "args": args } }));
        }
    }

    json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": parts,
            },
            "finishReason": "STOP",
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": completion_tokens,
            "totalTokenCount": prompt_tokens + completion_tokens,
        },
        "modelVersion": model,
    })
}

/// Dispatch handler for `/v1beta/models/*model` (multi-provider).
///
/// The Gemini SDK sends URLs like `/v1beta/models/gemini-pro:generateContent`
/// where the `:generateContent` / `:streamGenerateContent` / `:countTokens`
/// suffix specifies the action. Axum 0.7 can't have two colons in one path
/// segment, so we use a wildcard `/*model` and dispatch based on the suffix
/// here. The suffix is stripped before passing the model name to the inner
/// handlers.
pub async fn gemini_dispatch(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(raw): Path<String>,
    req: axum::body::Body,
) -> Result<axum::response::Response, StatusCode> {
    let (model, action) = split_gemini_model_action(&raw);
    let bytes = to_bytes(req, 10 * 1024 * 1024)
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    let json_req: GeminiGenerateContentRequest =
        serde_json::from_slice(&bytes).map_err(|_| StatusCode::BAD_REQUEST)?;

    match action.as_str() {
        "generateContent" => {
            let resp =
                gemini_generate_content(State(state), headers, Path(model), Json(json_req)).await?;
            Ok(resp.into_response())
        }
        "streamGenerateContent" => {
            let resp =
                gemini_stream_generate_content(State(state), headers, Path(model), Json(json_req))
                    .await?;
            Ok(resp)
        }
        "countTokens" => {
            let resp = gemini_count_tokens(State(state), Path(model), Json(json_req)).await?;
            Ok(resp.into_response())
        }
        _ => Err(StatusCode::NOT_FOUND),
    }
}

/// Dispatch handler for `/v1beta/models/*model` (single-provider).
pub async fn gemini_dispatch_single_provider(
    State(state): State<SingleProviderAppState>,
    headers: HeaderMap,
    Path(raw): Path<String>,
    req: axum::body::Body,
) -> Result<axum::response::Response, StatusCode> {
    let (model, action) = split_gemini_model_action(&raw);
    let bytes = to_bytes(req, 10 * 1024 * 1024)
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    let json_req: GeminiGenerateContentRequest =
        serde_json::from_slice(&bytes).map_err(|_| StatusCode::BAD_REQUEST)?;

    match action.as_str() {
        "generateContent" => {
            let resp = gemini_generate_content_single_provider(
                State(state),
                headers,
                Path(model),
                Json(json_req),
            )
            .await?;
            Ok(resp.into_response())
        }
        "streamGenerateContent" => {
            let resp = gemini_stream_generate_content_single_provider(
                State(state),
                headers,
                Path(model),
                Json(json_req),
            )
            .await?;
            Ok(resp)
        }
        "countTokens" => {
            let resp =
                gemini_count_tokens_single_provider(State(state), Path(model), Json(json_req))
                    .await?;
            Ok(resp.into_response())
        }
        _ => Err(StatusCode::NOT_FOUND),
    }
}

/// Split a wildcard path like `gemini-pro:generateContent` into
/// (`gemini-pro`, `generateContent`). If there's no colon, the action is
/// empty.
fn split_gemini_model_action(raw: &str) -> (String, String) {
    match raw.rfind(':') {
        Some(idx) => (raw[..idx].to_string(), raw[idx + 1..].to_string()),
        None => (raw.to_string(), String::new()),
    }
}

/// `POST /v1beta/models/:model:generateContent` - non-streaming Gemini API.
pub async fn gemini_generate_content(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(model): Path<String>,
    Json(req): Json<GeminiGenerateContentRequest>,
) -> Result<axum::response::Response, StatusCode> {
    let chat_req = gemini_to_chat_request(&req, &model, false);
    let chat_response = chat_completions(State(state), headers, Json(chat_req)).await?;
    let (chat_json, provider_response_headers) = match extract_json_response(chat_response).await {
        Ok(response) => response,
        Err(refusal) => return Ok(refusal),
    };
    Ok(with_provider_rate_limit_headers(
        Json(chat_response_to_gemini(chat_json, &model)).into_response(),
        &provider_response_headers,
    ))
}

/// `POST /v1beta/models/:model:generateContent` - single-provider variant.
pub async fn gemini_generate_content_single_provider(
    State(state): State<SingleProviderAppState>,
    headers: HeaderMap,
    Path(model): Path<String>,
    Json(req): Json<GeminiGenerateContentRequest>,
) -> Result<axum::response::Response, StatusCode> {
    let chat_req = gemini_to_chat_request(&req, &model, false);
    let chat_response =
        chat_completions_single_provider(State(state), headers, Json(chat_req)).await?;
    let (chat_json, provider_response_headers) = match extract_json_response(chat_response).await {
        Ok(response) => response,
        Err(refusal) => return Ok(refusal),
    };
    Ok(with_provider_rate_limit_headers(
        Json(chat_response_to_gemini(chat_json, &model)).into_response(),
        &provider_response_headers,
    ))
}

/// `POST /v1beta/models/:model:streamGenerateContent` - streaming Gemini API.
///
/// The Gemini SDK consumes Server-Sent Events where each `data:` line is a
/// single `GenerateContentResponse` (one per chunk). We reuse the OpenAI
/// streaming conversion and re-shape each chunk to the Gemini candidate
/// format. The final event carries `usageMetadata` and a `finishReason`.
pub async fn gemini_stream_generate_content(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(model): Path<String>,
    Json(req): Json<GeminiGenerateContentRequest>,
) -> Result<axum::response::Response, StatusCode> {
    let chat_req = gemini_to_chat_request(&req, &model, true);
    let chat_response = chat_completions(State(state), headers, Json(chat_req)).await?;
    let (chat_json, provider_response_headers) = match extract_json_response(chat_response).await {
        Ok(response) => response,
        Err(refusal) => return Ok(refusal),
    };
    let gemini_stream = chat_completion_to_gemini_sse_stream(chat_json, &model);
    let response = Sse::new(gemini_stream)
        .keep_alive(KeepAlive::default())
        .into_response();
    Ok(with_provider_rate_limit_headers(
        response,
        &provider_response_headers,
    ))
}

/// `POST /v1beta/models/:model:streamGenerateContent` - single-provider variant.
pub async fn gemini_stream_generate_content_single_provider(
    State(state): State<SingleProviderAppState>,
    headers: HeaderMap,
    Path(model): Path<String>,
    Json(req): Json<GeminiGenerateContentRequest>,
) -> Result<axum::response::Response, StatusCode> {
    let chat_req = gemini_to_chat_request(&req, &model, true);
    let chat_response =
        chat_completions_single_provider(State(state), headers, Json(chat_req)).await?;
    let (chat_json, provider_response_headers) = match extract_json_response(chat_response).await {
        Ok(response) => response,
        Err(refusal) => return Ok(refusal),
    };
    let gemini_stream = chat_completion_to_gemini_sse_stream(chat_json, &model);
    let response = Sse::new(gemini_stream)
        .keep_alive(KeepAlive::default())
        .into_response();
    Ok(with_provider_rate_limit_headers(
        response,
        &provider_response_headers,
    ))
}

/// Convert a non-streaming chat completion response into a Gemini-format
/// SSE stream. The SDK tolerates a single `data:` carrying the full
/// response followed by the standard `[DONE]` marker, so we reuse the
/// OpenAI streaming conversion's "split into content + tool call + final"
/// logic but re-shape each chunk to the Gemini candidate format.
fn chat_completion_to_gemini_sse_stream(
    response: serde_json::Value,
    model: &str,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let gemini_response = chat_response_to_gemini(response, model);
    let stream = async_stream::stream! {
        // Single chunk with the complete candidate (text + any tool calls).
        yield Ok(Event::default().data(gemini_response.to_string()));
        // The Gemini SDK does not require a [DONE] marker (it reads until
        // the stream closes), but sending one is harmless and matches the
        // OpenAI SSE convention some SDKs look for.
        yield Ok(Event::default().data("[DONE]"));
    };
    stream
}

/// `GET /v1beta/models` - list models in the Gemini API shape.
///
/// The Gemini CLI expects `{ "models": [{ "name": "models/<id>", ... }] }`.
pub async fn gemini_list_models(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let models = state
        .providers
        .list_all_models()
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?;

    let data: Vec<serde_json::Value> = models
        .into_iter()
        .map(|m| {
            // Pull context window from the curated catalog so Gemini CLI
            // discovers the real limit instead of falling back to its
            // own (smaller) hardcoded default.
            let catalog_entry = find_model(&m.provider, &m.id);
            let input_token_limit = catalog_entry
                .as_ref()
                .map(|c| c.max_input_tokens)
                .unwrap_or_else(|| fallback_context_limit() as i64);
            let output_token_limit = catalog_entry
                .as_ref()
                .map(|c| c.max_output_tokens)
                .unwrap_or(65_536);
            json!({
                "name": format!("models/{}", m.id),
                "version": "001",
                "displayName": m.display_name.unwrap_or_else(|| m.id.clone()),
                "description": format!("ReinCore-routed model: {}", m.id),
                "inputTokenLimit": input_token_limit,
                "outputTokenLimit": output_token_limit,
                "supportedGenerationMethods": [
                    "generateContent",
                    "streamGenerateContent",
                    "countTokens",
                ],
            })
        })
        .collect();

    Ok(Json(json!({ "models": data })))
}

/// `GET /v1beta/models` - single-provider variant.
pub async fn gemini_list_models_single_provider(
    State(state): State<SingleProviderAppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let models = state
        .provider
        .list_models()
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?;

    let data: Vec<serde_json::Value> = models
        .into_iter()
        .map(|m| {
            // Pull context window from the curated catalog so Gemini CLI
            // discovers the real limit instead of falling back to its
            // own (smaller) hardcoded default.
            let catalog_entry = find_model(&m.provider, &m.id);
            let input_token_limit = catalog_entry
                .as_ref()
                .map(|c| c.max_input_tokens)
                .unwrap_or_else(|| fallback_context_limit() as i64);
            let output_token_limit = catalog_entry
                .as_ref()
                .map(|c| c.max_output_tokens)
                .unwrap_or(65_536);
            json!({
                "name": format!("models/{}", m.id),
                "version": "001",
                "displayName": m.display_name.unwrap_or_else(|| m.id.clone()),
                "description": format!("ReinCore-routed model: {}", m.id),
                "inputTokenLimit": input_token_limit,
                "outputTokenLimit": output_token_limit,
                "supportedGenerationMethods": [
                    "generateContent",
                    "streamGenerateContent",
                    "countTokens",
                ],
            })
        })
        .collect();

    Ok(Json(json!({ "models": data })))
}

/// `POST /v1beta/models/:model:countTokens` - token counting.
pub async fn gemini_count_tokens(
    State(_state): State<AppState>,
    Path(_model): Path<String>,
    Json(req): Json<GeminiGenerateContentRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let chat_req = gemini_to_chat_request(&req, "", false);
    let combined: String = chat_req
        .messages
        .iter()
        .filter_map(|m| m.content_str())
        .collect::<Vec<_>>()
        .join("\n");
    let estimated_tokens = engine_core::count_text_tokens(&combined);

    Ok(Json(json!({
        "totalTokens": estimated_tokens,
    })))
}

/// `POST /v1beta/models/:model:countTokens` - single-provider variant.
pub async fn gemini_count_tokens_single_provider(
    State(_state): State<SingleProviderAppState>,
    Path(_model): Path<String>,
    Json(req): Json<GeminiGenerateContentRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let chat_req = gemini_to_chat_request(&req, "", false);
    let combined: String = chat_req
        .messages
        .iter()
        .filter_map(|m| m.content_str())
        .collect::<Vec<_>>()
        .join("\n");
    let estimated_tokens = engine_core::count_text_tokens(&combined);

    Ok(Json(json!({
        "totalTokens": estimated_tokens,
    })))
}
