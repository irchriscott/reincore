//! Server-side MCP tool loop. The session's authorized MCP tools are offered
//! to the model next to the client's own tools. The client never sees them,
//! so it cannot execute their calls: the engine runs those calls through the
//! shared tool services (authorization, loop protection, event recording,
//! memoization) and asks the model again until it answers or calls a client
//! tool.
use super::chat::{
    chat_completion_to_sse_stream, extract_json_response, with_provider_rate_limit_headers,
};
use super::*;
use engine_core::{MessageContent, ToolCall, ToolExecutorRegistry};
use std::collections::HashSet;
use std::future::Future;

/// Rounds of MCP calls allowed within one client turn before the engine
/// stops and hands the client what it has.
const MAX_MCP_ROUNDS: usize = 8;

/// Offer the session's authorized MCP tools to the model. A tool whose name
/// a client tool already uses is left out: that name belongs to the client.
pub(super) async fn inject_mcp_tools(
    registry: &ToolExecutorRegistry,
    session: &Session,
    req: &mut ChatCompletionRequest,
) {
    let tools = match registry
        .list_local_tools_in_context(&super::tools::local_context(session))
        .await
    {
        Ok(tools) => tools,
        Err(error) => {
            warn!(error = %error, "could not list MCP tools for the session");
            return;
        }
    };
    if tools.is_empty() {
        return;
    }
    let offered = req.tools.get_or_insert_with(Vec::new);
    for tool in tools {
        if !offered.iter().any(|existing| existing.name == tool.name) {
            offered.push(tool);
        }
    }
}

/// Run one client turn, executing the model's MCP tool calls in between
/// provider rounds. `round` runs a single request through the full chat
/// pipeline, which injects the MCP tools itself. Rounds run unstreamed
/// because each response is inspected before the client may see it; the
/// final one is streamed afterwards when the client asked for a stream.
pub(super) async fn run_with_mcp_tools<F, Fut>(
    db: &LocalDatabase,
    registry: &ToolExecutorRegistry,
    action_policy: &engine_core::action_gate::ActionPolicy,
    mut req: ChatCompletionRequest,
    round: F,
) -> Result<axum::response::Response, StatusCode>
where
    F: Fn(ChatCompletionRequest) -> Fut,
    Fut: Future<Output = Result<axum::response::Response, StatusCode>>,
{
    let wants_stream = req.stream.unwrap_or(false);
    req.stream = Some(false);
    if wants_stream {
        // Streaming requests were never batch-submitted; forcing the rounds
        // unstreamed must not change that.
        req.extra
            .insert("reincore_batch_eligible".to_string(), json!(false));
    }
    let client_tools: HashSet<String> = req
        .tools
        .iter()
        .flatten()
        .map(|tool| tool.name.clone())
        .collect();
    let mut rounds = 0;
    loop {
        let response = round(req.clone()).await?;
        // Batch submissions (202), refusals and errors are not completions:
        // they reach the client exactly as the pipeline produced them.
        if response.status() != StatusCode::OK {
            return Ok(response);
        }
        let (mut json, headers) = match extract_json_response(response).await {
            Ok(response) => response,
            Err(refusal) => return Ok(refusal),
        };
        let session_id = json
            .get("reincore_session_id")
            .and_then(|value| value.as_str())
            .and_then(|value| Uuid::parse_str(value).ok());
        let message = json
            .pointer("/choices/0/message")
            .cloned()
            .and_then(|value| serde_json::from_value::<ChatMessage>(value).ok());
        let (Some(session_id), Some(mut message)) = (session_id, message) else {
            return Ok(finish(json, &headers, wants_stream));
        };
        let mcp_tools = mcp_tools_for(db, registry, session_id, &client_tools).await;
        let calls: Vec<ToolCall> = message
            .tool_calls
            .iter()
            .flatten()
            .filter_map(|call| {
                let name = mcp_tool_name(&call.function.name, &mcp_tools)?;
                let mut call = call.clone();
                call.function.name = name.to_string();
                Some(call)
            })
            .collect();
        if calls.is_empty() {
            return Ok(finish(json, &headers, wants_stream));
        }
        if rounds == MAX_MCP_ROUNDS {
            warn!(
                session_id = %session_id,
                "MCP tool round limit reached; returning the turn to the client"
            );
            strip_mcp_calls(&mut json, &mcp_tools);
            return Ok(finish(json, &headers, wants_stream));
        }
        rounds += 1;

        // Client calls issued alongside MCP calls are dropped: their results
        // would never reach this loop. The model re-plans with the MCP
        // results and issues them again if it still needs them. Opaque
        // replay state described the full message, so it goes with them.
        if message.tool_calls.as_ref().map(Vec::len) != Some(calls.len()) {
            message.provider_state = None;
        }
        message.tool_calls = Some(calls.clone());
        req.messages.push(message);
        for call in calls {
            let safe_to_memoize = mcp_tools[&call.function.name];
            let content = execute_mcp_call(
                db,
                registry,
                action_policy,
                session_id,
                &call,
                safe_to_memoize,
            )
            .await;
            req.messages.push(ChatMessage {
                provider_state: None,
                role: "tool".to_string(),
                content: Some(MessageContent::text(content)),
                tool_calls: None,
                tool_call_id: Some(call.id),
                name: Some(call.function.name),
            });
        }
        // Later rounds carry history the client never sent; pin them to the
        // session the first round resolved.
        req.extra.insert(
            "reincore_session_id".to_string(),
            json!(session_id.to_string()),
        );
    }
}

/// The MCP tool a call addresses, if any. Models that saw the client's own
/// namespaced calls replayed in history (`muse.read_file`) sometimes give
/// MCP tools the same prefix (`muse.my_shop__list_orders`).
fn mcp_tool_name<'a>(name: &'a str, mcp_tools: &HashMap<String, bool>) -> Option<&'a str> {
    if mcp_tools.contains_key(name) {
        return Some(name);
    }
    let (_, unprefixed) = name.rsplit_once('.')?;
    mcp_tools.contains_key(unprefixed).then_some(unprefixed)
}

/// Authorized MCP tool names for the session, with their memoization
/// consent, minus names the client's own tools use.
async fn mcp_tools_for(
    db: &LocalDatabase,
    registry: &ToolExecutorRegistry,
    session_id: Uuid,
    client_tools: &HashSet<String>,
) -> HashMap<String, bool> {
    match super::tools::authorized_tools(db, registry, session_id).await {
        Ok((_, tools)) => tools
            .into_iter()
            .filter(|tool| !client_tools.contains(&tool.name))
            .map(|tool| (tool.name, tool.safe_to_memoize))
            .collect(),
        Err(status) => {
            warn!(session_id = %session_id, %status, "could not list MCP tools for the session");
            HashMap::new()
        }
    }
}

/// Execute one MCP call for the model and render its outcome as tool-message
/// content. Failures become content too, so the model can recover.
async fn execute_mcp_call(
    db: &LocalDatabase,
    registry: &ToolExecutorRegistry,
    action_policy: &engine_core::action_gate::ActionPolicy,
    session_id: Uuid,
    call: &ToolCall,
    safe_to_memoize: bool,
) -> String {
    let Ok(arguments) = serde_json::from_str::<serde_json::Value>(&call.function.arguments) else {
        return format!(
            "Tool call rejected: arguments for `{}` are not valid JSON.",
            call.function.name
        );
    };
    let request = super::tools::ToolCallRequest {
        session_id,
        tool_name: call.function.name.clone(),
        arguments,
        safe_to_memoize,
        is_subagent: false,
        tool_call_id: Some(call.id.clone()),
    };
    match super::tools::execute_local_tool_call(db, registry, request, action_policy).await {
        Ok((status, Json(body))) if status.is_success() => body
            .pointer("/result/content")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string(),
        Ok((status, _)) => format!(
            "Tool call to `{}` was blocked by ReinCore ({status}).",
            call.function.name
        ),
        Err(status) => format!("Tool call to `{}` failed ({status}).", call.function.name),
    }
}

/// Out of MCP rounds: hand the client only the calls it can execute.
fn strip_mcp_calls(json: &mut serde_json::Value, mcp_tools: &HashMap<String, bool>) {
    let Some(choice) = json.pointer_mut("/choices/0") else {
        return;
    };
    let remaining: Vec<serde_json::Value> = choice
        .pointer("/message/tool_calls")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter(|call| {
            !call
                .pointer("/function/name")
                .and_then(|value| value.as_str())
                .and_then(|name| mcp_tool_name(name, mcp_tools))
                .is_some()
        })
        .cloned()
        .collect();
    let Some(message) = choice
        .get_mut("message")
        .and_then(|value| value.as_object_mut())
    else {
        return;
    };
    if !remaining.is_empty() {
        message.insert("tool_calls".to_string(), json!(remaining));
        return;
    }
    message.remove("tool_calls");
    let has_text = message
        .get("content")
        .and_then(|value| value.as_str())
        .is_some_and(|text| !text.trim().is_empty());
    if !has_text {
        message.insert(
            "content".to_string(),
            json!(format!(
                "Stopped after {MAX_MCP_ROUNDS} rounds of MCP tool calls without a final answer."
            )),
        );
    }
    choice["finish_reason"] = json!("stop");
}

fn finish(
    json: serde_json::Value,
    headers: &engine_core::ProviderResponseHeaders,
    wants_stream: bool,
) -> axum::response::Response {
    // Checkpoints and other non-completion bodies stay JSON, as they always
    // have for streaming requests.
    let response = if wants_stream && json.get("choices").is_some() {
        Sse::new(chat_completion_to_sse_stream(json))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(json).into_response()
    };
    with_provider_rate_limit_headers(response, headers)
}

#[cfg(test)]
mod tests;
