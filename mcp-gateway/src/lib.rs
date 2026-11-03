//! Stateless Streamable HTTP MCP tools adapter for ReinCore engines.
//! Hosts authenticate every request; backends recheck session authorization.
mod protocol;
#[cfg(test)]
mod tests;

use async_trait::async_trait;
use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use engine_core::ToolDefinition;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::watch;
use uuid::Uuid;

pub const SESSION_HEADER: &str = "x-reincore-session";
pub const MAX_BODY_BYTES: usize = 2 * 1024 * 1024;

/// A per-request adapter carrying the host's verified identity, never raw JWTs.
#[async_trait]
pub trait Backend: Send + Sync {
    /// Authorize the current session and return an opaque cancellation scope.
    /// Include principal, key scope, workspace and resolved session identity.
    async fn authorize(&self, session: Uuid) -> Result<String, StatusCode>;
    async fn list_tools(&self, session: Uuid) -> Result<Vec<ToolDefinition>, StatusCode>;
    /// Return the existing HTTP tool service's result envelope.
    async fn call_tool(&self, session: Uuid, call: Call) -> Result<Value, StatusCode>;
}

pub struct Call {
    pub name: String,
    pub arguments: Value,
    pub safe_to_memoize: bool,
    pub request_group_id: Option<String>,
}

type Key = (String, String);
type Pending = Arc<Mutex<HashMap<Key, watch::Sender<bool>>>>;

/// Shared per engine instance, not global across hosts or processes.
#[derive(Clone, Default)]
pub struct Gateway {
    pending: Pending,
}

struct PendingGuard {
    pending: Pending,
    key: Key,
}
impl Drop for PendingGuard {
    fn drop(&mut self) {
        self.pending.lock().unwrap().remove(&self.key);
    }
}

impl Gateway {
    pub async fn handle(
        &self,
        backend: &dyn Backend,
        headers: &HeaderMap,
        body: &[u8],
    ) -> Response {
        if let Err(status) = protocol::validate_headers(headers) {
            return status.into_response();
        }

        if body.len() > MAX_BODY_BYTES {
            return StatusCode::PAYLOAD_TOO_LARGE.into_response();
        }

        let session = match session_id(headers) {
            Ok(id) => id,
            Err(status) => return status.into_response(),
        };

        let scope = match backend.authorize(session).await {
            Ok(scope) => scope,
            Err(status) => return status.into_response(),
        };

        let message: Value = match serde_json::from_slice(body) {
            Ok(message) => message,
            Err(_) => return protocol::error(Value::Null, -32700, "Parse error"),
        };

        if message.get("jsonrpc").and_then(Value::as_str) != Some("2.0") || !message.is_object() {
            return protocol::error(Value::Null, -32600, "Invalid Request");
        }

        let method = match message.get("method").and_then(Value::as_str) {
            Some(method) => method,
            None => return protocol::error(Value::Null, -32600, "Invalid Request"),
        };

        let params = message.get("params").cloned().unwrap_or(json!({}));
        if !params.is_object() {
            if message.get("id").is_none() {
                return StatusCode::BAD_REQUEST.into_response();
            }
            return protocol::error(
                message.get("id").cloned().unwrap_or(Value::Null),
                -32602,
                "Invalid params",
            );
        }

        let Some(id) = message.get("id").cloned() else {
            if method == "notifications/cancelled" {
                if let Some(request_id) =
                    params.get("requestId").filter(|id| protocol::valid_id(id))
                {
                    if let Some(cancel) = self
                        .pending
                        .lock()
                        .unwrap()
                        .get(&(scope, request_id.to_string()))
                    {
                        let _ = cancel.send(true);
                    }
                }
            } else if !method.starts_with("notifications/") {
                return StatusCode::BAD_REQUEST.into_response();
            }
            return StatusCode::ACCEPTED.into_response();
        };

        if !protocol::valid_id(&id) {
            return protocol::error(Value::Null, -32600, "Invalid request ID");
        }

        // Admit the request before discovery so cancellation, duplicate IDs and
        // capacity limits cover the entire operation, including initialization waits.
        let key = (scope, id.to_string());
        let (cancel, mut cancelled) = watch::channel(false);
        {
            let mut pending = self.pending.lock().unwrap();
            if pending.contains_key(&key) {
                return protocol::error(id, -32600, "Request ID already in flight");
            }
            if pending.len() >= 1024 {
                return StatusCode::TOO_MANY_REQUESTS.into_response();
            }
            pending.insert(key.clone(), cancel);
        }
        let _guard = PendingGuard {
            pending: self.pending.clone(),
            key,
        };
        let response_id = id.clone();
        let operation = async {
            match method {
                "initialize" => {
                    let Some(requested) = params.get("protocolVersion").and_then(Value::as_str)
                    else {
                        return protocol::error(id, -32602, "Missing protocolVersion");
                    };
                    if !params.get("capabilities").is_some_and(Value::is_object)
                        || !params.get("clientInfo").is_some_and(|v| {
                            v.get("name").is_some_and(Value::is_string)
                                && v.get("version").is_some_and(Value::is_string)
                        })
                    {
                        return protocol::error(id, -32602, "Invalid initialization parameters");
                    }
                    let version = if protocol::supported_version(requested) {
                        requested
                    } else {
                        "2025-06-18"
                    };
                    protocol::result(
                        id,
                        json!({
                            "protocolVersion": version,
                            "capabilities": { "tools": {} },
                            "serverInfo": { "name": "reincore", "version": env!("CARGO_PKG_VERSION") },
                        }),
                    )
                }
                "ping" => protocol::result(id, json!({})),
                "tools/list" => {
                    if params.get("cursor").is_some() {
                        return protocol::error(id, -32602, "Pagination cursors are not supported");
                    }
                    match backend.list_tools(session).await {
                        Ok(tools) => {
                            let tools: Vec<_> = tools
                                .into_iter()
                                .map(|tool| {
                                    json!({
                                        "name": tool.name,
                                        "description": tool.description,
                                        "inputSchema": tool.parameters
                                    })
                                })
                                .collect();
                            protocol::result(id, json!({ "tools": tools }))
                        }
                        Err(status) => status.into_response(),
                    }
                }
                "tools/call" => {
                    let call = match protocol::parse_call(params, headers) {
                        Ok(call) => call,
                        Err(()) => return protocol::error(id, -32602, "Invalid tool parameters"),
                    };
                    // Discovery uses the same authorized backend. Unknown tools never dispatch.
                    let tools = match backend.list_tools(session).await {
                        Ok(tools) => tools,
                        Err(status) => return status.into_response(),
                    };
                    if !tools.iter().any(|tool| tool.name == call.name) {
                        return protocol::error(id, -32602, "Tool not available for this session");
                    }

                    let outcome = backend.call_tool(session, call).await;

                    match outcome {
                        Ok(envelope) => match protocol::tool_result(envelope) {
                            Some(result) => protocol::result(id, result),
                            None => protocol::error(id, -32603, "Invalid tool service result"),
                        },
                        Err(status)
                            if matches!(
                                status,
                                StatusCode::FORBIDDEN
                                    | StatusCode::NOT_FOUND
                                    | StatusCode::UNAUTHORIZED
                            ) =>
                        {
                            status.into_response()
                        }
                        Err(_) => protocol::result(
                            id,
                            json!({
                                "content": [{
                                    "type": "text",
                                    "text": "Tool execution failed or was blocked; inspect the ReinCore session. Do not retry an ambiguous mutation."
                                }],
                                "isError": true
                            }),
                        ),
                    }
                }
                _ => protocol::error(id.clone(), -32601, "Method not found"),
            }
        };
        tokio::select! {
            biased;
            _ = cancelled.changed(), if method != "initialize" => StatusCode::NO_CONTENT.into_response(),
            outcome = tokio::time::timeout(Duration::from_secs(125), operation) => {
                outcome.unwrap_or_else(|_| protocol::error(response_id, -32000,
                    "MCP deadline exceeded; upstream outcome may be unknown"))
            }
        }
    }
}

pub fn session_id(headers: &HeaderMap) -> Result<Uuid, StatusCode> {
    if headers.get_all(SESSION_HEADER).iter().count() != 1 {
        return Err(StatusCode::BAD_REQUEST);
    }
    headers
        .get(SESSION_HEADER)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| Uuid::parse_str(h).ok())
        .filter(|id| !id.is_nil())
        .ok_or(StatusCode::BAD_REQUEST)
}

/// No browser origins are admitted; native clients omit Origin. This avoids
/// trusting attacker-controlled Host/forwarded headers for DNS-rebinding checks.
pub fn reject_origin(headers: &HeaderMap) -> Result<(), StatusCode> {
    if headers.contains_key("origin") {
        Err(StatusCode::FORBIDDEN)
    } else {
        Ok(())
    }
}

/// This stateless profile does not offer GET subscription streams or DELETE sessions.
pub async fn unsupported_transport(headers: HeaderMap) -> Response {
    match reject_origin(&headers) {
        Err(status) => status.into_response(),
        Ok(()) => (StatusCode::METHOD_NOT_ALLOWED, [("allow", "POST")]).into_response(),
    }
}
