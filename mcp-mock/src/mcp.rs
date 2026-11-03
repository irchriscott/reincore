//! An in-process MCP HTTP server for transport and dispatch tests.

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

/// A running mock server. Dropping the fixture stops its listener.
pub struct McpFixture {
    pub url: String,
    pub messages: Arc<Mutex<Vec<Value>>>,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for McpFixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[derive(Clone)]
struct Mock {
    messages: Arc<Mutex<Vec<Value>>>,
    sse: bool,
}

impl McpFixture {
    pub async fn start(sse: bool) -> Self {
        let messages = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new().route("/mcp", post(handle)).with_state(Mock {
            messages: messages.clone(),
            sse,
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            url,
            messages,
            task,
        }
    }

    pub fn calls(&self) -> usize {
        self.messages
            .lock()
            .unwrap()
            .iter()
            .filter(|v| v["method"] == "tools/call")
            .count()
    }

    pub fn config(&self, path: &str) -> Value {
        json!({
            "servers": [{
                "name": "fixture",
                "url": self.url,
                "allow_loopback_http": true,
                "timeout_ms": 1000,
                "access": { "mode": "local", "project_path": path },
                "tools": {
                    "read": {
                        "read_only": true,
                        "cache": {
                            "read_only": true,
                            "sharing": "private",
                            "semantic_reuse": false,
                            "ttl_seconds": 60,
                            "policy_version": "1",
                            "tool_version": "1",
                            "data_namespace": "fixture",
                            "data_version": "1",
                            "credential_identity": "fixture-service",
                            "credential_version": "1"
                        }
                    },
                    "write": { "read_only": false },
                    "slow": { "read_only": false },
                    "error": { "read_only": true }
                }
            }]
        })
    }
}

async fn handle(
    State(state): State<Mock>,
    headers: HeaderMap,
    Json(request): Json<Value>,
) -> Response {
    assert!(headers["accept"]
        .to_str()
        .unwrap()
        .contains("application/json"));
    assert!(headers["accept"]
        .to_str()
        .unwrap()
        .contains("text/event-stream"));

    if request.get("method").is_none() {
        assert_eq!(headers["mcp-session-id"], "fixture-session");
        assert_eq!(headers["mcp-protocol-version"], "2025-03-26");
        state
            .messages
            .lock()
            .unwrap()
            .push(json!({"method":"client/reply","params":request}));
        return StatusCode::ACCEPTED.into_response();
    }

    let method = request["method"].as_str().unwrap();
    if method != "initialize" {
        assert_eq!(headers["mcp-session-id"], "fixture-session");
        assert_eq!(headers["mcp-protocol-version"], "2025-03-26");
    }

    let authorization = headers
        .get("authorization")
        .map(|header| header.to_str().unwrap());

    state.messages.lock().unwrap().push(json!({
        "method": method,
        "params": request["params"],
        "authorization": authorization,
        "id": request["id"]
    }));

    if method.starts_with("notifications/") {
        return StatusCode::ACCEPTED.into_response();
    }

    let result = match method {
        "initialize" => {
            assert_eq!(request["params"]["protocolVersion"], "2025-06-18");
            json!({
                "protocolVersion": "2025-03-26",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "fixture", "version": "1" }
            })
        }
        "tools/list" => {
            if request["params"].get("cursor").is_none() {
                json!({
                    "tools": [{
                        "name": "read",
                        "description": "fixture read",
                        "inputSchema": { "type": "object" }
                    }],
                    "nextCursor": "page2"
                })
            } else {
                assert_eq!(request["params"]["cursor"], "page2");
                json!({
                    "tools": [
                        { "name": "write", "inputSchema": { "type": "object" } },
                        { "name": "slow", "inputSchema": { "type": "object" } },
                        { "name": "error", "inputSchema": { "type": "object" } },
                        { "name": "unlisted", "inputSchema": { "type": "object" } }
                    ]
                })
            }
        }
        "tools/call" => {
            assert!(request["params"]["arguments"].is_object());
            if request["params"]["arguments"]["expired"] == true {
                return StatusCode::NOT_FOUND.into_response();
            }
            if request["params"]["name"] == "slow" {
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            }
            json!({
                "content": [{ "type": "text", "text": "fixture result" }],
                "isError": request["params"]["name"] == "error"
            })
        }
        _ => panic!("unexpected MCP method"),
    };

    let value = json!({"jsonrpc":"2.0", "id":request["id"], "result":result});
    let mut response = if state.sse {
        let mut text = String::new();
        if method == "tools/call" {
            let ping = json!({
                "jsonrpc": "2.0",
                "id": "server-ping",
                "method": "ping"
            });
            let unsupported = json!({
                "jsonrpc": "2.0",
                "id": "unsupported",
                "method": "sampling/createMessage",
                "params": {}
            });
            text.push_str(&format!("data: {ping}\n\ndata: {unsupported}\n\n"));

            if request["params"]["arguments"]["changed"] == true {
                let changed = json!({
                    "jsonrpc": "2.0",
                    "method": "notifications/tools/list_changed"
                });
                text.push_str(&format!("data: {changed}\n\n"));
            }
        }
        let progress = json!({
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {}
        });
        text.push_str(&format!(
            "event: message\r\ndata: {progress}\r\n\r\n\
             event: message\r\ndata: {value}\r\n\r\n"
        ));
        ([("content-type", "text/event-stream")], text).into_response()
    } else {
        Json(value).into_response()
    };

    if method == "initialize" {
        response
            .headers_mut()
            .insert("mcp-session-id", "fixture-session".parse().unwrap());
    }
    response
}
