//! Host adapter: MCP and HTTP tools share execution and discovery services.
use super::*;
use axum::{body::Bytes, response::Response};
use mcp_gateway::{Backend, Call};

struct LocalBackend<'a> {
    db: &'a LocalDatabase,
    registry: &'a engine_core::ToolExecutorRegistry,
    action_policy: &'a engine_core::action_gate::ActionPolicy,
}

#[async_trait::async_trait]
impl Backend for LocalBackend<'_> {
    async fn authorize(&self, session_id: Uuid) -> Result<String, StatusCode> {
        let session = self
            .db
            .get_active_session_in_lineage(session_id)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .ok_or(StatusCode::NOT_FOUND)?;
        Ok(format!("local:{}", session.id))
    }

    async fn list_tools(
        &self,
        session: Uuid,
    ) -> Result<Vec<engine_core::ToolDefinition>, StatusCode> {
        Ok(
            super::tools::authorized_tools(self.db, self.registry, session)
                .await?
                .1,
        )
    }

    async fn call_tool(&self, session: Uuid, call: Call) -> Result<serde_json::Value, StatusCode> {
        let (status, Json(result)) = super::tools::execute_local_tool_call(
            self.db,
            self.registry,
            super::tools::ToolCallRequest {
                session_id: session,
                tool_name: call.name,
                arguments: call.arguments,
                safe_to_memoize: call.safe_to_memoize,
                is_subagent: false,
                tool_call_id: None,
            },
            self.action_policy,
        )
        .await?;
        if !status.is_success() {
            return Err(status);
        }
        Ok(result)
    }
}

pub async fn mcp(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let backend = LocalBackend {
        db: state.db.as_ref(),
        registry: state.tool_executors.as_ref(),
        action_policy: &state.action_policy,
    };
    state.mcp_gateway.handle(&backend, &headers, &body).await
}

pub async fn mcp_single_provider(
    State(state): State<SingleProviderAppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let backend = LocalBackend {
        db: state.db.as_ref(),
        registry: state.tool_executors.as_ref(),
        action_policy: &state.action_policy,
    };
    state.mcp_gateway.handle(&backend, &headers, &body).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::post, Router};
    use mcp_mock::McpFixture;
    use serde_json::Value;
    use std::sync::Arc;

    struct Server(tokio::task::JoinHandle<()>);

    impl Drop for Server {
        fn drop(&mut self) {
            self.0.abort();
        }
    }

    async fn rpc(client: &reqwest::Client, url: &str, session: Uuid, value: Value) -> Value {
        client
            .post(url)
            .header("accept", "application/json, text/event-stream")
            .header(mcp_gateway::SESSION_HEADER, session.to_string())
            .header("mcp-protocol-version", "2025-06-18")
            .header("x-reincore-tool-cache", "true")
            .json(&value)
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn gateway_real_http_dispatch_reuses_http_cache_and_cancels_upstream() {
        let upstream = McpFixture::start(true).await;
        let path = std::env::current_dir().unwrap();
        let config: engine_core::mcp::McpConfig =
            serde_json::from_value(upstream.config(path.to_str().unwrap())).unwrap();

        let db = LocalDatabase::open_in_memory().unwrap();
        let mut session = Session::new("MCP gateway", 100).unwrap();
        session.project_path = Some(path.to_str().unwrap().to_owned());
        db.insert_session(&session).unwrap();

        let mut denied = Session::new("other project", 100).unwrap();
        denied.project_path = Some(path.parent().unwrap().to_str().unwrap().to_owned());
        db.insert_session(&denied).unwrap();

        let mut state = AppState::new_with_project_path(db, Some(path.to_str().unwrap()));
        state.tool_executors = Arc::new(config.connect(false).await.unwrap());

        let db = state.db.clone();
        let app = Router::new()
            .route("/mcp", post(mcp).get(mcp_gateway::unsupported_transport))
            .route("/v1/tool-calls", post(tool_calls))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let url = format!("{base}/mcp");
        let _server = Server(tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        }));

        let client = reqwest::Client::new();
        let initialized = rpc(
            &client,
            &url,
            session.id,
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": { "name": "native-test", "version": "1" },
                },
            }),
        )
        .await;
        assert_eq!(initialized["result"]["serverInfo"]["name"], "reincore");

        let listed = rpc(
            &client,
            &url,
            session.id,
            json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }),
        )
        .await;
        assert_eq!(listed["result"]["tools"].as_array().unwrap().len(), 4);

        let read = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": { "name": "fixture__read", "arguments": { "query": "one" } },
        });
        let first = rpc(&client, &url, session.id, read.clone()).await;
        assert_eq!(first["result"]["content"][0]["text"], "fixture result");
        assert_eq!(first["result"]["_meta"]["reincore"]["cache_hit"], false);

        // A subsequent ordinary HTTP tool call hits the same cache and event service.
        let cached: Value = client
            .post(format!("{base}/v1/tool-calls"))
            .json(&json!({
                "session_id": session.id,
                "tool_name": "fixture__read",
                "arguments": { "query": "one" },
                "safe_to_memoize": true,
            }))
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(cached["cache_hit"], true);
        assert_eq!(upstream.calls(), 1);

        let mut second = read.clone();
        second["id"] = json!(4);
        second["params"]["arguments"]["query"] = json!("two");
        rpc(&client, &url, session.id, second.clone()).await;

        let mutation = rpc(
            &client,
            &url,
            session.id,
            json!({
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": { "name": "fixture__write" },
            }),
        )
        .await;
        assert_eq!(
            mutation["result"]["_meta"]["reincore"]["memoization_enabled"],
            false
        );

        second["id"] = json!(6);
        assert_eq!(
            rpc(&client, &url, session.id, second).await["result"]["_meta"]["reincore"]
                ["cache_hit"],
            false
        );
        assert_eq!(upstream.calls(), 4);
        assert_eq!(
            rpc(&client, &url, denied.id, read).await["error"]["code"],
            -32602
        );
        assert!(db.latest_seq(session.id).await.unwrap().unwrap() >= 10);
        assert_eq!(
            client
                .post(&url)
                .header("origin", "https://attacker.example")
                .header("accept", "application/json, text/event-stream")
                .header(mcp_gateway::SESSION_HEADER, session.id.to_string())
                .json(&json!({ "jsonrpc": "2.0", "id": 7, "method": "ping" }))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::FORBIDDEN
        );

        let task_client = client.clone();
        let task_url = url.clone();
        let session_id = session.id;
        let call = tokio::spawn(async move {
            task_client
                .post(&task_url)
                .header("accept", "application/json, text/event-stream")
                .header(mcp_gateway::SESSION_HEADER, session_id.to_string())
                .json(&json!({
                    "jsonrpc": "2.0", "id": "slow", "method": "tools/call",
                    "params": { "name": "fixture__slow" }
                }))
                .send()
                .await
                .unwrap()
        });
        tokio::time::timeout(Duration::from_secs(2), async {
            while upstream.calls() < 5 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let cancelled = client
            .post(&url)
            .header("accept", "application/json, text/event-stream")
            .header(mcp_gateway::SESSION_HEADER, session.id.to_string())
            .json(&json!({
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": { "requestId": "slow" },
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(cancelled.status(), StatusCode::ACCEPTED);

        let response = call.await.unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        assert!(response.bytes().await.unwrap().is_empty());

        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if upstream
                    .messages
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|m| m["method"] == "notifications/cancelled")
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(upstream.calls(), 5);
    }
}
