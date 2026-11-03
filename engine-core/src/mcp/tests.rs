use super::*;
use mcp_mock::McpFixture;

#[tokio::test]
async fn action_receipt_mcp_discovery_and_execution_correlate_outer_call_id() {
    use axum::{http::StatusCode, response::IntoResponse, routing::post, Json, Router};
    let app = Router::new().route("/mcp", post(|Json(request): Json<Value>| async move {
        let result = match request["method"].as_str().unwrap() {
            "initialize" => json!({"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"test","version":"1"}}),
            "notifications/initialized" => return StatusCode::ACCEPTED.into_response(),
            "tools/list" => json!({"tools":[{"name":"send","inputSchema":{"type":"object","properties":{"to":{"type":"string"}},"required":["to"]},"_meta":{"reincore/action_contract":{"action":"email.send","target_pointer":"/to","completion_template":"Email sent to {target}."}}}]}),
            "tools/call" => {
                assert_eq!(request["params"]["_meta"]["reincore/tool_call_id"], "outer-call-1");
                json!({"content":[],"structuredContent":{"reincore_action_receipts":[{
                    "version":1,"tool_call_id":request["params"]["_meta"]["reincore/tool_call_id"],
                    "action":"email.send","target":request["params"]["arguments"]["to"],"outcome":"succeeded","confirmation":"message-1"
                }]}})
            },
            method => panic!("unexpected method {method}"),
        };
        Json(json!({"jsonrpc":"2.0","id":request["id"],"result":result})).into_response()
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let config: ServerConfig = serde_json::from_value(json!({
        "name":"receipts","url":url,"allow_loopback_http":true,"timeout_ms":1000,
        "access":{"mode":"local","project_path":path()},"tools":{"send":{"read_only":false}}
    }))
    .unwrap();
    let executor = McpExecutor::connect(config, false).await.unwrap();
    let ctx = context();
    let definitions = executor.list_local_tools_in_context(&ctx).await.unwrap();
    let contract = definitions[0].action_contract.as_ref().unwrap();
    let args = json!({"to":"alice@example.com"});
    let call = ToolCall {
        id: "outer-call-1".into(),
        call_type: "function".into(),
        function: crate::FunctionCall {
            name: "receipts__send".into(),
            arguments: args.to_string(),
        },
    };
    let result = executor
        .execute_local_in_context(&call, &ctx)
        .await
        .unwrap();
    let value: Value = serde_json::from_str(&result.content).unwrap();
    let evidence = crate::action_receipts::collect(contract, &call.id, &args, &value);
    assert_eq!(
        evidence[0].outcome,
        crate::action_receipts::ActionOutcome::Succeeded
    );
    assert_eq!(evidence[0].confirmation.as_deref(), Some("message-1"));
    server.abort();
}

fn path() -> String {
    std::fs::canonicalize(std::env::current_dir().unwrap())
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned()
}

fn context() -> LocalToolCacheContext {
    LocalToolCacheContext {
        session_id: Uuid::new_v4(),
        project_path: Some(path()),
        workspace_id: None,
        project_id: None,
        user_id: None,
        client_id: None,
    }
}

fn call(name: &str) -> ToolCall {
    ToolCall {
        id: Uuid::new_v4().to_string(),
        call_type: "function".into(),
        function: crate::FunctionCall {
            name: format!("fixture__{name}"),
            arguments: "{}".into(),
        },
    }
}

async fn executor(fixture: &McpFixture) -> Arc<McpExecutor> {
    let config: McpConfig = serde_json::from_value(fixture.config(&path())).unwrap();
    Arc::new(
        McpExecutor::connect(config.servers.into_iter().next().unwrap(), false)
            .await
            .unwrap(),
    )
}

#[tokio::test]
async fn json_and_sse_discover_dispatch_and_invalidate() {
    for sse in [false, true] {
        let fixture = McpFixture::start(sse).await;
        let executor = executor(&fixture).await;
        let registry = ToolExecutorRegistry::new().register(executor.clone());
        let context = context();
        assert_eq!(
            registry
                .list_local_tools_in_context(&context)
                .await
                .unwrap()
                .len(),
            4
        );
        assert!(registry.resolve(&call("unlisted")).await.is_err());
        assert!(registry.execute(&call("read")).await.is_err());
        let before = executor
            .local_cache_policy(&call("read"), &context)
            .await
            .unwrap()
            .unwrap();
        let result = registry
            .resolve(&call("read"))
            .await
            .unwrap()
            .execute_local_in_context(&call("read"), &context)
            .await
            .unwrap();
        assert_eq!(result.is_error, Some(false));
        assert!(result.content.contains("fixture result"));
        executor
            .execute_local_in_context(&call("write"), &context)
            .await
            .unwrap();
        assert!(executor
            .local_cache_policy(&call("write"), &context)
            .await
            .unwrap()
            .is_none());
        assert_ne!(
            before,
            executor
                .local_cache_policy(&call("read"), &context)
                .await
                .unwrap()
                .unwrap()
        );
        let result = executor
            .execute_local_in_context(&call("error"), &context)
            .await
            .unwrap();
        assert_eq!(result.is_error, Some(true));
        let mut other = context.clone();
        other.project_path = Some("other".into());
        assert!(executor
            .execute_local_in_context(&call("read"), &other)
            .await
            .is_err());
        assert!(registry
            .list_local_tools_in_context(&other)
            .await
            .unwrap()
            .is_empty());
        assert_eq!(fixture.calls(), 3);
    }
}

#[tokio::test]
async fn timeout_cancels_without_retry_and_disables_reuse_after_uncertain_mutation() {
    let fixture = McpFixture::start(false).await;
    let executor = executor(&fixture).await;
    let error = executor
        .execute_local_in_context(&call("slow"), &context())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("timed out"));
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if fixture
                .messages
                .lock()
                .unwrap()
                .iter()
                .any(|v| v["method"] == "notifications/cancelled")
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(fixture.calls(), 1);
    assert!(executor
        .local_cache_policy(&call("read"), &context())
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn abort_sends_cancellation_and_invalidates_mutation() {
    let fixture = McpFixture::start(false).await;
    let executor = executor(&fixture).await;
    let cloned = executor.clone();
    let task = tokio::spawn(async move {
        cloned
            .execute_local_in_context(&call("slow"), &context())
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while fixture.calls() == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    task.abort();
    let _ = task.await;
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if fixture
                .messages
                .lock()
                .unwrap()
                .iter()
                .any(|v| v["method"] == "notifications/cancelled")
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(executor
        .local_cache_policy(&call("read"), &context())
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn cloud_scope_and_bearer_transport_are_enforced() {
    let fixture = McpFixture::start(false).await;
    let mut config = fixture.config(&path());
    let workspace = Uuid::new_v4();
    let project = Uuid::new_v4();
    let principal = Uuid::new_v4();
    config["servers"][0]["access"] = json!({
        "mode": "cloud",
        "workspace_id": workspace,
        "project_id": project,
        "principals": [principal]
    });
    // Unique variable so parallel fixtures cannot interfere.
    let env = format!("REINCORE_TEST_MCP_{}", Uuid::new_v4().simple());
    std::env::set_var(&env, "fixture-secret");
    config["servers"][0]["bearer_token_env"] = json!(env);
    let config: McpConfig = serde_json::from_value(config).unwrap();
    let registry = config.connect(true).await.unwrap();
    std::env::remove_var(&env);
    let tool = registry.resolve(&call("read")).await.unwrap();
    let c = ToolCacheContext {
        workspace_id: workspace,
        project_id: project,
        user_id: principal,
        session_id: Uuid::new_v4(),
    };
    tool.execute_in_context(&call("read"), &c).await.unwrap();
    for denied in [
        ToolCacheContext {
            workspace_id: Uuid::new_v4(),
            ..c.clone()
        },
        ToolCacheContext {
            project_id: Uuid::new_v4(),
            ..c.clone()
        },
        ToolCacheContext {
            user_id: Uuid::new_v4(),
            ..c.clone()
        },
    ] {
        assert!(tool
            .execute_in_context(&call("read"), &denied)
            .await
            .is_err());
        assert!(tool.cache_policy(&call("read"), &denied).await.is_err());
        assert!(registry
            .list_tools_in_context(&denied)
            .await
            .unwrap()
            .is_empty());
    }
    assert_eq!(fixture.calls(), 1);
    assert!(fixture
        .messages
        .lock()
        .unwrap()
        .iter()
        .all(|v| v["authorization"] == "Bearer fixture-secret"));
}

#[tokio::test]
async fn unsafe_configuration_fails_before_connection() {
    let fixture = McpFixture::start(false).await;
    let base = fixture.config(&path());
    for (field, value) in [
        ("url", json!("http://example.com/mcp")),
        ("url", json!("https://user:secret@example.com/mcp")),
        ("url", json!("https://example.com/mcp?token=secret")),
        ("timeout_ms", json!(0)),
        ("name", json!("bad__namespace")),
        (
            "access",
            json!({
                "mode": "cloud",
                "workspace_id": Uuid::new_v4(),
                "project_id": Uuid::new_v4(),
                "principals": []
            }),
        ),
    ] {
        let mut config = base.clone();
        config["servers"][0][field] = value;
        let config: McpConfig = serde_json::from_value(config).unwrap();
        assert!(config.connect(false).await.is_err());
    }
    let mut config = base;
    config["servers"][0]["tools"]["read"]["read_only"] = json!(false);
    let config: McpConfig = serde_json::from_value(config).unwrap();
    assert!(config.connect(false).await.is_err());
    assert!(fixture.messages.lock().unwrap().is_empty());
}

#[tokio::test]
async fn malformed_protocol_http_errors_and_redirects_fail_closed_without_secret_output() {
    use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::post, Router};
    let cases = [
        (
            StatusCode::OK,
            "application/json",
            json!({ "jsonrpc": "2.0", "id": 99, "result": {} }).to_string(),
        ),
        (
            StatusCode::OK,
            "application/json",
            json!({
                "jsonrpc": "2.0",
                "id": 0,
                "error": { "message": "upstream-secret" }
            })
            .to_string(),
        ),
        (
            StatusCode::OK,
            "application/json",
            json!({
                "jsonrpc": "2.0",
                "id": 0,
                "result": {
                    "protocolVersion": "unsupported",
                    "capabilities": { "tools": {} }
                }
            })
            .to_string(),
        ),
        (
            StatusCode::OK,
            "application/json",
            "upstream-secret invalid JSON".into(),
        ),
        (
            StatusCode::UNAUTHORIZED,
            "application/json",
            "upstream-secret".into(),
        ),
        (
            StatusCode::TEMPORARY_REDIRECT,
            "text/plain",
            "upstream-secret".into(),
        ),
        (
            StatusCode::OK,
            "application/json",
            "x".repeat(4 * 1024 * 1024 + 1),
        ),
    ];
    for (status, mime, body) in cases {
        let app = Router::new()
            .route(
                "/mcp",
                post(
                    |State((status, mime, body)): State<(StatusCode, &'static str, String)>| async move {
                        (
                            status,
                            [
                                ("content-type", mime),
                                ("location", "http://127.0.0.1:1/secret"),
                            ],
                            body,
                        ).into_response()
                    },
                ),
            )
            .with_state((status, mime, body));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let server: ServerConfig = serde_json::from_value(json!({
            "name": "bad",
            "url": url,
            "allow_loopback_http": true,
            "timeout_ms": 1000,
            "access": { "mode": "local", "project_path": path() },
            "tools": { "read": { "read_only": true } }
        }))
        .unwrap();
        let err = match McpExecutor::connect(server, false).await {
            Ok(_) => panic!("invalid server accepted"),
            Err(e) => e.to_string(),
        };
        assert!(!err.contains("upstream-secret"));
        assert!(!err.contains("127.0.0.1"));
        task.abort();
    }
}

#[tokio::test]
async fn server_requests_and_tool_list_changes_are_handled_safely() {
    let fixture = McpFixture::start(true).await;
    let executor = executor(&fixture).await;
    let mut changed = call("read");
    changed.function.arguments = json!({"changed":true}).to_string();
    executor
        .execute_local_in_context(&changed, &context())
        .await
        .unwrap();
    assert!(executor
        .local_cache_policy(&call("read"), &context())
        .await
        .unwrap()
        .is_none());
    assert!(executor
        .execute_local_in_context(&call("read"), &context())
        .await
        .is_err());
    let messages = fixture.messages.lock().unwrap();
    assert!(messages.iter().any(|m| m["method"] == "client/reply"
        && m["params"]["id"] == "server-ping"
        && m["params"]["result"].is_object()));
    assert!(messages.iter().any(|m| m["method"] == "client/reply"
        && m["params"]["id"] == "unsupported"
        && m["params"]["error"]["code"] == -32601));
}

#[tokio::test]
async fn identical_names_dispatch_only_inside_authorized_cloud_scope() {
    let first = McpFixture::start(false).await;
    let second = McpFixture::start(false).await;
    let workspace = Uuid::new_v4();
    let user = Uuid::new_v4();
    let a = ToolCacheContext {
        workspace_id: workspace,
        project_id: Uuid::new_v4(),
        user_id: user,
        session_id: Uuid::new_v4(),
    };
    let mut b = a.clone();
    b.project_id = Uuid::new_v4();
    let config = |fixture: &McpFixture, context: &ToolCacheContext| -> ServerConfig {
        let mut json = fixture.config(&path())["servers"][0].clone();
        json["access"] = json!({"mode":"cloud", "workspace_id":context.workspace_id, "project_id":context.project_id, "principals":[context.user_id]});
        serde_json::from_value(json).unwrap()
    };
    let registry = McpConfig {
        servers: vec![config(&first, &a), config(&second, &b)],
    }
    .connect(true)
    .await
    .unwrap();
    assert_eq!(registry.list_tools_in_context(&a).await.unwrap().len(), 4);
    assert_eq!(registry.list_tools_in_context(&b).await.unwrap().len(), 4);
    assert!(registry.resolve(&call("read")).await.is_err());
    for (context, expected) in [(&b, (0, 1)), (&a, (1, 1))] {
        let executor = registry
            .resolve_in_context(&call("read"), context)
            .await
            .unwrap();
        executor
            .execute_in_context(&call("read"), context)
            .await
            .unwrap();
        assert_eq!((first.calls(), second.calls()), expected);
        assert!(executor
            .cache_policy(&call("read"), context)
            .await
            .unwrap()
            .is_some());
    }
    for context in [
        ToolCacheContext {
            workspace_id: Uuid::new_v4(),
            ..a.clone()
        },
        ToolCacheContext {
            project_id: Uuid::new_v4(),
            ..a.clone()
        },
        ToolCacheContext {
            user_id: Uuid::new_v4(),
            ..a.clone()
        },
    ] {
        assert!(registry
            .resolve_in_context(&call("read"), &context)
            .await
            .is_err());
        assert!(registry
            .list_tools_in_context(&context)
            .await
            .unwrap()
            .is_empty());
    }
    assert_eq!((first.calls(), second.calls()), (1, 1));
    let ambiguous = ToolExecutorRegistry::new()
        .register(Arc::new(
            McpExecutor::connect(config(&first, &a), true)
                .await
                .unwrap(),
        ))
        .register(Arc::new(
            McpExecutor::connect(config(&second, &a), true)
                .await
                .unwrap(),
        ));
    assert!(ambiguous
        .resolve_in_context(&call("read"), &a)
        .await
        .is_err());
    assert!(McpConfig {
        servers: vec![config(&first, &a), config(&second, &a)]
    }
    .connect(true)
    .await
    .is_err());
}

#[tokio::test]
async fn invalid_inventory_is_rejected_before_any_connection() {
    let fixture = McpFixture::start(false).await;
    let valid = || {
        serde_json::from_value::<McpConfig>(fixture.config(&path()))
            .unwrap()
            .servers
            .remove(0)
    };
    let mut invalid = valid();
    invalid.name = "other".into();
    invalid.url = "not a URL".into();
    for second in [valid(), invalid] {
        assert!(McpConfig {
            servers: vec![valid(), second]
        }
        .connect(false)
        .await
        .is_err());
        assert!(fixture.messages.lock().unwrap().is_empty());
    }
    let file = std::env::temp_dir().join(format!("mcp-project-{}", Uuid::new_v4()));
    std::fs::write(&file, b"not a directory").unwrap();
    let mut invalid = valid();
    invalid.access = Access::Local {
        project_id: None,
        project_ids: vec![],
        project_path: Some(file.to_string_lossy().into_owned()),
    };
    assert!(invalid.validate(false).is_err());
    std::fs::remove_file(file).unwrap();
}

#[tokio::test]
async fn negotiated_headers_cr_streams_and_invalid_cursors() {
    use axum::{
        extract::State,
        http::{HeaderMap, StatusCode},
        response::IntoResponse,
        routing::post,
        Json, Router,
    };
    for cursor in [None, Some(json!("")), Some(json!(42)), Some(Value::Null)] {
        let invalid_cursor = cursor.is_some();
        let app = Router::new().route("/mcp", post(
            |State(cursor): State<Option<Value>>, headers: HeaderMap, Json(request): Json<Value>| async move {
                let method = request["method"].as_str().unwrap();
                let result = if method == "initialize" {
                    json!({"protocolVersion":"2025-06-18", "capabilities":{"tools":{}}, "serverInfo":{"name":"test","version":"1"}})
                } else {
                    assert_eq!(headers["mcp-protocol-version"], "2025-06-18");
                    assert_eq!(headers["mcp-session-id"], "test-session");
                    if method == "notifications/initialized" { return StatusCode::ACCEPTED.into_response(); }
                    let mut result = json!({"tools":[{"name":"read","inputSchema":{"type":"object"}}]});
                    if let Some(cursor) = cursor { result["nextCursor"] = cursor; }
                    result
                };
                let response = json!({"jsonrpc":"2.0", "id":request["id"], "result":result});
                ([ ("content-type", "text/event-stream"), ("mcp-session-id", "test-session") ], format!(": comment\rdata: {response}\r\r")).into_response()
            }
        )).with_state(cursor);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let config: ServerConfig = serde_json::from_value(json!({
            "name":"test", "url":url, "allow_loopback_http":true, "timeout_ms":1000,
            "access":{"mode":"local", "project_path":path()}, "tools":{"read":{"read_only":true}}
        }))
        .unwrap();
        assert_eq!(
            McpExecutor::connect(config, false).await.is_err(),
            invalid_cursor
        );
        task.abort();
    }
}

#[tokio::test]
async fn cloud_principal_lists_are_bounded_and_unique_before_connecting() {
    let fixture = McpFixture::start(false).await;
    let principal = Uuid::new_v4();
    for principals in [
        vec![principal, principal],
        (0..257).map(|_| Uuid::new_v4()).collect(),
    ] {
        let mut config = fixture.config("")["servers"][0].clone();
        config["access"] = json!({"mode":"cloud", "workspace_id":Uuid::new_v4(),
            "project_id":Uuid::new_v4(), "principals":principals});
        let mut config: ServerConfig = serde_json::from_value(config).unwrap();
        assert!(config.validate(true).is_err());
    }
    assert!(fixture.messages.lock().unwrap().is_empty());
}

#[tokio::test]
async fn local_scope_requires_exactly_one_valid_identifier() {
    let fixture = McpFixture::start(false).await;
    for access in [
        json!({"mode":"local"}),
        json!({"mode":"local","project_id":Uuid::nil()}),
        json!({"mode":"local","project_id":"invalid"}),
        json!({"mode":"local","project_id":Uuid::new_v4(),"project_path":path()}),
        json!({"mode":"local","project_path":""}),
    ] {
        let mut value = fixture.config(&path());
        value["servers"][0]["access"] = access;
        if let Ok(config) = serde_json::from_value::<McpConfig>(value) {
            assert!(config.connect(false).await.is_err());
        }
    }
    assert!(fixture.messages.lock().unwrap().is_empty());
}

#[tokio::test]
async fn local_multiple_project_grants_use_one_connection() {
    let fixture = McpFixture::start(false).await;
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let mut value = fixture.config(&path());
    value["servers"][0]["access"] = json!({"mode":"local", "project_ids":[a,b]});
    let registry = serde_json::from_value::<McpConfig>(value.clone())
        .unwrap()
        .connect(false)
        .await
        .unwrap();
    for id in [a, b] {
        let ctx = LocalToolCacheContext {
            project_id: Some(id),
            ..context()
        };
        let executor = registry.resolve(&call("read")).await.unwrap();
        executor
            .execute_local_in_context(&call("read"), &ctx)
            .await
            .unwrap();
    }
    for ids in [
        json!([]),
        json!([a, a]),
        json!([Uuid::nil()]),
        json!((0..257).map(|_| Uuid::new_v4()).collect::<Vec<_>>()),
    ] {
        value["servers"][0]["access"]["project_ids"] = ids;
        assert!(serde_json::from_value::<McpConfig>(value.clone())
            .unwrap()
            .connect(false)
            .await
            .is_err());
    }
    assert_eq!(fixture.calls(), 2);
    assert_eq!(
        fixture
            .messages
            .lock()
            .unwrap()
            .iter()
            .filter(|v| v["method"] == "initialize")
            .count(),
        1
    );
}

#[tokio::test]
async fn a_server_that_needs_sign_in_does_not_stop_the_others() {
    let fixture = McpFixture::start(false).await;
    let mut value = fixture.config(&path());
    let mut signed_out = value["servers"][0].clone();
    signed_out["name"] = json!("signed_out");
    signed_out["auth"] =
        json!({"type":"oauth","client_id":"c","token_endpoint":"http://127.0.0.1:9/token"});
    value["servers"].as_array_mut().unwrap().push(signed_out);
    let registry = serde_json::from_value::<McpConfig>(value)
        .unwrap()
        .connect(false)
        .await
        .unwrap();
    registry
        .resolve(&call("read"))
        .await
        .unwrap()
        .execute_local_in_context(&call("read"), &context())
        .await
        .unwrap();
}

#[tokio::test]
async fn local_inline_bearer_token_is_persisted_and_sent() {
    let fixture = McpFixture::start(false).await;
    let mut value = fixture.config(&path());
    value["servers"][0]["bearer_token"] = json!("local-fixture-token");
    let config: McpConfig = serde_json::from_value(value).unwrap();
    let persisted = serde_json::to_vec(&config).unwrap();
    let config: McpConfig = serde_json::from_slice(&persisted).unwrap();
    assert_eq!(
        config.servers[0].bearer_token.as_deref(),
        Some("local-fixture-token")
    );
    let registry = config.connect(false).await.unwrap();
    registry
        .resolve(&call("read"))
        .await
        .unwrap()
        .execute_local_in_context(&call("read"), &context())
        .await
        .unwrap();
    let messages = fixture.messages.lock().unwrap();
    assert!(!messages.is_empty());
    assert!(messages
        .iter()
        .all(|message| message["authorization"] == "Bearer local-fixture-token"));
}

#[tokio::test]
async fn local_inline_credentials_are_validated_before_connection() {
    let fixture = McpFixture::start(false).await;
    for token in [
        String::new(),
        " secret".into(),
        "secret\r\nInjected: value".into(),
        "secret token".into(),
        "x".repeat(16385),
    ] {
        let mut value = fixture.config(&path())["servers"][0].clone();
        value["bearer_token"] = json!(token);
        let mut config: ServerConfig = serde_json::from_value(value).unwrap();
        assert!(config.validate(false).is_err());
    }
    let mut value = fixture.config(&path())["servers"][0].clone();
    value["url"] = json!("https://mcp.example.com/mcp");
    value["bearer_token"] = json!("valid-token");
    let mut config: ServerConfig = serde_json::from_value(value).unwrap();
    config.validate(false).unwrap();
    config.bearer_token_env = Some("LEGACY_TOKEN".into());
    assert!(config.validate(false).is_err());
    config.bearer_token_env = None;
    assert!(config.validate(true).is_err());
    assert!(fixture.messages.lock().unwrap().is_empty());
}

#[tokio::test]
async fn local_discovery_lists_tools_without_scope_allowlist_or_execution() {
    for sse in [false, true] {
        let fixture = McpFixture::start(sse).await;
        let config = fixture.config(&path());
        let request: DiscoveryRequest = serde_json::from_value(json!({
            "url": config["servers"][0]["url"],
            "allow_loopback_http": true,
            "timeout_ms": 5000,
            "bearer_token": "discovery-secret"
        }))
        .unwrap();
        let tools = discover_local_tools(request).await.unwrap();
        assert!(tools.iter().any(|tool| tool.name == "read"));
        assert!(tools.iter().all(|tool| tool.input_schema.is_object()));
        assert_eq!(fixture.calls(), 0);
        let messages = fixture.messages.lock().unwrap();
        assert!(messages
            .iter()
            .any(|message| message["method"] == "tools/list"));
        assert!(messages
            .iter()
            .all(|message| message["authorization"] == "Bearer discovery-secret"));
    }
}

#[tokio::test]
async fn local_discovery_rejects_unsafe_endpoints_before_connecting() {
    let request: DiscoveryRequest = serde_json::from_value(json!({
        "url": "http://example.com/mcp", "timeout_ms": 5000,
        "bearer_token": "do-not-expose-this-secret"
    }))
    .unwrap();
    let result = discover_local_tools(request).await;
    let error = result.err().unwrap().to_string();
    assert!(!error.contains("do-not-expose-this-secret"));
    assert!(error.contains("HTTPS"));
}

#[tokio::test]
async fn cloud_discovery_rejects_unsafe_or_credentialless_inputs_before_connecting() {
    // Only public HTTPS without credentials, query, or fragment connects.
    for url in [
        "http://mcp.example.com/mcp",
        "https://127.0.0.1/mcp",
        "https://mcp.example.com/mcp?token=secret",
        "https://user:pass@mcp.example.com/mcp",
    ] {
        let request: CloudDiscoveryRequest = serde_json::from_value(json!({
            "url": url, "timeout_ms": 5000,
            "bearer_token": "cloud-discovery-secret"
        }))
        .unwrap();
        let result = discover_cloud_tools(request, None).await;
        let error = result.err().unwrap().to_string();
        assert!(!error.contains("cloud-discovery-secret"), "{url}: {error}");
    }
    // Remote endpoints require a tenant token; discovery never borrows
    // deployment environment secrets.
    let credentialless: CloudDiscoveryRequest = serde_json::from_value(json!({
        "url": "https://mcp.example.com/mcp", "timeout_ms": 5000
    }))
    .unwrap();
    assert!(discover_cloud_tools(credentialless, None).await.is_err());
    let short_timeout: CloudDiscoveryRequest = serde_json::from_value(json!({
        "url": "https://mcp.example.com/mcp", "timeout_ms": 50,
        "bearer_token": "cloud-discovery-secret"
    }))
    .unwrap();
    assert!(discover_cloud_tools(short_timeout, None).await.is_err());
    assert!(serde_json::from_value::<CloudDiscoveryRequest>(json!({
        "url": "https://mcp.example.com/mcp", "timeout_ms": 5000,
        "bearer_token_env": "DEPLOYMENT_SECRET"
    }))
    .is_err());
}

#[tokio::test]
async fn action_gate_mcp_denies_invalid_schema_and_operator_rule_before_dispatch() {
    let fixture = McpFixture::start(false).await;
    let mut config: McpConfig = serde_json::from_value(fixture.config(&path())).unwrap();
    config.servers[0].tools.get_mut("read").unwrap().action = crate::action_gate::ActionRule {
        allow: false,
        ..Default::default()
    };
    let executor = McpExecutor::connect(config.servers.remove(0), false)
        .await
        .unwrap();
    assert!(executor
        .execute_local_in_context(&call("read"), &context())
        .await
        .is_err());
    assert!(executor
        .local_cache_policy(&call("read"), &context())
        .await
        .is_err());
    let mut invalid = call("write");
    invalid.function.arguments = "[]".into();
    assert!(executor
        .execute_local_in_context(&invalid, &context())
        .await
        .is_err());
    assert_eq!(fixture.calls(), 0);
    executor
        .execute_local_in_context(&call("write"), &context())
        .await
        .unwrap();
    assert_eq!(fixture.calls(), 1);
}

#[tokio::test]
async fn discovered_tool_descriptions_carry_server_name() {
    let fixture = McpFixture::start(false).await;
    let executor = executor(&fixture).await;
    let definitions = executor
        .list_local_tools_in_context(&context())
        .await
        .unwrap();
    let read = definitions
        .iter()
        .find(|definition| definition.name == "fixture__read")
        .expect("fixture__read must be discovered");
    assert_eq!(read.description, "MCP server \"fixture\": fixture read");
}

#[test]
fn project_paths_compare_independent_of_spelling() {
    let root = std::env::temp_dir().join("reincore-project-key");
    std::fs::create_dir_all(root.join("sub")).unwrap();
    // On Windows the canonical form carries the verbatim `\\?\` prefix
    // while `display` does not, so this also covers prefixed spellings.
    let canonical = std::fs::canonicalize(&root)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let spelled =
        |suffix: &str| format!("{}{}{}", root.display(), std::path::MAIN_SEPARATOR, suffix);
    for variant in [
        canonical.clone(),
        spelled(""),
        spelled(&format!("sub{}..", std::path::MAIN_SEPARATOR)),
    ] {
        assert!(
            same_project(Some(&variant), Some(&canonical)),
            "{variant} should match {canonical}"
        );
    }
    assert!(!same_project(Some(&canonical), Some(&spelled("elsewhere"))));
    assert!(!same_project(None, Some(&canonical)));
    assert!(!same_project(Some(&canonical), None));
    std::fs::remove_dir_all(&root).ok();
}
