use super::*;
use axum::body::to_bytes;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Fake {
    principal: String,
    allowed: bool,
    calls: AtomicUsize,
    slow: bool,
    slow_discovery: bool,
    discoveries: AtomicUsize,
    dropped: Arc<AtomicBool>,
}

impl Fake {
    fn new() -> Self {
        Self {
            principal: "a".into(),
            allowed: true,
            calls: AtomicUsize::new(0),
            slow: false,
            slow_discovery: false,
            discoveries: AtomicUsize::new(0),
            dropped: Arc::new(AtomicBool::new(false)),
        }
    }
}

struct Dropped(Arc<AtomicBool>);

impl Drop for Dropped {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[async_trait]
impl Backend for Fake {
    async fn authorize(&self, session: Uuid) -> Result<String, StatusCode> {
        if !self.allowed {
            return Err(StatusCode::FORBIDDEN);
        }
        Ok(format!("{}:{session}", self.principal))
    }

    async fn list_tools(&self, _: Uuid) -> Result<Vec<ToolDefinition>, StatusCode> {
        self.discoveries.fetch_add(1, Ordering::SeqCst);
        if self.slow_discovery {
            let _dropped = Dropped(self.dropped.clone());
            std::future::pending::<()>().await;
        }
        Ok(vec![ToolDefinition {
            action_contract: None,
            name: "read".into(),
            description: "Read a document".into(),
            parameters: json!({ "type": "object" }),
            safe_to_memoize: false,
            memoization_policy: None,
        }])
    }

    async fn call_tool(&self, session: Uuid, call: Call) -> Result<Value, StatusCode> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let _dropped = Dropped(self.dropped.clone());
        if self.slow {
            tokio::time::sleep(Duration::from_secs(60)).await;
        }
        Ok(json!({
            "session_id": session,
            "cache_hit": false,
            "memoization_enabled": call.safe_to_memoize,
            "result": {
                "tool_call_id": "event-call",
                "role": "tool",
                "name": call.name,
                "content": json!({
                    "content": [{ "type": "text", "text": "document" }],
                    "structuredContent": { "answer": 42 },
                }).to_string(),
                "is_error": false,
            },
        }))
    }
}

fn headers(session: Uuid) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        "accept",
        "application/json, text/event-stream".parse().unwrap(),
    );
    headers.insert("content-type", "application/json".parse().unwrap());
    headers.insert(SESSION_HEADER, session.to_string().parse().unwrap());
    headers
}

async fn body(response: Response) -> Value {
    let bytes = to_bytes(response.into_body(), MAX_BODY_BYTES)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

async fn rpc(gateway: &Gateway, fake: &Fake, headers: &HeaderMap, value: Value) -> Value {
    body(
        gateway
            .handle(fake, headers, value.to_string().as_bytes())
            .await,
    )
    .await
}

#[tokio::test]
async fn initialization_discovery_and_structured_tool_results() {
    let gateway = Gateway::default();
    let fake = Fake::new();
    let headers = headers(Uuid::new_v4());
    for requested in ["2025-03-26", "2025-06-18", "future-version"] {
        let response = rpc(
            &gateway,
            &fake,
            &headers,
            json!({
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": {
                    "protocolVersion": requested,
                    "capabilities": {},
                    "clientInfo": { "name": "test", "version": "1" },
                },
            }),
        )
        .await;
        assert_eq!(response["id"], "init");
        assert!(protocol::supported_version(
            response["result"]["protocolVersion"].as_str().unwrap()
        ));
        assert!(response["result"]["capabilities"]["tools"].is_object());
    }

    let response = rpc(
        &gateway,
        &fake,
        &headers,
        json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }),
    )
    .await;
    assert_eq!(
        response["result"]["tools"][0]["inputSchema"]["type"],
        "object"
    );

    let response = rpc(
        &gateway,
        &fake,
        &headers,
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": { "name": "read" },
        }),
    )
    .await;
    assert_eq!(response["result"]["content"][0]["text"], "document");
    assert_eq!(response["result"]["structuredContent"]["answer"], 42);
    assert_eq!(
        response["result"]["_meta"]["reincore"]["memoization_enabled"],
        false
    );
    assert_eq!(
        response["result"]["_meta"]["reincore"]["tool_call_id"],
        "event-call"
    );
    assert_eq!(fake.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn invalid_requests_never_dispatch_and_notifications_have_empty_202() {
    let gateway = Gateway::default();
    let fake = Fake::new();
    let headers = headers(Uuid::new_v4());
    for (message, code) in [
        (json!([]), -32600),
        (
            json!({ "jsonrpc": "1.0", "id": 1, "method": "ping" }),
            -32600,
        ),
        (
            json!({ "jsonrpc": "2.0", "id": null, "method": "ping" }),
            -32600,
        ),
        (
            json!({ "jsonrpc": "2.0", "id": 1, "method": "resources/list" }),
            -32601,
        ),
        (
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": { "cursor": "anything" },
            }),
            -32602,
        ),
        (
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": { "name": "unlisted" },
            }),
            -32602,
        ),
        (
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": { "name": "read", "arguments": [] },
            }),
            -32602,
        ),
    ] {
        assert_eq!(
            rpc(&gateway, &fake, &headers, message).await["error"]["code"],
            code
        );
    }
    assert_eq!(
        body(gateway.handle(&fake, &headers, b"broken").await).await["error"]["code"],
        -32700
    );

    let response = gateway
        .handle(
            &fake,
            &headers,
            br#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        )
        .await;
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    assert!(to_bytes(response.into_body(), 10).await.unwrap().is_empty());
    assert_eq!(fake.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn transport_origin_session_version_and_authorization_fail_closed() {
    let gateway = Gateway::default();
    let mut fake = Fake::new();
    let original = headers(Uuid::new_v4());
    let request = br#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
    for (name, value, status) in [
        ("origin", "https://attacker.example", StatusCode::FORBIDDEN),
        (
            "mcp-protocol-version",
            "unsupported",
            StatusCode::BAD_REQUEST,
        ),
        (
            "content-type",
            "text/plain",
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
        ),
        ("accept", "application/json", StatusCode::NOT_ACCEPTABLE),
        (SESSION_HEADER, "not-a-uuid", StatusCode::BAD_REQUEST),
    ] {
        let mut headers = original.clone();
        headers.insert(name, value.parse().unwrap());
        assert_eq!(
            gateway.handle(&fake, &headers, request).await.status(),
            status
        );
    }

    let mut missing = original.clone();
    missing.remove(SESSION_HEADER);
    assert_eq!(
        gateway.handle(&fake, &missing, request).await.status(),
        StatusCode::BAD_REQUEST
    );

    fake.allowed = false;
    assert_eq!(
        gateway.handle(&fake, &original, request).await.status(),
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        unsupported_transport(HeaderMap::new()).await.status(),
        StatusCode::METHOD_NOT_ALLOWED
    );
}

#[tokio::test]
async fn cancellation_is_bound_to_identity_session_and_request_id() {
    let gateway = Gateway::default();
    let session = Uuid::new_v4();
    let request_headers = headers(session);

    let mut fake = Fake::new();
    fake.slow = true;
    let fake = Arc::new(fake);
    let task_gateway = gateway.clone();
    let task_fake = fake.clone();
    let task_headers = request_headers.clone();
    let task = tokio::spawn(async move {
        let request = json!({
            "jsonrpc": "2.0", "id": "pending", "method": "tools/call",
            "params": { "name": "read" }
        });
        task_gateway
            .handle(
                task_fake.as_ref(),
                &task_headers,
                request.to_string().as_bytes(),
            )
            .await
    });

    tokio::time::timeout(Duration::from_secs(2), async {
        while fake.calls.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    let cancel =
        br#"{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":"pending"}}"#;
    let mut other = Fake::new();
    other.principal = "b".into();
    assert_eq!(
        gateway
            .handle(&other, &request_headers, cancel)
            .await
            .status(),
        StatusCode::ACCEPTED
    );
    assert_eq!(
        gateway
            .handle(fake.as_ref(), &headers(Uuid::new_v4()), cancel)
            .await
            .status(),
        StatusCode::ACCEPTED
    );
    assert!(!fake.dropped.load(Ordering::SeqCst));

    let duplicate = rpc(
        &gateway,
        fake.as_ref(),
        &request_headers,
        json!({
            "jsonrpc": "2.0",
            "id": "pending",
            "method": "tools/call",
            "params": { "name": "read" },
        }),
    )
    .await;
    assert_eq!(duplicate["error"]["code"], -32600);
    assert_eq!(
        gateway
            .handle(fake.as_ref(), &request_headers, cancel)
            .await
            .status(),
        StatusCode::ACCEPTED
    );

    let response = task.await.unwrap();
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    assert!(to_bytes(response.into_body(), 10).await.unwrap().is_empty());
    assert!(fake.dropped.load(Ordering::SeqCst));
    assert!(gateway.pending.lock().unwrap().is_empty());
}

#[test]
fn cache_consent_header_and_per_call_override_never_change_arguments() {
    let mut headers = headers(Uuid::new_v4());
    headers.insert("x-reincore-tool-cache", "true".parse().unwrap());

    let arguments = json!({ "path": "file.txt", "session_id": "untrusted argument" });
    let request = json!({ "name": "read", "arguments": arguments });
    let parsed = protocol::parse_call(request.clone(), &headers).unwrap();
    assert!(parsed.safe_to_memoize);
    assert_eq!(parsed.arguments, arguments);

    let mut opt_out = request.clone();
    opt_out["_meta"] = json!({ "reincore/safe_to_memoize": false });
    assert!(
        !protocol::parse_call(opt_out, &headers)
            .unwrap()
            .safe_to_memoize
    );
    headers.insert("x-reincore-tool-cache", "invalid".parse().unwrap());
    assert!(protocol::parse_call(request, &headers).is_err());
}

#[tokio::test]
async fn duplicate_headers_and_invalid_notifications_do_not_dispatch() {
    let gateway = Gateway::default();
    let fake = Fake::new();
    for name in [
        "content-type",
        "mcp-protocol-version",
        "x-reincore-tool-cache",
    ] {
        let mut request_headers = headers(Uuid::new_v4());
        request_headers.append(name, "duplicate".parse().unwrap());
        request_headers.append(name, "duplicate".parse().unwrap());
        assert_eq!(
            gateway
                .handle(
                    &fake,
                    &request_headers,
                    br#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#
                )
                .await
                .status(),
            StatusCode::BAD_REQUEST
        );
    }
    let response = gateway
        .handle(
            &fake,
            &headers(Uuid::new_v4()),
            br#"{"jsonrpc":"2.0","method":"notifications/initialized","params":[]}"#,
        )
        .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(to_bytes(response.into_body(), MAX_BODY_BYTES)
        .await
        .unwrap()
        .is_empty());
    assert_eq!(fake.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn cancellation_and_duplicate_admission_cover_discovery() {
    let gateway = Gateway::default();
    let request_headers = headers(Uuid::new_v4());
    let mut fake = Fake::new();
    fake.slow_discovery = true;
    let fake = Arc::new(fake);
    let request =
        br#"{"jsonrpc":"2.0","id":"discovery","method":"tools/call","params":{"name":"read"}}"#;
    let task = {
        let gateway = gateway.clone();
        let fake = fake.clone();
        let headers = request_headers.clone();
        tokio::spawn(async move { gateway.handle(fake.as_ref(), &headers, request).await })
    };
    tokio::time::timeout(Duration::from_secs(2), async {
        while fake.discoveries.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let duplicate = gateway
        .handle(fake.as_ref(), &request_headers, request)
        .await;
    assert_eq!(body(duplicate).await["error"]["code"], -32600);
    assert_eq!(fake.discoveries.load(Ordering::SeqCst), 1);
    let cancel = br#"{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":"discovery"}}"#;
    assert_eq!(
        gateway
            .handle(fake.as_ref(), &request_headers, cancel)
            .await
            .status(),
        StatusCode::ACCEPTED
    );
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .unwrap()
            .unwrap()
            .status(),
        StatusCode::NO_CONTENT
    );
    assert!(fake.dropped.load(Ordering::SeqCst));
    assert_eq!(fake.calls.load(Ordering::SeqCst), 0);
    assert!(gateway.pending.lock().unwrap().is_empty());
}
