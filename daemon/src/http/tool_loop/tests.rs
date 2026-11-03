use super::*;
use axum::response::Response;
use engine_core::{EngineResult, ToolDefinition, ToolExecutor, ToolResult};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

struct Shop {
    calls: AtomicUsize,
}

#[async_trait::async_trait]
impl ToolExecutor for Shop {
    async fn list_tools(&self) -> EngineResult<Vec<ToolDefinition>> {
        Ok(vec![tool("my_shop__list_orders")])
    }
    async fn execute(&self, call: &ToolCall) -> EngineResult<ToolResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult {
            tool_call_id: call.id.clone(),
            role: "tool".into(),
            name: call.function.name.clone(),
            content: "8 orders".into(),
            is_error: None,
        })
    }
}

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition {
        action_contract: None,
        name: name.into(),
        description: "test".into(),
        parameters: json!({"type": "object"}),
        safe_to_memoize: false,
        memoization_policy: None,
    }
}

fn fixture() -> (LocalDatabase, Session, Arc<Shop>, ToolExecutorRegistry) {
    let db = LocalDatabase::open_in_memory().unwrap();
    let mut session = Session::new("shop", 100).unwrap();
    session.project_path = Some(std::env::current_dir().unwrap().to_str().unwrap().into());
    db.insert_session(&session).unwrap();
    let shop = Arc::new(Shop {
        calls: AtomicUsize::new(0),
    });
    let registry = ToolExecutorRegistry::new().register(shop.clone());
    (db, session, shop, registry)
}

fn request(tools: &[&str]) -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "test",
        "stream": true,
        "messages": [{"role": "user", "content": "list my orders"}],
        "tools": tools.iter().map(|name| tool(name)).collect::<Vec<_>>(),
    }))
    .unwrap()
}

fn completion(session: &Session, calls: &[(&str, &str)], content: Option<&str>) -> Response {
    let tool_calls: Vec<_> = calls
        .iter()
        .map(|(id, name)| {
            json!({"id": id, "type": "function", "function": {"name": name, "arguments": "{}"}})
        })
        .collect();
    let mut message = json!({"role": "assistant", "content": content});
    if !tool_calls.is_empty() {
        message["tool_calls"] = json!(tool_calls);
    }
    Json(json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "test",
        "reincore_session_id": session.id,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": if tool_calls.is_empty() { "stop" } else { "tool_calls" },
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }))
    .into_response()
}

async fn body_text(response: Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

#[tokio::test]
async fn injection_offers_session_tools_without_shadowing_client_tools() {
    let (_db, session, _shop, registry) = fixture();
    let mut req = request(&["powershell"]);
    inject_mcp_tools(&registry, &session, &mut req).await;
    let names: Vec<_> = req
        .tools
        .iter()
        .flatten()
        .map(|t| t.name.as_str())
        .collect();
    assert_eq!(names, ["powershell", "my_shop__list_orders"]);

    let mut colliding = request(&["my_shop__list_orders"]);
    inject_mcp_tools(&registry, &session, &mut colliding).await;
    assert_eq!(colliding.tools.unwrap().len(), 1);
}

#[tokio::test]
async fn mcp_calls_run_in_the_engine_and_only_the_answer_reaches_the_client() {
    let (db, session, shop, registry) = fixture();
    let seen = Mutex::new(Vec::<ChatCompletionRequest>::new());
    let response = run_with_mcp_tools(
        &db,
        &registry,
        &Default::default(),
        request(&["powershell"]),
        |req| {
            let round = {
                let mut seen = seen.lock().unwrap();
                seen.push(req);
                seen.len()
            };
            let session = session.clone();
            async move {
                Ok(if round == 1 {
                    completion(
                        &session,
                        &[
                            ("call_mcp", "my_shop__list_orders"),
                            ("call_ps", "powershell"),
                        ],
                        None,
                    )
                } else {
                    completion(&session, &[], Some("You have 8 orders."))
                })
            }
        },
    )
    .await
    .unwrap();

    assert_eq!(shop.calls.load(Ordering::SeqCst), 1);
    let seen = seen.into_inner().unwrap();
    assert_eq!(seen.len(), 2);
    assert!(seen.iter().all(|req| req.stream == Some(false)));
    let followup = &seen[1];
    assert_eq!(
        followup.extra.get("reincore_session_id"),
        Some(&json!(session.id.to_string()))
    );
    let assistant = &followup.messages[1];
    let replayed: Vec<_> = assistant
        .tool_calls
        .iter()
        .flatten()
        .map(|call| call.function.name.as_str())
        .collect();
    assert_eq!(
        replayed,
        ["my_shop__list_orders"],
        "the client call is dropped"
    );
    let result = &followup.messages[2];
    assert_eq!(result.tool_call_id.as_deref(), Some("call_mcp"));
    assert!(result.full_text().contains("8 orders"));

    // Recorded under the model's call id, so request-path replay dedupes it.
    let events = db.read_events(session.id, None, None).await.unwrap();
    let call_ids: Vec<_> = events
        .iter()
        .filter(|event| event.event_type == SessionEventType::ToolCall)
        .filter_map(|event| event.payload.pointer("/tool_call/id")?.as_str())
        .collect();
    assert_eq!(call_ids, ["call_mcp"]);

    // The client asked for a stream and gets the final answer as one.
    let body = body_text(response).await;
    assert!(body.contains("You have 8 orders."));
    assert!(!body.contains("my_shop__list_orders"));
}

#[tokio::test]
async fn round_limit_hands_back_only_client_executable_calls() {
    let (db, session, _shop, registry) = fixture();
    let mut req = request(&["powershell"]);
    req.stream = Some(false);
    let rounds = AtomicUsize::new(0);
    let response = run_with_mcp_tools(&db, &registry, &Default::default(), req, |_| {
        let id = format!("call_mcp_{}", rounds.fetch_add(1, Ordering::SeqCst));
        let session = session.clone();
        async move { Ok(completion(&session, &[(&id, "my_shop__list_orders")], None)) }
    })
    .await
    .unwrap();

    // Loop protection blocks the repeated call after a few rounds; blocked
    // calls still count, so the round limit is what ends the turn.
    assert_eq!(rounds.load(Ordering::SeqCst), MAX_MCP_ROUNDS + 1);
    let body: serde_json::Value = serde_json::from_str(&body_text(response).await).unwrap();
    assert!(body.pointer("/choices/0/message/tool_calls").is_none());
    assert_eq!(
        body.pointer("/choices/0/finish_reason"),
        Some(&json!("stop"))
    );
    assert!(body
        .pointer("/choices/0/message/content")
        .and_then(|value| value.as_str())
        .is_some_and(|text| text.contains("Stopped after")));
}

#[tokio::test]
async fn client_tool_calls_pass_through_untouched() {
    let (db, session, shop, registry) = fixture();
    let mut req = request(&["powershell"]);
    req.stream = Some(false);
    let response = run_with_mcp_tools(&db, &registry, &Default::default(), req, |_| {
        let session = session.clone();
        async move { Ok(completion(&session, &[("call_ps", "powershell")], None)) }
    })
    .await
    .unwrap();

    assert_eq!(shop.calls.load(Ordering::SeqCst), 0);
    let body: serde_json::Value = serde_json::from_str(&body_text(response).await).unwrap();
    assert_eq!(
        body.pointer("/choices/0/message/tool_calls/0/function/name"),
        Some(&json!("powershell"))
    );
}

#[tokio::test]
async fn client_namespaced_mcp_calls_run_as_the_mcp_tool() {
    // Seen live: after Muse Code replays `muse.read_skill`, the model calls
    // `muse.my_shop__list_orders`, which the client cannot execute.
    let (db, session, shop, registry) = fixture();
    let seen = Mutex::new(Vec::<ChatCompletionRequest>::new());
    run_with_mcp_tools(
        &db,
        &registry,
        &Default::default(),
        request(&["powershell"]),
        |req| {
            let round = {
                let mut seen = seen.lock().unwrap();
                seen.push(req);
                seen.len()
            };
            let session = session.clone();
            async move {
                Ok(if round == 1 {
                    completion(&session, &[("call_ns", "muse.my_shop__list_orders")], None)
                } else {
                    completion(&session, &[], Some("You have 8 orders."))
                })
            }
        },
    )
    .await
    .unwrap();

    assert_eq!(shop.calls.load(Ordering::SeqCst), 1);
    let seen = seen.into_inner().unwrap();
    let replayed = seen[1].messages[1].tool_calls.as_ref().unwrap();
    assert_eq!(replayed[0].function.name, "my_shop__list_orders");
}
