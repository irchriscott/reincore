use super::*;
use axum::{
    body::Body,
    http::Request,
    routing::{get, post},
    Router,
};
use mcp_mock::McpFixture;
use tower::ServiceExt;

async fn request(
    app: &Router,
    method: &str,
    uri: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method(method)
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}
#[tokio::test]
async fn configured_mcp_actual_routes_discover_cache_mutate_and_invalidate() {
    let upstream = McpFixture::start(true).await;
    let path = std::env::current_dir().unwrap();
    let config: engine_core::mcp::McpConfig =
        serde_json::from_value(upstream.config(path.to_str().unwrap())).unwrap();
    let db = LocalDatabase::open_in_memory().unwrap();
    let mut session = Session::new("MCP acceptance", 100).unwrap();
    session.project_path = Some(path.to_str().unwrap().to_owned());
    db.insert_session(&session).unwrap();
    let mut denied = Session::new("other project", 100).unwrap();
    denied.project_path = Some(path.parent().unwrap().to_str().unwrap().to_owned());
    db.insert_session(&denied).unwrap();
    let mut state = AppState::new_with_project_path(db, Some(path.to_str().unwrap()));
    state.tool_executors = std::sync::Arc::new(config.connect(false).await.unwrap());
    let app = Router::new()
        .route("/v1/tools", get(list_tools))
        .route("/v1/tool-calls", post(tool_calls))
        .with_state(state);
    let (status, list) = request(
        &app,
        "GET",
        &format!("/v1/tools?session_id={}", session.id),
        json!(null),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(list["data"].as_array().unwrap().len(), 4);
    assert!(list["data"]
        .as_array()
        .unwrap()
        .iter()
        .any(|tool| tool["name"] == "fixture__read"));
    let read = json!({
        "session_id": session.id,
        "tool_name": "fixture__read",
        "arguments": { "query": "same" },
        "safe_to_memoize": true
    });
    let (status, first) = request(&app, "POST", "/v1/tool-calls", read.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(first["cache_hit"], false);
    let (status, second) = request(&app, "POST", "/v1/tool-calls", read.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(second["cache_hit"], true);
    assert_eq!(upstream.calls(), 1);
    // Use a second warmed key for invalidation: the loop guard intentionally
    // rejects a third identical no-progress read, even across other tool calls.
    let invalidation_read = json!({
        "session_id": session.id,
        "tool_name": "fixture__read",
        "arguments": { "query": "invalidation" },
        "safe_to_memoize": true
    });
    let (status, warm) = request(&app, "POST", "/v1/tool-calls", invalidation_read.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(warm["cache_hit"], false);
    let (status, mutation) = request(
        &app,
        "POST",
        "/v1/tool-calls",
        json!({
            "session_id": session.id,
            "tool_name": "fixture__write",
            "arguments": {},
            "safe_to_memoize": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(mutation["cache_hit"], false);
    assert_eq!(mutation["memoization_enabled"], false);
    let (status, after) = request(&app, "POST", "/v1/tool-calls", invalidation_read).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(after["cache_hit"], false);
    assert_eq!(upstream.calls(), 4);
    let mut other = read;
    other["session_id"] = json!(denied.id);
    assert_eq!(
        request(&app, "POST", "/v1/tool-calls", other).await.0,
        StatusCode::FORBIDDEN
    );
    let (status, list) = request(
        &app,
        "GET",
        &format!("/v1/tools?session_id={}", denied.id),
        json!(null),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(list["data"].as_array().unwrap().is_empty());
    assert_eq!(upstream.calls(), 4);
}

#[tokio::test]
async fn project_scoped_server_covers_multiple_paths_without_duplicate_connections() {
    let upstream = McpFixture::start(false).await;
    let path = std::env::current_dir().unwrap();
    let project = Uuid::new_v4();
    let mut value = upstream.config(path.to_str().unwrap());
    value["servers"][0]["access"] = json!({"mode":"local", "project_id":project});
    let config: engine_core::mcp::McpConfig = serde_json::from_value(value).unwrap();
    let db = LocalDatabase::open_in_memory().unwrap();
    let mut cases = Vec::new();
    for (project_id, project_path, allowed) in [
        (Some(project), Some(path.clone()), true),
        (
            Some(project),
            Some(path.parent().unwrap().to_path_buf()),
            true,
        ),
        (Some(project), None, true),
        (Some(Uuid::new_v4()), Some(path.clone()), false),
        (None, Some(path.clone()), false),
    ] {
        let mut session = Session::new("project MCP", 100).unwrap();
        session.project_id = project_id;
        session.project_path = project_path.map(|p| p.to_string_lossy().into_owned());
        db.insert_session(&session).unwrap();
        cases.push((session.id, allowed));
    }
    let mut state = AppState::new_with_project_path(db, Some(path.to_str().unwrap()));
    state.tool_executors = std::sync::Arc::new(config.connect(false).await.unwrap());
    let app = Router::new()
        .route("/v1/tools", get(list_tools))
        .route("/v1/tool-calls", post(tool_calls))
        .with_state(state);
    for (session, allowed) in cases {
        let (status, list) = request(
            &app,
            "GET",
            &format!("/v1/tools?session_id={session}"),
            json!(null),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            list["data"].as_array().unwrap().len(),
            if allowed { 4 } else { 0 }
        );
        let (status, result) = request(&app, "POST", "/v1/tool-calls", json!({
            "session_id":session, "tool_name":"fixture__read", "arguments":{}, "safe_to_memoize":true
        })).await;
        if allowed {
            assert_eq!(status, StatusCode::OK);
            assert_eq!(
                result["cache_hit"], false,
                "different paths/sessions must not share relative-path results"
            );
        } else {
            assert!(!status.is_success());
        }
    }
    assert_eq!(upstream.calls(), 3);
    assert_eq!(
        upstream
            .messages
            .lock()
            .unwrap()
            .iter()
            .filter(|v| v["method"] == "initialize")
            .count(),
        1
    );
}
