use super::lifecycle::*;
use super::request_group::{resolve_request_group_id, PROVIDER_HEADER};
use super::*;

use axum::http::HeaderValue;
use engine_core::cost_outcome::{OutcomeReport, OutcomeSource, OutcomeVerdict, SuccessStatus};

#[tokio::test]
async fn replayed_context_after_reconnect_does_not_create_a_session_per_tool_call() {
    let path = std::env::temp_dir().join(format!(
        "reincore-rollover-replay-{}.sqlite",
        Uuid::new_v4()
    ));
    let db = LocalDatabase::open(&path).unwrap();
    let mut root = Session::new("Implement Portfolio CRUD API Endpoints", 100).unwrap();
    root.context_tokens = 296_317;
    db.insert_session(&root).unwrap();
    db.append_event(
        root.id,
        SessionEventType::TaskDecision,
        json!({"task_id":"portfolio", "entries":[]}),
        None,
    )
    .await
    .unwrap();
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model":"test", "messages":[{"role":"user", "content":"Option 3 - Change docker postgres port mapping"}]
    })).unwrap();
    let reset = engine_core::ContextWindowReset {
        limit_tokens: 128_000,
        input_tokens_before: 296_317,
        input_tokens_after: 66_720,
        reset_tokens: 229_597,
        ..Default::default()
    };
    let (first, concurrent) = tokio::join!(
        rollover_local_session_if_needed(&db, root.clone(), &reset, &request),
        rollover_local_session_if_needed(&db, root.clone(), &reset, &request)
    );
    let (first, from) = first.unwrap();
    assert_eq!(first.id, concurrent.unwrap().0.id);
    assert_eq!(from, Some(root.id));
    assert_eq!(first.task_summary, root.task_summary);
    drop(db);
    let db = LocalDatabase::open(&path).unwrap();
    for index in 0..35 {
        let active = db.get_session_async(first.id).await.unwrap().unwrap();
        let after = 66_720 + index * 700;
        let repeated = engine_core::ContextWindowReset {
            input_tokens_after: after,
            reset_tokens: reset.input_tokens_before - after,
            ..reset.clone()
        };
        let (mut active, from) = rollover_local_session_if_needed(&db, active, &repeated, &request)
            .await
            .unwrap();
        assert_eq!(active.id, first.id);
        assert_eq!(active.status, engine_core::SessionStatus::Active);
        assert_eq!(from, None);
        // Successful accounting stores the prepared prompt, not the raw replay.
        active.context_tokens = after;
        db.insert_session(&active).unwrap();
    }
    assert_eq!(
        db.list_sessions(&crate::db::SessionFilter::default())
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        db.read_task_decisions(first.id, "portfolio")
            .await
            .unwrap()
            .len(),
        1
    );
    let mut full = db.get_session_async(first.id).await.unwrap().unwrap();
    full.context_tokens = 128_000;
    db.insert_session(&full).unwrap();
    let (next, from) = rollover_local_session_if_needed(&db, full, &reset, &request)
        .await
        .unwrap();
    assert_eq!(from, Some(first.id));
    assert_eq!(next.rollover_index, 2);
    drop(db);
    std::fs::remove_file(path).unwrap();
}

#[tokio::test]
async fn client_compaction_is_rejected_without_session_events_or_provider_calls() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct Probe {
        config: engine_core::ProviderConfig,
        calls: Arc<AtomicUsize>,
    }
    #[async_trait::async_trait]
    impl engine_core::Provider for Probe {
        fn label(&self) -> &str {
            "test"
        }
        fn config(&self) -> &engine_core::ProviderConfig {
            &self.config
        }
        async fn list_models(&self) -> engine_core::EngineResult<Vec<engine_core::ProviderModel>> {
            Ok(vec![])
        }
        async fn complete(
            &self,
            _: &ChatCompletionRequest,
        ) -> engine_core::EngineResult<engine_core::ChatCompletionResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(engine_core::EngineError::Provider(
                "maintenance must not invoke provider".into(),
            ))
        }
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let single = SingleProviderAppState::new_with_project_path(
        LocalDatabase::open_in_memory().unwrap(),
        Arc::new(Probe {
            config: Default::default(),
            calls: calls.clone(),
        }),
        None,
    );
    let multi = AppState::new_with_project_path(LocalDatabase::open_in_memory().unwrap(), None);
    let prompt = "The conversation has grown too large for the context window and must be compacted now. Your task is to create a comprehensive, detailed summary of the entire conversation. This summary will be used to compact the conversation.";
    for stream in [false, true] {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"reincore-auto", "messages":[{"role":"user","content":prompt}], "stream":stream,
            "reincore_conversation_id":"copilot-maintenance"
        }))
        .unwrap();
        let responses = [
            chat_completions_single_provider(
                State(single.clone()),
                HeaderMap::new(),
                Json(req.clone()),
            )
            .await
            .unwrap(),
            chat_completions(State(multi.clone()), HeaderMap::new(), Json(req))
                .await
                .unwrap(),
        ];
        for response in responses {
            assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
            let body = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(error["error"]["code"], "client_compaction_disabled");
        }
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    for db in [single.db, multi.db] {
        assert!(db
            .list_sessions(&crate::db::SessionFilter::default())
            .unwrap()
            .is_empty());
    }
}

#[tokio::test]
async fn client_reminder_check_is_answered_quietly_without_provider_calls() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct Probe {
        config: engine_core::ProviderConfig,
        calls: Arc<AtomicUsize>,
    }
    #[async_trait::async_trait]
    impl engine_core::Provider for Probe {
        fn label(&self) -> &str {
            "test"
        }
        fn config(&self) -> &engine_core::ProviderConfig {
            &self.config
        }
        async fn list_models(&self) -> engine_core::EngineResult<Vec<engine_core::ProviderModel>> {
            Ok(vec![])
        }
        async fn complete(
            &self,
            _: &ChatCompletionRequest,
        ) -> engine_core::EngineResult<engine_core::ChatCompletionResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(engine_core::EngineError::Provider(
                "maintenance must not invoke provider".into(),
            ))
        }
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let single = SingleProviderAppState::new_with_project_path(
        LocalDatabase::open_in_memory().unwrap(),
        Arc::new(Probe {
            config: Default::default(),
            calls: calls.clone(),
        }),
        None,
    );
    let multi = AppState::new_with_project_path(LocalDatabase::open_in_memory().unwrap(), None);
    let prompt = "You are a reminder observer for the main agent. Do not answer the user. Record exactly one decision this round by calling submit_reminder_decision. Use decision=\"remind\" only when the remind condition stated in your duty is met; otherwise use decision=\"none\".";
    let req: ChatCompletionRequest = serde_json::from_value(json!({
        "model":"reincore-auto", "messages":[{"role":"user","content":prompt}],
        "reincore_conversation_id":"muse-maintenance"
    }))
    .unwrap();
    for response in [
        chat_completions_single_provider(
            State(single.clone()),
            HeaderMap::new(),
            Json(req.clone()),
        )
        .await
        .unwrap(),
        chat_completions(State(multi.clone()), HeaderMap::new(), Json(req))
            .await
            .unwrap(),
    ] {
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let completion: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let tool_call = completion
            .pointer("/choices/0/message/tool_calls/0")
            .expect("quiet answer carries the reminder decision");
        assert_eq!(
            tool_call
                .pointer("/function/name")
                .and_then(|value| value.as_str()),
            Some("submit_reminder_decision")
        );
        assert_eq!(
            tool_call
                .pointer("/function/arguments")
                .and_then(|value| value.as_str()),
            Some("{\"decision\":\"none\"}")
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    for db in [single.db, multi.db] {
        assert!(db
            .list_sessions(&crate::db::SessionFilter::default())
            .unwrap()
            .is_empty());
    }
}

#[tokio::test]
async fn client_feature_tip_is_answered_quietly_without_provider_calls() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct Probe {
        config: engine_core::ProviderConfig,
        calls: Arc<AtomicUsize>,
    }
    #[async_trait::async_trait]
    impl engine_core::Provider for Probe {
        fn label(&self) -> &str {
            "test"
        }
        fn config(&self) -> &engine_core::ProviderConfig {
            &self.config
        }
        async fn list_models(&self) -> engine_core::EngineResult<Vec<engine_core::ProviderModel>> {
            Ok(vec![])
        }
        async fn complete(
            &self,
            _: &ChatCompletionRequest,
        ) -> engine_core::EngineResult<engine_core::ChatCompletionResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(engine_core::EngineError::Provider(
                "maintenance must not invoke provider".into(),
            ))
        }
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let single = SingleProviderAppState::new_with_project_path(
        LocalDatabase::open_in_memory().unwrap(),
        Arc::new(Probe {
            config: Default::default(),
            calls: calls.clone(),
        }),
        None,
    );
    let multi = AppState::new_with_project_path(LocalDatabase::open_in_memory().unwrap(), None);
    let prompt = "You are picking one optional feature tip for a user who just finished a turn and is idle. Below is a list of candidate tips, each with an id and a note about when it helps. Choose the ONE tip most helpful to this user right now, judging from the conversation so far. Reply with exactly that tip's id and nothing else. If no tip is clearly helpful, reply exactly NONE.\n\nfile-mentions: user types or pastes file paths by hand or refers to specific files";
    let req: ChatCompletionRequest = serde_json::from_value(json!({
        "model":"reincore-auto", "messages":[{"role":"user","content":prompt}],
        "reincore_conversation_id":"muse-maintenance"
    }))
    .unwrap();
    for response in [
        chat_completions_single_provider(
            State(single.clone()),
            HeaderMap::new(),
            Json(req.clone()),
        )
        .await
        .unwrap(),
        chat_completions(State(multi.clone()), HeaderMap::new(), Json(req))
            .await
            .unwrap(),
    ] {
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let completion: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            completion
                .pointer("/choices/0/message/content")
                .and_then(|value| value.as_str()),
            Some("NONE")
        );
        assert!(completion
            .pointer("/choices/0/message/tool_calls")
            .map_or(true, |value| value.is_null()));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    for db in [single.db, multi.db] {
        assert!(db
            .list_sessions(&crate::db::SessionFilter::default())
            .unwrap()
            .is_empty());
    }
}

#[test]
fn provider_rate_limit_headers_are_forwarded_without_unrelated_headers() {
    let headers = engine_core::ProviderResponseHeaders::from([
        (
            "x-ratelimit-remaining-requests".to_string(),
            "17".to_string(),
        ),
        ("retry-after".to_string(), "2".to_string()),
        ("set-cookie".to_string(), "secret=value".to_string()),
    ]);
    let response = with_provider_rate_limit_headers(StatusCode::OK.into_response(), &headers);

    assert_eq!(
        response
            .headers()
            .get("x-ratelimit-remaining-requests")
            .and_then(|value| value.to_str().ok()),
        Some("17")
    );
    assert_eq!(
        response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok()),
        Some("2")
    );
    assert!(response.headers().get("set-cookie").is_none());
}

#[tokio::test]
async fn a_new_feature_reaches_the_provider_without_previous_task_context() {
    use std::sync::{Arc, Mutex};
    struct Capture {
        config: engine_core::ProviderConfig,
        requests: Arc<Mutex<Vec<ChatCompletionRequest>>>,
    }
    #[async_trait::async_trait]
    impl engine_core::Provider for Capture {
        fn label(&self) -> &str {
            "test"
        }
        fn config(&self) -> &engine_core::ProviderConfig {
            &self.config
        }
        async fn list_models(&self) -> engine_core::EngineResult<Vec<engine_core::ProviderModel>> {
            Ok(vec![])
        }
        async fn complete(
            &self,
            request: &ChatCompletionRequest,
        ) -> engine_core::EngineResult<engine_core::ChatCompletionResponse> {
            self.requests.lock().unwrap().push(request.clone());
            Ok(engine_core::ChatCompletionResponse {
                id: "new-feature-response".into(), model: request.model.clone(),
                message: serde_json::from_value(json!({"role":"assistant","content":"I will implement portfolio event sourcing."})).unwrap(),
                usage: None, raw: json!({"choices":[{"finish_reason":"stop"}]}),
                response_headers: Default::default(),
            })
        }
    }
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("Investigate pip", 1000).unwrap();
    db.insert_session(&session).unwrap();
    let old_request: ChatCompletionRequest = serde_json::from_value(json!({
        "model":"gpt-4o-mini", "messages":[{"role":"user","content":"Investigate the pip python.exe error"}]
    })).unwrap();
    let old_task = engine_core::resolve_task(Some(session.id), "old-request", &old_request, &[]);
    db.append_event(session.id, SessionEventType::LlmCall, json!({
        "task_id":old_task.task_id, "task_name":old_task.task_name, "request_group_id":"old-request",
        "request_messages":old_request.messages,
        "response_message":{"role":"assistant","content":"Completed the pip fix"}
    }), None).await.unwrap();
    engine_core::continuation::prepare(
        &db,
        Some(session.id),
        &old_task,
        "old-request",
        &mut old_request.clone(),
    )
    .await
    .unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let state = SingleProviderAppState::new_with_project_path(
        db,
        Arc::new(Capture {
            config: engine_core::ProviderConfig {
                label: "test".into(),
                standard_model: Some("gpt-4o-mini".into()),
                ..Default::default()
            },
            requests: requests.clone(),
        }),
        None,
    );
    let feature = "Next feature: Implement Portfolio Event Sourcing and History Reconstruction";
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model":"gpt-4o-mini", "stream":false,
        "reincore_session_id":session.id.to_string(), "reincore_request_group_id":"new-request",
        "messages":[
            {"role":"system","content":"You are a coding agent"},
            {"role":"user","content":"<conversation-summary>Investigate pip</conversation-summary>"},
            {"role":"assistant","content":"The pip Python installation needs repair"},
            {"role":"user","content":feature}
        ]
    })).unwrap();
    let response = chat_completions_single_provider(State(state), HeaderMap::new(), Json(request))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let captured = requests.lock().unwrap();
    let dispatched = captured
        .iter()
        .find(|request| {
            request
                .messages
                .iter()
                .any(|message| message.content_str() == Some(feature))
        })
        .expect("new feature must reach provider");
    let transcript = serde_json::to_string(&dispatched.messages).unwrap();
    assert!(
        !transcript.contains("pip"),
        "old conversation/continuation leaked: {transcript}"
    );
    assert!(transcript.contains(feature));
}

fn provider_registry_for_header_tests() -> engine_core::ProviderRegistry {
    struct Named {
        config: engine_core::ProviderConfig,
    }
    #[async_trait::async_trait]
    impl engine_core::Provider for Named {
        fn label(&self) -> &str {
            &self.config.label
        }
        fn config(&self) -> &engine_core::ProviderConfig {
            &self.config
        }
        async fn list_models(&self) -> engine_core::EngineResult<Vec<engine_core::ProviderModel>> {
            Ok(vec![])
        }
        async fn complete(
            &self,
            _: &ChatCompletionRequest,
        ) -> engine_core::EngineResult<engine_core::ChatCompletionResponse> {
            Err(engine_core::EngineError::Provider(
                "header test must not call providers".into(),
            ))
        }
    }
    fn named(label: &str, model: &str) -> Named {
        Named {
            config: engine_core::ProviderConfig {
                label: label.to_string(),
                standard_model: Some(model.to_string()),
                ..Default::default()
            },
        }
    }
    engine_core::ProviderRegistry::new()
        .register(std::sync::Arc::new(named("ollama", "kimi-k3:cloud")))
        .register(std::sync::Arc::new(named("meta", "muse-spark-1.3")))
}

#[test]
fn header_is_required_only_for_multi_provider_mode() {
    let headers = HeaderMap::new();
    let multi = provider_registry_for_header_tests();
    assert_eq!(
        provider_override_from_headers(&headers, &multi, "unknown-model"),
        Err(StatusCode::BAD_REQUEST)
    );
    assert_eq!(
        provider_override_from_headers(
            &headers,
            &engine_core::ProviderRegistry::new(),
            "unknown-model"
        ),
        Ok(None)
    );
}

#[test]
fn empty_header_is_rejected_in_multi_provider_mode() {
    let mut headers = HeaderMap::new();
    headers.insert(PROVIDER_HEADER, HeaderValue::from_static("   "));
    let multi = provider_registry_for_header_tests();
    assert_eq!(
        provider_override_from_headers(&headers, &multi, "unknown-model"),
        Err(StatusCode::BAD_REQUEST)
    );
}

#[test]
fn provider_label_is_trimmed_and_normalized() {
    let mut headers = HeaderMap::new();
    headers.insert(PROVIDER_HEADER, HeaderValue::from_static("  OpenAI  "));
    let multi = provider_registry_for_header_tests();
    assert_eq!(
        provider_override_from_headers(&headers, &multi, "unknown-model"),
        Ok(Some("openai".to_string()))
    );
}

#[test]
fn reincore_auto_never_requires_a_header() {
    // `reincore-auto` is the documented automatic routing handle: it routes
    // to the default provider even in multi-provider mode.
    let headers = HeaderMap::new();
    let multi = provider_registry_for_header_tests();
    assert_eq!(
        provider_override_from_headers(&headers, &multi, "reincore-auto"),
        Ok(None)
    );
    // Unknown models still require the header in multi-provider mode.
    assert_eq!(
        provider_override_from_headers(&headers, &multi, "unknown-model"),
        Err(StatusCode::BAD_REQUEST)
    );
}

#[test]
fn owned_model_defaults_provider_without_header() {
    // Muse Code sends model ids, not provider headers: a Muse model id
    // defaults to the Meta provider even in multi-provider mode.
    let headers = HeaderMap::new();
    let multi = provider_registry_for_header_tests();
    assert_eq!(
        provider_override_from_headers(&headers, &multi, "muse-spark-1.3"),
        Ok(Some("meta".to_string()))
    );
    assert_eq!(
        provider_override_from_headers(&headers, &multi, "kimi-k3:cloud"),
        Ok(Some("ollama".to_string()))
    );
}

#[test]
fn explicit_client_header_selects_the_delegation_profile() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "reincore-auto",
        "messages": [],
        "reincore_delegation_profile": "codex"
    }))
    .unwrap();
    let mut headers = HeaderMap::new();
    headers.insert(
        engine_core::DELEGATION_PROFILE_HEADER,
        HeaderValue::from_static("opencode"),
    );

    assert_eq!(
        delegation_profile(Some(&headers), &request).as_deref(),
        Some("opencode")
    );
}

#[tokio::test]
async fn task_endpoint_returns_separated_durable_task_groups() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("multi-task session", 1_000).unwrap();
    db.insert_session(&session).unwrap();

    for (task_id, task_name, task_type, group_id, prompt, input, output, cost) in [
        (
            "task-research",
            "Research regional suppliers",
            "research",
            "group-1",
            "Research regional suppliers",
            120,
            25,
            3,
        ),
        (
            "task-plan",
            "Plan the launch workshop",
            "planning",
            "group-2",
            "Plan the launch workshop",
            80,
            20,
            2,
        ),
        (
            "task-research",
            "Research regional suppliers",
            "research",
            "group-3",
            "Compare their delivery terms",
            60,
            15,
            1,
        ),
    ] {
        db.append_event(
            session.id,
            SessionEventType::LlmCall,
            json!({
                "request_messages": [{"role": "user", "content": prompt}],
                "response_message": {"role": "assistant", "content": "done"},
                "request_group_id": group_id,
                "task_id": task_id,
                "task_name": task_name,
                "task_type": task_type,
                "input_tokens": input,
                "output_tokens": output,
            }),
            Some(cost),
        )
        .await
        .unwrap();
    }

    let response = get_session_tasks(
        State(AppState::new_with_project_path(db, None)),
        Path(session.id),
    )
    .await
    .unwrap();
    let tasks = response.0;

    assert_eq!(tasks.len(), 2);
    assert_eq!(tasks[0].task_id, "task-research");
    assert_eq!(tasks[0].request_count, 2);
    assert_eq!(tasks[0].input_tokens, 180);
    assert_eq!(tasks[0].output_tokens, 40);
    assert_eq!(tasks[0].cost_cents, 4);
    assert_eq!(tasks[1].task_id, "task-plan");
    assert_eq!(tasks[1].request_count, 1);
}

#[tokio::test]
async fn timeline_and_event_page_keep_raw_events_behind_a_sequence_cursor() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("timeline session", 1_000).unwrap();
    db.insert_session(&session).unwrap();
    for seq in 1..=3 {
        db.append_event(
            session.id,
            SessionEventType::LlmCall,
            json!({
                "request_messages": [{"role": "user", "content": "Build the timeline"}],
                "response_message": {"role": "assistant", "content": "done"},
                "request_group_id": "group-1",
                "task_id": "task-1",
                "task_name": "Build the timeline",
                "task_type": "feature",
                "input_tokens": seq,
                "output_tokens": 1,
            }),
            None,
        )
        .await
        .unwrap();
    }
    let state = AppState::new_with_project_path(db, None);

    let timeline = get_session_timeline(State(state.clone()), Path(session.id))
        .await
        .unwrap()
        .0;
    assert_eq!(timeline.total_event_count, 3);
    assert_eq!(
        timeline.request_groups[0].prompt.as_deref(),
        Some("Build the timeline")
    );

    let first_page = get_session_event_page(
        State(state.clone()),
        Path(session.id),
        axum::extract::Query(SessionEventPageQuery {
            since_seq: Some(1),
            until_seq: Some(3),
            limit: Some(2),
        }),
    )
    .await
    .unwrap()
    .0;
    assert_eq!(first_page.events.len(), 2);
    assert_eq!(first_page.next_since_seq, Some(3));
    assert!(first_page.events[0]
        .payload
        .get("request_messages")
        .is_none());
    assert_eq!(
        first_page.events[0].payload["_reincore_omitted_fields"],
        json!(["request_messages"])
    );

    let complete_event = get_session_event(State(state), Path((session.id, 1)))
        .await
        .unwrap()
        .0;
    assert!(complete_event.payload.get("request_messages").is_some());
}

#[tokio::test]
async fn completing_session_leaves_success_unknown() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let mut session = Session::new("repeatable task", 1_000).unwrap();
    session.spend_cents = 200;
    db.insert_session(&session).unwrap();
    db.append_event(
        session.id,
        SessionEventType::LlmCall,
        json!({
            "request_messages": [{"role": "user", "content": "Do the task"}],
            "response_message": {"role": "assistant", "content": "done"},
            "request_group_id": "group-1",
            "task_id": "task-1",
            "task_name": "Repeatable task",
            "task_type": "general"
        }),
        Some(200),
    )
    .await
    .unwrap();

    let state = AppState::new_with_project_path(db, None);
    let response = complete_session(
        State(state.clone()),
        Path(session.id),
        Json(CompleteSessionRequest {
            reason: None,
            outcome_report: None,
        }),
    )
    .await
    .unwrap();

    assert_eq!(response.0.status, SessionStatus::Completed);
    let outcome = response.0.cost_outcome.expect("cost outcome");
    assert!(outcome.clean_completion);
    assert_eq!(outcome.completion_quality_bps, None);
    assert_eq!(outcome.cost_efficiency_score, None);
    assert_eq!(
        outcome.success_status,
        engine_core::cost_outcome::SuccessStatus::Unknown
    );
    let events = state.db.read_events(session.id, None, None).await.unwrap();
    assert_eq!(
        events.last().unwrap().event_type,
        SessionEventType::SessionCompleted
    );
}

#[tokio::test]
async fn completion_records_interventions_without_claiming_quality() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let mut session = Session::new("recovered task", 1_000).unwrap();
    session.spend_cents = 200;
    db.insert_session(&session).unwrap();
    for (repetition_count, message) in [(3, "try another approach"), (0, "")] {
        db.append_event(
            session.id,
            SessionEventType::LoopIntervention,
            json!({
                "repeated_hash": "hash",
                "repetition_count": repetition_count,
                "injected_message": message,
                "hard_stop": false
            }),
            None,
        )
        .await
        .unwrap();
    }

    let state = AppState::new_with_project_path(db, None);
    let response = complete_session(
        State(state),
        Path(session.id),
        Json(CompleteSessionRequest {
            reason: None,
            outcome_report: None,
        }),
    )
    .await
    .unwrap();
    let outcome = response.0.cost_outcome.expect("cost outcome");
    assert_eq!(outcome.loop_intervention_count, 1);
    assert!(!outcome.clean_completion);
    assert_eq!(outcome.completion_quality_bps, None);
    assert_eq!(outcome.cost_efficiency_score, None);
}

#[tokio::test]
async fn accepted_batch_submission_is_immediately_visible_as_a_task() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("batch task session", 1_000).unwrap();
    db.insert_session(&session).unwrap();
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "reincore-auto",
        "messages": [{"role": "user", "content": "Analyze the quarterly survey"}],
        "reincore_task_id": "task-survey",
        "reincore_task_name": "Analyze the quarterly survey",
        "reincore_task_type": "analysis"
    }))
    .unwrap();
    let task = engine_core::resolve_task(Some(session.id), "group-batch", &request, &[]);

    record_batch_submission(
        &db,
        Some(session.id),
        "group-batch",
        &task,
        &request,
        "batch-model",
        engine_core::ModelTier::Pro,
        "prefix-hash",
        "batch-123",
        2,
        4,
        None,
        engine_core::ContextAdjustment::default(),
    )
    .await
    .unwrap();

    let events = db.read_events(session.id, None, None).await.unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].payload["batch_submitted"], true);
    assert_eq!(events[0].payload["batch_id"], "batch-123");

    let response = get_session_tasks(
        State(AppState::new_with_project_path(db, None)),
        Path(session.id),
    )
    .await
    .unwrap();
    assert_eq!(response.0.len(), 1);
    assert_eq!(response.0[0].task_id, "task-survey");
    assert_eq!(response.0[0].name, "Analyze the quarterly survey");
    assert_eq!(response.0[0].request_count, 1);
    assert!(!response.0[0].completed);
    assert_eq!(response.0[0].cost_efficiency_score, None);
}

#[tokio::test]
async fn external_tool_execution_is_logged_once() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("external tool logging", 1_000).unwrap();
    db.insert_session(&session).unwrap();
    db.append_event(
        session.id,
        SessionEventType::LlmCall,
        json!({
            "request_group_id": "original-request-group",
            "request_messages": [{"role": "user", "content": "Read the file"}],
            "response_message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"a.rs\"}"}
                }]
            }
        }),
        None,
    )
    .await
    .unwrap();
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "reincore-auto",
        "messages": [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"a.rs\"}"}
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "file contents"
            }
        ]
    }))
    .unwrap();

    let resolved_group = resolve_request_group_id(&db, None, &request, Some(session.id))
        .await
        .unwrap();
    assert_eq!(resolved_group, "original-request-group");

    record_external_tool_executions(
        &db,
        session.id,
        &request.messages,
        &resolved_group,
        None,
        true,
    )
    .await
    .unwrap();
    record_external_tool_executions(
        &db,
        session.id,
        &request.messages,
        &resolved_group,
        None,
        true,
    )
    .await
    .unwrap();

    let events = db.read_events(session.id, None, None).await.unwrap();
    assert_eq!(events.len(), 3);
    assert_eq!(events[1].event_type, SessionEventType::ToolCall);
    assert_eq!(events[2].event_type, SessionEventType::ToolResult);
    assert_eq!(events[1].payload["execution_source"], "external_client");
    assert_eq!(events[1].payload["origin_llm_call_seq"], 1);
    assert_eq!(events[2].payload["origin_llm_call_seq"], 1);
    assert_eq!(
        events[1].payload["request_group_id"],
        "original-request-group"
    );
    assert_eq!(events[2].payload["result"]["tool_call_id"], "call-1");
    assert_eq!(events[2].payload["result"]["name"], "read_file");
    assert_eq!(events[2].payload["result"]["content"], "file contents");
}

#[tokio::test]
async fn completed_run_subagent_creates_one_correlated_child_session() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let mut session = Session::new_tracked(
        "parent task",
        1_000,
        Some("demo"),
        Some("/workspace/demo"),
        Some("vscode"),
        Some("parent-external"),
    )
    .unwrap();
    session.project_id = Some(Uuid::new_v4());
    db.insert_session(&session).unwrap();
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "reincore-auto",
        "messages": [
            {"role": "user", "content": "Review filters"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call-subagent",
                    "type": "function",
                    "function": {
                        "name": "runSubagent",
                        "arguments": "{\"description\":\"Explore filter architecture\",\"prompt\":\"Inspect filters\"}"
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call-subagent",
                "content": "report path"
            }
        ]
    }))
    .unwrap();

    record_external_tool_executions(
        &db,
        session.id,
        &request.messages,
        "spawn-group",
        None,
        true,
    )
    .await
    .unwrap();
    record_external_tool_executions(
        &db,
        session.id,
        &request.messages,
        "spawn-group",
        None,
        true,
    )
    .await
    .unwrap();

    let children = db.list_subagent_sessions("parent-external").unwrap();
    assert_eq!(children.len(), 1);
    let child = &children[0];
    assert_eq!(child.status, SessionStatus::Completed);
    assert_eq!(child.task_summary, "Explore filter architecture");
    assert_eq!(
        child.subagent_spawn_tool_call_id.as_deref(),
        Some("call-subagent")
    );
    assert_eq!(
        child.subagent_spawn_request_group_id.as_deref(),
        Some("spawn-group")
    );
    let child_events = db.read_events(child.id, None, None).await.unwrap();
    assert_eq!(child_events.len(), 2);
    assert_eq!(
        child_events[0].payload["execution_source"],
        "external_client_opaque"
    );

    let response = list_subagent_sessions(
        State(AppState::new_with_project_path(db, None)),
        Path(session.id),
    )
    .await
    .unwrap();
    let endpoint_json = serde_json::to_value(response.0).unwrap();
    assert_eq!(
        endpoint_json[0]["subagent_spawn_tool_call_id"],
        "call-subagent"
    );
    assert_eq!(
        endpoint_json[0]["subagent_spawn_request_group_id"],
        "spawn-group"
    );
}

#[tokio::test]
async fn disabled_subagent_tracking_keeps_delegation_in_parent_events() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let parent = Session::new_tracked(
        "parent task",
        1_000,
        Some("demo"),
        Some("/workspace/demo"),
        Some("vscode"),
        Some("parent-external"),
    )
    .unwrap();
    db.insert_session(&parent).unwrap();
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "reincore-auto",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-subagent",
                    "type": "function",
                    "function": {
                        "name": "runSubagent",
                        "arguments": "{\"prompt\":\"Inspect filters\"}"
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call-subagent",
                "content": "done"
            }
        ]
    }))
    .unwrap();

    record_external_tool_executions(
        &db,
        parent.id,
        &request.messages,
        "spawn-group",
        Some("vscode"),
        false,
    )
    .await
    .unwrap();

    assert!(db
        .list_subagent_sessions("parent-external")
        .unwrap()
        .is_empty());
    let events = db.read_events(parent.id, None, None).await.unwrap();
    assert!(events
        .iter()
        .any(|event| event.event_type == SessionEventType::ToolCall));
    assert!(events
        .iter()
        .any(|event| event.event_type == SessionEventType::ToolResult));
}

#[tokio::test]
async fn directly_streamed_subagent_is_not_duplicated_by_reconciliation() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let parent = Session::new_tracked(
        "parent task",
        1_000,
        Some("demo"),
        Some("/workspace/demo"),
        Some("vscode"),
        Some("parent-external"),
    )
    .unwrap();
    db.insert_session(&parent).unwrap();

    let mut direct_child = Session::new_tracked(
        "Inspect filters",
        0,
        Some("demo"),
        Some("/workspace/demo"),
        Some("vscode"),
        Some(engine_core::external_subagent_session_id(
            "parent-external",
            "call-subagent",
        )),
    )
    .unwrap();
    direct_child.is_subagent = true;
    direct_child.subagent_parent_session_id = Some("parent-external".to_string());
    direct_child.subagent_spawn_tool_call_id = Some("call-subagent".to_string());
    db.insert_session(&direct_child).unwrap();

    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "reincore-auto",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-subagent",
                    "type": "function",
                    "function": {
                        "name": "runSubagent",
                        "arguments": "{\"prompt\":\"Inspect filters\"}"
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call-subagent",
                "content": "done"
            }
        ]
    }))
    .unwrap();

    record_external_tool_executions(
        &db,
        parent.id,
        &request.messages,
        "spawn-group",
        Some("vscode"),
        true,
    )
    .await
    .unwrap();

    let children = db.list_subagent_sessions("parent-external").unwrap();
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].id, direct_child.id);
    assert_eq!(children[0].status, SessionStatus::Active);
}

#[tokio::test]
async fn profile_scoped_delegation_tools_create_child_sessions() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new_tracked(
        "cross-client parent",
        1_000,
        Some("demo"),
        Some("/workspace/demo"),
        Some("agent-client"),
        Some("cross-client-parent"),
    )
    .unwrap();
    db.insert_session(&session).unwrap();

    for (profile, name, call_id, arguments) in [
        (
            "codex",
            "spawn_agent",
            "codex-spawn",
            r#"{"message":"Inspect the Codex path"}"#,
        ),
        (
            "claude",
            "Agent",
            "claude-agent",
            r#"{"prompt":"Inspect the Claude path"}"#,
        ),
        (
            "github-copilot-cli",
            "Task",
            "copilot-task",
            r#"{"description":"Inspect the Copilot path"}"#,
        ),
        (
            "opencode",
            "subagent",
            "opencode-subagent",
            r#"{"task":"Inspect the OpenCode path"}"#,
        ),
        (
            "openclaw",
            "sessions_spawn",
            "openclaw-spawn",
            r#"{"message":"Inspect the OpenClaw path"}"#,
        ),
        (
            "cursor",
            "Task",
            "cursor-task",
            r#"{"instructions":"Inspect the Cursor path"}"#,
        ),
        (
            "gemini-cli",
            "codebase_investigator",
            "gemini-investigator",
            r#"{"prompt":"Inspect the Gemini path"}"#,
        ),
    ] {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "reincore-auto",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments}
                    }]
                },
                {"role": "tool", "tool_call_id": call_id, "content": "done"}
            ]
        }))
        .unwrap();
        record_external_tool_executions(
            &db,
            session.id,
            &request.messages,
            &format!("{profile}-group"),
            Some(profile),
            true,
        )
        .await
        .unwrap();
    }

    let children = db.list_subagent_sessions("cross-client-parent").unwrap();
    assert_eq!(children.len(), 7);
    for expected in [
        "Inspect the Codex path",
        "Inspect the Claude path",
        "Inspect the Copilot path",
        "Inspect the OpenCode path",
        "Inspect the OpenClaw path",
        "Inspect the Cursor path",
        "Inspect the Gemini path",
    ] {
        assert!(
            children.iter().any(|child| child.task_summary == expected),
            "missing child session for {expected}"
        );
    }
}

#[tokio::test]
async fn explicit_session_creation_persists_subagent_correlation() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let state = AppState::new_with_project_path(db, None);
    let request: CreateSessionRequest = serde_json::from_value(json!({
        "task_summary": "Explicit delegated task",
        "budget_cents": 250,
        "client_id": "custom-agent",
        "external_session_id": "child-external",
        "is_subagent": true,
        "subagent_parent_session_id": "parent-external",
        "subagent_spawn_tool_call_id": "spawn-42",
        "subagent_spawn_request_group_id": "group-42"
    }))
    .unwrap();

    let response = create_session(State(state.clone()), Json(request))
        .await
        .unwrap();
    let response = serde_json::to_value(response.0).unwrap();
    assert_eq!(response["is_subagent"], true);
    assert_eq!(response["subagent_parent_session_id"], "parent-external");
    assert_eq!(response["subagent_spawn_tool_call_id"], "spawn-42");
    assert_eq!(response["subagent_spawn_request_group_id"], "group-42");

    let session_id = Uuid::parse_str(response["session_id"].as_str().unwrap()).unwrap();
    let stored = state.db.get_session(session_id).unwrap().unwrap();
    assert!(stored.is_subagent);
    assert_eq!(
        stored.subagent_parent_session_id.as_deref(),
        Some("parent-external")
    );
    assert_eq!(
        stored.subagent_spawn_tool_call_id.as_deref(),
        Some("spawn-42")
    );
    assert_eq!(
        stored.subagent_spawn_request_group_id.as_deref(),
        Some("group-42")
    );
}

#[tokio::test]
async fn historical_external_tool_executions_are_backfilled() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("historical external tool", 1_000).unwrap();
    db.insert_session(&session).unwrap();
    db.append_event(
        session.id,
        SessionEventType::LlmCall,
        json!({
            "request_group_id": "historical-group",
            "request_messages": [{"role": "user", "content": "Search"}],
            "response_message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "historical-call",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"}
                }]
            }
        }),
        None,
    )
    .await
    .unwrap();
    db.append_event(
        session.id,
        SessionEventType::LlmCall,
        json!({
            "request_group_id": "continuation-group",
            "request_messages": [
                {"role": "user", "content": "Search"},
                {"role": "tool", "tool_call_id": "historical-call", "content": "match found"}
            ],
            "response_message": {"role": "assistant", "content": "Done"}
        }),
        None,
    )
    .await
    .unwrap();

    backfill_external_tool_executions(&db).await;
    backfill_external_tool_executions(&db).await;

    let events = db.read_events(session.id, None, None).await.unwrap();
    assert_eq!(events.len(), 4);
    assert_eq!(events[2].event_type, SessionEventType::ToolCall);
    assert_eq!(events[3].event_type, SessionEventType::ToolResult);
    assert_eq!(events[2].payload["request_group_id"], "historical-group");
    assert_eq!(events[3].payload["execution_source"], "external_client");
}

#[tokio::test]
async fn project_endpoint_wraps_session_backed_projects() {
    let db = LocalDatabase::open_in_memory().unwrap();
    for (name, path) in [("alpha", "/workspace/alpha"), ("beta", "/workspace/beta")] {
        let session = Session::new_tracked(
            format!("task for {name}"),
            1_000,
            Some(name),
            Some(path),
            Some("test"),
            Some(format!("conversation-{name}")),
        )
        .unwrap();
        db.insert_session(&session).unwrap();
    }

    let state = AppState::new_with_project_path(db, None);
    let response = list_projects(State(state)).await.unwrap();
    let value = serde_json::to_value(response.0).unwrap();

    assert_eq!(value["projects"].as_array().unwrap().len(), 2);
    assert_eq!(value["projects"][0]["session_count"], 1);
}

#[tokio::test]
async fn checkpointed_external_session_is_not_duplicated() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let (project_path, _) = crate::db::normalize_project_path("/workspace/test").unwrap();
    let mut session = Session::new_tracked(
        "checkpointed task",
        1_000,
        Some("test"),
        Some(project_path),
        Some("legacy-client"),
        Some("conversation-1"),
    )
    .unwrap();
    session.status = SessionStatus::Checkpointed;
    db.insert_session(&session).unwrap();
    // The session list only shows sessions that have at least one event.
    db.append_event(
        session.id,
        SessionEventType::LlmCall,
        json!({"request_messages": [{"role": "user", "content": "Initial"}], "response_message": {"role": "assistant", "content": "ok"}}),
        None,
    )
    .await
    .unwrap();

    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "reincore-auto",
        "messages": [{"role": "user", "content": "Continue the task"}],
        "reincore_external_session_id": "conversation-1",
        "reincore_client_id": "legacy-client",
        "reincore_project_name": "test",
        "reincore_project_path": "/workspace/test"
    }))
    .unwrap();

    assert_eq!(
        resolve_local_session(&db, &request, "test", "/workspace/test")
            .await
            .unwrap_err(),
        StatusCode::CONFLICT
    );
    assert_eq!(
        db.list_sessions(&crate::db::SessionFilter::default())
            .unwrap()
            .len(),
        1
    );
}

fn conversation_request(
    conversation_id: &str,
    prompt: &str,
    extra: serde_json::Value,
) -> ChatCompletionRequest {
    let mut value = json!({
        "model": "reincore-auto",
        "messages": [{"role": "user", "content": prompt}],
        "reincore_conversation_id": conversation_id,
        "reincore_client_id": "vscode",
        "reincore_project_name": "test",
        "reincore_project_path": "/workspace/test"
    });
    if let (Some(target), Some(values)) = (value.as_object_mut(), extra.as_object()) {
        target.extend(values.clone());
    }
    serde_json::from_value(value).unwrap()
}

#[tokio::test]
async fn backend_conversation_binding_is_idempotent_and_not_prompt_derived() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let first = conversation_request("chat-one", "Fix .egg-info/", json!({}));
    let repeated = conversation_request("chat-one", "Fix .egg-info/", json!({}));
    let separate_chat = conversation_request("chat-two", "Fix .egg-info/", json!({}));

    let first = resolve_local_session(&db, &first, "test", "/workspace/test")
        .await
        .unwrap()
        .unwrap();
    let repeated = resolve_local_session(&db, &repeated, "test", "/workspace/test")
        .await
        .unwrap()
        .unwrap();
    let separate_chat = resolve_local_session(&db, &separate_chat, "test", "/workspace/test")
        .await
        .unwrap()
        .unwrap();

    assert_eq!(first.id, repeated.id);
    assert_ne!(first.id, separate_chat.id);
}

#[tokio::test]
async fn legacy_vscode_uuid_per_request_is_replaced_by_backend_binding() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let make_request = |external_id: &str| {
        serde_json::from_value(json!({
            "model": "reincore-auto",
            "messages": [{"role": "user", "content": "Implement the Rust SDK"}],
            "reincore_external_session_id": external_id,
            "reincore_client_id": "vscode",
            "reincore_project_name": "test",
            "reincore_project_path": "/workspace/test"
        }))
        .unwrap()
    };
    let first = make_request("11111111-1111-4111-8111-111111111111");
    let repeated = make_request("22222222-2222-4222-8222-222222222222");

    let first = resolve_local_session(&db, &first, "test", "/workspace/test")
        .await
        .unwrap()
        .unwrap();
    let repeated = resolve_local_session(&db, &repeated, "test", "/workspace/test")
        .await
        .unwrap()
        .unwrap();

    assert_eq!(first.id, repeated.id);
    assert!(first
        .external_session_id
        .as_deref()
        .is_some_and(|id| id.starts_with("conversation:")));
}

#[tokio::test]
async fn parallel_first_requests_resolve_to_one_backend_session() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let first = conversation_request("parallel-chat", "Implement the SDK", json!({}));
    let second = conversation_request("parallel-chat", "Implement the SDK", json!({}));

    let (first, second) = tokio::join!(
        resolve_local_session(&db, &first, "test", "/workspace/test"),
        resolve_local_session(&db, &second, "test", "/workspace/test")
    );

    assert_eq!(
        first.unwrap().unwrap().id,
        second.unwrap().unwrap().id,
        "the binding transaction must select one winner"
    );
}

#[tokio::test]
async fn tracked_subagent_conversation_is_linked_to_its_parent() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let parent_request = conversation_request("main-chat", "Implement the SDK", json!({}));
    let parent = resolve_local_session(&db, &parent_request, "test", "/workspace/test")
        .await
        .unwrap()
        .unwrap();
    let child_request = conversation_request(
        "child-chat",
        "Read these files completely",
        json!({
            "reincore_parent_conversation_id": "main-chat",
            "reincore_is_subagent": true,
            "reincore_track_subagent_sessions": true,
            "reincore_subagent_spawn_tool_call_id": "call-subagent"
        }),
    );
    let child = resolve_local_session(&db, &child_request, "test", "/workspace/test")
        .await
        .unwrap()
        .unwrap();

    assert_ne!(parent.id, child.id);
    assert!(child.is_subagent);
    assert_eq!(child.subagent_parent_session_id, parent.external_session_id);
    assert_eq!(
        child.subagent_spawn_tool_call_id.as_deref(),
        Some("call-subagent")
    );
}

#[tokio::test]
async fn untracked_subagent_conversation_is_bound_to_the_parent_session() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let parent_request = conversation_request("main-chat", "Implement the SDK", json!({}));
    let parent = resolve_local_session(&db, &parent_request, "test", "/workspace/test")
        .await
        .unwrap()
        .unwrap();
    let child_request = conversation_request(
        "child-chat",
        "Read these files completely",
        json!({
            "reincore_parent_conversation_id": "main-chat",
            "reincore_is_subagent": true,
            "reincore_track_subagent_sessions": false,
            "reincore_subagent_spawn_tool_call_id": "call-subagent"
        }),
    );

    let child = resolve_local_session(&db, &child_request, "test", "/workspace/test")
        .await
        .unwrap()
        .unwrap();
    let child_again = resolve_local_session(&db, &child_request, "test", "/workspace/test")
        .await
        .unwrap()
        .unwrap();

    assert_eq!(parent.id, child.id);
    assert_eq!(parent.id, child_again.id);
}

#[tokio::test]
async fn a_declared_subagent_cannot_become_an_orphan_session() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let request = conversation_request(
        "child-chat",
        "Read these files completely",
        json!({
            "reincore_parent_conversation_id": "missing-main-chat",
            "reincore_is_subagent": true
        }),
    );

    assert_eq!(
        resolve_local_session(&db, &request, "test", "/workspace/test")
            .await
            .unwrap_err(),
        StatusCode::CONFLICT
    );
}

#[tokio::test]
async fn streaming_response_includes_prompt_injection_findings() {
    let response = json!({
        "id": "chat-1",
        "model": "test-model",
        "choices": [{
            "message": {"role": "assistant", "content": "done"},
            "finish_reason": "stop"
        }],
        "usage": {},
        "reincore_agent_health": {
            "findings": [{
                "kind": "prompt_injection",
                "severity": "warning",
                "message": "tool result contains prompt-injection indicators"
            }],
            "should_stop": false
        }
    });

    let response = Sse::new(chat_completion_to_sse_stream(response)).into_response();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();

    assert!(body.contains("reincore_agent_health"));
    assert!(body.contains("\"kind\":\"prompt_injection\""));
}

#[tokio::test]
async fn streaming_response_preserves_opaque_provider_state() {
    let message = json!({"role":"assistant","content":"visible", "provider_state":{
        "scope":"provider/model", "message_hash":"fingerprint", "request_group_id":"group", "payload":{"thinking":"private protocol state"}
    }});
    let response = json!({"choices":[{"message":message,"finish_reason":"stop"}]});
    let response = Sse::new(chat_completion_to_sse_stream(response)).into_response();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let chunks: Vec<serde_json::Value> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    assert_eq!(chunks[0]["reincore_provider_state"], message);
    assert!(chunks.iter().all(|c| c
        .pointer("/choices/0/delta/content")
        .and_then(|v| v.as_str())
        != Some("private protocol state")));
}

#[tokio::test]
async fn responses_and_messages_streams_emit_protocol_events() {
    let responses = json!({
        "id": "resp_1", "model": "test", "status": "completed",
        "output": [{"content": [{"type": "output_text", "text": "hello"}]}],
        "usage": {}
    });
    let response = Sse::new(responses_json_to_sse_stream(responses)).into_response();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    assert!(body.contains("event: response.output_text.delta"));
    assert!(body.contains("event: response.completed"));

    let message = json!({
        "id": "msg_1", "model": "test", "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 2, "output_tokens": 1}
    });
    let response = Sse::new(messages_json_to_sse_stream(message)).into_response();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    assert!(body.contains("event: message_start"));
    assert!(body.contains("event: content_block_delta"));
    assert!(body.contains("event: message_stop"));
}

#[test]
fn wrapper_adapters_preserve_tool_loops() {
    let responses: ResponsesApiRequest = serde_json::from_value(json!({
            "model": "auto",
            "input": [
                {"role": "user", "content": "read it"},
                {"type": "function_call", "call_id": "call_1", "name": "read_file", "arguments": "{\"path\":\"a.rs\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "contents"}
            ],
            "tools": [{"type": "function", "name": "read_file", "parameters": {"type": "object"}}]
        }))
        .unwrap();
    let chat = responses_to_chat_request(&responses);
    assert_eq!(chat.extra["reincore_delegation_profile"], "codex");
    assert_eq!(chat.tools.as_ref().unwrap()[0].name, "read_file");
    assert_eq!(
        chat.messages[1].tool_calls.as_ref().unwrap()[0].id,
        "call_1"
    );
    assert_eq!(chat.messages[2].role, "tool");
    assert_eq!(chat.messages[2].tool_call_id.as_deref(), Some("call_1"));
    let shaped = chat_response_to_responses(
        json!({
            "id": "resp_1",
            "choices": [{"message": {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_2", "type": "function",
                "function": {"name": "read_file", "arguments": "{}"}
            }]}}]
        }),
        "auto",
    );
    assert_eq!(shaped["output"][1]["type"], "function_call");
    assert_eq!(shaped["output"][1]["call_id"], "call_2");

    let messages: MessagesApiRequest = serde_json::from_value(json!({
            "model": "auto",
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "tool_1", "name": "read_file", "input": {"path": "a.rs"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "contents"}]}
            ],
            "tools": [{"name": "read_file", "input_schema": {"type": "object"}}]
        }))
        .unwrap();
    let chat = messages_to_chat_request(&messages);
    assert_eq!(chat.extra["reincore_delegation_profile"], "claude");
    assert_eq!(chat.tools.as_ref().unwrap()[0].name, "read_file");
    assert_eq!(
        chat.messages[0].tool_calls.as_ref().unwrap()[0].id,
        "tool_1"
    );
    assert_eq!(chat.messages[1].role, "tool");
    assert_eq!(chat.messages[1].tool_call_id.as_deref(), Some("tool_1"));

    let gemini: GeminiGenerateContentRequest = serde_json::from_value(json!({
        "contents": [
            {
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "name": "codebase_investigator",
                        "args": {"prompt": "inspect it"}
                    }
                }]
            },
            {
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": "codebase_investigator",
                        "response": {"result": "done"}
                    }
                }]
            }
        ],
        "tools": [{
            "functionDeclarations": [{
                "name": "codebase_investigator",
                "description": "Inspect a codebase",
                "parameters": {"type": "object"}
            }]
        }]
    }))
    .unwrap();
    let chat = gemini_to_chat_request(&gemini, "gemini-test", false);
    assert_eq!(chat.extra["reincore_delegation_profile"], "gemini");
    assert_eq!(
        chat.tools.as_ref().unwrap()[0].name,
        "codebase_investigator"
    );
    let tool_call = &chat.messages[0].tool_calls.as_ref().unwrap()[0];
    assert_eq!(tool_call.function.name, "codebase_investigator");
    assert_eq!(
        chat.messages[1].tool_call_id.as_deref(),
        Some(tool_call.id.as_str())
    );
}

#[tokio::test]
async fn batch_events_round_trip_through_sqlite_and_reject_missing_sessions() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("batch events", 500).unwrap();
    db.insert_session(&session).unwrap();
    let state = AppState::new_with_project_path(db, None);
    let request = || engine_core::provider::batch::BatchEventRequest {
        batch_id: "job".into(),
        provider: "custom".into(),
        custom_id: "one".into(),
        phase: "submitted".into(),
        data: json!({"request_messages":[{"role":"user","content":"hello"}]}),
    };
    let recorded = record_batch_event(State(state.clone()), Path(session.id), Json(request()))
        .await
        .unwrap();
    assert_eq!(recorded.0["seq"], 1);
    let events = state.db.read_events(session.id, None, None).await.unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event_type, SessionEventType::Batch);
    assert_eq!(events[0].payload["batch_id"], "job");
    assert_eq!(events[0].cost_cents, None);
    assert_eq!(
        record_batch_event(State(state), Path(Uuid::new_v4()), Json(request()))
            .await
            .unwrap_err(),
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn f9_reports_survive_reload_and_can_correct_an_earlier_verdict() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("verify a recovered result", 1000).unwrap();
    db.insert_session(&session).unwrap();
    db.append_event(
        session.id,
        SessionEventType::LoopIntervention,
        json!({"repetition_count": 2, "injected_message": "retry"}),
        None,
    )
    .await
    .unwrap();
    let state = AppState::new_with_project_path(db, None);
    let completed = complete_session(
        State(state.clone()),
        Path(session.id),
        Json(CompleteSessionRequest {
            reason: None,
            outcome_report: None,
        }),
    )
    .await
    .unwrap()
    .0;
    assert_eq!(
        completed.cost_outcome.unwrap().success_status,
        SuccessStatus::Unknown
    );
    for (verdict, expected) in [
        (OutcomeVerdict::Success, SuccessStatus::RecoveredSuccess),
        (OutcomeVerdict::Failure, SuccessStatus::ReportedFailure),
        (OutcomeVerdict::Unknown, SuccessStatus::Unknown),
    ] {
        let response = complete_session(
            State(state.clone()),
            Path(session.id),
            Json(CompleteSessionRequest {
                reason: None,
                outcome_report: Some(OutcomeReport {
                    source: OutcomeSource::User,
                    verdict,
                    evidence: "checked output".into(),
                }),
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(
            response.cost_outcome.as_ref().unwrap().success_status,
            expected
        );
        let events = state.db.read_diagnostic_events(session.id).await.unwrap();
        let loaded = state
            .db
            .get_session_async(session.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            engine_core::session_cost_outcome(&loaded, &events).success_status,
            expected
        );
    }
    let bad = complete_session(
        State(state.clone()),
        Path(session.id),
        Json(CompleteSessionRequest {
            reason: None,
            outcome_report: Some(OutcomeReport {
                source: OutcomeSource::Validator,
                verdict: OutcomeVerdict::Success,
                evidence: " ".into(),
            }),
        }),
    )
    .await
    .unwrap_err();
    assert_eq!(bad, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn f9_cache_endpoint_uses_tokens_and_projection_preserves_unknown_cost() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("cache test", 1000).unwrap();
    db.insert_session(&session).unwrap();
    for (input, cached, cost) in [
        (10, 10, json!({"amount_nano_usd": 100_000})),
        (90, 0, json!({"amount_nano_usd": null})),
    ] {
        db.append_event(
            session.id,
            SessionEventType::LlmCall,
            json!({"input_tokens": input, "cached_prompt_tokens": cached, "cost": cost,
                "predicted_cost_low_cents": 0, "predicted_cost_high_cents": 1,
                "request_messages": [{"role": "user", "content": "should not be loaded"}]}),
            Some(0),
        )
        .await
        .unwrap();
    }
    // JSON booleans must not become integer counts in SQLite's projection.
    // An explicit null cost is unknown even when legacy display cents are zero.
    db.append_event(
        session.id,
        SessionEventType::LlmCall,
        json!({"input_tokens": true, "cached_prompt_tokens": true, "cost": null,
            "predicted_cost_low_cents": 0, "predicted_cost_high_cents": 1}),
        Some(0),
    )
    .await
    .unwrap();
    let raw = db.read_events(session.id, None, None).await.unwrap();
    let light = db.read_diagnostic_events(session.id).await.unwrap();
    assert!(light
        .iter()
        .all(|e| e.payload.get("request_messages").is_none()));
    assert_eq!(
        engine_core::diagnostics::CallDiagnostics::from_events(&raw),
        engine_core::diagnostics::CallDiagnostics::from_events(&light)
    );
    let state = AppState::new_with_project_path(db, None);
    let detail = get_session(State(state.clone()), Path(session.id))
        .await
        .unwrap()
        .0;
    assert!(detail.cost_outcome.unwrap().diagnostics.is_none());
    let diagnostics = get_session_diagnostics(State(state.clone()), Path(session.id))
        .await
        .unwrap()
        .0;
    assert_eq!(
        diagnostics,
        engine_core::diagnostics::CallDiagnostics::from_events(&raw)
    );
    assert_eq!(
        get_session_diagnostics(State(state.clone()), Path(Uuid::new_v4()))
            .await
            .unwrap_err(),
        StatusCode::NOT_FOUND
    );
    let listed = list_sessions(
        State(state.clone()),
        axum::extract::Query(Default::default()),
    )
    .await
    .unwrap()
    .0;
    let listed = serde_json::to_value(listed).unwrap();
    assert!(listed["sessions"][0]["cost_outcome"]
        .get("diagnostics")
        .is_none());
    let cache = cache_hit_rate(State(state)).await.unwrap().0;
    assert_eq!(cache.total_prompt_tokens, 100);
    assert_eq!(cache.total_cached_prompt_tokens, 10);
    assert_eq!(cache.token_weighted_utilization_bps, Some(1000.0));
}

#[tokio::test]
async fn session_page_http_validates_limits_and_preserves_legacy_listing() {
    let db = LocalDatabase::open_in_memory().unwrap();
    for _ in 0..3 {
        let session = Session::new("pagination", 100).unwrap();
        db.insert_session(&session).unwrap();
        db.append_event(session.id, SessionEventType::LlmCall, json!({}), None)
            .await
            .unwrap();
    }
    let state = AppState::new_with_project_path(db, None);
    let page = list_sessions(
        State(state.clone()),
        axum::extract::Query(std::collections::HashMap::from([(
            "limit".into(),
            "2".into(),
        )])),
    )
    .await
    .unwrap()
    .0;
    assert_eq!(
        serde_json::to_value(&page).unwrap()["sessions"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    let cursor = page.next_cursor.unwrap();
    let last = list_sessions(
        State(state.clone()),
        axum::extract::Query(std::collections::HashMap::from([
            ("limit".into(), "2".into()),
            ("before".into(), cursor),
        ])),
    )
    .await
    .unwrap()
    .0;
    assert_eq!(
        serde_json::to_value(&last).unwrap()["sessions"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert!(last.next_cursor.is_none());
    let all = list_sessions(
        State(state.clone()),
        axum::extract::Query(Default::default()),
    )
    .await
    .unwrap()
    .0;
    assert_eq!(
        serde_json::to_value(&all).unwrap()["sessions"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    for (key, value) in [
        ("limit", "0"),
        ("limit", "201"),
        ("limit", "bad"),
        ("limit", "18446744073709551615"),
        ("before", "bad"),
        ("date_from", "bad"),
        ("project_id", "bad"),
    ] {
        let status = list_sessions(
            State(state.clone()),
            axum::extract::Query(std::collections::HashMap::from([(
                key.into(),
                value.into(),
            )])),
        )
        .await
        .err()
        .unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }
}

mod action_stream_tests {
    use super::*;
    use crate::http::chat::{live_chat_completion_to_sse_stream, LocalStreamContext};
    use engine_core::{ChatCompletionStreamChunk, ToolCallDelta};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct RecoveryProvider {
        config: engine_core::ProviderConfig,
        valid: bool,
    }
    #[async_trait::async_trait]
    impl engine_core::Provider for RecoveryProvider {
        fn label(&self) -> &str {
            "test"
        }
        fn config(&self) -> &engine_core::ProviderConfig {
            &self.config
        }
        async fn list_models(&self) -> engine_core::EngineResult<Vec<engine_core::ProviderModel>> {
            Ok(vec![])
        }
        async fn complete(
            &self,
            _: &ChatCompletionRequest,
        ) -> engine_core::EngineResult<engine_core::ChatCompletionResponse> {
            if !self.valid {
                return Err(engine_core::EngineError::Provider(
                    "test recovery failure".into(),
                ));
            }
            Ok(engine_core::ChatCompletionResponse {id:"recovered".into(),model:"test".into(),
                message:serde_json::from_value(json!({"role":"assistant","tool_calls":[{"id":"corrected","type":"function","function":{"name":"read","arguments":"{\"path\":\"a.rs\"}"}}]})).unwrap(),
                usage:None,raw:json!({"choices":[{"finish_reason":"tool_calls"}]}),response_headers:Default::default()})
        }
    }

    struct EmptyRetryProvider {
        config: engine_core::ProviderConfig,
        calls: Arc<AtomicUsize>,
    }
    #[async_trait::async_trait]
    impl engine_core::Provider for EmptyRetryProvider {
        fn label(&self) -> &str {
            "test"
        }
        fn config(&self) -> &engine_core::ProviderConfig {
            &self.config
        }
        fn supports_streaming(&self) -> bool {
            true
        }
        async fn list_models(&self) -> engine_core::EngineResult<Vec<engine_core::ProviderModel>> {
            Ok(vec![])
        }
        async fn complete(
            &self,
            _: &ChatCompletionRequest,
        ) -> engine_core::EngineResult<engine_core::ChatCompletionResponse> {
            Err(engine_core::EngineError::Provider(
                "unexpected non-streaming call".into(),
            ))
        }
        async fn complete_stream(
            &self,
            _: &ChatCompletionRequest,
        ) -> engine_core::EngineResult<engine_core::ProviderStream> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(engine_core::ProviderStream::without_headers(
                futures::stream::iter(vec![
                    Ok(ChatCompletionStreamChunk {
                        delta_content: Some("Recovered response".into()),
                        ..Default::default()
                    }),
                    Ok(ChatCompletionStreamChunk {
                        finish_reason: Some("stop".into()),
                        usage: Some(engine_core::TokenUsage {
                            prompt_tokens: 7,
                            completion_tokens: 2,
                            total_tokens: 9,
                            ..Default::default()
                        }),
                        ..Default::default()
                    }),
                ]),
            ))
        }
    }

    async fn context(policy: engine_core::action_gate::ActionPolicy) -> LocalStreamContext {
        let request: ChatCompletionRequest = serde_json::from_value(json!({"model":"test","messages":[{"role":"user","content":"read file"}],
            "tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object","required":["path"],"properties":{"path":{"type":"string"}}}}}]})).unwrap();
        let config = engine_core::ProviderConfig::default();
        let predictor = engine_core::CostPredictor::new();
        let group = Uuid::new_v4().to_string();
        LocalStreamContext {
            deduped_tool_results: Default::default(),
            loop_breaker: engine_core::CircuitBreaker::default_config(),
            claim_grounding: std::sync::Arc::new(Default::default()),
            db: Arc::new(LocalDatabase::open_in_memory().unwrap()),
            action_policy: Arc::new(policy),
            detection: std::sync::Arc::new(Default::default()),
            detection_judge: None,
            session_id: None,
            task: engine_core::resolve_task(None, &group, &request, &[]),
            request_group_id: group,
            task_context_messages_removed: 0,
            original_request: request.clone(),
            provider_request: request.clone(),
            provider_label: "test".into(),
            resolved_model: "test".into(),
            model_tier: engine_core::ModelTier::Pro,
            prefix_hash: String::new(),
            prediction: predictor
                .predict(&request, engine_core::ModelTier::Pro)
                .unwrap(),
            cost_predictor: predictor,
            classification: None,
            batch_requested: false,
            semantic: None,
            semantic_candidates: vec![],
            citation_attachments: None,
            initial_agent_health: Default::default(),
            prewarm_expected: false,
            prewarm_provider: Arc::new(RecoveryProvider {
                config: config.clone(),
                valid: false,
            }),
            prewarm_provider_config: config,

            provider_request_sent_at: chrono::Utc::now(),
            provider_request_started: std::time::Instant::now(),
            request_started: std::time::Instant::now(),
            provider_timer: engine_core::operational::Timer::new(
                engine_core::operational::Metric::ProviderOperation,
            ),
            context_adjustment: Default::default(),
            provider_span: tracing::Span::none(),
            stream_span: tracing::Span::none(),
        }
    }

    #[tokio::test]
    async fn redaction_stream_releases_sanitized_tools_without_recovery() {
        struct RetryProbe {
            config: engine_core::ProviderConfig,
            calls: Arc<AtomicUsize>,
        }
        #[async_trait::async_trait]
        impl engine_core::Provider for RetryProbe {
            fn label(&self) -> &str {
                "test"
            }
            fn config(&self) -> &engine_core::ProviderConfig {
                &self.config
            }
            async fn list_models(
                &self,
            ) -> engine_core::EngineResult<Vec<engine_core::ProviderModel>> {
                Ok(vec![])
            }
            async fn complete(
                &self,
                _: &ChatCompletionRequest,
            ) -> engine_core::EngineResult<engine_core::ChatCompletionResponse> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Err(engine_core::EngineError::Provider(
                    "privacy recovery must not run".into(),
                ))
            }
        }

        for command in [
            "Get-Content .env; Get-FileHash .env",
            "npm run configure",
            "echo PASSWORD=hunter123",
        ] {
            let calls = Arc::new(AtomicUsize::new(0));
            let mut ctx = context(Default::default()).await;
            ctx.claim_grounding = Arc::new(engine_core::claim_grounding::ClaimGroundingConfig {
                enabled: false,
                ..Default::default()
            });
            ctx.prewarm_provider = Arc::new(RetryProbe {
                config: Default::default(),
                calls: calls.clone(),
            });
            ctx.provider_request.messages.extend(
                serde_json::from_value::<Vec<engine_core::ChatMessage>>(json!([
                    {"role":"assistant","tool_calls":[{"id":"read-1","type":"function","function":{
                        "name":"read","arguments":"{\"path\":\".env\"}"}}]},
                    {"role":"tool","tool_call_id":"read-1","content":"PASSWORD=[REDACTED]"}
                ]))
                .unwrap(),
            );
            ctx.provider_request.tools = serde_json::from_value(json!([{
                "type":"function","function":{"name":"terminal","parameters":{"type":"object"}}
            }]))
            .unwrap();
            let mut proposed = tool(&serde_json::to_string(&json!({"command":command})).unwrap());
            proposed.delta_tool_calls.as_mut().unwrap()[0].function_name = Some("terminal".into());
            let upstream =
                engine_core::ProviderStream::without_headers(futures::stream::iter(vec![
                    Ok(proposed),
                    Ok(ChatCompletionStreamChunk {
                        finish_reason: Some("tool_calls".into()),
                        ..Default::default()
                    }),
                ]));
            let privacy = Arc::new(engine_core::PrivacyPolicy {
                provider_secret_action: engine_core::SecretAction::Redact,
                tool_secret_action: engine_core::SecretAction::Redact,
                ..Default::default()
            });
            let text = engine_core::privacy::scope_request_policy(Some(privacy), async {
                let response =
                    Sse::new(live_chat_completion_to_sse_stream(upstream, ctx)).into_response();
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                String::from_utf8(bytes.to_vec()).unwrap()
            })
            .await;

            assert!(text.contains("\"tool_calls\":["), "{text}");
            assert!(
                text.contains("\"reincore_tool_calls_validated\":true"),
                "{text}"
            );
            assert!(!text.contains("privacy_boundary"), "{text}");
            assert!(!text.contains("non-retryable"), "{text}");
            assert!(!text.contains("\"reincore_action_blocked\":true"), "{text}");
            assert!(text.contains("\"finish_reason\":\"tool_calls\""), "{text}");
            assert!(!text.contains("hunter123"), "{text}");
            if command.contains("hunter123") {
                assert!(text.contains("[REDACTED]"), "{text}");
            }
            assert_eq!(calls.load(Ordering::SeqCst), 0, "{text}");
        }
    }

    #[tokio::test]
    async fn empty_provider_stream_is_retried_and_usage_is_accumulated() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut ctx = context(Default::default()).await;
        ctx.prewarm_provider = Arc::new(EmptyRetryProvider {
            config: Default::default(),
            calls: calls.clone(),
        });
        let upstream =
            engine_core::ProviderStream::without_headers(futures::stream::iter(vec![Ok(
                ChatCompletionStreamChunk {
                    finish_reason: Some("stop".into()),
                    usage: Some(engine_core::TokenUsage {
                        prompt_tokens: 5,
                        total_tokens: 5,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )]));
        let response = Sse::new(live_chat_completion_to_sse_stream(upstream, ctx)).into_response();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(text.contains("Recovered response"), "{text}");
        assert!(text.contains("\"prompt_tokens\":12"), "{text}");
        assert!(text.contains("\"completion_tokens\":2"), "{text}");
        assert!(
            text.contains("\"reincore_provider_retry_count\":1"),
            "{text}"
        );
    }
    #[tokio::test]
    async fn claim_grounding_stream_reports_contradictions_and_honors_disable() {
        for enabled in [true, false] {
            let mut ctx = context(Default::default()).await;
            ctx.claim_grounding = Arc::new(engine_core::claim_grounding::ClaimGroundingConfig {
                enabled,
                ..Default::default()
            });
            ctx.provider_request.messages.extend(serde_json::from_value::<Vec<engine_core::ChatMessage>>(json!([
                {"role":"assistant","tool_calls":[{"id":"test","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"cargo test\"}"}}]},
                {"role":"tool","tool_call_id":"test","content":"{\"exit_code\":1,\"output\":\"1 passed; 1 failed\"}"}
            ])).unwrap());
            engine_core::claim_grounding::prepare(
                ctx.db.as_ref(),
                None,
                &mut ctx.provider_request,
                &ctx.claim_grounding.clone(),
            )
            .await;
            let chunks = vec![
                Ok(ChatCompletionStreamChunk {
                    delta_content: Some("Tests passed.".into()),
                    ..Default::default()
                }),
                Ok(ChatCompletionStreamChunk {
                    finish_reason: Some("stop".into()),
                    ..Default::default()
                }),
            ];
            let upstream =
                engine_core::ProviderStream::new(futures::stream::iter(chunks), Default::default());
            let response =
                Sse::new(live_chat_completion_to_sse_stream(upstream, ctx)).into_response();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let text = String::from_utf8(bytes.to_vec()).unwrap();
            assert!(text.contains("Tests passed."));
            assert_eq!(text.contains("contradicted_claim"), enabled);
            assert_eq!(text.contains("claim_grounding"), enabled);
        }
    }

    #[tokio::test]
    async fn action_gate_stream_assembles_snapshots_and_fragments_before_validation() {
        for snapshot in [true, false] {
            let mut first = tool(if snapshot {
                r#"{"path":"old.rs"}"#
            } else {
                r#"{"path":"#
            });
            let mut second = tool(if snapshot {
                r#"{"path":"a.rs"}"#
            } else {
                r#""a.rs"}"#
            });
            let first_delta = &mut first.delta_tool_calls.as_mut().unwrap()[0];
            first_delta.snapshot = snapshot;
            first_delta.function_name = Some(if snapshot { "read" } else { "re" }.into());
            let second_delta = &mut second.delta_tool_calls.as_mut().unwrap()[0];
            second_delta.snapshot = snapshot;
            second_delta.function_name = Some(if snapshot { "read" } else { "ad" }.into());
            let mut chunks = vec![Ok(first), Ok(second.clone())];
            if snapshot {
                chunks.push(Ok(second));
            }
            chunks.push(Ok(ChatCompletionStreamChunk {
                finish_reason: Some("tool_calls".into()),
                ..Default::default()
            }));
            let upstream =
                engine_core::ProviderStream::new(futures::stream::iter(chunks), Default::default());
            let stream =
                live_chat_completion_to_sse_stream(upstream, context(Default::default()).await);
            let bytes =
                axum::body::to_bytes(Sse::new(stream).into_response().into_body(), usize::MAX)
                    .await
                    .unwrap();
            let text = String::from_utf8(bytes.to_vec()).unwrap();
            assert!(
                text.contains("reincore_tool_calls_validated\":true"),
                "{text}"
            );
            let calls: Vec<serde_json::Value> = text
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .filter_map(|event| {
                    event["choices"][0]["delta"]["tool_calls"]
                        .as_array()
                        .cloned()
                })
                .flatten()
                .collect();
            assert_eq!(calls.len(), 1, "{text}");
            assert_eq!(calls[0]["function"]["name"], "read");
            assert_eq!(calls[0]["function"]["arguments"], r#"{"path":"a.rs"}"#);
        }
    }

    #[tokio::test]
    async fn action_recovery_stream_releases_only_the_corrected_validated_call() {
        for denied in [false, true] {
            let policy = engine_core::action_gate::ActionPolicy {
                default_allow: !denied,
                ..Default::default()
            };
            let mut ctx = context(policy).await;
            ctx.prewarm_provider = Arc::new(RecoveryProvider {
                config: Default::default(),
                valid: true,
            });
            let mut bad = tool("{}");
            bad.delta_tool_calls.as_mut().unwrap()[0].function_name = Some("readread".into());
            let upstream = engine_core::ProviderStream::new(
                futures::stream::iter(vec![
                    Ok(bad),
                    Ok(ChatCompletionStreamChunk {
                        finish_reason: Some("tool_calls".into()),
                        ..Default::default()
                    }),
                ]),
                Default::default(),
            );
            let response =
                Sse::new(live_chat_completion_to_sse_stream(upstream, ctx)).into_response();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let text = String::from_utf8(bytes.to_vec()).unwrap();
            assert_eq!(
                text.contains("reincore_action_blocked\":true"),
                denied,
                "{text}"
            );
            assert_eq!(text.contains("\"id\":\"corrected\""), !denied, "{text}");
            assert!(text.contains("action_denied"));
        }
    }

    #[tokio::test]
    async fn missing_fields_in_multiple_tools_recover_without_stopping_the_stream() {
        struct CorrectionProbe {
            config: engine_core::ProviderConfig,
            calls: Arc<AtomicUsize>,
        }
        #[async_trait::async_trait]
        impl engine_core::Provider for CorrectionProbe {
            fn label(&self) -> &str {
                "test"
            }
            fn config(&self) -> &engine_core::ProviderConfig {
                &self.config
            }
            async fn list_models(
                &self,
            ) -> engine_core::EngineResult<Vec<engine_core::ProviderModel>> {
                Ok(vec![])
            }
            async fn complete(
                &self,
                request: &ChatCompletionRequest,
            ) -> engine_core::EngineResult<engine_core::ChatCompletionResponse> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let feedback = request.messages.last().unwrap().full_text();
                for field in [
                    "list_dir",
                    "file_search",
                    "read",
                    "missing required property",
                    "expected string",
                ] {
                    assert!(feedback.contains(field), "{feedback}");
                }
                engine_core::Provider::complete(
                    &RecoveryProvider {
                        config: self.config.clone(),
                        valid: true,
                    },
                    request,
                )
                .await
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let mut ctx = context(engine_core::action_gate::ActionPolicy {
            stop_on_schema_validation_error: true,
            ..Default::default()
        })
        .await;
        ctx.claim_grounding = Arc::new(engine_core::claim_grounding::ClaimGroundingConfig {
            enabled: false,
            ..Default::default()
        });
        ctx.prewarm_provider = Arc::new(CorrectionProbe {
            config: Default::default(),
            calls: calls.clone(),
        });
        ctx.provider_request.tools = serde_json::from_value(json!([
            {
                "name":"read",
                "parameters":{
                    "type":"object",
                    "required":["path"],
                    "properties":{"path":{"type":"string"}}
                }
            },
            {
                "name":"list_dir",
                "parameters":{
                    "type":"object",
                    "required":["path"],
                    "properties":{"path":{"type":"string"}}
                }
            },
            {
                "name":"file_search",
                "parameters":{
                    "type":"object",
                    "required":["query"],
                    "properties":{"query":{"type":"string"}}
                }
            }
        ]))
        .unwrap();
        let mut chunks = Vec::new();
        for (index, name) in ["list_dir", "file_search", "read"].iter().enumerate() {
            let mut chunk = tool("{}");
            let delta = &mut chunk.delta_tool_calls.as_mut().unwrap()[0];
            delta.index = index;
            delta.id = Some(format!("invalid-{index}"));
            delta.function_name = Some((*name).into());
            chunks.push(Ok(chunk));
        }
        chunks.push(Ok(ChatCompletionStreamChunk {
            finish_reason: Some("tool_calls".into()),
            ..Default::default()
        }));
        let upstream = engine_core::ProviderStream::without_headers(futures::stream::iter(chunks));
        let response = Sse::new(live_chat_completion_to_sse_stream(upstream, ctx)).into_response();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1, "{text}");
        assert!(
            text.contains("\"reincore_tool_calls_validated\":true"),
            "{text}"
        );
        assert!(text.contains("\"finish_reason\":\"tool_calls\""), "{text}");
        assert!(text.contains("\"id\":\"corrected\""), "{text}");
        assert!(!text.contains("\"id\":\"invalid-"), "{text}");
        let events = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .collect::<Vec<_>>();
        let health = events
            .iter()
            .find_map(|event| event.get("reincore_agent_health"))
            .unwrap();
        assert_eq!(health["should_stop"], false, "{text}");
        assert!(
            health["interventions"]
                .as_array()
                .unwrap()
                .iter()
                .all(|record| record["outcome"] == "recovered"),
            "{text}"
        );
    }

    fn tool(arguments: &str) -> ChatCompletionStreamChunk {
        ChatCompletionStreamChunk {
            delta_tool_calls: Some(vec![ToolCallDelta {
                snapshot: false,
                index: 0,
                id: Some("call".into()),
                call_type: Some("function".into()),
                function_name: Some("read".into()),
                function_arguments: Some(arguments.into()),
            }]),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn action_gate_stream_holds_tools_but_delivers_text_before_finish() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let upstream = engine_core::ProviderStream::new(
            async_stream::stream! {while let Some(chunk)=rx.recv().await {yield chunk;}},
            Default::default(),
        );
        let stream =
            live_chat_completion_to_sse_stream(upstream, context(Default::default()).await);
        let mut body = Sse::new(stream)
            .into_response()
            .into_body()
            .into_data_stream();
        tx.send(Ok(ChatCompletionStreamChunk {
            delta_content: Some("visible text".into()),
            ..Default::default()
        }))
        .await
        .unwrap();
        tx.send(Ok(tool("{\"path\":\"a.rs\"}"))).await.unwrap();
        let role = body.next().await.unwrap().unwrap();
        assert!(!String::from_utf8_lossy(&role).contains("tool_calls"));
        let text = body.next().await.unwrap().unwrap();
        assert!(String::from_utf8_lossy(&text).contains("visible text"));
        assert!(tokio::time::timeout(Duration::from_millis(30), body.next())
            .await
            .is_err());
        tx.send(Ok(ChatCompletionStreamChunk {
            finish_reason: Some("tool_calls".into()),
            ..Default::default()
        }))
        .await
        .unwrap();
        drop(tx);
        let mut output = String::new();
        while let Some(chunk) = body.next().await {
            output.push_str(&String::from_utf8_lossy(&chunk.unwrap()));
        }
        assert!(output.contains("reincore_tool_calls_validated\":true"));
        assert!(output.contains("\"tool_calls\":["));
    }

    #[tokio::test]
    async fn action_gate_stream_denies_bad_arguments_policies_and_truncation() {
        for scenario in [
            "bad_schema",
            "denied",
            "truncated",
            "error",
            "oversize",
            "missing_snapshot",
            "malformed_name",
        ] {
            let policy = engine_core::action_gate::ActionPolicy {
                default_allow: scenario != "denied",
                stop_on_schema_validation_error: scenario == "bad_schema",
                ..Default::default()
            };
            let arguments = if scenario == "bad_schema" {
                "{}".into()
            } else if scenario == "oversize" {
                "x".repeat(engine_core::action_gate::MAX_TOOL_ARGUMENT_BYTES + 1)
            } else {
                "{\"path\":\"a.rs\"}".into()
            };
            let mut chunks = vec![
                Ok(ChatCompletionStreamChunk {
                    delta_content: Some("visible text".into()),
                    ..Default::default()
                }),
                Ok(tool(&arguments)),
            ];
            if scenario == "missing_snapshot" || scenario == "malformed_name" {
                let mut invalid = tool(&arguments);
                let delta = &mut invalid.delta_tool_calls.as_mut().unwrap()[0];
                delta.snapshot = true;
                if scenario == "missing_snapshot" {
                    delta.function_arguments = None;
                } else {
                    delta.function_name = Some("readread".into());
                }
                chunks.push(Ok(invalid));
            }
            if scenario == "error" {
                chunks.push(Err(engine_core::EngineError::Provider(
                    "upstream failed".into(),
                )));
            } else if scenario != "truncated" {
                chunks.push(Ok(ChatCompletionStreamChunk {
                    finish_reason: Some("tool_calls".into()),
                    ..Default::default()
                }));
            }
            let upstream =
                engine_core::ProviderStream::new(futures::stream::iter(chunks), Default::default());
            let response = Sse::new(live_chat_completion_to_sse_stream(
                upstream,
                context(policy).await,
            ))
            .into_response();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let text = String::from_utf8(bytes.to_vec()).unwrap();
            assert!(text.contains("visible text"));
            assert!(
                !text.contains("\"tool_calls\":["),
                "{scenario}: leaked tool call"
            );
            assert!(!text.contains("reincore_provider_state"));
            assert!(
                text.contains("reincore_action_blocked\":true") || text.contains("\"error\":"),
                "{scenario}"
            );
        }
    }
}

#[tokio::test]
async fn task_decision_endpoint_reads_scoped_answers_and_rejects_missing_sessions() {
    let db = LocalDatabase::open_in_memory().unwrap();
    let session = Session::new("Decisions", 100).unwrap();
    db.insert_session(&session).unwrap();
    let mut request:ChatCompletionRequest=serde_json::from_value(json!({"model":"test","messages":[{"role":"user","content":"Use PostgreSQL"}],"reincore_task_id":"task-one"})).unwrap();
    let task = engine_core::resolve_task(Some(session.id), "g1", &request, &[]);
    let scope = json!([session.project_id, session.project_path]).to_string();
    engine_core::task_decisions::prepare(
        &db,
        Some(session.id),
        &task,
        "g1",
        &scope,
        &mut request,
        true,
    )
    .await
    .unwrap();
    let mut state = AppState::new_with_project_path(db, None);
    state.task_decisions_enabled = false;
    let query = engine_core::task_decisions::DecisionQuery {
        question: Some("Which database should we use?".into()),
        ..Default::default()
    };
    let response = get_task_decisions(
        State(state.clone()),
        Path((session.id, "task-one".into())),
        axum::extract::Query(query),
    )
    .await
    .unwrap();
    assert_eq!(response.0["can_reuse"], true);
    assert_eq!(response.0["matching_decision"]["value"], "Use PostgreSQL");
    let missing = get_task_decisions(
        State(state),
        Path((Uuid::new_v4(), "task-one".into())),
        axum::extract::Query(Default::default()),
    )
    .await;
    assert_eq!(missing.unwrap_err(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn muse_code_models_lists_engine_models() {
    let state = AppState::new_with_project_path(LocalDatabase::open_in_memory().unwrap(), None);
    let Json(body) = super::base::muse_code_models(State(state)).await.unwrap();
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().unwrap();
    assert!(!data.is_empty());
    assert_eq!(data[0]["id"], "reincore-auto");
}

#[tokio::test]
async fn responses_stream_carries_item_ids_and_lifecycle_events() {
    let shaped = chat_response_to_responses(
        json!({
            "id": "resp_1",
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
        }),
        "auto",
    );
    assert_eq!(shaped["usage"]["input_tokens"], 10);
    assert_eq!(shaped["usage"]["output_tokens"], 3);
    assert!(shaped["usage"]["input_tokens_details"].is_object());
    assert!(shaped["usage"]["output_tokens_details"].is_object());
    // Strict Responses clients reject unknown top-level fields.
    for key in shaped.as_object().unwrap().keys() {
        assert!(
            !key.starts_with("reincore_"),
            "responses output leaks {key}"
        );
    }
    let item_id = shaped["output"][0]["id"].as_str().unwrap().to_string();
    assert!(item_id.starts_with("msg_"));

    let response = Sse::new(responses_json_to_sse_stream(shaped)).into_response();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    assert!(body.contains("event: response.in_progress"));
    assert!(body.contains("event: response.content_part.added"));
    assert!(body.contains("event: response.output_text.done"));
    assert!(body.contains("event: response.content_part.done"));
    assert!(body.contains(item_id.as_str()));
    // Every frame carries a monotonically increasing sequence_number,
    // required by Muse Code's Meta provider.
    let mut expected = 0;
    for line in body.lines().filter(|line| line.starts_with("data: {")) {
        let frame: serde_json::Value = serde_json::from_str(&line["data: ".len()..]).unwrap();
        assert_eq!(frame["sequence_number"], expected, "frame {line}");
        expected += 1;
    }
    assert!(expected > 3);
}

#[test]
fn namespace_tools_expand_to_nested_functions() {
    let tools = super::compatibility::api_tool_definitions(Some(&[json!({
        "type": "namespace",
        "name": "muse",
        "description": "Muse Code tool set.",
        "tools": [
            {"type": "function", "name": "read_file", "parameters": {"type": "object"}},
            {"type": "function", "name": "write_file", "parameters": {"type": "object"}}
        ]
    })]))
    .unwrap();
    let names: Vec<&str> = tools.iter().map(|tool| tool.name.as_str()).collect();
    assert_eq!(names, vec!["read_file", "write_file"]);
}

#[test]
fn compatibility_preserves_action_receipt_contracts_without_granting_permissions() {
    let flat = serde_json::json!({"name":"send","input_schema":{"type":"object"},"action_contract":{
        "action":"email.send","target_pointer":"/to","completion_template":"Email sent to {target}."}});
    for value in [
        flat.clone(),
        serde_json::json!({"type":"function","function":flat}),
    ] {
        let tools = super::compatibility::api_tool_definitions(Some(&[value])).unwrap();
        assert_eq!(
            tools[0].action_contract.as_ref().unwrap().action,
            "email.send"
        );
        assert!(!tools[0].safe_to_memoize);
    }
    let invalid = serde_json::json!({"name":"send","action_contract":{"action":"email.send"}});
    let tools = super::compatibility::api_tool_definitions(Some(&[invalid])).unwrap();
    assert!(tools[0].action_contract.is_none());
}
