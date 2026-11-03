use super::*;
use engine_core::tool_cache::{
    ExactToolCacheEnvelope, LocalToolCacheContext, ToolCachePolicy, ToolCacheSharing,
};
use engine_core::{
    EngineResult, ToolCall, ToolDefinition, ToolExecutor, ToolExecutorRegistry, ToolResult,
};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};

struct Executor {
    policy: Mutex<Option<ToolCachePolicy>>,
    calls: AtomicUsize,
    error: AtomicBool,
    rotate: AtomicBool,
}
#[async_trait::async_trait]
impl ToolExecutor for Executor {
    async fn list_tools(&self) -> EngineResult<Vec<ToolDefinition>> {
        Ok(vec![ToolDefinition {
            action_contract: None,
            name: "read_file".into(),
            description: "test".into(),
            parameters: json!({}),
            safe_to_memoize: true,
            memoization_policy: None,
        }])
    }
    async fn local_cache_policy(
        &self,
        _: &ToolCall,
        _: &LocalToolCacheContext,
    ) -> EngineResult<Option<ToolCachePolicy>> {
        Ok(self.policy.lock().unwrap().clone())
    }
    async fn execute(&self, call: &ToolCall) -> EngineResult<ToolResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.rotate.load(Ordering::SeqCst) {
            self.policy
                .lock()
                .unwrap()
                .as_mut()
                .unwrap()
                .data_version
                .push('x');
        }
        Ok(ToolResult {
            tool_call_id: call.id.clone(),
            role: "tool".into(),
            name: call.function.name.clone(),
            content: "contents".into(),
            is_error: Some(self.error.load(Ordering::SeqCst)),
        })
    }
    async fn execute_local_in_context(
        &self,
        call: &ToolCall,
        context: &LocalToolCacheContext,
    ) -> EngineResult<ToolResult> {
        let mut result = self.execute(call).await?;
        result.content = format!("{:?}: contents", context.project_path);
        Ok(result)
    }
}
fn policy() -> ToolCachePolicy {
    ToolCachePolicy {
        read_only: true,
        sharing: ToolCacheSharing::Private,
        semantic_reuse: false,
        ttl_seconds: 60,
        policy_version: "1".into(),
        tool_version: "1".into(),
        data_namespace: "executor-root".into(),
        data_version: "1".into(),
        credential_identity: "key-a".into(),
        credential_version: "1".into(),
    }
}
fn fixture() -> (LocalDatabase, Session, Arc<Executor>, ToolExecutorRegistry) {
    let db = LocalDatabase::open_in_memory().unwrap();
    let mut session = Session::new("cache test", 100).unwrap();
    session.project_path = Some(std::env::current_dir().unwrap().to_str().unwrap().into());
    db.insert_session(&session).unwrap();
    let executor = Arc::new(Executor {
        policy: Mutex::new(Some(policy())),
        calls: AtomicUsize::new(0),
        error: AtomicBool::new(false),
        rotate: AtomicBool::new(false),
    });
    let registry = ToolExecutorRegistry::new().register(executor.clone());
    (db, session, executor, registry)
}
async fn invoke(
    db: &LocalDatabase,
    registry: &ToolExecutorRegistry,
    session: &Session,
    consent: bool,
) -> serde_json::Value {
    let (status, Json(body)) = execute_local_tool_call(
        db,
        registry,
        ToolCallRequest {
            session_id: session.id,
            tool_name: "read_file".into(),
            arguments: json!({"path": "src/main.rs"}),
            safe_to_memoize: consent,
            is_subagent: false,
        },
        &Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(status, StatusCode::OK);
    body
}
fn cache_key(session: &Session, p: &ToolCachePolicy) -> engine_core::CacheKey {
    ExactToolCacheEnvelope::key(
        &json!({"adapter":"local-session-v1", "context":local_context(session)}),
        p,
        "read_file",
        &json!({"path":"src/main.rs"}),
    )
    .unwrap()
}

#[tokio::test]
async fn sqlite_exact_hit_rebinds_call_id_and_avoids_execution() {
    let (db, s, ex, reg) = fixture();
    let first = invoke(&db, &reg, &s, true).await;
    let second = invoke(&db, &reg, &s, true).await;
    assert_eq!(first["cache_hit"], false);
    assert_eq!(second["cache_hit"], true);
    assert_ne!(
        first["result"]["tool_call_id"],
        second["result"]["tool_call_id"]
    );
    assert_eq!(ex.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn sqlite_scope_changes_miss_including_shared_policy() {
    for boundary in ["session", "path", "project", "client"] {
        let (db, mut s, ex, reg) = fixture();
        ex.policy.lock().unwrap().as_mut().unwrap().sharing = ToolCacheSharing::Workspace;
        invoke(&db, &reg, &s, true).await;
        match boundary {
            "session" => s.id = Uuid::new_v4(),
            "path" => s.project_path = Some(std::env::temp_dir().to_str().unwrap().into()),
            "project" => s.project_id = Some(Uuid::new_v4()),
            "client" => s.client_id = Some("different-client".into()),
            _ => unreachable!(),
        }
        db.insert_session(&s).unwrap();
        assert_eq!(
            invoke(&db, &reg, &s, true).await["cache_hit"],
            false,
            "{boundary}"
        );
        assert_eq!(ex.calls.load(Ordering::SeqCst), 2);
    }
}

#[tokio::test]
async fn sqlite_all_executor_version_boundaries_miss() {
    for boundary in 0..7 {
        let (db, s, ex, reg) = fixture();
        invoke(&db, &reg, &s, true).await;
        {
            let mut guard = ex.policy.lock().unwrap();
            let p = guard.as_mut().unwrap();
            let field = match boundary {
                0 => &mut p.policy_version,
                1 => &mut p.tool_version,
                2 => &mut p.data_namespace,
                3 => &mut p.data_version,
                4 => &mut p.credential_identity,
                5 => &mut p.credential_version,
                _ => {
                    p.ttl_seconds = 30;
                    &mut p.policy_version
                }
            };
            field.push('x');
        }
        assert_eq!(invoke(&db, &reg, &s, true).await["cache_hit"], false);
        assert_eq!(ex.calls.load(Ordering::SeqCst), 2);
    }
}

#[tokio::test]
async fn sqlite_denies_missing_policy_consent_invalid_policy_and_unscoped_paths() {
    for boundary in 0..7 {
        let (db, mut s, ex, reg) = fixture();
        match boundary {
            0 => *ex.policy.lock().unwrap() = None,
            1 => ex.policy.lock().unwrap().as_mut().unwrap().read_only = false,
            2 => ex
                .policy
                .lock()
                .unwrap()
                .as_mut()
                .unwrap()
                .credential_identity
                .clear(),
            3 => s.project_path = None,
            4 => s.project_path = Some("relative/path".into()),
            5 => ex.policy.lock().unwrap().as_mut().unwrap().ttl_seconds = 0,
            _ => (),
        }
        db.insert_session(&s).unwrap();
        for _ in 0..2 {
            let body = invoke(&db, &reg, &s, boundary != 6).await;
            assert_eq!(body["cache_hit"], false);
            assert_eq!(body["memoization_enabled"], false);
        }
        assert_eq!(ex.calls.load(Ordering::SeqCst), 2);
    }
}

#[tokio::test]
async fn sqlite_rejects_expired_tampered_future_and_legacy_results() {
    for boundary in 0..4 {
        let (db, s, ex, reg) = fixture();
        invoke(&db, &reg, &s, true).await;
        let key = cache_key(&s, &policy());
        let mut entry = db.get_memoized(&key).await.unwrap().unwrap();
        match boundary {
            0 => entry.created_at -= chrono::Duration::seconds(61),
            1 => entry.result_payload["result"]["content"] = json!("tampered"),
            2 => entry.created_at += chrono::Duration::seconds(60),
            _ => entry.result_payload = entry.result_payload["result"].clone(),
        }
        db.put_memoized(key, entry).await.unwrap();
        assert_eq!(invoke(&db, &reg, &s, true).await["cache_hit"], false);
        assert_eq!(ex.calls.load(Ordering::SeqCst), 2);
    }
}

#[tokio::test]
async fn sqlite_errors_and_rotation_during_execution_are_not_cached() {
    for error in [false, true] {
        let (db, s, ex, reg) = fixture();
        ex.error.store(error, Ordering::SeqCst);
        ex.rotate.store(!error, Ordering::SeqCst);
        invoke(&db, &reg, &s, true).await;
        assert!(db
            .get_memoized(&cache_key(&s, &policy()))
            .await
            .unwrap()
            .is_none());
        assert_eq!(invoke(&db, &reg, &s, true).await["cache_hit"], false);
    }
}

#[tokio::test]
async fn sqlite_file_invalidation_matches_nested_arguments_literally() {
    use engine_core::MemoizationCache;
    let (db, s, _, reg) = fixture();
    invoke(&db, &reg, &s, true).await;
    let mut file = engine_core::FileContextAttachment {
        path: "src/%".into(),
        version: 2,
        content: None,
        content_hash: None,
    };
    assert_eq!(db.invalidate_for_file_update(&file).await.unwrap(), 0);
    file.path = "src/main.rs".into();
    assert_eq!(db.invalidate_for_file_update(&file).await.unwrap(), 1);
    assert_eq!(invoke(&db, &reg, &s, true).await["cache_hit"], false);
}

struct DefaultExecutor;
#[async_trait::async_trait]
impl ToolExecutor for DefaultExecutor {
    async fn list_tools(&self) -> EngineResult<Vec<ToolDefinition>> {
        Ok(vec![])
    }
    async fn execute(&self, _: &ToolCall) -> EngineResult<ToolResult> {
        unreachable!()
    }
}
#[tokio::test]
async fn default_executor_and_legacy_api_cannot_authorize_reuse() {
    use engine_core::MemoizationCache;
    let (db, s, _, reg) = fixture();
    let body = invoke(&db, &reg, &s, true).await;
    let call = ToolCall {
        id: "test".into(),
        call_type: "function".into(),
        function: engine_core::FunctionCall {
            name: "read_file".into(),
            arguments: r#"{"path":"src/main.rs"}"#.into(),
        },
    };
    assert!(DefaultExecutor
        .local_cache_policy(&call, &local_context(&s))
        .await
        .unwrap()
        .is_none());
    let tool = engine_core::cache::ToolDefinitionForCache {
        name: "read_file".into(),
        safe_to_memoize: true,
        policy: engine_core::MemoizationPolicy::Disabled,
    };
    let result = serde_json::from_value(body["result"].clone()).unwrap();
    db.store(&tool, &call, &result).await.unwrap();
    assert!(db.lookup(&tool, &call).await.unwrap().is_none());
    let old_key =
        engine_core::CacheKey::for_tool_call("read_file", &json!({"path":"src/main.rs"})).unwrap();
    assert!(db.get_memoized(&old_key).await.unwrap().is_none());
}

#[tokio::test]
async fn action_gate_executor_policy_denies_even_a_previously_cached_call() {
    let (db, session, executor, registry) = fixture();
    invoke(&db, &registry, &session, true).await;
    let before = executor.calls.load(Ordering::SeqCst);
    let (status, Json(body)) = execute_local_tool_call(
        &db,
        &registry,
        ToolCallRequest {
            session_id: session.id,
            tool_name: "read_file".into(),
            arguments: json!({"path":"src/main.rs"}),
            safe_to_memoize: true,
            is_subagent: false,
        },
        &engine_core::action_gate::ActionPolicy {
            default_allow: false,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["blocked"], true);
    assert_eq!(executor.calls.load(Ordering::SeqCst), before);
}
