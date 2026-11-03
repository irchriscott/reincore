use super::*;

#[derive(Deserialize)]
pub struct ToolListQuery {
    session_id: Uuid,
}

pub async fn list_tools(
    State(state): State<AppState>,
    axum::extract::Query(query): axum::extract::Query<ToolListQuery>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    list_local_tools(
        state.db.as_ref(),
        state.tool_executors.as_ref(),
        query.session_id,
    )
    .await
}
pub async fn list_tools_single_provider(
    State(state): State<SingleProviderAppState>,
    axum::extract::Query(query): axum::extract::Query<ToolListQuery>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    list_local_tools(
        state.db.as_ref(),
        state.tool_executors.as_ref(),
        query.session_id,
    )
    .await
}
async fn list_local_tools(
    db: &crate::db::LocalDatabase,
    registry: &engine_core::ToolExecutorRegistry,
    session_id: Uuid,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let (session_id, tools) = authorized_tools(db, registry, session_id).await?;
    Ok(Json(json!({"data": tools, "session_id": session_id})))
}

pub(super) async fn authorized_tools(
    db: &crate::db::LocalDatabase,
    registry: &engine_core::ToolExecutorRegistry,
    session_id: Uuid,
) -> Result<(Uuid, Vec<engine_core::ToolDefinition>), StatusCode> {
    let session = db
        .get_active_session_in_lineage(session_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let tools = registry
        .list_local_tools_in_context(&local_context(&session))
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?;

    Ok((session.id, tools))
}

#[derive(Deserialize)]
pub struct ToolCallRequest {
    #[allow(dead_code)]
    pub(super) session_id: Uuid,
    pub(super) tool_name: String,
    pub(super) arguments: serde_json::Value,
    /// Caller consent only; the executor must independently authorize reuse.
    #[serde(default)]
    pub(super) safe_to_memoize: bool,
    /// True when this tool call originates from a subagent (a child
    /// conversation spawned by VS Code's internal tool execution).
    #[serde(default)]
    pub(super) is_subagent: bool,
}

pub async fn tool_calls(
    State(state): State<AppState>,
    Json(req): Json<ToolCallRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), StatusCode> {
    execute_local_tool_call(
        state.db.as_ref(),
        state.tool_executors.as_ref(),
        req,
        &state.action_policy,
    )
    .await
}

pub(super) async fn execute_local_tool_call(
    db: &crate::db::LocalDatabase,
    executors: &engine_core::ToolExecutorRegistry,
    req: ToolCallRequest,
    action_policy: &engine_core::action_gate::ActionPolicy,
) -> Result<(StatusCode, Json<serde_json::Value>), StatusCode> {
    let requested_at = chrono::Utc::now();
    let session = db
        .get_active_session_in_lineage(req.session_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let mut tool_call = engine_core::ToolCall {
        id: Uuid::new_v4().to_string(),
        call_type: "function".to_string(),
        function: engine_core::FunctionCall {
            name: req.tool_name.clone(),
            arguments: req.arguments.to_string(),
        },
    };

    let context = local_context(&session);
    if let Err(error) = action_policy.prepare_tool_call(&mut tool_call) {
        let report = engine_core::action_gate::denial_report(&tool_call, &error.to_string());
        engine_core::log_agent_health(db, session.id, &report)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        return Ok((
            StatusCode::FORBIDDEN,
            Json(json!({"blocked":true,"agent_health":report})),
        ));
    }
    let executor = executors
        .resolve_local_in_context(&tool_call, &context)
        .await;
    if let Err(error) = &executor {
        let report = engine_core::action_gate::denial_report(&tool_call, &error.to_string());
        engine_core::log_agent_health(db, session.id, &report)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        return Ok((
            StatusCode::FORBIDDEN,
            Json(json!({"blocked":true,"agent_health":report})),
        ));
    }

    if let Some(report) = engine_core::detect_tool_loop(db, session.id, &tool_call, 32)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    {
        db.append_event(
            session.id,
            SessionEventType::ToolCall,
            serde_json::to_value(engine_core::event::ToolCallPayload {
                tool_call: tool_call.clone(),
                requested_at: Some(requested_at),
                is_subagent: req.is_subagent,
            })
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
            None,
        )
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        engine_core::log_agent_health(db, session.id, &report)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        return Ok((
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({
                "blocked": true,
                "session_id": session.id,
                "agent_health": report,
            })),
        ));
    }

    db.append_event(
        session.id,
        SessionEventType::ToolCall,
        serde_json::to_value(engine_core::event::ToolCallPayload {
            tool_call: tool_call.clone(),
            requested_at: Some(requested_at),
            is_subagent: req.is_subagent,
        })
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
        None,
    )
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let prepared_arguments = serde_json::from_str(&tool_call.function.arguments)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let policy = if req.safe_to_memoize && context.project_path.is_some() {
        match &executor {
            Ok(executor) => executor
                .local_cache_policy(&tool_call, &context)
                .await
                .ok()
                .flatten()
                .filter(|policy| policy.is_valid()),
            Err(_) => None,
        }
    } else {
        None
    };
    let scope = serde_json::json!({"adapter": "local-session-v1", "context": context});
    let key = policy
        .as_ref()
        .map(|policy| {
            engine_core::tool_cache::ExactToolCacheEnvelope::key(
                &scope,
                policy,
                &req.tool_name,
                &prepared_arguments,
            )
        })
        .transpose()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if let (Some(key), Some(policy)) = (&key, &policy) {
        if let Some(entry) = db.get_memoized(key).await.ok().flatten() {
            let policy_current = match &executor {
                Ok(executor) => {
                    executor
                        .local_cache_policy(&tool_call, &context)
                        .await
                        .ok()
                        .flatten()
                        .as_ref()
                        == Some(policy)
                }
                Err(_) => false,
            };

            let source_matches = db
                .get_session(session.id)
                .ok()
                .flatten()
                .is_some_and(|source| local_context(&source) == context);

            if let Some(result) = engine_core::tool_cache::ExactToolCacheEnvelope::validate(
                &entry,
                &scope,
                policy,
                &req.tool_name,
                &prepared_arguments,
                &tool_call.id,
            )
            .ok()
            .flatten()
            .filter(|_| source_matches && policy_current)
            {
                db.append_event(
                    session.id,
                    SessionEventType::ToolResult,
                    serde_json::to_value(engine_core::event::ToolResultPayload {
                        result: result.clone(),
                        cache_hit: true,
                        execution_started_at: None,
                        execution_completed_at: None,
                        execution_duration_ms: None,
                        is_subagent: req.is_subagent,
                    })
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
                    None,
                )
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                return Ok((
                    StatusCode::OK,
                    Json(json!({
                        "cache_hit": true, "memoization_enabled": true,
                        "session_id": session.id, "tool_name": req.tool_name, "result": result,
                    })),
                ));
            }
        }
    }

    let execution_started_at = chrono::Utc::now();
    let execution_started = std::time::Instant::now();
    let execution = match &executor {
        Ok(executor) => {
            executor
                .execute_local_in_context(&tool_call, &context)
                .await
        }
        Err(error) => Err(engine_core::EngineError::Provider(error.to_string())),
    };

    let result = match execution {
        Ok(result) => result,
        Err(error) => {
            let execution_completed_at = chrono::Utc::now();
            let execution_duration_ms =
                u64::try_from(execution_started.elapsed().as_millis()).unwrap_or(u64::MAX);
            let result = engine_core::ToolResult {
                tool_call_id: tool_call.id.clone(),
                role: "tool".to_string(),
                name: tool_call.function.name.clone(),
                content: engine_core::privacy::current_policy()
                    .sanitize_for_logging(&error.to_string()),
                is_error: Some(true),
            };
            let _ = db
                .append_event(
                    session.id,
                    SessionEventType::ToolResult,
                    serde_json::to_value(engine_core::event::ToolResultPayload {
                        result,
                        cache_hit: false,
                        execution_started_at: Some(execution_started_at),
                        execution_completed_at: Some(execution_completed_at),
                        execution_duration_ms: Some(execution_duration_ms),
                        is_subagent: req.is_subagent,
                    })
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
                    None,
                )
                .await;
            warn!(
                "Tool execution failed: {}",
                engine_core::privacy::current_policy().sanitize_for_logging(&error.to_string())
            );
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    let execution_completed_at = chrono::Utc::now();
    let execution_duration_ms =
        u64::try_from(execution_started.elapsed().as_millis()).unwrap_or(u64::MAX);

    if let (Some(key), Some(policy), Ok(executor)) = (&key, &policy, &executor) {
        let current = executor
            .local_cache_policy(&tool_call, &context)
            .await
            .ok()
            .flatten();
        let source_matches = db
            .get_session(session.id)
            .ok()
            .flatten()
            .is_some_and(|source| local_context(&source) == context);
        if current.as_ref() == Some(policy) && source_matches && result.name == req.tool_name {
            if let Ok(Some(entry)) = engine_core::tool_cache::ExactToolCacheEnvelope::entry(
                scope,
                policy.clone(),
                prepared_arguments.clone(),
                &result,
            ) {
                let _ = db.put_memoized(key.clone(), entry).await;
            }
        }
    }

    db.append_event(
        session.id,
        SessionEventType::ToolResult,
        serde_json::to_value(engine_core::event::ToolResultPayload {
            result: result.clone(),
            cache_hit: false,
            execution_started_at: Some(execution_started_at),
            execution_completed_at: Some(execution_completed_at),
            execution_duration_ms: Some(execution_duration_ms),
            is_subagent: req.is_subagent,
        })
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
        None,
    )
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok((
        StatusCode::OK,
        Json(json!({
            "cache_hit": false,
            "memoization_enabled": policy.is_some(),
            "session_id": session.id,
            "tool_name": req.tool_name,
            "result": result,
        })),
    ))
}

fn local_context(session: &Session) -> engine_core::tool_cache::LocalToolCacheContext {
    // Relative/unresolvable paths cannot define a stable repository boundary.
    let project_path = session
        .project_path
        .as_deref()
        .map(std::path::Path::new)
        .filter(|path| path.is_absolute())
        .and_then(|path| std::fs::canonicalize(path).ok())
        .and_then(|path| path.to_str().map(str::to_owned));
    engine_core::tool_cache::LocalToolCacheContext {
        session_id: session.id,
        project_path,
        workspace_id: session.workspace_id,
        project_id: session.project_id,
        user_id: session.user_id,
        client_id: session.client_id.clone(),
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod mcp_tests;
