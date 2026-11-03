use super::lifecycle::*;
use super::tools::{execute_local_tool_call, ToolCallRequest};
use super::*;

pub async fn create_session_single_provider(
    State(state): State<SingleProviderAppState>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<Json<CreateSessionResponse>, StatusCode> {
    let is_subagent = req.is_subagent.unwrap_or(false) || req.subagent_parent_session_id.is_some();
    let subagent_parent_session_id = req.subagent_parent_session_id.clone();
    let subagent_spawn_tool_call_id = req.subagent_spawn_tool_call_id.clone();
    let subagent_spawn_request_group_id = req.subagent_spawn_request_group_id.clone();
    let requested_project_name = req
        .project_name
        .unwrap_or_else(|| state.default_project_name.as_ref().clone());
    let requested_project_path = req
        .project_path
        .unwrap_or_else(|| state.default_project_path.as_ref().clone());
    let (project_id, project_name, project_path) = resolve_local_project_context(
        state.db.as_ref(),
        requested_project_name,
        requested_project_path,
    )
    .await?;

    let mut session = Session::new_tracked(
        req.task_summary,
        req.budget_cents,
        Some(project_name),
        Some(project_path),
        Some(req.client_id.unwrap_or_else(|| "local".to_string())),
        req.external_session_id,
    )
    .map_err(|_| StatusCode::BAD_REQUEST)?;
    session.project_id = project_id;
    session.mode = req
        .mode
        .as_deref()
        .and_then(|mode| mode.parse::<SessionMode>().ok())
        .unwrap_or(SessionMode::Interactive);
    session.is_subagent = is_subagent;
    session.subagent_parent_session_id = subagent_parent_session_id;
    session.subagent_spawn_tool_call_id = subagent_spawn_tool_call_id;
    session.subagent_spawn_request_group_id = subagent_spawn_request_group_id;

    let id = session.id;
    state
        .db
        .insert_session_async(session.clone())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    state.sessions.lock().await.insert(id, session.clone());

    Ok(Json(CreateSessionResponse {
        session_id: id.to_string(),
        status: SessionStatus::Active.as_str().to_string(),
        mode: session.mode.as_str().to_string(),
        budget_cents: req.budget_cents,
        project_id: session.project_id,
        project_name: session.project_name.clone(),
        project_path: session.project_path.clone(),
        external_session_id: session.external_session_id.clone(),
        is_subagent: session.is_subagent,
        subagent_parent_session_id: session.subagent_parent_session_id.clone(),
        subagent_spawn_tool_call_id: session.subagent_spawn_tool_call_id.clone(),
        subagent_spawn_request_group_id: session.subagent_spawn_request_group_id.clone(),
    }))
}

pub async fn list_sessions_single_provider(
    State(state): State<SingleProviderAppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<ListSessionsResponse>, StatusCode> {
    let limit = params
        .get("limit")
        .map(|v| v.parse::<usize>())
        .transpose()
        .map_err(|_| StatusCode::BAD_REQUEST)?
        .unwrap_or(if params.contains_key("before") {
            100
        } else {
            usize::MAX
        });
    if params.contains_key("limit") && !(1..=200).contains(&limit) {
        return Err(StatusCode::BAD_REQUEST);
    }
    let filter = crate::db::SessionFilter {
        limit: (limit != usize::MAX).then(|| limit + 1),
        before: params.get("before").map(String::as_str),
        path: params.get("path").map(|s| s.as_str()),
        status: params.get("status").map(|s| s.as_str()),
        mode: params.get("mode").map(|s| s.as_str()),
        project_id: params.get("project_id").map(|s| s.as_str()),
        date_from: params.get("date_from").map(|s| s.as_str()),
        date_to: params.get("date_to").map(|s| s.as_str()),
    };
    let include_checkpoints = params
        .get("checkpoints")
        .map(|s| s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let sessions = state
        .db
        .list_sessions_async(filter)
        .await
        .map_err(|e| match e {
            engine_core::EngineError::InvalidInput(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        })?;

    let mut sessions = sessions;
    let has_more = sessions.len() > limit;
    sessions.truncate(limit);
    let next_cursor = if has_more {
        sessions
            .last()
            .map(|s| format!("{}|{}", s.updated_at.to_rfc3339(), s.id))
    } else {
        None
    };
    attach_cost_outcomes(state.db.as_ref(), &mut sessions).await;
    let mut items = Vec::with_capacity(sessions.len());
    for session in sessions {
        let checkpoint_token = if include_checkpoints {
            state
                .db
                .get_latest_checkpoint_token_async(session.id)
                .await
                .ok()
                .flatten()
        } else {
            None
        };
        items.push(SessionListItem {
            session_id: session.id.to_string(),
            task_summary: session.task_summary,
            status: session.status.as_str().to_string(),
            mode: session.mode.as_str().to_string(),
            project_path: session.project_path,
            project_name: session.project_name,
            project_id: session.project_id,
            external_session_id: session.external_session_id,
            parent_session_id: session.parent_session_id,
            root_session_id: session.root_session_id,
            rollover_index: session.rollover_index,
            context_tokens: session.context_tokens,
            spend_cents: session.spend_cents,
            spend_nano_usd: session.spend_nano_usd,
            unknown_cost_calls: session.unknown_cost_calls,
            cost_outcome: session.cost_outcome,
            predicted_high_cents: session.predicted_high_cents,
            cost_accuracy_bps: session.cost_accuracy_bps,
            cost_accuracy_samples: session.cost_accuracy_samples,
            cache_hit_rate_bps: session.cache_hit_rate_bps,
            cache_hit_rate_samples: session.cache_hit_rate_samples,
            total_input_tokens: session.total_input_tokens,
            total_output_tokens: session.total_output_tokens,
            is_subagent: session.is_subagent,
            subagent_parent_session_id: session.subagent_parent_session_id,
            subagent_spawn_tool_call_id: session.subagent_spawn_tool_call_id,
            subagent_spawn_request_group_id: session.subagent_spawn_request_group_id,
            checkpoint_token,
            created_at: session.created_at,
            updated_at: session.updated_at,
        });
    }

    Ok(Json(ListSessionsResponse {
        sessions: items,
        next_cursor,
    }))
}

pub async fn get_session_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<Session>, StatusCode> {
    let mut session = state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;
    attach_cost_outcome(state.db.as_ref(), &mut session).await;
    Ok(Json(session))
}

/// GET /v1/sessions/:id/subagents - returns child subagent sessions for a
/// parent session. The parent is looked up by its UUID, and its
/// `external_session_id` is used to find child sessions where
/// `subagent_parent_session_id` matches.
pub async fn list_subagent_sessions_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<Vec<SessionListItem>>, StatusCode> {
    let session = state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let parent_ext_id = session
        .external_session_id
        .unwrap_or_else(|| session.id.to_string());
    let subagents = state
        .db
        .list_subagent_sessions_async(parent_ext_id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let mut subagents = subagents;
    attach_cost_outcomes(state.db.as_ref(), &mut subagents).await;
    let mut items = Vec::with_capacity(subagents.len());
    for s in subagents {
        items.push(SessionListItem {
            session_id: s.id.to_string(),
            task_summary: s.task_summary,
            status: s.status.as_str().to_string(),
            mode: s.mode.as_str().to_string(),
            project_path: s.project_path,
            project_name: s.project_name,
            project_id: s.project_id,
            external_session_id: s.external_session_id,
            parent_session_id: s.parent_session_id,
            root_session_id: s.root_session_id,
            rollover_index: s.rollover_index,
            context_tokens: s.context_tokens,
            spend_cents: s.spend_cents,
            spend_nano_usd: s.spend_nano_usd,
            unknown_cost_calls: s.unknown_cost_calls,
            cost_outcome: s.cost_outcome,
            predicted_high_cents: s.predicted_high_cents,
            cost_accuracy_bps: s.cost_accuracy_bps,
            cost_accuracy_samples: s.cost_accuracy_samples,
            cache_hit_rate_bps: s.cache_hit_rate_bps,
            cache_hit_rate_samples: s.cache_hit_rate_samples,
            total_input_tokens: s.total_input_tokens,
            total_output_tokens: s.total_output_tokens,
            is_subagent: s.is_subagent,
            subagent_parent_session_id: s.subagent_parent_session_id,
            subagent_spawn_tool_call_id: s.subagent_spawn_tool_call_id,
            subagent_spawn_request_group_id: s.subagent_spawn_request_group_id,
            checkpoint_token: None,
            created_at: s.created_at,
            updated_at: s.updated_at,
        });
    }
    Ok(Json(items))
}

#[derive(Deserialize)]
pub struct SessionEventPageQuery {
    pub since_seq: Option<i64>,
    pub until_seq: Option<i64>,
    pub limit: Option<usize>,
}

#[derive(Serialize)]
pub struct SessionEventPage {
    pub events: Vec<engine_core::SessionEvent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_since_seq: Option<i64>,
}

#[derive(Deserialize)]
pub struct SessionEventStreamQuery {
    pub after_seq: Option<i64>,
}

fn event_stream_cursor(headers: &HeaderMap, query: &SessionEventStreamQuery) -> i64 {
    let header_cursor = headers
        .get("last-event-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(0);
    query.after_seq.unwrap_or(0).max(header_cursor).max(0)
}

async fn session_event_stream_response(
    db: std::sync::Arc<crate::db::LocalDatabase>,
    headers: HeaderMap,
    id: Uuid,
    query: SessionEventStreamQuery,
) -> Result<axum::response::Response, StatusCode> {
    db.get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let after_seq = event_stream_cursor(&headers, &query);
    let events =
        engine_core::live_session_events(db.clone(), db.event_hub(), id, after_seq).map(|result| {
            let event = match result {
                Ok(event) => Event::default()
                    .event("session_event")
                    .id(event.seq.to_string())
                    .data(json!({ "schema_version": 1, "event": event }).to_string()),
                Err(error) => Event::default()
                    .event("stream_error")
                    .data(json!({ "error": error.to_string() }).to_string()),
            };
            Ok::<Event, Infallible>(event)
        });

    let mut response = Sse::new(events)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response();
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-cache, no-transform"),
    );
    response.headers_mut().insert(
        axum::http::HeaderName::from_static("x-accel-buffering"),
        axum::http::HeaderValue::from_static("no"),
    );
    Ok(response)
}

async fn read_session_event_page(
    db: &crate::db::LocalDatabase,
    id: Uuid,
    query: SessionEventPageQuery,
) -> Result<SessionEventPage, StatusCode> {
    let limit = query.limit.unwrap_or(100).clamp(1, 500);
    let mut events = db
        .read_event_page(
            id,
            query.since_seq,
            query.until_seq,
            limit.saturating_add(1),
        )
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let has_more = events.len() > limit;
    events.truncate(limit);
    let next_since_seq = has_more
        .then(|| events.last().map(|event| event.seq.saturating_add(1)))
        .flatten();
    Ok(SessionEventPage {
        events,
        next_since_seq,
    })
}

pub async fn get_session_events_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<Uuid>,
    axum::extract::Query(query): axum::extract::Query<SessionEventPageQuery>,
) -> Result<Json<SessionEventPage>, StatusCode> {
    Ok(Json(
        read_session_event_page(state.db.as_ref(), id, query).await?,
    ))
}

/// GET /v1/sessions/:id/events/stream - replaying live session events.
pub async fn stream_session_events_single_provider(
    State(state): State<SingleProviderAppState>,
    headers: HeaderMap,
    Path(id): Path<Uuid>,
    axum::extract::Query(query): axum::extract::Query<SessionEventStreamQuery>,
) -> Result<axum::response::Response, StatusCode> {
    session_event_stream_response(state.db, headers, id, query).await
}

/// GET /v1/sessions/:id/timeline - lightweight task/request projection.
pub async fn get_session_timeline_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<engine_core::SessionTimeline>, StatusCode> {
    let events = state
        .db
        .read_timeline_events(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(engine_core::session_timeline(&events)))
}

/// GET /v1/sessions/:id/events/page - bounded raw event range.
pub async fn get_session_event_page_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<Uuid>,
    axum::extract::Query(query): axum::extract::Query<SessionEventPageQuery>,
) -> Result<Json<SessionEventPage>, StatusCode> {
    Ok(Json(
        read_session_event_page(state.db.as_ref(), id, query).await?,
    ))
}

/// GET /v1/sessions/:id/event/:seq - one complete audit event for the inspector.
pub async fn get_session_event_single_provider(
    State(state): State<SingleProviderAppState>,
    Path((id, seq)): Path<(Uuid, i64)>,
) -> Result<Json<engine_core::SessionEvent>, StatusCode> {
    let event = state
        .db
        .read_events(id, Some(seq), Some(1))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .into_iter()
        .find(|event| event.seq == seq)
        .ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(event))
}

/// GET /v1/sessions/:id/tasks - durable task groups projected from LLM events.
pub async fn get_session_tasks_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<Vec<engine_core::SessionTask>>, StatusCode> {
    state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;
    let events = state
        .db
        .read_timeline_events(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(engine_core::session_tasks(&events)))
}

#[derive(Deserialize)]
pub struct CompleteSessionRequest {
    pub reason: Option<String>,
    pub outcome_report: Option<engine_core::cost_outcome::OutcomeReport>,
}

async fn complete_session_in_db(
    db: &crate::db::LocalDatabase,
    id: Uuid,
    reason: Option<String>,
    outcome_report: Option<engine_core::cost_outcome::OutcomeReport>,
) -> Result<Session, StatusCode> {
    let mut session = db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;
    if outcome_report
        .as_ref()
        .is_some_and(|report| report.evidence.trim().is_empty())
    {
        return Err(StatusCode::BAD_REQUEST);
    }
    if session.status != SessionStatus::Completed || outcome_report.is_some() {
        let reason = reason
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "completed by caller".to_string());
        db.append_event(
            id,
            SessionEventType::SessionCompleted,
            serde_json::to_value(engine_core::event::SessionCompletedPayload {
                reason,
                outcome_report: outcome_report,
            })
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
            None,
        )
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        session.status = SessionStatus::Completed;
        session.updated_at = chrono::Utc::now();
        db.insert_session_async(session.clone())
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }
    attach_cost_outcome(db, &mut session).await;
    Ok(session)
}

pub async fn complete_session_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<CompleteSessionRequest>,
) -> Result<Json<Session>, StatusCode> {
    let session =
        complete_session_in_db(state.db.as_ref(), id, req.reason, req.outcome_report).await?;
    state.sessions.lock().await.insert(id, session.clone());
    Ok(Json(session))
}

pub async fn pause_session_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<PauseSessionResponse>, StatusCode> {
    let mut session = state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let latest_seq = state
        .db
        .latest_seq(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .unwrap_or(0);

    let token = state
        .checkpoints
        .create(&mut session, latest_seq, "user-pause")
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    state
        .db
        .insert_session_async(session.clone())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    state
        .db
        .insert_checkpoint_async(token.clone())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    state.sessions.lock().await.insert(id, session);

    Ok(Json(PauseSessionResponse {
        session_id: id.to_string(),
        checkpoint_token: token.token,
        expires_at: token.expires_at.to_rfc3339(),
    }))
}

pub async fn resume_session_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(token_str): Path<String>,
) -> Result<Json<ResumeSessionResponse>, StatusCode> {
    let token = state
        .checkpoints
        .validate(&token_str)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let mut session = state
        .db
        .get_session_async(token.session_id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    state
        .checkpoints
        .resume(&token, &mut session, state.db.as_ref())
        .await
        .map_err(|error| checkpoint_resume_error_to_status(&error))?;

    state
        .db
        .insert_session_async(session.clone())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    state
        .sessions
        .lock()
        .await
        .insert(token.session_id, session.clone());

    Ok(Json(ResumeSessionResponse {
        session_id: token.session_id.to_string(),
        status: SessionStatus::Active.as_str().to_string(),
        resumed_at_seq: token.last_event_seq,
    }))
}

pub async fn tool_calls_single_provider(
    State(state): State<SingleProviderAppState>,
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

#[derive(Deserialize)]
pub struct CreateSessionRequest {
    task_summary: String,
    budget_cents: i64,
    project_path: Option<String>,
    project_name: Option<String>,
    client_id: Option<String>,
    external_session_id: Option<String>,
    mode: Option<String>,
    #[serde(default)]
    is_subagent: Option<bool>,
    subagent_parent_session_id: Option<String>,
    subagent_spawn_tool_call_id: Option<String>,
    subagent_spawn_request_group_id: Option<String>,
}

#[derive(Serialize)]
pub struct CreateSessionResponse {
    session_id: String,
    status: String,
    mode: String,
    budget_cents: i64,
    project_id: Option<Uuid>,
    project_name: Option<String>,
    project_path: Option<String>,
    external_session_id: Option<String>,
    is_subagent: bool,
    subagent_parent_session_id: Option<String>,
    subagent_spawn_tool_call_id: Option<String>,
    subagent_spawn_request_group_id: Option<String>,
}

pub async fn create_session(
    State(state): State<AppState>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<Json<CreateSessionResponse>, StatusCode> {
    let is_subagent = req.is_subagent.unwrap_or(false) || req.subagent_parent_session_id.is_some();
    let subagent_parent_session_id = req.subagent_parent_session_id.clone();
    let subagent_spawn_tool_call_id = req.subagent_spawn_tool_call_id.clone();
    let subagent_spawn_request_group_id = req.subagent_spawn_request_group_id.clone();
    let requested_project_name = req
        .project_name
        .unwrap_or_else(|| state.default_project_name.as_ref().clone());
    let requested_project_path = req
        .project_path
        .unwrap_or_else(|| state.default_project_path.as_ref().clone());
    let (project_id, project_name, project_path) = resolve_local_project_context(
        state.db.as_ref(),
        requested_project_name,
        requested_project_path,
    )
    .await?;
    let mut session = Session::new_tracked(
        req.task_summary,
        req.budget_cents,
        Some(project_name),
        Some(project_path),
        Some(req.client_id.unwrap_or_else(|| "local".to_string())),
        req.external_session_id,
    )
    .map_err(|_| StatusCode::BAD_REQUEST)?;
    session.project_id = project_id;
    session.mode = req
        .mode
        .as_deref()
        .and_then(|mode| mode.parse::<SessionMode>().ok())
        .unwrap_or(SessionMode::Interactive);
    session.is_subagent = is_subagent;
    session.subagent_parent_session_id = subagent_parent_session_id;
    session.subagent_spawn_tool_call_id = subagent_spawn_tool_call_id;
    session.subagent_spawn_request_group_id = subagent_spawn_request_group_id;
    let id = session.id;
    state
        .db
        .insert_session_async(session.clone())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    state.sessions.lock().await.insert(id, session.clone());

    Ok(Json(CreateSessionResponse {
        session_id: id.to_string(),
        status: SessionStatus::Active.as_str().to_string(),
        mode: session.mode.as_str().to_string(),
        budget_cents: req.budget_cents,
        project_id: session.project_id,
        project_name: session.project_name.clone(),
        project_path: session.project_path.clone(),
        external_session_id: session.external_session_id.clone(),
        is_subagent: session.is_subagent,
        subagent_parent_session_id: session.subagent_parent_session_id.clone(),
        subagent_spawn_tool_call_id: session.subagent_spawn_tool_call_id.clone(),
        subagent_spawn_request_group_id: session.subagent_spawn_request_group_id.clone(),
    }))
}

#[derive(Serialize)]
pub struct SessionListItem {
    session_id: String,
    task_summary: String,
    status: String,
    mode: String,
    project_path: Option<String>,
    project_name: Option<String>,
    project_id: Option<Uuid>,
    external_session_id: Option<String>,
    parent_session_id: Option<Uuid>,
    root_session_id: Option<Uuid>,
    rollover_index: u32,
    context_tokens: u64,
    spend_cents: i64,
    spend_nano_usd: Option<i64>,
    unknown_cost_calls: u64,
    cost_outcome: Option<engine_core::CostOutcome>,
    /// Latest pre-flight prediction's high bound, in cents. `None` for
    /// sessions created before the cost-calibration migration or free
    /// providers whose prediction is always zero.
    predicted_high_cents: Option<i64>,
    /// Running sample-weighted mean of `actual / predicted_high` in basis
    /// points (10_000 = prediction matched exactly). `None` until at
    /// least one calibrated call has been recorded.
    #[serde(
        rename = "actual_to_predicted_high_ratio_bps",
        alias = "cost_accuracy_bps"
    )]
    cost_accuracy_bps: Option<i64>,
    /// Number of LLM calls that contributed to `cost_accuracy_bps`.
    cost_accuracy_samples: Option<i64>,
    /// Running sample-weighted mean of `cached_prompt_tokens / prompt_tokens`
    /// in basis points (10_000 = 100% cache hit). `None` until at least one
    /// call with non-zero `prompt_tokens` has been recorded.
    #[serde(
        rename = "legacy_mean_call_cache_hit_rate_bps",
        alias = "cache_hit_rate_bps"
    )]
    cache_hit_rate_bps: Option<i64>,
    /// Number of LLM calls that contributed to `cache_hit_rate_bps`.
    cache_hit_rate_samples: Option<i64>,
    total_input_tokens: u64,
    total_output_tokens: u64,
    /// Whether this session was created by a subagent.
    is_subagent: bool,
    /// Parent session ID if this is a subagent session.
    subagent_parent_session_id: Option<String>,
    /// Exact delegation tool call that spawned this child.
    subagent_spawn_tool_call_id: Option<String>,
    /// Parent request group containing the spawn call.
    subagent_spawn_request_group_id: Option<String>,
    checkpoint_token: Option<String>,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Serialize)]
pub struct ListSessionsResponse {
    pub next_cursor: Option<String>,
    sessions: Vec<SessionListItem>,
}

#[derive(Serialize)]
pub struct ListProjectsResponse {
    pub projects: Vec<engine_core::ProjectUsage>,
}

pub async fn list_projects(
    State(state): State<AppState>,
) -> Result<Json<ListProjectsResponse>, StatusCode> {
    let projects = state
        .db
        .list_project_usage_async()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(ListProjectsResponse { projects }))
}

pub async fn list_projects_single_provider(
    State(state): State<SingleProviderAppState>,
) -> Result<Json<ListProjectsResponse>, StatusCode> {
    let projects = state
        .db
        .list_project_usage_async()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(ListProjectsResponse { projects }))
}

#[derive(Serialize)]
pub struct ListLocalProjectsResponse {
    projects: Vec<crate::db::LocalProject>,
}

#[derive(Serialize)]
pub struct ResolveLocalProjectResponse {
    available: bool,
    project: Option<crate::db::LocalProject>,
}

#[derive(Deserialize)]
pub struct CreateLocalProjectRequest {
    name: String,
    project_path: String,
}

#[derive(Deserialize)]
pub struct AssignLocalProjectDirectoryRequest {
    project_path: String,
}

fn local_project_mutation_status(error: &engine_core::EngineError) -> StatusCode {
    let message = error.to_string().to_ascii_lowercase();
    if message.contains("already assigned") || message.contains("unique constraint") {
        StatusCode::CONFLICT
    } else if message.contains("not found") {
        StatusCode::NOT_FOUND
    } else if message.contains("project path") || message.contains("project name") {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

async fn resolve_local_project_from_db(
    db: &crate::db::LocalDatabase,
    params: std::collections::HashMap<String, String>,
) -> Result<Json<ResolveLocalProjectResponse>, StatusCode> {
    let path = params
        .get("path")
        .map(String::as_str)
        .filter(|path| !path.trim().is_empty())
        .ok_or(StatusCode::BAD_REQUEST)?;
    let project = db
        .get_local_project_by_path_async(path.to_string())
        .await
        .map_err(|error| local_project_mutation_status(&error))?;
    Ok(Json(ResolveLocalProjectResponse {
        available: project.is_some(),
        project,
    }))
}

pub async fn list_local_projects(
    State(state): State<AppState>,
) -> Result<Json<ListLocalProjectsResponse>, StatusCode> {
    let projects = state
        .db
        .list_local_projects_async()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(ListLocalProjectsResponse { projects }))
}

pub async fn list_local_projects_single_provider(
    State(state): State<SingleProviderAppState>,
) -> Result<Json<ListLocalProjectsResponse>, StatusCode> {
    let projects = state
        .db
        .list_local_projects_async()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(ListLocalProjectsResponse { projects }))
}

pub async fn resolve_local_project(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<ResolveLocalProjectResponse>, StatusCode> {
    resolve_local_project_from_db(state.db.as_ref(), params).await
}

pub async fn resolve_local_project_single_provider(
    State(state): State<SingleProviderAppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<ResolveLocalProjectResponse>, StatusCode> {
    resolve_local_project_from_db(state.db.as_ref(), params).await
}

pub async fn create_local_project(
    State(state): State<AppState>,
    Json(req): Json<CreateLocalProjectRequest>,
) -> Result<Json<crate::db::LocalProject>, StatusCode> {
    state
        .db
        .create_local_project_async(req.name, req.project_path)
        .await
        .map(Json)
        .map_err(|error| local_project_mutation_status(&error))
}

pub async fn create_local_project_single_provider(
    State(state): State<SingleProviderAppState>,
    Json(req): Json<CreateLocalProjectRequest>,
) -> Result<Json<crate::db::LocalProject>, StatusCode> {
    state
        .db
        .create_local_project_async(req.name, req.project_path)
        .await
        .map(Json)
        .map_err(|error| local_project_mutation_status(&error))
}

pub async fn assign_local_project_directory(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<AssignLocalProjectDirectoryRequest>,
) -> Result<Json<crate::db::LocalProject>, StatusCode> {
    state
        .db
        .assign_local_project_directory_async(id, req.project_path)
        .await
        .map(Json)
        .map_err(|error| local_project_mutation_status(&error))
}

#[derive(Deserialize)]
pub struct RenameLocalProjectRequest {
    name: String,
}

pub async fn rename_local_project(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<RenameLocalProjectRequest>,
) -> Result<Json<crate::db::LocalProject>, StatusCode> {
    state
        .db
        .rename_local_project_async(id, req.name)
        .await
        .map(Json)
        .map_err(|error| local_project_mutation_status(&error))
}

pub async fn rename_local_project_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<RenameLocalProjectRequest>,
) -> Result<Json<crate::db::LocalProject>, StatusCode> {
    state
        .db
        .rename_local_project_async(id, req.name)
        .await
        .map(Json)
        .map_err(|error| local_project_mutation_status(&error))
}

pub async fn assign_local_project_directory_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<AssignLocalProjectDirectoryRequest>,
) -> Result<Json<crate::db::LocalProject>, StatusCode> {
    state
        .db
        .assign_local_project_directory_async(id, req.project_path)
        .await
        .map(Json)
        .map_err(|error| local_project_mutation_status(&error))
}

// ---- Session rename -------------------------------------------------------

#[derive(Deserialize)]
pub struct RenameSessionRequest {
    pub task_summary: String,
}

/// `PATCH /v1/sessions/:id` - rename a session by updating its task summary.
pub async fn rename_session(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<RenameSessionRequest>,
) -> Result<Json<Session>, StatusCode> {
    rename_session_inner(&state.db, id, req.task_summary).await
}

/// Single-provider variant of [`rename_session`].
pub async fn rename_session_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<RenameSessionRequest>,
) -> Result<Json<Session>, StatusCode> {
    rename_session_inner(&state.db, id, req.task_summary).await
}

async fn rename_session_inner(
    db: &crate::db::LocalDatabase,
    id: Uuid,
    task_summary: String,
) -> Result<Json<Session>, StatusCode> {
    let trimmed = task_summary.trim();
    if trimmed.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }
    db.update_session_summary_async(id, trimmed.to_string())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let session = db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(session))
}

/// Token-weighted cache utilization from recorded call events.
pub async fn cache_hit_rate(
    State(state): State<AppState>,
) -> Result<Json<engine_core::CacheHitRateBreakdown>, StatusCode> {
    let sessions = state
        .db
        .list_sessions_async(crate::db::SessionFilter::default())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(
        aggregate_cache_hit_rate_from_sessions(state.db.as_ref(), "all".to_string(), sessions)
            .await?,
    ))
}

/// `GET /v1/cache-hit-rate` - single-provider variant.
pub async fn cache_hit_rate_single_provider(
    State(state): State<SingleProviderAppState>,
) -> Result<Json<engine_core::CacheHitRateBreakdown>, StatusCode> {
    let sessions = state
        .db
        .list_sessions_async(crate::db::SessionFilter::default())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(
        aggregate_cache_hit_rate_from_sessions(state.db.as_ref(), "all".to_string(), sessions)
            .await?,
    ))
}

/// `GET /v1/checkpoint-redemptions` - successful checkpoint resumptions
/// across all top-level local sessions.
pub async fn checkpoint_redemptions(
    State(state): State<AppState>,
) -> Result<Json<engine_core::CheckpointRedemptionBreakdown>, StatusCode> {
    state
        .db
        .aggregate_checkpoint_redemptions_async()
        .await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

/// Single-provider variant of [`checkpoint_redemptions`].
pub async fn checkpoint_redemptions_single_provider(
    State(state): State<SingleProviderAppState>,
) -> Result<Json<engine_core::CheckpointRedemptionBreakdown>, StatusCode> {
    state
        .db
        .aggregate_checkpoint_redemptions_async()
        .await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

/// Aggregate cache-prefix hit rate in Rust from a session list. Mirrors the
/// cloud `aggregate_cache_hit_rate_from_sessions` fallback path.
async fn aggregate_cache_hit_rate_from_sessions(
    db: &crate::db::LocalDatabase,
    scope: String,
    sessions: Vec<Session>,
) -> Result<engine_core::CacheHitRateBreakdown, StatusCode> {
    let mut accumulator = engine_core::diagnostics::DiagnosticAccumulator::default();
    for session in sessions {
        accumulator
            .append_session(db, session.id)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }
    let mut result = accumulator.finish().cache;
    result.scope = scope;
    Ok(result)
}

pub async fn list_sessions(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<ListSessionsResponse>, StatusCode> {
    let limit = params
        .get("limit")
        .map(|v| v.parse::<usize>())
        .transpose()
        .map_err(|_| StatusCode::BAD_REQUEST)?
        .unwrap_or(if params.contains_key("before") {
            100
        } else {
            usize::MAX
        });
    if params.contains_key("limit") && !(1..=200).contains(&limit) {
        return Err(StatusCode::BAD_REQUEST);
    }
    let filter = crate::db::SessionFilter {
        limit: (limit != usize::MAX).then(|| limit + 1),
        before: params.get("before").map(String::as_str),
        path: params.get("path").map(|s| s.as_str()),
        status: params.get("status").map(|s| s.as_str()),
        mode: params.get("mode").map(|s| s.as_str()),
        project_id: params.get("project_id").map(|s| s.as_str()),
        date_from: params.get("date_from").map(|s| s.as_str()),
        date_to: params.get("date_to").map(|s| s.as_str()),
    };
    let include_checkpoints = params
        .get("checkpoints")
        .map(|s| s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let sessions = state
        .db
        .list_sessions_async(filter)
        .await
        .map_err(|e| match e {
            engine_core::EngineError::InvalidInput(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        })?;

    let mut sessions = sessions;
    let has_more = sessions.len() > limit;
    sessions.truncate(limit);
    let next_cursor = if has_more {
        sessions
            .last()
            .map(|s| format!("{}|{}", s.updated_at.to_rfc3339(), s.id))
    } else {
        None
    };
    attach_cost_outcomes(state.db.as_ref(), &mut sessions).await;
    let mut items = Vec::with_capacity(sessions.len());
    for session in sessions {
        let checkpoint_token = if include_checkpoints {
            state
                .db
                .get_latest_checkpoint_token_async(session.id)
                .await
                .ok()
                .flatten()
        } else {
            None
        };

        items.push(SessionListItem {
            session_id: session.id.to_string(),
            task_summary: session.task_summary,
            status: session.status.as_str().to_string(),
            mode: session.mode.as_str().to_string(),
            project_path: session.project_path,
            project_name: session.project_name,
            project_id: session.project_id,
            external_session_id: session.external_session_id,
            parent_session_id: session.parent_session_id,
            root_session_id: session.root_session_id,
            rollover_index: session.rollover_index,
            context_tokens: session.context_tokens,
            spend_cents: session.spend_cents,
            spend_nano_usd: session.spend_nano_usd,
            unknown_cost_calls: session.unknown_cost_calls,
            cost_outcome: session.cost_outcome,
            predicted_high_cents: session.predicted_high_cents,
            cost_accuracy_bps: session.cost_accuracy_bps,
            cost_accuracy_samples: session.cost_accuracy_samples,
            cache_hit_rate_bps: session.cache_hit_rate_bps,
            cache_hit_rate_samples: session.cache_hit_rate_samples,
            total_input_tokens: session.total_input_tokens,
            total_output_tokens: session.total_output_tokens,
            is_subagent: session.is_subagent,
            subagent_parent_session_id: session.subagent_parent_session_id,
            subagent_spawn_tool_call_id: session.subagent_spawn_tool_call_id,
            subagent_spawn_request_group_id: session.subagent_spawn_request_group_id,
            checkpoint_token,
            created_at: session.created_at,
            updated_at: session.updated_at,
        });
    }

    Ok(Json(ListSessionsResponse {
        sessions: items,
        next_cursor,
    }))
}

pub async fn get_session(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<Session>, StatusCode> {
    let mut session = state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;
    attach_cost_outcome(state.db.as_ref(), &mut session).await;
    Ok(Json(session))
}

/// GET /v1/sessions/:id/subagents - multi-provider variant.
pub async fn list_subagent_sessions(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<Vec<SessionListItem>>, StatusCode> {
    let session = state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let parent_ext_id = session
        .external_session_id
        .unwrap_or_else(|| session.id.to_string());
    let subagents = state
        .db
        .list_subagent_sessions_async(parent_ext_id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let mut subagents = subagents;
    attach_cost_outcomes(state.db.as_ref(), &mut subagents).await;
    let mut items = Vec::with_capacity(subagents.len());
    for s in subagents {
        items.push(SessionListItem {
            session_id: s.id.to_string(),
            task_summary: s.task_summary,
            status: s.status.as_str().to_string(),
            mode: s.mode.as_str().to_string(),
            project_path: s.project_path,
            project_name: s.project_name,
            project_id: s.project_id,
            external_session_id: s.external_session_id,
            parent_session_id: s.parent_session_id,
            root_session_id: s.root_session_id,
            rollover_index: s.rollover_index,
            context_tokens: s.context_tokens,
            spend_cents: s.spend_cents,
            spend_nano_usd: s.spend_nano_usd,
            unknown_cost_calls: s.unknown_cost_calls,
            cost_outcome: s.cost_outcome,
            predicted_high_cents: s.predicted_high_cents,
            cost_accuracy_bps: s.cost_accuracy_bps,
            cost_accuracy_samples: s.cost_accuracy_samples,
            cache_hit_rate_bps: s.cache_hit_rate_bps,
            cache_hit_rate_samples: s.cache_hit_rate_samples,
            total_input_tokens: s.total_input_tokens,
            total_output_tokens: s.total_output_tokens,
            is_subagent: s.is_subagent,
            subagent_parent_session_id: s.subagent_parent_session_id,
            subagent_spawn_tool_call_id: s.subagent_spawn_tool_call_id,
            subagent_spawn_request_group_id: s.subagent_spawn_request_group_id,
            checkpoint_token: None,
            created_at: s.created_at,
            updated_at: s.updated_at,
        });
    }
    Ok(Json(items))
}

async fn attach_cost_outcome(db: &crate::db::LocalDatabase, session: &mut Session) {
    attach_cost_outcomes(db, std::slice::from_mut(session)).await;
}

async fn attach_cost_outcomes(db: &crate::db::LocalDatabase, sessions: &mut [Session]) {
    if let Ok(mut outcomes) = db.session_outcome_summaries(sessions).await {
        for session in sessions {
            session.cost_outcome = outcomes.remove(&session.id);
        }
    }
}

/// Expensive call diagnostics are explicitly requested, never part of list/detail refreshes.
pub async fn get_session_diagnostics(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<engine_core::diagnostics::CallDiagnostics>, StatusCode> {
    session_diagnostics_in_db(state.db.as_ref(), id)
        .await
        .map(Json)
}

pub async fn get_session_diagnostics_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<engine_core::diagnostics::CallDiagnostics>, StatusCode> {
    session_diagnostics_in_db(state.db.as_ref(), id)
        .await
        .map(Json)
}

async fn session_diagnostics_in_db(
    db: &crate::db::LocalDatabase,
    id: Uuid,
) -> Result<engine_core::diagnostics::CallDiagnostics, StatusCode> {
    db.get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;
    let mut accumulator = engine_core::diagnostics::DiagnosticAccumulator::default();
    accumulator
        .append_session(db, id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(accumulator.finish())
}

pub async fn get_session_events(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    axum::extract::Query(query): axum::extract::Query<SessionEventPageQuery>,
) -> Result<Json<SessionEventPage>, StatusCode> {
    Ok(Json(
        read_session_event_page(state.db.as_ref(), id, query).await?,
    ))
}

/// GET /v1/sessions/:id/events/stream - replaying live session events.
pub async fn stream_session_events(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<Uuid>,
    axum::extract::Query(query): axum::extract::Query<SessionEventStreamQuery>,
) -> Result<axum::response::Response, StatusCode> {
    session_event_stream_response(state.db, headers, id, query).await
}

/// GET /v1/sessions/:id/timeline - lightweight task/request projection.
pub async fn get_session_timeline(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<engine_core::SessionTimeline>, StatusCode> {
    let events = state
        .db
        .read_timeline_events(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(engine_core::session_timeline(&events)))
}

/// GET /v1/sessions/:id/events/page - bounded raw event range.
pub async fn get_session_event_page(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    axum::extract::Query(query): axum::extract::Query<SessionEventPageQuery>,
) -> Result<Json<SessionEventPage>, StatusCode> {
    Ok(Json(
        read_session_event_page(state.db.as_ref(), id, query).await?,
    ))
}

/// GET /v1/sessions/:id/event/:seq - one complete audit event for the inspector.
pub async fn get_session_event(
    State(state): State<AppState>,
    Path((id, seq)): Path<(Uuid, i64)>,
) -> Result<Json<engine_core::SessionEvent>, StatusCode> {
    let event = state
        .db
        .read_events(id, Some(seq), Some(1))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .into_iter()
        .find(|event| event.seq == seq)
        .ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(event))
}

/// GET /v1/sessions/:id/tasks - durable task groups projected from LLM events.
pub async fn get_session_tasks(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<Vec<engine_core::SessionTask>>, StatusCode> {
    state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;
    let events = state
        .db
        .read_timeline_events(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(engine_core::session_tasks(&events)))
}

pub async fn complete_session(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<CompleteSessionRequest>,
) -> Result<Json<Session>, StatusCode> {
    let session =
        complete_session_in_db(state.db.as_ref(), id, req.reason, req.outcome_report).await?;
    state.sessions.lock().await.insert(id, session.clone());
    Ok(Json(session))
}

#[derive(Serialize)]
pub struct PauseSessionResponse {
    session_id: String,
    checkpoint_token: String,
    expires_at: String,
}

pub async fn pause_session(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<PauseSessionResponse>, StatusCode> {
    let mut session = state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let latest_seq = state
        .db
        .latest_seq(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .unwrap_or(0);

    let token = state
        .checkpoints
        .create(&mut session, latest_seq, "user-pause")
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    state
        .db
        .insert_session_async(session.clone())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    state
        .db
        .insert_checkpoint_async(token.clone())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    state.sessions.lock().await.insert(id, session);

    Ok(Json(PauseSessionResponse {
        session_id: id.to_string(),
        checkpoint_token: token.token,
        expires_at: token.expires_at.to_rfc3339(),
    }))
}

#[derive(Serialize)]
pub struct ResumeSessionResponse {
    pub(super) session_id: String,
    pub(super) status: String,
    pub(super) resumed_at_seq: i64,
}

pub async fn resume_session(
    State(state): State<AppState>,
    Path(token_str): Path<String>,
) -> Result<Json<ResumeSessionResponse>, StatusCode> {
    let token = state
        .checkpoints
        .validate(&token_str)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let mut session = state
        .db
        .get_session_async(token.session_id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    state
        .checkpoints
        .resume(&token, &mut session, state.db.as_ref())
        .await
        .map_err(|error| checkpoint_resume_error_to_status(&error))?;

    state
        .db
        .insert_session_async(session.clone())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    state
        .sessions
        .lock()
        .await
        .insert(token.session_id, session.clone());

    Ok(Json(ResumeSessionResponse {
        session_id: token.session_id.to_string(),
        status: SessionStatus::Active.as_str().to_string(),
        resumed_at_seq: token.last_event_seq,
    }))
}

/// Call-level cost calibration for local sessions, using the same core formulas as cloud.
pub async fn cost_diagnostics(
    State(state): State<AppState>,
) -> Result<Json<engine_core::CostAccuracyBreakdown>, StatusCode> {
    local_cost_diagnostics(state.db.as_ref()).await.map(Json)
}

pub async fn cost_diagnostics_single_provider(
    State(state): State<SingleProviderAppState>,
) -> Result<Json<engine_core::CostAccuracyBreakdown>, StatusCode> {
    local_cost_diagnostics(state.db.as_ref()).await.map(Json)
}

async fn local_cost_diagnostics(
    db: &crate::db::LocalDatabase,
) -> Result<engine_core::CostAccuracyBreakdown, StatusCode> {
    let sessions = db
        .list_sessions_async(crate::db::SessionFilter::default())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let mut accumulator = engine_core::diagnostics::DiagnosticAccumulator::default();
    for session in sessions {
        accumulator
            .append_session(db, session.id)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }
    let mut prediction = accumulator.finish().prediction;
    prediction.scope = "all".into();
    Ok(prediction)
}

/// Read decisions only after resolving the owning session.
pub async fn get_task_decisions(
    State(state): State<AppState>,
    Path((id, task_id)): Path<(Uuid, String)>,
    axum::extract::Query(query): axum::extract::Query<engine_core::task_decisions::DecisionQuery>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let session = state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;
    engine_core::task_decisions::view(state.db.as_ref(), &session, &task_id, &query)
        .await
        .map(Json)
        .map_err(|error| match error {
            engine_core::EngineError::InvalidInput(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        })
}

/// Read decisions only after resolving the owning session.
pub async fn get_task_decisions_single_provider(
    State(state): State<SingleProviderAppState>,
    Path((id, task_id)): Path<(Uuid, String)>,
    axum::extract::Query(query): axum::extract::Query<engine_core::task_decisions::DecisionQuery>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let session = state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;
    engine_core::task_decisions::view(state.db.as_ref(), &session, &task_id, &query)
        .await
        .map(Json)
        .map_err(|error| match error {
            engine_core::EngineError::InvalidInput(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        })
}
