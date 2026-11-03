use super::lifecycle::*;
use super::sessions::ResumeSessionResponse;
use super::*;

#[derive(Deserialize)]
pub struct FrameworkCheckpointRequest {
    pub reason: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct FrameworkCheckpointResponse {
    pub checkpoint: FrameworkCheckpoint,
}

pub async fn framework_checkpoint_session(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<FrameworkCheckpointRequest>,
) -> Result<Json<FrameworkCheckpointResponse>, StatusCode> {
    let mut session = state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let checkpoint = state
        .framework
        .checkpoint(
            &mut session,
            req.reason
                .unwrap_or_else(|| "framework-checkpoint".to_string()),
            state.db.as_ref(),
            req.metadata,
        )
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    state
        .db
        .insert_session_async(session.clone())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    state
        .db
        .insert_checkpoint_async(
            state
                .framework
                .validate_token(&checkpoint.checkpoint_token)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
        )
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    state.sessions.lock().await.insert(id, session);

    Ok(Json(FrameworkCheckpointResponse { checkpoint }))
}

pub async fn framework_checkpoint_session_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<FrameworkCheckpointRequest>,
) -> Result<Json<FrameworkCheckpointResponse>, StatusCode> {
    let mut session = state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let checkpoint = state
        .framework
        .checkpoint(
            &mut session,
            req.reason
                .unwrap_or_else(|| "framework-checkpoint".to_string()),
            state.db.as_ref(),
            req.metadata,
        )
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    state
        .db
        .insert_session_async(session.clone())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    state
        .db
        .insert_checkpoint_async(
            state
                .framework
                .validate_token(&checkpoint.checkpoint_token)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
        )
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    state.sessions.lock().await.insert(id, session);

    Ok(Json(FrameworkCheckpointResponse { checkpoint }))
}

pub async fn framework_resume_session(
    State(state): State<AppState>,
    Path(token): Path<String>,
) -> Result<Json<ResumeSessionResponse>, StatusCode> {
    let token = state
        .framework
        .validate_token(&token)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let mut session = state
        .db
        .get_session_async(token.session_id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    state
        .framework
        .resume_from_token(&token.token, &mut session, state.db.as_ref())
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

pub async fn framework_resume_session_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(token): Path<String>,
) -> Result<Json<ResumeSessionResponse>, StatusCode> {
    let token = state
        .framework
        .validate_token(&token)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let mut session = state
        .db
        .get_session_async(token.session_id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    state
        .framework
        .resume_from_token(&token.token, &mut session, state.db.as_ref())
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

#[derive(Deserialize)]
pub struct FrameworkMemoryRequest {
    pub system_prompt: Option<String>,
}

#[derive(Serialize)]
pub struct FrameworkMemoryResponse {
    pub snapshot: FrameworkMemorySnapshot,
}

pub async fn framework_memory_session(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<FrameworkMemoryRequest>,
) -> Result<Json<FrameworkMemoryResponse>, StatusCode> {
    let _session = state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let snapshot = state
        .framework
        .memory_snapshot(id, req.system_prompt.as_deref(), state.db.as_ref())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(FrameworkMemoryResponse { snapshot }))
}

pub async fn framework_memory_session_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<FrameworkMemoryRequest>,
) -> Result<Json<FrameworkMemoryResponse>, StatusCode> {
    let _session = state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let snapshot = state
        .framework
        .memory_snapshot(id, req.system_prompt.as_deref(), state.db.as_ref())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(FrameworkMemoryResponse { snapshot }))
}

#[derive(Deserialize)]
pub struct FrameworkTurnRequest {
    pub role: String,
    pub content: String,
}

pub async fn framework_record_turn(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<FrameworkTurnRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let _session = state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let event = state
        .framework
        .record_framework_turn(id, &req.role, &req.content, state.db.as_ref())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(json!({
        "seq": event.seq,
        "event_type": "llm_call",
        "session_id": id.to_string(),
    })))
}

pub async fn framework_record_turn_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<FrameworkTurnRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let _session = state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let event = state
        .framework
        .record_framework_turn(id, &req.role, &req.content, state.db.as_ref())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(json!({
        "seq": event.seq,
        "event_type": "llm_call",
        "session_id": id.to_string(),
    })))
}
