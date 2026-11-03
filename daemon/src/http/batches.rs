//! Durable session-bound batch endpoints. Provider data is accessed only after
//! engine session authorization and credential/endpoint binding checks.
use super::*;
use engine_core::{
    batch_jobs as jobs,
    provider::batch::{self, Batch, BatchError, CreateBatchRequest},
};

type BatchHttpError = (StatusCode, Json<serde_json::Value>);

fn batch_error(error: BatchError) -> BatchHttpError {
    (
        StatusCode::from_u16(error.status).unwrap_or(StatusCode::BAD_GATEWAY),
        Json(json!({"error":error.message})),
    )
}

fn failure(status: u16, message: &str) -> BatchHttpError {
    batch_error(BatchError::new(status, message))
}

pub async fn create_batch(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<CreateBatchRequest>,
) -> Result<Json<Batch>, BatchHttpError> {
    let provider = batch::select_provider(
        &state.providers,
        headers
            .get("x-reincore-provider")
            .and_then(|v| v.to_str().ok()),
    )
    .map_err(batch_error)?;

    provider
        .batch_client()
        .map_err(|_| failure(501, "provider does not support batch jobs"))?;

    let session_id = request
        .session_id
        .ok_or_else(|| failure(400, "session_id is required for durable batch accounting"))?;

    let session = state
        .db
        .get_session_async(session_id)
        .await
        .map_err(|_| failure(500, "session lookup failed"))?
        .ok_or_else(|| failure(404, "session not found"))?;

    let _ = session;
    let id = request.idempotency_key.unwrap_or_else(Uuid::new_v4);
    let job = jobs::create(
        state.db.as_ref(),
        provider.as_ref(),
        session_id,
        id,
        &request.requests,
    )
    .await
    .map_err(batch_error)?;

    jobs::spawn_polling(state.db.clone(), provider, id);
    Ok(Json(job.public()))
}

pub async fn get_batch(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<Batch>, BatchHttpError> {
    let provider = batch::select_provider(
        &state.providers,
        headers
            .get("x-reincore-provider")
            .and_then(|v| v.to_str().ok()),
    )
    .map_err(batch_error)?;

    provider
        .batch_client()
        .map_err(|_| failure(501, "provider does not support batch jobs"))?;

    let id = Uuid::parse_str(&id).map_err(|_| failure(400, "expected an engine batch UUID"))?;
    let job = jobs::get(state.db.as_ref(), id)
        .await
        .map_err(batch_error)?;

    let session = state
        .db
        .get_session_async(job.session_id)
        .await
        .map_err(|_| failure(500, "session lookup failed"))?
        .ok_or_else(|| failure(404, "session not found"))?;

    let _ = session;
    let job = jobs::refresh(state.db.as_ref(), provider.as_ref(), job)
        .await
        .map_err(batch_error)?;

    jobs::spawn_polling(state.db.clone(), provider, id);
    Ok(Json(job.public()))
}

pub async fn cancel_batch(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<Batch>, BatchHttpError> {
    let provider = batch::select_provider(
        &state.providers,
        headers
            .get("x-reincore-provider")
            .and_then(|v| v.to_str().ok()),
    )
    .map_err(batch_error)?;

    provider
        .batch_client()
        .map_err(|_| failure(501, "provider does not support batch jobs"))?;

    let id = Uuid::parse_str(&id).map_err(|_| failure(400, "expected an engine batch UUID"))?;
    let job = jobs::get(state.db.as_ref(), id)
        .await
        .map_err(batch_error)?;

    let session = state
        .db
        .get_session_async(job.session_id)
        .await
        .map_err(|_| failure(500, "session lookup failed"))?
        .ok_or_else(|| failure(404, "session not found"))?;

    let _ = session;
    let job = jobs::cancel(state.db.as_ref(), provider.as_ref(), job)
        .await
        .map_err(batch_error)?;

    jobs::spawn_polling(state.db.clone(), provider, id);
    Ok(Json(job.public()))
}

pub async fn results_batch(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, BatchHttpError> {
    let provider = batch::select_provider(
        &state.providers,
        headers
            .get("x-reincore-provider")
            .and_then(|v| v.to_str().ok()),
    )
    .map_err(batch_error)?;

    provider
        .batch_client()
        .map_err(|_| failure(501, "provider does not support batch jobs"))?;

    let id = Uuid::parse_str(&id).map_err(|_| failure(400, "expected an engine batch UUID"))?;
    let job = jobs::get(state.db.as_ref(), id)
        .await
        .map_err(batch_error)?;

    let session = state
        .db
        .get_session_async(job.session_id)
        .await
        .map_err(|_| failure(500, "session lookup failed"))?
        .ok_or_else(|| failure(404, "session not found"))?;

    let _ = session;
    let job = jobs::refresh(state.db.as_ref(), provider.as_ref(), job)
        .await
        .map_err(batch_error)?;
    jobs::spawn_polling(state.db.clone(), provider, id);

    if !job.finished() {
        return Err(failure(409, "batch results/accounting are not ready"));
    }
    let mut results = engine_core::action_gate::gate_batch_results(&job, &state.action_policy);
    engine_core::claim_grounding::annotate_batch_results(
        &mut results,
        state.claim_grounding.enabled,
    );
    Ok(Json(json!({"data":results, "batch":job.public()})))
}

pub async fn create_batch_single_provider(
    State(state): State<SingleProviderAppState>,
    Json(request): Json<CreateBatchRequest>,
) -> Result<Json<Batch>, BatchHttpError> {
    let provider = state.provider.clone();
    provider
        .batch_client()
        .map_err(|_| failure(501, "provider does not support batch jobs"))?;

    let session_id = request
        .session_id
        .ok_or_else(|| failure(400, "session_id is required for durable batch accounting"))?;

    let session = state
        .db
        .get_session_async(session_id)
        .await
        .map_err(|_| failure(500, "session lookup failed"))?
        .ok_or_else(|| failure(404, "session not found"))?;

    let _ = session;
    let id = request.idempotency_key.unwrap_or_else(Uuid::new_v4);
    let job = jobs::create(
        state.db.as_ref(),
        provider.as_ref(),
        session_id,
        id,
        &request.requests,
    )
    .await
    .map_err(batch_error)?;

    jobs::spawn_polling(state.db.clone(), provider, id);
    Ok(Json(job.public()))
}

pub async fn get_batch_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<String>,
) -> Result<Json<Batch>, BatchHttpError> {
    let provider = state.provider.clone();
    provider
        .batch_client()
        .map_err(|_| failure(501, "provider does not support batch jobs"))?;

    let id = Uuid::parse_str(&id).map_err(|_| failure(400, "expected an engine batch UUID"))?;
    let job = jobs::get(state.db.as_ref(), id)
        .await
        .map_err(batch_error)?;

    let session = state
        .db
        .get_session_async(job.session_id)
        .await
        .map_err(|_| failure(500, "session lookup failed"))?
        .ok_or_else(|| failure(404, "session not found"))?;

    let _ = session;
    let job = jobs::refresh(state.db.as_ref(), provider.as_ref(), job)
        .await
        .map_err(batch_error)?;

    jobs::spawn_polling(state.db.clone(), provider, id);
    Ok(Json(job.public()))
}

pub async fn cancel_batch_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<String>,
) -> Result<Json<Batch>, BatchHttpError> {
    let provider = state.provider.clone();
    provider
        .batch_client()
        .map_err(|_| failure(501, "provider does not support batch jobs"))?;

    let id = Uuid::parse_str(&id).map_err(|_| failure(400, "expected an engine batch UUID"))?;
    let job = jobs::get(state.db.as_ref(), id)
        .await
        .map_err(batch_error)?;

    let session = state
        .db
        .get_session_async(job.session_id)
        .await
        .map_err(|_| failure(500, "session lookup failed"))?
        .ok_or_else(|| failure(404, "session not found"))?;

    let _ = session;
    let job = jobs::cancel(state.db.as_ref(), provider.as_ref(), job)
        .await
        .map_err(batch_error)?;

    jobs::spawn_polling(state.db.clone(), provider, id);
    Ok(Json(job.public()))
}

pub async fn results_batch_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, BatchHttpError> {
    let provider = state.provider.clone();
    provider
        .batch_client()
        .map_err(|_| failure(501, "provider does not support batch jobs"))?;

    let id = Uuid::parse_str(&id).map_err(|_| failure(400, "expected an engine batch UUID"))?;
    let job = jobs::get(state.db.as_ref(), id)
        .await
        .map_err(batch_error)?;

    let session = state
        .db
        .get_session_async(job.session_id)
        .await
        .map_err(|_| failure(500, "session lookup failed"))?
        .ok_or_else(|| failure(404, "session not found"))?;

    let _ = session;
    let job = jobs::refresh(state.db.as_ref(), provider.as_ref(), job)
        .await
        .map_err(batch_error)?;

    jobs::spawn_polling(state.db.clone(), provider, id);
    if !job.finished() {
        return Err(failure(409, "batch results/accounting are not ready"));
    }

    let mut results = engine_core::action_gate::gate_batch_results(&job, &state.action_policy);
    engine_core::claim_grounding::annotate_batch_results(
        &mut results,
        state.claim_grounding.enabled,
    );
    Ok(Json(json!({"data":results, "batch":job.public()})))
}

/// Persist SDK batch activity after verifying access to the target session.
pub async fn record_batch_event(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(request): Json<batch::BatchEventRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let _session = state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let event = batch::record_event(state.db.as_ref(), id, request)
        .await
        .map_err(|error| match error {
            engine_core::EngineError::InvalidInput(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        })?;

    Ok(Json(
        json!({"seq":event.seq, "event_type":"batch", "session_id":id}),
    ))
}

/// Persist SDK batch activity after verifying access to the target session.
pub async fn record_batch_event_single_provider(
    State(state): State<SingleProviderAppState>,
    Path(id): Path<Uuid>,
    Json(request): Json<batch::BatchEventRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let _session = state
        .db
        .get_session_async(id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let event = batch::record_event(state.db.as_ref(), id, request)
        .await
        .map_err(|error| match error {
            engine_core::EngineError::InvalidInput(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        })?;

    Ok(Json(
        json!({"seq":event.seq, "event_type":"batch", "session_id":id}),
    ))
}
