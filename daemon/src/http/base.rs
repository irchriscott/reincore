use super::*;

pub async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let providers: Vec<&str> = state
        .providers
        .iter()
        .map(|provider| provider.label())
        .collect();
    Json(json!({
        "status": "ok",
        "edition": "engine",
        "mode": "multi_provider",
        "providers": providers,
        "default_project_path": state.default_project_path.as_str(),
    }))
}

pub async fn health_single_provider(
    State(state): State<SingleProviderAppState>,
) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "edition": "engine",
        "mode": "single_provider",
        "api_mode": state.api_mode.as_str(),
        "providers": [state.provider.label()],
        "default_project_path": state.default_project_path.as_str(),
    }))
}

pub async fn list_models(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let models = state
        .providers
        .list_all_models()
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?;
    let mut data: Vec<serde_json::Value> = models
        .into_iter()
        .map(|m| {
            // Look up context window from the curated catalog so CLI tools
            // (Codex, Copilot CLI, Aider, etc.) that read /v1/models can
            // discover ReinCore's real context limits without relying on
            // their own hardcoded defaults.
            let catalog_entry = find_model(&m.provider, &m.id);
            let context_window = catalog_entry
                .as_ref()
                .map(|c| c.max_input_tokens as u64)
                .unwrap_or_else(fallback_context_limit);
            let max_output_tokens = catalog_entry
                .as_ref()
                .map(|c| c.max_output_tokens as u64)
                .unwrap_or(65_536);
            json!({
                "id": m.id,
                "object": "model",
                "created": m.created_at.map(|dt| dt.timestamp()).unwrap_or(0),
                "owned_by": m.provider,
                "display_name": m.display_name.unwrap_or_else(|| m.id.clone()),
                "context_window": context_window,
                "max_input_tokens": context_window,
                "max_output_tokens": max_output_tokens,
            })
        })
        .collect();
    // Prepend the ReinCore Auto entry - ReinCore handles model selection and
    // tier degradation automatically, so this is the recommended default.
    data.insert(
        0,
        json!({
            "id": "reincore-auto",
            "object": "model",
            "created": 0,
            "owned_by": "reincore",
            "display_name": "ReinCore Auto",
            "context_window": 1_000_000,
            "max_input_tokens": 1_000_000,
            "max_output_tokens": 65_536,
        }),
    );
    Ok(Json(json!({
        "object": "list",
        "data": data,
    })))
}

/// `GET /muse-code/models` - model catalog for Muse Code.
///
/// Muse Code's Meta provider checks this endpoint before issuing
/// `POST /responses`. Same payload as `/v1/models`; ReinCore routes the
/// client-supplied model itself, so the catalog is informational.
pub async fn muse_code_models(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    list_models(State(state)).await
}

/// Fallback context limit used when a model is not in the curated catalog.
/// Reads `REINCORE_CONTEXT_LIMIT` env var (tokens), defaulting to 1_000_000.
/// This ensures the engine never assumes a small context window (which would
/// cause premature session rollovers) when the real limit is unknown.
pub fn fallback_context_limit() -> u64 {
    std::env::var("REINCORE_CONTEXT_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v: &u64| *v > 0)
        .unwrap_or(1_000_000)
}

pub async fn ready(State(state): State<AppState>) -> axum::response::Response {
    engine_core::resilience::readiness(state.db.ready().await, None)
}
pub async fn ready_single_provider(
    State(state): State<SingleProviderAppState>,
) -> axum::response::Response {
    engine_core::resilience::readiness(state.db.ready().await, None)
}
