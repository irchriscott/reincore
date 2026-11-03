use super::*;

// Engine configuration endpoints.
//
// This module exposes engine-wide settings (context compaction and future config
// options) over HTTP so the desktop frontend and CLI can read/write them
// through a uniform `/v1/config/*` API surface.

use engine_core::{CompressionMode, CompressionSettings, SettingsStore};

/// Response for `GET/PUT /v1/config/compaction`.
#[derive(Serialize)]
pub struct CompressionSettingsResponse {
    pub enabled: bool,
    pub mode: String,
}

/// Request body for `PUT /v1/config/compaction`.
#[derive(Deserialize)]
pub struct UpdateCompressionRequest {
    pub enabled: bool,
    pub mode: String,
}

/// `GET /v1/config/compaction` - read the current compaction settings.
pub async fn get_compression_config(
    State(state): State<AppState>,
) -> Result<Json<CompressionSettingsResponse>, (StatusCode, String)> {
    get_compression_config_inner(&state.db).await
}

/// Single-provider variant of [`get_compression_config`].
pub async fn get_compression_config_single_provider(
    State(state): State<SingleProviderAppState>,
) -> Result<Json<CompressionSettingsResponse>, (StatusCode, String)> {
    get_compression_config_inner(&state.db).await
}

async fn get_compression_config_inner(
    db: &crate::db::LocalDatabase,
) -> Result<Json<CompressionSettingsResponse>, (StatusCode, String)> {
    let settings = db
        .get_compression_settings()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(CompressionSettingsResponse {
        enabled: settings.is_enabled(),
        mode: settings
            .mode
            .unwrap_or(CompressionMode::Low)
            .as_str()
            .to_string(),
    }))
}

/// `PUT /v1/config/compaction` - update compaction settings.
pub async fn update_compression_config(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<UpdateCompressionRequest>,
) -> Result<Json<CompressionSettingsResponse>, (StatusCode, String)> {
    update_compression_config_inner(&state.db, req).await
}

/// Single-provider variant of [`update_compression_config`].
pub async fn update_compression_config_single_provider(
    State(state): State<SingleProviderAppState>,
    axum::Json(req): axum::Json<UpdateCompressionRequest>,
) -> Result<Json<CompressionSettingsResponse>, (StatusCode, String)> {
    update_compression_config_inner(&state.db, req).await
}

async fn update_compression_config_inner(
    db: &crate::db::LocalDatabase,
    req: UpdateCompressionRequest,
) -> Result<Json<CompressionSettingsResponse>, (StatusCode, String)> {
    let mode = CompressionMode::try_from(req.mode.as_str())
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let settings = if req.enabled {
        CompressionSettings::new(mode)
    } else {
        CompressionSettings::disabled()
    };

    db.set_compression_settings(settings)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(CompressionSettingsResponse {
        enabled: req.enabled,
        mode: mode.as_str().to_string(),
    }))
}
