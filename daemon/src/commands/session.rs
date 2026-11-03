use anyhow::Result;
use engine_core::{CompressionMode, ContextCompressor};
use serde_json::json;
use tracing::warn;
use uuid::Uuid;

use crate::db::LocalDatabase;
pub async fn handle_compress(
    session: String,
    mode: Option<String>,
    json: bool,
    db_path: String,
) -> Result<()> {
    let db_path = shellexpand::tilde(&db_path).to_string();
    let db = LocalDatabase::open(&db_path).unwrap_or_else(|e| {
        warn!(
            "Failed to open SQLite DB at {}: {}. Falling back to in-memory.",
            db_path, e
        );
        LocalDatabase::open_in_memory().expect("in-memory DB must open")
    });

    let session_id =
        Uuid::parse_str(&session).map_err(|_| anyhow::anyhow!("--session must be a valid UUID"))?;

    // Fall back to the user's configured compaction mode (stored in the DB
    // via `reincore config --compaction-mode`) when `--mode` is omitted, so
    // the manual command stays consistent with the in-request workflow. If
    // compaction is disabled (mode = None) and no `--mode` was given, use
    // `mid` as the explicit-offline-command default.
    let mode = match mode {
        Some(mode_str) => CompressionMode::try_from(mode_str.as_str())
            .map_err(|e| anyhow::anyhow!("invalid compaction mode: {}", e))?,
        None => {
            use engine_core::SettingsStore;
            db.get_compression_settings()
                .await?
                .mode
                .unwrap_or(CompressionMode::Mid)
        }
    };

    let compressor = ContextCompressor::new();
    let result = compressor.compress_session(&db, session_id, mode).await?;

    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!(
            "Compacted session {} with mode '{}'",
            session_id,
            mode.as_str()
        );
        println!(
            "  Original messages: {} -> retained: {}",
            result.original_message_count, result.compressed_message_count
        );
        if !result.summary.is_empty() {
            println!("  Summary: {}", result.summary);
        }
    }

    Ok(())
}

pub async fn handle_config(
    compaction: Option<String>,
    compaction_mode: Option<String>,
    db_path: String,
) -> Result<()> {
    let db_path = shellexpand::tilde(&db_path).to_string();
    let db = LocalDatabase::open(&db_path).unwrap_or_else(|e| {
        warn!(
            "Failed to open SQLite DB at {}: {}. Falling back to in-memory.",
            db_path, e
        );
        LocalDatabase::open_in_memory().expect("in-memory DB must open")
    });

    use engine_core::SettingsStore;
    let mut settings = db.get_compression_settings().await?;

    // `--compaction false` turns compaction off (mode = None).
    // `--compaction true` turns it on at the currently configured mode, or
    // `low` if no mode was set yet.
    if let Some(value) = compaction {
        let enabled = value.to_lowercase() == "true" || value == "1";
        settings.mode = if enabled {
            Some(settings.mode.unwrap_or(engine_core::CompressionMode::Low))
        } else {
            None
        };
    }

    // Setting a compaction mode also enables compaction.
    if let Some(mode_str) = compaction_mode {
        settings.mode = Some(
            CompressionMode::try_from(mode_str.as_str()).map_err(|e| anyhow::anyhow!("{}", e))?,
        );
    }

    db.set_compression_settings(settings).await?;

    println!("Compaction settings updated:");
    match settings.mode {
        Some(mode) => {
            println!("  Enabled: true");
            println!("  Mode: {}", mode.as_str());
        }
        None => {
            println!("  Enabled: false");
            println!("  Mode: (none)");
        }
    }

    Ok(())
}

pub async fn handle_pause(session: String, _reason: String, host: String, port: u16) -> Result<()> {
    let session_id =
        Uuid::parse_str(&session).map_err(|_| anyhow::anyhow!("--session must be a valid UUID"))?;

    let url = format!("http://{}:{}/v1/sessions/{}/pause", host, port, session_id);
    let client = reqwest::Client::new();
    match client.post(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            let body = resp
                .json::<serde_json::Value>()
                .await
                .unwrap_or_else(|_| json!({}));
            println!("Session paused successfully:");
            if let Some(token) = body.get("checkpoint_token").and_then(|v| v.as_str()) {
                println!("  Checkpoint token: {}", token);
            }
            if let Some(expires) = body.get("expires_at").and_then(|v| v.as_str()) {
                println!("  Expires at: {}", expires);
            }
        }
        Ok(resp) => {
            println!("Failed to pause session: {}", resp.status());
        }
        Err(e) => {
            println!("Could not reach ReinCore daemon: {}", e);
        }
    }
    Ok(())
}

pub async fn handle_resume(token: String, host: String, port: u16) -> Result<()> {
    let url = format!(
        "http://{}:{}/v1/checkpoints/{}/resume",
        host,
        port,
        urlencoding::encode(&token)
    );
    let client = reqwest::Client::new();
    match client.post(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            let body = resp
                .json::<serde_json::Value>()
                .await
                .unwrap_or_else(|_| json!({}));
            println!("Session resumed successfully:");
            if let Some(id) = body.get("session_id").and_then(|v| v.as_str()) {
                println!("  Session ID: {}", id);
            }
            if let Some(seq) = body.get("resumed_at_seq").and_then(|v| v.as_i64()) {
                println!("  Resumed at seq: {}", seq);
            }
        }
        Ok(resp) => {
            println!("Failed to resume session: {}", resp.status());
        }
        Err(e) => {
            println!("Could not reach ReinCore daemon: {}", e);
        }
    }
    Ok(())
}

pub async fn handle_sessions(
    path: Option<String>,
    checkpoints: bool,
    host: String,
    port: u16,
) -> Result<()> {
    let resolved_path = match path.as_deref() {
        Some(".") => std::env::current_dir()
            .ok()
            .and_then(|p| p.to_str().map(|s| s.to_string())),
        Some(p) => Some(p.to_string()),
        None => None,
    };

    let mut url = format!("http://{}:{}/v1/sessions", host, port);
    if let Some(p) = &resolved_path {
        url.push_str("?path=");
        url.push_str(&urlencoding::encode(p));
    }
    if checkpoints {
        url.push_str(if resolved_path.is_some() { "&" } else { "?" });
        url.push_str("checkpoints=true");
    }

    let client = reqwest::Client::new();
    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            let body = resp
                .json::<serde_json::Value>()
                .await
                .unwrap_or_else(|_| json!({}));
            let sessions = body
                .get("sessions")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            if sessions.is_empty() {
                println!("No sessions found.");
            } else {
                println!("Sessions:");
                for s in sessions {
                    let id = s
                        .get("session_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let summary = s.get("task_summary").and_then(|v| v.as_str()).unwrap_or("");
                    let status = s
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let path = s
                        .get("project_path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("(none)");
                    let project = s
                        .get("project_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("(unassigned)");
                    let spend = s.get("spend_cents").and_then(|v| v.as_i64()).unwrap_or(0);
                    let context = s
                        .get("context_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    println!(
                        "  {} - {} [{}] project={} path={} spend={}c context={} tokens",
                        id, summary, status, project, path, spend, context
                    );
                    if let Some(token) = s.get("checkpoint_token").and_then(|v| v.as_str()) {
                        println!("    checkpoint: {}", token);
                    }
                }
            }
        }
        Ok(resp) => {
            println!("Failed to list sessions: {}", resp.status());
        }
        Err(e) => {
            println!("Could not reach ReinCore daemon: {}", e);
        }
    }
    Ok(())
}
