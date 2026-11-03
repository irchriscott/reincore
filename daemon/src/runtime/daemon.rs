use anyhow::Result;
use axum::{
    body::to_bytes,
    extract::{DefaultBodyLimit, Request},
    http::HeaderName,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, patch, post},
    Router,
};
use colored::Colorize;
use engine_core::{otel_health, telemetry};
use serde_json::json;
use std::net::SocketAddr;

use crate::app::{AppState, SingleProviderAppState};
use crate::cli::DaemonConfig;
use crate::db::LocalDatabase;
use crate::http::{
    add_provider_credentials, assign_local_project_directory,
    assign_local_project_directory_single_provider, cache_hit_rate, cache_hit_rate_single_provider,
    chat_completions, chat_completions_single_provider, checkpoint_redemptions,
    checkpoint_redemptions_single_provider, complete_session, complete_session_single_provider,
    create_local_project, create_local_project_single_provider, create_session,
    create_session_single_provider, framework_checkpoint_session,
    framework_checkpoint_session_single_provider, framework_memory_session,
    framework_memory_session_single_provider, framework_record_turn,
    framework_record_turn_single_provider, framework_resume_session,
    framework_resume_session_single_provider, gemini_list_models,
    gemini_list_models_single_provider, get_budget, get_budget_history,
    get_budget_history_single_provider, get_budget_single_provider, get_compression_config,
    get_compression_config_single_provider, get_provider_config, get_routing_config, get_session,
    get_session_event, get_session_event_page, get_session_event_page_single_provider,
    get_session_event_single_provider, get_session_events, get_session_events_single_provider,
    get_session_single_provider, get_session_tasks, get_session_tasks_single_provider,
    get_session_timeline, get_session_timeline_single_provider, get_task_decisions,
    get_task_decisions_single_provider, health, health_single_provider, list_custom_models,
    list_local_projects, list_local_projects_single_provider, list_models,
    list_models_single_provider, list_ollama_models, list_projects, list_projects_single_provider,
    list_provider_catalog, list_providers, list_sessions, list_sessions_single_provider,
    list_subagent_sessions, list_subagent_sessions_single_provider, messages_api,
    messages_api_single_provider, messages_count_tokens, messages_count_tokens_single_provider,
    muse_code_models, muse_code_models_single_provider, pause_session,
    pause_session_single_provider, print_cumulative_cost, remove_provider_credentials,
    rename_local_project, rename_local_project_single_provider, rename_session,
    rename_session_single_provider, resolve_local_project, resolve_local_project_single_provider,
    responses_api, responses_api_single_provider, resume_session, resume_session_single_provider,
    set_budget, set_budget_single_provider, stream_session_events,
    stream_session_events_single_provider, tool_calls, tool_calls_single_provider,
    update_compression_config, update_compression_config_single_provider, update_provider_config,
    update_routing_config,
};

/// Large editor contexts must reach the compaction pipeline before they can
/// be reduced. Keep this aligned with the local request body limit.
const MAX_REQUEST_BODY_BYTES: usize = 32 * 1024 * 1024;

/// Warn when the administrator action policy is default-allow with no
/// per-tool rules: every tool is permitted unless a contract check denies
/// it. The policy itself is left untouched; this is operator guidance only.
fn warn_if_permissive_action_policy(policy: &engine_core::action_gate::ActionPolicy) {
    if policy.default_allow && policy.tools.is_empty() {
        eprintln!(
            "{}  {}",
            timestamp().dimmed(),
            "Administrator action policy is default-allow with no per-tool rules; \
             all tools are permitted unless a contract check denies them. \
             Add [action_policy] rules to config.toml to restrict tools."
                .yellow()
        );
    }
}

/// Format a timestamp as `HH:MM:SS` for log line prefixes.
fn timestamp() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}

async fn verified_otel_endpoint(
    otel: &crate::local_config::TelemetryRuntimeConfig,
) -> Option<String> {
    if !otel.enabled {
        return None;
    }
    let endpoint = otel.endpoint.clone()?;
    let health = otel_health::check(otel_health::OpenTelemetryHealthConfig {
        endpoint: endpoint.clone(),
        is_bundled_jaeger: endpoint == "http://localhost:4317",
    })
    .await;
    if health.is_healthy() {
        Some(endpoint)
    } else {
        eprintln!(
            "{}  {}",
            timestamp().dimmed(),
            format!("OpenTelemetry disabled: {}", health.message()).yellow()
        );
        None
    }
}

/// Middleware that logs each incoming request and its response.
///
/// **Always shown** (even without `--verbose`):
/// - A one-line summary: `[time] METHOD path HTTP/version → status` with the
///   status colored green (2xx/3xx) or red (4xx/5xx).
///
/// **Verbose only** (`--verbose` / debug level):
/// - Full request headers (with sensitive headers redacted) and body.
/// - Full response headers and body.
async fn log_request(req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let logged_uri = engine_core::privacy::current_policy().sanitize_for_logging(&uri.to_string());
    let http_version = req.version();
    let verbose = req
        .extensions()
        .get::<std::sync::Arc<bool>>()
        .map(|v| **v)
        .unwrap_or(false);

    // Capture the request body for verbose logging.
    let (parts, body) = req.into_parts();
    let body_bytes = match to_bytes(body, MAX_REQUEST_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!(%error, "request body exceeded the ReinCore limit");
            return (
                axum::http::StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "request body exceeds the {} MiB limit",
                    MAX_REQUEST_BODY_BYTES / (1024 * 1024)
                ),
            )
                .into_response();
        }
    };

    if verbose {
        let headers = &parts.headers;
        let body_str = String::from_utf8_lossy(&body_bytes);

        let header_str: String = headers
            .iter()
            .map(|(name, value)| {
                let val = if is_sensitive_header(name) {
                    "<redacted>".to_string()
                } else {
                    value.to_str().unwrap_or("<binary>").to_string()
                };
                format!("    {}: {}", name, val)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let body_display = if body_str.is_empty() {
            "(empty)".to_string()
        } else {
            engine_core::privacy::current_policy().sanitize_for_logging(&body_str)
        };

        println!(
            "\n{}  ┌─── Incoming Request ───────────────────────────────────\n\
             {}  │ {} {} {}\n\
             {}  │ Headers:\n\
             {}\n\
             {}  │ Body ({} bytes): {}\n\
             {}  └────────────────────────────────────────────────────────",
            timestamp().dimmed(),
            timestamp().dimmed(),
            method.to_string().cyan(),
            logged_uri.white(),
            format!("{:?}", http_version).dimmed(),
            timestamp().dimmed(),
            if header_str.is_empty() {
                "    (none)".to_string()
            } else {
                header_str
            },
            timestamp().dimmed(),
            body_bytes.len(),
            body_display,
            timestamp().dimmed(),
        );
    }

    // Reconstruct the request with the body so the handler can process it.
    let req = Request::from_parts(parts, axum::body::Body::from(body_bytes));

    // Run the handler and capture the response.
    let response = next.run(req).await;
    let status = response.status();

    // One-line summary, always shown.
    let status_code = status.as_u16();
    let status_colored = if status.is_success() || status.is_redirection() {
        format!(
            "{} {}",
            status_code,
            status.canonical_reason().unwrap_or("")
        )
        .green()
    } else {
        format!(
            "{} {}",
            status_code,
            status.canonical_reason().unwrap_or("")
        )
        .red()
        .bold()
    };

    let http_ver = format!("{:?}", http_version).dimmed();

    println!(
        "{}  {} {} {} → {}",
        timestamp().dimmed(),
        method.to_string().cyan(),
        logged_uri.white(),
        http_ver,
        status_colored,
    );

    // Verbose: log the response headers and body.
    if verbose && !status.is_success() {
        // For error responses, try to extract the error message from the body.
        let (parts, body) = response.into_parts();
        let body_bytes = to_bytes(body, 10 * 1024 * 1024).await.unwrap_or_default();
        let body_str = String::from_utf8_lossy(&body_bytes);

        // Try to parse the error message from JSON.
        let error_msg = if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body_str) {
            json.get("error")
                .or_else(|| json.get("message"))
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| body_str.to_string())
        } else {
            body_str.to_string()
        };
        let error_msg = engine_core::privacy::current_policy().sanitize_for_logging(&error_msg);

        eprintln!(
            "{}  {} Error response body: {}",
            timestamp().dimmed(),
            "✗".red().bold(),
            error_msg.red(),
        );

        return Response::from_parts(parts, axum::body::Body::from(body_bytes));
    }

    response
}

/// Headers whose values should not be logged in plaintext.
fn is_sensitive_header(name: &HeaderName) -> bool {
    let lower = name.as_str().to_lowercase();
    matches!(
        lower.as_str(),
        "authorization" | "x-api-key" | "anthropic-auth-token" | "cookie" | "set-cookie"
    )
}

// This is the CLI boundary, so keeping each flag visible is clearer than
// hiding them in a second configuration layer.
#[allow(clippy::too_many_arguments)]
pub async fn start_daemon(
    host: String,
    port: u16,
    db_path: String,
    verbose: bool,
    provider: Option<String>,
    mode: String,
    path: Option<String>,
    cloud: bool,
    model: Option<String>,
    shutdown_rx: Option<tokio::sync::oneshot::Receiver<()>>,
    startup_tx: Option<tokio::sync::oneshot::Sender<()>>,
) -> Result<()> {
    // Make sure ~/.reincore-io/config.toml exists on a fresh install so the
    // user has a writable routing/model configuration from the first run.
    if let Err(error) = crate::local_config::ensure_config_exists() {
        eprintln!(
            "{}  {}",
            timestamp().dimmed(),
            format!("Failed to create config.toml: {}", error).yellow()
        );
    }

    // If --provider was given, start in single-provider mode. This replaces
    // the old `provider start-proxy` subcommand and supports the same options
    // as `launch` (--mode, --cloud, --model, etc.) except for app-launching.
    if let Some(provider_label) = provider {
        let provider_label = provider_label.to_lowercase();

        // Ollama handles its own credentials/model selection (local vs. cloud),
        // so skip the generic API-key path for it.
        if provider_label != "ollama" {
            crate::launch::ensure_api_key_pub(&provider_label).await?;
        }

        // Resolve the provider. Ollama has special local/cloud setup; all
        // other providers go through the generic registry path.
        let provider_arc: std::sync::Arc<dyn engine_core::Provider> = if provider_label == "ollama"
        {
            crate::launch::setup_ollama_provider_pub(cloud, model, false).await?
        } else {
            crate::provider::resolve_provider_arc(&provider_label).await?
        };

        // Normalize mode: accept "agentic" (canonical), "agent", or
        // "responses" (legacy alias) - all map to the stored value
        // "agentic".  "chat" stays "chat".  The launch layer later
        // translates "agentic" -> wire API "responses" as needed.
        let canonical_mode = match mode.to_lowercase().as_str() {
            "chat" => "chat",
            "agentic" | "agent" | "responses" => "agentic",
            other => {
                return Err(anyhow::anyhow!(
                    "unknown mode '{}'. Use 'agentic' (default) or 'chat'.",
                    other
                ));
            }
        };

        return start_single_provider_daemon(
            host,
            port,
            db_path,
            verbose,
            provider_arc,
            canonical_mode,
            path,
            shutdown_rx,
            startup_tx,
        )
        .await;
    }

    let config = crate::local_config::load_local_config();
    warn_if_permissive_action_policy(&config.action_policy);
    let otel = config.otel();
    let endpoint = verified_otel_endpoint(&otel).await;
    let _telemetry_guard = telemetry::maybe_init_with_config_skip_health_check(
        otel.service_name.clone(),
        endpoint,
        otel.insecure,
    );
    if verbose {
        tracing::debug!("verbose logging requested via --verbose");
    }

    let startup_message = format!(
        "{} {}:{}",
        "ReinCore daemon starting on".green().bold(),
        host.white().bold(),
        port.to_string().white().bold(),
    );
    println!("{}  {}", timestamp().dimmed(), startup_message);

    let db_path = shellexpand::tilde(&db_path).to_string();
    let db = LocalDatabase::open(&db_path).unwrap_or_else(|e| {
        eprintln!(
            "{}  {}",
            timestamp().dimmed(),
            format!(
                "Failed to open SQLite DB at {}: {}. Falling back to in-memory.",
                db_path, e
            )
            .yellow()
        );
        LocalDatabase::open_in_memory().expect("in-memory DB must open")
    });

    let _metrics_server = engine_core::operational::start_from_env().await?;
    engine_core::operational::gauge(engine_core::operational::Gauge::RelayConfigured, Some(0.0));
    let monitor_db = db.clone();
    let _dependency_sampler = _metrics_server.as_ref().map(|_| {
        engine_core::operational::sampler(move || {
            let db = monitor_db.clone();
            async move {
                db.observe_dependencies().await;
            }
        })
    });

    let mut state = AppState::new_with_project_path(db, path.as_deref()).with_verbose(verbose);
    let mcp_settings = crate::mcp_config::settings()?;
    state.tool_executors = std::sync::Arc::new(crate::mcp_config::registry(&mcp_settings).await?);
    for provider in state.providers.iter() {
        engine_core::batch_jobs::resume_pending(state.db.clone(), provider.clone()).await?;
    }

    let app = Router::new()
        .route("/health", get(health))
        .route("/livez", get(engine_core::resilience::live))
        .route("/readyz", get(crate::http::ready))
        .route("/v1/models", get(list_models))
        // Alias without /v1 prefix - VS Code BYOK validation calls {base_url}/models.
        .route("/models", get(list_models))
        .route("/v1/projects", get(list_projects))
        .route(
            "/v1/local-projects",
            get(list_local_projects).post(create_local_project),
        )
        .route("/v1/local-projects/resolve", get(resolve_local_project))
        .route(
            "/v1/local-projects/:id/directories",
            post(assign_local_project_directory),
        )
        .route("/v1/local-projects/:id", patch(rename_local_project))
        .route("/v1/cache-hit-rate", get(cache_hit_rate))
        .route("/v1/cache-utilization", get(cache_hit_rate))
        .route(
            "/v1/sessions/:id/diagnostics",
            get(crate::http::get_session_diagnostics),
        )
        .route("/v1/cost-diagnostics", get(crate::http::cost_diagnostics))
        .route("/v1/checkpoint-redemptions", get(checkpoint_redemptions))
        // Provider & config management endpoints.
        .route("/v1/providers", get(list_providers))
        .route(
            "/v1/providers/:provider/credentials",
            post(add_provider_credentials).delete(remove_provider_credentials),
        )
        .route("/v1/providers/:provider/models", get(list_provider_catalog))
        .route("/v1/providers/ollama/models", post(list_ollama_models))
        .route("/v1/providers/custom/models", post(list_custom_models))
        .route(
            "/v1/config/routing",
            get(get_routing_config).put(update_routing_config),
        )
        .route(
            "/v1/config/providers/:provider",
            get(get_provider_config).put(update_provider_config),
        )
        .route("/v1/sessions", post(create_session).get(list_sessions))
        .route("/v1/sessions/:id", get(get_session).patch(rename_session))
        .route("/v1/sessions/:id/events", get(get_session_events))
        .route("/v1/sessions/:id/events/stream", get(stream_session_events))
        .route("/v1/sessions/:id/events/page", get(get_session_event_page))
        .route("/v1/sessions/:id/event/:seq", get(get_session_event))
        .route("/v1/sessions/:id/timeline", get(get_session_timeline))
        .route("/v1/sessions/:id/tasks", get(get_session_tasks))
        .route(
            "/v1/sessions/:id/tasks/:task_id/decisions",
            get(get_task_decisions),
        )
        .route("/v1/sessions/:id/subagents", get(list_subagent_sessions))
        .route("/v1/sessions/:id/pause", post(pause_session))
        .route("/v1/sessions/:id/complete", post(complete_session))
        .route(
            "/v1/sessions/:id/framework-checkpoint",
            post(framework_checkpoint_session),
        )
        .route(
            "/v1/sessions/:id/framework-memory",
            post(framework_memory_session),
        )
        .route(
            "/v1/sessions/:id/framework-turn",
            post(framework_record_turn),
        )
        .route("/v1/checkpoints/:token/resume", post(resume_session))
        .route(
            "/v1/framework-checkpoints/:token/resume",
            post(framework_resume_session),
        )
        .route(
            "/v1/sessions/:id/batch-events",
            post(crate::http::record_batch_event),
        )
        .route("/v1/batches", post(crate::http::create_batch))
        .route("/v1/batches/:id", get(crate::http::get_batch))
        .route("/v1/batches/:id/cancel", post(crate::http::cancel_batch))
        .route("/v1/batches/:id/results", get(crate::http::results_batch))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses_api))
        // Muse Code aliases: the Meta provider posts to {base_url}/responses
        // and reads its model catalog from {base_url}/muse-code/models.
        .route("/responses", post(responses_api))
        .route("/muse-code/models", get(muse_code_models))
        .route("/v1/messages", post(messages_api))
        .route("/v1/messages/count_tokens", post(messages_count_tokens))
        .merge(if mcp_settings.enabled {
            Router::new().route(
                "/mcp",
                post(crate::http::mcp)
                    .get(mcp_gateway::unsupported_transport)
                    .delete(mcp_gateway::unsupported_transport)
                    .layer(axum::extract::DefaultBodyLimit::max(
                        mcp_gateway::MAX_BODY_BYTES,
                    )),
            )
        } else {
            Router::new()
        })
        .route("/v1/tools", get(crate::http::list_tools))
        .route("/v1/tool-calls", post(tool_calls))
        // Budget endpoints.
        .route("/v1/budget", get(get_budget).post(set_budget))
        .route("/v1/budget/history", get(get_budget_history))
        // Context compaction settings endpoint. Keep the former path as a
        // compatibility alias for existing clients.
        .route(
            "/v1/config/compaction",
            get(get_compression_config).put(update_compression_config),
        )
        .route(
            "/v1/config/compression",
            get(get_compression_config).put(update_compression_config),
        )
        // Gemini native API (for Gemini CLI / google-generative-ai SDK).
        // The Gemini SDK sends URLs like /v1beta/models/gemini-pro:generateContent
        // where :generateContent is an action suffix. Axum 0.7 can't have two
        // colons in one path segment, so we use a wildcard /*model and
        // dispatch based on the suffix in a single handler.
        .route("/v1beta/models", get(gemini_list_models))
        .route("/v1beta/models/*model", post(crate::http::gemini_dispatch))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .layer(middleware::from_fn(log_request))
        .layer(axum::extract::Extension(std::sync::Arc::new(verbose)))
        .with_state(state)
        .layer(axum::middleware::from_fn_with_state(
            engine_core::resilience::Admission::from_env(32)?,
            engine_core::resilience::admit,
        ));

    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;

    // Notify any waiter that the server socket is bound and accepting.
    if let Some(tx) = startup_tx {
        let _ = tx.send(());
    }

    // Record host/port *and* our own PID so the desktop app can stop us
    // without relying on a network-accessible /shutdown endpoint.
    let daemon_config = DaemonConfig {
        host: host.clone(),
        port,
        pid: Some(std::process::id()),
    };
    if let Err(e) = daemon_config.save() {
        eprintln!(
            "{}  {}",
            timestamp().dimmed(),
            format!("Failed to save daemon config: {}", e).yellow()
        );
    }

    let server = axum::serve(listener, app);
    if let Some(rx) = shutdown_rx {
        server
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await?;
    } else {
        server.await?;
    }

    print_cumulative_cost();

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn start_single_provider_daemon(
    host: String,
    port: u16,
    db_path: String,
    verbose: bool,
    provider: std::sync::Arc<dyn engine_core::Provider>,
    mode: &str,
    project_path: Option<String>,
    shutdown_rx: Option<tokio::sync::oneshot::Receiver<()>>,
    startup_tx: Option<tokio::sync::oneshot::Sender<()>>,
) -> Result<()> {
    let config = crate::local_config::load_local_config();
    warn_if_permissive_action_policy(&config.action_policy);
    let otel = config.otel();
    let endpoint = verified_otel_endpoint(&otel).await;
    let _telemetry_guard = telemetry::maybe_init_with_config_skip_health_check(
        otel.service_name.clone(),
        endpoint,
        otel.insecure,
    );
    if verbose {
        tracing::debug!("verbose logging requested via --verbose");
    }

    let startup_message = format!(
        "{} {}:{}  {}  {}",
        "ReinCore single-provider proxy starting on".green().bold(),
        host.white().bold(),
        port.to_string().white().bold(),
        format!("provider={}", provider.label()).magenta(),
        format!("mode={}", mode).cyan(),
    );
    println!("{}  {}", timestamp().dimmed(), startup_message);

    let db_path = shellexpand::tilde(&db_path).to_string();
    let db = LocalDatabase::open(&db_path).unwrap_or_else(|e| {
        eprintln!(
            "{}  {}",
            timestamp().dimmed(),
            format!(
                "Failed to open SQLite DB at {}: {}. Falling back to in-memory.",
                db_path, e
            )
            .yellow()
        );
        LocalDatabase::open_in_memory().expect("in-memory DB must open")
    });

    let _metrics_server = engine_core::operational::start_from_env().await?;
    engine_core::operational::gauge(engine_core::operational::Gauge::RelayConfigured, Some(0.0));
    let monitor_db = db.clone();
    let _dependency_sampler = _metrics_server.as_ref().map(|_| {
        engine_core::operational::sampler(move || {
            let db = monitor_db.clone();
            async move {
                db.observe_dependencies().await;
            }
        })
    });
    let mut state =
        SingleProviderAppState::new_with_project_path(db, provider, project_path.as_deref())
            .with_api_mode(mode)
            .with_verbose(verbose);
    let mcp_settings = crate::mcp_config::settings()?;
    state.tool_executors = std::sync::Arc::new(crate::mcp_config::registry(&mcp_settings).await?);
    engine_core::batch_jobs::resume_pending(state.db.clone(), state.provider.clone()).await?;

    let app = Router::new()
        .route("/health", get(health_single_provider))
        .route("/livez", get(engine_core::resilience::live))
        .route("/readyz", get(crate::http::ready_single_provider))
        .route("/v1/models", get(list_models_single_provider))
        // Alias without /v1 prefix - VS Code BYOK validation calls {base_url}/models.
        .route("/models", get(list_models_single_provider))
        .route("/v1/projects", get(list_projects_single_provider))
        .route(
            "/v1/local-projects",
            get(list_local_projects_single_provider).post(create_local_project_single_provider),
        )
        .route(
            "/v1/local-projects/resolve",
            get(resolve_local_project_single_provider),
        )
        .route(
            "/v1/local-projects/:id/directories",
            post(assign_local_project_directory_single_provider),
        )
        .route(
            "/v1/local-projects/:id",
            patch(rename_local_project_single_provider),
        )
        .route("/v1/cache-hit-rate", get(cache_hit_rate_single_provider))
        .route("/v1/cache-utilization", get(cache_hit_rate_single_provider))
        .route(
            "/v1/sessions/:id/diagnostics",
            get(crate::http::get_session_diagnostics_single_provider),
        )
        .route(
            "/v1/cost-diagnostics",
            get(crate::http::cost_diagnostics_single_provider),
        )
        .route(
            "/v1/checkpoint-redemptions",
            get(checkpoint_redemptions_single_provider),
        )
        // Provider & config management endpoints (same handlers - they don't
        // depend on AppState).
        .route("/v1/providers", get(list_providers))
        .route(
            "/v1/providers/:provider/credentials",
            post(add_provider_credentials).delete(remove_provider_credentials),
        )
        .route("/v1/providers/:provider/models", get(list_provider_catalog))
        .route("/v1/providers/ollama/models", post(list_ollama_models))
        .route("/v1/providers/custom/models", post(list_custom_models))
        .route(
            "/v1/config/routing",
            get(get_routing_config).put(update_routing_config),
        )
        .route(
            "/v1/config/providers/:provider",
            get(get_provider_config).put(update_provider_config),
        )
        .route(
            "/v1/sessions",
            post(create_session_single_provider).get(list_sessions_single_provider),
        )
        .route(
            "/v1/sessions/:id",
            get(get_session_single_provider).patch(rename_session_single_provider),
        )
        .route(
            "/v1/sessions/:id/subagents",
            get(list_subagent_sessions_single_provider),
        )
        .route(
            "/v1/sessions/:id/events",
            get(get_session_events_single_provider),
        )
        .route(
            "/v1/sessions/:id/events/stream",
            get(stream_session_events_single_provider),
        )
        .route(
            "/v1/sessions/:id/events/page",
            get(get_session_event_page_single_provider),
        )
        .route(
            "/v1/sessions/:id/event/:seq",
            get(get_session_event_single_provider),
        )
        .route(
            "/v1/sessions/:id/timeline",
            get(get_session_timeline_single_provider),
        )
        .route(
            "/v1/sessions/:id/tasks",
            get(get_session_tasks_single_provider),
        )
        .route(
            "/v1/sessions/:id/tasks/:task_id/decisions",
            get(get_task_decisions_single_provider),
        )
        .route(
            "/v1/sessions/:id/pause",
            post(pause_session_single_provider),
        )
        .route(
            "/v1/sessions/:id/complete",
            post(complete_session_single_provider),
        )
        .route(
            "/v1/sessions/:id/framework-checkpoint",
            post(framework_checkpoint_session_single_provider),
        )
        .route(
            "/v1/sessions/:id/framework-memory",
            post(framework_memory_session_single_provider),
        )
        .route(
            "/v1/sessions/:id/framework-turn",
            post(framework_record_turn_single_provider),
        )
        .route(
            "/v1/checkpoints/:token/resume",
            post(resume_session_single_provider),
        )
        .route(
            "/v1/framework-checkpoints/:token/resume",
            post(framework_resume_session_single_provider),
        )
        .route(
            "/v1/sessions/:id/batch-events",
            post(crate::http::record_batch_event_single_provider),
        )
        .route(
            "/v1/batches",
            post(crate::http::create_batch_single_provider),
        )
        .route(
            "/v1/batches/:id",
            get(crate::http::get_batch_single_provider),
        )
        .route(
            "/v1/batches/:id/cancel",
            post(crate::http::cancel_batch_single_provider),
        )
        .route(
            "/v1/batches/:id/results",
            get(crate::http::results_batch_single_provider),
        )
        .route(
            "/v1/chat/completions",
            post(chat_completions_single_provider),
        )
        .route("/v1/responses", post(responses_api_single_provider))
        // Muse Code aliases (see multi-provider router above).
        .route("/responses", post(responses_api_single_provider))
        .route("/muse-code/models", get(muse_code_models_single_provider))
        .route("/v1/messages", post(messages_api_single_provider))
        .route(
            "/v1/messages/count_tokens",
            post(messages_count_tokens_single_provider),
        )
        .merge(if mcp_settings.enabled {
            Router::new().route(
                "/mcp",
                post(crate::http::mcp_single_provider)
                    .get(mcp_gateway::unsupported_transport)
                    .delete(mcp_gateway::unsupported_transport)
                    .layer(axum::extract::DefaultBodyLimit::max(
                        mcp_gateway::MAX_BODY_BYTES,
                    )),
            )
        } else {
            Router::new()
        })
        .route("/v1/tools", get(crate::http::list_tools_single_provider))
        .route("/v1/tool-calls", post(tool_calls_single_provider))
        // Budget endpoints (single-provider variants).
        .route(
            "/v1/budget",
            get(get_budget_single_provider).post(set_budget_single_provider),
        )
        .route(
            "/v1/budget/history",
            get(get_budget_history_single_provider),
        )
        // Context compaction settings endpoint (single-provider variant),
        // plus the former compatibility path.
        .route(
            "/v1/config/compaction",
            get(get_compression_config_single_provider)
                .put(update_compression_config_single_provider),
        )
        .route(
            "/v1/config/compression",
            get(get_compression_config_single_provider)
                .put(update_compression_config_single_provider),
        )
        // Gemini native API (for Gemini CLI / google-generative-ai SDK).
        .route("/v1beta/models", get(gemini_list_models_single_provider))
        .route(
            "/v1beta/models/*model",
            post(crate::http::gemini_dispatch_single_provider),
        )
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .layer(middleware::from_fn(log_request))
        .layer(axum::extract::Extension(std::sync::Arc::new(verbose)))
        .with_state(state)
        .layer(axum::middleware::from_fn_with_state(
            engine_core::resilience::Admission::from_env(32)?,
            engine_core::resilience::admit,
        ));

    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;

    // Notify any waiter that the server socket is bound and accepting.
    if let Some(tx) = startup_tx {
        let _ = tx.send(());
    }

    let daemon_config = DaemonConfig {
        host: host.clone(),
        port,
        pid: Some(std::process::id()),
    };
    if let Err(e) = daemon_config.save() {
        eprintln!(
            "{}  {}",
            timestamp().dimmed(),
            format!("Failed to save daemon config: {}", e).yellow()
        );
    }

    let server = axum::serve(listener, app);
    if let Some(rx) = shutdown_rx {
        server
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await?;
    } else {
        server.await?;
    }

    print_cumulative_cost();

    Ok(())
}

pub async fn show_status(host: String, port: u16) -> Result<()> {
    let url = format!("http://{}:{}/health", host, port);
    match reqwest::get(&url).await {
        Ok(resp) if resp.status().is_success() => {
            let body = resp
                .json::<serde_json::Value>()
                .await
                .unwrap_or_else(|_| json!({}));
            println!("ReinCore daemon is running on {}:{}", host, port);
            if let Some(edition) = body.get("edition").and_then(|v| v.as_str()) {
                println!("  Edition: {}", edition);
            }
        }
        Ok(resp) => {
            println!(
                "ReinCore daemon responded with non-OK status: {}",
                resp.status()
            );
        }
        Err(e) => {
            println!("ReinCore daemon is not running on {}:{}", host, port);
            println!("  Error: {}", e);
        }
    }
    Ok(())
}

pub async fn stop_daemon(_host: String, _port: u16) -> Result<()> {
    let config = DaemonConfig::load();
    match config.pid {
        Some(pid) => {
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                const CREATE_NO_WINDOW: u32 = 0x08000000;
                let status = std::process::Command::new("taskkill.exe")
                    .arg("/F")
                    .arg("/PID")
                    .arg(pid.to_string())
                    .creation_flags(CREATE_NO_WINDOW)
                    .status();
                match status {
                    Ok(s) if s.success() => {
                        println!("Stopped ReinCore daemon (PID {})", pid);
                    }
                    Ok(s) => {
                        println!("taskkill exited with status {} for PID {}", s, pid);
                    }
                    Err(e) => {
                        println!("Failed to run taskkill for PID {}: {}", pid, e);
                    }
                }
            }
            #[cfg(unix)]
            {
                match nix::sys::signal::kill(
                    nix::unistd::Pid::from_raw(pid as i32),
                    nix::sys::signal::Signal::SIGTERM,
                ) {
                    Ok(_) => {
                        println!("Sent SIGTERM to ReinCore daemon (PID {})", pid);
                    }
                    Err(e) => {
                        println!("Failed to send SIGTERM to PID {}: {}", pid, e);
                    }
                }
            }
        }
        None => {
            println!("No daemon PID recorded in ~/.reincore-io/daemon.json");
        }
    }
    Ok(())
}
