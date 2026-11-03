// ReinCore desktop app - library entry point.
//
// The Tauri app manages the ReinCore engine as a daemon.
//
// In **dev mode** (debug build), the daemon runs as an **in-process** tokio
// task. This means daemon logs appear directly in the Tauri dev console,
// making it easy to debug routing, SQL, and provider issues. The daemon's
// lifetime is tied to the desktop app - when you close the dev window, the
// daemon stops too.
//
// In **production** (release build), the daemon runs as a separate,
// **out-of-process** detached child process. This gives us two important
// properties:
//
//   1. The daemon keeps running after the desktop window is closed.
//   2. Re-launching the desktop app reconnects to an already-running daemon
//      instead of spawning a second one.
//
// On startup we read `~/.reincore-io/daemon.json` (the same file the CLI writes)
// to discover the daemon's host:port, then ping `/health`. If the daemon is
// already serving, we reuse it. If not, we start it (in-process in dev,
// out-of-process in production).

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use engine_core::{
    otel_health, CompressionMode, CompressionSettings, LoopSettings, PrivacyPolicy, SettingsStore,
};
use reincore::cli::DaemonConfig;
use reincore::db::LocalDatabase;
use reincore::local_config::read_provider_models;
use reincore::notifications::LocalNotificationSettings;
use serde::{Deserialize, Serialize};
use tauri::Manager;

/// State exposed to Tauri commands.
#[derive(Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub running: bool,
    pub host: String,
    pub port: u16,
    pub network_access: bool,
    pub mode: Option<String>,
    pub provider: Option<String>,
    pub api_mode: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DesktopCompressionSettings {
    pub enabled: bool,
    pub mode: String,
}

#[derive(Clone, Serialize)]
pub struct DesktopPrivacySettings {
    #[serde(flatten)]
    pub policy: PrivacyPolicy,
    pub environment_override: bool,
    pub config_path: String,
}

#[derive(Default, Deserialize)]
struct DaemonHealth {
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    api_mode: Option<String>,
    #[serde(default)]
    providers: Vec<String>,
}

/// Dev-mode handle to the in-process daemon's graceful-shutdown channel.
/// In release builds this is unused; the daemon is a separate OS process.
#[cfg(debug_assertions)]
#[derive(Default)]
pub struct InProcessDaemonHandle {
    pub shutdown_tx: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
}

fn default_daemon_config() -> DaemonConfig {
    DaemonConfig::load()
}

/// Discover the `reincore` (or `reincore.exe`) binary that should be spawned
/// as the out-of-process daemon. The desktop app looks in several places so
/// it works during development, from a Tauri bundle, and from a manual install:
///
/// 1. The same directory as the running desktop executable.
/// 2. Tauri's `resources/` folder next to (or inside) the app bundle.
/// 3. `target/{debug,release}/reincore[.exe]` one directory up from the exe,
///    which is the common dev layout.
/// 4. Any `reincore` binary on the user's PATH.
#[cfg_attr(debug_assertions, allow(dead_code))]
fn find_daemon_binary() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let exe_dir = exe.parent()?;

    let bin_name = if cfg!(windows) {
        "reincore.exe"
    } else {
        "reincore"
    };

    // 1. Same directory as the desktop binary.
    let candidate = exe_dir.join(bin_name);
    if candidate.exists() {
        return Some(candidate);
    }

    // 2. Tauri resources folder.
    #[cfg(windows)]
    let resources_dir = exe_dir.join("resources");
    #[cfg(target_os = "macos")]
    let resources_dir = exe_dir
        .parent()
        .map(|p| p.join("Resources"))
        .unwrap_or_else(|| exe_dir.join("resources"));
    #[cfg(all(not(windows), not(target_os = "macos")))]
    let resources_dir = exe_dir.join("resources");

    let candidate = resources_dir.join(bin_name);
    if candidate.exists() {
        return Some(candidate);
    }

    // 3. Cargo target dir fallback (dev layout: target/debug/reincore-app.exe
    //    next to target/debug/reincore.exe).
    for profile in ["debug", "release"] {
        let candidate = exe_dir
            .parent()
            .map(|p| p.join(profile).join(bin_name))
            .unwrap_or_default();
        if candidate.exists() {
            return Some(candidate);
        }
    }

    // 4. PATH fallback.
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let candidate = dir.join(bin_name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    None
}

fn daemon_connect_host(host: &str) -> &str {
    match host {
        "0.0.0.0" | "::" | "[::]" => "127.0.0.1",
        other => other,
    }
}

fn network_access_enabled(config: &DaemonConfig) -> bool {
    matches!(config.host.as_str(), "0.0.0.0" | "::" | "[::]")
}

async fn daemon_health(config: &DaemonConfig) -> Option<DaemonHealth> {
    let url = format!(
        "http://{}:{}/health",
        daemon_connect_host(&config.host),
        config.port
    );
    match reqwest::Client::new()
        .get(&url)
        .timeout(Duration::from_secs(2))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => resp.json::<DaemonHealth>().await.ok(),
        _ => None,
    }
}

async fn is_daemon_running(config: &DaemonConfig) -> bool {
    daemon_health(config).await.is_some()
}

/// Spawn the `reincore start` binary as a detached child process so it
/// survives the desktop app closing. (Production only - dev uses in-process.)
///
/// On Windows this uses `CREATE_NO_WINDOW | DETACHED_PROCESS` creation
/// flags. On Unix it calls `setsid()` in a `pre_exec` hook so the child
/// becomes a session leader and is not killed when the parent's terminal
/// session ends.
///
/// The daemon's stdout and stderr are redirected to `~/.reincore-io/daemon.log`
/// so startup failures can be inspected after the fact.
#[cfg_attr(debug_assertions, allow(dead_code))]
fn spawn_daemon_binary(config: &DaemonConfig) -> Result<std::process::Child, String> {
    let bin = find_daemon_binary().ok_or_else(|| {
        "Could not find the `reincore` daemon binary. Make sure it is built and either next to the desktop app, in resources/, or on PATH.".to_string()
    })?;

    let db_path = shellexpand::tilde("~/.reincore-io/reincore.db").to_string();
    let log_path = shellexpand::tilde("~/.reincore-io/daemon.log").to_string();

    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("Failed to open daemon log {}: {}", log_path, e))?;

    tracing::info!("Spawning daemon binary: {}", bin.display());

    let mut cmd = Command::new(&bin);
    cmd.arg("start")
        .arg("--host")
        .arg(&config.host)
        .arg("--port")
        .arg(config.port.to_string())
        .arg("--db-path")
        .arg(db_path)
        .stdout(log_file.try_clone().map_err(|e| e.to_string())?)
        .stderr(log_file)
        .stdin(Stdio::null());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        const DETACHED_PROCESS: u32 = 0x00000008;
        cmd.creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS);
    }

    #[cfg(unix)]
    unsafe {
        use std::os::unix::process::CommandExt;
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }

    let child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn daemon {}: {}", bin.display(), e))?;
    tracing::info!("Spawned daemon process with PID {}", child.id());
    Ok(child)
}

/// Restart the daemon in single-provider mode with telemetry carried over.
///
/// This is used by `restart_daemon` when the prior daemon was in single-provider
/// mode, so the restart preserves the provider, api_mode, and telemetry flag.
#[cfg_attr(debug_assertions, allow(dead_code))]
async fn spawn_daemon_binary_with_provider(
    config: &DaemonConfig,
    provider: &str,
    mode: &str,
) -> Result<std::process::Child, String> {
    let bin = find_daemon_binary().ok_or_else(|| {
        "Could not find the `reincore` daemon binary. Make sure it is built and either next to the desktop app, in resources/, or on PATH.".to_string()
    })?;

    let db_path = shellexpand::tilde("~/.reincore-io/reincore.db").to_string();
    let log_path = shellexpand::tilde("~/.reincore-io/daemon.log").to_string();

    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("Failed to open daemon log {}: {}", log_path, e))?;

    tracing::info!(
        "Spawning daemon binary (single-provider={}, mode={}): {}",
        provider,
        mode,
        bin.display()
    );

    let mut cmd = Command::new(&bin);
    cmd.arg("start")
        .arg("--provider")
        .arg(provider)
        .arg("--mode")
        .arg(mode)
        .arg("--host")
        .arg(&config.host)
        .arg("--port")
        .arg(config.port.to_string())
        .arg("--db-path")
        .arg(db_path);

    // Ollama: pass the saved standard model (and cloud mode when applicable)
    // so the restarted daemon doesn't try to prompt interactively for a model
    // with stdin detached. Without --model, `setup_ollama_local` falls back to
    // a blocking interactive prompt the detached daemon can never answer.
    //
    // Cloud mode mirrors `start_daemon_with_provider_impl`: prefer the local
    // Ollama server when it is running (it can also proxy `:cloud` models);
    // only use direct cloud mode when there is no local server AND the user
    // has an Ollama API key saved.
    if provider == "ollama" {
        if let Some(model) = read_provider_models("ollama")
            .standard
            .filter(|s| !s.trim().is_empty())
        {
            cmd.arg("--model").arg(model);
        }
        let local_server_running = reincore::launch::is_ollama_server_running_pub(
            reincore::launch::default_ollama_base_url(),
        )
        .await;
        let has_api_key = reincore::credentials::get_credential("ollama")
            .map(|cred| cred.kind == reincore::credentials::CredentialKind::ApiKey)
            .unwrap_or(false);
        if !local_server_running && has_api_key {
            cmd.arg("--cloud");
        }
    }

    cmd.stdout(log_file.try_clone().map_err(|e| e.to_string())?)
        .stderr(log_file)
        .stdin(Stdio::null());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        const DETACHED_PROCESS: u32 = 0x00000008;
        cmd.creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS);
    }

    #[cfg(unix)]
    unsafe {
        use std::os::unix::process::CommandExt;
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }

    let child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn daemon {}: {}", bin.display(), e))?;
    tracing::info!("Spawned daemon process with PID {}", child.id());
    Ok(child)
}

/// Start the daemon **in-process** as a tokio task (dev mode only).
///
/// In debug builds, running the daemon in-process means its log output
/// (request logs, SQL errors, provider debugging) goes directly to the same
/// stdout/stderr as the Tauri dev console, making it easy to see what the
/// daemon is doing without tailing a separate log file.
///
/// The daemon lifetime is tied to the desktop app - closing the window stops
/// the daemon. This is intentional in dev; production builds use the
/// out-of-process detached daemon instead.
#[cfg(debug_assertions)]
async fn start_daemon_in_process(config: &DaemonConfig) -> tokio::sync::oneshot::Sender<()> {
    let host = config.host.clone();
    let port = config.port;
    let db_path = shellexpand::tilde("~/.reincore-io/reincore.db").to_string();

    tracing::info!(
        "Starting ReinCore daemon **in-process** on {}:{} (dev mode)",
        host,
        port
    );

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let (startup_tx, startup_rx) = tokio::sync::oneshot::channel::<()>();

    tauri::async_runtime::spawn(async move {
        if let Err(e) = reincore::daemon::start_daemon(
            host,
            port,
            db_path,
            true, // verbose - show full request/response logs in dev
            None, // provider - multi-provider mode
            "interactive".to_string(),
            None,  // path
            false, // cloud
            None,  // model
            Some(shutdown_rx),
            Some(startup_tx),
        )
        .await
        {
            tracing::error!("In-process daemon exited with error: {}", e);
        }
    });

    // Wait until the daemon has bound its listener before returning so callers
    // can rely on it being ready immediately.
    let _ = startup_rx.await;
    shutdown_tx
}

/// Start the daemon **in-process** in single-provider mode (dev mode only).
///
/// This is the single-provider counterpart to [`start_daemon_in_process`]. It
/// is used by `restart_daemon` when the prior daemon was in single-provider
/// mode, so the restart preserves the provider and api_mode instead of
/// falling back to multi-provider mode.
///
/// For Ollama, the saved standard model (and cloud mode when the local
/// server is down but an API key exists) is passed through so the restarted
/// daemon doesn't block on an interactive model prompt.
#[cfg(debug_assertions)]
async fn start_daemon_in_process_with_provider(
    config: &DaemonConfig,
    provider: &str,
    mode: &str,
) -> tokio::sync::oneshot::Sender<()> {
    let host = config.host.clone();
    let port = config.port;
    let db_path = shellexpand::tilde("~/.reincore-io/reincore.db").to_string();
    let provider_label = provider.to_string();
    let wire_mode = mode.to_string();

    // Ollama needs its saved standard model passed explicitly (the in-process
    // daemon cannot prompt interactively), plus cloud mode when there is no
    // local server but an Ollama API key is saved.
    let (ollama_model, ollama_is_cloud) = if provider == "ollama" {
        let model = read_provider_models("ollama")
            .standard
            .filter(|s| !s.trim().is_empty());
        let local_server_running = reincore::launch::is_ollama_server_running_pub(
            reincore::launch::default_ollama_base_url(),
        )
        .await;
        let has_api_key = reincore::credentials::get_credential("ollama")
            .map(|cred| cred.kind == reincore::credentials::CredentialKind::ApiKey)
            .unwrap_or(false);
        (model, !local_server_running && has_api_key)
    } else {
        (None, false)
    };

    tracing::info!(
        "Starting ReinCore daemon **in-process** on {}:{} (dev mode, single-provider={}, mode={})",
        host,
        port,
        provider_label,
        wire_mode,
    );

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let (startup_tx, startup_rx) = tokio::sync::oneshot::channel::<()>();

    tauri::async_runtime::spawn(async move {
        if let Err(e) = reincore::daemon::start_daemon(
            host,
            port,
            db_path,
            true,                 // verbose
            Some(provider_label), // single-provider mode
            wire_mode,
            None,            // path
            ollama_is_cloud, // cloud
            ollama_model,    // model
            Some(shutdown_rx),
            Some(startup_tx),
        )
        .await
        {
            tracing::error!("In-process daemon exited with error: {}", e);
        }
    });

    let _ = startup_rx.await;
    shutdown_tx
}

/// Read telemetry settings from `~/.reincore-io/config.toml` for the desktop UI.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TelemetrySettings {
    pub enabled: bool,
}

fn telemetry_settings() -> TelemetrySettings {
    TelemetrySettings {
        enabled: reincore::local_config::load_local_config().otel().enabled,
    }
}

fn save_telemetry_settings(enabled: bool) -> Result<(), String> {
    let config = reincore::local_config::TelemetryConfig {
        enabled,
        ..Default::default()
    }
    .with_defaults();
    reincore::local_config::set_otel_config(&config).map_err(|e| e.to_string())
}

/// Ensure a daemon is running, blocking until it actually serves `/health`.
///
/// If a daemon is already serving (`/health` returns 200), reuse it. Otherwise
/// start one:
/// - In **dev** (debug build): run the daemon in-process as a tokio task so
///   its logs appear in the dev console.
/// - In **production** (release build): spawn a detached `reincore` binary
///   that outlives the desktop app.
///
/// Running this inside Tauri's `.setup()` hook guarantees the backend is up
/// before the webview loads, which prevents the early Vite proxy errors in
/// dev mode.
async fn ensure_daemon_running(app_handle: tauri::AppHandle) {
    let config = default_daemon_config();

    if is_daemon_running(&config).await {
        tracing::info!(
            "ReinCore daemon already running on {}:{} - reusing it.",
            config.host,
            config.port
        );
        return;
    }

    #[cfg(debug_assertions)]
    {
        // Dev mode: start the daemon in-process.
        let shutdown_tx = start_daemon_in_process(&config).await;
        if let Some(handle) = app_handle.try_state::<InProcessDaemonHandle>() {
            let mut guard = handle.shutdown_tx.lock().unwrap();
            *guard = Some(shutdown_tx);
        }
    }

    #[cfg(not(debug_assertions))]
    {
        // Production: spawn a detached out-of-process daemon.
        match spawn_daemon_binary(&config) {
            Ok(_) => {
                tracing::info!(
                    "Spawned ReinCore daemon on {}:{} as detached process.",
                    config.host,
                    config.port
                );
            }
            Err(e) => {
                tracing::error!("{}", e);
                return;
            }
        }
    }

    // Poll /health until the daemon is ready (up to ~15 s).
    for _ in 0..75 {
        tokio::time::sleep(Duration::from_millis(200)).await;
        if is_daemon_running(&config).await {
            tracing::info!(
                "ReinCore daemon is now serving on {}:{}.",
                config.host,
                config.port
            );
            return;
        }
    }
    tracing::warn!(
        "ReinCore daemon started but /health is not responding yet on {}:{}.",
        config.host,
        config.port
    );
}

/// Tauri command: return the daemon's address and whether it is actually
/// serving. The frontend should use this to display the current host:port and
/// a running/stopped badge.
#[tauri::command]
async fn daemon_status() -> DaemonStatus {
    let config = default_daemon_config();
    let health = daemon_health(&config).await;
    let running = health.is_some();
    let network_access = network_access_enabled(&config);
    let mode = health.as_ref().and_then(|value| value.mode.clone());
    let api_mode = health.as_ref().and_then(|value| value.api_mode.clone());
    let provider = health.as_ref().and_then(|value| {
        (value.mode.as_deref() == Some("single_provider"))
            .then(|| value.providers.first().cloned())
            .flatten()
    });
    DaemonStatus {
        running,
        host: config.host,
        port: config.port,
        network_access,
        mode,
        provider,
        api_mode,
    }
}

fn desktop_database() -> Result<LocalDatabase, String> {
    let path = shellexpand::tilde("~/.reincore-io/reincore.db").to_string();
    LocalDatabase::open(&path).map_err(|error| format!("Failed to open {}: {}", path, error))
}

#[tauri::command]
async fn get_compression_settings() -> Result<DesktopCompressionSettings, String> {
    let settings = desktop_database()?
        .get_compression_settings()
        .await
        .map_err(|error| error.to_string())?;
    Ok(DesktopCompressionSettings {
        enabled: settings.is_enabled(),
        mode: settings
            .mode
            .unwrap_or(CompressionMode::Low)
            .as_str()
            .to_string(),
    })
}

/// Stored loop-detection overrides. Unset fields fall back to the built-in
/// defaults (and to `REINCORE_LOOP_*` where the environment sets them), so an
/// empty object means "use the defaults", not "disable everything".
#[tauri::command]
async fn get_loop_settings() -> Result<LoopSettings, String> {
    desktop_database()?
        .get_loop_settings()
        .await
        .map_err(|error| error.to_string())
}

/// Returns the resolved configuration rather than the stored overrides, so the
/// UI can show what will actually apply after normalization clamps a
/// contradictory pair (a stop threshold below its own nudge threshold).
#[tauri::command]
async fn update_loop_settings(
    settings: LoopSettings,
) -> Result<engine_core::CircuitBreakerConfig, String> {
    desktop_database()?
        .set_loop_settings(settings.clone())
        .await
        .map_err(|error| error.to_string())?;
    Ok(engine_core::CircuitBreaker::new(settings.resolve())
        .config()
        .clone())
}

#[tauri::command]
async fn update_compression_settings(
    enabled: bool,
    mode: String,
) -> Result<DesktopCompressionSettings, String> {
    let parsed_mode =
        CompressionMode::try_from(mode.as_str()).map_err(|error| error.to_string())?;
    let settings = if enabled {
        CompressionSettings::new(parsed_mode)
    } else {
        CompressionSettings::disabled()
    };
    desktop_database()?
        .set_compression_settings(settings)
        .await
        .map_err(|error| error.to_string())?;

    Ok(DesktopCompressionSettings {
        enabled,
        mode: parsed_mode.as_str().to_string(),
    })
}

fn desktop_privacy_settings(policy: PrivacyPolicy) -> DesktopPrivacySettings {
    DesktopPrivacySettings {
        policy,
        environment_override: std::env::var_os("REINCORE_PRIVACY_POLICY").is_some(),
        config_path: reincore::local_config::local_config_path()
            .display()
            .to_string(),
    }
}

/// Return the policy saved in local config.toml. An environment override is
/// reported separately so the UI never presents the TOML value as effective
/// when the daemon will use REINCORE_PRIVACY_POLICY instead.
#[tauri::command]
fn get_privacy_settings() -> Result<DesktopPrivacySettings, String> {
    reincore::local_config::ensure_config_exists().map_err(|error| error.to_string())?;
    Ok(desktop_privacy_settings(
        reincore::local_config::load_local_config().privacy,
    ))
}

/// Save the normalized privacy policy to local config.toml. The separate
/// daemon process applies it after restart.
#[tauri::command]
fn update_privacy_settings(policy: PrivacyPolicy) -> Result<DesktopPrivacySettings, String> {
    let policy = policy.normalized();
    reincore::local_config::set_privacy_config(&policy).map_err(|error| error.to_string())?;
    engine_core::privacy::set_process_default_policy(policy.clone());
    Ok(desktop_privacy_settings(policy))
}

/// The live desktop-notification settings, shared between the poll loop and
/// the commands below.
#[derive(Clone)]
struct NotificationSettingsState(notifications::SharedSettings);

impl NotificationSettingsState {
    fn new(settings: LocalNotificationSettings) -> Self {
        Self(std::sync::Arc::new(std::sync::RwLock::new(settings)))
    }

    /// Publish a new value to the running poll loop. A poisoned lock is
    /// recovered rather than propagated: refusing to change a toast setting
    /// for the rest of the session because something once panicked is worse
    /// than the panic was.
    fn apply(&self, settings: LocalNotificationSettings) {
        match self.0.write() {
            Ok(mut guard) => *guard = settings,
            Err(poisoned) => *poisoned.into_inner() = settings,
        }
    }
}

/// Return the notification settings saved in local config.toml.
///
/// Reading the file rather than the live copy means a hand-edited
/// `[notifications]` table is picked up when the Settings page opens, and the
/// running poll loop is brought into line with it at the same time.
#[tauri::command]
fn get_notification_settings(
    state: tauri::State<'_, NotificationSettingsState>,
) -> Result<LocalNotificationSettings, String> {
    reincore::local_config::ensure_config_exists().map_err(|error| error.to_string())?;
    let settings = reincore::local_config::load_local_config().notifications;
    state.apply(settings);
    Ok(settings)
}

/// Save notification settings to local config.toml and apply them to the
/// running poll loop.
///
/// Unlike the privacy policy, this needs no restart to take effect: the poll
/// loop lives in this process and re-reads the settings on its next tick.
#[tauri::command]
fn update_notification_settings(
    settings: LocalNotificationSettings,
    state: tauri::State<'_, NotificationSettingsState>,
) -> Result<LocalNotificationSettings, String> {
    reincore::local_config::set_notifications_config(&settings)
        .map_err(|error| error.to_string())?;
    state.apply(settings);
    Ok(settings)
}

#[tauri::command]
fn update_network_access(enabled: bool) -> Result<DaemonStatus, String> {
    let mut config = default_daemon_config();
    config.host = if enabled {
        "0.0.0.0".to_string()
    } else {
        "127.0.0.1".to_string()
    };
    config.save().map_err(|error| error.to_string())?;

    Ok(DaemonStatus {
        running: false,
        host: config.host,
        port: config.port,
        network_access: enabled,
        mode: None,
        provider: None,
        api_mode: None,
    })
}

/// Tauri command: ping the daemon's `/health` endpoint and return whether it
/// is actually serving. This is the authoritative check.
#[tauri::command]
async fn ping_daemon() -> bool {
    is_daemon_running(&default_daemon_config()).await
}

/// Tauri command: stop the daemon by killing its recorded PID.
/// Returns `true` if the process no longer responds to `/health`.
///
/// In dev mode (in-process daemon), this sends a graceful shutdown signal to
/// the embedded Axum server, then waits for `/health` to stop responding.
#[tauri::command]
async fn stop_daemon(app_handle: tauri::AppHandle) -> bool {
    #[cfg(debug_assertions)]
    {
        let config = default_daemon_config();

        // Send the graceful shutdown signal to the in-process daemon.
        if let Some(handle) = app_handle.try_state::<InProcessDaemonHandle>() {
            let maybe_tx = {
                let mut guard = handle.shutdown_tx.lock().unwrap();
                guard.take()
            };
            if let Some(tx) = maybe_tx {
                let _ = tx.send(());
            }
        }

        // Wait up to ~3s for the daemon to actually stop responding.
        for _ in 0..15 {
            tokio::time::sleep(Duration::from_millis(200)).await;
            if !is_daemon_running(&config).await {
                return true;
            }
        }
        false
    }

    #[cfg(not(debug_assertions))]
    {
        let config = default_daemon_config();
        let Some(pid) = config.pid else {
            return false;
        };

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            let _ = std::process::Command::new("taskkill.exe")
                .arg("/F")
                .arg("/PID")
                .arg(pid.to_string())
                .creation_flags(CREATE_NO_WINDOW)
                .status();
        }
        #[cfg(unix)]
        {
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }
        }

        // Wait up to ~3s for the daemon to actually stop responding.
        for _ in 0..15 {
            tokio::time::sleep(Duration::from_millis(200)).await;
            if !is_daemon_running(&config).await {
                return true;
            }
        }
        false
    }
}

/// Tauri command: stop any existing daemon and spawn a fresh one.
///
/// In dev mode (in-process daemon), this sends a graceful shutdown signal,
/// waits for `/health` to stop responding, then starts a new in-process
/// daemon. In production, this kills the recorded PID and spawns a new
/// detached daemon.
///
/// **Mode preservation**: Before shutting down, this command queries the
/// running daemon's `/health` to discover its current mode (single-provider
/// vs. multi-provider). If the daemon was in single-provider mode, it is
/// restarted with the same provider and api_mode. Otherwise it falls back
/// to multi-provider mode.
#[tauri::command]
async fn restart_daemon(app_handle: tauri::AppHandle) -> bool {
    let config = default_daemon_config();

    // Capture the current daemon's mode *before* shutting it down so we can
    // restart in the same mode.
    let prior_health = daemon_health(&config).await;
    let single_provider_info: Option<(String, String)> = match &prior_health {
        Some(h) if h.mode.as_deref() == Some("single_provider") => {
            h.providers.first().cloned().map(|provider| {
                let mode = h.api_mode.clone().unwrap_or_else(|| "agentic".to_string());
                (provider, mode)
            })
        }
        _ => None,
    };

    #[cfg(not(debug_assertions))]
    {
        // 1. Kill any recorded daemon PID.
        if let Some(pid) = config.pid {
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                const CREATE_NO_WINDOW: u32 = 0x08000000;
                let _ = std::process::Command::new("taskkill.exe")
                    .arg("/F")
                    .arg("/PID")
                    .arg(pid.to_string())
                    .creation_flags(CREATE_NO_WINDOW)
                    .status();
            }
            #[cfg(unix)]
            {
                unsafe {
                    libc::kill(pid as i32, libc::SIGTERM);
                }
            }
        }

        // 2. Wait for the port to free up.
        for _ in 0..25 {
            tokio::time::sleep(Duration::from_millis(200)).await;
            if !is_daemon_running(&config).await {
                break;
            }
        }

        // 3. Spawn a new detached daemon - preserve single-provider mode.
        if let Some((provider, mode)) = &single_provider_info {
            let _ = spawn_daemon_binary_with_provider(&config, provider, mode).await;
        } else {
            let _ = spawn_daemon_binary(&config);
        }
    }

    #[cfg(debug_assertions)]
    {
        // 1. Send graceful shutdown to the in-process daemon, if any.
        if let Some(handle) = app_handle.try_state::<InProcessDaemonHandle>() {
            let maybe_tx = {
                let mut guard = handle.shutdown_tx.lock().unwrap();
                guard.take()
            };
            if let Some(tx) = maybe_tx {
                let _ = tx.send(());
            }
        }

        // 2. Wait for the port to free up.
        for _ in 0..25 {
            tokio::time::sleep(Duration::from_millis(200)).await;
            if !is_daemon_running(&config).await {
                break;
            }
        }

        // 3. Start a fresh in-process daemon and record its shutdown handle.
        //    Preserve single-provider mode if the old daemon was in it.
        let shutdown_tx = if let Some((provider, mode)) = &single_provider_info {
            start_daemon_in_process_with_provider(&config, provider, mode).await
        } else {
            start_daemon_in_process(&config).await
        };
        if let Some(handle) = app_handle.try_state::<InProcessDaemonHandle>() {
            let mut guard = handle.shutdown_tx.lock().unwrap();
            *guard = Some(shutdown_tx);
        }
    }

    // 4. Wait for /health to come back.
    for _ in 0..30 {
        tokio::time::sleep(Duration::from_millis(200)).await;
        if is_daemon_running(&config).await {
            return true;
        }
    }

    false
}

/// Tauri command: start (or restart) the daemon in **single-provider mode**.
///
/// **Reuse first**: if a daemon is already serving on the configured port in
/// the *same* single-provider mode with the *same* provider label and API
/// mode, it is reused as-is. This avoids tearing down a working proxy (and,
/// for providers like Ollama, re-running model discovery / server checks)
/// every time the user launches an app.
///
/// Otherwise the running daemon (if any) is killed and a fresh one is
/// started restricted to the given `provider` label (e.g. "openai",
/// "ollama"). The `mode` can be "agentic" (default, uses /v1/responses) or
/// "chat" (uses /v1/chat/completions).
///
/// `force_restart` (optional, defaults to `true`) bypasses the reuse check
/// so the daemon is rebuilt even when the requested provider/mode already
/// match. Callers that changed provider configuration (model tiers, API
/// keys) must pass `true` - the daemon freezes provider configs at startup,
/// so a restart is the only way new config takes effect.
///
#[tauri::command]
async fn start_daemon_with_provider(
    app_handle: tauri::AppHandle,
    provider: String,
    mode: String,
    force_restart: Option<bool>,
) -> Result<bool, String> {
    // Default to `true`: existing callers (Settings restart, Providers-page
    // config save, network-access toggle) all expect a rebuild so the daemon
    // picks up new configuration. Only callers that know the daemon just
    // needs to be running - like `launch_app` - opt into reuse.
    start_daemon_with_provider_impl(app_handle, provider, mode, force_restart.unwrap_or(true)).await
}

/// Implementation of [`start_daemon_with_provider`] with an explicit
/// `force_restart` flag (no defaulting here).
async fn start_daemon_with_provider_impl(
    app_handle: tauri::AppHandle,
    provider: String,
    mode: String,
    force_restart: bool,
) -> Result<bool, String> {
    let label = provider.to_lowercase();
    let config = default_daemon_config();

    // 1. Validate the provider label.
    let supported = reincore::provider::supported_providers();
    if !supported.contains(&label) {
        return Err(format!(
            "Unknown provider '{}'. Supported: {}",
            label,
            supported.join(", ")
        ));
    }

    // 2. Check if the provider is configured.
    //    - Copilot uses OAuth - can't be started from the desktop app.
    //    - Ollama doesn't need credentials for local mode.
    //    - All others need an env var or a saved credential.
    if label == "copilot" {
        return Err(
            "Copilot uses OAuth device flow. Run 'reincore provider add copilot' in the CLI first."
                .to_string(),
        );
    }

    let is_ollama = label == "ollama";
    let has_env = std::env::var(reincore::provider::api_key_env_var(&label))
        .ok()
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);
    let has_cred = reincore::credentials::get_credential(&label).is_some();

    if !is_ollama && !has_env && !has_cred {
        return Err(format!(
            "Provider '{}' is not configured. Add an API key in the Providers page first.",
            reincore::provider::provider_display_name(&label)
        ));
    }

    // Ollama requires an explicit standard model to be configured before it
    // can be launched. Without it the daemon would prompt interactively, which
    // is not possible from the desktop UI.
    let ollama_standard_model = if is_ollama {
        read_provider_models("ollama")
            .standard
            .filter(|s| !s.trim().is_empty())
    } else {
        None
    };
    if is_ollama && ollama_standard_model.is_none() {
        return Err(
            "Ollama is not configured. Please select a standard Ollama model on the Providers page first.".to_string()
        );
    }

    // Ollama cloud mode: prefer the local Ollama server when it is running -
    // it can serve both local models and (in recent Ollama versions) cloud
    // models proxied through it. Only fall back to direct cloud mode
    // (`--cloud`, talking to https://ollama.com) when there is no local
    // server AND the user has an Ollama API key saved. Without this, a
    // cloud-only user (API key + `:cloud` model, no local server) could
    // never launch from the desktop app.
    //
    // This runs *after* the reuse check below, so relaunching an app onto an
    // already-running daemon never pays the (up to 2 s) local-server probe.
    let mut ollama_is_cloud = false;

    // 3. Reuse: if a daemon is already serving on the configured port in
    //    single-provider mode with the *same* provider and API mode, keep it.
    //    Relaunching an app with the same provider should not tear down a
    //    healthy proxy and re-run provider setup (for Ollama that means
    //    re-checking the server and re-fetching installed models).
    //
    // Mode names are normalized before comparing because the CLI stores the
    // display label "agent" while the desktop app uses "agentic" - both are
    // the Responses wire API and are interchangeable here.
    if !force_restart {
        let normalize_mode = |m: &str| -> String {
            match m.to_lowercase().as_str() {
                "agent" | "responses" | "agentic" => "agentic".to_string(),
                other => other.to_string(),
            }
        };
        if let Some(health) = daemon_health(&config).await {
            let same_provider = health
                .providers
                .first()
                .map(|p| p.eq_ignore_ascii_case(&label))
                .unwrap_or(false);
            let same_mode = health
                .api_mode
                .as_deref()
                .map(|m| normalize_mode(m) == normalize_mode(&mode))
                .unwrap_or(false);
            if health.mode.as_deref() == Some("single_provider") && same_provider && same_mode {
                tracing::info!(
                    "ReinCore daemon already running in single-provider mode with provider='{}' mode='{}' - reusing it.",
                    label,
                    mode
                );
                return Ok(true);
            }
        }
    }

    // (Re)computing Ollama's local-vs-cloud decision only when we actually
    // need to start a fresh daemon.
    if is_ollama {
        let local_server_running = reincore::launch::is_ollama_server_running_pub(
            reincore::launch::default_ollama_base_url(),
        )
        .await;
        ollama_is_cloud = !local_server_running
            && reincore::credentials::get_credential("ollama")
                .map(|cred| cred.kind == reincore::credentials::CredentialKind::ApiKey)
                .unwrap_or(false);
    }

    // 4. Kill any existing daemon.
    #[cfg(not(debug_assertions))]
    {
        if let Some(pid) = config.pid {
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                const CREATE_NO_WINDOW: u32 = 0x08000000;
                let _ = std::process::Command::new("taskkill.exe")
                    .arg("/F")
                    .arg("/PID")
                    .arg(pid.to_string())
                    .creation_flags(CREATE_NO_WINDOW)
                    .status();
            }
            #[cfg(unix)]
            {
                unsafe {
                    libc::kill(pid as i32, libc::SIGTERM);
                }
            }
        }
        for _ in 0..25 {
            tokio::time::sleep(Duration::from_millis(200)).await;
            if !is_daemon_running(&config).await {
                break;
            }
        }
    }

    // 5. In dev mode, shut down the existing in-process daemon (if any) so
    //    we can bind the same port again in single-provider mode.
    #[cfg(debug_assertions)]
    {
        if let Some(handle) = app_handle.try_state::<InProcessDaemonHandle>() {
            let maybe_tx = {
                let mut guard = handle.shutdown_tx.lock().unwrap();
                guard.take()
            };
            if let Some(tx) = maybe_tx {
                let _ = tx.send(());
            }
        }

        for _ in 0..25 {
            tokio::time::sleep(Duration::from_millis(200)).await;
            if !is_daemon_running(&config).await {
                break;
            }
        }
    }

    // 6. Start the daemon in single-provider mode.
    #[cfg(debug_assertions)]
    {
        let host = config.host.clone();
        let port = config.port;
        let db_path = shellexpand::tilde("~/.reincore-io/reincore.db").to_string();
        let provider_label = label.clone();
        let wire_mode = mode.clone();

        tracing::info!(
            "Starting ReinCore daemon in-process on {}:{} with provider='{}' mode='{}'",
            host,
            port,
            provider_label,
            wire_mode
        );

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let (startup_tx, startup_rx) = tokio::sync::oneshot::channel::<()>();

        tauri::async_runtime::spawn(async move {
            if let Err(e) = reincore::daemon::start_daemon(
                host,
                port,
                db_path,
                true,
                Some(provider_label),
                wire_mode,
                None,
                ollama_is_cloud,
                ollama_standard_model.clone(),
                Some(shutdown_rx),
                Some(startup_tx),
            )
            .await
            {
                tracing::error!("In-process daemon exited with error: {}", e);
            }
        });

        // Wait until the single-provider daemon has bound its listener before
        // returning. This prevents the caller from racing ahead while the port
        // is still coming up.
        let _ = startup_rx.await;

        if let Some(handle) = app_handle.try_state::<InProcessDaemonHandle>() {
            let mut guard = handle.shutdown_tx.lock().unwrap();
            *guard = Some(shutdown_tx);
        }
    }

    #[cfg(not(debug_assertions))]
    {
        let bin = find_daemon_binary()
            .ok_or_else(|| "Could not find the `reincore` daemon binary.".to_string())?;
        let db_path = shellexpand::tilde("~/.reincore-io/reincore.db").to_string();
        let log_path = shellexpand::tilde("~/.reincore-io/daemon.log").to_string();
        let log_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|e| format!("Failed to open daemon log: {}", e))?;

        let mut cmd = Command::new(&bin);
        cmd.arg("start")
            .arg("--provider")
            .arg(&label)
            .arg("--mode")
            .arg(&mode)
            .arg("--host")
            .arg(&config.host)
            .arg("--port")
            .arg(config.port.to_string())
            .arg("--db-path")
            .arg(db_path);

        if label == "ollama" {
            if ollama_is_cloud {
                cmd.arg("--cloud");
            }
            if let Some(ref standard_model) = ollama_standard_model {
                cmd.arg("--model").arg(standard_model);
            }
        }

        cmd.stdout(std::fs::File::try_clone(&log_file).map_err(|e| e.to_string())?)
            .stderr(log_file)
            .stdin(Stdio::null());

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            const DETACHED_PROCESS: u32 = 0x00000008;
            cmd.creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS);
        }

        #[cfg(unix)]
        unsafe {
            use std::os::unix::process::CommandExt;
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }

        cmd.spawn()
            .map_err(|e| format!("Failed to spawn daemon: {}", e))?;
    }

    // 7. Wait for /health to come back.
    for _ in 0..30 {
        tokio::time::sleep(Duration::from_millis(200)).await;
        if is_daemon_running(&config).await {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Tauri command: resolve the path to the local SQLite database. Used by the
/// Settings page to show the user where their data lives.
#[tauri::command]
fn db_path() -> String {
    shellexpand::tilde("~/.reincore-io/reincore.db").to_string()
}

/// Tauri command: resolve the path to the user's ReinCore config directory.
#[tauri::command]
fn config_dir() -> String {
    let home = dirs::home_dir().unwrap_or_default();
    home.join(".reincore-io").display().to_string()
}

// Integrated app launch

/// Metadata about a known integrated app, returned by `list_integrated_apps`.
#[derive(Clone, Serialize, Deserialize)]
pub struct IntegratedApp {
    pub id: String,
    pub display_name: String,
    pub default_provider: String,
    pub installed: bool,
    pub install_hint: Option<String>,
}

/// Map an app id (e.g. "claude", "codex-app", "gemini-cli") to the
/// human-readable product name the user will recognise (e.g. "Claude Code",
/// "Codex", "Gemini CLI"). Falls back to the id itself for unknown apps.
fn app_display_name(id: &str) -> String {
    match id.to_lowercase().as_str() {
        "claude" => "Claude Code".to_string(),
        "codex" | "codex-app" => "Codex".to_string(),
        "copilot" => "GitHub Copilot".to_string(),
        "gemini" | "gemini-cli" => "Gemini CLI".to_string(),
        "cursor" => "Cursor".to_string(),
        "windsurf" => "Windsurf".to_string(),
        "aider" => "Aider".to_string(),
        "continue" => "Continue".to_string(),
        "open-code" | "opencode" | "open code" => "OpenCode".to_string(),
        "openclaw" | "open-claw" | "open claw" => "OpenClaw".to_string(),
        "muse" => "Muse Code".to_string(),
        "ollama" => "Ollama".to_string(),
        other => other.to_string(),
    }
}

/// Tauri command: return the full catalog of integrated apps that ReinCore
/// can launch, each annotated with its default provider, whether it appears
/// to be installed on this machine, and a short install hint if not.
///
/// This is async and offloads the `is_app_installed` checks (which call
/// `which::which()` and do filesystem probes) to a blocking thread so the
/// Tauri main thread / UI doesn't stall while the list is being built.
#[tauri::command]
async fn list_integrated_apps() -> Vec<IntegratedApp> {
    let defaults = reincore::launch::app_defaults();

    // Build the list of (app, provider) pairs to check, then move the
    // blocking work into spawn_blocking so the async runtime stays free.
    let pairs: Vec<(String, String)> = defaults
        .iter()
        .map(|(app, provider)| (app.to_string(), provider.to_string()))
        .collect();

    tauri::async_runtime::spawn_blocking(move || {
        pairs
            .into_iter()
            .map(|(app, provider)| {
                let installed = reincore::launch::is_app_installed(&app);
                let install_hint = reincore::launch::install_hint_for_app(&app);
                IntegratedApp {
                    id: app.clone(),
                    display_name: app_display_name(&app),
                    default_provider: provider,
                    installed,
                    install_hint,
                }
            })
            .collect()
    })
    .await
    .unwrap_or_default()
}

/// Tauri command: check whether a given app id is installed on this machine.
#[tauri::command]
fn is_app_installed_cmd(app: String) -> bool {
    reincore::launch::is_app_installed(&app)
}

/// The launch-routing state the frontend needs before starting an app, as
/// reported by `precheck_launch_app`. Mirrors the CLI's negotiation via
/// `reincore::launch_routing`, but exposes the switch conflict so the
/// GUI can render a dialog instead of a terminal prompt.
#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum LaunchPrecheck {
    /// The daemon is not running / will be started for the requested
    /// provider (or the user already resolved a switch to it).
    StartSingleProvider { provider: String },
    /// Reuse the running multi-provider daemon; the app must send
    /// `X-ReinCore-Provider: <provider>`.
    ReuseMultiProvider { provider: String },
    /// Reuse the running single-provider daemon (same provider + mode).
    ReuseSingleProvider { provider: String },
    /// The daemon runs a different single provider. The frontend must ask
    /// the user how to proceed and re-invoke `launch_app` with the chosen
    /// `switch_decision`.
    NeedsSwitchDecision { requested: String, current: String },
}

/// Tauri command: probe how a launch of `app` (with an optional provider
/// override and wire mode) would negotiate with the currently running
/// daemon. The Launch page calls this before enabling the Launch button and
/// uses `NeedsSwitchDecision` to show the 3-option dialog.
#[tauri::command]
async fn precheck_launch_app(
    app: String,
    provider: Option<String>,
    mode: Option<String>,
) -> Result<LaunchPrecheck, String> {
    let app_label = app.to_lowercase();
    let wire_mode = mode.unwrap_or_else(|| "agentic".to_string());

    // 1. Resolve the provider for this app.
    let provider_label =
        reincore::launch::resolve_provider(&app_label, provider).map_err(|e| e.to_string())?;

    let config = default_daemon_config();
    let daemon_config = reincore::cli::DaemonConfig {
        host: config.host.clone(),
        port: config.port,
        pid: None,
    };
    let plan = reincore::launch_routing::resolve_launch_plan(
        &daemon_config,
        &provider_label,
        Some(wire_mode.as_str()),
        Some(app.as_str()),
    )
    .await
    .map_err(|e| e.to_string())?;

    Ok(match plan {
        reincore::launch_routing::LaunchPlan::StartSingleProvider { provider, .. } => {
            LaunchPrecheck::StartSingleProvider { provider }
        }
        reincore::launch_routing::LaunchPlan::ReuseMultiProvider { provider } => {
            LaunchPrecheck::ReuseMultiProvider { provider }
        }
        reincore::launch_routing::LaunchPlan::ReuseSingleProvider { provider } => {
            LaunchPrecheck::ReuseSingleProvider { provider }
        }
        reincore::launch_routing::LaunchPlan::NeedsSwitchDecision { requested, current } => {
            LaunchPrecheck::NeedsSwitchDecision { requested, current }
        }
    })
}

/// Tauri command: launch an integrated app (e.g. `claude`, `codex`,
/// `gemini`) routed through ReinCore.
///
/// This mirrors `reincore launch <app>` from the CLI, including the
/// negotiation with whatever daemon is already running:
///
///   1. Resolve the provider (explicit override or the app's default).
///   2. Negotiate with the running daemon via
///      `reincore::launch_routing`:
///      - multi-provider daemon + provider configured -> reuse it and make
///        the app send `X-ReinCore-Provider`;
///      - single-provider daemon with the same provider (+mode) -> reuse it;
///      - single-provider daemon with a different provider -> the frontend
///        must first resolve the conflict via `precheck_launch_app` and pass
///        the user's choice as `switch_decision` ("use_requested" restarts
///        the daemon with the requested provider, "use_current" launches
///        on the running provider, "cancel" aborts).
///      - no daemon -> start one in single-provider mode.
///   3. Activate a persistent reincore profile for the app, if we have one.
///   4. Spawn the app in a new terminal window so the user can interact with
///      it while the daemon keeps serving in the background.
///
/// Returns a human-readable status message.
#[tauri::command]
async fn launch_app(
    app_handle: tauri::AppHandle,
    app: String,
    provider: Option<String>,
    mode: Option<String>,
    path: Option<String>,
    switch_decision: Option<String>,
) -> Result<String, String> {
    let app_label = app.to_lowercase();
    let wire_mode = mode.unwrap_or_else(|| "agentic".to_string());

    // 1. Resolve the provider for this app.
    let requested_provider =
        reincore::launch::resolve_provider(&app_label, provider).map_err(|e| e.to_string())?;

    // 2. Make sure the provider is configured (except ollama which is local).
    if requested_provider == "copilot" {
        return Err(
            "Copilot uses OAuth device flow. Run 'reincore provider add copilot' in the CLI first."
                .to_string(),
        );
    }

    // 3. Negotiate with whatever daemon is already running.
    let config = default_daemon_config();
    let daemon_config = reincore::cli::DaemonConfig {
        host: config.host.clone(),
        port: config.port,
        pid: None,
    };
    let plan = reincore::launch_routing::resolve_launch_plan(
        &daemon_config,
        &requested_provider,
        Some(wire_mode.as_str()),
        Some(app.as_str()),
    )
    .await
    .map_err(|e| e.to_string())?;

    // Resolve a switch conflict. `launch_app` may be called twice for one
    // user action: first without a decision (the precheck surfaced a
    // conflict), then with the user's choice.
    let decision: Option<reincore::launch_routing::SwitchDecision> = match &plan {
        reincore::launch_routing::LaunchPlan::NeedsSwitchDecision { requested, current } => {
            match switch_decision.as_deref().map(str::to_lowercase).as_deref() {
                Some("use_requested") => {
                    Some(reincore::launch_routing::SwitchDecision::UseRequested)
                }
                Some("use_current") => Some(reincore::launch_routing::SwitchDecision::UseCurrent),
                Some("cancel") | None => {
                    // No decision supplied: tell the frontend to ask the
                    // user (it should call again with one of the choices).
                    let display = |label: &str| reincore::provider::provider_display_name(label);
                    return Err(format!(
                        "CONFLICT:ReinCore is running in single-provider mode with '{}' but you asked for '{}'. Choose whether to switch (the daemon restarts) or keep the current provider.",
                        display(current),
                        display(requested)
                    ));
                }
                Some(other) => {
                    return Err(format!("Unknown switch decision '{other}'."));
                }
            }
        }
        _ => None,
    };

    let routing =
        reincore::launch_routing::plan_to_routing(&plan, decision).map_err(|e| e.to_string())?;
    // The provider the app will actually run on.
    let provider_label = routing.provider.clone();
    let provider_header = routing.provider_header.as_deref();

    // 4. Validate that the (possibly re-chosen) provider is configured.
    let is_ollama = provider_label == "ollama";
    let has_env = std::env::var(reincore::provider::api_key_env_var(&provider_label))
        .ok()
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);
    let has_cred = reincore::credentials::get_credential(&provider_label).is_some();
    if !is_ollama && !has_env && !has_cred {
        return Err(format!(
            "Provider '{}' is not configured. Add an API key on the Providers page first.",
            reincore::provider::provider_display_name(&provider_label)
        ));
    }
    if is_ollama {
        let has_standard_model = reincore::local_config::read_provider_models("ollama")
            .standard
            .map(|m| !m.trim().is_empty())
            .unwrap_or(false);
        if !has_standard_model {
            return Err(
                "Ollama needs a standard/default model. Select one on the Providers page first."
                    .to_string(),
            );
        }
    }

    // 5. Start (or restart) the daemon when the routing says so.
    let daemon_status = if routing.start_daemon {
        // The user chose to switch providers (or no daemon was running).
        start_daemon_with_provider_impl(
            app_handle,
            provider_label.clone(),
            wire_mode.clone(),
            true, // force restart - the negotiation already decided
        )
        .await?
    } else {
        true
    };
    if !daemon_status {
        return Err(
            "The daemon started but did not become healthy in time. Check the logs and try again."
                .to_string(),
        );
    }

    // 6. Resolve the daemon address for the proxy URLs.
    let config = default_daemon_config();
    let proxy_base_url = format!("http://{}:{}", config.host, config.port);
    let proxy_url = format!("{}/v1", proxy_base_url);

    // 7. Activate a persistent profile for the app (best-effort - not all
    //    apps have one; that's fine, the session-scoped env vars still
    //    apply to the spawned terminal).
    let wire_api = match (wire_mode.to_lowercase().as_str(), app_label.as_str()) {
        (_, "codex" | "codex-app" | "open-code" | "opencode" | "open code" | "muse") => "responses",
        (_, "openclaw" | "open-claw" | "open claw") => "chat",
        ("chat", _) => "chat",
        _ => "responses",
    };
    let profile_msg = match reincore::launch::activate_profile_for_app(
        &app_label,
        &proxy_base_url,
        &proxy_url,
        &provider_label,
        wire_api,
        provider_header,
    ) {
        Ok(reincore::launch::ProfileStatus::Handled) => {
            format!(
                "Activated a persistent ReinCore profile for '{}'. The app will route through {} even outside a ReinCore-launched terminal until you run 'reincore {} --restore'.",
                app_label, proxy_url, app_label
            )
        }
        Ok(reincore::launch::ProfileStatus::Unsupported) => {
            format!(
                "'{}' does not have a persistent profile - routing this session only.",
                app_label
            )
        }
        Err(e) => {
            format!(
                "Warning: could not activate a persistent profile for {}: {}",
                app_label, e
            )
        }
    };

    // 8. Resolve the working directory for the app. If a path is given but
    //    the directory doesn't exist, create it. If no path is given, the
    //    app launches from its own default (usually the user's home dir).
    let working_dir: Option<std::path::PathBuf> = if let Some(p) = path {
        let expanded = shellexpand::tilde(&p).to_string();
        let dir = std::path::PathBuf::from(&expanded);
        if !dir.exists() {
            std::fs::create_dir_all(&dir)
                .map_err(|e| format!("Failed to create directory '{}': {}", p, e))?;
        }
        if !dir.is_dir() {
            return Err(format!("'{}' exists but is not a directory.", p));
        }
        Some(dir)
    } else {
        None
    };

    // 9. Spawn the app in a new terminal window.
    if !reincore::launch::is_app_installed(&app_label) {
        let hint = reincore::launch::install_hint_for_app(&app_label)
            .unwrap_or_else(|| format!("Install '{}' and add it to your PATH.", app_label));
        return Ok(format!(
            "{}\n\n'{}' does not appear to be installed. {}\nThe daemon is running for provider '{}' at {}.",
            profile_msg, app_label, hint, provider_label, proxy_url
        ));
    }

    match reincore::launch::spawn_app_in_terminal(
        &app_label,
        &proxy_base_url,
        &proxy_url,
        working_dir.as_deref(),
        provider_header,
    ) {
        Ok(()) => Ok(format!(
            "{}\n\nStarted '{}' in a new terminal window, routed through provider '{}' at {}{}.",
            profile_msg,
            app_label,
            provider_label,
            proxy_url,
            if provider_header.is_some() {
                " (multi-provider routing)"
            } else {
                ""
            }
        )),
        Err(e) => Ok(format!(
            "{}\n\nWarning: could not auto-launch '{}': {}. The daemon is running at {} - launch the app manually and point it there.",
            profile_msg, app_label, e, proxy_url
        )),
    }
}

/// Read the saved local MCP startup settings.
#[tauri::command]
fn get_mcp_settings() -> Result<reincore::mcp_config::McpSettings, String> {
    reincore::mcp_config::settings().map_err(|error| error.to_string())
}

/// Persist MCP startup settings; the daemon must restart to apply them.
#[tauri::command]
fn update_mcp_settings(enabled: bool) -> Result<reincore::mcp_config::McpSettings, String> {
    reincore::mcp_config::set_enabled(enabled).map_err(|error| error.to_string())
}

/// Tauri command: get telemetry settings.
#[tauri::command]
fn get_telemetry_settings() -> TelemetrySettings {
    telemetry_settings()
}

/// Tauri command: update telemetry settings. When enabling telemetry, start
/// the bundled Jaeger container if it is not already running so traces have a
/// backend to land in.
#[tauri::command]
async fn update_telemetry_settings(enabled: bool) -> Result<TelemetrySettings, String> {
    save_telemetry_settings(enabled)?;

    if enabled {
        let running = reincore::telemetry_cmd::is_jaeger_running()
            .await
            .map_err(|e| e.to_string())?;
        if !running {
            reincore::telemetry_cmd::start_jaeger()
                .await
                .map_err(|e| e.to_string())?;
        }
    }

    Ok(telemetry_settings())
}

/// Tauri command: update telemetry settings and return the current backend
/// health so the settings page can show an immediate green/red status.
#[tauri::command]
async fn update_telemetry_settings_with_health(
    enabled: bool,
) -> Result<TelemetrySettingsWithHealth, String> {
    save_telemetry_settings(enabled)?;

    if enabled {
        let running = reincore::telemetry_cmd::is_jaeger_running()
            .await
            .map_err(|e| e.to_string())?;
        if !running {
            reincore::telemetry_cmd::start_jaeger()
                .await
                .map_err(|e| e.to_string())?;
        }
    }

    let settings = telemetry_settings();
    let health = check_telemetry_health().await?;
    Ok(TelemetrySettingsWithHealth {
        enabled: settings.enabled,
        health,
    })
}

/// Tauri command: check the runtime health of the configured OpenTelemetry
/// backend. Returns the health status message so the desktop app can warn the
/// user when telemetry is enabled but the collector/Jaeger container is not
/// reachable.
#[tauri::command]
async fn check_telemetry_health() -> Result<String, String> {
    let otel = reincore::local_config::load_local_config().otel();
    let endpoint = if otel.enabled {
        otel.endpoint.clone()
    } else {
        None
    };
    let Some(endpoint) = endpoint else {
        return Ok("OpenTelemetry is disabled".to_string());
    };
    let health = otel_health::check(otel_health::OpenTelemetryHealthConfig {
        endpoint: endpoint.clone(),
        is_bundled_jaeger: endpoint == "http://localhost:4317",
    })
    .await;
    Ok(health.message())
}

/// Tauri command: get telemetry settings plus backend health in one call.
/// This lets the settings page display a warning immediately on load.
#[tauri::command]
async fn get_telemetry_settings_with_health() -> Result<TelemetrySettingsWithHealth, String> {
    let settings = telemetry_settings();
    let health = check_telemetry_health().await?;
    Ok(TelemetrySettingsWithHealth {
        enabled: settings.enabled,
        health,
    })
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TelemetrySettingsWithHealth {
    pub enabled: bool,
    pub health: String,
}

/// Tauri command: open the Jaeger UI in the default browser.
#[tauri::command]
async fn open_telemetry_ui() -> Result<(), String> {
    reincore::telemetry_cmd::open_jaeger_ui()
        .await
        .map_err(|e| e.to_string())
}

/// Tauri command: open a native directory picker dialog and return the
/// chosen path, or `null` if the user cancels. Uses the app handle to
/// access the dialog plugin.
#[tauri::command]
async fn pick_directory(app: tauri::AppHandle) -> Option<String> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .set_title("Select a working directory")
        .pick_folder(move |path| {
            let result = path.map(|p| p.to_string());
            let _ = tx.send(result);
        });
    rx.await.unwrap_or(None)
}

mod mcp_auth;
use mcp_auth::{
    cancel_mcp_sign_in, delete_mcp_server, discover_mcp_tools, list_mcp_servers,
    mcp_oauth_metadata, save_mcp_server, sign_in_mcp_server,
};
mod notifications;
mod security;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Set up logging for the desktop app itself. The daemon is a separate
    // process, so its tracing setup cannot conflict with ours anymore.
    tracing_subscriber::fmt::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .manage(InProcessDaemonHandle::default())
        .manage(mcp_auth::McpSignIns::default())
        // Seeded from config here so the poll loop below and the commands
        // that edit it share one value. An unreadable config leaves the
        // defaults in place.
        .manage(NotificationSettingsState::new(
            reincore::local_config::load_local_config().notifications,
        ))
        .invoke_handler(tauri::generate_handler![
            daemon_status,
            ping_daemon,
            stop_daemon,
            restart_daemon,
            start_daemon_with_provider,
            get_compression_settings,
            update_compression_settings,
            get_privacy_settings,
            update_privacy_settings,
            get_notification_settings,
            update_notification_settings,
            get_loop_settings,
            update_loop_settings,
            update_network_access,
            discover_mcp_tools,
            list_mcp_servers,
            save_mcp_server,
            delete_mcp_server,
            mcp_oauth_metadata,
            sign_in_mcp_server,
            cancel_mcp_sign_in,
            get_mcp_settings,
            update_mcp_settings,
            get_telemetry_settings,
            get_telemetry_settings_with_health,
            check_telemetry_health,
            update_telemetry_settings,
            update_telemetry_settings_with_health,
            open_telemetry_ui,
            db_path,
            config_dir,
            list_integrated_apps,
            is_app_installed_cmd,
            precheck_launch_app,
            launch_app,
            pick_directory
        ])
        .setup(|app| {
            // Block the setup hook until the daemon is actually serving.
            // This prevents the webview from loading before the backend is
            // ready, which caused the early Vite proxy ECONNREFUSED errors.
            let handle = app.handle().clone();
            tauri::async_runtime::block_on(ensure_daemon_running(handle));

            // Install guards before the first page loads, after daemon startup.
            tauri::WebviewWindowBuilder::from_config(app, &app.config().app.windows[0])?
                .on_navigation(|url| security::navigation_allowed(url, cfg!(debug_assertions)))
                .on_new_window(|_, _| tauri::webview::NewWindowResponse::Deny)
                .build()?;

            // Desktop toasts. Started after the daemon is serving, so the
            // database exists to read; a failure to open it is logged and
            // never blocks the window.
            notifications::spawn(
                app.handle().clone(),
                std::path::PathBuf::from(
                    shellexpand::tilde("~/.reincore-io/reincore.db").to_string(),
                ),
                // Taken from managed state rather than a captured clone, so
                // the loop is provably reading what the commands write.
                app.state::<NotificationSettingsState>().0.clone(),
            );

            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_title("Reincore");

                // DevTools are opt-in via the RENCORE_DEVTOOLS env var (set
                // by `run.desktop.sh --devtools`). This avoids the default
                // behaviour where DevTools open on every dev launch.
                #[cfg(debug_assertions)]
                if std::env::var("RENCORE_DEVTOOLS").as_deref() == Ok("1") {
                    window.open_devtools();
                }

                let url = window.url();
                tracing::info!("Reincore main window URL: {:?}", url);
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running ReinCore desktop app");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn integrated_apps_list_contains_muse_code() {
        let apps = list_integrated_apps().await;
        let muse = apps
            .iter()
            .find(|app| app.id == "muse")
            .expect("launch list should include muse");
        assert_eq!(muse.display_name, "Muse Code");
        assert_eq!(muse.default_provider, "meta");
    }
}
