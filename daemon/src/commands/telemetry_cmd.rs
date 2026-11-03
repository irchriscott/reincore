use anyhow::{anyhow, Context, Result};
use colored::Colorize;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::cli::TelemetryCommands;

/// Name used for the bundled Jaeger container.
const JAEGER_CONTAINER: &str = "reincore-jaeger";

/// Jaeger OTLP gRPC port that the local daemon should export to.
pub const JAEGER_OTLP_ENDPOINT: &str = "http://localhost:4317";

/// Jaeger UI URL.
pub const JAEGER_UI_URL: &str = "http://localhost:16686";

/// Companion Jaeger v2 configuration mounted by the bundled Compose file.
const JAEGER_CONFIG_FILE: &str = "jaeger-config.yml";

/// Handle `reincore optl` subcommands.
pub async fn handle_telemetry_command(command: TelemetryCommands) -> Result<()> {
    match command {
        TelemetryCommands::Start => start_jaeger().await,
        TelemetryCommands::Stop => stop_jaeger().await,
        TelemetryCommands::Status => show_jaeger_status().await,
        TelemetryCommands::Open => open_jaeger_ui().await,
        TelemetryCommands::Enable => enable_telemetry().await,
        TelemetryCommands::Disable => disable_telemetry().await,
    }
}

/// Start the bundled Jaeger stack and persist OTLP settings in
/// `~/.reincore-io/config.toml` so future daemon starts export traces
/// automatically.
async fn enable_telemetry() -> Result<()> {
    if !docker_available() {
        return Err(anyhow!(
            "Docker is not installed or not in PATH. Install Docker, then run `reincore optl enable`."
        ));
    }
    if !docker_daemon_reachable() {
        return Err(anyhow!("{}", docker_error_message()));
    }

    start_jaeger().await?;
    let config = crate::local_config::TelemetryConfig {
        enabled: true,
        endpoint: Some(JAEGER_OTLP_ENDPOINT.to_string()),
        service_name: Some("reincore-local".to_string()),
        insecure: true,
    };
    crate::local_config::set_otel_config(&config)?;
    println!(
        "{} Telemetry enabled. {} settings saved to {}. Restart the daemon to begin trace export.",
        "✓".green().bold(),
        "OTLP".cyan(),
        crate::local_config::local_config_path()
            .display()
            .to_string()
            .dimmed()
    );
    Ok(())
}

/// Remove persisted OTLP settings from `~/.reincore-io/config.toml` so future
/// daemon starts do not export traces. Leaves the Jaeger container running so
/// existing traces remain viewable.
async fn disable_telemetry() -> Result<()> {
    crate::local_config::set_otel_field("enabled", "false")?;
    println!(
        "{} Telemetry disabled. Updated {}. Restart the daemon to stop trace export.",
        "✓".green().bold(),
        crate::local_config::local_config_path()
            .display()
            .to_string()
            .dimmed()
    );
    Ok(())
}

/// Resolve the bundled `jaeger.yml` compose file path.
///
/// In development this is the source tree layout. In a production build the
/// file is bundled next to the binary via Cargo's `include_str!` plus an
/// embedded-resources mechanism; we fall back to writing a known compose
/// files into `~/.reincore-io/` if the source-tree path does not exist.
fn jaeger_compose_path() -> Result<PathBuf> {
    // 1. Source tree layout (dev).
    let exe = std::env::current_exe().ok();
    let src_candidate = exe
        .as_ref()
        .and_then(|e| e.parent())
        .and_then(|p| p.parent())
        .map(|p| p.join("resources").join("jaeger.yml"));
    if let Some(ref p) = src_candidate {
        if jaeger_resource_pair_exists(p) {
            return Ok(p.clone());
        }
    }

    // 2. Tauri resources layout (binary next to resources/jaeger.yml).
    let resources_candidate = exe
        .as_ref()
        .and_then(|e| e.parent())
        .map(|p| p.join("resources").join("jaeger.yml"));
    if let Some(ref p) = resources_candidate {
        if jaeger_resource_pair_exists(p) {
            return Ok(p.clone());
        }
    }

    // 3. Fallback: write the embedded Compose and Jaeger config files next to
    //    each other. The Compose bind mount is relative to this directory.
    let dir = crate::config::local_config::config_home();
    std::fs::create_dir_all(&dir)?;
    let fallback = dir.join("jaeger.yml");
    std::fs::write(&fallback, include_str!("../../resources/jaeger.yml"))?;
    std::fs::write(
        dir.join(JAEGER_CONFIG_FILE),
        include_str!("../../resources/jaeger-config.yml"),
    )?;
    Ok(fallback)
}

fn jaeger_resource_pair_exists(compose_path: &std::path::Path) -> bool {
    compose_path.exists()
        && compose_path
            .parent()
            .map(|parent| parent.join(JAEGER_CONFIG_FILE).exists())
            .unwrap_or(false)
}

fn docker_command() -> Command {
    // Prefer `docker compose` (v2 plugin), fall back to `docker-compose`.
    let mut cmd = Command::new("docker");
    cmd.arg("compose");
    cmd
}

fn compose_args(path: &PathBuf) -> Vec<String> {
    vec![
        "--file".to_string(),
        path.display().to_string(),
        "--project-name".to_string(),
        "reincore-telemetry".to_string(),
    ]
}

/// Check whether the Docker daemon is reachable on this OS. On Windows this
/// specifically tests the Docker Desktop named pipe because `docker --version`
/// can succeed even when the engine is off.
fn docker_daemon_reachable() -> bool {
    #[cfg(target_os = "windows")]
    {
        // On Windows `docker info` can return success even when the Docker
        // Desktop WSL2 named pipe is not actually accepting compose traffic.
        // `docker ps` exercises the same Containers API that `docker compose`
        // uses, so it is a more reliable pre-flight check.
        Command::new("docker")
            .args(["ps", "--format", "{{.Names}}"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[cfg(not(target_os = "windows"))]
    {
        docker_available()
    }
}

fn docker_error_message() -> String {
    #[cfg(target_os = "windows")]
    {
        "Docker is installed but the daemon is not reachable. Make sure Docker Desktop is running (WSL2 backend recommended).".to_string()
    }

    #[cfg(target_os = "macos")]
    {
        "Docker is installed but the daemon is not reachable. Make sure Docker Desktop is running."
            .to_string()
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        "Docker is installed but the daemon is not reachable. Make sure the Docker service is running (e.g. `sudo systemctl start docker`).".to_string()
    }
}

/// Public helper for the desktop app: start the bundled Jaeger container.
pub async fn start_jaeger() -> Result<()> {
    let path = jaeger_compose_path()?;
    if !docker_available() {
        return Err(anyhow!(
            "Docker is not installed or not in PATH. Install Docker, then run `reincore optl start`."
        ));
    }
    if !docker_daemon_reachable() {
        return Err(anyhow!("{}", docker_error_message()));
    }

    println!(
        "{} Starting bundled Jaeger telemetry stack...",
        "▶".green().bold()
    );

    let mut cmd = docker_command();
    cmd.args(compose_args(&path)).args(["up", "-d", "--wait"]);
    let status = cmd
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("failed to run `docker compose up`")?;

    if !status.success() {
        return Err(anyhow!(
            "`docker compose up` failed (exit {:?})",
            status.code()
        ));
    }

    println!(
        "\n{} Jaeger is running. Traces UI: {}\n",
        "✓".green().bold(),
        JAEGER_UI_URL.cyan().underline()
    );
    println!(
        "   Export traces automatically with: {}\n",
        "reincore optl enable".yellow()
    );
    println!(
        "   Or manually restart the daemon with:\n   {}\n",
        format!(
            "OTEL_EXPORTER_OTLP_ENDPOINT={} OTEL_SERVICE_NAME=reincore-local reincore start",
            JAEGER_OTLP_ENDPOINT
        )
        .yellow()
    );
    Ok(())
}

async fn stop_jaeger() -> Result<()> {
    let path = jaeger_compose_path()?;
    if !docker_available() {
        return Err(anyhow!("Docker is not installed or not in PATH."));
    }
    if !docker_daemon_reachable() {
        return Err(anyhow!("{}", docker_error_message()));
    }

    println!(
        "{} Stopping bundled Jaeger telemetry stack...",
        "▶".cyan().bold()
    );

    let mut cmd = docker_command();
    cmd.args(compose_args(&path)).args(["down"]);
    let status = cmd
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("failed to run `docker compose down`")?;

    if !status.success() {
        return Err(anyhow!(
            "`docker compose down` failed (exit {:?})",
            status.code()
        ));
    }

    println!("{} Jaeger stopped.", "✓".green().bold());
    Ok(())
}

async fn show_jaeger_status() -> Result<()> {
    if !docker_available() {
        println!("{} Docker is not available.", "✗".red().bold());
        return Ok(());
    }
    if !docker_daemon_reachable() {
        println!(
            "{} {} Start it with: {}",
            "✗".red().bold(),
            docker_error_message(),
            "reincore optl start".cyan()
        );
        return Ok(());
    }

    let running = is_container_running(JAEGER_CONTAINER).await?;
    let ui_reachable = if running { ui_reachable().await } else { false };

    if running && ui_reachable {
        println!(
            "{} Jaeger is running. UI: {} | OTLP gRPC: {}",
            "✓".green().bold(),
            JAEGER_UI_URL.cyan().underline(),
            JAEGER_OTLP_ENDPOINT.cyan()
        );
    } else if running {
        println!(
            "{} Jaeger container is running but the UI at {} is not reachable yet\n   (it may still be starting, crash-looping, or unhealthy). Check: {}",
            "○".yellow().bold(),
            JAEGER_UI_URL.cyan().underline(),
            "docker logs reincore-jaeger".yellow()
        );
    } else {
        println!(
            "{} Jaeger is not running. Start it with: {}",
            "○".dimmed(),
            "reincore optl start".cyan()
        );
    }
    Ok(())
}

/// Probe the Jaeger UI over HTTP; returns true iff port 16686 serves an
/// HTTP response (any status code counts as reachable).
async fn ui_reachable() -> bool {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .ok();
    match client {
        Some(c) => c.get(JAEGER_UI_URL).send().await.is_ok(),
        None => false,
    }
}

/// Public helper for the desktop app: open the Jaeger UI in the default
/// browser, returning an error if Jaeger is not running.
pub async fn open_jaeger_ui() -> Result<()> {
    let running = is_jaeger_running().await?;
    if !running {
        return Err(anyhow!(
            "Jaeger is not running. Start it first with `reincore optl start`."
        ));
    }

    println!(
        "{} Opening {} ...",
        "▶".green().bold(),
        JAEGER_UI_URL.cyan()
    );
    open::that(JAEGER_UI_URL)
        .with_context(|| format!("failed to open browser at {}", JAEGER_UI_URL))?;
    Ok(())
}

fn docker_available() -> bool {
    Command::new("docker")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

async fn is_container_running(name: &str) -> Result<bool> {
    let output = Command::new("docker")
        .args([
            "ps",
            "--filter",
            &format!("name={}", name),
            "--format",
            "{{.Names}}\t{{.State}}",
        ])
        .output()
        .context("failed to run `docker ps`")?;

    if !output.status.success() {
        return Ok(false);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.lines().any(|line| {
        let mut parts = line.split('\t');
        let matches_name = parts.next().map(|n| n.trim() == name).unwrap_or(false);
        // `docker ps` includes restarting containers; report only truly
        // running ones so `status` doesn't claim the UI is up during a
        // crash loop.
        let is_running = parts.next().map(|s| s.trim() == "running").unwrap_or(false);
        matches_name && is_running
    }))
}

/// Public helper for the desktop app: check whether the bundled Jaeger
/// container is currently running.
pub async fn is_jaeger_running() -> Result<bool> {
    if !docker_available() {
        return Ok(false);
    }
    is_container_running(JAEGER_CONTAINER).await
}

#[cfg(test)]
mod tests {
    #[test]
    fn bundled_compose_mounts_a_persistent_named_volume() {
        let compose = include_str!("../../resources/jaeger.yml");
        assert!(compose.contains("jaegertracing/jaeger:2.20.0"));
        assert!(compose.contains("reincore-jaeger-data:/badger"));
        assert!(compose.contains("name: reincore-telemetry_reincore-jaeger-data"));
        assert!(compose.contains("./jaeger-config.yml:/etc/jaeger/config.yml:ro"));
        assert!(compose.contains("JAEGER_TRACE_TTL: ${JAEGER_TRACE_TTL:-720h}"));
    }

    #[test]
    fn bundled_jaeger_config_uses_non_ephemeral_badger_storage() {
        let config = include_str!("../../resources/jaeger-config.yml");
        assert!(config.contains("badger:"));
        assert!(config.contains("ephemeral: false"));
        assert!(config.contains("keys: /badger/keys"));
        assert!(config.contains("values: /badger/values"));
    }

    #[test]
    fn bundled_compose_fixes_badger_volume_permissions_before_start() {
        let compose = include_str!("../../resources/jaeger.yml");
        assert!(compose.contains("jaeger-init:"));
        assert!(compose.contains("busybox"));
        assert!(compose.contains("chown -R 10001:10001 /badger"));
        assert!(compose.contains("service_completed_successfully"));
    }
}
