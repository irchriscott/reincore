//! OpenTelemetry health check.
//!
//! When a user enables OTEL in config, the engine used to try to export traces
//! immediately. If the OTLP backend (e.g. the bundled Jaeger container) was not
//! running, every span export produced noise like:
//!
//!   OpenTelemetry trace error occurred. Exporter otlp encountered the
//!   following error(s): the grpc server returns error (The service is
//!   currently unavailable): , detailed error message: tcp connect error
//!
//! This module performs a lightweight pre-flight health check before the
//! exporter is installed. If the backend is unavailable, telemetry falls back
//! to plain console tracing without registering the OTLP pipeline at all.

use std::process::{Command, Stdio};
use std::time::Duration;

/// Result of an OpenTelemetry pre-flight health check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenTelemetryHealth {
    /// OTEL is disabled in configuration; no health check is needed.
    Disabled,
    /// The OTLP endpoint appears reachable and the backend container is running
    /// when bundled Jaeger is expected.
    Healthy,
    /// The OTLP endpoint is not reachable or the bundled backend is not running.
    Unhealthy {
        /// Human-readable explanation.
        reason: String,
        /// Whether the likely fix is starting the bundled Jaeger container.
        #[allow(dead_code)]
        needs_docker_check: bool,
    },
}

impl OpenTelemetryHealth {
    /// Return true if the OTLP exporter can be safely started.
    pub fn is_healthy(&self) -> bool {
        matches!(self, OpenTelemetryHealth::Healthy)
    }

    /// Return a user-facing message describing the health result.
    pub fn message(&self) -> String {
        match self {
            OpenTelemetryHealth::Disabled => "OpenTelemetry is disabled".to_string(),
            OpenTelemetryHealth::Healthy => {
                "OpenTelemetry backend is reachable; trace export enabled".to_string()
            }
            OpenTelemetryHealth::Unhealthy { reason, .. } => reason.clone(),
        }
    }
}

/// Configuration used by the health check.
#[derive(Debug, Clone)]
pub struct OpenTelemetryHealthConfig {
    /// The OTLP endpoint URL (e.g. `http://localhost:4317`).
    pub endpoint: String,
    /// Whether the endpoint is the bundled Jaeger default.
    pub is_bundled_jaeger: bool,
}

/// Check whether OpenTelemetry export should be allowed.
///
/// When the endpoint points at the bundled Jaeger container, this also checks
/// that Docker is available and the `reincore-jaeger` container is running.
/// For any other endpoint, it simply attempts a TCP connection to the host/port.
pub async fn check(config: OpenTelemetryHealthConfig) -> OpenTelemetryHealth {
    if config.endpoint.is_empty() {
        return OpenTelemetryHealth::Disabled;
    }

    // For the bundled Jaeger endpoint we can do a stronger check: make sure the
    // Docker daemon is up and the container is running. This gives the user a
    // much clearer error than a generic gRPC "service unavailable" log line.
    if config.is_bundled_jaeger {
        if !docker_available() {
            return OpenTelemetryHealth::Unhealthy {
                reason: "OpenTelemetry is enabled but Docker is not installed or not in PATH. \
                    Install Docker and run `reincore optl start`."
                    .to_string(),
                needs_docker_check: true,
            };
        }
        if !docker_daemon_reachable() {
            return OpenTelemetryHealth::Unhealthy {
                reason: docker_error_message(),
                needs_docker_check: true,
            };
        }
        match is_container_running("reincore-jaeger").await {
            Ok(true) => {}
            Ok(false) => {
                return OpenTelemetryHealth::Unhealthy {
                    reason:
                        "OpenTelemetry is enabled but the bundled Jaeger container is not running. \
                        Start it with `reincore optl start`."
                            .to_string(),
                    needs_docker_check: true,
                };
            }
            Err(err) => {
                return OpenTelemetryHealth::Unhealthy {
                    reason: format!(
                        "OpenTelemetry is enabled but failed to check Jaeger container status: {}. \
                            Start it with `reincore optl start`.",
                        err
                    ),
                    needs_docker_check: true,
                };
            }
        }
    }

    // General endpoint reachability test. gRPC OTLP typically uses port 4317.
    // We attempt a TCP connect with a short timeout so startup does not hang.
    match resolve_host_and_port(&config.endpoint) {
        Some((host, port)) => {
            let timeout = Duration::from_secs(2);
            match tokio::time::timeout(timeout, tokio::net::TcpStream::connect((host.as_str(), port))).await {
                Ok(Ok(_)) => OpenTelemetryHealth::Healthy,
                Ok(Err(err)) => OpenTelemetryHealth::Unhealthy {
                    reason: format!(
                        "OpenTelemetry endpoint {} is not reachable ({err}); trace export disabled",
                        config.endpoint
                    ),
                    needs_docker_check: config.is_bundled_jaeger,
                },
                Err(_) => OpenTelemetryHealth::Unhealthy {
                    reason: format!(
                        "OpenTelemetry endpoint {} did not respond within {:?}; trace export disabled",
                        config.endpoint, timeout
                    ),
                    needs_docker_check: config.is_bundled_jaeger,
                },
            }
        }
        None => OpenTelemetryHealth::Unhealthy {
            reason: format!(
                "OpenTelemetry endpoint {} could not be parsed; trace export disabled",
                config.endpoint
            ),
            needs_docker_check: false,
        },
    }
}

/// Parse a URL like `http://localhost:4317` into `("localhost", 4317)`.
fn resolve_host_and_port(endpoint: &str) -> Option<(String, u16)> {
    let url = url::Url::parse(endpoint).ok()?;
    let host = url.host_str()?.to_string();
    let port = url.port_or_known_default()?;
    Some((host, port))
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

/// Check whether the Docker daemon is reachable on this OS. On Windows this
/// specifically tests the Containers API because `docker --version` can
/// succeed even when the engine is off.
fn docker_daemon_reachable() -> bool {
    #[cfg(target_os = "windows")]
    {
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
        "OpenTelemetry is enabled but Docker is installed and not reachable. \
            Make sure Docker Desktop is running (WSL2 backend recommended)."
            .to_string()
    }

    #[cfg(target_os = "macos")]
    {
        "OpenTelemetry is enabled but Docker is installed and not reachable. \
            Make sure Docker Desktop is running."
            .to_string()
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        "OpenTelemetry is enabled but Docker is installed and not reachable. \
            Make sure the Docker service is running (e.g. `sudo systemctl start docker`)."
            .to_string()
    }
}

async fn is_container_running(name: &str) -> anyhow::Result<bool> {
    let output = Command::new("docker")
        .args(["ps", "--format", "{{.Names}}"])
        .output()
        .map_err(|e| anyhow::anyhow!("failed to run `docker ps`: {e}"))?;

    if !output.status.success() {
        return Err(anyhow::anyhow!(
            "`docker ps` exited with status {:?}",
            output.status.code()
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.lines().any(|line| line.trim() == name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_known_http_endpoint() {
        let parsed = resolve_host_and_port("http://localhost:4317").unwrap();
        assert_eq!(parsed, ("localhost".to_string(), 4317));
    }

    #[test]
    fn parse_known_https_endpoint() {
        let parsed = resolve_host_and_port("https://collector.example.com:443").unwrap();
        assert_eq!(parsed, ("collector.example.com".to_string(), 443));
    }

    #[test]
    fn parse_missing_scheme_fails() {
        assert!(resolve_host_and_port("localhost:4317").is_none());
    }
}
