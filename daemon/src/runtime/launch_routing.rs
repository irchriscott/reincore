//! Shared launch-routing logic used by both the CLI (`reincore launch`) and
//! the desktop app's `launch_app` Tauri command.
//!
//! Launching an app with a single provider has to negotiate with whatever
//! daemon state is already running on the machine. The daemon is either:
//!
//! - **Not running** - nothing to negotiate: start it in single-provider
//!   mode for the requested provider.
//! - **Running in multi-provider mode** - the daemon can already serve the
//!   requested provider (if it is configured) via the `X-ReinCore-Provider`
//!   header. Reuse the daemon and route the app's requests through it,
//!   instead of tearing it down.
//! - **Running in single-provider mode** - if the running provider matches
//!   the requested one, reuse it. Otherwise the user must decide between
//!   switching the daemon to the new provider (stop + restart, which kills
//!   any in-flight requests on the old provider) or keeping the running
//!   provider for this launch.

use crate::cli::DaemonConfig;
use anyhow::Result;
use colored::Colorize;
use engine_core::log;
use serde::{Deserialize, Serialize};

/// Normalized daemon state as reported by `GET /health`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DaemonState {
    /// `"multi_provider"` or `"single_provider"`.
    pub mode: String,
    /// The single provider's label (single-provider mode only).
    pub provider: Option<String>,
    /// The wire API mode: `"agentic"` or `"chat"` (single-provider mode only).
    pub api_mode: Option<String>,
    /// All registered provider labels (multi-provider mode), or the single
    /// provider repeated.
    pub providers: Vec<String>,
    /// Default project path of the running daemon (fallback for sessions
    /// that carry no project). Absent on older daemons.
    pub default_project_path: Option<String>,
}

impl DaemonState {
    pub fn is_single_provider(&self) -> bool {
        self.mode.eq_ignore_ascii_case("single_provider")
    }

    pub fn is_multi_provider(&self) -> bool {
        self.mode.eq_ignore_ascii_case("multi_provider")
    }

    /// Whether `label` is one of the configured providers of this daemon.
    pub fn has_provider(&self, label: &str) -> bool {
        self.providers.iter().any(|p| p.eq_ignore_ascii_case(label))
    }
}

/// Fetch `GET /health` from the daemon. `Ok(None)` means the daemon is not
/// reachable (not running, or running but wedged).
pub async fn probe_daemon_state(host: &str, port: u16) -> Result<Option<DaemonState>> {
    let url = format!("http://{host}:{port}/health");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();
    let resp = match client.get(&url).send().await {
        Ok(resp) => resp,
        Err(_) => return Ok(None),
    };
    if !resp.status().is_success() {
        return Ok(None);
    }
    let body: serde_json::Value = match resp.json().await {
        Ok(body) => body,
        Err(_) => return Ok(None),
    };

    let mode = body
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    if mode.is_empty() {
        return Ok(None);
    }

    let providers: Vec<String> = body
        .get("providers")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|p| p.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    let provider = if mode.eq_ignore_ascii_case("single_provider") {
        providers.first().cloned()
    } else {
        None
    };

    let api_mode = body
        .get("api_mode")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let default_project_path = body
        .get("default_project_path")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    Ok(Some(DaemonState {
        mode,
        provider,
        api_mode,
        providers,
        default_project_path,
    }))
}

/// The outcome of the launch negotiation with a running daemon.
#[derive(Clone, Debug, PartialEq)]
pub enum LaunchPlan {
    /// No daemon is running - start one in single-provider mode for the
    /// requested provider (the classic launch path). Any stale daemon on
    /// the port is stopped by the caller's start path.
    StartSingleProvider { provider: String },
    /// The daemon is already running in multi-provider mode and the requested
    /// provider is configured on it. Reuse the daemon as-is and make the app
    /// send `X-ReinCore-Provider: <provider>` on every request.
    ReuseMultiProvider { provider: String },
    /// The daemon is running in single-provider mode with the same provider.
    /// Reuse it as-is.
    ReuseSingleProvider { provider: String },
    /// The daemon is running in single-provider mode with a *different*
    /// provider. The caller must ask the user how to proceed:
    /// - [`SwitchDecision::UseRequested`] - stop the daemon and restart it
    ///   in single-provider mode for the requested provider.
    /// - [`SwitchDecision::UseCurrent`] - keep the running provider and
    ///   route the app to it instead of the requested one.
    /// - [`SwitchDecision::Cancel`] - abort the launch.
    NeedsSwitchDecision { requested: String, current: String },
}

/// How the user resolved a [`LaunchPlan::NeedsSwitchDecision`] conflict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwitchDecision {
    /// Stop the daemon and restart it with the requested provider.
    UseRequested,
    /// Keep the running provider and use it for this launch.
    UseCurrent,
    /// Abort the launch.
    Cancel,
}

/// The user-visible choice after [`resolve_launch_plan`] needs input.
#[derive(Clone, Debug, PartialEq)]
pub struct LaunchRouting {
    /// The provider the app will actually use (may differ from the request
    /// when the user chose to keep the running provider).
    pub provider: String,
    /// The provider label the app must send in `X-ReinCore-Provider` on
    /// every request. `None` when the daemon is single-provider for
    /// `provider` (the header is not needed and not checked).
    pub provider_header: Option<String>,
    /// Whether the daemon must be (re)started in single-provider mode for
    /// `provider` before the app can be launched. `false` means the running
    /// daemon is already usable as-is.
    pub start_daemon: bool,
    /// Stop the currently running daemon first (only set together with
    /// `start_daemon = true`).
    pub stop_first: bool,
}

/// Normalize a mode label to the canonical wire-API vocabulary used by the
/// daemon's `api_mode` field: "agentic" (a.k.a. "agent"/"responses") or
/// "chat". `None` means "unspecified - accept whatever is running".
fn normalize_mode(mode: Option<&str>) -> Option<String> {
    mode.map(|m| match m.to_lowercase().as_str() {
        "agent" | "responses" | "agentic" => "agentic".to_string(),
        other => other.to_string(),
    })
}

/// Decide what to do given the daemon state and the requested provider.
///
/// This is the negotiation core shared by the CLI and the desktop app; both
/// callers then map the [`LaunchPlan`] into their own UX (interactive
/// terminal prompt vs. GUI dialog).
///
/// `requested_mode` is the launch's wire-API mode ("agentic"/"chat"). It
/// only matters when a single-provider daemon is already running: the daemon
/// is only reusable when both the provider and the mode match. Multi-provider
/// daemons serve every request shape, so the mode is irrelevant there.
///
/// `app` is the app being launched. It gates multi-provider reuse: only apps
/// with a custom request-header mechanism (see
/// `launch::app_supports_provider_header`) can be routed through a
/// multi-provider daemon via `X-ReinCore-Provider`. For the others a running
/// multi-provider daemon is treated as a provider mismatch, so the user gets
/// the switch decision (stop + restart in single-provider mode).
pub async fn resolve_launch_plan(
    config: &DaemonConfig,
    requested_provider: &str,
    requested_mode: Option<&str>,
    app: Option<&str>,
) -> Result<LaunchPlan> {
    let requested = requested_provider.to_lowercase();

    let Some(state) = probe_daemon_state(config.host.as_str(), config.port).await? else {
        return Ok(LaunchPlan::StartSingleProvider {
            provider: requested,
        });
    };

    if state.is_multi_provider() {
        // Apps that cannot send custom request headers can't be routed
        // through a multi-provider daemon at all.
        let header_capable = app
            .map(crate::launch::app_supports_provider_header)
            .unwrap_or(true);
        if header_capable && state.has_provider(&requested) {
            Ok(LaunchPlan::ReuseMultiProvider {
                provider: requested,
            })
        } else if !header_capable {
            // The daemon must be restarted in single-provider mode for this
            // app; that is a destructive switch, so ask the user first.
            Ok(LaunchPlan::NeedsSwitchDecision {
                requested,
                current: "multi_provider".to_string(),
            })
        } else {
            Err(anyhow::anyhow!(
                "ReinCore is running in multi-provider mode, but provider '{}' is not configured on it. Add the provider (Settings -> Providers) or stop the daemon with 'reincore stop' and retry.",
                requested
            ))
        }
    } else if state.is_single_provider() {
        let current = state.provider.unwrap_or_default();
        let mode_matches = match (normalize_mode(requested_mode), state.api_mode.as_deref()) {
            (Some(wanted), Some(running)) => {
                normalize_mode(Some(running)).as_deref() == Some(wanted.as_str())
            }
            // Missing mode info on either side - don't block the reuse on it;
            // the daemon knows better than we do.
            _ => true,
        };
        if current.eq_ignore_ascii_case(&requested) && mode_matches {
            Ok(LaunchPlan::ReuseSingleProvider {
                provider: current.to_lowercase(),
            })
        } else {
            Ok(LaunchPlan::NeedsSwitchDecision { requested, current })
        }
    } else {
        // Unknown mode string - treat the daemon as unusable and fall back
        // to the classic "start our own" behavior. The caller's start path
        // will stop the stale daemon first.
        Ok(LaunchPlan::StartSingleProvider {
            provider: requested,
        })
    }
}

/// Map a [`LaunchPlan`] plus (for switch conflicts) the user's decision
/// into concrete routing instructions for the launch path.
pub fn plan_to_routing(
    plan: &LaunchPlan,
    decision: Option<SwitchDecision>,
) -> Result<LaunchRouting> {
    match plan {
        LaunchPlan::StartSingleProvider { provider } => Ok(LaunchRouting {
            provider: provider.clone(),
            provider_header: None,
            start_daemon: true,
            stop_first: true,
        }),
        LaunchPlan::ReuseMultiProvider { provider } => Ok(LaunchRouting {
            provider: provider.clone(),
            provider_header: Some(provider.clone()),
            start_daemon: false,
            stop_first: false,
        }),
        LaunchPlan::ReuseSingleProvider { provider } => Ok(LaunchRouting {
            provider: provider.clone(),
            provider_header: None,
            start_daemon: false,
            stop_first: false,
        }),
        LaunchPlan::NeedsSwitchDecision { requested, current } => match decision {
            Some(SwitchDecision::UseRequested) => Ok(LaunchRouting {
                provider: requested.clone(),
                provider_header: None,
                start_daemon: true,
                stop_first: true,
            }),
            Some(SwitchDecision::UseCurrent) => Ok(LaunchRouting {
                provider: current.clone(),
                provider_header: None,
                start_daemon: false,
                stop_first: false,
            }),
            Some(SwitchDecision::Cancel) => Err(anyhow::anyhow!("launch cancelled")),
            None => Err(anyhow::anyhow!(
                "launch requires a provider-switch decision"
            )),
        },
    }
}

/// Interactive prompt for the CLI: the daemon is running on `current` in
/// single-provider mode but the user asked for `requested`. Print the
/// consequences of switching and read one of the three choices.
///
/// `current == "multi_provider"` is the app-can't-send-headers case: the
/// daemon runs in multi-provider mode but the app needs a single-provider
/// daemon, so only "switch" and "cancel" are offered.
pub async fn prompt_switch_decision(requested: &str, current: &str) -> Result<SwitchDecision> {
    let current_display = if current == "multi_provider" {
        "multi-provider mode".to_string()
    } else {
        current.to_string()
    };
    println!();
    println!(
        "{}",
        log::warning(&format!(
            "ReinCore is running in {}, but '{}' needs a single-provider daemon for '{}'.",
            current_display, requested, requested
        ))
    );
    if current == "multi_provider" {
        println!(
            "{}",
            "Switching stops the running daemon (in-flight requests are aborted) and restarts it in single-provider mode.".dimmed()
        );
        println!();
        println!(
            "  1. Switch to a single-provider daemon for '{}' (restart the daemon)",
            requested
        );
        println!("  2. Cancel");
        print!("\nChoose an option [1/2]: ");
    } else {
        println!(
            "{}",
            format!(
                "Switching providers stops the running daemon (in-flight requests on '{}' are aborted) and restarts it with '{}'.",
                current, requested
            )
            .dimmed()
        );
        println!();
        println!("  1. Switch to '{}' (restart the daemon)", requested);
        println!("  2. Keep using '{}' for this launch", current);
        println!("  3. Cancel");
        print!("\nChoose an option [1/2/3]: ");
    }
    use std::io::Write;
    std::io::stdout().flush().ok();

    loop {
        if current != "multi_provider" {
            let choice = tokio::task::spawn_blocking(|| -> Result<String> {
                let mut input = String::new();
                std::io::stdin()
                    .read_line(&mut input)
                    .map_err(|e| anyhow::anyhow!("failed to read input: {}", e))?;
                Ok(input.trim().to_string())
            })
            .await??;

            match choice.as_str() {
                "1" => return Ok(SwitchDecision::UseRequested),
                "2" => return Ok(SwitchDecision::UseCurrent),
                "3" | "" => return Ok(SwitchDecision::Cancel),
                _ => println!("{}", log::warning("Enter 1, 2 or 3.")),
            }
        } else {
            let choice = tokio::task::spawn_blocking(|| -> Result<String> {
                let mut input = String::new();
                std::io::stdin()
                    .read_line(&mut input)
                    .map_err(|e| anyhow::anyhow!("failed to read input: {}", e))?;
                Ok(input.trim().to_string())
            })
            .await??;

            match choice.as_str() {
                "1" => return Ok(SwitchDecision::UseRequested),
                "2" | "" => return Ok(SwitchDecision::Cancel),
                _ => println!("{}", log::warning("Enter 1 or 2.")),
            }
        }
    }
}

/// High-level helper used by the CLI: negotiate with the running daemon and
/// ask the user interactively when a provider switch is required. Returns
/// the concrete routing the launch path should apply.
pub async fn resolve_launch_routing_interactive(
    config: &DaemonConfig,
    requested_provider: &str,
    requested_mode: Option<&str>,
    app: Option<&str>,
) -> Result<LaunchRouting> {
    let plan = resolve_launch_plan(config, requested_provider, requested_mode, app).await?;
    let decision = match &plan {
        LaunchPlan::NeedsSwitchDecision { requested, current } => {
            Some(prompt_switch_decision(requested, current).await?)
        }
        _ => None,
    };
    let routing = plan_to_routing(&plan, decision)?;

    match &plan {
        LaunchPlan::StartSingleProvider { .. } => {
            println!(
                "{}",
                log::info("No ReinCore daemon is running - starting one in single-provider mode.")
            );
        }
        LaunchPlan::ReuseMultiProvider { provider } => {
            println!(
                "{}",
                log::success(&format!(
                    "Reusing the running multi-provider daemon; routing requests to provider '{}' via the X-ReinCore-Provider header.",
                    provider
                ))
            );
        }
        LaunchPlan::ReuseSingleProvider { provider } => {
            println!(
                "{}",
                log::success(&format!(
                    "Reusing the running single-provider daemon for provider '{}'.",
                    provider
                ))
            );
        }
        LaunchPlan::NeedsSwitchDecision { .. } => {
            if decision == Some(SwitchDecision::UseRequested) {
                println!(
                    "{}",
                    log::info(
                        "Stopping the daemon and restarting it with the requested provider ..."
                    )
                );
            }
        }
    }

    Ok(routing)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daemon_state_classifies_mode() {
        let multi = DaemonState {
            mode: "multi_provider".to_string(),
            provider: None,
            api_mode: None,
            providers: vec!["openai".to_string(), "anthropic".to_string()],
            default_project_path: None,
        };
        assert!(multi.is_multi_provider());
        assert!(!multi.is_single_provider());
        assert!(multi.has_provider("OpenAI"));
        assert!(!multi.has_provider("groq"));

        let single = DaemonState {
            mode: "single_provider".to_string(),
            provider: Some("ollama".to_string()),
            api_mode: Some("agentic".to_string()),
            providers: vec!["ollama".to_string()],
            default_project_path: None,
        };
        assert!(single.is_single_provider());
        assert!(single.has_provider("OLLAMA"));
    }

    #[test]
    fn start_plan_maps_to_start_routing() {
        let plan = LaunchPlan::StartSingleProvider {
            provider: "openai".to_string(),
        };
        let routing = plan_to_routing(&plan, None).unwrap();
        assert_eq!(routing.provider, "openai");
        assert_eq!(routing.provider_header, None);
        assert!(routing.start_daemon);
        assert!(routing.stop_first);
    }

    #[test]
    fn multi_provider_plan_bakes_in_header() {
        let plan = LaunchPlan::ReuseMultiProvider {
            provider: "anthropic".to_string(),
        };
        let routing = plan_to_routing(&plan, None).unwrap();
        assert_eq!(routing.provider, "anthropic");
        assert_eq!(routing.provider_header.as_deref(), Some("anthropic"));
        assert!(!routing.start_daemon);
        assert!(!routing.stop_first);
    }

    #[test]
    fn matching_single_provider_plan_reuses_without_header() {
        let plan = LaunchPlan::ReuseSingleProvider {
            provider: "ollama".to_string(),
        };
        let routing = plan_to_routing(&plan, None).unwrap();
        assert_eq!(routing.provider, "ollama");
        assert_eq!(routing.provider_header, None);
        assert!(!routing.start_daemon);
        assert!(!routing.stop_first);
    }

    #[test]
    fn switch_decision_use_requested_restarts() {
        let plan = LaunchPlan::NeedsSwitchDecision {
            requested: "groq".to_string(),
            current: "openai".to_string(),
        };
        let routing = plan_to_routing(&plan, Some(SwitchDecision::UseRequested)).unwrap();
        assert_eq!(routing.provider, "groq");
        assert_eq!(routing.provider_header, None);
        assert!(routing.start_daemon);
        assert!(routing.stop_first);
    }

    #[test]
    fn switch_decision_use_current_keeps_daemon() {
        let plan = LaunchPlan::NeedsSwitchDecision {
            requested: "groq".to_string(),
            current: "openai".to_string(),
        };
        let routing = plan_to_routing(&plan, Some(SwitchDecision::UseCurrent)).unwrap();
        assert_eq!(routing.provider, "openai");
        assert_eq!(routing.provider_header, None);
        assert!(!routing.start_daemon);
        assert!(!routing.stop_first);
    }

    #[test]
    fn switch_decision_cancel_errors() {
        let plan = LaunchPlan::NeedsSwitchDecision {
            requested: "groq".to_string(),
            current: "openai".to_string(),
        };
        assert!(plan_to_routing(&plan, Some(SwitchDecision::Cancel)).is_err());
        // No decision supplied is equally fatal for the caller.
        assert!(plan_to_routing(&plan, None).is_err());
    }

    #[test]
    fn mode_labels_normalize_to_daemon_vocabulary() {
        assert_eq!(normalize_mode(Some("agent")), Some("agentic".to_string()));
        assert_eq!(
            normalize_mode(Some("responses")),
            Some("agentic".to_string())
        );
        assert_eq!(normalize_mode(Some("Agentic")), Some("agentic".to_string()));
        assert_eq!(normalize_mode(Some("chat")), Some("chat".to_string()));
        assert_eq!(normalize_mode(None), None);
    }
}
