use anyhow::Result;
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Default TCP port for the local daemon and every proxy it spawns.
pub const DEFAULT_PORT: u16 = 8989;

/// Default path (with `~` unexpanded) of the local SQLite database.
pub const DEFAULT_DB_PATH: &str = "~/.reincore-io/reincore.db";

/// Path to the small runtime file that records the active daemon's host/port.
pub fn daemon_config_path() -> PathBuf {
    crate::local_config::config_home().join("daemon.json")
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DaemonConfig {
    pub host: String,
    pub port: u16,
    /// Process ID of the currently running daemon, if known.
    pub pid: Option<u32>,
}

impl DaemonConfig {
    pub fn load() -> Self {
        std::fs::read_to_string(daemon_config_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| Self {
                host: "127.0.0.1".to_string(),
                port: DEFAULT_PORT,
                pid: None,
            })
    }

    pub fn save(&self) -> Result<()> {
        let path = daemon_config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(name = "reincore")]
#[command(about = "ReinCore daemon and CLI")]
#[command(alias = "rc")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Configure local MCP. Changes apply after restarting the daemon.
    Mcp {
        #[command(subcommand)]
        command: McpCommand,
    },
    /// Start the local ReinCore daemon.
    ///
    /// By default starts in multi-provider mode (all configured providers
    /// available). Pass `--provider` to restrict to a single provider - this
    /// is the replacement for the old `provider start-proxy` subcommand and
    /// supports the same options as `launch` (`--mode`, `--cloud`, `--model`,
    /// etc.) except for app-launching flags.
    Start {
        /// Restrict the daemon to a single provider (e.g. `openai`, `ollama`).
        /// When omitted, all configured providers are available.
        #[arg(long)]
        provider: Option<String>,

        /// API mode: "agentic" (Responses API, default) or "chat" (Chat Completions).
        /// Only meaningful with `--provider`. Agentic mode uses /v1/responses;
        /// chat mode uses /v1/chat/completions.
        #[arg(long, default_value = "agentic")]
        mode: String,

        /// Host address to bind to.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to listen on.
        #[arg(short, long, default_value_t = DEFAULT_PORT)]
        port: u16,

        /// Path to the local SQLite database.
        #[arg(short, long, default_value = "~/.reincore-io/reincore.db")]
        db_path: String,

        /// Working directory. Only meaningful with `--provider` for providers
        /// that care about the project path (e.g. Ollama local mode uses it
        /// for nothing, but it's accepted for parity with `launch`).
        #[arg(long)]
        path: Option<String>,

        /// Enable verbose logging.
        #[arg(short, long)]
        verbose: bool,

        /// Use the cloud variant of a provider that supports both local and
        /// cloud deployment (currently only `ollama`). When set, the user is
        /// prompted for an API key and a default cloud model name instead of
        /// picking from locally installed models.
        #[arg(long, default_value_t = false)]
        cloud: bool,

        /// Model name for providers that take a user-specified model (e.g.
        /// `ollama --cloud`). If omitted or invalid, you'll be prompted to
        /// enter one interactively.
        #[arg(long)]
        model: Option<String>,
    },

    /// Show or set budget limits.
    Budget {
        /// Budget scope: session, project, day, or month.
        #[arg(short, long, default_value = "session")]
        scope: String,

        /// Scope key: session_id, project name, or leave empty for day/month.
        #[arg(short, long)]
        key: Option<String>,

        /// Set the budget limit in dollars (e.g. 5.00 for $5).
        #[arg(short, long)]
        limit: Option<String>,

        /// Show history of budget updates for this scope.
        #[arg(long)]
        history: bool,

        /// Show spend and remaining budget for this scope.
        #[arg(long)]
        usage: bool,

        /// Path to the local SQLite database.
        #[arg(short, long, default_value = "~/.reincore-io/reincore.db")]
        db_path: String,
    },

    /// Show daemon status (running or not).
    Status,

    /// Stop the running local daemon.
    Stop,

    /// Compact previous chat messages for a session.
    #[command(name = "compact", alias = "compress")]
    Compact {
        /// Session UUID to compact.
        #[arg(short, long)]
        session: String,

        /// Compaction mode: low, mid, or max. Omit to use the configured
        /// compaction mode (see `reincore config --compaction-mode`).
        #[arg(short, long)]
        mode: Option<String>,

        /// Print the retained messages as JSON.
        #[arg(long)]
        json: bool,

        /// Path to the local SQLite database.
        #[arg(short, long, default_value = "~/.reincore-io/reincore.db")]
        db_path: String,
    },

    /// Configure global context compaction settings.
    Config {
        /// Enable or disable automatic compaction. `false` clears the
        /// configured mode (compaction off); `true` enables it at the
        /// currently configured mode (or `low` if none is set).
        #[arg(long = "compaction", alias = "compression", value_name = "true|false")]
        compaction: Option<String>,

        /// Set the compaction level: low, mid, or max. Setting a mode also
        /// enables compaction.
        #[arg(
            long = "compaction-mode",
            alias = "compression-mode",
            value_name = "MODE"
        )]
        compaction_mode: Option<String>,

        /// Path to the local SQLite database.
        #[arg(short, long, default_value = "~/.reincore-io/reincore.db")]
        db_path: String,
    },

    /// Pause a session and return a resumable checkpoint token.
    Pause {
        /// Session UUID to pause.
        #[arg(short, long)]
        session: String,

        /// Optional reason for the pause.
        #[arg(short, long, default_value = "user-pause")]
        reason: String,
    },

    /// Resume a session from a resumable checkpoint token.
    Resume {
        /// Resumable checkpoint token.
        #[arg(short, long)]
        token: String,
    },

    /// List sessions, optionally filtered by project path.
    Sessions {
        /// Project path to filter by. Use '.' for the current directory.
        #[arg(short, long)]
        path: Option<String>,

        /// Include checkpoint tokens in the output.
        #[arg(long)]
        checkpoints: bool,
    },

    /// Export a session's timeline (tasks, requests, events with sequence, tokens, spend) as JSON.
    Export {
        /// Session UUID to export.
        #[arg(short, long)]
        session: String,

        /// Write to this file instead of stdout.
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Inspect, test, or start a proxy for a specific provider.
    Provider {
        #[command(subcommand)]
        command: ProviderCommands,
    },

    /// Launch a coding/agent app and start its default single-provider proxy.
    Launch {
        /// App to launch: claude, codex, codex-app, open-code, openclaw,
        /// copilot, gemini, gemini-cli, cursor, windsurf, aider, continue,
        /// muse.
        app: String,

        /// Override the default provider for this app.
        #[arg(long)]
        provider: Option<String>,

        /// API mode: "agentic" (Responses API, default) or "chat" (Chat Completions).
        /// Agentic mode uses /v1/responses; chat mode uses /v1/chat/completions.
        #[arg(long, default_value = "agentic")]
        mode: String,

        /// Host address to bind to.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to listen on.
        #[arg(short, long, default_value_t = DEFAULT_PORT)]
        port: u16,

        /// Path to the local SQLite database.
        #[arg(short, long, default_value = "~/.reincore-io/reincore.db")]
        db_path: String,

        /// Working directory where the app should be launched.
        /// If the path doesn't exist, you'll be prompted to create it.
        #[arg(long)]
        path: Option<String>,

        /// Enable verbose logging.
        #[arg(short, long)]
        verbose: bool,

        /// Restore launched app profile
        #[arg(short, long)]
        restore: bool,

        /// Use the cloud variant of a provider that supports both local and
        /// cloud deployment (currently only `ollama`). When set, the user is
        /// prompted for an API key and a default cloud model name instead of
        /// picking from locally installed models.
        #[arg(long, default_value_t = false)]
        cloud: bool,

        /// Model name for providers that take a user-specified model (e.g.
        /// `ollama --cloud`). If omitted or invalid, you'll be prompted to
        /// enter one interactively.
        #[arg(long)]
        model: Option<String>,
    },

    /// Set up reincore with an app (write persistent config) without launching it.
    /// The app can then be launched manually from any directory.
    Setup {
        /// App to set up: claude, codex, codex-app, open-code, openclaw,
        /// copilot, gemini, gemini-cli, cursor, windsurf, aider, continue,
        /// muse.
        app: String,

        /// Override the default provider for this app.
        #[arg(long)]
        provider: Option<String>,

        /// API mode: "agentic" (Responses API, default) or "chat" (Chat Completions).
        #[arg(long, default_value = "agentic")]
        mode: String,

        /// Host address for the proxy (written into the app's config).
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port for the proxy (written into the app's config).
        #[arg(short, long, default_value_t = DEFAULT_PORT)]
        port: u16,
    },

    /// Explicit database backup, verification, restore and offline retention.
    Data {
        #[command(subcommand)]
        command: DataCommand,
    },

    /// Run database migrations and exit.
    Migrate {
        /// Path to the local SQLite database.
        #[arg(short, long, default_value = "~/.reincore-io/reincore.db")]
        db_path: String,
    },

    /// Restore an app's original config (revert reincore profile) without launching the proxy.
    Restore {
        /// App to restore: claude, codex, aider.
        app: String,
    },

    /// Manage the bundled Jaeger telemetry stack and OpenTelemetry export.
    ///
    /// `reincore otel start` launches the bundled Jaeger container and
    /// tells you the URL. `stop` tears it down. `status` checks whether the
    /// container is running. `open` launches the Jaeger UI in your browser.
    ///
    /// Once Jaeger is running, the daemon exports traces when restarted with
    /// `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317`. Use the desktop
    /// app's Settings page to toggle this automatically.
    #[command(name = "otel", alias = "telemetry")]
    Telemetry {
        #[command(subcommand)]
        command: TelemetryCommands,
    },

    /// Log out from a provider (delete stored credentials).
    Logout {
        /// Provider to log out from.
        provider: String,
    },

    /// Show or update the global routing policy in ~/.reincore-io/config.toml.
    /// Run with no flags to print the current policy.
    Routing {
        /// Routing mode: auto, custom, or locked.
        #[arg(long, value_name = "MODE")]
        mode: Option<String>,

        /// Provider strategy for deriving models (currently same_provider).
        #[arg(long, value_name = "STRATEGY")]
        provider_strategy: Option<String>,

        /// Minimum quality floor: economy, balanced, or high.
        #[arg(long, value_name = "QUALITY")]
        minimum_quality: Option<String>,

        /// Allow routing to a stronger tier than the standard model.
        #[arg(long, value_name = "true|false")]
        allow_upgrade: Option<String>,

        /// Allow budget pressure to degrade the selected tier.
        #[arg(long, value_name = "true|false")]
        allow_degradation: Option<String>,
    },

    /// Show or update a provider's model mappings and per-provider routing
    /// override in ~/.reincore-io/config.toml. Run with no model flags to print
    /// the current configuration for the provider.
    Model {
        /// Provider label, e.g. openai, anthropic, gemini, azure, groq, xai,
        /// deepseek, meta, copilot.
        provider: String,

        /// Set the standard (anchor) model.
        #[arg(long, value_name = "MODEL")]
        standard: Option<String>,

        /// Set the Lite tier model. Pass an empty string to clear it.
        #[arg(long, value_name = "MODEL")]
        lite: Option<String>,

        /// Set the Pro tier model. Pass an empty string to clear it.
        #[arg(long, value_name = "MODEL")]
        pro: Option<String>,

        /// Set the Ultra tier model. Pass an empty string to clear it.
        #[arg(long, value_name = "MODEL")]
        ultra: Option<String>,

        /// Per-provider routing mode override: auto, custom, or locked.
        #[arg(long, value_name = "MODE")]
        routing_mode: Option<String>,

        /// Per-provider minimum quality floor: economy, balanced, or high.
        #[arg(long, value_name = "QUALITY")]
        routing_minimum_quality: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
pub enum ProviderCommands {
    /// List configured providers and their tier mappings.
    List,

    /// Add a provider by entering its API key (or, for Copilot, running the
    /// OAuth device flow). The credential is saved to
    /// ~/.reincore-io/credentials.json so the provider is available in all
    /// future daemon starts without re-entering the key.
    Add {
        /// Provider label: openai, anthropic, gemini, azure, groq, xai,
        /// deepseek, meta, copilot, ollama.
        provider: String,

        /// API key (for non-OAuth providers). If omitted, you'll be prompted
        /// interactively. Ignored for Copilot (uses OAuth device flow).
        #[arg(short, long)]
        key: Option<String>,

        /// Use the cloud variant of a provider that supports both local and
        /// cloud deployment (currently only `ollama`). When set, models are
        /// fetched from the Ollama Cloud API instead of the local server.
        #[arg(long)]
        cloud: bool,
    },

    /// Remove a provider's stored credentials.
    Remove {
        /// Provider label to remove.
        provider: String,
    },

    /// Test a provider by sending a minimal chat completion.
    Test {
        /// Provider label, e.g. openai, anthropic, gemini, azure, groq, xai, deepseek, meta.
        provider: String,

        /// Optional prompt to send. Defaults to a simple hello.
        #[arg(short, long)]
        prompt: Option<String>,
    },

    /// Fetch and display available models from a provider (or all providers).
    Models {
        /// Provider label, or omit to list models from all configured providers.
        provider: Option<String>,
    },
}

#[derive(Clone, Debug)]
pub struct LaunchArgs {
    pub app: String,
    pub provider: Option<String>,
    pub mode: String,
    pub host: String,
    pub port: u16,
    pub db_path: String,
    pub path: Option<String>,
    pub verbose: bool,
    pub restore: bool,
    pub cloud: bool,
    pub model: Option<String>,
}

#[derive(Subcommand, Clone, Debug)]
pub enum TelemetryCommands {
    /// Start the bundled Jaeger container and print the trace UI URL.
    Start,

    /// Stop the bundled Jaeger container.
    Stop,

    /// Show whether the bundled Jaeger container is running.
    Status,

    /// Open the Jaeger UI in the default browser.
    Open,

    /// Start Jaeger and persist OTLP settings so the daemon exports traces.
    Enable,

    /// Remove persisted OTLP settings so the daemon stops exporting traces.
    Disable,
}

#[derive(Subcommand, Debug)]
pub enum McpCommand {
    /// List saved upstream servers.
    List,
    /// Add one server from a JSON ServerConfig file; restart to apply.
    Add {
        #[arg(required_unless_present = "stdin", conflicts_with = "stdin")]
        file: Option<std::path::PathBuf>,
        #[arg(long)]
        stdin: bool,
    },
    /// Replace an existing server; JSON name must match. Restart to apply.
    Update {
        name: String,
        #[arg(required_unless_present = "stdin", conflicts_with = "stdin")]
        file: Option<std::path::PathBuf>,
        #[arg(long)]
        stdin: bool,
    },
    /// Delete a saved server; restart to apply.
    Delete { name: String },
    /// Sign in to a saved server with OAuth in your browser; restart to apply.
    ///
    /// Without --client-id, ReinCore uses the server's current OAuth client,
    /// or registers one when the server allows it.
    SignIn {
        name: String,
        /// Client ID of an OAuth client you created with the provider.
        #[arg(long)]
        client_id: Option<String>,
        /// Environment variable holding that client's secret.
        #[arg(long, requires = "client_id")]
        client_secret_env: Option<String>,
        /// A scope to request; repeat for several. Defaults to what the server asks for.
        #[arg(long = "scope")]
        scopes: Vec<String>,
    },
    /// Enable the local MCP gateway and configured upstream tools at startup.
    Enable,
    /// Disable local MCP at the next daemon startup.
    Disable,
    /// Print saved MCP state and the effective configuration file path.
    Status,
}

#[cfg(test)]
mod mcp_input_tests {
    use super::*;
    #[test]
    fn input_requires_exactly_one_file_or_stdin() {
        for action in ["add", "update"] {
            let mut args = vec!["reincore", "mcp"];
            args.push(action);
            if action == "update" {
                args.push("repo");
            }
            assert!(Cli::try_parse_from(args.clone()).is_err());
            let mut stdin = args.clone();
            stdin.push("--stdin");
            assert!(Cli::try_parse_from(stdin.clone()).is_ok());
            stdin.push("server.json");
            assert!(Cli::try_parse_from(stdin).is_err());
            args.push("server.json");
            assert!(Cli::try_parse_from(args).is_ok());
        }
    }
}

#[cfg(test)]
mod local_defaults_tests {
    use super::*;

    #[test]
    fn daemon_commands_default_to_reincore_io_home_and_8989() {
        assert_eq!(DEFAULT_PORT, 8989);
        assert_eq!(DEFAULT_DB_PATH, "~/.reincore-io/reincore.db");
        match Cli::try_parse_from(["reincore", "start"]).unwrap().command {
            Commands::Start { port, db_path, .. } => {
                assert_eq!(port, DEFAULT_PORT);
                assert_eq!(db_path, DEFAULT_DB_PATH);
            }
            other => panic!("unexpected command: {other:?}"),
        }
        match Cli::try_parse_from(["reincore", "launch", "codex"])
            .unwrap()
            .command
        {
            Commands::Launch { port, db_path, .. } => {
                assert_eq!(port, DEFAULT_PORT);
                assert_eq!(db_path, DEFAULT_DB_PATH);
            }
            other => panic!("unexpected command: {other:?}"),
        }
        match Cli::try_parse_from(["reincore", "setup", "codex"])
            .unwrap()
            .command
        {
            Commands::Setup { port, .. } => assert_eq!(port, DEFAULT_PORT),
            other => panic!("unexpected command: {other:?}"),
        }
        match Cli::try_parse_from(["reincore", "migrate"])
            .unwrap()
            .command
        {
            Commands::Migrate { db_path } => {
                assert_eq!(db_path, DEFAULT_DB_PATH)
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn daemon_state_file_lives_under_reincore_io() {
        let path = daemon_config_path();
        assert!(
            path.ends_with(".reincore-io/daemon.json")
                || path.ends_with(".reincore-io\\daemon.json"),
            "unexpected daemon path: {}",
            path.display()
        );
    }
}

#[derive(Subcommand, Debug)]
pub enum DataCommand {
    Backup {
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Verify {
        #[arg(long)]
        source: PathBuf,
    },
    Restore {
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Retain {
        #[arg(long)]
        source: PathBuf,
        /// RFC3339 cutoff; only completed, unlinked sessions without live checkpoints qualify.
        #[arg(long, conflicts_with = "retention_days")]
        before: Option<chrono::DateTime<chrono::Utc>>,
        /// Override the configured privacy retention_days for this run.
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..=3650))]
        retention_days: Option<u32>,
        #[arg(long, default_value_t = 100)]
        limit: usize,
        #[arg(long, requires_all = ["offline", "backup"])]
        apply: bool,
        /// Acknowledge that all engines and workers using this DB have been stopped.
        #[arg(long)]
        offline: bool,
        /// New pre-deletion snapshot file; required with --apply.
        #[arg(long)]
        backup: Option<PathBuf>,
    },
}

#[cfg(test)]
mod data_command_tests {
    use super::*;
    #[test]
    fn retention_apply_requires_offline_and_backup() {
        let args = [
            "rc",
            "data",
            "retain",
            "--source",
            "test.db",
            "--before",
            "2026-01-01T00:00:00Z",
        ];
        assert!(Cli::try_parse_from(args).is_ok());
        let by_age = [
            "rc",
            "data",
            "retain",
            "--source",
            "test.db",
            "--retention-days",
            "7",
        ];
        assert!(Cli::try_parse_from(by_age).is_ok());
        let mut apply = args.to_vec();
        apply.push("--apply");
        assert!(Cli::try_parse_from(&apply).is_err());
        apply.extend(["--offline", "--backup", "new.db"]);
        assert!(Cli::try_parse_from(&apply).is_ok());
    }
}
