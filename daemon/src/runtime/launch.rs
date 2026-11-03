use anyhow::Result;
use std::collections::HashMap;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::cli::{DaemonConfig, LaunchArgs};
use crate::credentials::{self, CredentialKind};
use crate::local_config::read_provider_models;
use crate::provider::provider_start_proxy;
use colored::Colorize;
use engine_core::log;
use engine_core::model_library::{default_model_for_provider_and_tier, find_model, Modality};
use engine_core::provider::{OllamaMode, OllamaModelInfo, OllamaProvider};

/// Arguments for the `setup` command (configure an app without launching it).
pub struct SetupArgs {
    pub app: String,
    pub provider: Option<String>,
    pub mode: String,
    pub host: String,
    pub port: u16,
}

/// Known coding/agent apps and their default provider.
pub fn app_defaults() -> HashMap<&'static str, &'static str> {
    [
        ("claude", "anthropic"),
        ("codex", "openai"),
        ("codex-app", "openai"),
        ("copilot", "copilot"),
        ("gemini", "gemini"),
        ("gemini-cli", "gemini"),
        ("cursor", "openai"),
        ("windsurf", "openai"),
        ("aider", "openai"),
        ("continue", "openai"),
        ("open-code", "openai"),
        ("opencode", "openai"),
        ("openclaw", "anthropic"),
        ("open-claw", "anthropic"),
        ("muse", "meta"),
        ("ollama", "ollama"),
    ]
    .into_iter()
    .collect()
}

/// Return the executable aliases commonly used to invoke an app from the shell.
fn app_command_aliases(app: &str) -> Vec<&str> {
    match app.to_lowercase().as_str() {
        "claude" => vec!["claude", "claude-code"],
        "codex" => vec!["codex", "codex-cli"],
        "copilot" => vec!["github-copilot-cli", "copilot"],
        "gemini" | "gemini-cli" => vec!["gemini"],
        "cursor" => vec!["cursor"],
        "windsurf" => vec!["windsurf"],
        "aider" => vec!["aider", "aider-chat"],
        "continue" => vec!["continue"],
        "open-code" | "opencode" | "open code" => vec!["open-code", "opencode"],
        "openclaw" | "open-claw" | "open claw" => vec!["openclaw", "open-claw"],
        "muse" => vec!["muse"],
        "ollama" => vec!["ollama"],
        _ => vec![app],
    }
}

/// Known app install directories on Windows that are not always on PATH.
fn windows_app_install_dirs() -> HashMap<&'static str, Vec<PathBuf>> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let local_programs = home.join("AppData").join("Local").join("Programs");

    // Claude Code's installer drops the `claude` binary in `~/.local/bin`,
    // which is not on PATH by default on Windows.
    let claude_local_bin = home.join(".local").join("bin");

    [
        (
            "codex",
            vec![local_programs.join("OpenAI").join("Codex").join("bin")],
        ),
        (
            "claude",
            vec![
                local_programs.join("Anthropic").join("Claude"),
                claude_local_bin,
            ],
        ),
        ("muse", vec![local_programs.join("muse")]),
    ]
    .into_iter()
    .collect()
}

/// On Windows, the per-user bin directory that some installers (notably
/// Claude Code) use without adding it to PATH.
#[cfg(target_os = "windows")]
fn user_local_bin() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".local")
        .join("bin")
}

/// True if `dir` is already listed in the PATH environment variable.
#[cfg(target_os = "windows")]
fn is_on_path(dir: &Path) -> bool {
    let path = std::env::var("PATH").unwrap_or_default();
    let dir_lower = dir.to_string_lossy().to_lowercase();
    path.split(';').any(|p| {
        let p = p.trim();
        !p.is_empty() && p.to_lowercase() == dir_lower
    })
}

/// Resolve the provider label for a given app, using an explicit override if provided.
pub fn resolve_provider(app: &str, provider_override: Option<String>) -> Result<String> {
    if let Some(provider) = provider_override {
        return Ok(provider.to_lowercase());
    }

    let defaults = app_defaults();
    defaults
        .get(app.to_lowercase().as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| {
            let known: Vec<_> = defaults.keys().copied().collect();
            anyhow::anyhow!(
                "unknown app '{}'. Known apps: {}. Use --provider to specify one explicitly.",
                app,
                known.join(", ")
            )
        })
}

/// Check whether an alias resolves via a login shell's PATH.
#[cfg(unix)]
fn check_alias_via_login_shell(alias: &str) -> bool {
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
    match std::process::Command::new(&shell)
        .arg("-lc")
        .arg(format!("command -v {}", alias))
        .output()
    {
        Ok(output) => output.status.success() && !output.stdout.is_empty(),
        Err(_) => false,
    }
}

/// Check whether an alias resolves via Windows' `where`.
#[cfg(target_os = "windows")]
fn check_alias_via_where(alias: &str) -> bool {
    match std::process::Command::new("where").arg(alias).output() {
        Ok(output) => output.status.success() && !output.stdout.is_empty(),
        Err(_) => false,
    }
}

/// Check whether the named app appears to be installed on this system.
pub fn is_app_installed(app: &str) -> bool {
    let app_lower = app.to_lowercase();
    let aliases = app_command_aliases(&app_lower);

    for alias in &aliases {
        if which::which(alias).is_ok() {
            return true;
        }
    }

    #[cfg(unix)]
    {
        for alias in &aliases {
            if check_alias_via_login_shell(alias) {
                return true;
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        for alias in &aliases {
            if check_alias_via_where(alias) {
                return true;
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        let home = dirs::home_dir().unwrap_or_else(|| Path::new(".").to_path_buf());
        let local_app_data = std::env::var("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home.clone());
        let program_files = std::env::var("ProgramFiles").map(PathBuf::from).ok();
        let program_files_x86 = std::env::var("ProgramFiles(x86)").map(PathBuf::from).ok();

        let mut candidate_paths: Vec<PathBuf> = match app_lower.as_str() {
            "claude" => vec![
                local_app_data.join("Anthropic").join("Claude"),
                program_files
                    .clone()
                    .map(|p| p.join("Anthropic").join("Claude"))
                    .unwrap_or_default(),
                program_files_x86
                    .clone()
                    .map(|p| p.join("Anthropic").join("Claude"))
                    .unwrap_or_default(),
                user_local_bin(),
            ],
            "codex" => vec![
                local_app_data.join("Codex"),
                program_files
                    .clone()
                    .map(|p| p.join("Codex"))
                    .unwrap_or_default(),
            ],
            "copilot" => vec![
                local_app_data.join("GitHub").join("Copilot"),
                program_files
                    .clone()
                    .map(|p| p.join("GitHub").join("Copilot"))
                    .unwrap_or_default(),
            ],
            // Gemini CLI is an npm package; the binary is in the global npm
            // bin dir. We add a few common npm install locations on Windows
            // as best-effort discovery paths.
            "gemini" | "gemini-cli" => vec![
                home.join("AppData").join("Roaming").join("npm"),
                local_app_data.join("npm"),
            ],
            "cursor" => vec![
                local_app_data.join("Cursor"),
                program_files
                    .clone()
                    .map(|p| p.join("Cursor"))
                    .unwrap_or_default(),
            ],
            "windsurf" => vec![
                local_app_data.join("Windsurf"),
                program_files
                    .clone()
                    .map(|p| p.join("Windsurf"))
                    .unwrap_or_default(),
            ],
            _ => vec![],
        };

        if let Some(extra) = windows_app_install_dirs().get(app_lower.as_str()) {
            candidate_paths.extend(extra.iter().cloned());
        }

        for path in candidate_paths {
            if path.exists() {
                return true;
            }
        }

        let start_menu = local_app_data
            .join("Microsoft")
            .join("Windows")
            .join("Start Menu")
            .join("Programs");
        if start_menu.exists() {
            let needle = app_lower.clone();
            if let Ok(entries) = std::fs::read_dir(&start_menu) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_lowercase();
                    if name.contains(&needle) {
                        return true;
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        let applications = Path::new("/Applications");
        let home_apps = dirs::home_dir()
            .unwrap_or_else(|| Path::new(".").to_path_buf())
            .join("Applications");

        let candidate_names: Vec<String> = match app_lower.as_str() {
            "claude" => vec!["Claude".to_string(), "Claude Code".to_string()],
            "codex" => vec!["Codex".to_string()],
            "copilot" => vec!["GitHub Copilot".to_string()],
            "gemini" | "gemini-cli" => vec!["Gemini CLI".to_string()],
            "cursor" => vec!["Cursor".to_string()],
            "windsurf" => vec!["Windsurf".to_string()],
            _ => vec![app_lower.clone()],
        };

        for name in candidate_names {
            if applications.join(format!("{}.app", name)).exists()
                || home_apps.join(format!("{}.app", name)).exists()
            {
                return true;
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        let home = dirs::home_dir().unwrap_or_else(|| Path::new(".").to_path_buf());
        let desktop_dirs = vec![
            home.join(".local").join("share").join("applications"),
            Path::new("/usr").join("share").join("applications"),
            Path::new("/usr")
                .join("local")
                .join("share")
                .join("applications"),
        ];

        for dir in desktop_dirs {
            if !dir.exists() {
                continue;
            }
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_lowercase();
                    if name.contains(&app_lower) {
                        return true;
                    }
                }
            }
        }
    }

    false
}

/// Environment variables that apps read to determine their API endpoint.
fn app_env_vars(
    app: &str,
    proxy_base_url: &str,
    proxy_url: &str,
    provider_header: Option<&str>,
) -> Vec<(&'static str, String)> {
    let provider_header_value = provider_header.map(str::to_string);
    match app.to_lowercase().as_str() {
        "claude" => {
            let mut vars = vec![
                // Claude Code appends /v1/messages to ANTHROPIC_BASE_URL itself,
                // so it must be the base URL WITHOUT /v1.
                ("ANTHROPIC_BASE_URL", proxy_base_url.to_string()),
                ("ANTHROPIC_AUTH_TOKEN", "reincore".to_string()),
                // Claude Code sometimes drops the Authorization header when
                // routing through a custom ANTHROPIC_BASE_URL, causing the proxy
                // to return 404 and Claude Code to surface a generic "model
                // issue" error. Force the header through via ANTHROPIC_CUSTOM_HEADERS.
                // See: https://github.com/anthropics/claude-code/issues/44752
                (
                    "ANTHROPIC_CUSTOM_HEADERS",
                    "Authorization: Bearer reincore".to_string(),
                ),
                // Use a Claude model name that Claude Code recognizes as having
                // a 1M context window (claude-sonnet-5 in our catalog has
                // max_input_tokens: 1_000_000). The proxy ignores the
                // client-supplied model and routes the request itself, so this
                // name is purely for Claude Code's internal context-window
                // tracking - it determines when Claude Code triggers its own
                // session compaction. With a 1M value, Claude Code won't
                // compact before ReinCore's engine does.
                ("ANTHROPIC_MODEL", "claude-sonnet-5".to_string()),
                // Enable gateway model discovery so Claude Code populates the
                // /model picker from the proxy's /v1/models endpoint.
                (
                    "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY",
                    "1".to_string(),
                ),
                ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1".to_string()),
            ];
            // Multi-provider routing: add X-ReinCore-Provider to the custom
            // headers (newline-separated list).
            if let Some(header) = provider_header_value {
                if let Some(custom) = vars
                    .iter_mut()
                    .find(|(k, _)| *k == "ANTHROPIC_CUSTOM_HEADERS")
                {
                    custom.1 = format!("{}\nX-ReinCore-Provider: {}", custom.1, header);
                }
            }
            vars
        }
        "codex" => {
            let mut vars = vec![
                ("OPENAI_BASE_URL", proxy_url.to_string()),
                ("OPENAI_API_KEY", "reincore".to_string()),
            ];
            if let Some(header) = provider_header_value {
                vars.push(("REINCORE_PROVIDER", header));
            }
            vars
        }
        "open-code" | "opencode" | "open code" => {
            // OpenCode does NOT read OPENAI_BASE_URL - it's driven entirely by
            // the `reincore` provider written into opencode.json by
            // activate_opencode_profile. Injecting OPENAI_* here would only
            // (wrongly) surface the built-in `openai` provider in its picker.
            let _ = (proxy_base_url, proxy_url);
            vec![]
        }
        // OpenClaw is driven entirely by the `reincore` provider written into
        // ~/.openclaw/openclaw.json by activate_openclaw_profile. OpenClaw does
        // NOT read OPENAI_*/ANTHROPIC_* env vars for a custom base URL, so we
        // inject nothing here (setting them would only surface OpenClaw's
        // built-in providers with a bogus key).
        "openclaw" | "open-claw" | "open claw" => {
            let _ = (proxy_base_url, proxy_url);
            vec![]
        }
        "copilot" => vec![
            // GitHub Copilot CLI in Copilot endpoint mode. The CLI is pointed
            // at reincore's proxy, which uses the CopilotProvider to forward
            // requests to api.githubcopilot.com using a GitHub PAT.
            // NOTE: the Copilot CLI has no mechanism for custom request
            // headers, so it cannot be routed through a *multi-provider*
            // daemon; the launcher restarts the daemon in single-provider
            // mode for copilot instead (see app_supports_provider_header).
            ("COPILOT_PROVIDER_TYPE", "openai".to_string()),
            ("COPILOT_PROVIDER_BASE_URL", proxy_url.to_string()),
            ("COPILOT_PROVIDER_API_KEY", "reincore".to_string()),
            ("COPILOT_MODEL", "reincore-auto".to_string()),
        ],
        // Gemini CLI uses the native Google Gemini API (not the
        // OpenAI-compatible shim), so it must point at reincore's proxy as the
        // base URL. The proxy serves /v1beta/models/...:generateContent and
        // /v1beta/models/...:streamGenerateContent, translating to the core
        // provider pipeline.
        "gemini" | "gemini-cli" => {
            let mut vars = vec![
                // Override the Gemini API base URL. The Gemini CLI appends
                // `/v1beta/models/<model>:generateContent` to this, so we supply
                // the bare host:port root.
                ("GOOGLE_GEMINI_BASE_URL", proxy_base_url.to_string()),
                // A non-empty API key is required by the SDK even when the
                // endpoint is a proxy that ignores it; the proxy authenticates
                // via the configured backend provider.
                ("GEMINI_API_KEY", "reincore".to_string()),
            ];
            // Multi-provider routing: the Gemini CLI parses custom request
            // headers from GEMINI_CLI_CUSTOM_HEADERS (comma-separated
            // `Name: value` pairs).
            if let Some(header) = provider_header_value {
                vars.push((
                    "GEMINI_CLI_CUSTOM_HEADERS",
                    format!("X-ReinCore-Provider: {}", header),
                ));
            }
            vars
        }
        "cursor" => vec![
            // Cursor's custom-endpoint support is GUI-settings-only; there
            // is no env-var mechanism for custom request headers, so a
            // *multi-provider* daemon cannot be reused for cursor (see
            // app_supports_provider_header).
            ("OPENAI_API_KEY", "reincore".to_string()),
            ("OPENAI_BASE_URL", proxy_url.to_string()),
        ],
        "windsurf" => vec![
            ("OPENAI_API_KEY", "reincore".to_string()),
            ("OPENAI_BASE_URL", proxy_url.to_string()),
        ],
        "aider" => {
            let mut vars = vec![
                ("OPENAI_API_KEY", "reincore".to_string()),
                ("OPENAI_BASE_URL", proxy_url.to_string()),
            ];
            // Multi-provider routing: aider merges `extra_headers` from the
            // special `aider/extra_params` entry in the model-settings file;
            // activate_aider_profile writes it, we just point aider at it.
            if provider_header_value.is_some() {
                let settings_path = dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".aider.model.settings.reincore.yml");
                vars.push((
                    "AIDER_MODEL_SETTINGS_FILE",
                    settings_path.display().to_string(),
                ));
            }
            vars
        }
        "continue" => vec![
            ("OPENAI_API_KEY", "reincore".to_string()),
            ("OPENAI_BASE_URL", proxy_url.to_string()),
        ],
        // Ollama is a provider, not a client app. When launched via
        // `reincore launch ollama`, we only start the proxy - there's no
        // separate app process to spawn, so no env vars are needed.
        "ollama" => vec![],
        // Muse Code takes its endpoint exclusively via CLI flags
        // (`--provider/--base-url/--model`, see `app_extra_args`), which the
        // launch script passes explicitly - no env vars needed.
        "muse" => vec![],
        _ => vec![],
    }
}

/// Extra CLI arguments appended after the app executable in the generated
/// launch script. Most apps are fully configured through env vars and their
/// persistent profile, so this is empty - except Muse Code, whose endpoint
/// (base URL) has no env-var or persistent-settings equivalent and must be
/// passed as flags on every invocation.
fn app_extra_args(app: &str, proxy_base_url: &str) -> Vec<String> {
    match app.to_lowercase().as_str() {
        "muse" => vec![
            "--provider".to_string(),
            "meta".to_string(),
            "--base-url".to_string(),
            proxy_base_url.to_string(),
            "--model".to_string(),
            "reincore-auto".to_string(),
        ],
        _ => vec![],
    }
}

/// Whether `app` can send the `X-ReinCore-Provider` request header when
/// routed through a multi-provider daemon. Apps without such a mechanism
/// (Cursor/Windsurf GUI-only custom endpoints, the Copilot CLI, Continue)
/// must use a single-provider daemon instead - the launch negotiation then
/// treats a running multi-provider daemon as unusable for those apps.
///
/// Header mechanisms per app:
/// - claude: `ANTHROPIC_CUSTOM_HEADERS` (newline-separated `Name: value`)
/// - codex: `http_headers` in `model_providers.<id>` (config.toml)
/// - opencode/openclaw: `headers` in the provider config JSON
/// - gemini: `GEMINI_CLI_CUSTOM_HEADERS` (comma-separated `Name: value`)
/// - aider: `extra_params.extra_headers` via the model-settings file
///
/// Apps without such a mechanism (muse, cursor/windsurf GUI-only custom
/// endpoints, the Copilot CLI, Continue) launch against a single-provider
/// daemon instead. Muse Code still works against a multi-provider daemon
/// when invoked manually: it names models (`reincore-auto` or a provider's
/// model id), which the daemon resolves without headers.
pub fn app_supports_provider_header(app: &str) -> bool {
    matches!(
        app.to_lowercase().as_str(),
        "claude"
            | "codex"
            | "codex-app"
            | "open-code"
            | "opencode"
            | "open code"
            | "openclaw"
            | "open-claw"
            | "open claw"
            | "gemini"
            | "gemini-cli"
            | "aider"
    )
}

/// Environment variables to explicitly unset in the spawned terminal before
/// setting the reincore proxy vars. This prevents conflicts where the app
/// sees both a reincore auth token and a real API key inherited from the
/// parent process (e.g. Claude Code warns when both ANTHROPIC_AUTH_TOKEN and
/// ANTHROPIC_API_KEY are set).
fn app_env_vars_to_unset(app: &str) -> Vec<&'static str> {
    match app.to_lowercase().as_str() {
        "claude" => vec!["ANTHROPIC_API_KEY"],
        _ => vec![],
    }
}

// Persistent per-app profiles: create / activate / restore.
//
// Unlike the env vars above (which only apply to the one process reincore
// spawns), these functions edit each app's own persistent config file so
// *every* invocation of that app routes through reincore until restored -
// the same idea as `reincore <app> --restore` reverting things back.
//
// Every edit goes through `activate_generic`/`restore_generic`, which back
// up the entire original config file (or record that it didn't exist) before
// touching anything, and restore that exact backup on `--restore`. This is
// deliberately simple and file-level rather than trying to reverse a partial
// edit, so restore can't drift from the real original.

/// Where reincore keeps the backups needed to restore an app's config.
fn reincore_state_dir() -> PathBuf {
    crate::config::local_config::config_home().join("profiles")
}

fn state_file_for(app_key: &str, suffix: &str) -> PathBuf {
    reincore_state_dir().join(format!("{}.{}", app_key.to_lowercase(), suffix))
}

/// Back up `config_path` (if we haven't already) and hand its current
/// contents to `mutate`, writing whatever it returns back to the file.
/// Safe to call repeatedly - a second activate without an intervening
/// restore does NOT overwrite the original backup, so the real "before
/// reincore" state is never lost.
fn activate_generic<F>(app_key: &str, config_path: &Path, mutate: F) -> Result<()>
where
    F: FnOnce(String) -> Result<String>,
{
    std::fs::create_dir_all(reincore_state_dir())
        .map_err(|e| anyhow::anyhow!("failed to create reincore state dir: {}", e))?;
    let backup_path = state_file_for(app_key, "backup");

    if !backup_path.exists() {
        let backup_contents = match std::fs::read_to_string(config_path) {
            Ok(original) => format!("PRESENT\n{}", original),
            Err(_) => "ABSENT\n".to_string(),
        };
        std::fs::write(&backup_path, backup_contents).map_err(|e| {
            anyhow::anyhow!("failed to write backup {}: {}", backup_path.display(), e)
        })?;
    }

    let current = std::fs::read_to_string(config_path).unwrap_or_default();
    let updated = mutate(current)?;

    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(config_path, updated)
        .map_err(|e| anyhow::anyhow!("failed to write {}: {}", config_path.display(), e))?;
    Ok(())
}

/// Put back whatever `activate_generic` backed up. Returns `Ok(false)` (not
/// an error) if there was nothing to restore, so callers can tell "restored"
/// apart from "there was no reincore profile active for this app".
fn restore_generic(app_key: &str, config_path: &Path) -> Result<bool> {
    let backup_path = state_file_for(app_key, "backup");
    if !backup_path.exists() {
        return Ok(false);
    }
    let backup = std::fs::read_to_string(&backup_path)
        .map_err(|e| anyhow::anyhow!("failed to read backup {}: {}", backup_path.display(), e))?;

    if let Some(_rest) = backup.strip_prefix("ABSENT\n") {
        if config_path.exists() {
            std::fs::remove_file(config_path).map_err(|e| {
                anyhow::anyhow!("failed to remove {}: {}", config_path.display(), e)
            })?;
        }
    } else if let Some(original) = backup.strip_prefix("PRESENT\n") {
        std::fs::write(config_path, original)
            .map_err(|e| anyhow::anyhow!("failed to restore {}: {}", config_path.display(), e))?;
    } else {
        return Err(anyhow::anyhow!(
            "backup file {} is in an unexpected format; not restoring",
            backup_path.display()
        ));
    }

    std::fs::remove_file(&backup_path).ok();
    Ok(true)
}

/// True if `line` is a top-level `key = ...` assignment for the given key -
/// not a table header, not an indented/nested key inside some other table,
/// and not an unrelated key that merely starts with the same prefix (e.g.
/// `key` = "model" shouldn't match a `model_provider` line).
fn is_top_level_key_line(line: &str, key: &str) -> bool {
    if line.starts_with(' ') || line.starts_with('\t') {
        return false;
    }
    match line.strip_prefix(key) {
        Some(rest) => rest.trim_start().starts_with('='),
        None => false,
    }
}

/// Set (or add) a top-level `key = <raw_value>` line in a TOML file, without
/// pulling in a TOML parser. `raw_value` must already include any quoting
/// (e.g. `"\"reincore-auto\""` for a basic string, `"'C:\\path'"` for a
/// literal string) - the caller decides that so paths with backslashes can
/// use TOML literal (single-quoted) strings and avoid escaping entirely.
fn set_toml_top_level_key_raw(current: &str, key: &str, raw_value: &str) -> String {
    let new_line = format!("{} = {}", key, raw_value);
    let mut out = String::new();
    let mut replaced = false;
    for line in current.lines() {
        if is_top_level_key_line(line, key) {
            out.push_str(&new_line);
            out.push('\n');
            replaced = true;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    if !replaced {
        out = format!("{}\n{}", new_line, out);
    }
    out
}

/// Set (or add) a `[table_header]` block, replacing its entire existing body
/// (up to the next `[`-starting line or EOF) if the header already exists, or
/// appending a fresh block at the end otherwise. Used for
/// `[model_providers.reincore]`, which has more than one field and so can't
/// be handled by `set_toml_top_level_key_raw`.
fn set_toml_table_block(current: &str, table_header: &str, body_lines: &[String]) -> String {
    let header_line = format!("[{}]", table_header);
    let lines: Vec<&str> = current.lines().collect();

    let start_idx = lines.iter().position(|line| line.trim() == header_line);

    let mut new_block = String::new();
    new_block.push_str(&header_line);
    new_block.push('\n');
    for line in body_lines {
        new_block.push_str(line);
        new_block.push('\n');
    }

    match start_idx {
        Some(start) => {
            let end = lines
                .iter()
                .enumerate()
                .skip(start + 1)
                .find(|(_, line)| line.starts_with('['))
                .map(|(i, _)| i)
                .unwrap_or(lines.len());

            let mut out = String::new();
            for line in &lines[..start] {
                out.push_str(line);
                out.push('\n');
            }
            out.push_str(&new_block);
            for line in &lines[end..] {
                out.push_str(line);
                out.push('\n');
            }
            out
        }
        None => {
            let mut out = current.to_string();
            if !out.is_empty() && !out.ends_with('\n') {
                out.push('\n');
            }
            out.push('\n');
            out.push_str(&new_block);
            out
        }
    }
}

/// $CODEX_HOME, defaulting to ~/.codex.
fn codex_home() -> PathBuf {
    std::env::var("CODEX_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".codex")
        })
}

/// Build the single, honest "auto" catalog entry for Codex's model picker.
///
/// reincore's single-provider proxy (`chat_completions_single_provider`)
/// always resolves the real model itself, via its classifier + budget-tier
/// degradation - it never reads the client-supplied `model` field for
/// routing. So this entry exists only so Codex has something valid to
/// display/select; it has no effect on which real model actually serves the
/// request. Context window / input modalities are pulled from the real
/// catalog record for this provider's "pro" tier model, purely for a more
/// accurate-looking display - not because it's the model that will actually
/// run.
///
/// NOTE: `engine_core::provider_model` is a guessed module path for
/// `default_model_for_provider_and_tier` / `find_model` / `Modality` - adjust
/// to wherever engine_core actually re-exports them if this doesn't compile.
fn build_codex_catalog_entry(provider_label: &str) -> serde_json::Value {
    let representative = default_model_for_provider_and_tier(provider_label, "pro")
        .and_then(|model_id| find_model(provider_label, &model_id));

    let context_window = representative
        .as_ref()
        .map(|m| m.max_input_tokens)
        .unwrap_or(1_000_000);

    let input_modalities: Vec<&str> = representative
        .as_ref()
        .map(|m| {
            m.supported_input_data
                .iter()
                .map(|modality| match modality {
                    Modality::Text => "text",
                    Modality::Image => "image",
                    Modality::Audio => "audio",
                    Modality::Video => "video",
                })
                .collect()
        })
        .unwrap_or_else(|| vec!["text"]);

    serde_json::json!({
        "models": [{
            "base_instructions": "",
            "context_window": context_window,
            "default_verbosity": "low",
            "display_name": "reincore (auto)",
            "experimental_supported_tools": [],
            "input_modalities": input_modalities,
            "priority": 0,
            "shell_type": "default",
            "slug": "reincore-auto",
            "support_verbosity": true,
            "supported_in_api": true,
            "supported_reasoning_levels": [],
            "supports_parallel_tool_calls": false,
            "supports_reasoning_summaries": false,
            "truncation_policy": { "limit": 10000, "mode": "tokens" },
            "visibility": "list"
        }]
    })
}

/// Point Codex at reincore with a stable `model_providers.reincore` entry -
/// unlike some other launchers, which mint a freshly-named provider id every
/// run (leaving old, orphaned `[model_providers.*]` blocks behind), this
/// always edits the same `reincore` id in place, so repeated launches don't
/// accumulate cruft in config.toml.
fn activate_codex_profile(
    proxy_url: &str,
    provider_label: &str,
    mode: &str,
    provider_header: Option<&str>,
) -> Result<()> {
    let home = codex_home();
    std::fs::create_dir_all(&home)
        .map_err(|e| anyhow::anyhow!("failed to create {}: {}", home.display(), e))?;

    // Entirely our own file - safe to overwrite wholesale every time.
    let catalog_path = home.join("reincore-models.json");
    let catalog = build_codex_catalog_entry(provider_label);
    std::fs::write(&catalog_path, serde_json::to_string_pretty(&catalog)?)
        .map_err(|e| anyhow::anyhow!("failed to write {}: {}", catalog_path.display(), e))?;

    let config_path = home.join("config.toml");
    activate_generic("codex", &config_path, |current| {
        let mut out = current;
        out = set_toml_top_level_key_raw(&out, "model", "\"reincore-auto\"");
        out = set_toml_top_level_key_raw(&out, "model_provider", "\"reincore\"");
        // Literal (single-quoted) TOML string - sidesteps backslash escaping
        // entirely, which matters for Windows paths.
        out = set_toml_top_level_key_raw(
            &out,
            "model_catalog_json",
            &format!("'{}'", catalog_path.display()),
        );
        let mut lines = vec![
            "name = \"ReinCore\"".to_string(),
            format!("base_url = \"{}\"", proxy_url),
            format!("wire_api = \"{}\"", mode),
        ];
        // Multi-provider routing: Codex's `http_headers` map carries the
        // X-ReinCore-Provider header on every request.
        if let Some(header) = provider_header {
            lines.push(format!(
                "http_headers = {{ \"X-ReinCore-Provider\" = \"{}\" }}",
                header
            ));
        }
        out = set_toml_table_block(&out, "model_providers.reincore", &lines);
        Ok(out)
    })
}

fn restore_codex_profile() -> Result<bool> {
    let home = codex_home();
    // Best-effort cleanup of our own file; ignore if it's already gone.
    std::fs::remove_file(home.join("reincore-models.json")).ok();
    restore_generic("codex", &home.join("config.toml"))
}

/// ~/.claude/settings.json
fn claude_settings_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".claude")
        .join("settings.json")
}

/// Claude Code reads the `env` object in settings.json at startup and
/// applies it "no matter how claude was launched" - exactly the persistent
/// hook we need. We merge into the existing JSON (preserving every other
/// key) rather than overwriting the file.
fn activate_claude_profile(proxy_base_url: &str, provider_header: Option<&str>) -> Result<()> {
    activate_generic("claude", &claude_settings_path(), |current| {
        let mut settings: serde_json::Value = if current.trim().is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_str(&current).unwrap_or_else(|_| serde_json::json!({}))
        };
        if !settings.is_object() {
            settings = serde_json::json!({});
        }

        let obj = settings.as_object_mut().expect("just ensured object");
        let env = obj.entry("env").or_insert_with(|| serde_json::json!({}));
        if !env.is_object() {
            *env = serde_json::json!({});
        }
        let env_obj = env.as_object_mut().expect("just ensured object");
        env_obj.insert(
            "ANTHROPIC_BASE_URL".to_string(),
            serde_json::json!(proxy_base_url),
        );
        env_obj.insert(
            "ANTHROPIC_AUTH_TOKEN".to_string(),
            serde_json::json!("reincore"),
        );
        // Claude Code sometimes drops the Authorization header when routing
        // through a custom ANTHROPIC_BASE_URL, causing the proxy to return
        // 404 and Claude Code to surface a generic "model issue" error. Force
        // the header through via ANTHROPIC_CUSTOM_HEADERS.
        // See: https://github.com/anthropics/claude-code/issues/44752
        env_obj.insert(
            "ANTHROPIC_CUSTOM_HEADERS".to_string(),
            serde_json::json!(match provider_header {
                Some(header) => format!(
                    "Authorization: Bearer reincore\nX-ReinCore-Provider: {}",
                    header
                ),
                None => "Authorization: Bearer reincore".to_string(),
            }),
        );
        // Claude Code warns when both ANTHROPIC_AUTH_TOKEN and
        // ANTHROPIC_API_KEY are set. The proxy uses AUTH_TOKEN, so explicitly
        // blank out API_KEY here to override any value inherited from the
        // user's shell environment.
        env_obj.insert("ANTHROPIC_API_KEY".to_string(), serde_json::json!(""));
        // Use a Claude model name that Claude Code recognizes as having
        // a 1M context window (claude-sonnet-5 in our catalog has
        // max_input_tokens: 1_000_000). The proxy ignores the
        // client-supplied model and routes the request itself, so this
        // name is purely for Claude Code's internal context-window
        // tracking - it determines when Claude Code triggers its own
        // session compaction. With a 1M value, Claude Code won't
        // compact before ReinCore's engine does.
        env_obj.insert(
            "ANTHROPIC_MODEL".to_string(),
            serde_json::json!("claude-sonnet-5"),
        );
        // Enable gateway model discovery so Claude Code populates the /model
        // picker from the proxy's /v1/models endpoint.
        env_obj.insert(
            "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY".to_string(),
            serde_json::json!("1"),
        );
        env_obj.insert(
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC".to_string(),
            serde_json::json!("1"),
        );

        serde_json::to_string_pretty(&settings)
            .map_err(|e| anyhow::anyhow!("failed to serialize settings.json: {}", e))
    })?;

    // Also update the VS Code extension settings so the extension's own
    // login check sees the gateway credential. Best-effort: if VS Code
    // isn't installed, skip silently.
    if let Err(e) = activate_claude_vscode_profile(proxy_base_url, provider_header) {
        eprintln!(
            "Warning: could not update VS Code extension settings: {}",
            e
        );
    }

    Ok(())
}

fn restore_claude_profile() -> Result<bool> {
    let settings_restored = restore_generic("claude", &claude_settings_path())?;
    // Also restore the VS Code extension settings. Best-effort: if there's
    // no VS Code settings file or no reincore profile in it, we still report
    // success based on the ~/.claude/settings.json restoration above.
    let _ = restore_claude_vscode_profile();
    Ok(settings_restored)
}

/// Locate the VS Code user `settings.json`. Returns `None` if the file
/// doesn't exist (VS Code not installed or never opened).
fn vscode_settings_path() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let candidate = if cfg!(target_os = "windows") {
        // %APPDATA%\Code\User\settings.json
        std::env::var("APPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home.join("AppData").join("Roaming"))
            .join("Code")
            .join("User")
            .join("settings.json")
    } else if cfg!(target_os = "macos") {
        home.join("Library")
            .join("Application Support")
            .join("Code")
            .join("User")
            .join("settings.json")
    } else {
        home.join(".config")
            .join("Code")
            .join("User")
            .join("settings.json")
    };
    if candidate.exists() {
        Some(candidate)
    } else {
        None
    }
}

/// Update the VS Code `claudeCode.environmentVariables` array to point the
/// Claude Code extension at the reincore proxy. The extension checks
/// credentials from this setting before launching, so it's the reliable
/// place for the gateway credential (values in ~/.claude/settings.json reach
/// the spawned process but not the extension's own login check).
fn activate_claude_vscode_profile(
    proxy_base_url: &str,
    provider_header: Option<&str>,
) -> Result<()> {
    let path = vscode_settings_path().ok_or_else(|| {
        anyhow::anyhow!("VS Code settings.json not found; skipping VS Code extension profile")
    })?;

    activate_generic("claude_vscode", &path, |current| {
        let mut settings: serde_json::Value = if current.trim().is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_str(&current).unwrap_or_else(|_| serde_json::json!({}))
        };
        if !settings.is_object() {
            settings = serde_json::json!({});
        }

        let obj = settings.as_object_mut().expect("just ensured object");
        // The extension expects an array of { "name": ..., "value": ... } objects.
        let custom_headers = match provider_header {
            Some(header) => format!(
                "Authorization: Bearer reincore\nX-ReinCore-Provider: {}",
                header
            ),
            None => "Authorization: Bearer reincore".to_string(),
        };
        let env_vars = serde_json::json!([
            { "name": "ANTHROPIC_BASE_URL", "value": proxy_base_url },
            { "name": "ANTHROPIC_AUTH_TOKEN", "value": "reincore" },
            { "name": "ANTHROPIC_CUSTOM_HEADERS", "value": custom_headers },
            { "name": "ANTHROPIC_API_KEY", "value": "" },
            { "name": "ANTHROPIC_MODEL", "value": "claude-sonnet-5" },
            { "name": "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "value": "1" },
            { "name": "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "value": "1" },
        ]);
        obj.insert("claudeCode.environmentVariables".to_string(), env_vars);

        serde_json::to_string_pretty(&settings)
            .map_err(|e| anyhow::anyhow!("failed to serialize VS Code settings.json: {}", e))
    })
}

/// Restore the VS Code extension settings to their pre-reincore state.
fn restore_claude_vscode_profile() -> Result<bool> {
    let path = match vscode_settings_path() {
        Some(p) => p,
        None => return Ok(false),
    };
    restore_generic("claude_vscode", &path)
}

/// ~/.aider.conf.yml
fn aider_config_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".aider.conf.yml")
}

/// Set (or add) a flat, top-level `key: value` line in a simple YAML file,
/// without pulling in a YAML crate. Only matches un-indented lines so it
/// can't accidentally clobber a same-named key nested under a mapping.
fn set_yaml_top_level_key(current: &str, key: &str, value: &str) -> String {
    let prefix = format!("{}:", key);
    let mut found = false;
    let mut out = String::new();
    for line in current.lines() {
        if !line.starts_with(' ') && !line.starts_with('\t') && line.starts_with(&prefix) {
            out.push_str(&format!("{}: \"{}\"", key, value));
            out.push('\n');
            found = true;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    if !found {
        out.push_str(&format!("{}: \"{}\"\n", key, value));
    }
    out
}

/// Aider loads `~/.aider.conf.yml` automatically on every run. `openai-api-base`
/// / `openai-api-key` there are the persistent equivalent of the
/// `--openai-api-base` / `OPENAI_API_KEY` flags.
fn activate_aider_profile(proxy_url: &str, provider_header: Option<&str>) -> Result<()> {
    activate_generic("aider", &aider_config_path(), |current| {
        let updated = set_yaml_top_level_key(&current, "openai-api-base", proxy_url);
        let updated = set_yaml_top_level_key(&updated, "openai-api-key", "reincore");
        Ok(updated)
    })?;

    // Multi-provider routing: aider only accepts extra request headers via
    // the model-settings file (the special `aider/extra_params` entry, whose
    // `extra_params.extra_headers` is merged into every litellm completion
    // call). Write our own file so we don't clobber a user's settings file.
    if let Some(header) = provider_header {
        let path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".aider.model.settings.reincore.yml");
        let content = format!(
            "# Written by reincore - routed through a multi-provider daemon.\n\
             # Loaded via AIDER_MODEL_SETTINGS_FILE by the reincore launcher.\n\
             - name: aider/extra_params\n\
             \x20 extra_params:\n\
             \x20   extra_headers:\n\
             \x20     X-ReinCore-Provider: {}\n",
            header
        );
        std::fs::write(&path, content)
            .map_err(|e| anyhow::anyhow!("failed to write {}: {}", path.display(), e))?;
    }

    Ok(())
}

fn restore_aider_profile() -> Result<bool> {
    restore_generic("aider", &aider_config_path())
}

/// OpenCode's global config: `$XDG_CONFIG_HOME/opencode/opencode.json`,
/// falling back to `~/.config/opencode/opencode.json` (this layout is used on
/// every OS, including Windows - OpenCode does not use %APPDATA%).
fn opencode_config_path() -> PathBuf {
    let base = std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".config")
        });
    base.join("opencode").join("opencode.json")
}

/// OpenCode discovers custom OpenAI-compatible providers through its own JSON
/// config - it does NOT read `OPENAI_BASE_URL`. We register a single
/// `reincore` provider (merging into any existing config) and select its
/// `reincore-auto` model. The proxy resolves the real model itself, so the
/// model id here is purely a stable handle for OpenCode's `/models` picker.
///
/// npm package depends on the wire protocol the proxy is serving for this
/// provider: `/v1/responses` -> `@ai-sdk/openai`, otherwise
/// `/v1/chat/completions` -> `@ai-sdk/openai-compatible`.
fn activate_opencode_profile(
    proxy_url: &str,
    mode: &str,
    provider_header: Option<&str>,
) -> Result<()> {
    let npm = if mode == "responses" {
        "@ai-sdk/openai"
    } else {
        "@ai-sdk/openai-compatible"
    };
    activate_generic("opencode", &opencode_config_path(), |current| {
        let mut config: serde_json::Value = if current.trim().is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_str(&current).unwrap_or_else(|_| serde_json::json!({}))
        };
        if !config.is_object() {
            config = serde_json::json!({});
        }

        let obj = config.as_object_mut().expect("just ensured object");
        obj.entry("$schema")
            .or_insert_with(|| serde_json::json!("https://opencode.ai/config.json"));

        let provider = obj
            .entry("provider")
            .or_insert_with(|| serde_json::json!({}));
        if !provider.is_object() {
            *provider = serde_json::json!({});
        }
        let mut reincore_headers = serde_json::json!({
            "X-ReinCore-Client": "opencode"
        });
        if let Some(header) = provider_header {
            reincore_headers["X-ReinCore-Provider"] = serde_json::json!(header);
        }
        provider
            .as_object_mut()
            .expect("just ensured object")
            .insert(
                "reincore".to_string(),
                serde_json::json!({
                    "npm": npm,
                    "name": "ReinCore",
                    "options": {
                        "baseURL": proxy_url,
                        "apiKey": "reincore",
                        "headers": reincore_headers
                    },
                    "models": {
                        "reincore-auto": { "name": "reincore (auto)" }
                    }
                }),
            );

        obj.insert(
            "model".to_string(),
            serde_json::json!("reincore/reincore-auto"),
        );

        serde_json::to_string_pretty(&config)
            .map_err(|e| anyhow::anyhow!("failed to serialize opencode.json: {}", e))
    })
}

fn restore_opencode_profile() -> Result<bool> {
    restore_generic("opencode", &opencode_config_path())
}

/// OpenClaw's Gateway config: `~/.openclaw/openclaw.json` (honoring
/// `OPENCLAW_CONFIG_PATH`). Same layout on every OS - OpenClaw does not use
/// %APPDATA%.
fn openclaw_config_path() -> PathBuf {
    std::env::var("OPENCLAW_CONFIG_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".openclaw")
                .join("openclaw.json")
        })
}

/// OpenClaw reaches custom OpenAI-compatible endpoints through a
/// `models.providers.<id>` entry in openclaw.json - it does NOT read
/// OPENAI_BASE_URL / ANTHROPIC_BASE_URL for a custom base URL. We register a
/// single `reincore` provider (merging into any existing config, preserving
/// the user's channels/agents/etc.) on the OpenAI Chat Completions wire
/// (`api: "openai-completions"`, so the proxy is launched in "chat" mode) and
/// select `reincore/reincore-auto` as the primary model. The proxy resolves
/// the real model itself, so the model id here is just a stable handle for
/// OpenClaw's model picker.
///
/// OpenClaw validates its config strictly and refuses to boot on unknown keys,
/// so we only write documented schema fields (`models.mode`,
/// `models.providers.*`, and `agents.defaults.model.primary`).
fn activate_openclaw_profile(proxy_url: &str, provider_header: Option<&str>) -> Result<()> {
    activate_generic("openclaw", &openclaw_config_path(), |current| {
        let mut config: serde_json::Value = if current.trim().is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_str(&current).unwrap_or_else(|_| serde_json::json!({}))
        };
        if !config.is_object() {
            config = serde_json::json!({});
        }
        let obj = config.as_object_mut().expect("just ensured object");

        // agents.defaults.model.primary = "reincore/reincore-auto"
        let agents = obj.entry("agents").or_insert_with(|| serde_json::json!({}));
        if !agents.is_object() {
            *agents = serde_json::json!({});
        }
        let defaults = agents
            .as_object_mut()
            .expect("just ensured object")
            .entry("defaults")
            .or_insert_with(|| serde_json::json!({}));
        if !defaults.is_object() {
            *defaults = serde_json::json!({});
        }
        let model = defaults
            .as_object_mut()
            .expect("just ensured object")
            .entry("model")
            .or_insert_with(|| serde_json::json!({}));
        if !model.is_object() {
            *model = serde_json::json!({});
        }
        model.as_object_mut().expect("just ensured object").insert(
            "primary".to_string(),
            serde_json::json!("reincore/reincore-auto"),
        );

        // models.mode = "merge"; models.providers.reincore = { ... }
        let models = obj.entry("models").or_insert_with(|| serde_json::json!({}));
        if !models.is_object() {
            *models = serde_json::json!({});
        }
        let models_obj = models.as_object_mut().expect("just ensured object");
        models_obj.insert("mode".to_string(), serde_json::json!("merge"));
        let providers = models_obj
            .entry("providers")
            .or_insert_with(|| serde_json::json!({}));
        if !providers.is_object() {
            *providers = serde_json::json!({});
        }
        let mut reincore_headers = serde_json::json!({
            "X-ReinCore-Client": "openclaw"
        });
        if let Some(header) = provider_header {
            reincore_headers["X-ReinCore-Provider"] = serde_json::json!(header);
        }
        providers
            .as_object_mut()
            .expect("just ensured object")
            .insert(
                "reincore".to_string(),
                serde_json::json!({
                    "baseUrl": proxy_url,
                    "apiKey": "reincore",
                    "api": "openai-completions",
                    "headers": reincore_headers,
                    "models": [
                        { "id": "reincore-auto", "name": "ReinCore (auto)" }
                    ]
                }),
            );

        serde_json::to_string_pretty(&config)
            .map_err(|e| anyhow::anyhow!("failed to serialize openclaw.json: {}", e))
    })
}

fn restore_openclaw_profile() -> Result<bool> {
    restore_generic("openclaw", &openclaw_config_path())
}

/// Gemini CLI's user config lives in `~/.gemini/`. The base URL is set via
/// the `GOOGLE_GEMINI_BASE_URL` env var (which the CLI resolves from
/// `~/.gemini/.env` on startup), and authentication via `GEMINI_API_KEY`.
/// Both can reference env-var interpolation from `settings.json`, but the
/// simplest, most reliable hook is to write them into the `.env` file the
/// CLI loads automatically - see the Gemini CLI configuration docs:
/// https://geminicli.com/docs/reference/configuration#environment-variables-and-env-files
///
/// We merge into the existing `.env` (preserving any other keys) rather than
/// overwriting it, and back up the original via `activate_generic` so
/// `reincore gemini --restore` reverts cleanly.
fn gemini_env_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".gemini")
        .join(".env")
}

/// Set (or add) a `KEY=value` line in a `.env` file, replacing any existing
/// line that starts with `KEY=`. Matches the dotenv format the Gemini CLI
/// loads - no quoting needed for URLs or the placeholder key.
fn set_env_line(current: &str, key: &str, value: &str) -> String {
    let prefix = format!("{}=", key);
    let new_line = format!("{}={}", key, value);
    let mut replaced = false;
    let mut out = String::new();
    for line in current.lines() {
        if line.starts_with(&prefix) {
            out.push_str(&new_line);
            out.push('\n');
            replaced = true;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    if !replaced {
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&new_line);
        out.push('\n');
    }
    out
}

/// Activate a persistent reincore profile for Gemini CLI by writing
/// `GOOGLE_GEMINI_BASE_URL` and `GEMINI_API_KEY` into `~/.gemini/.env`.
fn activate_gemini_profile(proxy_base_url: &str, provider_header: Option<&str>) -> Result<()> {
    activate_generic("gemini", &gemini_env_path(), |current| {
        let updated = set_env_line(&current, "GOOGLE_GEMINI_BASE_URL", proxy_base_url);
        let updated = set_env_line(&updated, "GEMINI_API_KEY", "reincore");
        // Multi-provider routing: the Gemini CLI parses custom request
        // headers from GEMINI_CLI_CUSTOM_HEADERS (comma-separated
        // `Name: value` pairs).
        let updated = match provider_header {
            Some(header) => set_env_line(
                &updated,
                "GEMINI_CLI_CUSTOM_HEADERS",
                &format!("X-ReinCore-Provider: {}", header),
            ),
            None => updated,
        };
        Ok(updated)
    })
}

fn restore_gemini_profile() -> Result<bool> {
    restore_generic("gemini", &gemini_env_path())
}

/// Whether a persistent profile was actually written/restored for this app.
pub enum ProfileStatus {
    Handled,
    Unsupported,
}

/// Create + activate a persistent reincore profile for apps we have a
/// verified, safe config mechanism for. Apps not listed here (Cursor,
/// Windsurf, Copilot, Continue) don't have a config file we're
/// confident is both discoverable and safe to script - Cursor/Windsurf's
/// custom-endpoint support is GUI-settings-only (and for Cursor, only covers
/// plan mode, not the coding agent), and Copilot doesn't support a custom
/// base URL at all. Those apps keep using the existing session-scoped env var
/// launch only.
pub fn activate_profile_for_app(
    app: &str,
    proxy_base_url: &str,
    proxy_url: &str,
    provider_label: &str,
    mode: &str,
    provider_header: Option<&str>,
) -> Result<ProfileStatus> {
    match app.to_lowercase().as_str() {
        "codex" | "codex-app" => {
            activate_codex_profile(proxy_url, provider_label, mode, provider_header)?;
            Ok(ProfileStatus::Handled)
        }
        "open-code" | "opencode" | "open code" => {
            activate_opencode_profile(proxy_url, mode, provider_header)?;
            Ok(ProfileStatus::Handled)
        }
        "openclaw" | "open-claw" | "open claw" => {
            activate_openclaw_profile(proxy_url, provider_header)?;
            Ok(ProfileStatus::Handled)
        }
        "claude" => {
            activate_claude_profile(proxy_base_url, provider_header)?;
            Ok(ProfileStatus::Handled)
        }
        "aider" => {
            activate_aider_profile(proxy_url, provider_header)?;
            Ok(ProfileStatus::Handled)
        }
        // Gemini CLI reads `GOOGLE_GEMINI_BASE_URL` + `GEMINI_API_KEY` from
        // `~/.gemini/.env` on startup, so we persist them there rather than
        // relying on session-scoped env vars. Falls back to the session-scoped
        // env vars injected by `app_env_vars` for `reincore launch gemini`.
        "gemini" | "gemini-cli" => {
            activate_gemini_profile(proxy_base_url, provider_header)?;
            Ok(ProfileStatus::Handled)
        }
        _ => Ok(ProfileStatus::Unsupported),
    }
}

/// Revert whatever `activate_profile_for_app` changed, for `reincore <app>
/// --restore`.
pub fn restore_profile_for_app(app: &str) -> Result<ProfileStatus> {
    match app.to_lowercase().as_str() {
        "codex" | "codex-app" => Ok(if restore_codex_profile()? {
            ProfileStatus::Handled
        } else {
            ProfileStatus::Unsupported
        }),
        "open-code" | "opencode" | "open code" => Ok(if restore_opencode_profile()? {
            ProfileStatus::Handled
        } else {
            ProfileStatus::Unsupported
        }),
        "openclaw" | "open-claw" | "open claw" => Ok(if restore_openclaw_profile()? {
            ProfileStatus::Handled
        } else {
            ProfileStatus::Unsupported
        }),
        "claude" => Ok(if restore_claude_profile()? {
            ProfileStatus::Handled
        } else {
            ProfileStatus::Unsupported
        }),
        "aider" => Ok(if restore_aider_profile()? {
            ProfileStatus::Handled
        } else {
            ProfileStatus::Unsupported
        }),
        "gemini" | "gemini-cli" => Ok(if restore_gemini_profile()? {
            ProfileStatus::Handled
        } else {
            ProfileStatus::Unsupported
        }),
        _ => Ok(ProfileStatus::Unsupported),
    }
}

/// Resolve the actual command to invoke in the launched terminal, preferring
/// a fully-qualified path over a bare alias name (a freshly spawned shell
/// doesn't necessarily see the same PATH the user's interactive shell does).
fn resolve_app_executable(app: &str) -> String {
    let aliases = app_command_aliases(app);

    for alias in &aliases {
        if let Ok(path) = which::which(alias) {
            return path.display().to_string();
        }
    }

    #[cfg(target_os = "windows")]
    {
        let app_lower = app.to_lowercase();
        if let Some(dirs) = windows_app_install_dirs().get(app_lower.as_str()) {
            for dir in dirs {
                for alias in &aliases {
                    for ext in ["exe", "cmd", "bat"] {
                        let candidate = dir.join(format!("{}.{}", alias, ext));
                        if candidate.exists() {
                            return candidate.display().to_string();
                        }
                    }
                }
            }
        }
    }

    #[cfg(unix)]
    {
        let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
        for alias in &aliases {
            if let Ok(output) = std::process::Command::new(&shell)
                .arg("-lc")
                .arg(format!("command -v {}", alias))
                .output()
            {
                if output.status.success() {
                    let resolved = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !resolved.is_empty() {
                        return resolved;
                    }
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        for alias in &aliases {
            if let Ok(output) = std::process::Command::new("where").arg(alias).output() {
                if output.status.success() {
                    if let Some(first_line) = String::from_utf8_lossy(&output.stdout).lines().next()
                    {
                        let resolved = first_line.trim();
                        if !resolved.is_empty() {
                            return resolved.to_string();
                        }
                    }
                }
            }
        }
    }

    aliases
        .first()
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| app.to_string())
}

/// Build a unique path in the system temp dir for a one-shot launch script.
fn unique_script_path(app: &str, ext: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let filename = format!(
        "reincore_launch_{}_{}_{}.{}",
        app.to_lowercase(),
        std::process::id(),
        nanos,
        ext
    );
    std::env::temp_dir().join(filename)
}

/// Escape a value for safe inclusion in a POSIX shell command string.
#[cfg(not(target_os = "windows"))]
fn shell_escape(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// Write a one-shot PowerShell script that sets the app's env vars, runs it,
/// and then pauses so the window stays open. Written to a file and referenced
/// with a single `-File` argument - rather than an inline
/// `"$env:X = 'a'; $env:Y = 'b'; app"` `-Command` string - because `wt.exe`
/// parses its own argument list for unescaped `;` to split multiple actions,
/// which fractures one launch into several broken tabs. A single `-File`
/// argument has no `;` in it at all.
#[cfg(target_os = "windows")]
fn write_windows_launch_script(
    app: &str,
    command_name: &str,
    env_vars: &[(&'static str, String)],
    unset_vars: &[&'static str],
    working_dir: Option<&Path>,
    extra_args: &[String],
) -> Result<PathBuf> {
    let mut script = String::new();
    // Change to the working directory before launching the app, if specified.
    if let Some(dir) = working_dir {
        script.push_str(&format!(
            "Set-Location '{}'\n",
            dir.display().to_string().replace('\'', "''")
        ));
    }
    // Save the original values of env vars we're about to unset/override so
    // they can be restored after the app exits (e.g. ANTHROPIC_API_KEY,
    // ANTHROPIC_AUTH_TOKEN). This prevents reincore from clobbering the
    // user's real credentials permanently.
    for key in unset_vars {
        let k = key.trim();
        script.push_str(&format!("$__reincore_backup_{} = $env:{}\n", k, k));
    }
    for (key, value) in env_vars {
        let k = key.trim();
        script.push_str(&format!("$__reincore_backup_{} = $env:{}\n", k, k));
        script.push_str(&format!("$env:{} = '{}'\n", k, value.replace('\'', "''")));
    }
    // Call operator `&` is required to execute a string; works for both a
    // bare alias like `codex` and a fully-qualified path with spaces.
    // Per-app extra args (e.g. Muse Code's --base-url) follow the command.
    let mut invocation = format!("& '{}'", command_name.trim().replace('\'', "''"));
    for arg in extra_args {
        invocation.push_str(&format!(" '{}'", arg.replace('\'', "''")));
    }
    invocation.push('\n');
    script.push_str(&invocation);
    // Restore the original env var values after the app exits.
    script.push_str("Write-Host \"\"\n");
    for key in unset_vars {
        let k = key.trim();
        script.push_str(&format!(
            "if ($null -eq $__reincore_backup_{k}) {{ Remove-Item Env:{k} -ErrorAction SilentlyContinue }} else {{ $env:{k} = $__reincore_backup_{k} }}\n",
            k = k
        ));
    }
    for (key, _) in env_vars {
        let k = key.trim();
        script.push_str(&format!(
            "if ($null -eq $__reincore_backup_{k}) {{ Remove-Item Env:{k} -ErrorAction SilentlyContinue }} else {{ $env:{k} = $__reincore_backup_{k} }}\n",
            k = k
        ));
    }
    script.push_str("Read-Host \"Press Enter to close this window\" | Out-Null\n");

    let path = unique_script_path(app, "ps1");
    std::fs::write(&path, script)
        .map_err(|e| anyhow::anyhow!("failed to write launch script {}: {}", path.display(), e))?;
    Ok(path)
}

#[cfg(target_os = "windows")]
fn launch_windows_terminal(app: &str, script_path: &Path) -> Result<()> {
    let mut wt_cmd = std::process::Command::new("wt.exe");
    wt_cmd
        .arg("new-tab")
        .arg("--title")
        .arg(format!("{} (reincore)", app))
        .arg("powershell.exe")
        .arg("-NoExit")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-File")
        .arg(script_path);

    if wt_cmd.spawn().is_ok() {
        return Ok(());
    }

    std::process::Command::new("cmd")
        .arg("/c")
        .arg("start")
        .arg(format!("{} (reincore)", app))
        .arg("powershell.exe")
        .arg("-NoExit")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-File")
        .arg(script_path)
        .spawn()
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!("failed to open new console window: {}", e))
}

/// Write a one-shot POSIX shell script, for the same reason as its Windows
/// counterpart: a single referenced file sidesteps nested-quoting issues.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn write_posix_launch_script(
    app: &str,
    command_name: &str,
    env_vars: &[(&'static str, String)],
    unset_vars: &[&'static str],
    working_dir: Option<&Path>,
    extra_args: &[String],
) -> Result<PathBuf> {
    let mut script = String::from("#!/bin/sh\n");
    // Change to the working directory before launching the app, if specified.
    if let Some(dir) = working_dir {
        script.push_str(&format!(
            "cd {}\n",
            shell_escape(&dir.display().to_string())
        ));
    }
    // Save the original values of env vars we're about to unset/override so
    // they can be restored after the app exits.
    for key in unset_vars {
        let k = key.trim();
        script.push_str(&format!("__reincore_backup_{}=\"${}\"\n", k, k));
        script.push_str(&format!("unset {}\n", k));
    }
    for (key, value) in env_vars {
        let k = key.trim();
        script.push_str(&format!("__reincore_backup_{}=\"${}\"\n", k, k));
        script.push_str(&format!("export {}={}\n", k, shell_escape(value)));
    }
    script.push_str(&shell_escape(command_name));
    for arg in extra_args {
        script.push(' ');
        script.push_str(&shell_escape(arg));
    }
    script.push('\n');
    // Restore the original env var values after the app exits.
    script.push_str("echo\n");
    for key in unset_vars {
        let k = key.trim();
        script.push_str(&format!(
            "if [ -n \"$__reincore_backup_{k}\" ]; then export {k}=\"$__reincore_backup_{k}\"; else unset {k}; fi\n",
            k = k
        ));
    }
    for (key, _) in env_vars {
        let k = key.trim();
        script.push_str(&format!(
            "if [ -n \"$__reincore_backup_{k}\" ]; then export {k}=\"$__reincore_backup_{k}\"; else unset {k}; fi\n",
            k = k
        ));
    }
    script.push_str("printf 'Press Enter to close this window...'\nread _\n");

    let ext = if cfg!(target_os = "macos") {
        "command"
    } else {
        "sh"
    };
    let path = unique_script_path(app, ext);
    std::fs::write(&path, script)
        .map_err(|e| anyhow::anyhow!("failed to write launch script {}: {}", path.display(), e))?;

    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(&path)
        .map_err(|e| anyhow::anyhow!("failed to stat launch script: {}", e))?
        .permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms)
        .map_err(|e| anyhow::anyhow!("failed to chmod launch script: {}", e))?;

    Ok(path)
}

#[cfg(target_os = "linux")]
fn launch_linux_terminal(script_path: &Path) -> Result<()> {
    let terminals = [
        "gnome-terminal",
        "konsole",
        "xfce4-terminal",
        "alacritty",
        "kitty",
        "wezterm",
        "xterm",
    ];
    for term in &terminals {
        if which::which(term).is_err() {
            continue;
        }
        let mut cmd = std::process::Command::new(term);
        match *term {
            "gnome-terminal" | "xfce4-terminal" => {
                cmd.arg("--").arg("bash").arg(script_path);
            }
            _ => {
                cmd.arg("-e").arg("bash").arg(script_path);
            }
        }
        if cmd.spawn().is_ok() {
            return Ok(());
        }
    }
    Err(anyhow::anyhow!(
        "could not find a supported terminal emulator on Linux"
    ))
}

/// Spawn the target CLI app in a new terminal tab/window so reincore can keep
/// running in the original terminal and the user can still see app logs.
/// If `working_dir` is given, the app is launched from that directory.
/// When `provider_header` is set (multi-provider daemon), the app is
/// configured to send `X-ReinCore-Provider: <label>` on every request.
pub fn spawn_app_in_terminal(
    app: &str,
    proxy_base_url: &str,
    proxy_url: &str,
    working_dir: Option<&Path>,
    provider_header: Option<&str>,
) -> Result<()> {
    let command_name = resolve_app_executable(app);
    let env_vars = app_env_vars(app, proxy_base_url, proxy_url, provider_header);
    let unset_vars = app_env_vars_to_unset(app);
    let extra_args = app_extra_args(app, proxy_base_url);

    #[cfg(target_os = "windows")]
    {
        let script_path = write_windows_launch_script(
            app,
            &command_name,
            &env_vars,
            &unset_vars,
            working_dir,
            &extra_args,
        )?;
        launch_windows_terminal(app, &script_path)?;
    }

    #[cfg(target_os = "macos")]
    {
        let script_path = write_posix_launch_script(
            app,
            &command_name,
            &env_vars,
            &unset_vars,
            working_dir,
            &extra_args,
        )?;
        std::process::Command::new("open")
            .arg("-a")
            .arg("Terminal")
            .arg(&script_path)
            .spawn()
            .map_err(|e| anyhow::anyhow!("failed to open Terminal.app: {}", e))?;
    }

    #[cfg(target_os = "linux")]
    {
        let script_path = write_posix_launch_script(
            app,
            &command_name,
            &env_vars,
            &unset_vars,
            working_dir,
            &extra_args,
        )?;
        launch_linux_terminal(&script_path)?;
    }

    Ok(())
}

/// Return a short, OS-specific install hint for a known app, or `None` if we
/// don't have one. The hint is printed when `is_app_installed` reports the
/// app is missing so the user knows how to get it rather than staring at a
/// bare "not installed" warning.
pub fn install_hint_for_app(app: &str) -> Option<String> {
    let web = |url: &str| format!("See {} for installation instructions.", url);
    let (windows, macos, linux, url): (Option<&str>, Option<&str>, Option<&str>, &str) =
        match app.to_lowercase().as_str() {
            "claude" => (
                Some("irm https://claude.ai/install.ps1 | iex"),
                Some("curl -fsSL https://claude.ai/install.sh | bash"),
                Some("curl -fsSL https://claude.ai/install.sh | bash"),
                "https://claude.ai/download",
            ),
            "codex" | "codex-app" => (
                Some("npm install -g @openai/codex"),
                Some("npm install -g @openai/codex"),
                Some("npm install -g @openai/codex"),
                "https://github.com/openai/codex",
            ),
            "open-code" | "opencode" | "open code" => (
                Some("npm install -g opencode-ai"),
                Some("brew install anomalyco/tap/opencode"),
                Some("curl -fsSL https://opencode.ai/install | bash"),
                "https://opencode.ai/docs/",
            ),
            "openclaw" | "open-claw" | "open claw" => (
                Some("iwr -useb https://openclaw.ai/install.ps1 | iex"),
                Some("curl -fsSL https://openclaw.ai/install.sh | bash"),
                Some("curl -fsSL https://openclaw.ai/install.sh | bash"),
                "https://docs.openclaw.ai/install",
            ),
            "copilot" => (
                Some("gh extension install github/gh-copilot"),
                Some("gh extension install github/gh-copilot"),
                Some("gh extension install github/gh-copilot"),
                "https://docs.github.com/copilot",
            ),
            "gemini" | "gemini-cli" => (
                Some("npm install -g @google/gemini-cli"),
                Some("npm install -g @google/gemini-cli"),
                Some("npm install -g @google/gemini-cli"),
                "https://geminicli.com/docs/get-started/installation",
            ),
            "cursor" => (None, None, None, "https://cursor.com/download"),
            "windsurf" => (None, None, None, "https://windsurf.com/download"),
            "aider" => (
                Some("pipx install aider-chat"),
                Some("pipx install aider-chat"),
                Some("pipx install aider-chat"),
                "https://aider.chat/docs/install.html",
            ),
            "continue" => (None, None, None, "https://continue.dev"),
            "ollama" => (
                Some("winget install Ollama.Ollama"),
                Some("brew install ollama"),
                Some("curl -fsSL https://ollama.com/install.sh | sh"),
                "https://ollama.com/download",
            ),
            _ => return None,
        };

    let cmd = if cfg!(target_os = "windows") {
        windows
    } else if cfg!(target_os = "macos") {
        macos
    } else {
        linux
    };

    Some(match cmd {
        Some(c) => format!("Install it with: {}\n{}", c, web(url)),
        None => web(url),
    })
}

/// Map a provider label to the environment variable its API key is expected
/// to live in.
fn api_key_env_var(provider: &str) -> String {
    match provider.to_lowercase().as_str() {
        "anthropic" => "ANTHROPIC_API_KEY".to_string(),
        "openai" => "OPENAI_API_KEY".to_string(),
        "copilot" => "COPILOT_CLIENT_ID".to_string(), // Optional override; defaults are built in
        other => format!("{}_API_KEY", other.to_uppercase()),
    }
}

/// Make sure an API key is available for the given provider before the proxy
/// starts. If one isn't already set, prompt the user for it interactively.
///
/// NOTE: this assumes `provider_start_proxy` sources credentials from an
/// environment variable named `<PROVIDER>_API_KEY`. If keys are actually read
/// from `db_path` / a config file instead, point this at that instead.
/// Public wrapper for `ensure_api_key` so `start --provider` can reuse the
/// same API-key resolution logic as `launch`.
pub async fn ensure_api_key_pub(provider_label: &str) -> Result<()> {
    ensure_api_key(provider_label).await
}

/// Public wrapper for `setup_ollama_provider` so `start --provider ollama`
/// can reuse the same Ollama local/cloud setup logic as `launch`.
pub async fn setup_ollama_provider_pub(
    cloud: bool,
    model: Option<String>,
    interactive_tiers: bool,
) -> Result<std::sync::Arc<dyn engine_core::Provider>> {
    setup_ollama_provider(cloud, model, interactive_tiers).await
}

async fn ensure_api_key(provider_label: &str) -> Result<()> {
    // Copilot uses built-in OAuth credentials - no API key needed.
    if provider_label.to_lowercase() == "copilot" {
        return Ok(());
    }

    let env_var = api_key_env_var(provider_label);

    // Check if the key is already in the environment.
    if std::env::var(&env_var)
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
    {
        return Ok(());
    }

    // Check the credential store for a saved key.
    if let Some(cred) = credentials::get_credential(provider_label) {
        if cred.kind == CredentialKind::ApiKey {
            std::env::set_var(&env_var, &cred.value);
            println!(
                "{}",
                log::info(&format!(
                    "Using saved {} for provider '{}'.",
                    env_var, provider_label
                ))
            );
            return Ok(());
        }
    }

    // No key found - prompt the user.
    println!(
        "{}",
        log::warning(&format!(
            "No {} found for provider '{}'.",
            env_var, provider_label
        ))
    );

    // Use a provider-specific prompt message.
    let prompt = format!("Enter your {} API key: ", provider_label);
    print!("{}", prompt);
    use std::io::Write;
    std::io::stdout().flush().ok();

    let key = tokio::task::spawn_blocking(|| -> Result<String> {
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|e| anyhow::anyhow!("failed to read API key from stdin: {}", e))?;
        Ok(input.trim().to_string())
    })
    .await??;

    if key.is_empty() {
        return Err(anyhow::anyhow!(
            "no API key entered; cannot start the proxy for provider '{}'.",
            provider_label
        ));
    }

    // Set it in the environment for this session.
    std::env::set_var(&env_var, &key);
    println!(
        "{}",
        log::success(&format!("{} set for this session.", env_var))
    );

    // Save it to the credential store for future launches.
    if let Err(e) = credentials::store_api_key(provider_label, &key) {
        eprintln!(
            "{}",
            log::warning(&format!(
                "could not save API key to credential store: {}",
                e
            ))
        );
    } else {
        println!(
            "{}",
            log::success("API key saved to ~/.reincore-io/credentials.json for future use.")
        );
    }

    Ok(())
}

/// Poll until the proxy is actually accepting TCP connections, or timeout.
fn wait_for_proxy_ready(host: &str, port: u16, timeout: Duration) -> bool {
    let addr = format!("{}:{}", host, port);
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if TcpStream::connect(&addr).is_ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

/// Project name for a launch working directory: the latest folder name.
/// Falls back to `"reincore"` for roots or non-UTF-8 names.
pub fn launch_project_name(path: &Path) -> String {
    path.file_name()
        .and_then(|part| part.to_str())
        .filter(|part| !part.is_empty())
        .unwrap_or("reincore")
        .to_string()
}

/// Register the launch working directory as a local project on the running
/// daemon so sessions opened from the launched app attribute to it. The
/// project name is the latest folder name. A `409 CONFLICT` (already
/// assigned) counts as success; any other failure is returned so the caller
/// can warn without aborting the launch.
pub async fn ensure_launch_project(host: &str, port: u16, path: &Path) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap_or_default();
    let response = client
        .post(format!(
            "http://{}:{}/v1/local-projects",
            host.trim_end_matches('/'),
            port
        ))
        .json(&serde_json::json!({
            "name": launch_project_name(path),
            "project_path": path.to_string_lossy(),
        }))
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("could not reach the daemon to register the project: {}", e))?;
    if response.status().is_success()
        || response.status() == reqwest::StatusCode::CONFLICT
    {
        return Ok(());
    }
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    Err(anyhow::anyhow!(
        "daemon refused to register the project ({}): {}",
        status,
        body.chars().take(300).collect::<String>()
    ))
}

/// Validate or create the working directory for launching an app.
/// If the path doesn't exist, prompt the user interactively to create it.
/// Returns the resolved path, or an error if the user declines / the path
/// is invalid.
async fn resolve_launch_path(path: &Option<String>) -> Result<Option<PathBuf>> {
    // When no --path is given, default to the current working directory of
    // the reincore process so the app launches where the user invoked reincore.
    let path = match path {
        Some(p) => p,
        None => {
            let cwd = std::env::current_dir()
                .map_err(|e| anyhow::anyhow!("failed to get current directory: {}", e))?;
            return Ok(Some(cwd));
        }
    };

    let expanded = shellexpand::tilde(path).to_string();
    let path_buf = PathBuf::from(&expanded);

    if path_buf.exists() {
        if !path_buf.is_dir() {
            return Err(anyhow::anyhow!("'{}' exists but is not a directory.", path));
        }
        return Ok(Some(path_buf));
    }

    // Path doesn't exist - ask the user if they want to create it.
    println!("The path '{}' does not exist.", path);
    print!("Create it? [y/N] ");
    use std::io::Write;
    std::io::stdout().flush().ok();

    let answer = tokio::task::spawn_blocking(|| -> Result<String> {
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|e| anyhow::anyhow!("failed to read input: {}", e))?;
        Ok(input.trim().to_lowercase())
    })
    .await??;

    if answer == "y" || answer == "yes" {
        std::fs::create_dir_all(&path_buf)
            .map_err(|e| anyhow::anyhow!("failed to create directory '{}': {}", path, e))?;
        println!("Created '{}'.", path);
        Ok(Some(path_buf))
    } else {
        Err(anyhow::anyhow!(
            "path '{}' does not exist and was not created. Aborting.",
            path
        ))
    }
}

/// Set up reincore with an app without launching it. Writes the persistent
/// config so the user can launch the app manually from any directory.
pub async fn handle_setup(args: SetupArgs) -> Result<()> {
    let SetupArgs {
        app,
        provider,
        mode,
        host,
        port,
    } = args;

    let wire_api = match (mode.to_lowercase().as_str(), app.to_lowercase().as_str()) {
        (_, "codex" | "codex-app" | "open-code" | "opencode" | "open code" | "muse") => "responses",
        // OpenClaw only speaks the OpenAI Chat Completions wire
        // (`api: "openai-completions"`); it has no /v1/responses client.
        (_, "openclaw" | "open-claw" | "open claw") => "chat",
        ("chat", _) => "chat",
        ("agentic" | "agent" | "responses", _) => "responses",
        (other, _) => {
            return Err(anyhow::anyhow!(
                "unknown mode '{}'. Use 'agentic' (default) or 'chat'.",
                other
            ));
        }
    };

    let provider_label = resolve_provider(&app, provider)?;
    let proxy_base_url = format!("http://{}:{}", host, port);
    let proxy_url = format!("{}/v1", proxy_base_url);

    if !is_app_installed(&app) {
        eprintln!(
            "{}",
            log::warning(&format!(
                "'{}' does not appear to be installed on this system.",
                app
            ))
        );
        if let Some(hint) = install_hint_for_app(&app) {
            eprintln!("  {}", hint.dimmed());
        }
    }

    // `reincore setup` configures the app for whatever daemon is running
    // (or, if none is running, for the single-provider daemon a later
    // `reincore launch` would start). Probe the daemon so multi-provider
    // routing gets the X-ReinCore-Provider header baked into the profile.
    let provider_header = match crate::launch_routing::probe_daemon_state(&host, port).await? {
        Some(state)
            if state.is_multi_provider() && state.has_provider(&provider_label.to_lowercase()) =>
        {
            Some(provider_label.to_lowercase())
        }
        _ => None,
    };

    match activate_profile_for_app(
        &app,
        &proxy_base_url,
        &proxy_url,
        &provider_label,
        wire_api,
        provider_header.as_deref(),
    ) {
        Ok(ProfileStatus::Handled) => {
            println!(
                "{}",
                log::success(&format!(
                    "Set up a persistent reincore profile for '{}'. The app will route through reincore at {} when launched.",
                    app, proxy_url
                ))
            );
            println!(
                "  {}",
                format!(
                    "Run `reincore {} --restore` (or `reincore restore {}`) to revert.",
                    app, app
                )
                .dimmed()
            );
            if app.to_lowercase() == "claude" {
                println!(
                    "  {}",
                    "Tip: if Claude Code shows a model error, run /logout inside it to clear any saved login.".yellow()
                );
            }
            Ok(())
        }
        Ok(ProfileStatus::Unsupported) => {
            println!(
                "{}",
                log::info(&format!(
                    "'{}' doesn't have a persistent reincore profile. Use `reincore launch {}` to route a session through reincore.",
                    app, app
                ))
            );
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Prompt the user to select their GitHub account type and return the
/// appropriate instance URL. For GitHub Enterprise Cloud with data residency,
/// the user provides their instance URL.
async fn prompt_github_instance() -> Result<String> {
    // If the env var is already set, use it without prompting.
    if let Ok(instance) = std::env::var("COPILOT_GITHUB_INSTANCE") {
        return Ok(instance);
    }

    println!(
        "\n┌─── {} ─────────────────────────",
        "Select GitHub Account Type".cyan().bold()
    );
    println!("│  1. {}", "GitHub (github.com)".white());
    println!(
        "│  2. {}",
        "GitHub Enterprise Cloud with data residency (*.ghe.com)".white()
    );
    println!("└────────────────────────────────────────────────────────");
    print!("Enter choice [1 or 2, default 1]: ");
    use std::io::Write;
    std::io::stdout().flush().ok();

    let answer = tokio::task::spawn_blocking(|| -> Result<String> {
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|e| anyhow::anyhow!("failed to read input: {}", e))?;
        Ok(input.trim().to_lowercase())
    })
    .await??;

    if answer == "2" || answer == "enterprise" || answer == "ghe" {
        print!("Enter your GitHub Enterprise instance URL (e.g. https://enterprise.ghe.com): ");
        std::io::stdout().flush().ok();

        let instance = tokio::task::spawn_blocking(|| -> Result<String> {
            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .map_err(|e| anyhow::anyhow!("failed to read input: {}", e))?;
            Ok(input.trim().to_string())
        })
        .await??;

        if instance.is_empty() {
            return Err(anyhow::anyhow!(
                "GitHub Enterprise instance URL is required"
            ));
        }

        // Ensure it starts with https://
        let instance = if instance.starts_with("http") {
            instance.trim_end_matches('/').to_string()
        } else {
            format!("https://{}", instance.trim_end_matches('/'))
        };

        println!(
            "{} {}",
            "Using GitHub Enterprise instance:".dimmed(),
            instance.green()
        );
        Ok(instance)
    } else {
        Ok("https://github.com".to_string())
    }
}

/// Log out from a provider by removing stored credentials.
pub fn handle_logout(provider: &str) -> Result<()> {
    let removed = credentials::remove_credential(provider)?;
    if removed {
        println!(
            "{}",
            log::success(&format!(
                "Credentials for '{}' removed from ~/.reincore-io/credentials.json.",
                provider
            ))
        );
    } else {
        println!(
            "{}",
            log::warning(&format!("No stored credentials found for '{}'.", provider))
        );
    }
    Ok(())
}

// Ollama provider setup (local + cloud).
//
// `reincore launch ollama` (default) → local mode: detect the ollama binary,
// fetch installed models from the running server's `/api/tags`, and prompt
// the user to pick one. No API key, no model degradation (ollama models have
// no known per-token pricing in the catalog).
//
// `reincore launch ollama --cloud` → cloud mode: prompt for an API key and a
// default cloud model name, then validate the model against the endpoint's
// `/api/tags` listing before starting the proxy.

/// Default local Ollama server root URL.
const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434";

/// Fixed base URL for the Ollama Cloud API (see https://docs.ollama.com/cloud).
/// ollama.com acts as a remote Ollama host exposing the same `/api/chat` and
/// `/api/tags` endpoints as a local server, authenticated via
/// `Authorization: Bearer $OLLAMA_API_KEY`.
const OLLAMA_CLOUD_BASE_URL: &str = "https://ollama.com";

/// Public accessor for the Ollama Cloud base URL.
pub fn ollama_cloud_base_url() -> &'static str {
    OLLAMA_CLOUD_BASE_URL
}

/// Public wrappers for Ollama local-mode checks, used by `provider list`.
pub fn is_ollama_installed_pub() -> bool {
    is_ollama_installed()
}

pub async fn is_ollama_server_running_pub(base_url: &str) -> bool {
    is_ollama_server_running(base_url).await
}

pub fn default_ollama_base_url() -> &'static str {
    DEFAULT_OLLAMA_BASE_URL
}

/// Fetch the list of installed Ollama models from a local server or the
/// Ollama Cloud API. Returns an empty vector if the server is unreachable.
/// Used by `provider add ollama` to present a selection list.
pub async fn fetch_ollama_models_pub(
    base_url: &str,
    api_key: Option<&str>,
) -> Vec<OllamaModelInfo> {
    OllamaProvider::fetch_models_detailed(base_url, api_key)
        .await
        .unwrap_or_default()
}

/// Prompt the user to select a model from a numbered list. Returns the
/// selected model name, or an error if the list is empty or the user
/// enters an empty line to abort.
pub async fn prompt_select_ollama_model_pub(
    models: &[OllamaModelInfo],
    prompt: &str,
) -> Result<String> {
    prompt_select_model_detailed(models, prompt).await
}

/// Check whether the `ollama` binary is installed and on PATH.
fn is_ollama_installed() -> bool {
    which::which("ollama").is_ok()
}

/// Check whether a local Ollama server is reachable at `base_url` by
/// polling `/api/tags` for a short period.
async fn is_ollama_server_running(base_url: &str) -> bool {
    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap_or_default();
    for _ in 0..10 {
        if client
            .get(&url)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
        {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    false
}

/// Prompt the user to select a model from a list, returning the selected
/// model name. `prompt` is printed before the numbered list.
/// Render each Ollama model's parameter size alongside its name so the user
/// can pick a stronger model for a tier.
///
/// This loops until the user makes a valid selection so that setup cannot
/// proceed without explicitly choosing at least the standard model.
async fn prompt_select_model_detailed(models: &[OllamaModelInfo], prompt: &str) -> Result<String> {
    if models.is_empty() {
        return Err(anyhow::anyhow!("no models available to select"));
    }

    let read_line = || -> Result<String> {
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|e| anyhow::anyhow!("failed to read input: {}", e))?;
        Ok(input.trim().to_lowercase())
    };

    loop {
        println!("\n{}", prompt.cyan().bold());
        for (i, m) in models.iter().enumerate() {
            let size = m
                .parameter_size
                .as_deref()
                .map(|s| format!("  ({})", s))
                .unwrap_or_default();
            println!(
                "  {}. {}{}",
                format!("{}", i + 1).yellow(),
                m.name.white(),
                size.dimmed()
            );
        }
        print!("\nEnter choice [1-{}]: ", models.len());
        use std::io::Write;
        std::io::stdout().flush().ok();

        let answer = tokio::task::spawn_blocking(read_line).await??;

        if let Ok(idx) = answer.parse::<usize>() {
            if idx >= 1 && idx <= models.len() {
                return Ok(models[idx - 1].name.clone());
            }
        }
        if let Some(m) = models.iter().find(|m| m.name.to_lowercase() == answer) {
            return Ok(m.name.clone());
        }
        println!(
            "{}",
            "Invalid selection. Please enter a number from the list or the exact model name.".red()
        );
    }
}

/// Optionally configure per-tier Ollama models. Ollama has no per-model
/// pricing, so this is purely quality routing (bigger models for harder
/// requests) - the `parameter_size` shown next to each model guides the
/// choice. Unset tiers fall back to the standard model. Returns the explicit
/// tier mappings the user selected (empty when they keep a single model).
async fn prompt_optional_ollama_tiers(
    models: &[OllamaModelInfo],
    standard: &str,
) -> Result<Vec<engine_core::ProviderModelMapping>> {
    use std::io::Write;

    if models.is_empty() {
        return Ok(Vec::new());
    }

    print!(
        "\nConfigure different models per routing tier? Ollama has no per-model \
pricing, so this only affects quality routing (bigger models for harder \
tasks). [y/N]: "
    );
    std::io::stdout().flush().ok();
    let answer = tokio::task::spawn_blocking(|| -> Result<String> {
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|e| anyhow::anyhow!("failed to read input: {}", e))?;
        Ok(input.trim().to_lowercase())
    })
    .await??;
    if answer != "y" && answer != "yes" {
        return Ok(Vec::new());
    }

    let mut mappings = Vec::new();
    for (tier, tier_name) in [
        (engine_core::ModelTier::Lite, "Lite"),
        (engine_core::ModelTier::Pro, "Pro"),
        (engine_core::ModelTier::Ultra, "Ultra"),
    ] {
        println!(
            "\n{}",
            format!("{} tier (0 = use standard model '{}')", tier_name, standard)
                .cyan()
                .bold()
        );
        for (i, m) in models.iter().enumerate() {
            let size = m
                .parameter_size
                .as_deref()
                .map(|s| format!("  ({})", s))
                .unwrap_or_default();
            println!(
                "  {}. {}{}",
                format!("{}", i + 1).yellow(),
                m.name.white(),
                size.dimmed()
            );
        }
        print!("\nEnter choice [0-{}]: ", models.len());
        std::io::stdout().flush().ok();
        let choice = tokio::task::spawn_blocking(|| -> Result<String> {
            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .map_err(|e| anyhow::anyhow!("failed to read input: {}", e))?;
            Ok(input.trim().to_string())
        })
        .await??;
        if let Ok(idx) = choice.parse::<usize>() {
            if idx >= 1 && idx <= models.len() {
                mappings.push(engine_core::ProviderModelMapping {
                    tier,
                    provider_model: models[idx - 1].name.clone(),
                });
            }
            // 0 (or out-of-range) leaves the tier to fall back to standard.
        }
    }
    Ok(mappings)
}

/// Set up the Ollama provider in local mode. Detects the ollama binary,
/// verifies the server is running, fetches installed models, and prompts
/// the user to select one. Returns a ready-to-use provider.
async fn setup_ollama_local(
    model: Option<String>,
    interactive_tiers: bool,
) -> Result<std::sync::Arc<dyn engine_core::Provider>> {
    if !is_ollama_installed() {
        eprintln!("{}", log::error("Ollama is not installed on this system."));
        eprintln!(
            "  {}",
            "Install it with: winget install Ollama.Ollama  (Windows) / brew install ollama  (macOS) / curl -fsSL https://ollama.com/install.sh | sh  (Linux)".dimmed()
        );
        eprintln!(
            "  {}",
            "See https://ollama.com/download for installation instructions.".dimmed()
        );
        return Err(anyhow::anyhow!(
            "ollama binary not found; install Ollama and try again"
        ));
    }

    let base_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_BASE_URL.to_string());

    println!(
        "{}",
        log::info(&format!(
            "Checking for a running Ollama server at {} ...",
            base_url
        ))
    );
    if !is_ollama_server_running(&base_url).await {
        eprintln!(
            "{}",
            log::error(&format!(
                "No Ollama server reachable at {}. Start it with `ollama serve` (or launch the Ollama desktop app) and try again.",
                base_url
            ))
        );
        return Err(anyhow::anyhow!("ollama server not running at {}", base_url));
    }

    println!(
        "{}",
        log::info(&format!("Fetching installed models from {} ...", base_url))
    );
    let models = OllamaProvider::fetch_models_detailed(&base_url, None)
        .await
        .map_err(|e| anyhow::anyhow!("failed to fetch Ollama models: {}", e))?;

    if models.is_empty() {
        eprintln!(
            "{}",
            log::warning("No models installed. Pull one with `ollama pull <model>` and try again.")
        );
        eprintln!(
            "  {}",
            "Browse available models at https://ollama.com/library".dimmed()
        );
        return Err(anyhow::anyhow!("no Ollama models installed"));
    }

    let selected = match model.filter(|s| !s.trim().is_empty()) {
        Some(m) => {
            let m = m.trim().to_string();
            println!(
                "{}",
                log::info(&format!("Using configured Ollama model '{}'", m))
            );
            m
        }
        None => {
            // Use the saved standard model from ~/.reincore-io/config.toml if it
            // exists, so `reincore launch ollama` doesn't prompt every time.
            if let Some(saved) = read_provider_models("ollama")
                .standard
                .filter(|s| !s.trim().is_empty())
            {
                println!(
                    "{}",
                    log::info(&format!(
                        "Using saved Ollama standard model '{}' from config.toml",
                        saved
                    ))
                );
                saved
            } else {
                prompt_select_model_detailed(&models, "Select the standard Ollama model to use:")
                    .await?
            }
        }
    };

    println!(
        "{}",
        log::success(&format!("Using Ollama model '{}'", selected))
    );

    let tier_models = if interactive_tiers {
        prompt_optional_ollama_tiers(&models, &selected).await?
    } else {
        Vec::new()
    };

    // Persist the selected standard model so future launches (and the desktop
    // app) can treat Ollama as configured without prompting again.
    crate::local_config::set_provider_model_field("ollama", "standard", &selected).unwrap_or_else(
        |e| {
            eprintln!(
                "{}",
                log::warning(&format!("Could not save Ollama standard model: {}", e))
            );
        },
    );

    let config = engine_core::ProviderConfig {
        label: "ollama".to_string(),
        base_url,
        api_key: String::new(),
        standard_model: Some(selected),
        models: tier_models,
        ..Default::default()
    };

    let provider = OllamaProvider::new(config, OllamaMode::Local)?;
    Ok(std::sync::Arc::new(provider) as std::sync::Arc<dyn engine_core::Provider>)
}

/// Set up the Ollama provider in cloud mode. Prompts for an API key and a
/// default cloud model name, validates the model against the endpoint's
/// `/api/tags` listing, and returns a ready-to-use provider.
///
/// If `model` is provided via `--model`, it is validated directly; if it's
/// missing or invalid, the user is prompted to enter one interactively.
///
/// The Ollama Cloud API lives at a fixed base URL (`https://ollama.com`),
/// authenticated via `Authorization: Bearer $OLLAMA_API_KEY`. See
/// https://docs.ollama.com/cloud.
async fn setup_ollama_cloud(
    model: Option<String>,
    interactive_tiers: bool,
) -> Result<std::sync::Arc<dyn engine_core::Provider>> {
    let base_url = OLLAMA_CLOUD_BASE_URL.to_string();

    // Check for a saved API key first, then the OLLAMA_API_KEY env var, then
    // prompt interactively.
    let saved_cred = credentials::get_credential("ollama");
    let api_key = if let Some(cred) = &saved_cred {
        if cred.kind == CredentialKind::ApiKey && !cred.value.is_empty() {
            println!("{}", log::info("Using saved Ollama cloud API key."));
            cred.value.clone()
        } else {
            prompt_for_ollama_cloud_api_key().await?
        }
    } else if let Ok(key) = std::env::var("OLLAMA_API_KEY") {
        if !key.trim().is_empty() {
            println!(
                "{}",
                log::info("Using OLLAMA_API_KEY from the environment.")
            );
            key
        } else {
            prompt_for_ollama_cloud_api_key().await?
        }
    } else {
        prompt_for_ollama_cloud_api_key().await?
    };

    // Resolve the model: use --model if provided, otherwise prompt. If the
    // provided model fails validation, fall through to the interactive
    // prompt so the user can pick a valid one.
    let model = match model {
        Some(m) if !m.trim().is_empty() => {
            let m = m.trim().to_string();
            println!(
                "{}",
                log::info(&format!(
                    "Validating model '{}' against {} ...",
                    m, base_url
                ))
            );
            match OllamaProvider::validate_model(&base_url, Some(&api_key), &m).await {
                Ok(()) => {
                    println!("{}", log::success(&format!("Model '{}' is available.", m)));
                    m
                }
                Err(e) => {
                    println!(
                        "{}",
                        log::warning(&format!(
                            "Model '{}' is not available: {}. Falling back to interactive selection.",
                            m, e
                        ))
                    );
                    prompt_for_ollama_cloud_model(&base_url, &api_key).await?
                }
            }
        }
        _ => {
            // Use the saved standard model from ~/.reincore-io/config.toml if it
            // exists and is valid, so `reincore launch ollama --cloud` doesn't
            // prompt every time.
            if let Some(saved) = read_provider_models("ollama")
                .standard
                .filter(|s| !s.trim().is_empty())
            {
                println!(
                    "{}",
                    log::info(&format!(
                        "Validating saved Ollama standard model '{}' against {} ...",
                        saved, base_url
                    ))
                );
                match OllamaProvider::validate_model(&base_url, Some(&api_key), &saved).await {
                    Ok(()) => {
                        println!(
                            "{}",
                            log::success(&format!(
                                "Using saved Ollama standard model '{}' from config.toml",
                                saved
                            ))
                        );
                        saved
                    }
                    Err(e) => {
                        println!(
                            "{}",
                            log::warning(&format!(
                                "Saved model '{}' is not available: {}. Falling back to interactive selection.",
                                saved, e
                            ))
                        );
                        prompt_for_ollama_cloud_model(&base_url, &api_key).await?
                    }
                }
            } else {
                prompt_for_ollama_cloud_model(&base_url, &api_key).await?
            }
        }
    };

    // Offer per-tier model selection (quality routing). Best-effort: if the
    // model list can't be fetched, skip tiers and keep the single model.
    let tier_models = match OllamaProvider::fetch_models_detailed(&base_url, Some(&api_key)).await {
        Ok(detailed) => {
            if interactive_tiers {
                prompt_optional_ollama_tiers(&detailed, &model)
                    .await
                    .unwrap_or_default()
            } else {
                Vec::new()
            }
        }
        Err(_) => Vec::new(),
    };

    // Persist the selected default cloud model so future launches can reuse it
    // without prompting every time.
    crate::local_config::set_provider_model_field("ollama", "standard", &model).unwrap_or_else(
        |e| {
            eprintln!(
                "{}",
                log::warning(&format!("Could not save Ollama standard model: {}", e))
            );
        },
    );

    let config = engine_core::ProviderConfig {
        label: "ollama".to_string(),
        base_url,
        api_key,
        standard_model: Some(model),
        models: tier_models,
        ..Default::default()
    };

    let provider = OllamaProvider::new(config, OllamaMode::Cloud)?;
    Ok(std::sync::Arc::new(provider) as std::sync::Arc<dyn engine_core::Provider>)
}

/// Prompt the user for an Ollama cloud model name and validate it against
/// the endpoint's `/api/tags` listing. Re-prompts on invalid input until
/// the user enters a valid model or an empty line (which aborts).
async fn prompt_for_ollama_cloud_model(base_url: &str, api_key: &str) -> Result<String> {
    use std::io::Write;
    loop {
        print!("Enter the default cloud model name (e.g. glm-5.2:cloud): ");
        std::io::stdout().flush().ok();
        let model = tokio::task::spawn_blocking(|| -> Result<String> {
            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .map_err(|e| anyhow::anyhow!("failed to read input: {}", e))?;
            Ok(input.trim().to_string())
        })
        .await??;

        if model.is_empty() {
            return Err(anyhow::anyhow!("default cloud model name is required"));
        }

        println!(
            "{}",
            log::info(&format!(
                "Validating model '{}' against {} ...",
                model, base_url
            ))
        );
        match OllamaProvider::validate_model(base_url, Some(api_key), &model).await {
            Ok(()) => {
                println!(
                    "{}",
                    log::success(&format!("Model '{}' is available.", model))
                );
                return Ok(model);
            }
            Err(e) => {
                println!(
                    "{}",
                    log::warning(&format!(
                        "Model '{}' is not available: {}. Try again.",
                        model, e
                    ))
                );
            }
        }
    }
}

/// Prompt the user for an Ollama cloud API key, save it to the credential
/// store, and return it.
async fn prompt_for_ollama_cloud_api_key() -> Result<String> {
    println!(
        "\n  {}",
        "Create an API key at https://ollama.com/settings/keys"
            .cyan()
            .dimmed()
    );
    print!("Enter your Ollama API key: ");
    use std::io::Write;
    std::io::stdout().flush().ok();
    let key = tokio::task::spawn_blocking(|| -> Result<String> {
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|e| anyhow::anyhow!("failed to read API key from stdin: {}", e))?;
        Ok(input.trim().to_string())
    })
    .await??;

    if key.is_empty() {
        return Err(anyhow::anyhow!("API key is required for Ollama cloud mode"));
    }

    if let Err(e) = credentials::store_api_key("ollama", &key) {
        eprintln!(
            "{}",
            log::warning(&format!(
                "could not save Ollama API key to credential store: {}",
                e
            ))
        );
    } else {
        println!(
            "{}",
            log::success(
                "Ollama cloud API key saved to ~/.reincore-io/credentials.json for future use."
            )
        );
    }

    Ok(key)
}

/// Set up the Ollama provider based on the `--cloud` flag. Returns a
/// ready-to-use provider.
async fn setup_ollama_provider(
    cloud: bool,
    model: Option<String>,
    interactive_tiers: bool,
) -> Result<std::sync::Arc<dyn engine_core::Provider>> {
    if cloud {
        println!(
            "{}",
            log::info("Setting up Ollama in cloud mode ...").cyan()
        );
        setup_ollama_cloud(model, interactive_tiers).await
    } else {
        println!(
            "{}",
            log::info("Setting up Ollama in local mode ...").cyan()
        );
        setup_ollama_local(model, interactive_tiers).await
    }
}

pub async fn handle_launch(args: LaunchArgs) -> Result<()> {
    let LaunchArgs {
        app,
        provider,
        mode,
        host,
        port,
        db_path,
        path,
        verbose,
        restore,
        cloud,
        model,
    } = args;

    // Normalize mode: "agentic"/"agent" -> "responses" wire API, "chat" -> "chat".
    // Note: Codex (OpenAI) no longer supports wire_api = "chat" and requires
    // "responses". For Codex we always use "responses" regardless of --mode.
    // Muse Code likewise only speaks the Responses dialect (/responses).
    let wire_api = match (mode.to_lowercase().as_str(), app.to_lowercase().as_str()) {
        (_, "codex" | "codex-app" | "open-code" | "opencode" | "open code" | "muse") => "responses",
        // OpenClaw only speaks the OpenAI Chat Completions wire
        // (`api: "openai-completions"`); it has no /v1/responses client.
        (_, "openclaw" | "open-claw" | "open claw") => "chat",
        ("chat", _) => "chat",
        ("agentic" | "agent" | "responses", _) => "responses",
        (other, _) => {
            return Err(anyhow::anyhow!(
                "unknown mode '{}'. Use 'agentic' (default) or 'chat'.",
                other
            ));
        }
    };

    // Display label for logging (not the wire API value).
    let mode_label = if wire_api == "responses" {
        "agent"
    } else {
        "chat"
    };

    // `reincore <app> --restore`: revert whatever a previous activation
    // changed, and do nothing else (no proxy, no API key prompt, no launch).
    if restore {
        return match restore_profile_for_app(&app)? {
            ProfileStatus::Handled => {
                println!(
                    "{}",
                    log::success(&format!(
                        "Restored '{}' to its original configuration.",
                        app
                    ))
                );
                Ok(())
            }
            ProfileStatus::Unsupported => {
                println!(
                    "{}",
                    log::info(&format!(
                        "'{}' has no persistent reincore profile to restore (nothing to do).",
                        app
                    ))
                );
                Ok(())
            }
        };
    }

    let provider_label = resolve_provider(&app, provider)?;

    // Claude Code appends `/v1/messages` to ANTHROPIC_BASE_URL itself, so the
    // base URL must NOT include `/v1`. OpenAI-based apps (codex, cursor, etc.)
    // expect OPENAI_BASE_URL to already include `/v1`.
    let proxy_base_url = format!("http://{}:{}", host, port);
    let proxy_url = format!("{}/v1", proxy_base_url);

    // Resolve the working directory for the app. If --path was given but the
    // directory doesn't exist, prompt the user to create it (or abort).
    let launch_path = resolve_launch_path(&path).await?;

    // Negotiate with whatever daemon is already running on this host:port.
    // This decides whether we (a) start a new single-provider daemon, (b)
    // reuse the running multi-provider daemon and route via the
    // X-ReinCore-Provider header, (c) reuse a matching single-provider
    // daemon, or (d) ask the user about switching providers (stop + restart).
    let daemon_config = DaemonConfig {
        host: host.clone(),
        port,
        pid: None,
    };
    let routing = crate::launch_routing::resolve_launch_routing_interactive(
        &daemon_config,
        &provider_label,
        Some(mode.as_str()),
        Some(app.as_str()),
    )
    .await?;
    // The provider the app will actually run on. This differs from the
    // request when the user chose "keep the current provider".
    let provider_label = routing.provider;
    let provider_header = routing.provider_header.as_deref();

    // Ollama handles its own credentials/model selection (local vs. cloud),
    // so skip the generic `ensure_api_key` path for it.
    if provider_label.to_lowercase() != "ollama" {
        // Make sure we actually have an API key before starting the proxy at
        // all - prompts interactively (and blocks) only if no key is present.
        ensure_api_key(&provider_label).await?;
    }

    // Provider construction + daemon startup are only needed when the
    // routing decision says we must start a daemon. When the routing reuses
    // an already-running daemon (multi-provider or matching single-provider),
    // there is nothing to set up - the daemon has the provider configured
    // already.
    let ollama_provider: Option<std::sync::Arc<dyn engine_core::Provider>> =
        if routing.start_daemon && provider_label.to_lowercase() == "ollama" {
            let saved_standard = read_provider_models("ollama")
                .standard
                .filter(|s| !s.trim().is_empty());
            let interactive_tiers = saved_standard.is_none();
            Some(setup_ollama_provider(cloud, model, interactive_tiers).await?)
        } else {
            None
        };

    // For the Copilot provider, perform the OAuth device flow and tier
    // detection BEFORE starting the proxy, so the user authenticates
    // before anything starts listening.
    let copilot_provider: Option<std::sync::Arc<dyn engine_core::Provider>> = if routing
        .start_daemon
        && provider_label.to_lowercase() == "copilot"
    {
        // Check if we have a saved OAuth token from a previous login.
        let saved_token = credentials::get_credential("copilot");
        if let Some(cred) = &saved_token {
            if cred.kind == CredentialKind::OAuthToken {
                // Pre-populate the global OAuth token cache so the device
                // flow is skipped if the token is still valid.
                let oauth_cache = engine_core::provider::get_global_oauth_token();
                *oauth_cache.lock().unwrap() = Some(cred.value.clone());
                println!("{}", log::info("Using saved GitHub Copilot OAuth token."));
            }
        }

        // Ask the user which GitHub account type they want to log in with.
        // (Only if no saved token or if the saved token is invalid, the
        // device flow will be triggered by check_tier below.)
        let github_instance = prompt_github_instance().await?;

        println!(
            "{}",
            log::info("Authenticating with GitHub Copilot...").cyan()
        );
        let client_id = engine_core::provider::CopilotProvider::resolve_client_id();
        let client_secret = engine_core::provider::CopilotProvider::resolve_client_secret();
        let local_config = crate::local_config::load_local_config();
        let model_policy = crate::provider::configured_model_policy(&local_config, "copilot");
        let config = engine_core::ProviderConfig {
            label: "copilot".to_string(),
            base_url: "https://api.githubcopilot.com".to_string(),
            api_key: String::new(),
            standard_model: model_policy.standard_model,
            models: model_policy.models,
            routing: model_policy.routing.clone(),
            ..Default::default()
        };

        // Determine the model family (check env var or prompt).
        let env_family_set = std::env::var("COPILOT_MODEL_FAMILY").is_ok();
        let family = if env_family_set {
            std::env::var("COPILOT_MODEL_FAMILY")
                .unwrap_or_default()
                .parse::<engine_core::provider::CopilotModelFamily>()
                .unwrap_or(engine_core::provider::CopilotModelFamily::OpenAI)
        } else {
            // Default to OpenAI - we'll prompt after tier detection.
            engine_core::provider::CopilotModelFamily::OpenAI
        };

        let provider = engine_core::provider::CopilotProvider::new(
            client_id,
            client_secret,
            github_instance.clone(),
            config,
            family,
        )?;

        // This triggers the OAuth device flow and token exchange.
        // If a saved token was pre-populated in the cache and is still
        // valid, the device flow is skipped.
        let is_free_tier = provider.check_tier().await?;

        // Save the OAuth token for future launches (if the device flow ran
        // and produced a new token).
        let oauth_cache = engine_core::provider::get_global_oauth_token();
        let current_token = oauth_cache.lock().unwrap().clone();
        if let Some(token) = &current_token {
            // Only save if it's different from what's already stored.
            let should_save = match &saved_token {
                Some(cred) => cred.value != *token,
                None => true,
            };
            if should_save {
                if let Err(e) = credentials::store_oauth_token("copilot", token) {
                    eprintln!(
                        "{}",
                        log::warning(&format!("could not save OAuth token: {}", e))
                    );
                } else {
                    println!(
                        "{}",
                        log::success("GitHub Copilot OAuth token saved for future use.")
                    );
                }
            }
        }

        if is_free_tier {
            println!(
                "{}",
                log::info("GitHub Copilot free tier detected. Using limited model.")
            );
            Some(std::sync::Arc::new(provider) as std::sync::Arc<dyn engine_core::Provider>)
        } else {
            // Paid tier - prompt for model family if not set via env var.
            if !env_family_set {
                let family = crate::provider::prompt_copilot_model_family();
                // Reconstruct with the selected family.
                let client_id = engine_core::provider::CopilotProvider::resolve_client_id();
                let client_secret = engine_core::provider::CopilotProvider::resolve_client_secret();
                let local_config = crate::local_config::load_local_config();
                let model_policy =
                    crate::provider::configured_model_policy(&local_config, "copilot");
                let config = engine_core::ProviderConfig {
                    label: "copilot".to_string(),
                    base_url: "https://api.githubcopilot.com".to_string(),
                    api_key: String::new(),
                    standard_model: model_policy.standard_model,
                    models: model_policy.models,
                    routing: model_policy.routing.clone(),
                    ..Default::default()
                };
                let new_provider = engine_core::provider::CopilotProvider::new(
                    client_id,
                    client_secret,
                    github_instance.clone(),
                    config,
                    family,
                )?;
                println!(
                    "{}",
                    log::success(&format!(
                        "GitHub Copilot paid tier detected. Using {} model family.",
                        family.as_str()
                    ))
                );
                Some(std::sync::Arc::new(new_provider) as std::sync::Arc<dyn engine_core::Provider>)
            } else {
                println!(
                    "{}",
                    log::success(&format!(
                        "GitHub Copilot paid tier detected. Using {} model family.",
                        family.as_str()
                    ))
                );
                Some(std::sync::Arc::new(provider) as std::sync::Arc<dyn engine_core::Provider>)
            }
        }
    } else {
        None
    };

    // Start the proxy FIRST, in the background, so it is already listening
    // before we ever try to launch (or point) an app at it. When the routing
    // decision reused a running daemon this block is skipped entirely and
    // the app talks to the daemon that is already up.
    let proxy_handle = if routing.start_daemon {
        // The user chose to switch providers (or no daemon was running):
        // stop the existing daemon so the new one can bind the same port.
        if routing.stop_first {
            crate::daemon::stop_daemon(host.clone(), port).await?;
        }

        let proxy_host = host.clone();
        let proxy_provider_label = provider_label.clone();
        let proxy_mode = mode_label.to_string();
        let proxy_project_path = launch_path
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned());
        Some(tokio::spawn(async move {
            if let Some(provider) = copilot_provider {
                // For copilot, start the daemon directly with the pre-authenticated provider.
                crate::daemon::start_single_provider_daemon(
                    proxy_host,
                    port,
                    db_path,
                    verbose,
                    provider,
                    &proxy_mode,
                    proxy_project_path.clone(),
                    None,
                    None,
                )
                .await
            } else if let Some(provider) = ollama_provider {
                // For ollama, start the daemon directly with the pre-configured
                // provider (local or cloud mode already resolved).
                crate::daemon::start_single_provider_daemon(
                    proxy_host,
                    port,
                    db_path,
                    verbose,
                    provider,
                    &proxy_mode,
                    proxy_project_path.clone(),
                    None,
                    None,
                )
                .await
            } else {
                provider_start_proxy(
                    proxy_provider_label,
                    proxy_host,
                    port,
                    db_path,
                    verbose,
                    &proxy_mode,
                    proxy_project_path,
                )
                .await
            }
        }))
    } else {
        None
    };

    // Don't launch the app (or hand off control) until the proxy is actually
    // accepting connections.
    let host_check = host.clone();
    let ready = tokio::task::spawn_blocking(move || {
        wait_for_proxy_ready(&host_check, port, Duration::from_secs(5))
    })
    .await?;

    if !ready {
        eprintln!(
            "{}",
            log::warning(&format!(
                "Proxy on {}:{} did not become ready within 5s; the app may fail to connect.",
                host, port
            ))
        );
    }

    // Register the launch working directory as a local project on the
    // daemon (fresh or reused) so sessions opened from the launched app
    // attribute to the active path under the latest folder name.
    // Best-effort: a warning is enough, the app can still run.
    if let Some(dir) = launch_path.as_deref() {
        if let Err(e) = ensure_launch_project(&host, port, dir).await {
            eprintln!("{}", log::warning(&format!("{}", e)));
        }
    }

    // On Windows, the Claude Code installer places `claude.exe` in
    // `~/.local/bin` but does not add that directory to PATH. Detect that
    // situation and patch up PATH for this process so the launch succeeds,
    // while telling the user how to make it permanent.
    #[cfg(target_os = "windows")]
    {
        let app_lower = app.to_lowercase();
        if app_lower == "claude" {
            let local_bin = user_local_bin();
            let claude_exe = local_bin.join("claude.exe");
            if claude_exe.exists() && !is_on_path(&local_bin) {
                if let Ok(existing) = std::env::var("PATH") {
                    std::env::set_var("PATH", format!("{};{}", local_bin.display(), existing));
                }
                println!(
                    "Found 'claude' in {} (not on PATH). Added it to PATH for this session.",
                    local_bin.display()
                );
                println!(
                    "To make this permanent, add {} to your PATH:",
                    local_bin.display()
                );
                println!(
                    "  [Environment]::SetEnvironmentVariable('PATH', \"{};$env:PATH\", 'User')",
                    local_bin.display()
                );
            }
        }
    }

    if !is_app_installed(&app) {
        eprintln!(
            "{}",
            log::warning(&format!(
                "'{}' does not appear to be installed on this system.",
                app
            ))
        );
        if let Some(hint) = install_hint_for_app(&app) {
            eprintln!("  {}", hint.dimmed());
        }
        eprintln!(
            "  {}",
            format!(
                "Proxy is running. Configure {} to use {} as its API/base URL when ready.",
                app, proxy_url
            )
            .dimmed()
        );
        if let Some(handle) = proxy_handle {
            handle.await??;
        }
    } else {
        println!(
            "{}",
            log::info(&format!(
                "'{}' appears to be installed. Routing all {} requests through reincore at {} ...",
                app, app, proxy_url
            ))
        );
        println!(
            "{}",
            log::info(&format!(
                "Proxy running for provider '{}' on {} ...",
                provider_label, proxy_url
            ))
        );

        // Persistently point the app's own config at reincore, if we have a
        // verified way to do so, so it keeps working even outside a
        // reincore-launched terminal - until `reincore <app> --restore`.
        match activate_profile_for_app(
            &app,
            &proxy_base_url,
            &proxy_url,
            &provider_label,
            wire_api,
            provider_header,
        ) {
            Ok(ProfileStatus::Handled) => {
                println!(
                    "{}",
                    log::success(&format!(
                        "Activated a persistent reincore profile for '{}'. Run `reincore {} --restore` to revert it.",
                        app, app
                    ))
                );
                // Claude Code: a saved claude.ai login can conflict with the
                // gateway credential and cause "model issue" errors. Tell the
                // user to run /logout inside Claude Code if they see auth errors.
                if app.to_lowercase() == "claude" {
                    println!(
                        "  {}",
                        "Tip: if Claude Code shows a model error, run /logout inside it to clear any saved login.".yellow()
                    );
                }
            }
            Ok(ProfileStatus::Unsupported) => {
                println!(
                    "{}",
                    log::info(&format!(
                        "'{}' doesn't have a persistent reincore profile yet - routing this session only.",
                        app
                    ))
                );
            }
            Err(e) => {
                eprintln!(
                    "{}",
                    log::warning(&format!(
                        "could not activate a persistent profile for {}: {}",
                        app, e
                    ))
                );
            }
        }

        match spawn_app_in_terminal(
            &app,
            &proxy_base_url,
            &proxy_url,
            launch_path.as_deref(),
            provider_header,
        ) {
            Ok(()) => {
                println!(
                    "{}",
                    log::success(&format!("Started {} in a new terminal tab/window.", app))
                );
                println!(
                    "{}",
                    log::info(&format!(
                        "Proxy running for provider '{}' on {}. Press Ctrl+C to stop.",
                        provider_label, proxy_url
                    ))
                );
                if let Some(handle) = proxy_handle {
                    handle.await??;
                }
            }
            Err(e) => {
                eprintln!(
                    "{}",
                    log::warning(&format!("could not auto-launch {}: {}", app, e))
                );
                eprintln!(
                    "  {}",
                    format!(
                        "Proxy is running; launch {} manually and point it to {}.",
                        app, proxy_url
                    )
                    .dimmed()
                );
                if let Some(handle) = proxy_handle {
                    handle.await??;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_project_name_uses_latest_folder() {
        assert_eq!(
            launch_project_name(Path::new("/workspace/my-app")),
            "my-app"
        );
        assert_eq!(
            launch_project_name(Path::new("/workspace/my-app/")),
            "my-app"
        );
        // Roots and empty names fall back instead of producing a blank project.
        assert_eq!(launch_project_name(Path::new("/")), "reincore");
    }

    #[test]
    fn muse_launch_defaults_to_meta() {
        assert_eq!(resolve_provider("muse", None).unwrap(), "meta");
        assert_eq!(app_defaults().get("muse"), Some(&"meta"));
        assert_eq!(app_command_aliases("muse"), vec!["muse"]);
        // Muse Code has no header mechanism, so launch uses a
        // single-provider daemon (like the Copilot CLI).
        assert!(!app_supports_provider_header("muse"));
    }

    #[test]
    fn muse_spawn_carries_endpoint_flags() {
        // The base URL has no env-var equivalent, so it travels as CLI
        // flags alongside the documented default model.
        assert_eq!(
            app_extra_args("muse", "http://127.0.0.1:8989"),
            vec![
                "--provider".to_string(),
                "meta".to_string(),
                "--base-url".to_string(),
                "http://127.0.0.1:8989".to_string(),
                "--model".to_string(),
                "reincore-auto".to_string(),
            ]
        );
        assert!(app_extra_args("codex", "http://127.0.0.1:8989").is_empty());
    }
}
