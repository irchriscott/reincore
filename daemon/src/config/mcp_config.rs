//! Local MCP installation and persisted startup settings for the local engine.
use anyhow::{Context, Result};
use serde::Serialize;
use std::{
    ffi::OsStr,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

const DEFAULT_MCP_JSON: &str = include_str!("../../resources/mcp.json");

#[derive(Serialize)]
pub struct McpSettings {
    pub enabled: bool,
    pub config_path: String,
    pub environment_override: bool,
    pub default_config_path: String,
    pub restart_required: bool,
}

pub fn default_path() -> Result<PathBuf> {
    Ok(dirs::home_dir()
        .context("cannot find the home directory")?
        .join(super::local_config::CONFIG_DIR_NAME)
        .join("mcp.json"))
}

fn install_default(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            file.write_all(DEFAULT_MCP_JSON.as_bytes())?;
            file.sync_all()?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn read_document(path: &Path) -> Result<toml_edit::DocumentMut> {
    fs::read_to_string(path)?
        .parse()
        .context("invalid local config.toml")
}

fn enabled(document: &toml_edit::DocumentMut) -> Result<bool> {
    match document.get("mcp") {
        None => Ok(false),
        Some(table) => {
            let table = table.as_table().context("[mcp] must be a table")?;
            match table.get("enabled") {
                None => Ok(false),
                Some(value) => value.as_bool().context("mcp.enabled must be true or false"),
            }
        }
    }
}

// Resolve relative paths against config.toml, consistently for CLI, app and daemon.
fn resolve_path(path: &OsStr, config: &Path, home: &Path) -> Result<PathBuf> {
    let text = path.to_string_lossy();
    anyhow::ensure!(
        !text.trim().is_empty(),
        "MCP configuration path cannot be blank"
    );
    if text == "~" {
        return Ok(home.to_path_buf());
    }
    if let Some(relative) = text.strip_prefix("~/").or_else(|| text.strip_prefix("~\\")) {
        return Ok(home.join(relative));
    }
    anyhow::ensure!(
        !text.starts_with('~'),
        "only ~/ home expansion is supported"
    );
    let path = PathBuf::from(path);
    if path.is_absolute() {
        return Ok(path);
    }
    let config = std::path::absolute(config)?;
    Ok(config
        .parent()
        .context("config.toml has no parent directory")?
        .join(path))
}

fn effective_path(
    document: &toml_edit::DocumentMut,
    config: &Path,
    home: &Path,
    override_path: Option<&OsStr>,
) -> Result<PathBuf> {
    let configured = document
        .get("mcp")
        .map(|item| item.as_table().context("[mcp] must be a table"))
        .transpose()?
        .and_then(|table| table.get("config_path"))
        .map(|item| item.as_str().context("mcp.config_path must be a string"))
        .transpose()?;
    if let Some(path) = override_path {
        return resolve_path(path, config, home).context("invalid REINCORE_MCP_CONFIG");
    }
    match configured {
        Some(path) => resolve_path(OsStr::new(path), config, home),
        None => Ok(home
            .join(super::local_config::CONFIG_DIR_NAME)
            .join("mcp.json")),
    }
}

pub fn settings() -> Result<McpSettings> {
    let config = crate::local_config::ensure_config_exists()?;
    let home = dirs::home_dir().context("cannot find the home directory")?;
    let default = default_path()?;
    install_default(&default)?;
    let document = read_document(&config)?;
    let override_path = std::env::var_os("REINCORE_MCP_CONFIG");
    let effective = effective_path(&document, &config, &home, override_path.as_deref())?;
    Ok(McpSettings {
        enabled: enabled(&document)?,
        config_path: effective.to_string_lossy().into_owned(),
        environment_override: override_path.is_some(),
        default_config_path: default.to_string_lossy().into_owned(),
        restart_required: true,
    })
}

fn write_enabled(path: &Path, value: bool) -> Result<()> {
    let canonical = fs::canonicalize(path)?;
    let path = canonical.as_path();
    let _lock = lock_config(path)?;
    let mut document = read_document(path)?;
    // Validate existing settings before editing; never overwrite malformed data.
    enabled(&document)?;
    if document.get("mcp").is_none() {
        document["mcp"] = toml_edit::Item::Table(toml_edit::Table::new());
    }
    document["mcp"]["enabled"] = toml_edit::value(value);
    atomic_write(path, document.to_string().as_bytes())?;
    Ok(())
}

pub fn set_enabled(value: bool) -> Result<McpSettings> {
    settings()?;
    write_enabled(&crate::local_config::local_config_path(), value)?;
    settings()
}

pub async fn registry(settings: &McpSettings) -> Result<engine_core::ToolExecutorRegistry> {
    if !settings.enabled {
        return Ok(engine_core::ToolExecutorRegistry::new());
    }
    let bytes = tokio::fs::read(&settings.config_path)
        .await
        .context("cannot read local MCP configuration")?;
    let config: engine_core::mcp::McpConfig =
        serde_json::from_slice(&bytes).context("invalid local MCP configuration JSON")?;
    // OAuth sign-ins are kept beside the configuration, not in it. Without
    // them, servers that need a sign-in are skipped rather than every server.
    let path = Path::new(&settings.config_path);
    let sign_ins = crate::mcp_credentials::load(path).unwrap_or_else(|error| {
        tracing::warn!("MCP sign-ins unavailable: {error:#}");
        Default::default()
    });
    Ok(config
        .connect_with(false, |server| {
            crate::mcp_credentials::credential(path, server, &sign_ins)
        })
        .await?)
}

pub async fn command(action: &crate::cli::McpCommand) -> Result<()> {
    use crate::cli::McpCommand;
    if let McpCommand::SignIn {
        name,
        client_id,
        client_secret_env,
        scopes,
    } = action
    {
        return sign_in(
            name,
            client_id.clone(),
            client_secret_env.as_deref(),
            scopes,
        )
        .await;
    }
    if matches!(
        action,
        McpCommand::List
            | McpCommand::Add { .. }
            | McpCommand::Update { .. }
            | McpCommand::Delete { .. }
    ) {
        let path = PathBuf::from(settings()?.config_path);
        manage_servers(&path, action)?;
        return Ok(());
    }
    let settings = match action {
        crate::cli::McpCommand::Enable => set_enabled(true)?,
        crate::cli::McpCommand::Disable => set_enabled(false)?,
        crate::cli::McpCommand::Status => settings()?,
        _ => unreachable!(),
    };
    println!(
        "MCP configured: {}",
        if settings.enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!("Configuration: {}", settings.config_path);
    if settings.environment_override {
        println!("REINCORE_MCP_CONFIG overrides the config.toml MCP path.");
    }
    println!("Changes take effect after restarting the local daemon.");
    Ok(())
}

// A cross-process lock prevents concurrent commands from losing each other's edits.
struct ConfigLock(PathBuf);
impl Drop for ConfigLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn lock_config(path: &Path) -> Result<ConfigLock> {
    let mut lock_path = path.as_os_str().to_owned();
    lock_path.push(".lock");
    let lock_path = PathBuf::from(lock_path);
    let file = OpenOptions::new().write(true).create_new(true).open(&lock_path)
        .context("configuration is locked; after a crashed command remove its .lock file only when no writer is running")?;
    drop(file);
    Ok(ConfigLock(lock_path))
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let temp = path.with_extension(format!("{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| -> Result<()> {
        let permissions = fs::metadata(path)?.permissions();
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.set_permissions(permissions)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}

pub fn list_servers() -> Result<engine_core::mcp::McpConfig> {
    Ok(serde_json::from_slice(&fs::read(settings()?.config_path)?)?)
}

pub fn save_server(
    server: engine_core::mcp::ServerConfig,
    original_name: Option<String>,
) -> Result<engine_core::mcp::McpConfig> {
    save_server_signed_in(server, original_name, None)
}

/// Saves a server, with a completed OAuth sign-in when one is given: its
/// settings replace the server's `auth` and its tokens are stored beside the
/// configuration. Without one, an OAuth server keeps the settings of its
/// stored sign-in while the endpoint is unchanged, as cloud servers do.
pub fn save_server_signed_in(
    server: engine_core::mcp::ServerConfig,
    original_name: Option<String>,
    sign_in: Option<crate::mcp_credentials::SignIn>,
) -> Result<engine_core::mcp::McpConfig> {
    let path = PathBuf::from(settings()?.config_path);
    save_signed_in(&path, server, original_name, sign_in)
}

pub(crate) fn save_signed_in(
    path: &Path,
    mut server: engine_core::mcp::ServerConfig,
    original_name: Option<String>,
    sign_in: Option<crate::mcp_credentials::SignIn>,
) -> Result<engine_core::mcp::McpConfig> {
    let name = original_name.clone().unwrap_or_else(|| server.name.clone());
    // Read before writing anything, so an unreadable sign-in file fails the save.
    let stored = crate::mcp_credentials::load(path)?;
    match &sign_in {
        Some(sign_in) => {
            anyhow::ensure!(
                sign_in.url == server.url,
                "the sign-in was for another endpoint URL; sign in again"
            );
            server.auth = Some(sign_in.auth.clone());
            // Sign-in tokens take the place of any configured credential.
            server.bearer_token = None;
            server.bearer_token_env = None;
            server.secret = None;
            server.secret_env = None;
            server.session_token_env = None;
        }
        None if matches!(
            server.auth,
            Some(engine_core::mcp::AuthConfig::Oauth { .. })
        ) =>
        {
            if let Some(earlier) = stored
                .servers
                .get(&name)
                .filter(|earlier| earlier.url == server.url)
            {
                server.auth = Some(earlier.auth.clone());
            }
        }
        None => {}
    }
    let config = mutate_servers(path, name.clone(), Some(server), original_name.is_none())?;
    if let Some(sign_in) = sign_in {
        crate::mcp_credentials::save(path, &name, sign_in)?;
    }
    // A server whose endpoint or settings changed no longer uses its sign-in.
    crate::mcp_credentials::retain_matching(path, &config)?;
    Ok(config)
}

/// `reincore mcp sign-in`: signs in to a saved server in the browser and
/// stores the sign-in beside the configuration.
async fn sign_in(
    name: &str,
    client_id: Option<String>,
    client_secret_env: Option<&str>,
    scopes: &[String],
) -> Result<()> {
    use crate::mcp_sign_in;
    let path = PathBuf::from(settings()?.config_path);
    let config: engine_core::mcp::McpConfig =
        serde_json::from_slice(&fs::read(&path).context("cannot read local MCP configuration")?)
            .context("invalid local MCP configuration JSON")?;
    let server = config
        .servers
        .into_iter()
        .find(|server| server.name == name)
        .with_context(|| format!("no saved MCP server is named {name}"))?;
    let earlier = crate::mcp_credentials::load(&path)?
        .servers
        .remove(name)
        .filter(|earlier| earlier.url == server.url);
    let client_secret = client_secret_env
        .map(|variable| {
            std::env::var(variable)
                .with_context(|| format!("environment variable {variable} is not set"))
        })
        .transpose()?;
    // Without --client-id, sign in again with the server's current client.
    let client_id = client_id.or_else(|| match &server.auth {
        Some(engine_core::mcp::AuthConfig::Oauth { client_id, .. }) => Some(client_id.clone()),
        _ => None,
    });
    let (client_id, client_secret) =
        mcp_sign_in::client_for(earlier.as_ref(), client_id, client_secret);
    let request = mcp_sign_in::SignInRequest {
        url: server.url.clone(),
        client_id,
        client_secret,
        scopes: (!scopes.is_empty()).then(|| scopes.to_vec()),
        timeout_ms: server.timeout_ms,
    };
    let open = |url: &str| {
        println!("Opening your browser to sign in. If it does not open, visit:\n{url}");
        let _ = open::that_detached(url);
        Ok(())
    };
    let cancelled = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    let signed_in = mcp_sign_in::sign_in(request, open, cancelled)
        .await
        .map_err(anyhow::Error::msg)?;
    save_server_signed_in(server, Some(name.to_owned()), Some(signed_in))?;
    println!("Signed in to {name}. Restart the daemon to apply changes.");
    Ok(())
}

pub fn delete_server(name: String) -> Result<engine_core::mcp::McpConfig> {
    let path = PathBuf::from(settings()?.config_path);
    let config = mutate_servers(&path, name, None, false)?;
    crate::mcp_credentials::retain_matching(&path, &config)?;
    Ok(config)
}

fn manage_servers(path: &Path, action: &crate::cli::McpCommand) -> Result<()> {
    use crate::cli::McpCommand;
    let (name, replacement, add) = match action {
        McpCommand::List => {
            let config: engine_core::mcp::McpConfig = serde_json::from_slice(&fs::read(path)?)?;
            println!("{}", serde_json::to_string_pretty(&config)?);
            return Ok(());
        }
        McpCommand::Add { file, .. } | McpCommand::Update { file, .. } => {
            let server = engine_core::mcp::read_server_config(file.as_deref())?;
            let name = match action {
                McpCommand::Update { name, .. } => name.clone(),
                _ => server.name.clone(),
            };
            (name, Some(server), matches!(action, McpCommand::Add { .. }))
        }
        McpCommand::Delete { name } => (name.clone(), None, false),
        _ => anyhow::bail!("expected a server management command"),
    };
    let config = mutate_servers(path, name, replacement, add)?;
    crate::mcp_credentials::retain_matching(path, &config)?;
    println!("Saved. Restart the daemon to apply changes.");
    Ok(())
}

fn mutate_servers(
    path: &Path,
    name: String,
    mut replacement: Option<engine_core::mcp::ServerConfig>,
    add: bool,
) -> Result<engine_core::mcp::McpConfig> {
    if let Some(server) = &mut replacement {
        server.validate(false)?;
        anyhow::ensure!(name == server.name, "replacement server name must match");
    }
    // Resolve aliases before locking or replacing so a symlink stays a symlink.
    let canonical = fs::canonicalize(path)?;
    let path = canonical.as_path();
    let _lock = lock_config(path)?;
    let mut config: engine_core::mcp::McpConfig = serde_json::from_slice(&fs::read(path)?)?;
    let existing = config.servers.iter().position(|server| server.name == name);
    if add {
        anyhow::ensure!(existing.is_none(), "MCP server already exists");
        anyhow::ensure!(config.servers.len() < 32, "too many MCP servers");
        config.servers.push(replacement.unwrap());
    } else {
        let index = existing.context("MCP server not found")?;
        if let Some(server) = replacement {
            config.servers[index] = server;
        } else {
            config.servers.remove(index);
        }
    }
    atomic_write(path, serde_json::to_string_pretty(&config)?.as_bytes())?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oauth_servers_keep_their_sign_in_settings_when_edited() {
        use crate::mcp_credentials::{load, SignIn};
        use engine_core::mcp::{AuthConfig, AuthSecret, ServerConfig};
        let root = std::env::temp_dir().join(format!("mcp-save-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("mcp.json");
        fs::write(&path, r#"{"servers":[]}"#).unwrap();
        let oauth = |scopes: &[&str]| AuthConfig::Oauth {
            client_id: "client".into(),
            token_endpoint: "https://auth.example.com/token".into(),
            scopes: scopes.iter().map(|scope| scope.to_string()).collect(),
            resource: None,
            client_auth: Default::default(),
            issuer: None,
            managed_client: false,
        };
        let server = |name: &str, auth: AuthConfig| -> ServerConfig {
            serde_json::from_value(serde_json::json!({
                "name": name,
                "url": "https://mcp.example.com/mcp",
                "timeout_ms": 1000,
                "auth": auth,
                "secret_env": "LEFT_OVER",
                "access": {"mode": "local", "project_ids": [uuid::Uuid::new_v4()]},
                "tools": {"read": {"read_only": true}}
            }))
            .unwrap()
        };
        let signed_in = SignIn {
            url: "https://mcp.example.com/mcp".into(),
            auth: oauth(&["read"]),
            secret: AuthSecret {
                refresh_token: Some("refresh".into()),
                ..AuthSecret::default()
            },
            registered: true,
        };

        // A sign-in replaces any configured credential source.
        let config = save_signed_in(
            &path,
            server("docs", AuthConfig::Bearer),
            None,
            Some(signed_in),
        )
        .unwrap();
        assert_eq!(config.servers[0].auth, Some(oauth(&["read"])));
        assert!(config.servers[0].secret_env.is_none());

        // Editing the OAuth settings in JSON keeps the sign-in and the
        // settings its tokens were issued for.
        let mut edited = server("docs", oauth(&["read", "write"]));
        edited.secret_env = None;
        let config = save_signed_in(&path, edited, Some("docs".into()), None).unwrap();
        assert_eq!(config.servers[0].auth, Some(oauth(&["read"])));
        assert!(load(&path).unwrap().servers["docs"].registered);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn path_precedence_and_resolution() {
        let home = std::env::temp_dir().join("mcp-home");
        let config = home.join("settings/config.toml");
        let empty = "".parse().unwrap();
        assert_eq!(
            effective_path(&empty, &config, &home, None).unwrap(),
            home.join(crate::local_config::CONFIG_DIR_NAME)
                .join("mcp.json")
        );
        let document = "[mcp]\nconfig_path = 'servers/custom.json'"
            .parse()
            .unwrap();
        assert_eq!(
            effective_path(&document, &config, &home, None).unwrap(),
            home.join("settings/servers/custom.json")
        );
        assert_eq!(
            effective_path(
                &document,
                &config,
                &home,
                Some(OsStr::new("~/override.json"))
            )
            .unwrap(),
            home.join("override.json")
        );
        let absolute = home.join("absolute.json");
        assert_eq!(
            effective_path(&document, &config, &home, Some(absolute.as_os_str())).unwrap(),
            absolute
        );
        for value in [
            "'~/.reincore-io/custom.json'",
            "'~\\.reincore-io/custom.json'",
        ] {
            let document = format!("[mcp]\nconfig_path = {value}").parse().unwrap();
            assert_eq!(
                effective_path(&document, &config, &home, None).unwrap(),
                home.join(crate::local_config::CONFIG_DIR_NAME)
                    .join("custom.json")
            );
        }
        for value in ["''", "'   '", "42", "'~another/file.json'"] {
            let document = format!("[mcp]\nconfig_path = {value}").parse().unwrap();
            assert!(effective_path(&document, &config, &home, None).is_err());
        }
        assert!(effective_path(&empty, &config, &home, Some(OsStr::new(" "))).is_err());
    }

    #[test]
    fn server_crud_preserves_file_on_conflict_and_invalid_input() {
        use crate::cli::McpCommand;
        let root = std::env::temp_dir().join(format!("mcp-crud-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("mcp.json");
        let input = root.join("server.json");
        install_default(&path).unwrap();
        let mut server = serde_json::json!({
            "name":"repo",
            "url":"http://127.0.0.1:9000/mcp",
            "allow_loopback_http":true,
            "bearer_token":"local-persisted-token",
            "timeout_ms":1000,
            "access":{
                "mode":"local",
                "project_path":root
            },
            "tools": {
                "read": {
                    "read_only": true
                }
            }
        });
        fs::write(&input, server.to_string()).unwrap();
        manage_servers(
            &path,
            &McpCommand::Add {
                file: Some(input.clone()),
                stdin: false,
            },
        )
        .unwrap();
        let original = fs::read(&path).unwrap();
        assert!(manage_servers(
            &path,
            &McpCommand::Add {
                file: Some(input.clone()),
                stdin: false
            }
        )
        .is_err());
        assert_eq!(fs::read(&path).unwrap(), original);
        server["timeout_ms"] = serde_json::json!(2000);
        fs::write(&input, server.to_string()).unwrap();
        manage_servers(
            &path,
            &McpCommand::Update {
                name: "repo".into(),
                file: Some(input.clone()),
                stdin: false,
            },
        )
        .unwrap();
        let updated = fs::read(&path).unwrap();
        let config: engine_core::mcp::McpConfig = serde_json::from_slice(&updated).unwrap();
        assert_eq!(config.servers[0].timeout_ms, 2000);
        assert_eq!(
            config.servers[0].bearer_token.as_deref(),
            Some("local-persisted-token")
        );
        server["url"] = serde_json::json!("https://user:secret@example.com/mcp");
        fs::write(&input, server.to_string()).unwrap();
        assert!(manage_servers(
            &path,
            &McpCommand::Update {
                name: "repo".into(),
                file: Some(input.clone()),
                stdin: false
            }
        )
        .is_err());
        assert_eq!(fs::read(&path).unwrap(), updated);
        manage_servers(
            &path,
            &McpCommand::Delete {
                name: "repo".into(),
            },
        )
        .unwrap();
        assert!(manage_servers(
            &path,
            &McpCommand::Delete {
                name: "repo".into()
            }
        )
        .is_err());
        let config: engine_core::mcp::McpConfig =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert!(config.servers.is_empty());
        fs::remove_file(input).unwrap();
        fs::remove_file(path).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn bundled_default_is_valid_and_existing_files_are_preserved() {
        let root = std::env::temp_dir().join(format!("reincore-mcp-{}", uuid::Uuid::new_v4()));
        let path = root.join("mcp.json");
        install_default(&path).unwrap();
        let config: engine_core::mcp::McpConfig =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert!(config.servers.is_empty());
        fs::write(&path, "custom configuration").unwrap();
        install_default(&path).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "custom configuration");
        fs::remove_file(path).unwrap();
        fs::remove_dir(root).unwrap();
    }
    #[test]
    fn toggle_preserves_other_configuration_and_rejects_invalid_values() {
        let path = std::env::temp_dir().join(format!("reincore-mcp-{}.toml", uuid::Uuid::new_v4()));
        fs::write(
            &path,
            r#"# preserve me
[routing]
mode = "locked"
"#,
        )
        .unwrap();
        assert!(!enabled(&read_document(&path).unwrap()).unwrap());
        let before = fs::read(&path).unwrap();
        let lock = lock_config(&fs::canonicalize(&path).unwrap()).unwrap();
        assert!(write_enabled(&path, true).is_err());
        assert_eq!(fs::read(&path).unwrap(), before);
        drop(lock);
        write_enabled(&path, true).unwrap();
        assert!(enabled(&read_document(&path).unwrap()).unwrap());
        assert!(fs::read_to_string(&path).unwrap().contains("# preserve me"));
        assert_eq!(
            read_document(&path).unwrap()["routing"]["mode"].as_str(),
            Some("locked")
        );
        write_enabled(&path, false).unwrap();
        assert!(!enabled(&read_document(&path).unwrap()).unwrap());
        fs::write(&path, "[mcp]\nenabled = 'not a boolean'\n").unwrap();
        assert!(write_enabled(&path, true).is_err());
        fs::remove_file(path).unwrap();
    }
    #[tokio::test]
    async fn disabled_does_not_read_or_connect_and_enabled_validates_configuration() {
        let mut settings = McpSettings {
            enabled: false,
            config_path: "missing-mcp-config.json".into(),
            environment_override: false,
            default_config_path: String::new(),
            restart_required: true,
        };
        assert!(registry(&settings)
            .await
            .unwrap()
            .list_all_tools()
            .await
            .unwrap()
            .is_empty());
        settings.enabled = true;
        assert!(registry(&settings).await.is_err());
    }
}
