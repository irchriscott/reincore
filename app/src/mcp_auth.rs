//! MCP server authentication commands for the desktop app. OAuth sign-in runs
//! in the system browser (see reincore::mcp_sign_in). A finished sign-in
//! is held in memory until the server is saved with it, then kept beside
//! mcp.json (see reincore::mcp_credentials).
use async_trait::async_trait;
use engine_core::{
    mcp::{
        oauth, AuthConfig, AuthSecret, Credential, CredentialLease, CredentialStore,
        DiscoveredTool, DiscoveryRequest, McpConfig, ServerConfig,
    },
    EngineError, EngineResult,
};
use reincore::{
    mcp_config, mcp_credentials,
    mcp_credentials::SignIn,
    mcp_sign_in::{self, message},
};
use serde::Serialize;
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use tauri::State;
use tokio::sync::{oneshot, Mutex};

// How long a finished sign-in waits for its server to be saved.
const SIGN_IN_KEPT: Duration = Duration::from_secs(3600);

struct Finished {
    sign_in: SignIn,
    finished_at: Instant,
}

type FinishedMap = Arc<Mutex<HashMap<String, Finished>>>;

#[derive(Default)]
pub struct McpSignIns {
    finished: FinishedMap,
    cancel: std::sync::Mutex<Option<oneshot::Sender<()>>>,
}

/// A finished sign-in, identified for discovery and saving.
#[derive(Serialize)]
pub struct FinishedSignIn {
    id: String,
    auth: AuthConfig,
    expires_in_seconds: u64,
}

/// Servers plus the names of those with a usable stored sign-in.
#[derive(Serialize)]
pub struct LocalMcpServers {
    servers: Vec<ServerConfig>,
    signed_in: Vec<String>,
}

fn config_path() -> Result<PathBuf, String> {
    mcp_config::settings()
        .map(|settings| PathBuf::from(settings.config_path))
        .map_err(|error| error.to_string())
}

/// Sign-in files are read under a lock another ReinCore process may hold, so
/// file work runs on a blocking thread, never the UI or async threads.
async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(work)
        .await
        .map_err(|error| error.to_string())?
}

pub fn with_sign_ins(config: McpConfig) -> Result<LocalMcpServers, String> {
    // An unreadable sign-in file shows every OAuth server as signed out
    // instead of hiding the whole list; saving reports the file itself.
    let sign_ins = mcp_credentials::load(&config_path()?).unwrap_or_else(|error| {
        tracing::warn!("MCP sign-ins unavailable: {error:#}");
        Default::default()
    });
    let signed_in = config
        .servers
        .iter()
        .filter(|server| {
            sign_ins
                .servers
                .get(&server.name)
                .is_some_and(|sign_in| sign_in.matches(server))
        })
        .map(|server| server.name.clone())
        .collect();
    Ok(LocalMcpServers {
        servers: config.servers,
        signed_in,
    })
}

#[tauri::command]
pub async fn mcp_oauth_metadata(
    url: String,
    timeout_ms: u64,
) -> Result<oauth::AuthorizationMetadata, String> {
    oauth::discover(
        &url,
        false,
        Duration::from_millis(timeout_ms.clamp(100, 120_000)),
    )
    .await
    .map_err(message)
}

/// Signs in to an MCP server in the system browser. Signing in again with the
/// client of an earlier sign-in (saved for `server_name`, or finished in this
/// session) reuses its client secret.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn sign_in_mcp_server(
    state: State<'_, McpSignIns>,
    url: String,
    client_id: Option<String>,
    client_secret: Option<String>,
    scopes: Option<Vec<String>>,
    timeout_ms: u64,
    server_name: Option<String>,
) -> Result<FinishedSignIn, String> {
    // Starting a sign-in abandons any earlier one still waiting.
    let (cancel, cancelled) = oneshot::channel();
    *state.cancel.lock().unwrap() = Some(cancel);

    let saved = match server_name {
        Some(name) => {
            let path = config_path()?;
            blocking(move || {
                mcp_credentials::load(&path)
                    .map(|mut sign_ins| sign_ins.servers.remove(&name))
                    .map_err(|error| error.to_string())
            })
            .await?
        }
        None => None,
    };
    let unsaved: Vec<SignIn> = {
        let finished = state.finished.lock().await;
        finished
            .values()
            .map(|entry| entry.sign_in.clone())
            .collect()
    };
    let earlier = saved.into_iter().chain(unsaved).find(|earlier| {
        earlier.url == url
            && matches!(&earlier.auth, AuthConfig::Oauth { client_id: id, .. } if Some(id) == client_id.as_ref())
    });
    let (client_id, client_secret) =
        mcp_sign_in::client_for(earlier.as_ref(), client_id, client_secret);

    let sign_in = mcp_sign_in::sign_in(
        mcp_sign_in::SignInRequest {
            url,
            client_id,
            client_secret,
            scopes,
            timeout_ms,
        },
        |authorization_url| {
            tauri_plugin_opener::open_url(authorization_url, None::<&str>)
                .map_err(|error| format!("Could not open the browser: {error}"))
        },
        async {
            // A newer sign-in replacing this sender cancels this one too.
            let _ = cancelled.await;
        },
    )
    .await?;

    let id = uuid::Uuid::new_v4().to_string();
    let auth = sign_in.auth.clone();
    let mut finished = state.finished.lock().await;
    finished.retain(|_, entry| entry.finished_at.elapsed() < SIGN_IN_KEPT);
    finished.insert(
        id.clone(),
        Finished {
            sign_in,
            finished_at: Instant::now(),
        },
    );
    Ok(FinishedSignIn {
        id,
        auth,
        expires_in_seconds: SIGN_IN_KEPT.as_secs(),
    })
}

#[tauri::command]
pub fn cancel_mcp_sign_in(state: State<'_, McpSignIns>) {
    if let Some(cancel) = state.cancel.lock().unwrap().take() {
        let _ = cancel.send(());
    }
}

/// Keeps tokens renewed while probing with a sign-in that is not saved yet,
/// so a rotated refresh token is what gets saved.
struct FinishedStore {
    finished: FinishedMap,
    id: String,
}

struct FinishedLease {
    guard: tokio::sync::OwnedMutexGuard<HashMap<String, Finished>>,
    id: String,
    secret: AuthSecret,
}

#[async_trait]
impl CredentialStore for FinishedStore {
    async fn lease(&self) -> EngineResult<Box<dyn CredentialLease>> {
        let guard = self.finished.clone().lock_owned().await;
        let secret = guard
            .get(&self.id)
            .map(|entry| entry.sign_in.secret.clone())
            .ok_or_else(|| EngineError::ToolExecution(oauth::SIGN_IN_AGAIN.into()))?;
        Ok(Box::new(FinishedLease {
            guard,
            id: self.id.clone(),
            secret,
        }))
    }
}

#[async_trait]
impl CredentialLease for FinishedLease {
    fn secret(&self) -> &AuthSecret {
        &self.secret
    }

    async fn commit(mut self: Box<Self>, secret: AuthSecret) -> EngineResult<()> {
        let id = self.id.clone();
        if let Some(entry) = self.guard.get_mut(&id) {
            entry.sign_in.secret = secret;
        }
        Ok(())
    }
}

/// Lists a server's tools. The credential is the connection's own, a finished
/// sign-in (`sign_in_id`), or the stored sign-in of the saved server
/// `server_name` while its endpoint is unchanged.
#[tauri::command]
pub async fn discover_mcp_tools(
    state: State<'_, McpSignIns>,
    mut connection: DiscoveryRequest,
    sign_in_id: Option<String>,
    server_name: Option<String>,
) -> Result<Vec<DiscoveredTool>, String> {
    let credential = if let Some(id) = sign_in_id {
        let finished = state.finished.lock().await;
        let entry = finished
            .get(&id)
            .ok_or("The sign-in expired. Sign in again.")?;
        if entry.sign_in.url != connection.url {
            return Err("The sign-in was for another endpoint URL. Sign in again.".into());
        }
        connection.auth = Some(entry.sign_in.auth.clone());
        Some(Credential {
            secret: entry.sign_in.secret.clone(),
            store: Some(Arc::new(FinishedStore {
                finished: state.finished.clone(),
                id,
            })),
        })
    } else if let (Some(name), Some(AuthConfig::Oauth { .. })) = (server_name, &connection.auth) {
        let path = config_path()?;
        let url = connection.url.clone();
        let (auth, credential) = blocking(move || {
            let sign_ins = mcp_credentials::load(&path).map_err(|error| error.to_string())?;
            // OAuth settings come from the sign-in, as when saving.
            let auth = sign_ins
                .servers
                .get(&name)
                .filter(|sign_in| sign_in.url == url)
                .map(|sign_in| sign_in.auth.clone())
                .ok_or("Sign in to this server before loading its tools.")?;
            let credential = mcp_credentials::stored(&path, &name, &url, &auth, &sign_ins);
            Ok((auth, credential))
        })
        .await?;
        connection.auth = Some(auth);
        credential
    } else {
        None
    };
    engine_core::mcp::discover_local_tools_with(connection, credential)
        .await
        .map_err(message)
}

#[tauri::command]
pub async fn list_mcp_servers() -> Result<LocalMcpServers, String> {
    blocking(|| with_sign_ins(mcp_config::list_servers().map_err(|error| error.to_string())?)).await
}

/// Saves a server, with a finished sign-in when `sign_in_id` is given.
#[tauri::command]
pub async fn save_mcp_server(
    state: State<'_, McpSignIns>,
    server: ServerConfig,
    original_name: Option<String>,
    sign_in_id: Option<String>,
) -> Result<LocalMcpServers, String> {
    // Held until saved, so the sign-in cannot be renewed or used twice meanwhile.
    let mut finished = state.finished.lock().await;
    let sign_in = match &sign_in_id {
        Some(id) => Some(
            finished
                .get(id)
                .map(|entry| entry.sign_in.clone())
                .ok_or("The sign-in expired. Sign in again.")?,
        ),
        None => None,
    };
    let config = blocking(move || {
        mcp_config::save_server_signed_in(server, original_name, sign_in)
            .map_err(|error| error.to_string())
    })
    .await?;
    if let Some(id) = sign_in_id {
        finished.remove(&id);
    }
    drop(finished);
    blocking(|| with_sign_ins(config)).await
}

#[tauri::command]
pub async fn delete_mcp_server(name: String) -> Result<LocalMcpServers, String> {
    blocking(move || {
        with_sign_ins(mcp_config::delete_server(name).map_err(|error| error.to_string())?)
    })
    .await
}
