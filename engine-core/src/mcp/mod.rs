//! Administrator-configured MCP tools over Streamable HTTP.
pub mod auth;
pub mod network;
pub mod oauth;

pub use auth::{
    AuthConfig, AuthSecret, ClientAuth, Credential, CredentialLease, CredentialStore, MemoryStore,
};

use crate::tool_cache::{LocalToolCacheContext, ToolCacheContext, ToolCachePolicy};
use crate::{
    EngineError, EngineResult, ToolCall, ToolDefinition, ToolExecutor, ToolExecutorRegistry,
    ToolResult,
};
use async_trait::async_trait;
use futures::{stream, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashSet},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
};
use transport::Transport;
use uuid::Uuid;

fn error(message: &str) -> EngineError {
    EngineError::ToolExecution(message.into())
}

pub const AUTH_REJECTED: &str =
    "MCP server rejected the credentials (HTTP 401 or 403); credentials redacted";

/// The upstream refused the credentials, or a sign-in can no longer be
/// renewed: both call for new credentials rather than a retry.
pub fn is_auth_rejected(error: &EngineError) -> bool {
    matches!(error, EngineError::ToolExecution(message)
        if message.starts_with(AUTH_REJECTED) || message == oauth::SIGN_IN_AGAIN)
}

/// Why an upstream refused the credentials: the HTTP status, the `error` of
/// its Bearer challenge (RFC 6750, such as `insufficient_scope`) and the reason
/// of a Google API error (such as `SERVICE_DISABLED`). Only short codes are
/// kept, never upstream text, so the details are safe to show administrators.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct AuthRejection {
    pub status: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

pub(crate) fn rejection_code(value: &str) -> Option<String> {
    (!value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-')))
    .then(|| value.to_owned())
}

impl AuthRejection {
    fn message(&self) -> String {
        let mut message = format!("{AUTH_REJECTED}: status={}", self.status);
        for (name, value) in [("error", &self.error), ("reason", &self.reason)] {
            if let Some(value) = value {
                message.push_str(&format!(" {name}={value}"));
            }
        }
        message
    }
}

/// The details of a refusal by the upstream, or `None` for other failures,
/// including a sign-in that could not be renewed.
pub fn auth_rejection(error: &EngineError) -> Option<AuthRejection> {
    let EngineError::ToolExecution(message) = error else {
        return None;
    };
    let details = message.strip_prefix(AUTH_REJECTED)?.strip_prefix(": ")?;
    let mut rejection = AuthRejection::default();
    for part in details.split(' ') {
        match part.split_once('=')? {
            ("status", value) => rejection.status = value.parse().ok()?,
            ("error", value) => rejection.error = rejection_code(value),
            ("reason", value) => rejection.reason = rejection_code(value),
            _ => {}
        }
    }
    Some(rejection)
}

fn valid_env_name(name: &str) -> bool {
    !name.is_empty() && name.trim() == name && !name.contains(['=', '\0'])
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct McpConfig {
    pub servers: Vec<ServerConfig>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    pub name: String,
    pub url: String,
    pub bearer_token_env: Option<String>,
    /// Local configuration credential. Cloud transport receives its token separately.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bearer_token: Option<String>,
    /// How to authenticate. Absent means a bearer token when one is
    /// configured, and no authentication otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<AuthConfig>,
    /// Local credential for `auth`: the bearer token, header value, password,
    /// client secret, service account key or AWS secret access key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret_env: Option<String>,
    /// AWS session token for temporary credentials.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_token_env: Option<String>,
    /// Non-secret headers sent with every request.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub allow_loopback_http: bool,
    pub timeout_ms: u64,
    pub access: Access,
    pub tools: BTreeMap<String, ToolPolicy>,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum Access {
    Local {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        project_id: Option<Uuid>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        project_ids: Vec<Uuid>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        project_path: Option<String>,
    },
    Cloud {
        workspace_id: Uuid,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        project_id: Option<Uuid>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        project_ids: Vec<Uuid>,
        principals: Vec<Uuid>,
    },
}

impl Access {
    pub fn projects(&self) -> Vec<Uuid> {
        match self {
            Self::Local {
                project_id,
                project_ids,
                ..
            }
            | Self::Cloud {
                project_id,
                project_ids,
                ..
            } => project_id
                .iter()
                .copied()
                .chain(project_ids.iter().copied())
                .collect(),
        }
    }

    pub fn allows_project(&self, project: Uuid) -> bool {
        !project.is_nil() && self.projects().contains(&project)
    }

    fn valid_projects(&self) -> bool {
        let ids = self.projects();
        let exclusive = match self {
            Self::Local {
                project_id,
                project_ids,
                ..
            }
            | Self::Cloud {
                project_id,
                project_ids,
                ..
            } => project_id.is_none() || project_ids.is_empty(),
        };
        exclusive
            && !ids.is_empty()
            && ids.len() <= 256
            && ids.iter().all(|id| !id.is_nil())
            && ids.iter().collect::<HashSet<_>>().len() == ids.len()
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolPolicy {
    #[serde(default)]
    pub action: crate::action_gate::ActionRule,
    /// Explicit administrator decision; upstream annotations never grant permission.
    pub read_only: bool,
    pub cache: Option<ToolCachePolicy>,
}

/// Read one bounded server object from a file, or stdin when explicitly selected by the CLI.
pub fn read_server_config(path: Option<&std::path::Path>) -> EngineResult<ServerConfig> {
    use std::io::Read;
    let reader: Box<dyn Read> = match path {
        Some(path) => {
            Box::new(std::fs::File::open(path).map_err(|_| error("cannot read server JSON file"))?)
        }
        None => Box::new(std::io::stdin()),
    };
    let mut bytes = Vec::new();
    reader
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| error("cannot read server JSON"))?;
    if bytes.len() > 1024 * 1024 {
        return Err(error("server JSON exceeds 1 MiB"));
    }
    serde_json::from_slice(&bytes).map_err(|_| error("invalid server JSON object"))
}

/// No configuration means no upstream tools. Invalid configured servers fail startup.
pub async fn configured_registry(cloud: bool) -> EngineResult<ToolExecutorRegistry> {
    match std::env::var("REINCORE_MCP_CONFIG") {
        Err(std::env::VarError::NotPresent) => Ok(ToolExecutorRegistry::new()),
        Err(_) => Err(error("invalid REINCORE_MCP_CONFIG")),
        Ok(path) => {
            let bytes = tokio::fs::read(path)
                .await
                .map_err(|_| error("cannot read MCP configuration"))?;
            let config: McpConfig = serde_json::from_slice(&bytes)
                .map_err(|_| error("invalid MCP configuration JSON"))?;
            config.connect(cloud).await
        }
    }
}

impl McpConfig {
    pub async fn connect(self, cloud: bool) -> EngineResult<ToolExecutorRegistry> {
        self.connect_with(cloud, |_| None).await
    }

    /// Connects every server, taking credentials the host stores outside the
    /// configuration (OAuth sign-ins) from `credential`. Servers it returns
    /// nothing for use the configuration's own credential sources.
    pub async fn connect_with(
        self,
        cloud: bool,
        credential: impl Fn(&ServerConfig) -> Option<Credential>,
    ) -> EngineResult<ToolExecutorRegistry> {
        if self.servers.len() > 32 {
            return Err(error("too many MCP servers"));
        }
        let mut servers = self.servers;
        let mut names = HashSet::new();
        // Validate the complete configuration before contacting any upstream.
        for server in &mut servers {
            server.validate(cloud)?;
            let scopes = match &server.access {
                Access::Cloud { workspace_id, .. } => server
                    .access
                    .projects()
                    .into_iter()
                    .map(|id| Some((*workspace_id, id)))
                    .collect(),
                Access::Local { .. } => vec![None],
            };
            for scope in scopes {
                if !names.insert((scope, server.name.clone())) {
                    return Err(error("duplicate MCP server name in project"));
                }
            }
        }
        let servers: Vec<_> = servers
            .into_iter()
            .map(|server| {
                let credential = credential(&server);
                (server, credential)
            })
            .collect();
        let executors: Vec<Option<McpExecutor>> = stream::iter(servers)
            .map(|(server, credential)| async move {
                let name = server.name.clone();
                match tokio::time::timeout(
                    std::time::Duration::from_secs(125),
                    McpExecutor::connect_with(server, cloud, credential),
                )
                .await
                {
                    Err(_) => Err(error("MCP initialization deadline exceeded")),
                    // Sign-ins expire and credentials get revoked in normal use;
                    // that server's tools are left out instead of every server's.
                    Ok(Err(failure)) if is_auth_rejected(&failure) => {
                        tracing::warn!(server = %name, "MCP server skipped: {failure}");
                        Ok(None)
                    }
                    Ok(connected) => connected.map(Some),
                }
            })
            .buffered(8)
            .try_collect()
            .await?;
        let mut registry = ToolExecutorRegistry::new();
        for executor in executors.into_iter().flatten() {
            registry = registry.register(Arc::new(executor));
        }
        Ok(registry)
    }
}

impl ServerConfig {
    /// Validate persisted configuration without connecting or reading credentials.
    pub fn validate(&mut self, cloud: bool) -> EngineResult<()> {
        if cloud && (self.bearer_token.is_some() || self.secret.is_some()) {
            return Err(error("inline MCP credentials are local only"));
        }
        if self.name.is_empty()
            || self.name.contains("__")
            || self.name.len() > 32
            || !self
                .name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            return Err(error("invalid MCP server name"));
        }
        let valid_projects = self.access.valid_projects();
        match &mut self.access {
            Access::Local {
                project_id,
                project_ids,
                project_path,
            } if !cloud => {
                if !(valid_projects && project_path.is_none()
                    || project_id.is_none() && project_ids.is_empty() && project_path.is_some())
                {
                    return Err(error(
                        "local MCP requires project_ids (or legacy project_id) OR project_path",
                    ));
                }
                if let Some(project_path) = project_path {
                    let path = std::path::Path::new(project_path);
                    if !path.is_absolute() || !path.is_dir() {
                        return Err(error(
                            "MCP project path must be an existing absolute directory",
                        ));
                    }
                    *project_path = std::fs::canonicalize(path)
                        .ok()
                        .and_then(|p| p.to_str().map(str::to_owned))
                        .ok_or_else(|| error("MCP project path cannot be resolved"))?;
                }
            }
            Access::Cloud {
                principals,
                workspace_id,
                ..
            } if cloud
                && !principals.is_empty()
                && principals.len() <= 256
                && principals.iter().collect::<HashSet<_>>().len() == principals.len()
                && !workspace_id.is_nil()
                && valid_projects
                && principals.iter().all(|p| !p.is_nil()) => {}
            _ => {
                return Err(error(
                    "MCP access must explicitly match engine mode and authorized principals",
                ))
            }
        }

        if self.tools.is_empty() || self.tools.len() > 256 {
            return Err(error("MCP requires a bounded tool allowlist"));
        }

        for (name, policy) in &self.tools {
            if name.is_empty() || name.len() > 128 {
                return Err(error("invalid MCP tool name"));
            }
            if let Some(cache) = &policy.cache {
                if !policy.read_only || !cache.is_valid() {
                    return Err(error(
                        "invalid MCP cache policy; mutations cannot be cached",
                    ));
                }
            }
        }

        self.validate_connection(cloud)?;
        Ok(())
    }

    pub fn effective_auth(&self) -> AuthConfig {
        self.auth.clone().unwrap_or(
            if self.bearer_token.is_some() || self.bearer_token_env.is_some() {
                AuthConfig::Bearer
            } else {
                AuthConfig::None
            },
        )
    }

    /// Endpoint, authentication settings and local credential sources,
    /// without reading environment variables or contacting the server.
    fn validate_connection(&self, cloud: bool) -> EngineResult<url::Url> {
        let url = transport::validate_endpoint(self)?;
        let auth = self.effective_auth();
        auth.validate(cloud)?;
        auth::validate_headers(&self.headers)?;
        for name in [
            &self.bearer_token_env,
            &self.secret_env,
            &self.session_token_env,
        ]
        .into_iter()
        .flatten()
        {
            if !valid_env_name(name) {
                return Err(error(
                    "invalid MCP credential environment variable reference",
                ));
            }
        }
        let sources = [
            self.bearer_token.is_some(),
            self.bearer_token_env.is_some(),
            self.secret.is_some(),
            self.secret_env.is_some(),
        ];
        if sources.iter().filter(|set| **set).count() > 1 {
            return Err(error(
                "configure one of bearer_token, bearer_token_env, secret or secret_env",
            ));
        }
        if (self.bearer_token.is_some() || self.bearer_token_env.is_some())
            && auth != AuthConfig::Bearer
        {
            return Err(error("bearer_token applies only to bearer authentication"));
        }
        if self.session_token_env.is_some() && !matches!(auth, AuthConfig::AwsSigv4 { .. }) {
            return Err(error("session_token_env applies only to AWS SigV4"));
        }
        if let Some(value) = self.bearer_token.as_ref().or(self.secret.as_ref()) {
            auth.validate_secret(&AuthSecret::from_secret(value.clone()))?;
        }
        // Sign-in tokens are never configuration: the host supplies them from
        // its own store when it connects (see McpConfig::connect_with).
        if matches!(auth, AuthConfig::Oauth { .. })
            && (self.secret.is_some() || self.secret_env.is_some())
        {
            return Err(error(
                "OAuth sign-in tokens come from signing in, not from secret or secret_env",
            ));
        }
        if !network::loopback(&url) && self.auth.is_none() && auth == AuthConfig::None {
            return Err(error(
                "remote MCP requires authentication; set auth to {\"type\":\"none\"} for a public server",
            ));
        }
        Ok(url)
    }

    /// The credential local configuration names, read from inline values or
    /// environment variables.
    fn local_credential(&self) -> EngineResult<Option<Credential>> {
        let read = |name: &Option<String>| {
            name.as_ref()
                .map(|name| {
                    std::env::var(name)
                        .map_err(|_| error("MCP credential environment variable missing"))
                })
                .transpose()
        };
        let secret = match self.bearer_token.as_ref().or(self.secret.as_ref()) {
            Some(value) => Some(value.clone()),
            None => read(&self.bearer_token_env)?.or(read(&self.secret_env)?),
        };
        let session_token = read(&self.session_token_env)?;
        Ok((secret.is_some() || session_token.is_some()).then(|| {
            Credential::new(AuthSecret {
                secret,
                session_token,
                ..AuthSecret::default()
            })
        }))
    }
}

/// Connection-only input for local configuration preview; does not grant tool access.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryRequest {
    pub url: String,
    pub bearer_token: Option<String>,
    pub bearer_token_env: Option<String>,
    #[serde(default)]
    pub auth: Option<AuthConfig>,
    #[serde(default)]
    pub secret: Option<String>,
    #[serde(default)]
    pub secret_env: Option<String>,
    #[serde(default)]
    pub session_token_env: Option<String>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub allow_loopback_http: bool,
    pub timeout_ms: u64,
}

#[derive(Serialize)]
pub struct DiscoveredTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// Initialize and list tools without executing tools or persisting configuration.
pub async fn discover_local_tools(request: DiscoveryRequest) -> EngineResult<Vec<DiscoveredTool>> {
    discover_local_tools_with(request, None).await
}

/// Like `discover_local_tools`, with a host-stored credential (an OAuth
/// sign-in, finished or saved) in place of the request's credential sources.
pub async fn discover_local_tools_with(
    request: DiscoveryRequest,
    credential: Option<Credential>,
) -> EngineResult<Vec<DiscoveredTool>> {
    let config = ServerConfig {
        name: String::new(),
        url: request.url,
        bearer_token: request.bearer_token,
        bearer_token_env: request.bearer_token_env,
        auth: request.auth,
        secret: request.secret,
        secret_env: request.secret_env,
        session_token_env: request.session_token_env,
        headers: request.headers,
        allow_loopback_http: request.allow_loopback_http,
        timeout_ms: request.timeout_ms,
        access: Access::Local {
            project_id: None,
            project_ids: Vec::new(),
            project_path: None,
        },
        tools: BTreeMap::new(),
    };
    config.validate_connection(false)?;
    tokio::time::timeout(std::time::Duration::from_millis(config.timeout_ms), async {
        let transport = Arc::new(Transport::connect(&config, credential).await?);
        discovered_from_inventory(discover_inventory(&transport).await?)
    })
    .await
    .map_err(|_| error("MCP discovery timed out"))?
}

/// Connection-only input for cloud configuration preview; does not grant tool access.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CloudDiscoveryRequest {
    pub url: String,
    pub bearer_token: Option<String>,
    #[serde(default)]
    pub auth: Option<AuthConfig>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    pub timeout_ms: u64,
}

/// Initialize and list tools over tenant-supplied credentials without persisting
/// configuration or granting tool access. Only public HTTPS endpoints are
/// reachable; deployment environment secrets are never consulted.
pub async fn discover_cloud_tools(
    request: CloudDiscoveryRequest,
    credential: Option<Credential>,
) -> EngineResult<Vec<DiscoveredTool>> {
    let auth = request.auth.unwrap_or(if request.bearer_token.is_some() {
        AuthConfig::Bearer
    } else {
        AuthConfig::None
    });
    let credential = match (credential, request.bearer_token) {
        (Some(_), Some(_)) => return Err(error("configure one MCP credential")),
        (Some(credential), None) => Some(credential),
        (None, Some(token)) => {
            let secret = AuthSecret::from_secret(token);
            auth.validate_secret(&secret)?;
            Some(Credential::new(secret))
        }
        (None, None) => None,
    };
    let config = ServerConfig {
        name: String::new(),
        url: request.url,
        bearer_token: None,
        bearer_token_env: None,
        auth: Some(auth),
        secret: None,
        secret_env: None,
        session_token_env: None,
        headers: request.headers,
        allow_loopback_http: false,
        timeout_ms: request.timeout_ms,
        access: Access::Cloud {
            workspace_id: Uuid::nil(),
            project_id: None,
            project_ids: Vec::new(),
            principals: Vec::new(),
        },
        tools: BTreeMap::new(),
    };
    network::validate_cloud_url(&config.url)?;
    config.validate_connection(true)?;
    tokio::time::timeout(std::time::Duration::from_millis(config.timeout_ms), async {
        let transport = Arc::new(Transport::connect_cloud(&config, credential).await?);
        discovered_from_inventory(discover_inventory(&transport).await?)
    })
    .await
    .map_err(|_| error("MCP discovery timed out"))?
}

/// Bounded conversion of a raw tools/list inventory into discovery results.
fn discovered_from_inventory(
    inventory: BTreeMap<String, Value>,
) -> EngineResult<Vec<DiscoveredTool>> {
    inventory
        .into_iter()
        .map(|(name, tool)| {
            if name.is_empty() || name.len() > 128 {
                return Err(error("invalid MCP tool name"));
            }
            let input_schema = tool
                .get("inputSchema")
                .filter(|schema| schema.is_object())
                .ok_or_else(|| error("invalid MCP input schema"))?
                .clone();
            Ok(DiscoveredTool {
                name,
                description: tool
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned(),
                input_schema,
            })
        })
        .collect()
}

async fn discover_inventory(transport: &Arc<Transport>) -> EngineResult<BTreeMap<String, Value>> {
    let mut discovered = BTreeMap::new();
    let mut discovery_bytes = 0;
    let mut cursor = None;
    let mut cursors = HashSet::new();

    for page in 0..32 {
        let result = transport
            .rpc(
                "tools/list",
                cursor
                    .as_ref()
                    .map(|c| json!({"cursor": c}))
                    .unwrap_or(json!({})),
            )
            .await?;

        discovery_bytes += result.to_string().len();
        if discovery_bytes > 4 * 1024 * 1024 {
            return Err(error("MCP discovery exceeds aggregate 4 MiB limit"));
        }
        let tools = result
            .get("tools")
            .and_then(Value::as_array)
            .ok_or_else(|| error("invalid MCP discovery response"))?;

        for tool in tools {
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| error("invalid MCP tool definition"))?;
            if discovered.insert(name.to_owned(), tool.clone()).is_some() {
                return Err(error("duplicate discovered MCP tool"));
            }
        }

        if discovered.len() > 4096 {
            return Err(error("MCP discovery exceeds tool limit"));
        }

        cursor = match result.get("nextCursor") {
            None => None,
            Some(Value::String(value)) if !value.is_empty() && value.len() <= 4096 => {
                Some(value.clone())
            }
            _ => return Err(error("invalid MCP discovery cursor")),
        };

        if cursor.is_none() {
            break;
        }

        if page == 31 || !cursors.insert(cursor.clone()) {
            return Err(error("MCP discovery pagination exceeded limit"));
        }
    }

    Ok(discovered)
}

pub struct McpExecutor {
    config: ServerConfig,
    transport: Arc<Transport>,
    definitions: Vec<ToolDefinition>,
    epoch: String,
    generation: AtomicU64,
    mutations: AtomicUsize,
    uncertain: AtomicBool,
}

impl McpExecutor {
    /// Invalidated sessions and changed tool metadata must be rediscovered.
    pub fn metadata_current(&self) -> bool {
        self.transport.metadata_current()
    }

    pub async fn connect(config: ServerConfig, cloud: bool) -> EngineResult<Self> {
        Self::connect_with(config, cloud, None).await
    }

    /// Like `connect`, with a host-stored credential (an OAuth sign-in)
    /// taking the place of the configuration's credential sources.
    pub async fn connect_with(
        mut config: ServerConfig,
        cloud: bool,
        credential: Option<Credential>,
    ) -> EngineResult<Self> {
        config.validate(cloud)?;

        let transport = Transport::connect(&config, credential).await?;
        Self::discover(config, transport).await
    }

    /// Tenant credentials are passed directly; no deployment environment lookup.
    pub async fn connect_cloud(
        mut config: ServerConfig,
        credential: Option<Credential>,
    ) -> EngineResult<Self> {
        network::validate_cloud_url(&config.url)?;
        config.validate(true)?;
        let transport = Transport::connect_cloud(&config, credential).await?;
        Self::discover(config, transport).await
    }

    async fn discover(config: ServerConfig, transport: Transport) -> EngineResult<Self> {
        let transport = Arc::new(transport);
        let discovered = discover_inventory(&transport).await?;

        let mut definitions = Vec::new();
        for (name, policy) in &config.tools {
            let tool = discovered
                .get(name)
                .ok_or_else(|| error("configured MCP tool was not discovered"))?;
            let schema = tool
                .get("inputSchema")
                .filter(|s| s.is_object())
                .ok_or_else(|| error("invalid MCP input schema"))?;
            definitions.push(ToolDefinition {
                action_contract: tool
                    .get("_meta")
                    .and_then(|meta| meta.get("reincore/action_contract"))
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(|_| error("invalid MCP action contract"))?,
                name: format!("{}__{}", config.name, name),
                description: tool
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned(),
                parameters: schema.clone(),
                safe_to_memoize: policy.cache.is_some(),
                memoization_policy: None,
            });
        }

        Ok(Self {
            config,
            transport,
            definitions,
            epoch: Uuid::new_v4().to_string(),
            generation: AtomicU64::new(0),
            mutations: AtomicUsize::new(0),
            uncertain: AtomicBool::new(false),
        })
    }

    fn cloud_allowed(&self, c: &ToolCacheContext) -> bool {
        matches!(
            &self.config.access,
            Access::Cloud { workspace_id, principals, .. }
                if *workspace_id == c.workspace_id
                    && self.config.access.allows_project(c.project_id)
                    && principals.contains(&c.user_id)
        )
    }

    fn local_allowed(&self, c: &LocalToolCacheContext) -> bool {
        match &self.config.access {
            Access::Local {
                project_path: None, ..
            } => c
                .project_id
                .is_some_and(|id| self.config.access.allows_project(id)),
            Access::Local {
                project_id: None,
                project_ids,
                project_path: Some(path),
            } if project_ids.is_empty() => c.project_path.as_ref() == Some(path),
            _ => false,
        }
    }

    fn tool(&self, call: &ToolCall) -> EngineResult<(&str, &ToolPolicy)> {
        let name = call
            .function
            .name
            .strip_prefix(&format!("{}__", self.config.name))
            .ok_or_else(|| error("MCP tool not allowed"))?;
        self.config
            .tools
            .get_key_value(name)
            .map(|(n, p)| (n.as_str(), p))
            .ok_or_else(|| error("MCP tool not allowed"))
    }

    fn policy(&self, call: &ToolCall) -> EngineResult<Option<ToolCachePolicy>> {
        let (_, tool) = self.tool(call)?;
        crate::action_gate::validate_call(call, &self.definitions)?;
        let args = serde_json::from_str(&call.function.arguments)
            .map_err(|_| error("invalid tool arguments"))?;
        tool.action.check(&args)?;
        if !self.transport.metadata_current()
            || self.mutations.load(Ordering::SeqCst) != 0
            || self.uncertain.load(Ordering::SeqCst)
        {
            return Ok(None);
        }
        tool.cache
            .clone()
            .map(|mut p| {
                // Boot namespace prevents persisted entries surviving configuration/credential reloads.
                p.data_version = crate::CacheKey::for_tool_call(
                    "mcp-data",
                    &json!([
                        p.data_version,
                        self.epoch,
                        self.generation.load(Ordering::SeqCst)
                    ]),
                )?
                .0;
                Ok(p)
            })
            .transpose()
    }

    async fn dispatch(&self, call: &ToolCall) -> EngineResult<ToolResult> {
        if !self.transport.metadata_current() {
            return Err(error("MCP tool metadata changed; restart to rediscover"));
        }
        let (name, policy) = self.tool(call)?;
        crate::action_gate::validate_call(call, &self.definitions)?;
        let arguments: Value = serde_json::from_str(&call.function.arguments)
            .map_err(|_| error("invalid MCP arguments JSON"))?;
        if !arguments.is_object() {
            return Err(error("MCP arguments must be an object"));
        }
        policy.action.check(&arguments)?;
        let mut mutation = if !policy.read_only {
            self.mutations.fetch_add(1, Ordering::SeqCst);
            self.generation.fetch_add(1, Ordering::SeqCst);
            Some(Mutation {
                executor: self,
                complete: false,
            })
        } else {
            None
        };
        let mut params = json!({"name": name, "arguments": arguments});
        if self
            .definitions
            .iter()
            .any(|tool| tool.name == call.function.name && tool.action_contract.is_some())
        {
            params["_meta"] = json!({"reincore/tool_call_id":call.id});
        }
        let result = self.transport.rpc("tools/call", params).await?;
        if !result.get("content").is_some_and(Value::is_array)
            || result.get("isError").is_some_and(|v| !v.is_boolean())
        {
            return Err(error("invalid MCP tool result; outcome may be unknown"));
        }
        if let Some(mutation) = &mut mutation {
            mutation.complete = true;
        }
        Ok(ToolResult {
            tool_call_id: call.id.clone(),
            role: "tool".into(),
            name: call.function.name.clone(),
            content: result.to_string(),
            is_error: Some(
                result
                    .get("isError")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            ),
        })
    }
}

struct Mutation<'a> {
    executor: &'a McpExecutor,
    complete: bool,
}

impl Drop for Mutation<'_> {
    fn drop(&mut self) {
        if !self.complete {
            self.executor.uncertain.store(true, Ordering::SeqCst);
        }
        self.executor.generation.fetch_add(1, Ordering::SeqCst);
        self.executor.mutations.fetch_sub(1, Ordering::SeqCst);
    }
}

#[async_trait]
impl ToolExecutor for McpExecutor {
    async fn list_tools(&self) -> EngineResult<Vec<ToolDefinition>> {
        Ok(self.definitions.clone())
    }

    async fn list_tools_in_context(
        &self,
        c: &ToolCacheContext,
    ) -> EngineResult<Vec<ToolDefinition>> {
        Ok(if self.cloud_allowed(c) {
            self.definitions.clone()
        } else {
            vec![]
        })
    }

    async fn list_local_tools_in_context(
        &self,
        c: &LocalToolCacheContext,
    ) -> EngineResult<Vec<ToolDefinition>> {
        Ok(if self.local_allowed(c) {
            self.definitions.clone()
        } else {
            vec![]
        })
    }

    async fn cache_policy(
        &self,
        call: &ToolCall,
        c: &ToolCacheContext,
    ) -> EngineResult<Option<ToolCachePolicy>> {
        if !self.cloud_allowed(c) {
            return Err(error("MCP access denied"));
        }
        self.policy(call)
    }

    async fn local_cache_policy(
        &self,
        call: &ToolCall,
        c: &LocalToolCacheContext,
    ) -> EngineResult<Option<ToolCachePolicy>> {
        if !self.local_allowed(c) {
            return Err(error("MCP access denied"));
        }
        self.policy(call)
    }

    async fn execute_in_context(
        &self,
        call: &ToolCall,
        c: &ToolCacheContext,
    ) -> EngineResult<ToolResult> {
        if !self.cloud_allowed(c) {
            return Err(error("MCP access denied"));
        }
        self.dispatch(call).await
    }

    async fn execute_local_in_context(
        &self,
        call: &ToolCall,
        c: &LocalToolCacheContext,
    ) -> EngineResult<ToolResult> {
        if !self.local_allowed(c) {
            return Err(error("MCP access denied"));
        }
        self.dispatch(call).await
    }

    async fn execute(&self, _: &ToolCall) -> EngineResult<ToolResult> {
        Err(error("MCP execution requires authorized session context"))
    }
}

#[cfg(test)]
mod tests;
mod transport;
