//! OAuth for upstream MCP servers: authorization discovery (RFC 9728 protected
//! resource metadata, RFC 8414 server metadata), dynamic client registration
//! (RFC 7591), user sign-in with PKCE and refresh, client credentials, and
//! Google service account assertions (RFC 7523).
use super::{
    auth::{self, validate_scopes, AuthConfig, AuthSecret, ClientAuth, CredentialStore},
    error, network, transport,
};
use crate::EngineResult;
use base64::{
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
    Engine as _,
};
use futures::StreamExt;
use regex::Regex;
use reqwest::{
    header::{ACCEPT, AUTHORIZATION, WWW_AUTHENTICATE},
    StatusCode,
};
use ring::rand::SecureRandom;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};

const LIMIT: usize = 1024 * 1024;
const EXPIRY_MARGIN_SECONDS: i64 = 60;
// Token requests run while a credential is leased (a row or file lock), so a
// hung authorization server must not hold it for long.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

pub const SIGN_IN_AGAIN: &str = "MCP sign-in expired or was revoked; sign in to the server again";
pub const CLIENT_ID_REQUIRED: &str =
    "This MCP server does not register clients automatically; enter an OAuth client ID";
pub const SIGN_IN_UNAVAILABLE: &str = "This MCP server does not publish OAuth sign-in endpoints";
// Token endpoint refusals a person can act on.
pub const CLIENT_REJECTED: &str = "OAuth client rejected; check the client ID and secret";
pub const GRANT_UNSUPPORTED: &str =
    "The authorization server does not let this client get tokens this way; client credentials are not supported by Google, for example";
pub const SCOPES_REJECTED: &str = "OAuth scopes rejected by the authorization server";
pub const MANAGED_CLIENT_UNAVAILABLE: &str =
    "ReinCore's Google sign-in is not available for this server; use your own OAuth client";

pub const GOOGLE_TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";
const MANAGED_CLIENT_ID: &str = "REINCORE_MCP_GOOGLE_CLIENT_ID";
const MANAGED_CLIENT_SECRET: &str = "REINCORE_MCP_GOOGLE_CLIENT_SECRET";

/// ReinCore's own Google OAuth client, from `REINCORE_MCP_GOOGLE_CLIENT_ID`
/// and `REINCORE_MCP_GOOGLE_CLIENT_SECRET`, so administrators can sign in to
/// Google MCP servers without creating a client. Blank values mean unset.
pub fn managed_google_client() -> Option<(String, String)> {
    let read = |name: &str, max: usize| {
        std::env::var(name)
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| visible(value, max))
    };
    Some((
        read(MANAGED_CLIENT_ID, 512)?,
        read(MANAGED_CLIENT_SECRET, 4096)?,
    ))
}

fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

fn random_token() -> EngineResult<String> {
    let mut bytes = [0u8; 32];
    ring::rand::SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| error("secure random unavailable"))?;
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}

fn visible(value: &str, max: usize) -> bool {
    !value.is_empty() && value.len() <= max && value.bytes().all(|b| (0x21..=0x7e).contains(&b))
}

async fn read_limited(response: reqwest::Response) -> EngineResult<Vec<u8>> {
    let mut stream = response.bytes_stream();
    let mut buffer = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| error("OAuth response interrupted"))?;
        if buffer.len() + chunk.len() > LIMIT {
            return Err(error("OAuth response too large"));
        }
        buffer.extend_from_slice(&chunk);
    }
    Ok(buffer)
}

async fn get_json(client: &reqwest::Client, url: &str, public_only: bool) -> Option<Value> {
    let url = network::validate_endpoint_url(url, public_only).ok()?;
    let response = client
        .get(url)
        .header(ACCEPT, "application/json")
        .send()
        .await
        .ok()?;
    if response.status() != StatusCode::OK {
        return None;
    }
    serde_json::from_slice::<Value>(&read_limited(response).await.ok()?)
        .ok()
        .filter(Value::is_object)
}

fn strings(document: &Value, field: &str) -> Vec<String> {
    document
        .get(field)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .filter(|value| visible(value, 256))
                .take(256)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// What an MCP server publishes about signing in to it.
#[derive(Clone, Debug, Default, Serialize)]
pub struct AuthorizationMetadata {
    /// An unauthenticated initialize was refused with 401 or 403.
    pub requires_auth: bool,
    /// The resource indicator (RFC 8707) tokens are requested for.
    pub resource: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issuer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authorization_endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_endpoint: Option<String>,
    /// Clients register themselves, so no client ID has to be entered.
    pub dynamic_registration: bool,
    /// ReinCore's own Google OAuth client can sign in to this server.
    pub managed_client: bool,
    pub scopes_supported: Vec<String>,
    /// Scopes the server asked for when it refused the request.
    pub challenge_scopes: Vec<String>,
    pub grant_types_supported: Vec<String>,
    #[serde(skip)]
    registration_endpoint: Option<String>,
    #[serde(skip)]
    code_challenge_methods: Vec<String>,
    #[serde(skip)]
    client_auth_methods: Vec<String>,
    #[serde(skip)]
    server_scopes: Vec<String>,
}

struct Challenge {
    requires_auth: bool,
    resource_metadata: Option<String>,
    scopes: Vec<String>,
}

/// The `error` of a `Bearer` challenge, such as `invalid_token`.
pub(super) fn bearer_error(header: &str) -> Option<String> {
    bearer_parameters(header)?
        .into_iter()
        .find_map(|(name, value)| (name == "error").then_some(value))
}

/// Parameters of a `Bearer` challenge in a WWW-Authenticate header.
fn bearer_challenge(header: &str) -> Option<(Option<String>, Vec<String>)> {
    let mut resource_metadata = None;
    let mut scopes = Vec::new();
    for (name, value) in bearer_parameters(header)? {
        match name.as_str() {
            "resource_metadata" if resource_metadata.is_none() => resource_metadata = Some(value),
            "scope" => {
                scopes = value
                    .split_whitespace()
                    .filter(|scope| auth::valid_scope(scope))
                    .map(str::to_owned)
                    .collect()
            }
            _ => {}
        }
    }
    Some((resource_metadata, scopes))
}

/// The parameters of the first `Bearer` challenge, names lowercased.
fn bearer_parameters(header: &str) -> Option<Vec<(String, String)>> {
    static PARAMETER: OnceLock<Regex> = OnceLock::new();
    let parameter = PARAMETER.get_or_init(|| {
        Regex::new(r#"([A-Za-z0-9_\-]+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^\s,"]+))"#)
            .expect("valid challenge pattern")
    });
    let lower = header.to_ascii_lowercase();
    let start = lower.match_indices("bearer").find_map(|(index, _)| {
        let before = lower[..index].chars().next_back();
        let after = lower[index + 6..].chars().next();
        (before.map_or(true, |c| c == ',' || c.is_whitespace())
            && after.map_or(true, char::is_whitespace))
        .then_some(index + 6)
    })?;
    Some(
        parameter
            .captures_iter(&header[start..])
            .map(|capture| {
                let value = capture
                    .get(2)
                    .map(|quoted| quoted.as_str().replace("\\\"", "\"").replace("\\\\", "\\"))
                    .or_else(|| capture.get(3).map(|token| token.as_str().to_owned()))
                    .unwrap_or_default();
                (capture[1].to_ascii_lowercase(), value)
            })
            .collect(),
    )
}

async fn probe(client: &reqwest::Client, endpoint: &url::Url) -> EngineResult<Challenge> {
    let response = client
        .post(endpoint.clone())
        .header(ACCEPT, "application/json, text/event-stream")
        .json(&transport::initialize_request())
        .send()
        .await
        .map_err(|_| error("MCP server unreachable during authorization discovery"))?;
    let (resource_metadata, scopes) = response
        .headers()
        .get_all(WWW_AUTHENTICATE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .find_map(bearer_challenge)
        .unwrap_or_default();
    Ok(Challenge {
        requires_auth: matches!(
            response.status(),
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN
        ),
        resource_metadata,
        scopes,
    })
}

/// Well-known locations for `name` under `url`, most specific first.
fn well_known(url: &url::Url, name: &str) -> Vec<String> {
    let origin = url.origin().ascii_serialization();
    let path = url.path().trim_end_matches('/');
    let mut candidates = Vec::new();
    if !path.is_empty() {
        candidates.push(format!("{origin}/.well-known/{name}{path}"));
    }
    candidates.push(format!("{origin}/.well-known/{name}"));
    candidates
}

async fn authorization_server(
    client: &reqwest::Client,
    issuer: &str,
    public_only: bool,
) -> Option<Value> {
    let url = url::Url::parse(issuer).ok()?;
    let origin = url.origin().ascii_serialization();
    let path = url.path().trim_end_matches('/');
    // The order the MCP authorization specification gives.
    let candidates = if path.is_empty() {
        vec![
            format!("{origin}/.well-known/oauth-authorization-server"),
            format!("{origin}/.well-known/openid-configuration"),
        ]
    } else {
        vec![
            format!("{origin}/.well-known/oauth-authorization-server{path}"),
            format!("{origin}/.well-known/openid-configuration{path}"),
            format!("{origin}{path}/.well-known/openid-configuration"),
        ]
    };
    for candidate in candidates {
        if let Some(document) = get_json(client, &candidate, public_only).await {
            // RFC 8414: metadata for another issuer must not be used.
            if document
                .get("issuer")
                .and_then(Value::as_str)
                .is_some_and(|found| found.trim_end_matches('/') == issuer.trim_end_matches('/'))
            {
                return Some(document);
            }
        }
    }
    None
}

fn browser_endpoint(raw: &str, public_only: bool) -> Option<String> {
    let url = url::Url::parse(raw).ok()?;
    (raw.len() <= 2048
        && url.fragment().is_none()
        && url.username().is_empty()
        && (url.scheme() == "https"
            || !public_only && url.scheme() == "http" && network::loopback(&url)))
    .then(|| raw.to_owned())
}

/// Finds how to sign in to the MCP server at `url`. Nothing is sent that
/// identifies a user; the server sees one unauthenticated initialize.
pub async fn discover(
    url: &str,
    public_only: bool,
    timeout: Duration,
) -> EngineResult<AuthorizationMetadata> {
    let endpoint = network::validate_endpoint_url(url, public_only)?;
    let client = network::client(public_only)?;
    tokio::time::timeout(timeout, discover_with(&client, endpoint, public_only))
        .await
        .map_err(|_| error("MCP authorization discovery timed out"))?
}

async fn discover_with(
    client: &reqwest::Client,
    endpoint: url::Url,
    public_only: bool,
) -> EngineResult<AuthorizationMetadata> {
    let challenge = probe(client, &endpoint).await?;
    let mut metadata = AuthorizationMetadata {
        requires_auth: challenge.requires_auth,
        resource: endpoint.as_str().trim_end_matches('/').to_owned(),
        challenge_scopes: challenge.scopes,
        ..AuthorizationMetadata::default()
    };
    let mut candidates: Vec<String> = challenge.resource_metadata.into_iter().collect();
    candidates.extend(well_known(&endpoint, "oauth-protected-resource"));
    let mut issuer = None;
    let mut protected = false;
    for candidate in candidates {
        let Some(document) = get_json(client, &candidate, public_only).await else {
            continue;
        };
        // RFC 9728: metadata describing a resource on another origin is ignored.
        match document.get("resource").and_then(Value::as_str) {
            Some(resource)
                if url::Url::parse(resource)
                    .is_ok_and(|parsed| parsed.origin() == endpoint.origin())
                    && resource.len() <= 2048 =>
            {
                metadata.resource = resource.to_owned();
            }
            Some(_) => continue,
            None => {}
        }
        metadata.scopes_supported = strings(&document, "scopes_supported");
        issuer = strings(&document, "authorization_servers")
            .into_iter()
            .next();
        protected = true;
        break;
    }
    // Servers that predate protected resource metadata host their
    // authorization server at their own origin.
    let issuer = issuer.unwrap_or_else(|| endpoint.origin().ascii_serialization());
    if let Some(document) = authorization_server(client, &issuer, public_only).await {
        let text = |field: &str| document.get(field).and_then(Value::as_str);
        metadata.issuer = Some(issuer.clone());
        metadata.authorization_endpoint =
            text("authorization_endpoint").and_then(|raw| browser_endpoint(raw, public_only));
        metadata.token_endpoint = text("token_endpoint")
            .filter(|raw| network::validate_endpoint_url(raw, public_only).is_ok())
            .map(str::to_owned);
        metadata.registration_endpoint = text("registration_endpoint")
            .filter(|raw| network::validate_endpoint_url(raw, public_only).is_ok())
            .map(str::to_owned);
        metadata.dynamic_registration = metadata.registration_endpoint.is_some();
        metadata.managed_client = google(Some(issuer.as_str()))
            && metadata.token_endpoint.as_deref() == Some(GOOGLE_TOKEN_ENDPOINT)
            && managed_google_client().is_some();
        metadata.code_challenge_methods = strings(&document, "code_challenge_methods_supported");
        metadata.client_auth_methods = strings(&document, "token_endpoint_auth_methods_supported");
        metadata.grant_types_supported = strings(&document, "grant_types_supported");
        metadata.server_scopes = strings(&document, "scopes_supported");
        if !protected {
            metadata.scopes_supported = metadata.server_scopes.clone();
        }
    }
    Ok(metadata)
}

fn google(issuer: Option<&str>) -> bool {
    issuer.is_some_and(|issuer| issuer.trim_end_matches('/') == "https://accounts.google.com")
}

pub fn validate_redirect_uri(uri: &str) -> EngineResult<()> {
    let url = url::Url::parse(uri).map_err(|_| error("invalid OAuth redirect URI"))?;
    let local = matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"));
    if uri.len() > 2048
        || url.fragment().is_some()
        || !url.username().is_empty()
        || url.password().is_some()
        || !(url.scheme() == "https" || url.scheme() == "http" && local)
    {
        return Err(error("invalid OAuth redirect URI"));
    }
    Ok(())
}

/// A sign-in to begin. Without a client ID the client registers itself.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthorizationRequest {
    pub url: String,
    pub redirect_uri: String,
    #[serde(default)]
    pub client_id: Option<String>,
    #[serde(default)]
    pub client_secret: Option<String>,
    /// Requested scopes; defaults to what the server asks for or supports.
    #[serde(default)]
    pub scopes: Option<Vec<String>>,
    pub timeout_ms: u64,
    /// Sign in with ReinCore's own Google client instead of a client ID.
    #[serde(default)]
    pub managed_client: bool,
}

/// Non-secret state of a sign-in in progress. Its [`AuthSecret`] carries the
/// client secret and the PKCE verifier.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PendingAuthorization {
    pub url: String,
    pub redirect_uri: String,
    pub auth: AuthConfig,
}

pub struct StartedAuthorization {
    /// Where to send the user's browser.
    pub authorization_url: String,
    /// The value the redirect must return; hosts look the sign-in up by it.
    pub state: String,
    pub pending: PendingAuthorization,
    pub secret: AuthSecret,
}

async fn register(
    client: &reqwest::Client,
    endpoint: &str,
    redirect_uri: &str,
    public_only: bool,
) -> EngineResult<(String, Option<String>, ClientAuth)> {
    let failed = || error("OAuth client registration failed");
    let response = client
        .post(network::validate_endpoint_url(endpoint, public_only)?)
        .timeout(REQUEST_TIMEOUT)
        .header(ACCEPT, "application/json")
        .json(&json!({
            "client_name": "ReinCore",
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }))
        .send()
        .await
        .map_err(|_| failed())?;
    if !response.status().is_success() {
        return Err(failed());
    }
    let body: Value =
        serde_json::from_slice(&read_limited(response).await?).map_err(|_| failed())?;
    let client_id = body
        .get("client_id")
        .and_then(Value::as_str)
        .filter(|id| visible(id, 512))
        .ok_or_else(failed)?
        .to_owned();
    let client_secret = body
        .get("client_secret")
        .and_then(Value::as_str)
        .filter(|secret| visible(secret, 4096))
        .map(str::to_owned);
    let client_auth = match (
        &client_secret,
        body.get("token_endpoint_auth_method")
            .and_then(Value::as_str),
    ) {
        (None, _) => ClientAuth::None,
        (Some(_), Some("client_secret_post")) => ClientAuth::ClientSecretPost,
        (Some(_), _) => ClientAuth::ClientSecretBasic,
    };
    Ok((client_id, client_secret, client_auth))
}

/// Discovers the server's authorization server and prepares a sign-in URL.
pub async fn start(
    request: AuthorizationRequest,
    public_only: bool,
) -> EngineResult<StartedAuthorization> {
    if !(100..=120_000).contains(&request.timeout_ms) {
        return Err(error("MCP timeout_ms must be 100..120000"));
    }
    validate_redirect_uri(&request.redirect_uri)?;
    if let Some(scopes) = &request.scopes {
        validate_scopes(scopes)?;
    }
    if request
        .client_id
        .as_deref()
        .is_some_and(|id| !visible(id, 512))
        || request
            .client_secret
            .as_deref()
            .is_some_and(|secret| !visible(secret, 4096))
        || request.client_secret.is_some() && request.client_id.is_none()
        || request.managed_client && request.client_id.is_some()
    {
        return Err(error("invalid OAuth client ID or secret"));
    }
    let client = network::client(public_only)?;
    let timeout = Duration::from_millis(request.timeout_ms);
    tokio::time::timeout(timeout, async move {
        let endpoint = network::validate_endpoint_url(&request.url, public_only)?;
        let metadata = discover_with(&client, endpoint, public_only).await?;
        let (Some(authorization_endpoint), Some(token_endpoint)) =
            (&metadata.authorization_endpoint, &metadata.token_endpoint)
        else {
            return Err(error(SIGN_IN_UNAVAILABLE));
        };
        if !metadata.code_challenge_methods.is_empty()
            && !metadata
                .code_challenge_methods
                .iter()
                .any(|method| method == "S256")
        {
            return Err(error(
                "the authorization server does not support PKCE (S256)",
            ));
        }
        let managed_client = request.managed_client;
        let (client_id, client_secret, client_auth) = match request.client_id {
            // Only the client ID is kept: the secret is read again whenever
            // tokens are requested.
            None if managed_client => {
                let (client_id, _) = managed_google_client()
                    .filter(|_| {
                        google(metadata.issuer.as_deref())
                            && token_endpoint == GOOGLE_TOKEN_ENDPOINT
                    })
                    .ok_or_else(|| error(MANAGED_CLIENT_UNAVAILABLE))?;
                (client_id, None, ClientAuth::ClientSecretBasic)
            }
            Some(client_id) => {
                let methods = &metadata.client_auth_methods;
                let auth = match &request.client_secret {
                    None => ClientAuth::None,
                    Some(_)
                        if !methods.iter().any(|m| m == "client_secret_basic")
                            && methods.iter().any(|m| m == "client_secret_post") =>
                    {
                        ClientAuth::ClientSecretPost
                    }
                    Some(_) => ClientAuth::ClientSecretBasic,
                };
                (client_id, request.client_secret, auth)
            }
            None => match &metadata.registration_endpoint {
                Some(registration) => {
                    register(&client, registration, &request.redirect_uri, public_only).await?
                }
                None => return Err(error(CLIENT_ID_REQUIRED)),
            },
        };
        let mut scopes = request.scopes.unwrap_or_else(|| {
            if metadata.challenge_scopes.is_empty() {
                metadata.scopes_supported.clone()
            } else {
                metadata.challenge_scopes.clone()
            }
        });
        scopes.truncate(64);
        // Servers that model refresh as a scope only issue refresh tokens with it.
        if (metadata
            .server_scopes
            .iter()
            .chain(&metadata.scopes_supported))
        .any(|scope| scope == "offline_access")
            && !scopes.iter().any(|scope| scope == "offline_access")
            && scopes.len() < 64
        {
            scopes.push("offline_access".into());
        }
        let mut unique = std::collections::HashSet::new();
        scopes.retain(|scope| auth::valid_scope(scope) && unique.insert(scope.clone()));

        let verifier = random_token()?;
        let state = random_token()?;
        let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
        let mut url =
            url::Url::parse(authorization_endpoint).map_err(|_| error(SIGN_IN_UNAVAILABLE))?;
        {
            let mut query = url.query_pairs_mut();
            query
                .append_pair("response_type", "code")
                .append_pair("client_id", &client_id)
                .append_pair("redirect_uri", &request.redirect_uri)
                .append_pair("state", &state)
                .append_pair("code_challenge", &challenge)
                .append_pair("code_challenge_method", "S256")
                .append_pair("resource", &metadata.resource);
            if !scopes.is_empty() {
                query.append_pair("scope", &scopes.join(" "));
            }
            // Google only issues refresh tokens when asked this way. Other
            // providers would show a consent screen on every sign-in.
            if google(metadata.issuer.as_deref()) {
                query
                    .append_pair("access_type", "offline")
                    .append_pair("prompt", "consent");
            }
        }
        Ok(StartedAuthorization {
            authorization_url: url.into(),
            state,
            pending: PendingAuthorization {
                url: request.url,
                redirect_uri: request.redirect_uri,
                auth: AuthConfig::Oauth {
                    client_id,
                    token_endpoint: token_endpoint.clone(),
                    scopes,
                    resource: Some(metadata.resource.clone()),
                    client_auth,
                    issuer: metadata.issuer.clone(),
                    managed_client,
                },
            },
            secret: AuthSecret {
                secret: client_secret,
                code_verifier: Some(verifier),
                ..AuthSecret::default()
            },
        })
    })
    .await
    .map_err(|_| error("MCP authorization discovery timed out"))?
}

/// Exchanges the code the authorization server returned for tokens.
pub async fn complete(
    pending: &PendingAuthorization,
    secret: &AuthSecret,
    code: &str,
    public_only: bool,
    timeout: Duration,
) -> EngineResult<(AuthConfig, AuthSecret)> {
    if !visible(code, 4096) {
        return Err(error("invalid OAuth authorization code"));
    }
    let verifier = secret
        .code_verifier
        .clone()
        .ok_or_else(|| error("OAuth sign-in state is incomplete"))?;
    let client = OauthClient::from_config(&pending.auth, secret.secret.clone())?;
    let mut form = vec![
        ("grant_type", "authorization_code".to_owned()),
        ("code", code.to_owned()),
        ("redirect_uri", pending.redirect_uri.clone()),
        ("code_verifier", verifier),
    ];
    if let Some(resource) = &client.resource {
        form.push(("resource", resource.clone()));
    }
    let http = network::client(public_only)?;
    let tokens = tokio::time::timeout(
        timeout,
        token_request(
            &http,
            &client.token_endpoint,
            Some(&client),
            form,
            public_only,
        ),
    )
    .await
    .map_err(|_| error("OAuth token request timed out"))??;
    let mut auth = pending.auth.clone();
    if let (AuthConfig::Oauth { scopes, .. }, Some(granted)) = (&mut auth, tokens.scopes) {
        if !granted.is_empty() && validate_scopes(&granted).is_ok() {
            *scopes = granted;
        }
    }
    Ok((
        auth,
        AuthSecret {
            secret: secret.secret.clone(),
            access_token: Some(tokens.access_token),
            refresh_token: tokens.refresh_token,
            expires_at: tokens.expires_at,
            ..AuthSecret::default()
        },
    ))
}

struct OauthClient {
    client_id: String,
    token_endpoint: String,
    client_auth: ClientAuth,
    client_secret: Option<String>,
    scopes: Vec<String>,
    audience: Option<String>,
    resource: Option<String>,
}

impl OauthClient {
    fn from_config(config: &AuthConfig, client_secret: Option<String>) -> EngineResult<Self> {
        match config {
            AuthConfig::Oauth {
                client_id,
                scopes,
                resource,
                managed_client: true,
                ..
            } => {
                // A client the deployment no longer has cannot renew these
                // tokens: it was removed or replaced.
                let (client_id, client_secret) = managed_google_client()
                    .filter(|(managed, _)| managed == client_id)
                    .ok_or_else(|| error(SIGN_IN_AGAIN))?;
                Ok(Self {
                    client_id,
                    token_endpoint: GOOGLE_TOKEN_ENDPOINT.into(),
                    client_auth: ClientAuth::ClientSecretBasic,
                    client_secret: Some(client_secret),
                    scopes: scopes.clone(),
                    audience: None,
                    resource: resource.clone(),
                })
            }
            AuthConfig::Oauth {
                client_id,
                token_endpoint,
                scopes,
                resource,
                client_auth,
                ..
            } => Ok(Self {
                client_id: client_id.clone(),
                token_endpoint: token_endpoint.clone(),
                client_auth: *client_auth,
                client_secret,
                scopes: scopes.clone(),
                audience: None,
                resource: resource.clone(),
            }),
            AuthConfig::ClientCredentials {
                token_endpoint,
                client_id,
                scopes,
                audience,
                resource,
                client_auth,
            } => Ok(Self {
                client_id: client_id.clone(),
                token_endpoint: token_endpoint.clone(),
                client_auth: *client_auth,
                client_secret,
                scopes: scopes.clone(),
                audience: audience.clone(),
                resource: resource.clone(),
            }),
            _ => Err(error("not an OAuth authentication type")),
        }
    }
}

struct Tokens {
    access_token: String,
    refresh_token: Option<String>,
    expires_at: Option<i64>,
    scopes: Option<Vec<String>>,
}

fn parse_tokens(body: &[u8], now: i64) -> EngineResult<Tokens> {
    #[derive(Deserialize)]
    struct Raw {
        access_token: String,
        #[serde(default)]
        token_type: Option<String>,
        #[serde(default)]
        expires_in: Option<Value>,
        #[serde(default)]
        refresh_token: Option<String>,
        #[serde(default)]
        scope: Option<String>,
    }
    let raw: Raw =
        serde_json::from_slice(body).map_err(|_| error("invalid OAuth token response"))?;
    if !visible(&raw.access_token, 16384)
        || raw
            .refresh_token
            .as_deref()
            .is_some_and(|token| !visible(token, 16384))
    {
        return Err(error("invalid OAuth token response"));
    }
    if raw
        .token_type
        .as_deref()
        .is_some_and(|kind| !kind.eq_ignore_ascii_case("bearer"))
    {
        return Err(error(
            "unsupported OAuth token type; only bearer tokens are supported",
        ));
    }
    let expires_in = match raw.expires_in {
        Some(Value::Number(seconds)) => seconds.as_i64(),
        Some(Value::String(seconds)) => seconds.parse().ok(),
        _ => None,
    };
    Ok(Tokens {
        access_token: raw.access_token,
        refresh_token: raw.refresh_token,
        expires_at: expires_in.map(|seconds| now + seconds.clamp(0, 31_536_000)),
        scopes: raw.scope.map(|scope| {
            scope
                .split_whitespace()
                .filter(|scope| auth::valid_scope(scope))
                .map(str::to_owned)
                .collect()
        }),
    })
}

async fn token_request(
    http: &reqwest::Client,
    endpoint: &str,
    client: Option<&OauthClient>,
    mut form: Vec<(&'static str, String)>,
    public_only: bool,
) -> EngineResult<Tokens> {
    let url = network::validate_endpoint_url(endpoint, public_only)?;
    let mut request = http
        .post(url)
        .timeout(REQUEST_TIMEOUT)
        .header(ACCEPT, "application/json");
    if let Some(client) = client {
        match (client.client_auth, client.client_secret.as_deref()) {
            (ClientAuth::None, _) => form.push(("client_id", client.client_id.clone())),
            (ClientAuth::ClientSecretPost, Some(secret)) => {
                form.push(("client_id", client.client_id.clone()));
                form.push(("client_secret", secret.to_owned()));
            }
            (ClientAuth::ClientSecretBasic, Some(secret)) => {
                // RFC 6749 section 2.3.1: both parts are form-encoded first.
                let encode = |value: &str| {
                    url::form_urlencoded::byte_serialize(value.as_bytes()).collect::<String>()
                };
                request = request.header(
                    AUTHORIZATION,
                    auth::sensitive(format!(
                        "Basic {}",
                        STANDARD.encode(format!(
                            "{}:{}",
                            encode(&client.client_id),
                            encode(secret)
                        ))
                    ))?,
                );
            }
            _ => return Err(error("OAuth client secret missing")),
        }
    }
    let response = request
        .form(&form)
        .send()
        .await
        .map_err(|_| error("OAuth token request failed"))?;
    let status = response.status();
    let body = read_limited(response).await?;
    if !status.is_success() {
        let code = serde_json::from_slice::<Value>(&body)
            .ok()
            .and_then(|value| {
                value
                    .get("error")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            });
        return Err(error(match code.as_deref() {
            Some("invalid_grant") => SIGN_IN_AGAIN,
            Some("invalid_client") => CLIENT_REJECTED,
            // The client is valid but may not use this grant type.
            Some("unauthorized_client" | "unsupported_grant_type") => GRANT_UNSUPPORTED,
            Some("invalid_scope") => SCOPES_REJECTED,
            _ => "OAuth token request failed",
        }));
    }
    parse_tokens(&body, now())
}

fn usable(expires_at: i64, now: i64) -> bool {
    expires_at.saturating_sub(EXPIRY_MARGIN_SECONDS) > now
}

/// Short-lived access tokens, fetched on demand and cached until close to
/// expiry or rejected.
pub(super) struct TokenSource {
    grant: Grant,
    http: reqwest::Client,
    public_only: bool,
    cached: tokio::sync::Mutex<Option<(String, i64)>>,
}

enum Grant {
    Refresh(OauthClient, Arc<dyn CredentialStore>),
    ClientCredentials(OauthClient),
    ServiceAccount(ServiceAccount),
}

impl TokenSource {
    fn new(grant: Grant, public_only: bool) -> EngineResult<Self> {
        Ok(Self {
            grant,
            http: network::client(public_only)?,
            public_only,
            cached: tokio::sync::Mutex::new(None),
        })
    }

    pub(super) fn refresh(
        config: &AuthConfig,
        client_secret: Option<String>,
        store: Arc<dyn CredentialStore>,
        public_only: bool,
    ) -> EngineResult<Self> {
        Self::new(
            Grant::Refresh(OauthClient::from_config(config, client_secret)?, store),
            public_only,
        )
    }

    pub(super) fn client_credentials(
        config: &AuthConfig,
        client_secret: String,
        public_only: bool,
    ) -> EngineResult<Self> {
        Self::new(
            Grant::ClientCredentials(OauthClient::from_config(config, Some(client_secret))?),
            public_only,
        )
    }

    pub(super) fn service_account(
        account: ServiceAccount,
        public_only: bool,
    ) -> EngineResult<Self> {
        Self::new(Grant::ServiceAccount(account), public_only)
    }

    /// A usable token other than `rejected`.
    pub(super) async fn token(&self, rejected: Option<&str>) -> EngineResult<String> {
        let now = now();
        let mut cached = self.cached.lock().await;
        if let Some((token, expires_at)) = cached.as_ref() {
            if usable(*expires_at, now) && rejected != Some(token.as_str()) {
                return Ok(token.clone());
            }
        }
        let (token, expires_at) = match &self.grant {
            Grant::Refresh(client, store) => {
                self.refreshed(client, store.as_ref(), rejected, now)
                    .await?
            }
            Grant::ClientCredentials(client) => {
                let mut form = vec![("grant_type", "client_credentials".to_owned())];
                if !client.scopes.is_empty() {
                    form.push(("scope", client.scopes.join(" ")));
                }
                if let Some(audience) = &client.audience {
                    form.push(("audience", audience.clone()));
                }
                if let Some(resource) = &client.resource {
                    form.push(("resource", resource.clone()));
                }
                let tokens = token_request(
                    &self.http,
                    &client.token_endpoint,
                    Some(client),
                    form,
                    self.public_only,
                )
                .await?;
                (tokens.access_token, tokens.expires_at.unwrap_or(now + 300))
            }
            Grant::ServiceAccount(account) => {
                let form = vec![
                    (
                        "grant_type",
                        "urn:ietf:params:oauth:grant-type:jwt-bearer".to_owned(),
                    ),
                    ("assertion", account.assertion(now)?),
                ];
                let tokens =
                    token_request(&self.http, &account.token_uri, None, form, self.public_only)
                        .await?;
                (tokens.access_token, tokens.expires_at.unwrap_or(now + 300))
            }
        };
        *cached = Some((token.clone(), expires_at));
        Ok(token)
    }

    async fn refreshed(
        &self,
        client: &OauthClient,
        store: &dyn CredentialStore,
        rejected: Option<&str>,
        now: i64,
    ) -> EngineResult<(String, i64)> {
        let lease = store.lease().await?;
        let latest = lease.secret().clone();
        if let Some(token) = latest.access_token.as_deref() {
            let expires_at = latest.expires_at.unwrap_or(i64::MAX);
            // Another replica may already have refreshed this credential.
            if usable(expires_at, now) && rejected != Some(token) {
                return Ok((token.to_owned(), expires_at));
            }
        }
        let refresh_token = latest
            .refresh_token
            .clone()
            .ok_or_else(|| error(SIGN_IN_AGAIN))?;
        let mut form = vec![
            ("grant_type", "refresh_token".to_owned()),
            ("refresh_token", refresh_token),
        ];
        if let Some(resource) = &client.resource {
            form.push(("resource", resource.clone()));
        }
        let tokens = token_request(
            &self.http,
            &client.token_endpoint,
            Some(client),
            form,
            self.public_only,
        )
        .await?;
        let token = tokens.access_token.clone();
        let expires_at = tokens.expires_at.unwrap_or(i64::MAX);
        lease
            .commit(AuthSecret {
                access_token: Some(tokens.access_token),
                expires_at: tokens.expires_at,
                refresh_token: tokens
                    .refresh_token
                    .or_else(|| latest.refresh_token.clone()),
                ..latest
            })
            .await?;
        Ok((token, expires_at))
    }
}

fn google_token_uri() -> String {
    "https://oauth2.googleapis.com/token".into()
}

pub(super) struct ServiceAccount {
    email: String,
    token_uri: String,
    key_id: Option<String>,
    key: ring::signature::RsaKeyPair,
    scopes: Vec<String>,
    subject: Option<String>,
}

impl ServiceAccount {
    pub(super) fn parse(
        json: &str,
        scopes: &[String],
        subject: Option<&str>,
        public_only: bool,
    ) -> EngineResult<Self> {
        #[derive(Deserialize)]
        struct Key {
            #[serde(rename = "type")]
            kind: String,
            client_email: String,
            private_key: String,
            #[serde(default)]
            private_key_id: Option<String>,
            #[serde(default = "google_token_uri")]
            token_uri: String,
        }
        let invalid = || error("invalid Google service account key");
        if json.len() > 16384 {
            return Err(invalid());
        }
        let key: Key = serde_json::from_str(json).map_err(|_| invalid())?;
        if key.kind != "service_account"
            || !visible(&key.client_email, 320)
            || key
                .private_key_id
                .as_deref()
                .is_some_and(|id| !visible(id, 256))
        {
            return Err(invalid());
        }
        network::validate_endpoint_url(&key.token_uri, public_only)?;
        let body = key
            .private_key
            .trim()
            .strip_prefix("-----BEGIN PRIVATE KEY-----")
            .and_then(|rest| rest.trim_end().strip_suffix("-----END PRIVATE KEY-----"))
            .ok_or_else(invalid)?;
        let der = STANDARD
            .decode(
                body.chars()
                    .filter(|c| !c.is_whitespace())
                    .collect::<String>(),
            )
            .map_err(|_| invalid())?;
        let pair = ring::signature::RsaKeyPair::from_pkcs8(&der).map_err(|_| invalid())?;
        Ok(Self {
            email: key.client_email,
            token_uri: key.token_uri,
            key_id: key.private_key_id,
            key: pair,
            scopes: scopes.to_vec(),
            subject: subject.map(str::to_owned),
        })
    }

    fn assertion(&self, now: i64) -> EngineResult<String> {
        let mut header = json!({"alg": "RS256", "typ": "JWT"});
        if let Some(key_id) = &self.key_id {
            header["kid"] = json!(key_id);
        }
        let mut claims = json!({
            "iss": self.email,
            "aud": self.token_uri,
            "iat": now,
            "exp": now + 3600,
            "scope": self.scopes.join(" "),
        });
        if let Some(subject) = &self.subject {
            claims["sub"] = json!(subject);
        }
        let input = format!(
            "{}.{}",
            URL_SAFE_NO_PAD.encode(header.to_string()),
            URL_SAFE_NO_PAD.encode(claims.to_string())
        );
        let mut signature = vec![0; self.key.public().modulus_len()];
        self.key
            .sign(
                &ring::signature::RSA_PKCS1_SHA256,
                &ring::rand::SystemRandom::new(),
                input.as_bytes(),
                &mut signature,
            )
            .map_err(|_| error("service account signing failed"))?;
        Ok(format!("{input}.{}", URL_SAFE_NO_PAD.encode(signature)))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{auth::MemoryStore, transport::Transport, Access, Credential, ServerConfig};
    use super::*;
    use axum::{
        extract::{Form, State},
        http::{HeaderMap, StatusCode},
        response::IntoResponse,
        routing::{get, post},
        Json, Router,
    };
    use std::{
        collections::{BTreeMap, HashMap, HashSet},
        sync::Mutex,
    };

    #[test]
    fn bearer_challenges_yield_metadata_and_scopes() {
        let (metadata, scopes) = bearer_challenge(
            r#"Basic realm="x", Bearer error="invalid_token", resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource", scope="files:read files:write""#,
        )
        .unwrap();
        assert_eq!(
            metadata.as_deref(),
            Some("https://mcp.example.com/.well-known/oauth-protected-resource")
        );
        assert_eq!(scopes, ["files:read", "files:write"]);
        assert!(bearer_challenge(r#"Basic realm="bearer""#).is_none());
        assert_eq!(bearer_challenge("Bearer").unwrap(), (None, vec![]));
        assert_eq!(
            bearer_error(r#"Bearer realm="https://accounts.google.com/", error="invalid_token""#)
                .as_deref(),
            Some("invalid_token")
        );
    }

    #[tokio::test]
    async fn refusals_keep_their_codes_but_not_upstream_text() {
        let app = Router::new().route(
            "/mcp",
            post(|| async {
                (
                    StatusCode::FORBIDDEN,
                    [(
                        "www-authenticate",
                        r#"Bearer error="insufficient_scope", error_description="secret-looking text""#,
                    )],
                    Json(json!({"error": {"code": 403, "status": "PERMISSION_DENIED",
                        "message": "Gmail MCP API has not been used in project 123",
                        "details": [{"reason": "SERVICE_DISABLED"}]}})),
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let error = Transport::connect_with(
            &server(&url, AuthConfig::Bearer),
            Some(Credential::new(AuthSecret::from_secret("token"))),
            false,
        )
        .await
        .err()
        .unwrap();
        assert!(super::super::is_auth_rejected(&error));
        assert!(
            !error.to_string().contains("secret-looking")
                && !error.to_string().contains("project 123")
        );
        assert_eq!(
            super::super::auth_rejection(&error),
            Some(super::super::AuthRejection {
                status: 403,
                error: Some("insufficient_scope".into()),
                reason: Some("SERVICE_DISABLED".into()),
            })
        );
        assert!(super::super::auth_rejection(&super::error(SIGN_IN_AGAIN)).is_none());
        task.abort();
    }

    #[derive(Default)]
    struct Upstream {
        base: String,
        valid: HashSet<String>,
        forms: Vec<HashMap<String, String>>,
    }

    type Shared = Arc<Mutex<Upstream>>;

    async fn mcp(
        State(state): State<Shared>,
        headers: HeaderMap,
        Json(request): Json<Value>,
    ) -> axum::response::Response {
        let (base, valid) = {
            let state = state.lock().unwrap();
            (state.base.clone(), state.valid.clone())
        };
        let token = headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "));
        if !token.is_some_and(|token| valid.contains(token)) {
            let challenge = format!(
                r#"Bearer resource_metadata="{base}/.well-known/oauth-protected-resource/mcp", scope="tools.read""#
            );
            return (StatusCode::UNAUTHORIZED, [("www-authenticate", challenge)]).into_response();
        }
        let result = match request["method"].as_str().unwrap() {
            "initialize" => {
                json!({"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "oauth", "version": "1"}})
            }
            "tools/list" => json!({"tools": [{"name": "read", "inputSchema": {"type": "object"}}]}),
            _ => return StatusCode::ACCEPTED.into_response(),
        };
        Json(json!({"jsonrpc": "2.0", "id": request["id"], "result": result})).into_response()
    }

    async fn token(
        State(state): State<Shared>,
        headers: HeaderMap,
        Form(form): Form<HashMap<String, String>>,
    ) -> axum::response::Response {
        let mut state = state.lock().unwrap();
        state.forms.push(form.clone());
        let base = state.base.clone();
        let issued = |access: &str, refresh: Option<&str>| {
            let mut body =
                json!({"access_token": access, "token_type": "Bearer", "expires_in": 3600});
            if let Some(refresh) = refresh {
                body["refresh_token"] = json!(refresh);
            }
            body
        };
        let body = match form["grant_type"].as_str() {
            "authorization_code" => {
                assert_eq!(form["code"], "code-1");
                assert_eq!(form["client_id"], "registered-client");
                assert_eq!(
                    form["redirect_uri"],
                    "https://app.example.com/mcp/oauth/callback"
                );
                assert_eq!(form["resource"], format!("{base}/mcp"));
                assert!(form["code_verifier"].len() >= 43);
                let mut body = issued("access-1", Some("refresh-1"));
                body["scope"] = json!("tools.read offline_access");
                body
            }
            "refresh_token" if form["refresh_token"] == "refresh-1" => {
                issued("access-2", Some("refresh-2"))
            }
            "client_credentials" if form.get("audience").is_none() => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "unsupported_grant_type"})),
                )
                    .into_response()
            }
            "client_credentials" => {
                assert_eq!(
                    headers["authorization"],
                    format!("Basic {}", STANDARD.encode("machine:machine-secret%3A1"))
                );
                assert_eq!(form["audience"], "mcp");
                assert!(!form.contains_key("client_secret"));
                issued("access-cc", None)
            }
            _ => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "invalid_grant"})),
                )
                    .into_response()
            }
        };
        Json(body).into_response()
    }

    async fn upstream() -> (String, Shared, tokio::task::JoinHandle<()>) {
        let state = Shared::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        state.lock().unwrap().base = base.clone();
        let metadata = base.clone();
        let issuer = base.clone();
        let app = Router::new()
            .route("/mcp", post(mcp))
            .route(
                "/.well-known/oauth-protected-resource/mcp",
                get(move || async move {
                    Json(json!({
                        "resource": format!("{metadata}/mcp"),
                        "authorization_servers": [format!("{metadata}/auth")],
                        "scopes_supported": ["tools.read", "tools.write"]
                    }))
                }),
            )
            .route(
                "/.well-known/oauth-authorization-server/auth",
                get(move || async move {
                    Json(json!({
                        "issuer": format!("{issuer}/auth"),
                        "authorization_endpoint": format!("{issuer}/auth/authorize"),
                        "token_endpoint": format!("{issuer}/auth/token"),
                        "registration_endpoint": format!("{issuer}/auth/register"),
                        "code_challenge_methods_supported": ["S256"],
                        "scopes_supported": ["tools.read", "tools.write", "offline_access"]
                    }))
                }),
            )
            .route(
                "/auth/register",
                post(|Json(request): Json<Value>| async move {
                    assert_eq!(request["token_endpoint_auth_method"], "none");
                    (
                        StatusCode::CREATED,
                        Json(json!({"client_id": "registered-client"})),
                    )
                }),
            )
            .route("/auth/token", post(token))
            .with_state(state.clone());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (base, state, task)
    }

    fn server(url: &str, auth: AuthConfig) -> ServerConfig {
        ServerConfig {
            name: "oauth".into(),
            url: url.into(),
            bearer_token_env: None,
            bearer_token: None,
            auth: Some(auth),
            secret: None,
            secret_env: None,
            session_token_env: None,
            headers: BTreeMap::new(),
            allow_loopback_http: true,
            timeout_ms: 5000,
            access: Access::Local {
                project_id: None,
                project_ids: vec![],
                project_path: None,
            },
            tools: BTreeMap::new(),
        }
    }

    #[tokio::test]
    async fn sign_in_registers_a_client_and_refreshes_rejected_tokens() {
        let (base, state, task) = upstream().await;
        let url = format!("{base}/mcp");
        let timeout = Duration::from_secs(5);

        let metadata = discover(&url, false, timeout).await.unwrap();
        assert!(metadata.requires_auth);
        assert!(metadata.dynamic_registration);
        assert!(!metadata.managed_client);
        // ReinCore's Google client is refused for servers Google does not sign in to.
        let managed = start(
            AuthorizationRequest {
                url: url.clone(),
                redirect_uri: "https://app.example.com/mcp/oauth/callback".into(),
                client_id: None,
                client_secret: None,
                scopes: None,
                timeout_ms: 5000,
                managed_client: true,
            },
            false,
        )
        .await;
        assert_eq!(
            managed.err().map(|error| error.to_string()),
            Some(error(MANAGED_CLIENT_UNAVAILABLE).to_string())
        );
        assert_eq!(metadata.challenge_scopes, ["tools.read"]);
        assert_eq!(metadata.scopes_supported, ["tools.read", "tools.write"]);
        assert_eq!(metadata.token_endpoint, Some(format!("{base}/auth/token")));

        let started = start(
            AuthorizationRequest {
                url: url.clone(),
                redirect_uri: "https://app.example.com/mcp/oauth/callback".into(),
                client_id: None,
                client_secret: None,
                scopes: None,
                timeout_ms: 5000,
                managed_client: false,
            },
            false,
        )
        .await
        .unwrap();
        let authorization = url::Url::parse(&started.authorization_url).unwrap();
        let query: HashMap<_, _> = authorization.query_pairs().into_owned().collect();
        let verifier = started.secret.code_verifier.clone().unwrap();
        assert_eq!(query["state"], started.state);
        assert_eq!(query["client_id"], "registered-client");
        assert_eq!(query["scope"], "tools.read offline_access");
        assert_eq!(query["resource"], url);
        // Google's offline parameters are not sent to other providers.
        assert!(!query.contains_key("prompt") && !query.contains_key("access_type"));
        assert!(google(Some("https://accounts.google.com/")));
        assert!(!google(Some("https://accounts.google.com.evil.example")));
        assert_eq!(
            query["code_challenge"],
            URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
        );

        let (auth, secret) = complete(&started.pending, &started.secret, "code-1", false, timeout)
            .await
            .unwrap();
        assert_eq!(secret.access_token.as_deref(), Some("access-1"));
        assert_eq!(secret.refresh_token.as_deref(), Some("refresh-1"));
        assert!(secret.code_verifier.is_none());
        assert!(secret.expires_at.unwrap() > now() + 3000);

        // A token the server no longer accepts is refreshed and the request
        // sent once more; the rotated refresh token is kept.
        state.lock().unwrap().valid = HashSet::from(["access-2".into()]);
        let store = MemoryStore::new(secret.clone());
        let transport = Transport::connect_with(
            &server(&url, auth),
            Some(Credential {
                secret,
                store: Some(store.clone()),
            }),
            false,
        )
        .await
        .unwrap();
        let transport = Arc::new(transport);
        transport.rpc("tools/list", json!({})).await.unwrap();
        let stored = store.lease().await.unwrap().secret().clone();
        assert_eq!(stored.access_token.as_deref(), Some("access-2"));
        assert_eq!(stored.refresh_token.as_deref(), Some("refresh-2"));

        // Once the refresh token is refused too, the failure asks for a new sign-in.
        state.lock().unwrap().valid.clear();
        let error = transport.rpc("tools/list", json!({})).await.unwrap_err();
        assert!(super::super::is_auth_rejected(&error), "{error}");
        task.abort();
    }

    #[test]
    fn managed_google_client_secret_comes_from_the_deployment_and_goes_to_google() {
        let managed = |client_id: &str, token_endpoint: &str| AuthConfig::Oauth {
            client_id: client_id.into(),
            token_endpoint: token_endpoint.into(),
            scopes: vec!["https://www.googleapis.com/auth/gmail.readonly".into()],
            resource: None,
            client_auth: ClientAuth::None,
            issuer: Some("https://accounts.google.com".into()),
            managed_client: true,
        };
        std::env::set_var(MANAGED_CLIENT_ID, "reincore.apps.googleusercontent.com");
        std::env::set_var(MANAGED_CLIENT_SECRET, "deployment-secret");
        let config = managed("reincore.apps.googleusercontent.com", GOOGLE_TOKEN_ENDPOINT);
        config.validate(true).unwrap();
        // No client secret is stored with a managed sign-in.
        assert!(config
            .validate_secret(&AuthSecret::from_secret("stored"))
            .is_err());
        let client = OauthClient::from_config(&config, Some("ignored".into())).unwrap();
        assert_eq!(client.client_secret.as_deref(), Some("deployment-secret"));
        assert_eq!(client.token_endpoint, GOOGLE_TOKEN_ENDPOINT);
        assert_eq!(client.client_auth, ClientAuth::ClientSecretBasic);
        // Never sent elsewhere, and a replaced client needs a new sign-in.
        assert!(managed(
            "reincore.apps.googleusercontent.com",
            "https://evil.example/token"
        )
        .validate(true)
        .is_err());
        assert_eq!(
            OauthClient::from_config(
                &managed("old.apps.googleusercontent.com", GOOGLE_TOKEN_ENDPOINT),
                None
            )
            .err()
            .map(|error| error.to_string()),
            Some(error(SIGN_IN_AGAIN).to_string())
        );
        std::env::set_var(MANAGED_CLIENT_SECRET, "  ");
        assert!(managed_google_client().is_none());
    }

    #[tokio::test]
    async fn client_credentials_use_basic_client_authentication() {
        let (base, state, task) = upstream().await;
        state.lock().unwrap().valid = HashSet::from(["access-cc".into()]);
        let url = format!("{base}/mcp");
        let auth = AuthConfig::ClientCredentials {
            token_endpoint: format!("{base}/auth/token"),
            client_id: "machine".into(),
            scopes: vec![],
            audience: Some("mcp".into()),
            resource: None,
            client_auth: ClientAuth::ClientSecretBasic,
        };
        let credential = Credential::new(AuthSecret::from_secret("machine-secret:1"));
        Transport::connect_with(&server(&url, auth), Some(credential), false)
            .await
            .unwrap();
        // One token serves every request of the session.
        assert_eq!(state.lock().unwrap().forms.len(), 1);

        // A token endpoint that refuses the grant says so instead of looking
        // like an unreachable server.
        let refused = AuthConfig::ClientCredentials {
            token_endpoint: format!("{base}/auth/token"),
            client_id: "machine".into(),
            scopes: vec![],
            audience: None,
            resource: None,
            client_auth: ClientAuth::ClientSecretBasic,
        };
        let error = Transport::connect_with(
            &server(&url, refused),
            Some(Credential::new(AuthSecret::from_secret("machine-secret:1"))),
            false,
        )
        .await
        .err()
        .unwrap();
        assert_eq!(
            error.to_string(),
            super::error(GRANT_UNSUPPORTED).to_string()
        );

        let wrong = Credential::new(AuthSecret::from_secret("wrong"));
        let error = Transport::connect_with(&server(&url, AuthConfig::Bearer), Some(wrong), false)
            .await
            .err()
            .unwrap();
        assert!(super::super::is_auth_rejected(&error));
        task.abort();
    }

    // Needs an RSA key, which is not committed: generate one with
    // `openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out key.pem`
    // and point REINCORE_TEST_RSA_KEY_PEM at it.
    #[test]
    #[ignore = "requires REINCORE_TEST_RSA_KEY_PEM"]
    fn service_account_assertions_are_signed_rs256_jwts() {
        use ring::signature::KeyPair;
        let pem =
            std::fs::read_to_string(std::env::var("REINCORE_TEST_RSA_KEY_PEM").unwrap()).unwrap();
        let key = json!({
            "type": "service_account",
            "client_email": "agent@project.iam.gserviceaccount.com",
            "private_key_id": "key-1",
            "private_key": pem,
            "token_uri": "https://oauth2.googleapis.com/token"
        })
        .to_string();
        let scopes = vec!["https://www.googleapis.com/auth/bigquery".to_owned()];
        let account = ServiceAccount::parse(&key, &scopes, Some("user@example.com"), true).unwrap();
        let assertion = account.assertion(1_700_000_000).unwrap();
        let parts: Vec<_> = assertion.split('.').collect();
        let header: Value =
            serde_json::from_slice(&URL_SAFE_NO_PAD.decode(parts[0]).unwrap()).unwrap();
        let claims: Value =
            serde_json::from_slice(&URL_SAFE_NO_PAD.decode(parts[1]).unwrap()).unwrap();
        assert_eq!(
            header,
            json!({"alg": "RS256", "typ": "JWT", "kid": "key-1"})
        );
        assert_eq!(
            claims,
            json!({
                "iss": "agent@project.iam.gserviceaccount.com",
                "sub": "user@example.com",
                "aud": "https://oauth2.googleapis.com/token",
                "iat": 1_700_000_000,
                "exp": 1_700_003_600,
                "scope": "https://www.googleapis.com/auth/bigquery"
            })
        );
        ring::signature::UnparsedPublicKey::new(
            &ring::signature::RSA_PKCS1_2048_8192_SHA256,
            account.key.public_key().as_ref(),
        )
        .verify(
            format!("{}.{}", parts[0], parts[1]).as_bytes(),
            &URL_SAFE_NO_PAD.decode(parts[2]).unwrap(),
        )
        .unwrap();
        assert!(ServiceAccount::parse(
            &key.replace("service_account", "user"),
            &scopes,
            None,
            true
        )
        .is_err());
    }

    #[tokio::test]
    async fn servers_without_published_endpoints_cannot_start_a_sign_in() {
        let app = Router::new().route(
            "/mcp",
            post(|| async { (StatusCode::UNAUTHORIZED, [("www-authenticate", "Bearer")]) }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let metadata = discover(&url, false, Duration::from_secs(5)).await.unwrap();
        assert!(metadata.requires_auth);
        assert!(metadata.token_endpoint.is_none());
        let request = |managed_client| AuthorizationRequest {
            url: url.clone(),
            redirect_uri: "https://app.example.com/callback".into(),
            client_id: None,
            client_secret: None,
            scopes: None,
            timeout_ms: 5000,
            managed_client,
        };
        assert_eq!(
            start(request(false), false)
                .await
                .err()
                .unwrap()
                .to_string(),
            error(SIGN_IN_UNAVAILABLE).to_string()
        );
        assert!(validate_redirect_uri("http://app.example.com/callback").is_err());
        assert!(validate_redirect_uri("http://localhost:5173/mcp/oauth/callback").is_ok());
        task.abort();
    }
}
