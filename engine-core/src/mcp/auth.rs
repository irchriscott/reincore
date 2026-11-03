//! Upstream MCP authentication.
//!
//! [`AuthConfig`] is the non-secret half: it is stored with the server and may
//! be shown to administrators. [`AuthSecret`] holds what must never be shown
//! again; cloud hosts encrypt it, local configuration reads it inline or from
//! environment variables. [`Authenticator`] turns both into request headers,
//! renewing short-lived tokens and signing requests where a type needs it.
use super::{error, network, oauth};
use crate::EngineResult;
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use hmac::{Hmac, Mac};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
};

pub const MAX_HEADERS: usize = 16;
const MAX_SECRET: usize = 16384;
const MAX_TEXT: usize = 2048;
const MAX_SCOPES: usize = 64;
pub const CREDENTIAL_MISSING: &str =
    "MCP credential missing for this authentication type; set it before connecting";

// Headers the transport owns, or that would change how the request is framed.
const RESERVED_HEADERS: &[&str] = &[
    "accept",
    "connection",
    "content-length",
    "content-type",
    "expect",
    "host",
    "keep-alive",
    "last-event-id",
    "mcp-protocol-version",
    "mcp-session-id",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "x-amz-date",
    "x-amz-security-token",
];

/// How the OAuth client proves its identity at the token endpoint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientAuth {
    /// A public client: only the client ID is sent.
    #[default]
    None,
    ClientSecretBasic,
    ClientSecretPost,
}

fn client_secret_basic() -> ClientAuth {
    ClientAuth::ClientSecretBasic
}

/// How ReinCore authenticates to an upstream MCP server. Secrets are held
/// separately in [`AuthSecret`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum AuthConfig {
    /// A public server.
    #[default]
    None,
    /// `Authorization: Bearer <secret>`.
    Bearer,
    /// `<name>: <secret>`, for API keys or custom authorization schemes.
    Header { name: String },
    /// `Authorization: Basic base64(username:secret)`.
    Basic { username: String },
    /// User sign-in (authorization code with PKCE). Tokens come from a
    /// completed sign-in and are refreshed automatically.
    Oauth {
        client_id: String,
        token_endpoint: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        scopes: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resource: Option<String>,
        #[serde(default)]
        client_auth: ClientAuth,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        issuer: Option<String>,
        /// Signed in with ReinCore's own Google OAuth client. Its secret comes
        /// from the deployment when tokens are requested and is never stored
        /// with the server (see [`oauth::managed_google_client`]).
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        managed_client: bool,
    },
    /// Machine-to-machine OAuth; the secret is the client secret.
    ClientCredentials {
        token_endpoint: String,
        client_id: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        scopes: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        audience: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resource: Option<String>,
        #[serde(default = "client_secret_basic")]
        client_auth: ClientAuth,
    },
    /// A Google service account; the secret is its JSON key.
    GoogleServiceAccount {
        scopes: Vec<String>,
        /// Workspace user to impersonate through domain-wide delegation.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        subject: Option<String>,
    },
    /// AWS Signature Version 4; the secret is the secret access key.
    AwsSigv4 {
        access_key_id: String,
        region: String,
        service: String,
    },
}

/// Secret material for one server. Which fields apply depends on the type:
/// `secret` is the bearer token, header value, password, OAuth client secret,
/// service account key or AWS secret access key.
#[derive(Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AuthSecret {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret: Option<String>,
    /// AWS temporary credentials.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub access_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    /// Unix seconds; absent when the authorization server did not say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
    /// PKCE verifier of a sign-in that has not completed yet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_verifier: Option<String>,
}

impl std::fmt::Debug for AuthSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AuthSecret(redacted)")
    }
}

impl AuthSecret {
    pub fn from_secret(secret: impl Into<String>) -> Self {
        Self {
            secret: Some(secret.into()),
            ..Self::default()
        }
    }
}

/// Persists tokens renewed at runtime. A lease holds the latest stored secret
/// exclusively, so replicas sharing a credential never refresh it concurrently
/// with a refresh token another replica has already rotated.
#[async_trait]
pub trait CredentialStore: Send + Sync {
    async fn lease(&self) -> EngineResult<Box<dyn CredentialLease>>;
}

#[async_trait]
pub trait CredentialLease: Send {
    fn secret(&self) -> &AuthSecret;
    async fn commit(self: Box<Self>, secret: AuthSecret) -> EngineResult<()>;
}

/// Keeps renewed tokens in memory only.
pub struct MemoryStore(Arc<tokio::sync::Mutex<AuthSecret>>);

impl MemoryStore {
    pub fn new(secret: AuthSecret) -> Arc<Self> {
        Arc::new(Self(Arc::new(tokio::sync::Mutex::new(secret))))
    }
}

struct MemoryLease(tokio::sync::OwnedMutexGuard<AuthSecret>);

#[async_trait]
impl CredentialStore for MemoryStore {
    async fn lease(&self) -> EngineResult<Box<dyn CredentialLease>> {
        Ok(Box::new(MemoryLease(self.0.clone().lock_owned().await)))
    }
}

#[async_trait]
impl CredentialLease for MemoryLease {
    fn secret(&self) -> &AuthSecret {
        &self.0
    }
    async fn commit(mut self: Box<Self>, secret: AuthSecret) -> EngineResult<()> {
        *self.0 = secret;
        Ok(())
    }
}

/// A resolved credential. Without a store, renewed tokens live in memory.
#[derive(Clone)]
pub struct Credential {
    pub secret: AuthSecret,
    pub store: Option<Arc<dyn CredentialStore>>,
}

impl Credential {
    pub fn new(secret: AuthSecret) -> Self {
        Self {
            secret,
            store: None,
        }
    }
}

fn ensure(condition: bool, message: &str) -> EngineResult<()> {
    if condition {
        Ok(())
    } else {
        Err(error(message))
    }
}

fn bounded(value: &str, max: usize) -> bool {
    !value.is_empty() && value.len() <= max
}

fn visible(value: &str) -> bool {
    value.bytes().all(|b| (0x21..=0x7e).contains(&b))
}

fn printable(value: &str) -> bool {
    !value.chars().any(char::is_control)
}

/// RFC 6749 scope-token characters.
pub(super) fn valid_scope(scope: &str) -> bool {
    bounded(scope, 256)
        && scope
            .bytes()
            .all(|b| b == 0x21 || (0x23..=0x5b).contains(&b) || (0x5d..=0x7e).contains(&b))
}

pub(super) fn validate_scopes(scopes: &[String]) -> EngineResult<()> {
    ensure(
        scopes.len() <= MAX_SCOPES
            && scopes.iter().all(|scope| valid_scope(scope))
            && scopes.iter().collect::<HashSet<_>>().len() == scopes.len(),
        "invalid OAuth scopes",
    )
}

/// A resource indicator names the MCP server, so it follows the endpoint
/// rules: HTTPS, or loopback HTTP (127.0.0.1, ::1, or localhost) for local servers.
fn validate_uri(value: &str, public_only: bool) -> EngineResult<()> {
    let url = url::Url::parse(value).map_err(|_| error("invalid OAuth resource"))?;
    ensure(
        value.len() <= MAX_TEXT
            && url.fragment().is_none()
            && (url.scheme() == "https"
                || !public_only && url.scheme() == "http" && network::loopback(&url)),
        "invalid OAuth resource",
    )
}

fn header_name(name: &str, allow_authorization: bool) -> EngineResult<HeaderName> {
    let parsed =
        HeaderName::from_bytes(name.as_bytes()).map_err(|_| error("invalid header name"))?;
    ensure(
        name.len() <= 64
            && !RESERVED_HEADERS.contains(&parsed.as_str())
            && (allow_authorization || parsed != AUTHORIZATION),
        "header name is reserved or invalid",
    )?;
    Ok(parsed)
}

fn header_value(value: &str) -> bool {
    bounded(value, MAX_SECRET)
        && value.trim() == value
        && value
            .bytes()
            .all(|b| b == b'\t' || (0x20..=0x7e).contains(&b))
}

/// Non-secret extra headers sent with every request, such as a quota project.
/// Credentials belong in an authentication type, so `Authorization` is refused.
pub fn validate_headers(headers: &BTreeMap<String, String>) -> EngineResult<HeaderMap> {
    ensure(headers.len() <= MAX_HEADERS, "too many MCP headers")?;
    let mut map = HeaderMap::new();
    for (name, value) in headers {
        ensure(
            value.len() <= 4096 && header_value(value),
            "invalid MCP header value",
        )?;
        let name = header_name(name, false)?;
        ensure(!map.contains_key(&name), "duplicate MCP header")?;
        map.insert(
            name,
            HeaderValue::from_str(value).map_err(|_| error("invalid MCP header value"))?,
        );
    }
    Ok(map)
}

impl AuthConfig {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Bearer => "bearer",
            Self::Header { .. } => "header",
            Self::Basic { .. } => "basic",
            Self::Oauth { .. } => "oauth",
            Self::ClientCredentials { .. } => "client_credentials",
            Self::GoogleServiceAccount { .. } => "google_service_account",
            Self::AwsSigv4 { .. } => "aws_sigv4",
        }
    }

    pub fn needs_secret(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Checks the non-secret settings. `public_only` applies the cloud egress
    /// rules to the endpoints the engine will call.
    pub fn validate(&self, public_only: bool) -> EngineResult<()> {
        match self {
            Self::None | Self::Bearer => Ok(()),
            Self::Header { name } => header_name(name, true).map(|_| ()),
            Self::Basic { username } => ensure(
                bounded(username, 256) && printable(username) && !username.contains(':'),
                "invalid Basic username",
            ),
            Self::Oauth {
                client_id,
                token_endpoint,
                scopes,
                resource,
                issuer,
                managed_client,
                ..
            } => {
                ensure(
                    bounded(client_id, 512) && visible(client_id),
                    "invalid OAuth client ID",
                )?;
                // The deployment's client secret only ever goes to Google.
                ensure(
                    !managed_client || token_endpoint == oauth::GOOGLE_TOKEN_ENDPOINT,
                    "ReinCore's Google client signs in to Google only",
                )?;
                network::validate_endpoint_url(token_endpoint, public_only)?;
                validate_scopes(scopes)?;
                resource
                    .as_deref()
                    .map(|resource| validate_uri(resource, public_only))
                    .transpose()?;
                ensure(
                    issuer
                        .as_deref()
                        .map_or(true, |issuer| bounded(issuer, MAX_TEXT) && visible(issuer)),
                    "invalid OAuth issuer",
                )
            }
            Self::ClientCredentials {
                token_endpoint,
                client_id,
                scopes,
                audience,
                resource,
                client_auth,
            } => {
                ensure(
                    bounded(client_id, 512) && visible(client_id),
                    "invalid OAuth client ID",
                )?;
                network::validate_endpoint_url(token_endpoint, public_only)?;
                validate_scopes(scopes)?;
                resource
                    .as_deref()
                    .map(|resource| validate_uri(resource, public_only))
                    .transpose()?;
                ensure(
                    audience.as_deref().map_or(true, |audience| {
                        bounded(audience, MAX_TEXT) && visible(audience)
                    }),
                    "invalid OAuth audience",
                )?;
                ensure(
                    *client_auth != ClientAuth::None,
                    "client credentials need client_secret_basic or client_secret_post",
                )
            }
            Self::GoogleServiceAccount { scopes, subject } => {
                ensure(
                    !scopes.is_empty(),
                    "a Google service account needs at least one scope",
                )?;
                validate_scopes(scopes)?;
                ensure(
                    subject
                        .as_deref()
                        .map_or(true, |subject| bounded(subject, 320) && visible(subject)),
                    "invalid service account subject",
                )
            }
            Self::AwsSigv4 {
                access_key_id,
                region,
                service,
            } => {
                let lower = |value: &str, max| {
                    bounded(value, max)
                        && value
                            .bytes()
                            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
                };
                ensure(
                    (16..=128).contains(&access_key_id.len())
                        && access_key_id
                            .bytes()
                            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit()),
                    "invalid AWS access key ID",
                )?;
                ensure(lower(region, 32), "invalid AWS region")?;
                ensure(lower(service, 64), "invalid AWS service name")
            }
        }
    }

    /// Checks a credential an administrator entered for this type. Tokens
    /// from a completed OAuth sign-in are checked when they are issued.
    pub fn validate_secret(&self, secret: &AuthSecret) -> EngineResult<()> {
        ensure(
            secret.access_token.is_none()
                && secret.refresh_token.is_none()
                && secret.expires_at.is_none()
                && secret.code_verifier.is_none()
                && (secret.session_token.is_none() || matches!(self, Self::AwsSigv4 { .. })),
            "credential fields do not match the authentication type",
        )?;
        let value = secret.secret.as_deref();
        match self {
            Self::None => ensure(
                value.is_none(),
                "this authentication type takes no credential",
            ),
            Self::Bearer => ensure(
                value.is_some_and(|v| bounded(v, MAX_SECRET) && visible(v)),
                "invalid MCP bearer token",
            ),
            Self::Header { .. } => ensure(value.is_some_and(header_value), "invalid header value"),
            Self::Basic { .. } => ensure(
                value.is_some_and(|v| bounded(v, 4096) && printable(v)),
                "invalid Basic password",
            ),
            Self::Oauth { managed_client, .. } => ensure(
                if *managed_client {
                    value.is_none()
                } else {
                    value.map_or(true, |v| bounded(v, 4096) && visible(v))
                },
                "invalid OAuth client secret",
            ),
            Self::ClientCredentials { .. } => ensure(
                value.is_some_and(|v| bounded(v, 4096) && visible(v)),
                "invalid OAuth client secret",
            ),
            Self::GoogleServiceAccount { scopes, subject } => {
                let value = value.ok_or_else(|| error(CREDENTIAL_MISSING))?;
                oauth::ServiceAccount::parse(value, scopes, subject.as_deref(), false).map(|_| ())
            }
            Self::AwsSigv4 { .. } => ensure(
                value.is_some_and(|v| bounded(v, 256) && visible(v))
                    && secret
                        .session_token
                        .as_deref()
                        .map_or(true, |t| bounded(t, 8192) && visible(t)),
                "invalid AWS secret access key or session token",
            ),
        }
    }
}

/// Adds authentication to outgoing requests.
pub(super) struct Authenticator(Method);

// One per connection, so the size of the token variant does not matter.
#[allow(clippy::large_enum_variant)]
enum Method {
    None,
    Static(HeaderName, HeaderValue),
    Token(oauth::TokenSource),
    Sigv4(Sigv4),
}

pub(super) fn sensitive(value: String) -> EngineResult<HeaderValue> {
    let mut value = HeaderValue::from_str(&value).map_err(|_| error("invalid MCP credential"))?;
    value.set_sensitive(true);
    Ok(value)
}

pub(super) fn bearer(token: &str) -> EngineResult<HeaderValue> {
    if !bounded(token, MAX_SECRET) || !visible(token) {
        return Err(error("invalid MCP bearer token"));
    }
    sensitive(format!("Bearer {token}"))
}

impl Authenticator {
    pub(super) fn new(
        config: &AuthConfig,
        credential: Option<Credential>,
        public_only: bool,
    ) -> EngineResult<Self> {
        let value = || {
            credential
                .as_ref()
                .and_then(|c| c.secret.secret.clone())
                .ok_or_else(|| error(CREDENTIAL_MISSING))
        };
        Ok(Self(match config {
            AuthConfig::None => Method::None,
            AuthConfig::Bearer => Method::Static(AUTHORIZATION, bearer(&value()?)?),
            AuthConfig::Header { name } => {
                let value = value()?;
                if !header_value(&value) {
                    return Err(error("invalid header value"));
                }
                Method::Static(header_name(name, true)?, sensitive(value)?)
            }
            AuthConfig::Basic { username } => Method::Static(
                AUTHORIZATION,
                sensitive(format!(
                    "Basic {}",
                    STANDARD.encode(format!("{username}:{}", value()?))
                ))?,
            ),
            AuthConfig::Oauth { .. } => {
                let credential = credential.ok_or_else(|| error(oauth::SIGN_IN_AGAIN))?;
                let store = credential
                    .store
                    .unwrap_or_else(|| MemoryStore::new(credential.secret.clone()));
                Method::Token(oauth::TokenSource::refresh(
                    config,
                    credential.secret.secret,
                    store,
                    public_only,
                )?)
            }
            AuthConfig::ClientCredentials { .. } => Method::Token(
                oauth::TokenSource::client_credentials(config, value()?, public_only)?,
            ),
            AuthConfig::GoogleServiceAccount { scopes, subject } => {
                Method::Token(oauth::TokenSource::service_account(
                    oauth::ServiceAccount::parse(
                        &value()?,
                        scopes,
                        subject.as_deref(),
                        public_only,
                    )?,
                    public_only,
                )?)
            }
            AuthConfig::AwsSigv4 {
                access_key_id,
                region,
                service,
            } => Method::Sigv4(Sigv4 {
                access_key_id: access_key_id.clone(),
                secret_access_key: value()?,
                session_token: credential
                    .as_ref()
                    .and_then(|c| c.secret.session_token.clone()),
                region: region.clone(),
                service: service.clone(),
            }),
        }))
    }

    /// Authenticates one request. Returns the bearer token used when it can be
    /// renewed, so a rejection can ask for a different one.
    pub(super) async fn apply(
        &self,
        url: &url::Url,
        headers: &mut HeaderMap,
        body: &[u8],
    ) -> EngineResult<Option<String>> {
        match &self.0 {
            Method::None => Ok(None),
            Method::Static(name, value) => {
                headers.insert(name.clone(), value.clone());
                Ok(None)
            }
            Method::Token(source) => {
                let token = source.token(None).await?;
                headers.insert(AUTHORIZATION, bearer(&token)?);
                Ok(Some(token))
            }
            Method::Sigv4(signer) => {
                signer.sign("POST", url, headers, body, chrono::Utc::now())?;
                Ok(None)
            }
        }
    }

    /// After HTTP 401: true when a different token is now available, in which
    /// case the rejected request may be sent once more.
    pub(super) async fn renew(&self, rejected: Option<&str>) -> bool {
        match (&self.0, rejected) {
            (Method::Token(source), Some(rejected)) => source
                .token(Some(rejected))
                .await
                .is_ok_and(|token| token != rejected),
            _ => false,
        }
    }
}

struct Sigv4 {
    access_key_id: String,
    secret_access_key: String,
    session_token: Option<String>,
    region: String,
    service: String,
}

fn hmac(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC accepts keys of any length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

// RFC 3986 unreserved characters pass through; everything else is %XX.
fn uri_encode(value: &str, keep_slash: bool) -> String {
    value
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            b'/' if keep_slash => "/".into(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

impl Sigv4 {
    fn sign(
        &self,
        method: &str,
        url: &url::Url,
        headers: &mut HeaderMap,
        body: &[u8],
        now: chrono::DateTime<chrono::Utc>,
    ) -> EngineResult<()> {
        let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
        let date = &amz_date[..8];
        let host = match (url.host_str(), url.port()) {
            (Some(host), Some(port)) => format!("{host}:{port}"),
            (Some(host), None) => host.to_owned(),
            _ => return Err(error("invalid MCP endpoint")),
        };
        let mut signed = vec![("host", host), ("x-amz-date", amz_date.clone())];
        if let Some(token) = &self.session_token {
            signed.push(("x-amz-security-token", token.clone()));
        }
        let canonical_headers: String = signed
            .iter()
            .map(|(name, value)| format!("{name}:{}\n", value.trim()))
            .collect();
        let signed_headers = signed
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
            .join(";");
        // Services other than S3 encode the already-encoded path once more.
        let path = if url.path().is_empty() {
            "/"
        } else {
            url.path()
        };
        let mut query: Vec<(String, String)> = url
            .query_pairs()
            .map(|(k, v)| (uri_encode(&k, false), uri_encode(&v, false)))
            .collect();
        query.sort();
        let query = query
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&");
        let canonical_request = format!(
            "{method}\n{}\n{query}\n{canonical_headers}\n{signed_headers}\n{}",
            uri_encode(path, true),
            hex::encode(Sha256::digest(body))
        );
        let scope = format!("{date}/{}/{}/aws4_request", self.region, self.service);
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
            hex::encode(Sha256::digest(canonical_request.as_bytes()))
        );
        let key = [date, &self.region, &self.service, "aws4_request"]
            .iter()
            .fold(
                format!("AWS4{}", self.secret_access_key).into_bytes(),
                |key, part| hmac(&key, part.as_bytes()),
            );
        let signature = hex::encode(hmac(&key, string_to_sign.as_bytes()));
        headers.insert(
            "x-amz-date",
            HeaderValue::from_str(&amz_date).map_err(|_| error("invalid AWS date"))?,
        );
        if let Some(token) = &self.session_token {
            headers.insert("x-amz-security-token", sensitive(token.clone())?);
        }
        headers.insert(
            AUTHORIZATION,
            sensitive(format!(
                "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
                self.access_key_id
            ))?,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sigv4_matches_the_aws_test_suite() {
        // aws-sig-v4-test-suite "get-vanilla" and "post-vanilla".
        let signer = Sigv4 {
            access_key_id: "AKIDEXAMPLE".into(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".into(),
            session_token: None,
            region: "us-east-1".into(),
            service: "service".into(),
        };
        let now = chrono::DateTime::parse_from_rfc3339("2015-08-30T12:36:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let url = url::Url::parse("https://example.amazonaws.com/").unwrap();
        for (method, signature) in [
            (
                "GET",
                "5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31",
            ),
            (
                "POST",
                "5da7c1a2acd57cee7505fc6676e4e544621c30862966e37dddb68e92efbe5d6b",
            ),
        ] {
            let mut headers = HeaderMap::new();
            signer.sign(method, &url, &mut headers, b"", now).unwrap();
            assert_eq!(headers["x-amz-date"], "20150830T123600Z");
            assert_eq!(
                headers[AUTHORIZATION].to_str().unwrap(),
                format!("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, SignedHeaders=host;x-amz-date, Signature={signature}")
            );
            assert!(headers[AUTHORIZATION].is_sensitive());
        }
    }

    #[test]
    fn configuration_round_trips_and_rejects_unsafe_settings() {
        let config: AuthConfig = serde_json::from_value(json!({
            "type": "client_credentials",
            "token_endpoint": "https://auth.example.com/oauth/token",
            "client_id": "reincore",
            "scopes": ["tools.read"]
        }))
        .unwrap();
        assert_eq!(
            config,
            AuthConfig::ClientCredentials {
                token_endpoint: "https://auth.example.com/oauth/token".into(),
                client_id: "reincore".into(),
                scopes: vec!["tools.read".into()],
                audience: None,
                resource: None,
                client_auth: ClientAuth::ClientSecretBasic,
            }
        );
        config.validate(true).unwrap();
        assert_eq!(
            serde_json::from_value::<AuthConfig>(json!({"type": "none"})).unwrap(),
            AuthConfig::None
        );
        assert!(serde_json::from_value::<AuthConfig>(json!({"type": "api_key_query"})).is_err());
        assert!(serde_json::from_value::<AuthConfig>(
            json!({"type": "basic", "username": "me", "password": "x"})
        )
        .is_err());

        for invalid in [
            json!({"type": "header", "name": "Content-Type"}),
            json!({"type": "header", "name": "bad header"}),
            json!({"type": "basic", "username": "user:name"}),
            json!({"type": "oauth", "client_id": "x", "token_endpoint": "https://127.0.0.1/token"}),
            json!({"type": "oauth", "client_id": "x", "token_endpoint": "http://auth.example.com/token"}),
            json!({"type": "oauth", "client_id": "x", "token_endpoint": "https://auth.example.com/token", "scopes": ["a b"]}),
            json!({"type": "client_credentials", "client_id": "x", "token_endpoint": "https://auth.example.com/token", "client_auth": "none"}),
            json!({"type": "google_service_account", "scopes": []}),
            json!({"type": "aws_sigv4", "access_key_id": "short", "region": "us-east-1", "service": "aws-mcp"}),
            json!({"type": "aws_sigv4", "access_key_id": "AKIAIOSFODNN7EXAMPLE", "region": "US-EAST-1", "service": "aws-mcp"}),
        ] {
            let config: AuthConfig = serde_json::from_value(invalid.clone()).unwrap();
            assert!(config.validate(true).is_err(), "{invalid}");
        }
        // A local server may be a loopback HTTP resource; a cloud one may not.
        let loopback = AuthConfig::Oauth {
            client_id: "x".into(),
            token_endpoint: "https://auth.example.com/token".into(),
            scopes: vec![],
            resource: Some("http://127.0.0.1:9000/mcp".into()),
            client_auth: ClientAuth::None,
            issuer: None,
            managed_client: false,
        };
        loopback.validate(false).unwrap();
        assert!(loopback.validate(true).is_err());
        // Authorization is a valid custom scheme header, but not an extra header.
        AuthConfig::Header {
            name: "Authorization".into(),
        }
        .validate(true)
        .unwrap();
        assert!(validate_headers(&BTreeMap::from([("Authorization".into(), "x".into())])).is_err());
        assert!(validate_headers(&BTreeMap::from([(
            "x-goog-user-project".into(),
            "p\r\n".into()
        )]))
        .is_err());
        validate_headers(&BTreeMap::from([(
            "x-goog-user-project".into(),
            "my-project".into(),
        )]))
        .unwrap();
    }

    #[test]
    fn secrets_are_checked_against_their_type_and_never_printed() {
        let secret = AuthSecret::from_secret("token with space");
        assert!(AuthConfig::Bearer.validate_secret(&secret).is_err());
        AuthConfig::Header {
            name: "x-api-key".into(),
        }
        .validate_secret(&secret)
        .unwrap();
        AuthConfig::Basic {
            username: "me@example.com".into(),
        }
        .validate_secret(&secret)
        .unwrap();
        assert!(AuthConfig::None.validate_secret(&secret).is_err());
        assert!(AuthConfig::Bearer
            .validate_secret(&AuthSecret {
                refresh_token: Some("injected".into()),
                ..AuthSecret::from_secret("token")
            })
            .is_err());
        assert!(AuthConfig::GoogleServiceAccount {
            scopes: vec!["scope".into()],
            subject: None
        }
        .validate_secret(&AuthSecret::from_secret("{\"type\":\"service_account\"}"))
        .is_err());
        assert_eq!(format!("{secret:?}"), "AuthSecret(redacted)");
    }

    #[tokio::test]
    async fn static_types_set_their_headers() {
        let url = url::Url::parse("https://mcp.example.com/mcp").unwrap();
        let credential = || Some(Credential::new(AuthSecret::from_secret("s3cret")));
        for (config, name, value) in [
            (AuthConfig::Bearer, "authorization", "Bearer s3cret"),
            (
                AuthConfig::Header {
                    name: "X-Api-Key".into(),
                },
                "x-api-key",
                "s3cret",
            ),
            (
                AuthConfig::Basic {
                    username: "me".into(),
                },
                "authorization",
                "Basic bWU6czNjcmV0",
            ),
        ] {
            let auth = Authenticator::new(&config, credential(), true).unwrap();
            let mut headers = HeaderMap::new();
            assert!(auth
                .apply(&url, &mut headers, b"{}")
                .await
                .unwrap()
                .is_none());
            assert_eq!(headers[name], value);
            assert!(headers[name].is_sensitive());
            assert!(!auth.renew(None).await);
        }
        assert!(Authenticator::new(&AuthConfig::Bearer, None, true).is_err());
        let none = Authenticator::new(&AuthConfig::None, None, true).unwrap();
        let mut headers = HeaderMap::new();
        none.apply(&url, &mut headers, b"").await.unwrap();
        assert!(headers.is_empty());
    }
}
