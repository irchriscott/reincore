//! GitHub Copilot provider adapter.
//!
//! Uses the GitHub OAuth Device Flow to authenticate the user, then exchanges
//! the OAuth token for a short-lived Copilot token via the internal token
//! endpoint. The Copilot token is used to authenticate requests to
//! `api.githubcopilot.com`, which speaks the OpenAI Chat Completions format
//! and provides access to all Copilot models (OpenAI, Anthropic, etc.).
//!
//! ## Setup
//!
//! Create a GitHub OAuth App at https://github.com/settings/developers:
//! - Homepage URL: any URL
//! - Authorization callback URL: `http://localhost:8989/callback` (or your reincore port)
//!
//! Set the client ID and secret as env vars:
//! ```bash
//! export COPILOT_CLIENT_ID=Ov23li...
//! export COPILOT_CLIENT_SECRET=...
//! ```
//!
//! ## Tiers
//!
//! - **Free tier**: Limited to a single model. All tier mappings (Lite/Pro/Ultra)
//!   resolve to the same model.
//! - **Paid tier**: Access to multiple models from different providers
//!   (OpenAI, Anthropic). The user selects which provider's models to use,
//!   and the tier mappings resolve to models from that provider.

use crate::error::{describe_request_error as engine_request_error, EngineError, EngineResult};
use crate::models::{
    ChatCompletionRequest, ChatMessage, MessageContent, ModelTier, ToolCall, ToolDefinition,
};
use crate::provider::{
    capture_response_headers, ChatCompletionResponse, Provider, ProviderConfig, ProviderModel,
};
use async_trait::async_trait;
use colored::Colorize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::debug;

/// Re-export of `crate::log` to avoid conflict with the `log` crate that
/// `tracing` re-exports. Using `rc_log::` makes it unambiguous.
use crate::log as rc_log;

/// The GitHub Copilot token endpoint (business/enterprise only).
const GITHUB_COPILOT_TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";

/// The default Copilot API base URL (business/enterprise).
const COPILOT_API_BASE_URL_DEFAULT: &str = "https://api.githubcopilot.com";

/// The Copilot API base URL for individual subscribers.
const COPILOT_API_BASE_URL_INDIVIDUAL: &str = "https://api.individual.githubcopilot.com";

/// Default OAuth App credentials shipped with reincore. Users can override
/// these with `COPILOT_CLIENT_ID` and `COPILOT_CLIENT_SECRET` env vars if they
/// want to use their own OAuth App.
///
/// To create your own OAuth App: https://github.com/settings/developers
/// - Homepage URL: https://github.com/irchriscott/reincorehq
/// - Authorization callback URL: http://localhost:8989/callback
const DEFAULT_CLIENT_ID: &str = "Ov23lie0bU8HNzPXxMYN"; // Replace with your OAuth App client ID
const DEFAULT_CLIENT_SECRET: &str = "b028eb6e39883055636b9e8e1eeefef9b1c606a3"; // Replace with your OAuth App client secret

/// The scope requested for Copilot access.
const COPILOT_OAUTH_SCOPE: &str = "read:user";

/// Models available on the free Copilot tier.
const FREE_TIER_MODEL: &str = "gpt-4o-2024-08-06";

/// Models available on the paid Copilot tier, per provider family.
/// Each tuple is (Lite, Pro, Ultra).
const PAID_TIER_OPENAI_MODELS: (&str, &str, &str) =
    ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol");

const PAID_TIER_ANTHROPIC_MODELS: (&str, &str, &str) =
    ("claude-haiku-4-5", "claude-sonnet-5", "claude-fable-5");

/// Cached Copilot token with its expiry and tier info.
struct CachedToken {
    /// The token used for API requests (Copilot token for business, raw OAuth
    /// token for individual).
    copilot_token: String,
    /// The API base URL to use (varies by plan).
    api_base_url: String,
    expires_at: Instant,
    /// Whether the account is on the free tier (limited models).
    is_free_tier: bool,
}

/// Which model family to use on the paid tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopilotModelFamily {
    OpenAI,
    Anthropic,
}

impl CopilotModelFamily {
    pub fn as_str(&self) -> &'static str {
        match self {
            CopilotModelFamily::OpenAI => "openai",
            CopilotModelFamily::Anthropic => "anthropic",
        }
    }
}

impl std::str::FromStr for CopilotModelFamily {
    type Err = EngineError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "openai" | "gpt" | "oai" => Ok(Self::OpenAI),
            "anthropic" | "claude" => Ok(Self::Anthropic),
            _ => Err(EngineError::InvalidInput(format!(
                "unknown Copilot model family: {value}"
            ))),
        }
    }
}

/// Global OAuth token cache - shared across all CopilotProvider instances
/// so that the pre-auth check_tier() call and the proxy's provider don't
/// each trigger a separate device flow login.
static GLOBAL_OAUTH_TOKEN: std::sync::OnceLock<Arc<Mutex<Option<String>>>> =
    std::sync::OnceLock::new();

/// Get the global OAuth token cache.
pub fn get_global_oauth_token() -> Arc<Mutex<Option<String>>> {
    GLOBAL_OAUTH_TOKEN
        .get_or_init(|| Arc::new(Mutex::new(None)))
        .clone()
}

/// Global Copilot token cache - shared across all CopilotProvider instances.
static GLOBAL_COPILOT_TOKEN: std::sync::OnceLock<Arc<Mutex<Option<CachedToken>>>> =
    std::sync::OnceLock::new();

/// Get the global Copilot token cache.
fn global_copilot_token() -> Arc<Mutex<Option<CachedToken>>> {
    GLOBAL_COPILOT_TOKEN
        .get_or_init(|| Arc::new(Mutex::new(None)))
        .clone()
}

/// Multi-tenant Copilot token cache for the cloud path, keyed by the
/// caller's OAuth token. Unlike the single-entry `GLOBAL_COPILOT_TOKEN`
/// (single-user CLI), the cloud engine builds a fresh `CopilotProvider` per
/// request for many different users, so the exchanged Copilot token / tier
/// must be cached per OAuth token - never shared across tenants - while
/// still being reused across the short-lived per-request instances.
static CLOUD_COPILOT_TOKENS: std::sync::OnceLock<
    Mutex<std::collections::HashMap<String, CachedToken>>,
> = std::sync::OnceLock::new();

/// Get the cloud (token-keyed) Copilot token cache.
fn cloud_copilot_tokens() -> &'static Mutex<std::collections::HashMap<String, CachedToken>> {
    CLOUD_COPILOT_TOKENS.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// GitHub Copilot provider implementation.
pub struct CopilotProvider {
    /// The GitHub OAuth client ID.
    client_id: String,
    /// The GitHub OAuth client secret.
    client_secret: String,
    /// The GitHub instance URL (e.g. "https://github.com" or
    /// "https://enterprise.ghe.com").
    github_instance: String,
    /// Provider config (model tier mappings, etc.).
    config: ProviderConfig,
    /// HTTP client.
    client: reqwest::Client,
    /// Which model family to use on the paid tier (OpenAI or Anthropic).
    /// Ignored on the free tier.
    model_family: CopilotModelFamily,
    /// Cloud/multi-tenant path: a pre-obtained OAuth token. When set, the
    /// interactive device flow is skipped and the token-keyed
    /// `CLOUD_COPILOT_TOKENS` cache is used instead of the single-user
    /// process-global caches. `None` keeps the original single-user CLI
    /// behavior (device flow + global caches).
    oauth_override: Option<String>,
}

impl CopilotProvider {
    /// Resolve the OAuth client ID, using the env var override or the
    /// default shipped with reincore.
    pub fn resolve_client_id() -> String {
        std::env::var("COPILOT_CLIENT_ID").unwrap_or_else(|_| DEFAULT_CLIENT_ID.to_string())
    }

    /// Resolve the OAuth client secret, using the env var override or the
    /// default shipped with reincore.
    pub fn resolve_client_secret() -> String {
        std::env::var("COPILOT_CLIENT_SECRET").unwrap_or_else(|_| DEFAULT_CLIENT_SECRET.to_string())
    }

    /// Resolve the GitHub instance URL. Defaults to `https://github.com`.
    /// Can be overridden with the `COPILOT_GITHUB_INSTANCE` env var (e.g.
    /// `https://enterprise.ghe.com`).
    pub fn resolve_github_instance() -> String {
        std::env::var("COPILOT_GITHUB_INSTANCE")
            .unwrap_or_else(|_| "https://github.com".to_string())
    }

    /// Create a new Copilot provider from OAuth credentials and provider config.
    /// `github_instance` is the base URL of the GitHub instance (e.g.
    /// "https://github.com" or "https://enterprise.ghe.com").
    pub fn new(
        client_id: String,
        client_secret: String,
        github_instance: String,
        config: ProviderConfig,
        model_family: CopilotModelFamily,
    ) -> EngineResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            // Separate from the overall timeout: an unreachable endpoint should
            // fail in seconds, not after a generation-length wait.
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| EngineError::Provider(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            client_id,
            client_secret,
            github_instance,
            config,
            client,
            model_family,
            oauth_override: None,
        })
    }

    /// Create a pre-authenticated Copilot provider for the multi-tenant
    /// cloud engine. `oauth_token` is a GitHub OAuth token already obtained
    /// out-of-band (the control plane runs the device flow at setup time and
    /// stores the token). No device flow is performed; the token is exchanged
    /// for a Copilot token lazily and cached per token in
    /// `CLOUD_COPILOT_TOKENS` so it is never shared across tenants. The tier
    /// (free vs paid) is detected live from GitHub. `github_instance` is the
    /// base URL of the GitHub instance the token was issued for.
    pub fn with_oauth_token(
        oauth_token: String,
        github_instance: String,
        config: ProviderConfig,
        model_family: CopilotModelFamily,
    ) -> EngineResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            // Separate from the overall timeout: an unreachable endpoint should
            // fail in seconds, not after a generation-length wait.
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| EngineError::Provider(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            client_id: Self::resolve_client_id(),
            client_secret: Self::resolve_client_secret(),
            github_instance,
            config,
            client,
            model_family,
            oauth_override: Some(oauth_token),
        })
    }

    /// Get the API base URL for the GitHub instance. For github.com this is
    /// `https://api.github.com`. For GHE instances, the API is at
    /// `<instance>/api/v3`.
    fn github_instance_api(&self) -> String {
        if self.github_instance == "https://github.com" {
            "https://api.github.com".to_string()
        } else {
            format!("{}/api/v3", self.github_instance.trim_end_matches('/'))
        }
    }

    /// Get the OAuth token, performing the device flow if needed.
    /// Uses a global cache so all CopilotProvider instances share the same
    /// OAuth token (prevents re-login when the proxy creates a new instance).
    pub async fn get_oauth_token(&self) -> EngineResult<String> {
        // Cloud path: a token was supplied out-of-band; never run the
        // interactive device flow (there is no TTY in the cloud engine).
        if let Some(token) = &self.oauth_override {
            return Ok(token.clone());
        }

        let oauth_cache = get_global_oauth_token();

        // Check if we already have a cached OAuth token.
        {
            let cached = oauth_cache.lock().unwrap();
            if let Some(token) = &*cached {
                return Ok(token.clone());
            }
        }

        // Perform the device flow.
        let token = self.device_flow_login().await?;

        // Cache it.
        {
            let mut cached = oauth_cache.lock().unwrap();
            *cached = Some(token.clone());
        }

        Ok(token)
    }

    /// Run the GitHub OAuth Device Flow to obtain an access token.
    async fn device_flow_login(&self) -> EngineResult<String> {
        debug!(
            "{}",
            rc_log::provider_line(
                "copilot",
                &format!("Starting OAuth device flow on {}", self.github_instance)
            )
        );

        let device_code_url = format!("{}/login/device/code", self.github_instance);
        let oauth_token_url = format!("{}/login/oauth/access_token", self.github_instance);

        // Step 1: Request a device code.
        let response = self
            .client
            .post(&device_code_url)
            .header("Accept", "application/json")
            .header("User-Agent", "reincore/1.0.0")
            .form(&[
                ("client_id", self.client_id.as_str()),
                ("scope", COPILOT_OAUTH_SCOPE),
            ])
            .send()
            .await
            .map_err(|e| {
                EngineError::Provider(format!(
                    "failed to request device code: {}",
                    engine_request_error(&e)
                ))
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(EngineError::Provider(format!(
                "GitHub device code request failed ({}): {} - ensure COPILOT_CLIENT_ID is set correctly",
                status, body
            )));
        }

        let device_resp: serde_json::Value = response.json().await.map_err(|e| {
            EngineError::Provider(format!("failed to parse device code response: {e}"))
        })?;

        let user_code = device_resp
            .get("user_code")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                EngineError::Provider("device code response missing 'user_code'".to_string())
            })?
            .to_string();
        let device_code = device_resp
            .get("device_code")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                EngineError::Provider("device code response missing 'device_code'".to_string())
            })?
            .to_string();
        let verification_uri = device_resp
            .get("verification_uri")
            .and_then(|v| v.as_str())
            .unwrap_or("https://github.com/login/device")
            .to_string();
        let interval = device_resp
            .get("interval")
            .and_then(|v| v.as_u64())
            .unwrap_or(5);
        let expires_in = device_resp
            .get("expires_in")
            .and_then(|v| v.as_u64())
            .unwrap_or(900);

        // Step 2: Show the user the code and URL.
        println!(
            "\n┌─── {} ──────────────────────────────",
            "GitHub Copilot Login".cyan().bold()
        );
        println!("│ {}", "To authenticate with GitHub Copilot:".white());
        println!(
            "│  1. {} {}",
            "Open:".dimmed(),
            verification_uri.green().underline()
        );
        println!(
            "│  2. {} {}",
            "Enter the code:".dimmed(),
            user_code.yellow().bold()
        );
        println!("│ {}", "Waiting for authorization...".dimmed());
        println!("└────────────────────────────────────────────────────────\n");

        // Step 3: Poll for the access token.
        let poll_interval = Duration::from_secs(interval);
        let deadline = Instant::now() + Duration::from_secs(expires_in);

        loop {
            if Instant::now() > deadline {
                return Err(EngineError::Provider(
                    "GitHub device flow timed out waiting for user authorization".to_string(),
                ));
            }

            tokio::time::sleep(poll_interval).await;

            let response = self
                .client
                .post(&oauth_token_url)
                .header("Accept", "application/json")
                .header("User-Agent", "reincore/1.0.0")
                .form(&[
                    ("client_id", self.client_id.as_str()),
                    ("client_secret", self.client_secret.as_str()),
                    ("device_code", device_code.as_str()),
                    ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ])
                .send()
                .await
                .map_err(|e| {
                    EngineError::Provider(format!(
                        "failed to poll for token: {}",
                        engine_request_error(&e)
                    ))
                })?;

            let token_resp: serde_json::Value = response.json().await.map_err(|e| {
                EngineError::Provider(format!("failed to parse token response: {e}"))
            })?;

            // Check for error responses.
            if let Some(error) = token_resp.get("error").and_then(|v| v.as_str()) {
                match error {
                    "authorization_pending" => continue, // Keep polling
                    "slow_down" => {
                        tokio::time::sleep(poll_interval).await;
                        continue;
                    }
                    "expired_token" => {
                        return Err(EngineError::Provider(
                            "GitHub device code expired. Please try again.".to_string(),
                        ));
                    }
                    "access_denied" => {
                        return Err(EngineError::Provider(
                            "User denied authorization request.".to_string(),
                        ));
                    }
                    _ => {
                        let desc = token_resp
                            .get("error_description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown error");
                        return Err(EngineError::Provider(format!(
                            "GitHub OAuth error: {} - {}",
                            error, desc
                        )));
                    }
                }
            }

            // Success - we got an access token.
            let access_token = token_resp
                .get("access_token")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    EngineError::Provider("token response missing 'access_token'".to_string())
                })?
                .to_string();

            println!("{}\n", rc_log::success("GitHub authentication successful!"));
            return Ok(access_token);
        }
    }

    /// Resolve the Copilot token and API endpoint. For individual subscribers,
    /// the raw OAuth token is used directly with `api.individual.githubcopilot.com`.
    /// For business/enterprise, the OAuth token is exchanged for a Copilot
    /// token via `copilot_internal/v2/token` and used with `api.githubcopilot.com`.
    ///
    /// Returns (token, api_base_url, is_free_tier).
    async fn get_copilot_token(&self) -> EngineResult<(String, String, bool)> {
        // Cloud path: cache keyed by the caller's OAuth token so tenants are
        // isolated but the exchange is reused across per-request instances.
        if let Some(oauth_token) = self.oauth_override.clone() {
            {
                let cache = cloud_copilot_tokens().lock().unwrap();
                if let Some(token) = cache.get(&oauth_token) {
                    if token.expires_at > Instant::now() + Duration::from_secs(60) {
                        return Ok((
                            token.copilot_token.clone(),
                            token.api_base_url.clone(),
                            token.is_free_tier,
                        ));
                    }
                }
            }

            let (api_token, api_base_url, is_free_tier) =
                self.resolve_copilot_token(&oauth_token).await?;

            {
                let mut cache = cloud_copilot_tokens().lock().unwrap();
                cache.insert(
                    oauth_token,
                    CachedToken {
                        copilot_token: api_token.clone(),
                        api_base_url: api_base_url.clone(),
                        expires_at: Instant::now() + Duration::from_secs(1800),
                        is_free_tier,
                    },
                );
            }

            return Ok((api_token, api_base_url, is_free_tier));
        }

        // CLI path: single-user process-global cache + interactive device flow.
        let copilot_cache = global_copilot_token();

        // Check cache first.
        {
            let cached = copilot_cache.lock().unwrap();
            if let Some(token) = &*cached {
                if token.expires_at > Instant::now() + Duration::from_secs(60) {
                    return Ok((
                        token.copilot_token.clone(),
                        token.api_base_url.clone(),
                        token.is_free_tier,
                    ));
                }
            }
        }

        // Get the OAuth token (performs device flow if needed).
        let oauth_token = self.get_oauth_token().await?;

        let (api_token, api_base_url, is_free_tier) =
            self.resolve_copilot_token(&oauth_token).await?;

        // Cache the result.
        {
            let mut cached = copilot_cache.lock().unwrap();
            *cached = Some(CachedToken {
                copilot_token: api_token.clone(),
                api_base_url: api_base_url.clone(),
                expires_at: Instant::now() + Duration::from_secs(1800),
                is_free_tier,
            });
        }

        Ok((api_token, api_base_url, is_free_tier))
    }

    /// Best-effort free-tier guess from the GitHub *account* plan. Only used
    /// as a fallback when `/copilot_internal/user` is unavailable; the account
    /// plan is not the Copilot subscription, so this defaults to `true` (free)
    /// on any uncertainty.
    async fn github_account_is_free(&self, oauth_token: &str) -> bool {
        let user_api_url = format!("{}/user", self.github_instance_api());
        let response = match self
            .client
            .get(&user_api_url)
            .header("Authorization", format!("Bearer {}", oauth_token))
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .header("User-Agent", "reincore/1.0.0")
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => resp,
            _ => return true,
        };
        let raw: serde_json::Value = match response.json().await {
            Ok(v) => v,
            Err(_) => return true,
        };
        raw.get("plan")
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .map(|name| name.to_ascii_lowercase().contains("free"))
            .unwrap_or(true)
    }

    /// Given an OAuth token, detect the tier and resolve the Copilot API
    /// token + endpoint. Individual subscribers use the raw OAuth token with
    /// the individual endpoint; business/enterprise exchange it for a
    /// short-lived Copilot token. Shared by the CLI and cloud paths (which
    /// differ only in how they obtain and cache the OAuth token).
    ///
    /// Returns (token, api_base_url, is_free_tier). Does not cache.
    async fn resolve_copilot_token(
        &self,
        oauth_token: &str,
    ) -> EngineResult<(String, String, bool)> {
        // The authoritative source for the Copilot *subscription* tier is the
        // Copilot-specific user endpoint. The GitHub *account* plan (/user ->
        // plan.name) is a different thing entirely: the vast majority of
        // Copilot Pro / Pro+ subscribers have a "free" personal GitHub account,
        // so keying the tier off it misreports paying subscribers as free tier.
        // Derive both the tier and the endpoint from `/copilot_internal/user`,
        // and only fall back to the account plan when that endpoint is
        // unavailable.
        let copilot_user_url = format!("{}/copilot_internal/user", self.github_instance_api());
        let copilot_response = self
            .client
            .get(&copilot_user_url)
            .header("Authorization", format!("Bearer {}", oauth_token))
            .header("Accept", "application/json")
            .header("User-Agent", "reincore/1.0.0")
            .header("Editor-Version", "vscode/1.99.0")
            .header("Editor-Plugin-Version", "copilot-chat/0.25.0")
            .send()
            .await;

        // `copilot_plan` is one of "free", "individual" (Copilot Pro / Pro+),
        // "business", or "enterprise"; `access_type_sku` carries finer detail
        // such as "free_limited_copilot" or "copilot_pro_plus". Either
        // signalling "free" marks the free tier.
        let copilot_info: Option<serde_json::Value> = match copilot_response {
            Ok(resp) => {
                let resp_status = resp.status();
                if resp_status.is_success() {
                    let raw: serde_json::Value = resp.json().await.map_err(|e| {
                        EngineError::Provider(format!("failed to parse Copilot user info: {e}"))
                    })?;
                    debug!(
                        "{}",
                        rc_log::provider_line("copilot", &format!("Copilot user info: {}", raw))
                    );
                    Some(raw)
                } else {
                    debug!(
                        "{}",
                        rc_log::provider_line(
                            "copilot",
                            &format!("Copilot user info request failed ({})", resp_status)
                        )
                    );
                    None
                }
            }
            Err(e) => {
                debug!(
                    "{}",
                    rc_log::provider_line(
                        "copilot",
                        &format!("Copilot user info request failed: {}", e)
                    )
                );
                None
            }
        };

        let (is_free_tier, is_individual) = if let Some(info) = &copilot_info {
            let copilot_plan = info
                .get("copilot_plan")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            let access_sku = info
                .get("access_type_sku")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_ascii_lowercase();

            // Free tier only when the Copilot plan/SKU explicitly says so; a
            // populated non-free plan (individual/business/enterprise) is paid.
            let is_free = copilot_plan.contains("free") || access_sku.contains("free");
            // Business/enterprise use the exchanged token + default endpoint;
            // everything else (free, individual/Pro/Pro+) is an individual
            // subscriber using the raw OAuth token + individual endpoint.
            let is_individual =
                !(copilot_plan.contains("business") || copilot_plan.contains("enterprise"));

            debug!(
                "{}",
                rc_log::provider_line(
                    "copilot",
                    &format!(
                        "Copilot plan: {} (sku: {}, free: {}, individual: {})",
                        copilot_plan, access_sku, is_free, is_individual
                    )
                )
            );
            (is_free, is_individual)
        } else {
            // Fallback: /copilot_internal/user unavailable. Best-effort guess
            // from the GitHub account plan (safer default: free), and assume an
            // individual endpoint.
            let is_free = self.github_account_is_free(oauth_token).await;
            debug!(
                "{}",
                rc_log::provider_line(
                    "copilot",
                    &format!(
                        "Copilot user info unavailable; falling back to GitHub account plan (free: {})",
                        is_free
                    )
                )
            );
            (is_free, true)
        };

        let (api_token, api_base_url) = if is_individual {
            // Individual plan: use the raw OAuth token directly, no exchange.
            debug!(
                "{}",
                rc_log::provider_line(
                    "copilot",
                    "Individual plan: using raw OAuth token with individual endpoint"
                )
            );
            (
                oauth_token.to_string(),
                COPILOT_API_BASE_URL_INDIVIDUAL.to_string(),
            )
        } else {
            // Business/enterprise: exchange the OAuth token for a Copilot token.
            debug!(
                "{}",
                rc_log::provider_line(
                    "copilot",
                    "Business/enterprise plan: exchanging OAuth token for Copilot token"
                )
            );
            let response = self
                .client
                .get(GITHUB_COPILOT_TOKEN_URL)
                .header("Authorization", format!("Bearer {}", oauth_token))
                .header("Accept", "application/json")
                .header("User-Agent", "reincore/1.0.0")
                .header("Editor-Version", "vscode/1.99.0")
                .header("Editor-Plugin-Version", "copilot-chat/0.25.0")
                .header("Copilot-Integration-Id", "vscode-chat")
                .send()
                .await
                .map_err(|e| {
                    EngineError::Provider(format!(
                        "failed to fetch Copilot token: {}",
                        engine_request_error(&e)
                    ))
                })?;

            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                return Err(EngineError::Provider(format!(
                    "GitHub Copilot token exchange failed ({}): {} - ensure your GitHub account has Copilot access",
                    status, body
                )));
            }

            let raw: serde_json::Value = response.json().await.map_err(|e| {
                EngineError::Provider(format!("failed to parse Copilot token response: {e}"))
            })?;

            let copilot_token = raw
                .get("token")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    EngineError::Provider(
                        "Copilot token response missing 'token' field".to_string(),
                    )
                })?
                .to_string();

            (copilot_token, COPILOT_API_BASE_URL_DEFAULT.to_string())
        };

        Ok((api_token, api_base_url, is_free_tier))
    }

    /// Eagerly check the subscription tier. Performs the device flow if
    /// needed, then checks the Copilot user info to detect the plan.
    /// Should be called before the proxy starts so the user can be prompted
    /// for a model family (if paid) before any requests are made.
    ///
    /// Returns `true` if the account is on the free tier, `false` if paid.
    pub async fn check_tier(&self) -> EngineResult<bool> {
        crate::privacy::current_policy().check_provider(self.label())?;
        let (_, _, is_free_tier) = self.get_copilot_token().await?;
        Ok(is_free_tier)
    }

    /// Set the model family to use (for paid tier). Called after the user
    /// has been prompted to select a family.
    pub fn set_model_family(&mut self, family: CopilotModelFamily) {
        self.model_family = family;
    }

    /// Get the current model family.
    pub fn model_family(&self) -> CopilotModelFamily {
        self.model_family
    }

    /// Resolve the model for a given tier, taking into account whether the
    /// account is on the free or paid tier.
    fn resolve_model(&self, tier: ModelTier, is_free_tier: bool) -> EngineResult<String> {
        if is_free_tier {
            // Free tier: all tiers resolve to the same limited model.
            return Ok(FREE_TIER_MODEL.to_string());
        }

        // Paid tier: explicit per-tier configuration (control-plane JWT tier
        // mappings or a configured standard model) overrides the built-in
        // family presets. This lets a workspace pin standard/lite/pro/ultra
        // to specific Copilot models. Fall back to the family presets when the
        // config carries no model information (the common single-user case).
        if self.config.standard_model.is_some() || !self.config.models.is_empty() {
            return self.config.model_for_tier(tier);
        }

        // No explicit config: use the built-in model family presets.
        let models = match self.model_family {
            CopilotModelFamily::OpenAI => &PAID_TIER_OPENAI_MODELS,
            CopilotModelFamily::Anthropic => &PAID_TIER_ANTHROPIC_MODELS,
        };

        let model = match tier {
            ModelTier::Lite => models.0,
            ModelTier::Pro => models.1,
            ModelTier::Ultra => models.2,
        };

        Ok(model.to_string())
    }

    fn auth_headers(&self, copilot_token: &str) -> std::collections::HashMap<String, String> {
        let mut headers = std::collections::HashMap::new();
        headers.insert(
            "Authorization".to_string(),
            format!("Bearer {}", copilot_token),
        );
        headers.insert("Editor-Version".to_string(), "vscode/1.99.0".to_string());
        headers.insert(
            "Editor-Plugin-Version".to_string(),
            "copilot-chat/0.25.0".to_string(),
        );
        headers.insert(
            "Copilot-Integration-Id".to_string(),
            "vscode-chat".to_string(),
        );
        headers
    }

    fn build_body(
        &self,
        request: &ChatCompletionRequest,
        provider_model: String,
    ) -> EngineResult<serde_json::Value> {
        let mut messages = Vec::new();
        for m in &request.messages {
            let mut wire = copilot_message_from_chat_message(m);
            if super::replay::enabled(request) {
                super::replay::apply(
                    m,
                    &mut wire,
                    &super::replay::scope(&self.config, &provider_model),
                );
            }
            messages.push(wire);
        }

        let mut body = serde_json::json!({
            "model": provider_model,
            "messages": messages,
        });

        if let Some(tools) = &request.tools {
            let tool_specs: Vec<serde_json::Value> =
                tools.iter().map(copilot_tool_definition).collect();
            body["tools"] = serde_json::Value::Array(tool_specs);
        }

        if let Some(tool_choice) = &request.tool_choice {
            body["tool_choice"] = super::provider::tool_choice_without_allowed_tools(tool_choice);
        }

        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }

        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }

        Ok(body)
    }
}

#[async_trait]
impl Provider for CopilotProvider {
    fn label(&self) -> &str {
        &self.config.label
    }

    fn config(&self) -> &ProviderConfig {
        &self.config
    }

    async fn complete(
        &self,
        request: &ChatCompletionRequest,
    ) -> EngineResult<ChatCompletionResponse> {
        let prepared_request = crate::privacy::prepare_provider_request(request, self.label())?;
        let request = &prepared_request;
        let tier = request
            .extra
            .get("reincore_model_tier")
            .and_then(|v| v.as_str())
            .and_then(|value| value.parse().ok())
            .unwrap_or(ModelTier::Pro);

        // Get the Copilot token and API endpoint (performs device flow + token exchange if needed).
        let (copilot_token, api_base_url, is_free_tier) = self.get_copilot_token().await?;

        // Resolve the model based on tier and subscription level.
        let provider_model = if request
            .extra
            .get("reincore_title_standard_model")
            .and_then(|v| v.as_bool())
            == Some(true)
        {
            self.config.model_for_request(request, tier)?
        } else {
            self.resolve_model(tier, is_free_tier)?
        };
        let mut body = self.build_body(request, provider_model.clone())?;

        // Force stream=false - we don't support streaming through the
        // Copilot provider. The proxy handles non-streaming responses.
        if let Some(obj) = body.as_object_mut() {
            obj.insert("stream".to_string(), serde_json::json!(false));
            obj.remove("stream_options");
        }

        debug!(
            "{}",
            rc_log::provider_line(
                "copilot",
                &format!(
                    "Request to {}: {}",
                    api_base_url,
                    crate::privacy::current_policy()
                        .sanitize_for_logging(&serde_json::to_string(&body).unwrap_or_default())
                )
            )
        );

        let mut req_builder = self
            .client
            .post(format!("{}/chat/completions", api_base_url))
            .header("Content-Type", "application/json");

        // Use the Copilot token for API authentication.
        for (key, value) in self.auth_headers(&copilot_token) {
            req_builder = req_builder.header(key, value);
        }

        let response = req_builder
            .json(&body)
            .send()
            .await
            .map_err(EngineError::from_request_error)?;

        let status = response.status();
        let response_headers = capture_response_headers(response.headers());
        // Read the response as text first, then try to parse as JSON.
        // This avoids "failed to read response" errors when the API returns
        // non-JSON error bodies (e.g. plain text or HTML).
        let response_text = response
            .text()
            .await
            .map_err(|e| EngineError::Provider(format!("failed to read response body: {e}")))?;

        let raw: serde_json::Value = serde_json::from_str(&response_text).unwrap_or_else(|_| {
            serde_json::json!({
                "raw_response": response_text,
            })
        });

        if !status.is_success() {
            return Err(EngineError::Provider(format!(
                "Copilot returned {status}: {}",
                crate::privacy::current_policy().sanitize_for_logging(&raw.to_string())
            )));
        }

        let mut parsed = parse_completion_response(raw)?;
        super::replay::capture_response(
            &mut parsed,
            super::replay::request_scope(
                &self.config,
                body["model"].as_str().unwrap_or_default(),
                request,
            ),
            false,
        );
        parsed.response_headers = response_headers;
        Ok(parsed)
    }

    async fn list_models(&self) -> EngineResult<Vec<ProviderModel>> {
        crate::privacy::current_policy().check_provider(self.label())?;
        // Get the Copilot token to detect the tier.
        let (_, _, is_free_tier) = self.get_copilot_token().await?;

        if is_free_tier {
            // Free tier: only one model available.
            return Ok(vec![ProviderModel {
                id: FREE_TIER_MODEL.to_string(),
                provider: self.label().to_string(),
                display_name: Some(format!("{} (free tier)", FREE_TIER_MODEL)),
                created_at: None,
                metadata: None,
            }]);
        }

        // Paid tier: return models from the selected family.
        let models = match self.model_family {
            CopilotModelFamily::OpenAI => &PAID_TIER_OPENAI_MODELS,
            CopilotModelFamily::Anthropic => &PAID_TIER_ANTHROPIC_MODELS,
        };

        Ok(vec![
            ProviderModel {
                id: models.0.to_string(),
                provider: self.label().to_string(),
                display_name: Some(format!("{} (lite)", models.0)),
                created_at: None,
                metadata: None,
            },
            ProviderModel {
                id: models.1.to_string(),
                provider: self.label().to_string(),
                display_name: Some(format!("{} (pro)", models.1)),
                created_at: None,
                metadata: None,
            },
            ProviderModel {
                id: models.2.to_string(),
                provider: self.label().to_string(),
                display_name: Some(format!("{} (ultra)", models.2)),
                created_at: None,
                metadata: None,
            },
        ])
    }

    async fn check_tier(&self) -> EngineResult<Option<bool>> {
        let is_free_tier = self.check_tier().await?;
        Ok(Some(is_free_tier))
    }

    fn model_family(&self) -> Option<&str> {
        Some(self.model_family.as_str())
    }
}

fn copilot_message_from_chat_message(msg: &ChatMessage) -> serde_json::Value {
    let mut value = serde_json::json!({
        "role": msg.role,
    });

    if let Some(content) = &msg.content {
        value["content"] = serde_json::to_value(content).unwrap_or_default();
    }

    if let Some(tool_calls) = &msg.tool_calls {
        value["tool_calls"] =
            serde_json::json!(tool_calls.iter().map(copilot_tool_call).collect::<Vec<_>>());
    }

    if let Some(tool_call_id) = &msg.tool_call_id {
        value["tool_call_id"] = serde_json::json!(tool_call_id);
    }

    if let Some(name) = &msg.name {
        value["name"] = serde_json::json!(name);
    }

    value
}

#[cfg(test)]
mod replay_tests {
    use super::*;

    #[test]
    fn copilot_replays_only_matching_model_state() {
        let config = ProviderConfig {
            label: "copilot".into(),
            ..Default::default()
        };
        let provider = CopilotProvider::new(
            String::new(),
            String::new(),
            "https://github.com".into(),
            config.clone(),
            CopilotModelFamily::OpenAI,
        )
        .unwrap();
        let mut request: ChatCompletionRequest = serde_json::from_value(
            serde_json::json!({"model":"m","messages":[{"role":"assistant","content":"working"}]}),
        )
        .unwrap();
        request.extra.insert(
            "reincore_provider_replay_enabled".into(),
            serde_json::json!(true),
        );
        let state = super::super::replay::capture(super::super::replay::scope(&config, "m"), &serde_json::json!({"reasoning_details":[{"type":"reasoning.encrypted","data":"opaque"}]}), false).unwrap().bind(&request.messages[0]);
        request.messages[0].provider_state = Some(state);
        let body = provider.build_body(&request, "m".into()).unwrap();
        assert_eq!(
            body["messages"][0]["reasoning_details"][0]["data"],
            "opaque"
        );
        let changed = provider.build_body(&request, "different".into()).unwrap();
        assert!(changed["messages"][0].get("reasoning_details").is_none());
        request.extra.insert(
            "reincore_provider_replay_enabled".into(),
            serde_json::json!(false),
        );
        let disabled = provider.build_body(&request, "m".into()).unwrap();
        assert!(disabled["messages"][0].get("reasoning_details").is_none());
        assert!(disabled.get("reincore_provider_replay_enabled").is_none());
    }
}

fn copilot_tool_definition(tool: &ToolDefinition) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
    })
}

fn copilot_tool_call(call: &ToolCall) -> serde_json::Value {
    serde_json::json!({
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.function.name,
            "arguments": call.function.arguments,
        }
    })
}

fn parse_completion_response(raw: serde_json::Value) -> EngineResult<ChatCompletionResponse> {
    let id = raw
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let model = raw
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let choice = raw
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
        .ok_or_else(|| EngineError::Provider("no choices in response".to_string()))?;

    let message = choice
        .get("message")
        .ok_or_else(|| EngineError::Provider("no message in first choice".to_string()))?;

    let content = message
        .get("content")
        .and_then(|c| c.as_str())
        .map(|s| MessageContent::text(s.to_string()));

    let tool_calls: Option<Vec<ToolCall>> = message
        .get("tool_calls")
        .and_then(|tc| tc.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|tc| {
                    let id = tc.get("id").and_then(|v| v.as_str())?.to_string();
                    let function = tc.get("function")?;
                    let name = function.get("name").and_then(|v| v.as_str())?.to_string();
                    let arguments = function
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}")
                        .to_string();
                    Some(ToolCall {
                        id,
                        call_type: "function".to_string(),
                        function: crate::models::FunctionCall { name, arguments },
                    })
                })
                .collect()
        });

    let _finish_reason = choice
        .get("finish_reason")
        .and_then(|f| f.as_str())
        .unwrap_or("stop")
        .to_string();

    let usage = raw
        .get("usage")
        .and_then(super::provider::parse_openai_stream_usage);

    Ok(ChatCompletionResponse {
        id,
        model,
        message: ChatMessage {
            provider_state: None,
            role: "assistant".to_string(),
            content,
            tool_calls,
            tool_call_id: None,
            name: None,
        },
        usage,
        raw,
        response_headers: Default::default(),
    })
}

// Tier detection is now done via the `copilot_internal/user` endpoint in
// `get_copilot_token()`, which returns the `copilot_plan` field directly.
