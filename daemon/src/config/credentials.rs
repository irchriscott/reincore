//! Credential store for persisting provider API keys and OAuth tokens.
//!
//! Credentials are stored in `~/.reincore-io/credentials.json` with file
//! permissions restricted to the current user (0600 on Unix). This allows
//! reincore to reuse credentials across launches without prompting the user
//! each time.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// The path to the credentials file.
fn credentials_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(super::local_config::CONFIG_DIR_NAME)
        .join("credentials.json")
}

/// A single stored credential.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredCredential {
    /// The credential value (API key, OAuth token, etc.).
    pub value: String,
    /// The type of credential.
    pub kind: CredentialKind,
    /// When the credential was stored (Unix timestamp).
    pub stored_at: i64,
}

/// The type of credential.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CredentialKind {
    /// A provider API key (e.g. OPENAI_API_KEY).
    ApiKey,
    /// A GitHub OAuth token (for Copilot).
    OAuthToken,
}

/// The credential store, mapping provider labels to credentials.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CredentialStore {
    pub credentials: HashMap<String, StoredCredential>,
}

impl CredentialStore {
    /// Load the credential store from disk. Returns an empty store if the
    /// file doesn't exist.
    pub fn load() -> Self {
        let path = credentials_path();
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Save the credential store to disk with restricted permissions.
    pub fn save(&self) -> Result<()> {
        let path = credentials_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, serde_json::to_string_pretty(self)?)?;

        // Restrict file permissions to the current user only.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&path)?.permissions();
            perms.set_mode(0o600);
            std::fs::set_permissions(&path, perms)?;
        }

        // On Windows, restrict access to the current user via icacls.
        #[cfg(target_os = "windows")]
        {
            let _ = std::process::Command::new("icacls")
                .arg(&path)
                .args([
                    "/inheritance:r",
                    "/grant:r",
                    &format!("{}:F", whoami_user()),
                ])
                .output();
        }

        Ok(())
    }

    /// Get a credential for a provider. Returns `None` if not stored.
    pub fn get(&self, provider: &str) -> Option<&StoredCredential> {
        self.credentials.get(&provider.to_lowercase())
    }

    /// Store a credential for a provider.
    pub fn set(&mut self, provider: &str, value: String, kind: CredentialKind) {
        self.credentials.insert(
            provider.to_lowercase(),
            StoredCredential {
                value,
                kind,
                stored_at: chrono::Utc::now().timestamp(),
            },
        );
    }

    /// Remove a credential for a provider. Returns `true` if a credential
    /// was removed.
    pub fn remove(&mut self, provider: &str) -> bool {
        self.credentials.remove(&provider.to_lowercase()).is_some()
    }

    /// Check if a credential exists for a provider.
    #[allow(dead_code)]
    pub fn has(&self, provider: &str) -> bool {
        self.credentials.contains_key(&provider.to_lowercase())
    }
}

/// Get the current username for Windows ACL commands.
#[cfg(target_os = "windows")]
fn whoami_user() -> String {
    std::env::var("USERNAME").unwrap_or_else(|_| "Everyone".to_string())
}

/// Store an API key for a provider.
pub fn store_api_key(provider: &str, key: &str) -> Result<()> {
    let mut store = CredentialStore::load();
    store.set(provider, key.to_string(), CredentialKind::ApiKey);
    store.save()
}

/// Store an OAuth token for a provider.
pub fn store_oauth_token(provider: &str, token: &str) -> Result<()> {
    let mut store = CredentialStore::load();
    store.set(provider, token.to_string(), CredentialKind::OAuthToken);
    store.save()
}

/// Get a stored credential for a provider.
pub fn get_credential(provider: &str) -> Option<StoredCredential> {
    CredentialStore::load().get(provider).cloned()
}

/// Remove a stored credential for a provider.
pub fn remove_credential(provider: &str) -> Result<bool> {
    let mut store = CredentialStore::load();
    let removed = store.remove(provider);
    if removed {
        store.save()?;
    }
    Ok(removed)
}
