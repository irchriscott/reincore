//! OAuth sign-ins to local MCP servers. Tokens are not configuration: they live
//! beside mcp.json in `<name>.credentials.json` (owner-only on Unix), keyed by
//! server name, and are used only while the server keeps the endpoint and
//! settings they were issued for. The daemon renews them in place; a lock file
//! serializes it with the desktop app so a rotated refresh token is never lost.
use anyhow::{Context, Result};
use async_trait::async_trait;
use engine_core::{
    mcp::{
        oauth, AuthConfig, AuthSecret, Credential, CredentialLease, CredentialStore, McpConfig,
        ServerConfig,
    },
    EngineError, EngineResult,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

const LOCK_WAIT: Duration = Duration::from_secs(15);
// A writer holds the lock for one token request at most; older locks were
// left by a process that exited mid-write.
const STALE_LOCK: Duration = Duration::from_secs(120);

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignIns {
    #[serde(default)]
    pub servers: BTreeMap<String, SignIn>,
}

/// A completed sign-in: the tokens and the settings they were issued for.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignIn {
    pub url: String,
    pub auth: AuthConfig,
    pub secret: AuthSecret,
    /// ReinCore registered the OAuth client itself during this sign-in.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub registered: bool,
}

impl SignIn {
    /// Tokens go only to the endpoint and settings they were issued for.
    pub fn matches(&self, server: &ServerConfig) -> bool {
        self.url == server.url && server.auth.as_ref() == Some(&self.auth)
    }
}

/// Where sign-ins for the MCP configuration at `config_path` are kept.
pub fn path_for(config_path: &Path) -> PathBuf {
    config_path.with_extension("credentials.json")
}

struct Lock(PathBuf);

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn try_lock(path: &Path) -> Result<Option<Lock>> {
    let mut lock = path.as_os_str().to_owned();
    lock.push(".lock");
    let lock = PathBuf::from(lock);
    match OpenOptions::new().write(true).create_new(true).open(&lock) {
        Ok(_) => Ok(Some(Lock(lock))),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let stale = fs::metadata(&lock)
                .and_then(|metadata| metadata.modified())
                .ok()
                .and_then(|modified| modified.elapsed().ok())
                .is_some_and(|age| age > STALE_LOCK);
            if stale {
                let _ = fs::remove_file(&lock);
            }
            Ok(None)
        }
        Err(error) => Err(error).context("cannot lock MCP sign-ins"),
    }
}

fn lock(path: &Path) -> Result<Lock> {
    let deadline = std::time::Instant::now() + LOCK_WAIT;
    loop {
        if let Some(lock) = try_lock(path)? {
            return Ok(lock);
        }
        anyhow::ensure!(
            std::time::Instant::now() < deadline,
            "MCP sign-ins are locked by another ReinCore process"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

async fn lock_async(path: &Path) -> Result<Lock> {
    let deadline = tokio::time::Instant::now() + LOCK_WAIT;
    loop {
        if let Some(lock) = try_lock(path)? {
            return Ok(lock);
        }
        anyhow::ensure!(
            tokio::time::Instant::now() < deadline,
            "MCP sign-ins are locked by another ReinCore process"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn read(path: &Path) -> Result<SignIns> {
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).with_context(|| {
            format!(
                "invalid MCP sign-in file {}; delete it, then sign in to its servers again",
                path.display()
            )
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(SignIns::default()),
        Err(error) => Err(error).context("cannot read MCP sign-ins"),
    }
}

fn write(path: &Path, sign_ins: &SignIns) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension(format!("{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| -> Result<()> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temp)?;
        file.write_all(&serde_json::to_vec_pretty(sign_ins)?)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

/// Every stored sign-in for the configuration at `config_path`.
pub fn load(config_path: &Path) -> Result<SignIns> {
    read(&path_for(config_path))
}

/// Stores a completed sign-in for `name`, replacing an earlier one.
pub fn save(config_path: &Path, name: &str, sign_in: SignIn) -> Result<()> {
    let path = path_for(config_path);
    let _lock = lock(&path)?;
    let mut sign_ins = read(&path)?;
    sign_ins.servers.insert(name.to_owned(), sign_in);
    write(&path, &sign_ins)
}

/// Drops sign-ins whose server was removed, or no longer has the endpoint or
/// settings they were issued for.
pub fn retain_matching(config_path: &Path, config: &McpConfig) -> Result<()> {
    let path = path_for(config_path);
    if !path.exists() {
        return Ok(());
    }
    let _lock = lock(&path)?;
    let mut sign_ins = read(&path)?;
    let before = sign_ins.servers.len();
    sign_ins.servers.retain(|name, sign_in| {
        config
            .servers
            .iter()
            .any(|server| &server.name == name && sign_in.matches(server))
    });
    if sign_ins.servers.len() != before {
        write(&path, &sign_ins)?;
    }
    Ok(())
}

/// The stored sign-in for `server`, renewed in place when used.
pub fn credential(
    config_path: &Path,
    server: &ServerConfig,
    sign_ins: &SignIns,
) -> Option<Credential> {
    let auth = server.auth.as_ref()?;
    stored(config_path, &server.name, &server.url, auth, sign_ins)
}

/// The sign-in stored for `name`, if it was issued for `url` and `auth`.
pub fn stored(
    config_path: &Path,
    name: &str,
    url: &str,
    auth: &AuthConfig,
    sign_ins: &SignIns,
) -> Option<Credential> {
    let sign_in = sign_ins
        .servers
        .get(name)
        .filter(|sign_in| sign_in.url == url && &sign_in.auth == auth)?;
    Some(Credential {
        secret: sign_in.secret.clone(),
        store: Some(Arc::new(FileStore {
            path: path_for(config_path),
            name: name.to_owned(),
            url: sign_in.url.clone(),
            auth: sign_in.auth.clone(),
        })),
    })
}

fn storage_error(error: anyhow::Error) -> EngineError {
    EngineError::ToolExecution(format!("MCP sign-in storage unavailable: {error}"))
}

pub struct FileStore {
    path: PathBuf,
    name: String,
    url: String,
    auth: AuthConfig,
}

struct FileLease {
    _lock: Lock,
    path: PathBuf,
    name: String,
    secret: AuthSecret,
}

#[async_trait]
impl CredentialStore for FileStore {
    async fn lease(&self) -> EngineResult<Box<dyn CredentialLease>> {
        let lock = lock_async(&self.path).await.map_err(storage_error)?;
        let sign_ins = read(&self.path).map_err(storage_error)?;
        // A sign-in replaced or removed since connecting is not this one.
        let secret = sign_ins
            .servers
            .get(&self.name)
            .filter(|sign_in| sign_in.url == self.url && sign_in.auth == self.auth)
            .map(|sign_in| sign_in.secret.clone())
            .ok_or_else(|| EngineError::ToolExecution(oauth::SIGN_IN_AGAIN.into()))?;
        Ok(Box::new(FileLease {
            _lock: lock,
            path: self.path.clone(),
            name: self.name.clone(),
            secret,
        }))
    }
}

#[async_trait]
impl CredentialLease for FileLease {
    fn secret(&self) -> &AuthSecret {
        &self.secret
    }

    async fn commit(self: Box<Self>, secret: AuthSecret) -> EngineResult<()> {
        let mut sign_ins = read(&self.path).map_err(storage_error)?;
        let sign_in = sign_ins
            .servers
            .get_mut(&self.name)
            .ok_or_else(|| EngineError::ToolExecution(oauth::SIGN_IN_AGAIN.into()))?;
        sign_in.secret = secret;
        write(&self.path, &sign_ins).map_err(storage_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn server(url: &str, auth: AuthConfig) -> ServerConfig {
        serde_json::from_value(json!({
            "name": "docs",
            "url": url,
            "timeout_ms": 1000,
            "auth": auth,
            "access": {
                "mode": "local",
                "project_ids": [uuid::Uuid::new_v4()]
            },
            "tools": {
                "read": {"read_only": true}
            }
        }))
        .unwrap()
    }

    fn oauth(client_id: &str) -> AuthConfig {
        AuthConfig::Oauth {
            client_id: client_id.into(),
            token_endpoint: "https://auth.example.com/token".into(),
            scopes: vec!["read".into()],
            resource: None,
            client_auth: Default::default(),
            issuer: None,
            managed_client: false,
        }
    }

    #[tokio::test]
    async fn sign_ins_renew_in_place_and_follow_their_server() {
        let root = std::env::temp_dir().join(format!("mcp-sign-ins-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let config_path = root.join("mcp.json");
        assert_eq!(path_for(&config_path), root.join("mcp.credentials.json"));
        let url = "https://mcp.example.com/mcp";
        let sign_in = SignIn {
            url: url.into(),
            auth: oauth("client"),
            secret: AuthSecret {
                access_token: Some("access-1".into()),
                refresh_token: Some("refresh-1".into()),
                ..AuthSecret::default()
            },
            registered: false,
        };
        save(&config_path, "docs", sign_in).unwrap();

        // Used only while the endpoint and settings are unchanged.
        let current = server(url, oauth("client"));
        let sign_ins = load(&config_path).unwrap();
        assert!(credential(
            &config_path,
            &server("https://other.example.com/mcp", oauth("client")),
            &sign_ins
        )
        .is_none());
        assert!(credential(&config_path, &server(url, oauth("another")), &sign_ins).is_none());
        let credential = credential(&config_path, &current, &sign_ins).unwrap();

        // Renewal is written back and serialized by the lock.
        let store = credential.store.unwrap();
        let lease = store.lease().await.unwrap();
        assert!(try_lock(&path_for(&config_path)).unwrap().is_none());
        let renewed = AuthSecret {
            access_token: Some("access-2".into()),
            refresh_token: Some("refresh-2".into()),
            ..AuthSecret::default()
        };
        lease.commit(renewed.clone()).await.unwrap();
        assert_eq!(load(&config_path).unwrap().servers["docs"].secret, renewed);
        drop(try_lock(&path_for(&config_path)).unwrap().unwrap());

        // Changing the server's settings drops the sign-in, after which a
        // connection made earlier is told to sign in again.
        let changed = McpConfig {
            servers: vec![server(url, oauth("another"))],
        };
        retain_matching(&config_path, &changed).unwrap();
        assert!(load(&config_path).unwrap().servers.is_empty());
        let error = match store.lease().await {
            Ok(_) => panic!("removed sign-in leased"),
            Err(error) => error,
        };
        assert!(engine_core::mcp::is_auth_rejected(&error));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(path_for(&config_path))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        fs::remove_dir_all(root).unwrap();
    }
}
