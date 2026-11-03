//! OAuth sign-in to local MCP servers, following RFC 8252 for native apps: the
//! system browser signs in and returns to a one-time listener on 127.0.0.1.
//! Shared by the desktop app and `reincore mcp sign-in`; the result is stored
//! beside mcp.json (see [`crate::mcp_credentials`]).
use crate::mcp_credentials::SignIn;
use engine_core::{
    mcp::{oauth, AuthConfig},
    EngineError,
};
use std::{collections::HashMap, future::Future, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

/// How long the browser has to come back.
pub const SIGN_IN_WAIT: Duration = Duration::from_secs(300);
pub const CANCELLED: &str = "Sign-in cancelled.";

pub struct SignInRequest {
    pub url: String,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub scopes: Option<Vec<String>>,
    pub timeout_ms: u64,
}

/// An engine failure as a person reads it, without the error category.
pub fn message(error: EngineError) -> String {
    match error {
        EngineError::ToolExecution(message) => message,
        other => other.to_string(),
    }
}

/// The client to sign in with, given an earlier sign-in to the same server.
/// A typed secret is used as given. Signing in again with the earlier client
/// reuses its stored secret, except for a client ReinCore registered itself:
/// its redirect URI names the loopback port of its first sign-in, which
/// authorization servers may match exactly, so a new client is registered.
pub fn client_for(
    earlier: Option<&SignIn>,
    client_id: Option<String>,
    client_secret: Option<String>,
) -> (Option<String>, Option<String>) {
    let client_id = client_id.filter(|id| !id.is_empty());
    let client_secret = client_secret.filter(|secret| !secret.is_empty());
    if client_secret.is_some() {
        return (client_id, client_secret);
    }
    let earlier = earlier.filter(|earlier| {
        matches!(&earlier.auth, AuthConfig::Oauth { client_id: id, .. } if Some(id) == client_id.as_ref())
    });
    match earlier {
        Some(earlier) if earlier.registered => (None, None),
        Some(earlier) => (client_id, earlier.secret.secret.clone()),
        None => (client_id, None),
    }
}

/// What one request to the loopback listener means for the sign-in.
#[derive(Debug, PartialEq)]
enum Redirect {
    Code(String),
    Refused(String),
    // Not the redirect: the browser's favicon request, or a stray visitor.
    Ignored,
}

fn redirect_outcome(request: &str, state: &str) -> Redirect {
    let Some(target) = request
        .lines()
        .next()
        .and_then(|line| line.strip_prefix("GET "))
        .and_then(|rest| rest.split(' ').next())
    else {
        return Redirect::Ignored;
    };
    let Ok(url) = reqwest::Url::parse(&format!("http://127.0.0.1{target}")) else {
        return Redirect::Ignored;
    };
    let params: HashMap<String, String> = url.query_pairs().into_owned().collect();
    if url.path() != "/" || params.get("state").map(String::as_str) != Some(state) {
        return Redirect::Ignored;
    }
    if let Some(error) = params.get("error") {
        return Redirect::Refused(params.get("error_description").cloned().unwrap_or_else(|| {
            match error.as_str() {
                "access_denied" => "Sign-in was cancelled at the provider.".into(),
                other => format!("The provider refused the sign-in ({other})."),
            }
        }));
    }
    match params.get("code") {
        Some(code) if !code.is_empty() => Redirect::Code(code.clone()),
        _ => Redirect::Refused("The provider did not return a sign-in code.".into()),
    }
}

fn page(status: &str, heading: &str, detail: &str) -> Vec<u8> {
    let escape = |text: &str| {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    };
    let body = format!(
        "<!doctype html><meta charset=\"utf-8\"><title>ReinCore</title>\
         <body style=\"font-family:system-ui,sans-serif;display:grid;place-items:center;min-height:90vh;color:#e5e5e5;background:#111\">\
         <div style=\"text-align:center;max-width:28rem\"><h1 style=\"font-size:1.25rem\">{}</h1><p style=\"color:#a3a3a3\">{}</p></div>",
        escape(heading),
        escape(detail)
    );
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

async fn wait_for_code(listener: &TcpListener, state: &str) -> Result<String, String> {
    loop {
        let (mut socket, _) = listener
            .accept()
            .await
            .map_err(|_| "The sign-in listener stopped.".to_string())?;
        let mut buffer = vec![0; 16384];
        let read =
            match tokio::time::timeout(Duration::from_secs(5), socket.read(&mut buffer)).await {
                Ok(Ok(read)) => read,
                _ => continue,
            };
        let request = String::from_utf8_lossy(&buffer[..read]);
        let outcome = redirect_outcome(&request, state);
        let response = match &outcome {
            Redirect::Code(_) => page(
                "200 OK",
                "Signed in",
                "You can close this tab and return to ReinCore.",
            ),
            Redirect::Refused(reason) => page("200 OK", "Sign-in was not completed", reason),
            Redirect::Ignored => page("404 Not Found", "Not found", ""),
        };
        let _ = socket.write_all(&response).await;
        let _ = socket.shutdown().await;
        match outcome {
            Redirect::Code(code) => return Ok(code),
            Redirect::Refused(reason) => return Err(reason),
            Redirect::Ignored => continue,
        }
    }
}

/// Signs in to the MCP server at `request.url`: `open` sends the browser to
/// the authorization URL, and the code it brings back is exchanged for
/// tokens. Resolving `cancelled` abandons the sign-in at any step.
pub async fn sign_in(
    request: SignInRequest,
    open: impl FnOnce(&str) -> Result<(), String>,
    cancelled: impl Future<Output = ()>,
) -> Result<SignIn, String> {
    let flow = async {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .map_err(|error| format!("Could not start the sign-in listener: {error}"))?;
        let port = listener
            .local_addr()
            .map_err(|error| error.to_string())?
            .port();
        let registered = request.client_id.is_none();
        let started = oauth::start(
            oauth::AuthorizationRequest {
                url: request.url.clone(),
                redirect_uri: format!("http://127.0.0.1:{port}/"),
                client_id: request.client_id,
                client_secret: request.client_secret,
                scopes: request.scopes,
                timeout_ms: request.timeout_ms,
                managed_client: false,
            },
            false,
        )
        .await
        .map_err(message)?;
        open(&started.authorization_url)?;
        let code = tokio::select! {
            code = wait_for_code(&listener, &started.state) => code?,
            _ = tokio::time::sleep(SIGN_IN_WAIT) => return Err("Sign-in timed out. Sign in again.".to_string()),
        };
        drop(listener);
        let (auth, secret) = oauth::complete(
            &started.pending,
            &started.secret,
            &code,
            false,
            Duration::from_secs(30),
        )
        .await
        .map_err(message)?;
        Ok(SignIn {
            url: request.url,
            auth,
            secret,
            registered,
        })
    };
    tokio::select! {
        biased;
        _ = cancelled => Err(CANCELLED.into()),
        result = flow => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::mcp::AuthSecret;

    #[test]
    fn only_the_matching_redirect_completes_a_sign_in() {
        let request = |target: &str| format!("GET {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        assert_eq!(
            redirect_outcome(&request("/?code=abc&state=s1"), "s1"),
            Redirect::Code("abc".into())
        );
        assert_eq!(
            redirect_outcome(&request("/favicon.ico"), "s1"),
            Redirect::Ignored
        );
        assert_eq!(
            redirect_outcome(&request("/?code=abc&state=other"), "s1"),
            Redirect::Ignored
        );
        assert_eq!(
            redirect_outcome("POST /?code=abc&state=s1 HTTP/1.1", "s1"),
            Redirect::Ignored
        );
        assert_eq!(
            redirect_outcome(&request("/?error=access_denied&state=s1"), "s1"),
            Redirect::Refused("Sign-in was cancelled at the provider.".into())
        );
        assert_eq!(
            redirect_outcome(&request("/?state=s1"), "s1"),
            Redirect::Refused("The provider did not return a sign-in code.".into())
        );
        let html =
            String::from_utf8(page("200 OK", "Sign-in was not completed", "<script>")).unwrap();
        assert!(html.contains("&lt;script&gt;") && !html.contains("<script>"));
    }

    #[tokio::test]
    async fn the_listener_waits_past_unrelated_requests_for_the_redirect() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let browser = tokio::spawn(async move {
            let mut responses = Vec::new();
            for target in [
                "/favicon.ico",
                "/?code=wrong&state=forged",
                "/?code=abc&state=s1",
            ] {
                let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
                stream
                    .write_all(
                        format!("GET {target} HTTP/1.1\r\nHost: {address}\r\n\r\n").as_bytes(),
                    )
                    .await
                    .unwrap();
                let mut response = String::new();
                stream.read_to_string(&mut response).await.unwrap();
                responses.push(response);
            }
            responses
        });
        assert_eq!(wait_for_code(&listener, "s1").await.unwrap(), "abc");
        let responses = browser.await.unwrap();
        assert!(responses[0].starts_with("HTTP/1.1 404"));
        assert!(responses[1].starts_with("HTTP/1.1 404"));
        assert!(responses[2].starts_with("HTTP/1.1 200") && responses[2].contains("Signed in"));
    }

    #[test]
    fn signing_in_again_reuses_only_clients_the_administrator_created() {
        let earlier = |registered: bool| SignIn {
            url: "https://mcp.example.com/mcp".into(),
            auth: AuthConfig::Oauth {
                client_id: "client".into(),
                token_endpoint: "https://auth.example.com/token".into(),
                scopes: vec![],
                resource: None,
                client_auth: Default::default(),
                issuer: None,
                managed_client: false,
            },
            secret: AuthSecret::from_secret("stored-secret"),
            registered,
        };
        let own = earlier(false);
        assert_eq!(
            client_for(Some(&own), Some("client".into()), None),
            (Some("client".into()), Some("stored-secret".into()))
        );
        assert_eq!(
            client_for(Some(&own), Some("client".into()), Some("typed".into())),
            (Some("client".into()), Some("typed".into()))
        );
        assert_eq!(
            client_for(Some(&own), Some("other".into()), None),
            (Some("other".into()), None)
        );
        assert_eq!(
            client_for(Some(&earlier(true)), Some("client".into()), None),
            (None, None)
        );
        assert_eq!(
            client_for(None, Some(String::new()), Some(String::new())),
            (None, None)
        );
    }

    #[tokio::test]
    async fn a_loopback_server_signs_in_and_saves() {
        use axum::{
            http::StatusCode,
            routing::{get, post},
            Form, Json, Router,
        };
        use serde_json::json;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let issuer = base.clone();
        let app = Router::new()
            .route("/mcp", post(|| async { StatusCode::UNAUTHORIZED }))
            .route(
                "/.well-known/oauth-authorization-server",
                get(move || async move {
                    Json(json!({
                        "issuer": issuer,
                        "authorization_endpoint": format!("{issuer}/authorize"),
                        "token_endpoint": format!("{issuer}/token"),
                        "code_challenge_methods_supported": ["S256"]
                    }))
                }),
            )
            .route(
                "/token",
                post(|Form(form): Form<HashMap<String, String>>| async move {
                    assert_eq!(form["code"], "code-1");
                    assert_eq!(form["client_id"], "client");
                    Json(json!({"access_token": "access", "refresh_token": "refresh",
                        "token_type": "Bearer", "expires_in": 3600}))
                }),
            );
        let upstream = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        // The browser signs in and follows the redirect back to the listener.
        let browser = |authorization: &str| {
            let authorization = reqwest::Url::parse(authorization).unwrap();
            let query: HashMap<String, String> = authorization.query_pairs().into_owned().collect();
            assert!(!query.contains_key("prompt"));
            tokio::spawn(async move {
                let redirect = format!(
                    "{}?code=code-1&state={}",
                    query["redirect_uri"], query["state"]
                );
                let client = reqwest::Client::builder().no_proxy().build().unwrap();
                client.get(redirect).send().await.unwrap();
            });
            Ok(())
        };
        let request = SignInRequest {
            url: format!("{base}/mcp"),
            client_id: Some("client".into()),
            client_secret: None,
            scopes: None,
            timeout_ms: 5000,
        };
        let signed_in = sign_in(request, browser, std::future::pending())
            .await
            .unwrap();
        assert!(!signed_in.registered);
        assert_eq!(signed_in.secret.refresh_token.as_deref(), Some("refresh"));

        // Its settings, including the loopback resource, are valid to save.
        let root = std::env::temp_dir().join(format!("mcp-sign-in-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("mcp.json");
        std::fs::write(&path, r#"{"servers":[]}"#).unwrap();
        let server = serde_json::from_value(json!({
            "name": "local",
            "url": format!("{base}/mcp"),
            "allow_loopback_http": true,
            "timeout_ms": 5000,
            "auth": {"type": "none"},
            "access": {"mode": "local", "project_ids": [uuid::Uuid::new_v4()]},
            "tools": {"read": {"read_only": true}}
        }))
        .unwrap();
        let config =
            crate::mcp_config::save_signed_in(&path, server, None, Some(signed_in)).unwrap();
        assert!(matches!(
            &config.servers[0].auth,
            Some(AuthConfig::Oauth { resource: Some(resource), .. }) if *resource == format!("{base}/mcp")
        ));
        assert!(crate::mcp_credentials::load(&path)
            .unwrap()
            .servers
            .contains_key("local"));
        std::fs::remove_dir_all(root).unwrap();
        upstream.abort();
    }

    #[tokio::test]
    async fn cancelling_abandons_the_sign_in() {
        let request = SignInRequest {
            // Nothing listens on port 9: the flow would fail, but cancelling wins.
            url: "http://127.0.0.1:9/mcp".into(),
            client_id: None,
            client_secret: None,
            scopes: None,
            timeout_ms: 60_000,
        };
        let result = sign_in(request, |_| Ok(()), async {}).await;
        assert_eq!(result.err().as_deref(), Some(CANCELLED));
    }
}
