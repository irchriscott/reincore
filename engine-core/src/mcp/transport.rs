//! MCP transport utilities.
use super::{
    auth::{self, Authenticator, Credential},
    error, network, oauth, rejection_code, AuthRejection, ServerConfig,
};
use crate::EngineResult;
use futures::StreamExt;
use reqwest::{
    header::{HeaderMap, HeaderValue, CONTENT_TYPE, WWW_AUTHENTICATE},
    Client, StatusCode,
};
use serde_json::{json, Value};
use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

const VERSION: &str = "2025-06-18";

const LIMIT: usize = 4 * 1024 * 1024;

pub(super) fn initialize_request() -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "reincore",
                "version": env!("CARGO_PKG_VERSION")
            }
        }
    })
}

pub(super) struct Transport {
    client: Client,
    url: url::Url,
    headers: HeaderMap,
    auth: Authenticator,
    timeout: Duration,
    next_id: AtomicU64,
    metadata_changed: AtomicBool,
}

/// Endpoint and timeout bounds. Authentication is validated with the rest of
/// the server configuration.
pub(super) fn validate_endpoint(config: &ServerConfig) -> EngineResult<url::Url> {
    let url = url::Url::parse(&config.url).map_err(|_| error("invalid MCP endpoint"))?;
    let loopback = network::loopback(&url);

    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !(url.scheme() == "https"
            || (url.scheme() == "http" && loopback && config.allow_loopback_http))
    {
        return Err(error("MCP requires HTTPS or explicit loopback HTTP (127.0.0.1, ::1, or localhost); endpoint credentials/query forbidden"));
    }

    if !(100..=120_000).contains(&config.timeout_ms) {
        return Err(error("MCP timeout_ms must be 100..120000"));
    }

    Ok(url)
}

impl Transport {
    /// A supplied credential takes the place of the configuration's own
    /// credential sources.
    pub async fn connect(
        config: &ServerConfig,
        credential: Option<Credential>,
    ) -> EngineResult<Self> {
        let credential = match credential {
            Some(credential) => Some(credential),
            None => config.local_credential()?,
        };
        Self::connect_with(config, credential, false).await
    }

    pub async fn connect_cloud(
        config: &ServerConfig,
        credential: Option<Credential>,
    ) -> EngineResult<Self> {
        network::validate_cloud_url(&config.url)?;
        Self::connect_with(config, credential, true).await
    }

    pub(super) async fn connect_with(
        config: &ServerConfig,
        credential: Option<Credential>,
        public_only: bool,
    ) -> EngineResult<Self> {
        let url = validate_endpoint(config)?;
        let mut headers = auth::validate_headers(&config.headers)?;
        headers.insert(
            "accept",
            HeaderValue::from_static("application/json, text/event-stream"),
        );
        let auth = Authenticator::new(&config.effective_auth(), credential, public_only)?;

        let mut transport = Self {
            client: network::client(public_only)?,
            url,
            headers,
            auth,
            timeout: Duration::from_millis(config.timeout_ms),
            next_id: AtomicU64::new(1),
            metadata_changed: AtomicBool::new(false),
        };

        let (result, session) = tokio::time::timeout(
            transport.timeout,
            transport.exchange(initialize_request(), Some(0)),
        )
        .await
        .map_err(|_| error("MCP initialization timed out"))??;

        let version = match result.get("protocolVersion").and_then(Value::as_str) {
            Some("2025-03-26") => "2025-03-26",
            Some("2025-06-18") => "2025-06-18",
            _ => return Err(error("unsupported MCP protocol version")),
        };
        if !result
            .get("capabilities")
            .and_then(|c| c.get("tools"))
            .is_some_and(Value::is_object)
        {
            return Err(error("missing MCP tools capability"));
        }
        transport
            .headers
            .insert("mcp-protocol-version", HeaderValue::from_static(version));

        if let Some(mut session) = session {
            if session.as_bytes().is_empty()
                || session.as_bytes().len() > 1024
                || !session.as_bytes().iter().all(|b| (0x21..=0x7e).contains(b))
            {
                return Err(error("invalid MCP session header"));
            }
            session.set_sensitive(true);
            transport.headers.insert("mcp-session-id", session);
        }

        transport
            .notify("notifications/initialized", json!({}))
            .await?;
        Ok(transport)
    }

    pub fn metadata_current(&self) -> bool {
        !self.metadata_changed.load(Ordering::SeqCst)
    }

    async fn notify(&self, method: &str, params: Value) -> EngineResult<()> {
        tokio::time::timeout(
            Duration::from_secs(2),
            self.exchange(
                json!({
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params
                }),
                None,
            ),
        )
        .await
        .map_err(|_| error("MCP notification timed out"))??;
        Ok(())
    }

    pub async fn rpc(self: &Arc<Self>, method: &str, params: Value) -> EngineResult<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut cancellation = Cancellation {
            transport: self.clone(),
            id,
            armed: true,
        };
        let response = tokio::time::timeout(
            self.timeout,
            self.exchange(
                json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "method": method,
                    "params": params
                }),
                Some(id),
            ),
        )
        .await;
        match response {
            Ok(Ok((value, _))) => {
                cancellation.armed = false;
                Ok(value)
            }
            Ok(Err(e)) => Err(e),
            Err(_) => Err(error("MCP request timed out; outcome unknown, not retried")),
        }
    }

    async fn message(&self, value: Value, id: u64) -> EngineResult<Option<Value>> {
        if value.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            return Err(error("invalid MCP JSON-RPC version"));
        }
        if let Some(method) = value.get("method").and_then(Value::as_str) {
            if let Some(request_id) = value.get("id") {
                // Ping requires no optional capability. Unsupported server-initiated
                // capabilities receive a JSON-RPC error rather than being executed.
                let reply = if method == "ping" {
                    json!({ "jsonrpc": "2.0", "id": request_id, "result": {} })
                } else {
                    json!({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": "Client capability not supported"
                        }
                    })
                };

                let response = self
                    .send(&reply)
                    .await
                    .map_err(|_| error("MCP client response failed"))?;

                if response.status() != reqwest::StatusCode::ACCEPTED {
                    return Err(error("MCP client response rejected"));
                }
            } else if method == "notifications/tools/list_changed" {
                self.metadata_changed.store(true, Ordering::SeqCst);
            }
            return Ok(None);
        }
        rpc_result(value, id)
    }

    /// Sends one message with fresh authentication. A 401 means the server did
    /// not process the request, so it is sent once more when a renewed token
    /// is available.
    async fn send(&self, payload: &Value) -> EngineResult<reqwest::Response> {
        let body = serde_json::to_vec(payload).map_err(|_| error("invalid MCP request"))?;
        let mut retried = false;
        loop {
            let mut headers = self.headers.clone();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            let token = self.auth.apply(&self.url, &mut headers, &body).await?;
            let response = self
                .client
                .post(self.url.clone())
                .headers(headers)
                .body(body.clone())
                .send()
                .await
                .map_err(|_| error("MCP transport failed; outcome unknown, not retried"))?;
            if response.status() == StatusCode::UNAUTHORIZED
                && !retried
                && self.auth.renew(token.as_deref()).await
            {
                retried = true;
                continue;
            }
            return Ok(response);
        }
    }

    async fn exchange(
        &self,
        payload: Value,
        id: Option<u64>,
    ) -> EngineResult<(Value, Option<HeaderValue>)> {
        let response = self.send(&payload).await?;

        if id.is_none() {
            return if response.status() == reqwest::StatusCode::ACCEPTED {
                Ok((Value::Null, None))
            } else {
                Err(error("MCP notification rejected"))
            };
        }

        if response.status() == reqwest::StatusCode::NOT_FOUND
            && self.headers.contains_key("mcp-session-id")
        {
            self.metadata_changed.store(true, Ordering::SeqCst);
        }
        if matches!(
            response.status(),
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN
        ) {
            let rejection = rejection(response).await;
            tracing::warn!(
                status = rejection.status,
                error = rejection.error.as_deref().unwrap_or(""),
                reason = rejection.reason.as_deref().unwrap_or(""),
                "MCP server refused the credentials"
            );
            return Err(error(&rejection.message()));
        }
        if !response.status().is_success() {
            return Err(error(
                "MCP HTTP request rejected; credentials and response redacted; no retry",
            ));
        }

        let session = response.headers().get("mcp-session-id").cloned();
        let mime = response
            .headers()
            .get("content-type")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("");
        let sse = match mime {
            "text/event-stream" => true,
            "application/json" => false,
            _ => return Err(error("unsupported MCP response content type")),
        };

        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();
        let mut total = 0;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|_| error("MCP response interrupted; outcome unknown"))?;
            total += chunk.len();
            if total > LIMIT {
                return Err(error("MCP response exceeds 4 MiB"));
            }
            buffer.extend_from_slice(&chunk);
            if sse {
                // Parse complete lines as bytes so chunk boundaries cannot split UTF-8.
                while let Some((end, delimiter)) = frame_end(&buffer) {
                    let frame = std::str::from_utf8(&buffer[..end])
                        .map_err(|_| error("invalid MCP SSE UTF-8"))?;
                    let normalized = frame.replace("\r\n", "\n").replace('\r', "\n");
                    let data = normalized
                        .lines()
                        .filter_map(|l| {
                            l.strip_prefix("data:")
                                .map(|s| s.strip_prefix(' ').unwrap_or(s))
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !data.is_empty() {
                        let value: Value = serde_json::from_str(&data)
                            .map_err(|_| error("invalid MCP SSE JSON"))?;
                        if let Some(result) = self.message(value, id.unwrap()).await? {
                            return Ok((result, session));
                        }
                    }
                    buffer.drain(..end + delimiter);
                }
            }
        }

        if sse {
            return Err(error("MCP stream ended without a matching result"));
        }
        let value: Value =
            serde_json::from_slice(&buffer).map_err(|_| error("invalid MCP response JSON"))?;
        let result = self
            .message(value, id.unwrap())
            .await?
            .ok_or_else(|| error("MCP response ID mismatch"))?;
        Ok((result, session))
    }
}

fn frame_end(bytes: &[u8]) -> Option<(usize, usize)> {
    let mut previous = None;
    let mut i = 0;
    while i < bytes.len() {
        let width = match bytes[i] {
            b'\r' if bytes.get(i + 1) == Some(&b'\n') => 2,
            b'\r' | b'\n' => 1,
            _ => {
                previous = None;
                i += 1;
                continue;
            }
        };
        if let Some(start) = previous {
            return Some((start, i + width - start));
        }
        previous = Some(i);
        i += width;
    }
    None
}

/// The codes explaining a 401 or 403: the Bearer challenge's `error`, and the
/// reason of a Google API error body such as
/// `{"error": {"status": "PERMISSION_DENIED", "details": [{"reason": "SERVICE_DISABLED"}]}}`.
async fn rejection(response: reqwest::Response) -> AuthRejection {
    const BODY_LIMIT: usize = 65536;
    let status = response.status().as_u16();
    let challenge = response
        .headers()
        .get_all(WWW_AUTHENTICATE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .find_map(oauth::bearer_error)
        .and_then(|error| rejection_code(&error));
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(Ok(chunk)) = stream.next().await {
        if body.len() + chunk.len() > BODY_LIMIT {
            break;
        }
        body.extend_from_slice(&chunk);
    }
    let reason = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .filter(|error| error.is_object())
                .cloned()
        })
        .and_then(|error| {
            let listed = |field: &str| {
                error
                    .get(field)
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .find_map(|item| item.get("reason").and_then(Value::as_str))
                    .map(str::to_owned)
            };
            listed("details").or_else(|| listed("errors")).or_else(|| {
                error
                    .get("status")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
        })
        .and_then(|reason| rejection_code(&reason));
    AuthRejection {
        status,
        error: challenge,
        reason,
    }
}

fn rpc_result(value: Value, id: u64) -> EngineResult<Option<Value>> {
    if value.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err(error("invalid MCP JSON-RPC version"));
    }

    if value.get("method").is_some() {
        if value.get("id").is_some() {
            return Err(error("MCP server requested unsupported client capability"));
        }
        return Ok(None);
    }

    if value.get("id").and_then(Value::as_u64) != Some(id) {
        return Err(error("MCP response ID mismatch"));
    }

    if value.get("error").is_some() {
        return Err(error(
            "MCP RPC error; upstream details redacted; not retried",
        ));
    }

    value
        .get("result")
        .cloned()
        .map(Some)
        .ok_or_else(|| error("missing MCP result"))
}

struct Cancellation {
    transport: Arc<Transport>,
    id: u64,
    armed: bool,
}

impl Drop for Cancellation {
    fn drop(&mut self) {
        if self.armed {
            if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                let transport = self.transport.clone();
                let id = self.id;
                runtime.spawn(async move {
                    let _ = transport
                        .notify(
                            "notifications/cancelled",
                            json!({
                                "requestId": id,
                                "reason": "ReinCore request cancelled"
                            }),
                        )
                        .await;
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::frame_end;

    #[test]
    fn sse_frames_accept_all_line_endings_and_incomplete_chunks() {
        for delimiter in ["\n\n", "\r\r", "\r\n\r\n", "\n\r\n", "\r\n\n"] {
            let frame = format!("data: {{}}{delimiter}next");
            assert_eq!(frame_end(frame.as_bytes()), Some((8, delimiter.len())));
        }
        assert_eq!(frame_end(b"data: {}\r\n"), None);
        assert_eq!(frame_end(b"data: {}\r"), None);
        assert_eq!(frame_end(b"data: {}"), None);
    }
    #[tokio::test]
    async fn resolved_credentials_are_sent_without_environment_lookup() {
        let fixture = mcp_mock::McpFixture::start(false).await;
        let mut config: super::ServerConfig =
            serde_json::from_value(fixture.config("")["servers"][0].clone()).unwrap();
        config.bearer_token_env = Some("THIS_ENV_IS_NEVER_READ_BY_TENANT_TRANSPORT".into());
        let credential = || {
            Some(super::Credential::new(
                super::auth::AuthSecret::from_secret("tenant-only-token"),
            ))
        };
        super::Transport::connect_with(&config, credential(), false)
            .await
            .unwrap();
        let messages = fixture.messages.lock().unwrap();
        assert!(messages
            .iter()
            .all(|v| v["authorization"] == "Bearer tenant-only-token"));
        drop(messages);
        assert!(
            super::Transport::connect_cloud(&config, credential())
                .await
                .is_err(),
            "public-only cloud transport must deny this loopback fixture"
        );
    }
}
