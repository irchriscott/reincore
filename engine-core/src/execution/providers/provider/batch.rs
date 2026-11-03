//! Shared provider-backed batch lifecycle and wire types.
//!
//! Validates requests and adapts submission, status, cancellation, and result
//! retrieval for OpenAI, Anthropic, and Gemini. Jobs remain at the provider;
//! result normalization preserves caller IDs, provider JSON, and item failures.
use super::Provider;
use crate::{ChatCompletionRequest, EngineError, EngineResult};
use reqwest::{Client, Method};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BatchRequest {
    pub custom_id: String,
    #[serde(deserialize_with = "deserialize_batch_body")]
    pub body: ChatCompletionRequest,
}

// Batch callers omit the model; provider policy resolves it before submission.
fn deserialize_batch_body<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<ChatCompletionRequest, D::Error> {
    let mut body = Value::deserialize(deserializer)?;
    if let Some(object) = body.as_object_mut() {
        object.insert("model".into(), Value::String("reincore-auto".into()));
    }
    serde_json::from_value(body).map_err(serde::de::Error::custom)
}

/// Resolve the active provider, requiring an explicit header in multi-provider mode.
pub fn select_provider(
    registry: &crate::ProviderRegistry,
    label: Option<&str>,
) -> BatchResultValue<std::sync::Arc<dyn Provider>> {
    if registry.is_empty() {
        return Err(BatchError::new(503, "no provider is configured"));
    }
    if let Some(label) = label.map(str::trim).filter(|label| !label.is_empty()) {
        return registry
            .get_by_label(label)
            .ok_or_else(|| BatchError::new(400, "requested provider is not configured"));
    }
    if registry.is_multi_provider() {
        return Err(BatchError::new(
            400,
            "X-ReinCore-Provider is required in multi-provider mode",
        ));
    }
    Ok(registry.iter().next().expect("nonempty registry").clone())
}

#[derive(Debug, Deserialize)]
pub struct CreateBatchRequest {
    pub session_id: Option<uuid::Uuid>,
    /// Reuse this engine job ID after a timeout; never submit a new ID blindly.
    pub idempotency_key: Option<uuid::Uuid>,
    pub requests: Vec<BatchRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Batch {
    pub id: String,
    pub provider: String,
    pub status: String,
    pub request_counts: Value,
    pub raw: Value,
}

impl Batch {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status.as_str(),
            "completed" | "partial_failed" | "failed" | "cancelled" | "expired"
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    pub custom_id: String,
    pub response: Option<Value>,
    /// OpenAI-compatible completion for SDK agent orchestration; raw response is retained.
    #[serde(default)]
    pub completion: Option<Value>,
    pub error: Option<Value>,
    pub raw: Value,
}

#[derive(Debug)]
pub struct BatchError {
    pub status: u16,
    pub message: String,
}

impl BatchError {
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

pub type BatchResultValue<T> = Result<T, BatchError>;

#[derive(Clone, Copy)]
pub enum BatchProtocol {
    OpenAi,
    Anthropic,
    Gemini,
}

pub struct BatchClient {
    client: Client,
    base_url: String,
    headers: HashMap<String, String>,
    protocol: BatchProtocol,
    provider: String,
}

impl BatchClient {
    /// Opaque credential/endpoint identity; credentials never enter job records.
    pub fn scope(&self) -> String {
        let headers: std::collections::BTreeMap<_, _> = self.headers.iter().collect();
        crate::CacheKey::for_tool_call(
            "batch-provider-scope-v1",
            &json!({"base_url":self.base_url,"provider":self.provider,"headers":headers}),
        )
        .expect("serializable scope")
        .0
    }

    pub fn usage(&self, result: &BatchResult) -> Option<super::TokenUsage> {
        let usage = result.response.as_ref()?.get("usage")?;
        match self.protocol {
            BatchProtocol::Anthropic => {
                usage["input_tokens"]
                    .as_u64()?
                    .checked_add(usage["cache_read_input_tokens"].as_u64().unwrap_or(0))?
                    .checked_add(usage["cache_creation_input_tokens"].as_u64().unwrap_or(0))?
                    .checked_add(usage["output_tokens"].as_u64()?)?;
                super::anthropic::parse_usage(usage)
            }
            _ => super::provider::parse_openai_stream_usage(usage),
        }
    }
    pub fn new(
        client: Client,
        base_url: String,
        headers: HashMap<String, String>,
        protocol: BatchProtocol,
        provider: String,
    ) -> Self {
        Self {
            client,
            base_url,
            headers,
            protocol,
            provider,
        }
    }

    fn collection(&self) -> &'static str {
        match self.protocol {
            BatchProtocol::OpenAi | BatchProtocol::Gemini => "batches",
            BatchProtocol::Anthropic => "v1/messages/batches",
        }
    }

    fn check_privacy(&self) -> BatchResultValue<()> {
        crate::privacy::current_policy()
            .check_provider(&self.provider)
            .map_err(|_| BatchError::new(403, "provider is not approved by the privacy policy"))
    }

    // IDs are data, never URL paths. This also prevents absolute-URL/SSRF input.
    fn url(
        &self,
        collection: &str,
        id: Option<&str>,
        action: Option<&str>,
    ) -> BatchResultValue<reqwest::Url> {
        let mut url = reqwest::Url::parse(&format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            collection
        ))
        .map_err(|_| BatchError::new(500, "invalid provider base URL"))?;
        if let Some(id) = id {
            let id = if matches!(self.protocol, BatchProtocol::Gemini) && collection == "batches" {
                id.strip_prefix("batches/").unwrap_or(id)
            } else {
                id
            };
            if id.is_empty() || id == "." || id == ".." {
                return Err(BatchError::new(400, "invalid batch or file ID"));
            }
            let mut path = url
                .path_segments_mut()
                .map_err(|_| BatchError::new(500, "invalid provider URL"))?;
            path.push(id);
            if let Some(action) = action {
                path.push(action);
            }
        }
        Ok(url)
    }

    fn request(&self, method: Method, url: reqwest::Url) -> reqwest::RequestBuilder {
        let mut request = self.client.request(method, url);
        for (key, value) in &self.headers {
            request = request.header(key, value);
        }
        request
    }

    async fn send(request: reqwest::RequestBuilder) -> BatchResultValue<reqwest::Response> {
        let response = request.send().await.map_err(|e| {
            BatchError::new(
                if e.is_timeout() { 504 } else { 502 },
                "batch provider request failed",
            )
        })?;
        if !response.status().is_success() {
            let status = response.status().as_u16();
            // Do not expose upstream bodies, which may contain credentials or submitted prompts.
            return Err(BatchError::new(
                status,
                format!("batch provider returned HTTP {status}"),
            ));
        }
        Ok(response)
    }

    async fn json(request: reqwest::RequestBuilder) -> BatchResultValue<Value> {
        Self::send(request)
            .await?
            .json()
            .await
            .map_err(|_| BatchError::new(502, "invalid batch provider JSON"))
    }

    fn gemini_url(
        &self,
        collection: &str,
        id: Option<&str>,
        action: Option<&str>,
        upload: bool,
    ) -> BatchResultValue<reqwest::Url> {
        let mut url = reqwest::Url::parse(&self.base_url)
            .map_err(|_| BatchError::new(500, "invalid Gemini base URL"))?;
        let version = url
            .path()
            .trim_end_matches('/')
            .strip_suffix("/openai")
            .ok_or_else(|| BatchError::new(400, "Gemini batch base URL must end in /openai"))?
            .to_string();
        url.set_path(&format!(
            "{}{version}/{collection}",
            if upload {
                "/upload"
            } else if action == Some("download") {
                "/download"
            } else {
                ""
            }
        ));
        if let Some(id) = id {
            let id = id.strip_prefix(&format!("{collection}/")).unwrap_or(id);
            if id.is_empty() || id == "." || id == ".." || id.contains('/') {
                return Err(BatchError::new(400, "invalid Gemini resource ID"));
            }
            let value = format!(
                "{id}{}",
                action.map(|s| format!(":{s}")).unwrap_or_default()
            );
            url.path_segments_mut()
                .map_err(|_| BatchError::new(500, "invalid Gemini URL"))?
                .push(&value);
        }
        Ok(url)
    }

    fn gemini_request(
        &self,
        method: Method,
        url: reqwest::Url,
    ) -> BatchResultValue<reqwest::RequestBuilder> {
        let key = self
            .headers
            .get("Authorization")
            .and_then(|h| h.strip_prefix("Bearer "))
            .ok_or_else(|| BatchError::new(401, "Gemini API key is missing"))?;
        Ok(self
            .client
            .request(method, url)
            .header("x-goog-api-key", key))
    }

    async fn gemini_upload(&self, input: String, display_name: &str) -> BatchResultValue<String> {
        let origin = self.gemini_url("files", None, None, true)?;
        let start = self
            .gemini_request(Method::POST, origin.clone())?
            .header("X-Goog-Upload-Protocol", "resumable")
            .header("X-Goog-Upload-Command", "start")
            .header("X-Goog-Upload-Header-Content-Length", input.len())
            .header("X-Goog-Upload-Header-Content-Type", "application/jsonl")
            .json(&json!({"file":{"display_name":display_name}}));
        let response = Self::send(start).await?;
        let upload_url = response
            .headers()
            .get("x-goog-upload-url")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| reqwest::Url::parse(v).ok())
            .ok_or_else(|| BatchError::new(502, "missing Gemini upload URL"))?;
        if upload_url.origin() != origin.origin() {
            return Err(BatchError::new(502, "Gemini upload URL changed origin"));
        }
        let raw = Self::json(
            self.gemini_request(Method::POST, upload_url)?
                .header("X-Goog-Upload-Offset", "0")
                .header("X-Goog-Upload-Command", "upload, finalize")
                .body(input),
        )
        .await?;
        raw["file"]["name"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| BatchError::new(502, "missing Gemini input file name"))
    }

    fn normalize(&self, raw: Value) -> BatchResultValue<Batch> {
        let id = raw["id"]
            .as_str()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| BatchError::new(502, "missing batch ID"))?
            .to_string();
        let status = match self.protocol {
            BatchProtocol::OpenAi | BatchProtocol::Gemini => raw["status"].as_str(),
            BatchProtocol::Anthropic => raw["processing_status"].as_str(),
        }
        .ok_or_else(|| BatchError::new(502, "missing batch status"))?;
        let status = match status {
            "in_progress" => "in_progress",
            "canceling" => "cancelling",
            "ended" => "completed",
            "canceled" => "cancelled",
            s => s,
        }
        .to_string();
        Ok(Batch {
            id,
            provider: self.provider.clone(),
            status,
            request_counts: raw["request_counts"].clone(),
            raw,
        })
    }

    pub async fn create(&self, requests: Vec<(String, Value)>) -> BatchResultValue<Batch> {
        self.create_tagged(requests, "reincore-batch").await
    }

    pub async fn create_tagged(
        &self,
        requests: Vec<(String, Value)>,
        tag: &str,
    ) -> BatchResultValue<Batch> {
        self.check_privacy()?;
        let body = match self.protocol {
            BatchProtocol::Anthropic => {
                json!({"requests": requests.into_iter().map(|(id, params)| json!({"custom_id": id, "params": params})).collect::<Vec<_>>()})
            }
            BatchProtocol::OpenAi | BatchProtocol::Gemini => {
                let input = requests
                    .into_iter()
                    .map(|(id, body)| super::openai::build_batch_jsonl_line(&id, body))
                    .collect::<Vec<_>>()
                    .join("\n")
                    + "\n";
                if matches!(self.protocol, BatchProtocol::Gemini) {
                    let file_id = self.gemini_upload(input, tag).await?;
                    json!({"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"})
                } else {
                    let upload = self
                        .request(Method::POST, self.url("files", None, None)?)
                        .multipart(
                            reqwest::multipart::Form::new()
                                .text("purpose", "batch")
                                .part(
                                    "file",
                                    reqwest::multipart::Part::text(input)
                                        .file_name("reincore-batch.jsonl"),
                                ),
                        );
                    let uploaded = Self::json(upload).await?;
                    let file_id = uploaded["id"]
                        .as_str()
                        .ok_or_else(|| BatchError::new(502, "missing batch input file ID"))?;
                    json!({"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"})
                }
            }
        };
        self.normalize(
            Self::json(
                self.request(Method::POST, self.url(self.collection(), None, None)?)
                    .json(&body),
            )
            .await?,
        )
    }

    pub async fn get(&self, id: &str) -> BatchResultValue<Batch> {
        self.check_privacy()?;
        self.normalize(
            Self::json(self.request(Method::GET, self.url(self.collection(), Some(id), None)?))
                .await?,
        )
    }

    pub async fn cancel(&self, id: &str) -> BatchResultValue<Batch> {
        self.check_privacy()?;
        if matches!(self.protocol, BatchProtocol::Gemini) {
            Self::send(self.gemini_request(
                Method::POST,
                self.gemini_url("batches", Some(id), Some("cancel"), false)?,
            )?)
            .await?;
            return self.get(id).await;
        }
        self.normalize(
            Self::json(self.request(
                Method::POST,
                self.url(self.collection(), Some(id), Some("cancel"))?,
            ))
            .await?,
        )
    }

    pub async fn results(&self, id: &str) -> BatchResultValue<Vec<BatchResult>> {
        self.check_privacy()?;
        let batch = self.get(id).await?;
        if !batch.is_terminal() {
            return Err(BatchError::new(409, "batch results are not ready"));
        }
        let mut urls = Vec::new();
        match self.protocol {
            BatchProtocol::Anthropic => {
                urls.push(self.url(self.collection(), Some(id), Some("results"))?)
            }
            BatchProtocol::OpenAi | BatchProtocol::Gemini => {
                for field in ["output_file_id", "error_file_id"] {
                    if let Some(file) = batch.raw[field].as_str() {
                        if matches!(self.protocol, BatchProtocol::Gemini) {
                            let mut url =
                                self.gemini_url("files", Some(file), Some("download"), false)?;
                            url.set_query(Some("alt=media"));
                            urls.push(url);
                        } else {
                            urls.push(self.url("files", Some(file), Some("content"))?);
                        }
                    }
                }
            }
        }
        let mut results = Vec::new();
        for url in urls {
            let request = if matches!(self.protocol, BatchProtocol::Gemini) {
                self.gemini_request(Method::GET, url)?
            } else {
                self.request(Method::GET, url)
            };
            let text = Self::send(request)
                .await?
                .text()
                .await
                .map_err(|_| BatchError::new(502, "failed to read batch results"))?;
            for line in text.lines().filter(|line| !line.trim().is_empty()) {
                let raw: Value = serde_json::from_str(line)
                    .map_err(|_| BatchError::new(502, "malformed batch result JSONL"))?;
                results.push(normalize_result(raw, self.protocol)?);
            }
        }
        Ok(results)
    }
}

pub fn normalize_result(raw: Value, protocol: BatchProtocol) -> BatchResultValue<BatchResult> {
    let custom_id = raw["custom_id"]
        .as_str()
        .ok_or_else(|| BatchError::new(502, "missing batch result custom_id"))?
        .to_string();
    let (response, error) = match protocol {
        BatchProtocol::OpenAi | BatchProtocol::Gemini => {
            let response = &raw["response"];
            let status = response["status_code"].as_u64();
            if !raw["error"].is_null() {
                (None, Some(raw["error"].clone()))
            } else if status.is_some_and(|s| (200..300).contains(&s)) {
                (Some(response["body"].clone()), None)
            } else {
                (
                    None,
                    Some(json!({"status_code": response["status_code"], "body": response["body"]})),
                )
            }
        }
        BatchProtocol::Anthropic => {
            let result = &raw["result"];
            if result["type"] == "succeeded" {
                (Some(result["message"].clone()), None)
            } else {
                (None, Some(result.clone()))
            }
        }
    };
    Ok(BatchResult {
        custom_id,
        completion: response
            .as_ref()
            .map(|value| normalize_completion(value, protocol)),
        response,
        error,
        raw,
    })
}

pub fn validate_requests(requests: &[BatchRequest]) -> EngineResult<()> {
    if requests.is_empty() || requests.len() > 50_000 {
        return Err(EngineError::InvalidInput(
            "batches require 1..=50000 requests".into(),
        ));
    }
    let mut ids = HashSet::new();
    for request in requests {
        if request.custom_id.is_empty() || !ids.insert(&request.custom_id) {
            return Err(EngineError::InvalidInput(
                "batch custom_id values must be nonempty and unique".into(),
            ));
        }
        if request.body.stream == Some(true) || request.body.messages.is_empty() {
            return Err(EngineError::InvalidInput(
                "batch requests require messages and cannot stream".into(),
            ));
        }
        if [
            "reincore_session_id",
            "reincore_external_session_id",
            "reincore_budget_cents",
        ]
        .iter()
        .any(|key| request.body.extra.contains_key(*key))
        {
            return Err(EngineError::InvalidInput(
                "native batches do not support ReinCore session or budget options".into(),
            ));
        }
    }
    Ok(())
}

pub async fn create(provider: &dyn Provider, requests: &[BatchRequest]) -> BatchResultValue<Batch> {
    validate_requests(requests).map_err(|e| BatchError::new(400, e.to_string()))?;
    let client = provider
        .batch_client()
        .map_err(|_| BatchError::new(501, "provider does not support the batch lifecycle"))?;
    // Prepare every request before uploading anything: invalid input cannot partially submit.
    let bodies = requests
        .iter()
        .map(|r| {
            provider
                .prepare_batch_body(&r.body)
                .map(|body| (r.custom_id.clone(), body))
        })
        .collect::<EngineResult<Vec<_>>>()
        .map_err(|e| BatchError::new(400, e.to_string()))?;
    client.create(bodies).await
}

#[cfg(test)]
mod tests {
    //! Tests for provider batch protocols and result normalization.
    //!
    //! Local HTTP fixtures verify upload formats, authentication, cancellation,
    //! result files, error handling, model selection, and request validation.

    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn server(
        responses: Vec<(u16, String)>,
    ) -> (String, tokio::task::JoinHandle<Vec<String>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let upload_url = format!("{url}/upload/finalize");
        let task = tokio::spawn(async move {
            let mut requests = Vec::new();
            for (status, body) in responses {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut input = Vec::new();
                loop {
                    let mut chunk = [0; 8192];
                    let n = socket.read(&mut chunk).await.unwrap();
                    assert!(n > 0);
                    input.extend_from_slice(&chunk[..n]);
                    if let Some(end) = input.windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&input[..end]);
                        let length = headers
                            .lines()
                            .find_map(|line| {
                                line.to_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|s| s.trim().parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        if input.len() >= end + 4 + length {
                            break;
                        }
                    }
                }
                requests.push(String::from_utf8(input).unwrap());
                let (body, extra) = if body == "__upload__" {
                    (
                        "{}".to_string(),
                        format!("X-Goog-Upload-URL: {upload_url}\r\n"),
                    )
                } else {
                    (body, String::new())
                };
                socket.write_all(format!("HTTP/1.1 {status} Test\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
            }
            requests
        });
        (url, task)
    }

    fn client(url: String, protocol: BatchProtocol) -> BatchClient {
        BatchClient::new(
            Client::new(),
            url,
            HashMap::from([("Authorization".into(), "Bearer test".into())]),
            protocol,
            "test".into(),
        )
    }

    #[tokio::test]
    async fn openai_batch_lifecycle_includes_success_and_error_files() {
        let done = json!({"id":"job", "status":"completed", "output_file_id":"output", "error_file_id":"errors"});
        let (url, task) = server(vec![
        (200, json!({"id":"input"}).to_string()),
        (200, json!({"id":"job", "status":"validating"}).to_string()),
        (200, json!({"id":"job", "status":"cancelling"}).to_string()),
        (200, done.to_string()),
        (200, json!({"custom_id":"second", "response":{"status_code":200,"body":{"choices":[]}},"error":null}).to_string()+"\n"),
        (200, json!({"custom_id":"first", "response":null,"error":{"code":"expired"}}).to_string()+"\n"),
    ]).await;
        let client = client(url, BatchProtocol::OpenAi);
        assert_eq!(
            client
                .create(vec![
                    ("first".into(), json!({"model":"a"})),
                    ("second".into(), json!({"model":"b"}))
                ])
                .await
                .unwrap()
                .status,
            "validating"
        );
        assert_eq!(client.cancel("job").await.unwrap().status, "cancelling");
        let results = client.results("job").await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].custom_id, "second");
        assert!(results[0].response.is_some());
        assert_eq!(results[1].error.as_ref().unwrap()["code"], "expired");
        let requests = task.await.unwrap();
        assert!(requests[0].contains("\"custom_id\":\"first\""));
        assert!(requests[0].contains("\"custom_id\":\"second\""));
        assert!(requests[1].starts_with("POST /batches "));
        assert!(requests[2].starts_with("POST /batches/job/cancel "));
        assert!(requests
            .iter()
            .all(|r| r.to_lowercase().contains("authorization: bearer test")));
    }

    #[tokio::test]
    async fn anthropic_batch_lifecycle_retains_partial_failures() {
        let (url, task) = server(vec![
        (
            200,
            json!({"id":"job","processing_status":"in_progress"}).to_string(),
        ),
        (
            200,
            json!({"id":"job","processing_status":"canceling"}).to_string(),
        ),
        (
            200,
            json!({"id":"job","processing_status":"ended"}).to_string(),
        ),
        (
            200,
            format!(
                "{}\n{}\n",
                json!({"custom_id":"first","result":{"type":"succeeded","message":{"content":[]}}}),
                json!({"custom_id":"second","result":{"type":"canceled"}})
            ),
        ),
    ])
    .await;
        let client = client(url, BatchProtocol::Anthropic);
        client
            .create(vec![
                ("first".into(), json!({"model":"test"})),
                ("second".into(), json!({"model":"test"})),
            ])
            .await
            .unwrap();
        assert_eq!(client.cancel("job").await.unwrap().status, "cancelling");
        let results = client.results("job").await.unwrap();
        assert!(results[0].response.is_some());
        assert_eq!(results[1].error.as_ref().unwrap()["type"], "canceled");
        let requests = task.await.unwrap();
        let input: Value =
            serde_json::from_str(requests[0].split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(input["requests"].as_array().unwrap().len(), 2);
        assert!(requests[3].starts_with("GET /v1/messages/batches/job/results "));
    }

    #[tokio::test]
    async fn not_ready_and_provider_errors_are_explicit() {
        let (url, task) = server(vec![
            (200, json!({"id":"job","status":"in_progress"}).to_string()),
            (429, "secret upstream content".into()),
        ])
        .await;
        let client = client(url, BatchProtocol::OpenAi);
        assert_eq!(client.results("job").await.unwrap_err().status, 409);
        let err = client.get("job").await.unwrap_err();
        assert_eq!(err.status, 429);
        assert!(!err.message.contains("secret"));
        task.await.unwrap();
    }

    #[test]
    fn validates_entire_input_and_encodes_ids() {
        let request: BatchRequest = serde_json::from_value(json!({"custom_id":"a", "body":{"model":"test", "messages":[{"role":"user","content":"x"}]}})).unwrap();
        assert!(validate_requests(&[request.clone()]).is_ok());
        assert!(validate_requests(&[request.clone(), request]).is_err());
        assert!(validate_requests(&[]).is_err());
        let session_request: BatchRequest = serde_json::from_value(json!({"custom_id":"one", "body":{"model":"test", "messages":[{"role":"user","content":"x"}], "reincore_session_id":"session"}})).unwrap();
        assert!(validate_requests(&[session_request]).is_err());
        let client = client("https://provider.test".into(), BatchProtocol::OpenAi);
        assert_eq!(
            client.url("batches", Some("job/1?#"), None).unwrap().path(),
            "/batches/job%2F1%3F%23"
        );
        assert!(client.url("batches", Some(".."), None).is_err());
    }

    #[test]
    fn batch_adapter_uses_configured_model() {
        use crate::provider::{AnthropicProvider, GeminiProvider, OpenAiProvider, ProviderConfig};
        let config = ProviderConfig {
            standard_model: Some("configured-model".into()),
            ..ProviderConfig::default()
        };
        let providers: Vec<Box<dyn Provider>> = vec![
            Box::new(OpenAiProvider::new(config.clone()).unwrap()),
            Box::new(AnthropicProvider::new(config.clone()).unwrap()),
            Box::new(GeminiProvider::new(config).unwrap()),
        ];
        for provider in providers {
            for model in ["explicit-model", "reincore-auto", ""] {
                let request: ChatCompletionRequest = serde_json::from_value(
                    json!({"model":model, "messages":[{"role":"user","content":"hello"}]}),
                )
                .unwrap();
                assert_eq!(
                    provider.prepare_batch_body(&request).unwrap()["model"],
                    "configured-model"
                );
            }
        }
    }

    #[tokio::test]
    async fn gemini_uses_native_file_upload_download_and_cancellation() {
        let (url, task) = server(vec![
            (200, "__upload__".into()),
            (200, json!({"file":{"name":"files/input"}}).to_string()),
            (200, json!({"id":"job","status":"validating"}).to_string()),
            (200, "{}".into()),
            (200, json!({"id":"job","status":"cancelled"}).to_string()),
            (
                200,
                json!({"id":"job","status":"completed","output_file_id":"files/output"})
                    .to_string(),
            ),
            (
                200,
                json!({"custom_id":"one","response":{"status_code":200,"body":{"choices":[]}}})
                    .to_string(),
            ),
        ])
        .await;
        let client = client(format!("{url}/v1beta/openai"), BatchProtocol::Gemini);
        client
            .create(vec![("one".into(), json!({"model":"gemini-test"}))])
            .await
            .unwrap();
        assert_eq!(client.cancel("job").await.unwrap().status, "cancelled");
        assert_eq!(client.results("job").await.unwrap()[0].custom_id, "one");
        let requests = task.await.unwrap();
        assert!(requests[0].starts_with("POST /upload/v1beta/files "));
        assert!(requests[0].to_lowercase().contains("x-goog-api-key: test"));
        assert!(requests[1].contains("\"custom_id\":\"one\""));
        assert!(requests[2].starts_with("POST /v1beta/openai/batches "));
        assert!(requests[3].starts_with("POST /v1beta/batches/job:cancel "));
        assert!(requests[6].starts_with("GET /download/v1beta/files/output:download?alt=media "));
    }

    #[test]
    fn model_is_optional_and_provider_selection_is_explicit_only_when_needed() {
        use crate::provider::{OpenAiProvider, ProviderConfig};
        use std::sync::Arc;
        let request: CreateBatchRequest = serde_json::from_value(json!({"requests":[{"custom_id":"one","body":{"messages":[{"role":"user","content":"hello"}]}}]})).unwrap();
        validate_requests(&request.requests).unwrap();
        assert_eq!(request.requests[0].body.model, "reincore-auto");
        let registry = crate::ProviderRegistry::new();
        assert_eq!(select_provider(&registry, None).err().unwrap().status, 503);
        let registry = registry.register(Arc::new(
            OpenAiProvider::new(ProviderConfig {
                label: "first".into(),
                ..ProviderConfig::default()
            })
            .unwrap(),
        ));
        assert_eq!(select_provider(&registry, None).unwrap().label(), "first");
        assert_eq!(
            select_provider(&registry, Some("unknown"))
                .err()
                .unwrap()
                .status,
            400
        );
        let registry = registry.register(Arc::new(
            OpenAiProvider::new(ProviderConfig {
                label: "second".into(),
                ..ProviderConfig::default()
            })
            .unwrap(),
        ));
        for label in [None, Some(" ")] {
            assert_eq!(select_provider(&registry, label).err().unwrap().status, 400);
        }
        assert_eq!(
            select_provider(&registry, Some(" SECOND "))
                .unwrap()
                .label(),
            "second"
        );
    }

    #[test]
    fn anthropic_results_normalize_tools_and_cache_usage_for_agents() {
        let item = normalize_result(json!({"custom_id":"one","result":{"type":"succeeded","message":{"content":[{"type":"text","text":"hello"},{"type":"tool_use","id":"call","name":"lookup","input":{"x":1}}],"usage":{"input_tokens":10,"output_tokens":2,"cache_read_input_tokens":3,"cache_creation_input_tokens":4}}}}), BatchProtocol::Anthropic).unwrap();
        let completion = item.completion.unwrap();
        assert_eq!(
            completion["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"x\":1}"
        );
        assert_eq!(completion["usage"]["prompt_tokens"], 17);
        assert_eq!(completion["usage"]["total_tokens"], 19);
        assert_eq!(completion["usage"]["cached_prompt_tokens"], 3);
        assert_eq!(item.response.unwrap()["content"][1]["type"], "tool_use");
    }

    #[tokio::test]
    async fn batch_activity_is_persisted_without_charging_the_session() {
        use crate::{EventStore, SessionEventType};
        let store = crate::in_memory_stores::InMemoryEventStore::default();
        let session = uuid::Uuid::new_v4();
        for phase in ["submitted", "completed", "tools", "failed"] {
            record_event(
                &store,
                session,
                BatchEventRequest {
                    batch_id: "job".into(),
                    provider: "custom".into(),
                    custom_id: "one".into(),
                    phase: phase.into(),
                    data: json!({"test":true}),
                },
            )
            .await
            .unwrap();
        }
        let events = store.read_events(session, None, None).await.unwrap();
        assert_eq!(events.len(), 4);
        assert!(events
            .iter()
            .all(|e| e.event_type == SessionEventType::Batch
                && e.cost_cents.is_none()
                && e.payload["accounting"] == "unsettled"));
        assert_eq!(events[3].seq, 4);
        assert!(record_event(
            &store,
            session,
            BatchEventRequest {
                batch_id: "job".into(),
                provider: "custom".into(),
                custom_id: "one".into(),
                phase: "invalid".into(),
                data: json!({})
            }
        )
        .await
        .is_err());
        assert_eq!(
            store.read_events(session, None, None).await.unwrap().len(),
            4
        );
    }
}

/// Normalize native Anthropic messages without discarding provider output.
fn normalize_completion(value: &Value, protocol: BatchProtocol) -> Value {
    if !matches!(protocol, BatchProtocol::Anthropic) {
        let mut completion = value.clone();
        if let Some(usage) = completion.get_mut("usage").and_then(Value::as_object_mut) {
            let cached = usage
                .get("cached_prompt_tokens")
                .cloned()
                .unwrap_or_else(|| {
                    usage
                        .get("prompt_tokens_details")
                        .and_then(|d| d.get("cached_tokens"))
                        .cloned()
                        .unwrap_or(json!(0))
                });
            usage.insert("cached_prompt_tokens".into(), cached);
        }
        return completion;
    }
    let blocks = value["content"].as_array().cloned().unwrap_or_default();
    let text = blocks
        .iter()
        .filter(|b| b["type"] == "text")
        .filter_map(|b| b["text"].as_str())
        .collect::<Vec<_>>()
        .join("");
    let calls: Vec<Value> = blocks.iter().filter(|b| b["type"] == "tool_use").map(|b| json!({"id":b["id"],"type":"function","function":{"name":b["name"],"arguments":b["input"].to_string()}})).collect();
    let input = value["usage"]["input_tokens"].as_u64().unwrap_or(0);
    let cached = value["usage"]["cache_read_input_tokens"]
        .as_u64()
        .unwrap_or(0);
    let created = value["usage"]["cache_creation_input_tokens"]
        .as_u64()
        .unwrap_or(0);
    let output = value["usage"]["output_tokens"].as_u64().unwrap_or(0);
    let prompt = input.saturating_add(cached).saturating_add(created);
    json!({"id":value["id"],"model":value["model"],"choices":[{"message":{"role":"assistant","content":text,"tool_calls":calls},"finish_reason":value["stop_reason"]}],"usage":{"prompt_tokens":prompt,"completion_tokens":output,"total_tokens":prompt.saturating_add(output),"cached_prompt_tokens":cached,"prompt_tokens_details":{"cached_tokens":cached},"cache_creation_input_tokens":created}})
}

/// SDK-reported batch activity. Observability only; never settles billing.
#[derive(Deserialize)]
pub struct BatchEventRequest {
    pub batch_id: String,
    pub provider: String,
    pub custom_id: String,
    pub phase: String,
    #[serde(default)]
    pub data: Value,
}

pub async fn record_event<E: crate::EventStore>(
    store: &E,
    session_id: uuid::Uuid,
    request: BatchEventRequest,
) -> EngineResult<crate::SessionEvent> {
    if request.batch_id.is_empty()
        || request.provider.is_empty()
        || request.custom_id.is_empty()
        || !matches!(
            request.phase.as_str(),
            "submitted" | "completed" | "failed" | "tools"
        )
    {
        return Err(EngineError::InvalidInput("invalid batch event".into()));
    }
    store.append_event(session_id, crate::SessionEventType::Batch, json!({"source":"sdk","batch_id":request.batch_id,"provider":request.provider,"custom_id":request.custom_id,"phase":request.phase,"data":request.data,"accounting":"unsettled"}), None).await
}
