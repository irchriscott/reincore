use super::LocalDatabase;
use engine_core::{
    batch_jobs::{BatchJob, BatchJobStore},
    EngineError, EngineResult,
};
use rusqlite::{params, OptionalExtension};
use uuid::Uuid;
#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{
        batch_jobs as jobs,
        provider::{batch::BatchRequest, GeminiProvider, ProviderConfig},
        EventStore, Session,
    };
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn server(
        responses: Vec<(u16, String)>,
    ) -> (String, tokio::task::JoinHandle<Vec<String>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let upload = format!("{base}/finalize");
        let task = tokio::spawn(async move {
            let mut requests = Vec::new();
            for (status, body) in responses {
                let (mut socket, _) =
                    tokio::time::timeout(std::time::Duration::from_secs(10), listener.accept())
                        .await
                        .unwrap()
                        .unwrap();
                let mut data = Vec::new();
                loop {
                    let mut buf = [0; 8192];
                    let n = socket.read(&mut buf).await.unwrap();
                    assert!(n > 0);
                    data.extend_from_slice(&buf[..n]);
                    if let Some(end) = data.windows(4).position(|w| w == b"\r\n\r\n") {
                        let len = String::from_utf8_lossy(&data[..end])
                            .lines()
                            .find_map(|l| {
                                l.to_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|v| v.trim().parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        if data.len() >= end + 4 + len {
                            break;
                        }
                    }
                }
                requests.push(String::from_utf8(data).unwrap());
                let (body, extra) = if body == "upload" {
                    ("{}".into(), format!("x-goog-upload-url: {upload}\r\n"))
                } else {
                    (body, String::new())
                };
                socket.write_all(format!("HTTP/1.1 {status} Test\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n{body}",body.len()).as_bytes()).await.unwrap();
            }
            requests
        });
        (base, task)
    }
    fn provider(base: &str, key: &str) -> GeminiProvider {
        GeminiProvider::new(ProviderConfig {
            label: "gemini".into(),
            base_url: format!("{base}/v1beta/openai"),
            api_key: key.into(),
            standard_model: Some("gemini-3.8-flash".into()),
            ..Default::default()
        })
        .unwrap()
    }
    fn requests(ids: &[&str]) -> Vec<BatchRequest> {
        ids.iter().map(|id|serde_json::from_value(json!({"custom_id":id,"body":{"messages":[{"role":"user","content":"Say OK"}],"max_tokens":32}})).unwrap()).collect()
    }
    async fn due(db: &LocalDatabase, id: Uuid) -> jobs::BatchJob {
        let mut job = jobs::get(db, id).await.unwrap();
        job.next_poll_at = chrono::Utc::now();
        assert!(db.update_batch_job(&job).await.unwrap());
        jobs::get(db, id).await.unwrap()
    }

    #[tokio::test]
    async fn batch_redaction_metadata_is_non_blocking_and_survives_persistence() {
        let (base, wire) = server(vec![
            (200, "upload".into()),
            (200, json!({"file":{"name":"files/input"}}).to_string()),
            (
                200,
                json!({"id":"remote","status":"validating"}).to_string(),
            ),
        ])
        .await;
        let db = LocalDatabase::open_in_memory().unwrap();
        let session = Session::new("batch privacy", 1000).unwrap();
        db.insert_session(&session).unwrap();
        let input: BatchRequest = serde_json::from_value(json!({
            "custom_id":"one",
            "body":{
                "messages":[
                    {"role":"user","content":"Inspect the environment file"},
                    {"role":"assistant","tool_calls":[{
                        "id":"read-1","type":"function","function":{
                            "name":"read_file","arguments":"{\"path\":\".env\"}"
                        }
                    }]},
                    {"role":"tool","tool_call_id":"read-1","content":"PASSWORD=hunter123"}
                ],
                "tools":[{"type":"function","function":{
                    "name":"write_file","parameters":{"type":"object"}
                }}]
            }
        }))
        .unwrap();
        let job = jobs::create(
            &db,
            &provider(&base, "test-key"),
            session.id,
            Uuid::new_v4(),
            &[input],
        )
        .await
        .unwrap();
        let saved = jobs::get(&db, job.id).await.unwrap();
        let disposition = saved.items[0].privacy_redaction.as_ref().unwrap();
        assert_eq!(disposition.tool_call_ids, ["read-1"]);
        assert_eq!(disposition.targets, [".env"]);
        assert!(disposition.retryable);
        assert!(!disposition.user_action_required);
        assert!(!disposition.execution_blocked);
        assert!(!saved.items[0].body.to_string().contains("hunter123"));
        let requests = wire.await.unwrap();
        assert!(!requests.iter().any(|request| request.contains("hunter123")));
    }

    #[tokio::test]
    async fn gemini_durable_upload_poll_restart_partial_results_and_exact_accounting() {
        let done = json!({"id":"remote","status":"completed","output_file_id":"files/output","error_file_id":"files/errors"});
        let (base,wire)=server(vec![
        (200,"upload".into()),(200,json!({"file":{"name":"files/input"}}).to_string()),
        (200,json!({"id":"remote","status":"validating"}).to_string()),
        (200,json!({"id":"remote","status":"in_progress"}).to_string()),
        (200,done.to_string()),(200,done.to_string()),
        (200,json!({"custom_id":"ok","response":{"status_code":200,"body":{"model":"gemini-3.8-flash","choices":[{"message":{"role":"assistant","content":"OK"}}],"usage":{"prompt_tokens":100,"completion_tokens":10,"prompt_tokens_details":{"cached_tokens":20}}}}}).to_string()),
        (200,json!({"custom_id":"failed","error":{"code":"invalid_request"}}).to_string()),
    ]).await;
        let path = std::env::temp_dir().join(format!("reincore-batch-{}.db", Uuid::new_v4()));
        let db = LocalDatabase::open(&path).unwrap();
        let session = Session::new("batch", 1000).unwrap();
        db.insert_session(&session).unwrap();
        let provider = provider(&base, "test-key");
        let id = Uuid::new_v4();
        let input = requests(&["ok", "failed", "missing"]);
        let job = jobs::create(&db, &provider, session.id, id, &input)
            .await
            .unwrap();
        assert_eq!(
            jobs::create(&db, &provider, session.id, id, &input)
                .await
                .unwrap()
                .id,
            id
        );
        assert_eq!(
            jobs::refresh(&db, &provider, job).await.unwrap().status,
            "in_progress"
        );
        drop(db);
        let db = LocalDatabase::open(&path).unwrap();
        assert_eq!(db.pending_batch_jobs().await.unwrap().len(), 1);
        let job = jobs::refresh(&db, &provider, due(&db, id).await)
            .await
            .unwrap();
        assert_eq!(job.status, "partial_failed");
        assert_eq!(job.results.as_ref().unwrap().len(), 3);
        let duplicate = jobs::refresh(&db, &provider, job.clone()).await.unwrap();
        assert!(duplicate.finished());
        let session = db.get_session(session.id).unwrap().unwrap();
        assert_eq!(session.spend_nano_usd, Some(49_500));
        assert_eq!(session.unknown_cost_calls, 2);
        assert_eq!(
            db.read_events(session.id, None, None).await.unwrap().len(),
            3
        );
        assert!(db.pending_batch_jobs().await.unwrap().is_empty());
        let wire = wire.await.unwrap();
        assert_eq!(wire.len(), 8);
        assert!(wire[0].starts_with("POST /upload/v1beta/files "));
        assert!(wire[0]
            .to_lowercase()
            .contains("x-goog-upload-command: start"));
        assert!(wire[1]
            .to_lowercase()
            .contains("x-goog-upload-command: upload, finalize"));
        assert!(wire[1].contains("\"custom_id\":\"ok\""));
        assert!(wire[1].contains("gemini-3.8-flash"));
        assert!(wire[2].starts_with("POST /v1beta/openai/batches "));
        assert!(wire[2].contains("\"input_file_id\":\"files/input\""));
        assert!(wire[6].starts_with("GET /download/v1beta/files/output:download?alt=media "));
        assert!(wire[6].to_lowercase().contains("x-goog-api-key: test-key"));
        assert!(!serde_json::to_string(&job).unwrap().contains("test-key"));
        drop(db);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn ambiguous_submission_retry_does_not_resubmit_and_credentials_are_bound() {
        let (base, wire) = server(vec![
            (200, "upload".into()),
            (200, json!({"file":{"name":"files/in"}}).to_string()),
            (503, "sensitive upstream body".into()),
        ])
        .await;
        let db = LocalDatabase::open_in_memory().unwrap();
        let session = Session::new("batch", 1000).unwrap();
        db.insert_session(&session).unwrap();
        let id = Uuid::new_v4();
        let p = provider(&base, "secret");
        let input = requests(&["one"]);
        let job = jobs::create(&db, &p, session.id, id, &input).await.unwrap();
        assert_eq!(job.status, "submission_unknown");
        assert_eq!(
            jobs::create(&db, &p, session.id, id, &input)
                .await
                .unwrap()
                .status,
            "submission_unknown"
        );
        assert_eq!(
            jobs::create(&db, &p, Uuid::new_v4(), id, &input)
                .await
                .unwrap_err()
                .status,
            409
        );
        assert_eq!(
            jobs::refresh(&db, &provider(&base, "another-key"), job)
                .await
                .unwrap_err()
                .status,
            403
        );
        assert_eq!(wire.await.unwrap().len(), 3);
        assert_eq!(db.get_session(session.id).unwrap().unwrap().spend_cents, 0);
    }

    #[tokio::test]
    async fn durable_receipts_recover_after_partial_settlement_without_double_billing() {
        let db = LocalDatabase::open_in_memory().unwrap();
        let session = Session::new("batch", 1000).unwrap();
        db.insert_session(&session).unwrap();
        let p = provider("http://127.0.0.1:1", "key");
        let cost = engine_core::pricing::estimate(
            "gemini",
            "gemini-3.8-flash",
            &engine_core::TokenUsage {
                prompt_tokens: 100,
                ..Default::default()
            },
            false,
            true,
        );
        let receipt = engine_core::accounting::CompletionAccounting {
            id: Uuid::new_v4(),
            session_id: session.id,
            recorded_at: chrono::Utc::now(),
            cost_cents: cost.display_cents(),
            cost: Some(cost),
            purpose: Some("batch".into()),
            payload: json!({"input_tokens":100,"output_tokens":0}),
        };
        use engine_core::accounting::AccountingStore;
        db.record_accounting(&receipt).await.unwrap();
        let job = jobs::BatchJob {
            id: Uuid::new_v4(),
            session_id: session.id,
            provider: "gemini".into(),
            scope: engine_core::Provider::batch_client(&p).unwrap().scope(),
            request_fingerprint: "fixture".into(),
            items: vec![jobs::JobItem {
                tools: vec![],
                privacy_redaction: None,
                custom_id: "one".into(),
                body: json!({}),
                accounting_id: receipt.id,
            }],
            remote: Some(engine_core::provider::batch::Batch {
                id: "remote".into(),
                provider: "gemini".into(),
                status: "completed".into(),
                request_counts: json!({}),
                raw: json!({}),
            }),
            status: "reconciling".into(),
            results: Some(vec![engine_core::provider::batch::BatchResult {
                custom_id: "one".into(),
                response: Some(json!({})),
                completion: None,
                error: None,
                raw: json!({}),
            }]),
            receipts: vec![receipt],
            revision: 0,
            next_poll_at: chrono::Utc::now(),
            created_at: chrono::Utc::now(),
            error: None,
        };
        db.insert_batch_job(&job).await.unwrap();
        let (a, b) = tokio::join!(
            jobs::refresh(&db, &p, job.clone()),
            jobs::refresh(&db, &p, job)
        );
        a.unwrap();
        b.unwrap();
        assert_eq!(
            db.get_session(session.id).unwrap().unwrap().spend_nano_usd,
            Some(37_500)
        );
        assert_eq!(
            db.read_events(session.id, None, None).await.unwrap().len(),
            1
        );
    }

    #[tokio::test]
    async fn batch_recovery_omits_raw_and_signed_state_instead_of_rewriting_it() {
        let db = LocalDatabase::open_in_memory().unwrap();
        let session = Session::new("batch privacy", 1000).unwrap();
        db.insert_session(&session).unwrap();
        let job = jobs::BatchJob {
            id: Uuid::new_v4(),
            session_id: session.id,
            provider: "gemini".into(),
            scope: "hashed-scope".into(),
            request_fingerprint: "fingerprint".into(),
            items: vec![jobs::JobItem {
                tools: vec![],
                privacy_redaction: None,
                custom_id: "one".into(),
                body: json!({"messages":[{"content":"api_key=abcdefghijk"}]}),
                accounting_id: Uuid::new_v4(),
            }],
            remote: Some(engine_core::provider::batch::Batch {
                id: "remote".into(),
                provider: "gemini".into(),
                status: "completed".into(),
                request_counts: json!({}),
                raw: json!({"signature":"opaque-signed-provider-state"}),
            }),
            status: "completed".into(),
            results: Some(vec![engine_core::provider::batch::BatchResult {
                custom_id: "one".into(),
                response: None,
                completion: Some(json!({
                    "reasoning_content":"ephemeral",
                    "choices":[{"message":{"content":"ok","provider_state":{"signature":"signed"}}}]
                })),
                error: None,
                raw: json!({"signature":"another-opaque-value"}),
            }]),
            receipts: vec![],
            revision: 0,
            next_poll_at: chrono::Utc::now(),
            created_at: chrono::Utc::now(),
            error: None,
        };
        db.insert_batch_job(&job).await.unwrap();
        let stored = db.get_batch_job(job.id).await.unwrap().unwrap();
        assert_eq!(
            stored.items[0].body["messages"][0]["content"],
            "api_key=[REDACTED]"
        );
        assert!(stored.remote.as_ref().unwrap().raw.is_null());
        let result = &stored.results.as_ref().unwrap()[0];
        assert!(result.raw.is_null());
        let completion = result.completion.as_ref().unwrap();
        assert!(completion.get("reasoning_content").is_none());
        assert!(completion["choices"][0]["message"]
            .get("provider_state")
            .is_none());
    }

    #[tokio::test]
    #[ignore = "paid Gemini sandbox: explicitly configure GEMINI_API_KEY and REINCORE_BATCH_SANDBOX_MODEL"]
    async fn gemini_batch_sandbox_upload_submit_poll_download_and_accounting() {
        let key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY required");
        let model = std::env::var("REINCORE_BATCH_SANDBOX_MODEL")
            .expect("choose a supported sandbox model explicitly");
        let provider = GeminiProvider::new(ProviderConfig {
            label: "gemini".into(),
            api_key: key,
            base_url: "https://generativelanguage.googleapis.com/v1beta/openai".into(),
            standard_model: Some(model),
            ..Default::default()
        })
        .unwrap();
        let path =
            std::env::temp_dir().join(format!("reincore-batch-sandbox-{}.db", Uuid::new_v4()));
        let db = LocalDatabase::open(&path).unwrap();
        let session = Session::new("paid batch sandbox", 100).unwrap();
        db.insert_session(&session).unwrap();
        let id = Uuid::new_v4();
        let mut job = jobs::create(&db, &provider, session.id, id, &requests(&["sandbox"]))
            .await
            .unwrap();
        eprintln!("Sandbox job {id}; recovery database {}", path.display());
        let timeout = std::env::var("REINCORE_BATCH_SANDBOX_TIMEOUT_SECONDS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(1800);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout);
        while !job.finished() && std::time::Instant::now() < deadline {
            assert_ne!(
                job.status, "submission_unknown",
                "inspect provider jobs before any retry"
            );
            job = jobs::refresh(&db, &provider, job).await.unwrap();
            if !job.finished() {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            }
        }
        if !job.finished() {
            let _ = jobs::cancel(&db, &provider, job).await;
            panic!("sandbox timed out; inspect persisted job {id}");
        }
        assert_eq!(job.status, "completed");
        assert_eq!(job.receipts.len(), 1);
        assert!(job.receipts[0].amount_nano_usd().is_some());
        assert_eq!(
            db.get_session(session.id).unwrap().unwrap().spend_nano_usd,
            job.receipts[0].amount_nano_usd()
        );
    }

    #[tokio::test]
    async fn cancellation_intent_survives_restart_and_accounts_partial_success() {
        let done =
            json!({"id":"batches/remote","status":"cancelled","output_file_id":"files/output"});
        let output = json!({"custom_id":"ok","response":{"status_code":200,"body":{"model":"gemini-3.8-flash","usage":{"prompt_tokens":100,"completion_tokens":0},"choices":[]}}});
        let (base, wire) = server(vec![
            (200, "upload".into()),
            (200, json!({"file":{"name":"files/input"}}).to_string()),
            (
                200,
                json!({"id":"batches/remote","status":"in_progress"}).to_string(),
            ),
            (
                200,
                json!({"id":"batches/remote","status":"in_progress"}).to_string(),
            ),
            (200, "{}".into()),
            (200, done.to_string()),
            (200, done.to_string()),
            (200, output.to_string()),
        ])
        .await;
        let path = std::env::temp_dir().join(format!("reincore-cancel-{}.db", Uuid::new_v4()));
        let db = LocalDatabase::open(&path).unwrap();
        let session = Session::new("cancel", 1000).unwrap();
        db.insert_session(&session).unwrap();
        let p = provider(&base, "key");
        let mut job = jobs::create(
            &db,
            &p,
            session.id,
            Uuid::new_v4(),
            &requests(&["ok", "cancelled"]),
        )
        .await
        .unwrap();
        job.status = "cancelling".into();
        assert!(db.update_batch_job(&job).await.unwrap());
        drop(db);
        let db = LocalDatabase::open(&path).unwrap();
        let job = jobs::refresh(&db, &p, due(&db, job.id).await)
            .await
            .unwrap();
        assert_eq!(job.status, "partial_failed");
        assert_eq!(
            db.get_session(session.id).unwrap().unwrap().spend_nano_usd,
            Some(37_500)
        );
        let wire = wire.await.unwrap();
        assert!(wire[3].starts_with("GET /v1beta/openai/batches/remote "));
        assert!(wire[4].starts_with("POST /v1beta/batches/remote:cancel "));
        assert!(wire[4].to_lowercase().contains("x-goog-api-key: key"));
        assert_eq!(wire.iter().filter(|r| r.contains(":cancel ")).count(), 1);
        drop(db);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn invalid_result_identity_withholds_all_accounting() {
        for ids in [["one", "one"], ["one", "another-job"]] {
            let done = json!({"id":"remote","status":"completed","output_file_id":"files/output"});
            let output=ids.iter().map(|id|json!({"custom_id":id,"response":{"status_code":200,"body":{"usage":{"prompt_tokens":100,"completion_tokens":1}}}}).to_string()).collect::<Vec<_>>().join("\n");
            let (base, wire) = server(vec![
                (200, "upload".into()),
                (200, json!({"file":{"name":"files/input"}}).to_string()),
                (200, done.to_string()),
                (200, done.to_string()),
                (200, done.to_string()),
                (200, output),
            ])
            .await;
            let db = LocalDatabase::open_in_memory().unwrap();
            let session = Session::new("identity", 1000).unwrap();
            db.insert_session(&session).unwrap();
            let p = provider(&base, "key");
            let job = jobs::create(
                &db,
                &p,
                session.id,
                Uuid::new_v4(),
                &requests(&["one", "two"]),
            )
            .await
            .unwrap();
            assert_eq!(jobs::refresh(&db, &p, job).await.unwrap_err().status, 502);
            assert!(db
                .read_events(session.id, None, None)
                .await
                .unwrap()
                .is_empty());
            assert_eq!(db.get_session(session.id).unwrap().unwrap().spend_cents, 0);
            wire.await.unwrap();
        }
    }

    #[tokio::test]
    async fn batch_routes_require_session_and_reject_changed_credentials() {
        use axum::{
            body::Body,
            http::Request,
            routing::{get, post},
            Router,
        };
        use std::sync::Arc;
        use tower::ServiceExt;
        let (base, wire) = server(vec![
            (200, "upload".into()),
            (200, json!({"file":{"name":"files/input"}}).to_string()),
            (503, "unknown".into()),
        ])
        .await;
        let db = LocalDatabase::open_in_memory().unwrap();
        let session = Session::new("route", 1000).unwrap();
        db.insert_session(&session).unwrap();
        let state = crate::app::SingleProviderAppState::new_with_project_path(
            db,
            Arc::new(provider(&base, "key")),
            None,
        );
        let routes = || {
            Router::new()
                .route(
                    "/v1/batches",
                    post(crate::http::create_batch_single_provider),
                )
                .route(
                    "/v1/batches/:id",
                    get(crate::http::get_batch_single_provider),
                )
        };
        let app = routes().with_state(state.clone());
        let id = Uuid::new_v4();
        for (session_id, expected) in [
            (None, 400),
            (Some(Uuid::new_v4()), 404),
            (Some(session.id), 200),
        ] {
            let body =
                json!({"session_id":session_id,"idempotency_key":id,"requests":requests(&["one"])});
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/batches")
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), expected);
        }
        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/batches/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), 200);
        let mut changed = state;
        changed.provider = Arc::new(provider(&base, "rotated"));
        let response = routes()
            .with_state(changed)
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/batches/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), 403);
        assert_eq!(wire.await.unwrap().len(), 3);
    }
}
fn storage(e: impl std::fmt::Display) -> EngineError {
    EngineError::Storage(e.to_string())
}

#[async_trait::async_trait]
impl BatchJobStore for LocalDatabase {
    async fn pending_batch_jobs(&self) -> EngineResult<Vec<BatchJob>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let mut stmt = conn
                .prepare("SELECT record FROM batch_jobs WHERE active=1")
                .map_err(storage)?;
            let rows = stmt
                .query_map([], |r| r.get::<_, String>(0))
                .map_err(storage)?;
            rows.map(|r| serde_json::from_str(&r.map_err(storage)?).map_err(EngineError::from))
                .collect()
        })
        .await
        .map_err(storage)?
    }
    async fn insert_batch_job(&self, job: &BatchJob) -> EngineResult<bool> {
        let db = self.clone();
        let job = job.durable_copy()?;
        tokio::task::spawn_blocking(move || {
            let conn=db.conn.lock().unwrap();
            Ok(conn.execute("INSERT INTO batch_jobs(id,session_id,revision,record) VALUES (?1,?2,0,?3) ON CONFLICT(id) DO NOTHING",
                params![job.id.to_string(),job.session_id.to_string(),serde_json::to_string(&job)?]).map_err(storage)? == 1)
        }).await.map_err(storage)?
    }
    async fn get_batch_job(&self, id: Uuid) -> EngineResult<Option<BatchJob>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let record: Option<String> = conn
                .query_row(
                    "SELECT record FROM batch_jobs WHERE id=?1",
                    [id.to_string()],
                    |r| r.get(0),
                )
                .optional()
                .map_err(storage)?;
            record
                .map(|s| serde_json::from_str(&s).map_err(EngineError::from))
                .transpose()
        })
        .await
        .map_err(storage)?
    }
    async fn update_batch_job(&self, job: &BatchJob) -> EngineResult<bool> {
        let db = self.clone();
        let mut job = job.durable_copy()?;
        tokio::task::spawn_blocking(move || {
            let expected=job.revision;job.revision+=1;
            let conn=db.conn.lock().unwrap();
            Ok(conn.execute("UPDATE batch_jobs SET revision=?1, record=?2, active=?5 WHERE id=?3 AND revision=?4",
                params![job.revision,serde_json::to_string(&job)?,job.id.to_string(),expected,!job.finished() && job.status != "submission_unknown"]).map_err(storage)? == 1)
        }).await.map_err(storage)?
    }
}
