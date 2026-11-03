//! Durable batch orchestration. Persist intent before submission, and immutable
//! receipts before settlement. Ambiguous submissions are never retried as new work.
use crate::{
    accounting::{AccountingStore, CompletionAccounting},
    provider::batch::{
        self, Batch, BatchClient, BatchError, BatchRequest, BatchResult, BatchResultValue,
    },
    EngineError, EngineResult, Provider,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobItem {
    /// Original normalized contract; older jobs cannot authorize tool actions.
    #[serde(default)]
    pub tools: Vec<crate::ToolDefinition>,
    /// Sanitized redaction metadata captured for guidance and auditing.
    /// Redaction is not an execution boundary; this metadata must never
    /// withhold otherwise valid batch actions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privacy_redaction: Option<crate::PrivacyRedactionDisposition>,
    pub custom_id: String,
    pub body: Value,
    pub accounting_id: Uuid,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchJob {
    pub id: Uuid,
    pub session_id: Uuid,
    pub provider: String,
    pub scope: String,
    pub request_fingerprint: String,
    pub items: Vec<JobItem>,
    pub remote: Option<Batch>,
    pub status: String,
    pub results: Option<Vec<BatchResult>>,
    pub receipts: Vec<CompletionAccounting>,
    pub revision: i64,
    pub next_poll_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub error: Option<String>,
}
impl BatchJob {
    pub fn public(&self) -> Batch {
        Batch {
            id: self.id.to_string(),
            provider: self.provider.clone(),
            status: self.status.clone(),
            request_counts: self
                .remote
                .as_ref()
                .map(|r| r.request_counts.clone())
                .unwrap_or(Value::Null),
            raw: json!({"session_id":self.session_id,"provider_batch_id":self.remote.as_ref().map(|r| &r.id),
                "next_poll_at":self.next_poll_at,"error":self.error,"accounting_complete":self.finished()}),
        }
    }
    pub fn finished(&self) -> bool {
        matches!(
            self.status.as_str(),
            "completed" | "partial_failed" | "failed" | "cancelled" | "expired"
        )
    }
    pub fn check_scope(&self, client: &BatchClient) -> BatchResultValue<()> {
        if self.scope != client.scope() {
            return Err(BatchError::new(
                403,
                "batch provider credentials or endpoint do not match",
            ));
        }
        Ok(())
    }

    /// Build the recovery record written to durable storage. Provider-owned
    /// raw payloads may be signed, so they are omitted rather than rewritten.
    pub fn durable_copy(&self) -> EngineResult<Self> {
        let policy = crate::privacy::current_policy();
        if !policy.enabled {
            return Ok(self.clone());
        }
        let mut job = self.clone();
        for item in &mut job.items {
            policy.sanitize_persistent_value(&mut item.body);
            for tool in &mut item.tools {
                let mut value = serde_json::to_value(&*tool)?;
                policy.sanitize_persistent_value(&mut value);
                *tool = serde_json::from_value(value)?;
            }
        }
        if let Some(remote) = &mut job.remote {
            policy.sanitize_persistent_value(&mut remote.request_counts);
            remote.raw = Value::Null;
        }
        if let Some(results) = &mut job.results {
            for result in results {
                for value in [
                    &mut result.response,
                    &mut result.completion,
                    &mut result.error,
                ] {
                    if let Some(value) = value {
                        policy.sanitize_persistent_value(value);
                    }
                }
                result.raw = Value::Null;
            }
        }
        for receipt in &mut job.receipts {
            policy.sanitize_persistent_value(&mut receipt.payload);
        }
        if let Some(error) = &mut job.error {
            *error = policy.sanitize_persistent_text(error);
        }
        Ok(job)
    }
}

#[async_trait::async_trait]
pub trait BatchJobStore: AccountingStore {
    async fn pending_batch_jobs(&self) -> EngineResult<Vec<BatchJob>>;
    async fn insert_batch_job(&self, job: &BatchJob) -> EngineResult<bool>;
    async fn get_batch_job(&self, id: Uuid) -> EngineResult<Option<BatchJob>>;
    /// Compare-and-swap; incoming revision is the expected current revision.
    async fn update_batch_job(&self, job: &BatchJob) -> EngineResult<bool>;
}
fn storage(_: EngineError) -> BatchError {
    BatchError::new(500, "batch persistence failed; retry the same job ID")
}
pub async fn get<S: BatchJobStore + ?Sized>(store: &S, id: Uuid) -> BatchResultValue<BatchJob> {
    store
        .get_batch_job(id)
        .await
        .map_err(storage)?
        .ok_or_else(|| BatchError::new(404, "batch job not found"))
}
async fn save<S: BatchJobStore + ?Sized>(store: &S, job: &mut BatchJob) -> BatchResultValue<bool> {
    if store.update_batch_job(job).await.map_err(storage)? {
        job.revision += 1;
        Ok(true)
    } else {
        *job = get(store, job.id).await?;
        Ok(false)
    }
}

pub async fn create<S: BatchJobStore + ?Sized>(
    store: &S,
    provider: &dyn Provider,
    session_id: Uuid,
    id: Uuid,
    requests: &[BatchRequest],
) -> BatchResultValue<BatchJob> {
    if id.is_nil() {
        return Err(BatchError::new(400, "idempotency_key must not be nil"));
    }
    let privacy = crate::privacy::current_policy();
    if requests
        .iter()
        .any(|request| privacy.contains_secret(&request.custom_id))
    {
        return Err(BatchError::new(
            400,
            "batch custom_id contains sensitive data blocked by privacy policy",
        ));
    }
    batch::validate_requests(requests).map_err(|e| BatchError::new(400, e.to_string()))?;
    let client = provider
        .batch_client()
        .map_err(|_| BatchError::new(501, "provider does not support batch jobs"))?;
    let bodies = requests
        .iter()
        .map(|r| {
            provider
                .prepare_batch_body(&r.body)
                .map(|body| (r.custom_id.clone(), body))
        })
        .collect::<EngineResult<Vec<_>>>()
        .map_err(|e| BatchError::new(400, e.to_string()))?;
    let privacy_redactions = requests
        .iter()
        .map(|request| {
            privacy
                .prepare_provider_request(&request.body, provider.label())
                .map(|prepared| crate::privacy::redaction_disposition(&prepared, &privacy))
        })
        .collect::<EngineResult<Vec<_>>>()
        .map_err(|e| BatchError::new(400, e.to_string()))?;
    let fingerprint = crate::CacheKey::for_tool_call("batch-request-v1", &json!(bodies))
        .expect("serializable request")
        .0;
    let now = Utc::now();
    let mut job = BatchJob {
        id,
        session_id,
        provider: provider.label().into(),
        scope: client.scope(),
        request_fingerprint: fingerprint,
        items: bodies
            .iter()
            .zip(requests)
            .zip(privacy_redactions)
            .map(|(((id, body), request), privacy_redaction)| JobItem {
                custom_id: id.clone(),
                tools: request.body.tools.clone().unwrap_or_default(),
                privacy_redaction,
                body: body.clone(),
                accounting_id: Uuid::new_v4(),
            })
            .collect(),
        remote: None,
        status: "submitting".into(),
        results: None,
        receipts: vec![],
        revision: 0,
        next_poll_at: now,
        created_at: now,
        error: None,
    };
    if !store.insert_batch_job(&job).await.map_err(storage)? {
        let existing = get(store, id).await?;
        if existing.session_id != session_id
            || existing.scope != job.scope
            || existing.request_fingerprint != job.request_fingerprint
        {
            return Err(BatchError::new(
                409,
                "idempotency_key already belongs to another batch request",
            ));
        }
        return Ok(existing);
    }
    // Only the successful insert owner may submit. A crash here requires provider
    // investigation, never blind resubmission (Google creation is not idempotent).
    match client
        .create_tagged(bodies, &format!("reincore-{}", job.id))
        .await
    {
        Ok(remote) => {
            job.remote = Some(remote);
            job.status = "queued".into();
        }
        Err(error) => {
            job.status = "submission_unknown".into();
            job.error = Some(format!(
                "Submission outcome unknown (HTTP {}); inspect provider jobs before retrying",
                error.status
            ));
        }
    }
    if !save(store, &mut job).await? {
        return Err(BatchError::new(
            409,
            "batch changed during submission; retrieve its existing job ID",
        ));
    }
    Ok(job)
}

pub async fn refresh<S: BatchJobStore + ?Sized>(
    store: &S,
    provider: &dyn Provider,
    mut job: BatchJob,
) -> BatchResultValue<BatchJob> {
    let client = provider
        .batch_client()
        .map_err(|_| BatchError::new(501, "provider does not support batch jobs"))?;

    job.check_scope(&client)?;

    if job.finished() {
        return Ok(job);
    }

    let Some(remote) = job.remote.clone() else {
        if job.status == "submitting" && Utc::now() - job.created_at > chrono::Duration::minutes(5)
        {
            job.status = "submission_unknown".into();
            job.error =
                Some("Submission interrupted; inspect provider jobs before retrying".into());
            save(store, &mut job).await?;
        }
        return Ok(job);
    };

    if job.results.is_none() {
        if Utc::now() < job.next_poll_at {
            return Ok(job);
        }
        // Claim this polling interval before network I/O. Other processes return
        // the saved snapshot; after a crash the interval expires automatically.
        job.next_poll_at = Utc::now() + chrono::Duration::seconds(30);
        if !save(store, &mut job).await? {
            return Ok(job);
        }
        let cancelling = job.status == "cancelling";
        let snapshot = client.get(&remote.id).await?;
        let snapshot = if cancelling && !snapshot.is_terminal() && snapshot.status != "cancelling" {
            client.cancel(&remote.id).await?
        } else {
            snapshot
        };
        job.status = if snapshot.is_terminal() {
            "reconciling".into()
        } else if cancelling {
            "cancelling".into()
        } else {
            "in_progress".into()
        };
        job.error = None;
        job.remote = Some(snapshot.clone());
        if snapshot.is_terminal() {
            let downloaded = client.results(&remote.id).await?;
            let mut mapped = std::collections::HashMap::new();
            let expected_ids: std::collections::HashSet<_> =
                job.items.iter().map(|i| i.custom_id.as_str()).collect();
            for result in downloaded {
                if !expected_ids.contains(result.custom_id.as_str())
                    || mapped.insert(result.custom_id.clone(), result).is_some()
                {
                    return Err(BatchError::new(
                        502,
                        "unknown or duplicate result custom_id; accounting withheld",
                    ));
                }
            }
            let at = Utc::now();
            let mut results = Vec::new();
            for item in &job.items {
                let result = mapped
                    .remove(&item.custom_id)
                    .unwrap_or_else(|| BatchResult {
                        custom_id: item.custom_id.clone(),
                        response: None,
                        completion: None,
                        error: Some(
                            json!({"code":"missing_result","batch_status":snapshot.status}),
                        ),
                        raw: Value::Null,
                    });
                let usage = client.usage(&result);
                let model = result
                    .response
                    .as_ref()
                    .and_then(|r| r["model"].as_str())
                    .or_else(|| item.body["model"].as_str())
                    .unwrap_or("unknown");
                let mut cost = usage
                    .as_ref()
                    .map(|usage| {
                        crate::pricing::estimate_from_catalog(
                            crate::model_library::model_catalog(),
                            provider.label(),
                            model,
                            usage,
                            true,
                            at,
                        )
                    })
                    .unwrap_or_else(|| {
                        crate::pricing::CostEstimate::unknown(
                            provider.label(),
                            model,
                            "batch item has no usage; not assumed free",
                        )
                    });
                cost.batch_submitted = true;
                let tokens = usage.unwrap_or_default();
                job.receipts.push(CompletionAccounting {id:item.accounting_id, session_id:job.session_id,recorded_at:at,
                    cost_cents:cost.display_cents(),cost:Some(cost),purpose:Some("batch".into()),
                    payload:json!({"batch_job_id":job.id,"provider_batch_id":remote.id,"custom_id":item.custom_id,
                        "model":model,"input_tokens":tokens.prompt_tokens,"output_tokens":tokens.completion_tokens,
                        "response_message":result.completion.as_ref().and_then(|c| c.pointer("/choices/0/message")),"error":result.error})});
                results.push(result);
            }
            job.results = Some(results);
        }
        if !save(store, &mut job).await? {
            return Ok(job);
        }
    }
    if job.results.is_some() {
        // Immutable receipts are already durable. Concurrent/restarted workers
        // stage identical IDs/records and the existing ledger settles only once.
        for receipt in &job.receipts {
            store.stage_accounting(receipt).await.map_err(storage)?;
            store.settle_accounting(receipt.id).await.map_err(storage)?;
        }
        job.error = None;
        let successes = job
            .results
            .as_ref()
            .unwrap()
            .iter()
            .filter(|r| r.error.is_none())
            .count();
        job.status = if successes == job.items.len() {
            "completed".into()
        } else if successes > 0 {
            "partial_failed".into()
        } else {
            match job.remote.as_ref().unwrap().status.as_str() {
                "cancelled" => "cancelled",
                "expired" => "expired",
                _ => "failed",
            }
            .into()
        };
        save(store, &mut job).await?;
    }
    Ok(job)
}

pub async fn cancel<S: BatchJobStore + ?Sized>(
    store: &S,
    provider: &dyn Provider,
    mut job: BatchJob,
) -> BatchResultValue<BatchJob> {
    let client = provider
        .batch_client()
        .map_err(|_| BatchError::new(501, "provider does not support batch jobs"))?;
    job.check_scope(&client)?;
    if job.finished() || job.results.is_some() {
        return refresh(store, provider, job).await;
    }
    let remote = job.remote.clone().ok_or_else(|| {
        BatchError::new(
            409,
            "submission outcome unknown; cancellation requires a provider job ID",
        )
    })?;
    job.status = "cancelling".into();
    if !save(store, &mut job).await? {
        return Ok(job);
    }
    // Cancellation is durable intent and safe to repeat; partial successes still
    // go through ordinary result reconciliation instead of becoming free work.
    let remote = client.cancel(&remote.id).await?;
    job.remote = Some(remote);
    job.next_poll_at = Utc::now();
    save(store, &mut job).await?;
    refresh(store, provider, job).await
}

/// Runtime credentials stay in memory. After restart an authenticated get/results
/// request resumes polling using the saved job and re-supplied credentials.
pub fn spawn_polling<S: BatchJobStore + 'static>(
    store: std::sync::Arc<S>,
    provider: std::sync::Arc<dyn Provider>,
    id: Uuid,
) {
    type WorkerKey = (usize, Uuid);
    static ACTIVE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<WorkerKey>>> =
        std::sync::OnceLock::new();
    let key = (std::sync::Arc::as_ptr(&store) as usize, id);
    if !ACTIVE
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .insert(key)
    {
        return;
    }
    struct Guard(WorkerKey);
    impl Drop for Guard {
        fn drop(&mut self) {
            ACTIVE.get().unwrap().lock().unwrap().remove(&self.0);
        }
    }
    let guard = Guard(key);
    // Polling persists provider results long after the request that started
    // it, so it keeps that request's privacy policy.
    tokio::spawn(crate::privacy::inherit_request_policy(async move {
        let _guard = guard;
        loop {
            let job = match get(store.as_ref(), id).await {
                Ok(job) => job,
                Err(error) if error.status == 404 => break,
                Err(_) => {
                    tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                    continue;
                }
            };
            if job.finished() || job.status == "submission_unknown" {
                break;
            }
            if let Err(error) = refresh(store.as_ref(), provider.as_ref(), job).await {
                if let Ok(mut saved) = get(store.as_ref(), id).await {
                    if !saved.finished() {
                        saved.error = Some(format!(
                            "Polling/reconciliation failed (HTTP {}); will retry",
                            error.status
                        ));
                        let _ = save(store.as_ref(), &mut saved).await;
                    }
                }
            }
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        }
    }));
}

pub async fn submit_chat<S: BatchJobStore + 'static>(
    store: std::sync::Arc<S>,
    provider: std::sync::Arc<dyn Provider>,
    session: Option<Uuid>,
    request: &crate::ChatCompletionRequest,
) -> EngineResult<crate::provider::BatchSubmission> {
    let session = session
        .ok_or_else(|| EngineError::InvalidInput("batch work requires an engine session".into()))?;
    let id = request
        .extra
        .get("reincore_batch_idempotency_key")
        .and_then(Value::as_str)
        .map(Uuid::parse_str)
        .transpose()
        .map_err(|_| EngineError::InvalidInput("invalid batch idempotency key".into()))?
        .unwrap_or_else(Uuid::new_v4);

    let mut body = request.clone();
    for field in [
        "reincore_session_id",
        "reincore_external_session_id",
        "reincore_budget_cents",
        "reincore_batch_idempotency_key",
    ] {
        body.extra.remove(field);
    }

    let job = create(
        store.as_ref(),
        provider.as_ref(),
        session,
        id,
        &[BatchRequest {
            custom_id: "request-1".into(),
            body,
        }],
    )
    .await
    .map_err(|e| EngineError::Provider(format!("{} (engine batch {id})", e.message)))?;

    spawn_polling(store, provider, id);

    Ok(crate::provider::BatchSubmission {
        batch_id: id.to_string(),
        status: job.status,
    })
}

pub async fn resume_pending<S: BatchJobStore + 'static>(
    store: std::sync::Arc<S>,
    provider: std::sync::Arc<dyn Provider>,
) -> EngineResult<()> {
    let Ok(client) = provider.batch_client() else {
        return Ok(());
    };
    for job in store.pending_batch_jobs().await? {
        if job.scope == client.scope() {
            spawn_polling(store.clone(), provider.clone(), job.id);
        }
    }
    Ok(())
}

/// Conservative preflight factor, using both ordinary input and output rates.
/// Missing prices do not invent a universal batch discount.
pub fn prediction_factor(provider: &str, model: &str, input_tokens: u64) -> Option<f64> {
    let record = crate::model_library::find_model(provider, model)?;
    let schedule = record.pricing.as_ref()?;
    let rates = schedule.rates_at(input_tokens, Utc::now())?;
    let ratio = |regular: i64, batch: Option<i64>| -> Option<f64> {
        let batch = batch?;
        if regular == 0 {
            return (batch == 0).then_some(1.0);
        }
        Some(batch as f64 / regular as f64)
    };
    Some(ratio(rates.input, rates.batch_input)?.max(ratio(rates.output, rates.batch_output)?))
}
