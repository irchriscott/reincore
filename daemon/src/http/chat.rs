// Chat handlers coordinate request identity, provider execution, and durable usage
// accounting. Shared core helpers supply model routing, context preparation, and
// cache policy; live response bodies perform finalization as the stream is consumed.
use super::lifecycle::*;
use super::logging::*;
use super::request_group::*;
use super::*;
use tracing::Instrument;

/// Extract the JSON body from an `axum::response::Response`. Used by
/// the Responses API and Messages API handlers which delegate to
/// `chat_completions` / `chat_completions_single_provider` and need to
/// read the JSON response body.
/// Read a delegated chat response body.
///
/// A refusal (loop hard stop, upstream error) is returned as `Err` carrying the
/// original status and body, so wrappers forward it instead of reshaping an
/// error payload into a bogus successful completion.
pub(super) async fn extract_json_response(
    response: axum::response::Response,
) -> Result<(serde_json::Value, engine_core::ProviderResponseHeaders), axum::response::Response> {
    let status = response.status();
    let response_headers = response
        .headers()
        .iter()
        .filter(|(name, _)| engine_core::is_rate_limit_response_header(name.as_str()))
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_string(), value.to_string()))
        })
        .collect();
    let bytes = axum::body::to_bytes(response.into_body(), 10 * 1024 * 1024)
        .await
        .map_err(|e| {
            warn!("failed to read response body: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        })?;
    let json: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        warn!("failed to parse response body as JSON: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR.into_response()
    })?;
    if !status.is_success() {
        return Err((status, Json(json)).into_response());
    }
    Ok((json, response_headers))
}

pub(super) fn with_provider_rate_limit_headers(
    mut response: axum::response::Response,
    headers: &engine_core::ProviderResponseHeaders,
) -> axum::response::Response {
    for (name, value) in headers {
        if !engine_core::is_rate_limit_response_header(name) {
            continue;
        }
        let Ok(name) = axum::http::HeaderName::from_bytes(name.as_bytes()) else {
            continue;
        };
        let Ok(value) = axum::http::HeaderValue::from_str(value) else {
            continue;
        };
        response.headers_mut().insert(name, value);
    }
    response
}

/// Convert a non-streaming chat completion response into an SSE stream.
/// This is used when the client requests `stream: true` but the provider
/// only supports non-streaming responses. We split the response into
/// chunks: first a content chunk, then tool call chunks (if any), then a
/// final chunk with the finish reason and usage.
pub(super) fn chat_completion_to_sse_stream(
    response: serde_json::Value,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let id = response
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let model = response
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let created = response
        .get("created")
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| chrono::Utc::now().timestamp());

    let choice = response
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
        .cloned()
        .unwrap_or(json!({}));

    let message = choice.get("message").cloned().unwrap_or(json!({}));
    let finish_reason = choice
        .get("finish_reason")
        .and_then(|f| f.as_str())
        .unwrap_or("stop")
        .to_string();

    let content = message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    let tool_calls = message
        .get("tool_calls")
        .and_then(|t| t.as_array())
        .cloned();
    let usage = response.get("usage").cloned().unwrap_or(json!({}));
    let agent_health = response.get("reincore_agent_health").cloned();
    let request_group_id = response.get("reincore_request_group_id").cloned();
    let provider_retry_count = response.get("reincore_provider_retry_count").cloned();
    let task_id = response.get("reincore_task_id").cloned();
    let task_name = response.get("reincore_task_name").cloned();
    let task_type = response.get("reincore_task_type").cloned();
    let task_is_new = response.get("reincore_task_is_new").cloned();
    let task_context_messages_removed = response
        .get("reincore_task_context_messages_removed")
        .cloned();
    let request_group_step_count = response.get("reincore_request_group_step_count").cloned();
    let request_group_total_actual_cost_cents = response
        .get("reincore_request_group_total_actual_cost_cents")
        .cloned();
    let request_group_total_tokens = response.get("reincore_request_group_total_tokens").cloned();
    let predicted_cost_high_cents = response.get("reincore_predicted_cost_high_cents").cloned();
    let cost = response.get("reincore_cost").cloned();
    let cost_accuracy_bps = response
        .get("reincore_actual_to_predicted_high_ratio_bps")
        .cloned();
    let cache_hit_rate_bps = response.get("reincore_cache_hit_rate_bps").cloned();
    let semantic_cache_hit = response.get("reincore_cache_hit").cloned();
    let semantic_cache_provenance = response.get("reincore_semantic_cache_provenance").cloned();
    let semantic_retrieval_candidates = response
        .get("reincore_semantic_retrieval_candidates")
        .cloned();

    let tool_calls_validated = response
        .get("reincore_tool_calls_validated")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let replay_message = message
        .get("provider_state")
        .filter(|v| !v.is_null())
        .map(|_| message.clone());
    let stream = async_stream::stream! {
        if let Some(message) = replay_message {
            yield Ok(Event::default().data(json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": null}],
                "reincore_provider_state": message
            }).to_string()));
        }
        // If there's text content, send it as a single chunk.
        if !content.is_empty() {
            let chunk = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": null
                }]
            });
            yield Ok(Event::default().data(chunk.to_string()));
        }

        // If there are tool calls, send each one as a chunk.
        if let Some(tc_arr) = &tool_calls {
            for (i, tc) in tc_arr.iter().enumerate() {
                let function = tc.get("function").cloned().unwrap_or(json!({}));
                let tc_id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string();
                let name = function.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                let arguments = function.get("arguments").and_then(|a| a.as_str()).unwrap_or("").to_string();

                let chunk = json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "reincore_tool_calls_validated": tool_calls_validated,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [{
                                "index": i,
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments,
                                }
                            }]
                        },
                        "finish_reason": null
                    }]
                });
                yield Ok(Event::default().data(chunk.to_string()));
            }
        }

        // If content is empty and there are no tool calls, send an empty
        // content chunk so the client knows the role.
        if content.is_empty() && tool_calls.is_none() {
            let chunk = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "",
                    },
                    "finish_reason": null
                }]
            });
            yield Ok(Event::default().data(chunk.to_string()));
        }

        // Final chunk with finish reason and usage.
        let mut final_chunk = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }],
            "usage": usage
        });
        if let Some(agent_health) = agent_health {
            final_chunk["reincore_agent_health"] = agent_health;
        }
        if let Some(request_group_id) = request_group_id {
            final_chunk["reincore_request_group_id"] = request_group_id;
        }
        if let Some(provider_retry_count) = provider_retry_count {
            final_chunk["reincore_provider_retry_count"] = provider_retry_count;
        }
        if let Some(task_id) = task_id {
            final_chunk["reincore_task_id"] = task_id;
        }
        if let Some(task_name) = task_name {
            final_chunk["reincore_task_name"] = task_name;
        }
        if let Some(task_type) = task_type {
            final_chunk["reincore_task_type"] = task_type;
        }
        if let Some(task_is_new) = task_is_new {
            final_chunk["reincore_task_is_new"] = task_is_new;
        }
        if let Some(removed) = task_context_messages_removed {
            final_chunk["reincore_task_context_messages_removed"] = removed;
        }
        if let Some(request_group_step_count) = request_group_step_count {
            final_chunk["reincore_request_group_step_count"] = request_group_step_count;
        }
        if let Some(request_group_total_actual_cost_cents) = request_group_total_actual_cost_cents {
            final_chunk["reincore_request_group_total_actual_cost_cents"] = request_group_total_actual_cost_cents;
        }
        if let Some(request_group_total_tokens) = request_group_total_tokens {
            final_chunk["reincore_request_group_total_tokens"] = request_group_total_tokens;
        }
        if let Some(predicted_cost_high_cents) = predicted_cost_high_cents {
            final_chunk["reincore_predicted_cost_high_cents"] = predicted_cost_high_cents;
        }
        if let Some(cost) = cost {
            final_chunk["reincore_cost"] = cost;
        }
        if let Some(cost_accuracy_bps) = cost_accuracy_bps {
            final_chunk["reincore_actual_to_predicted_high_ratio_bps"] = cost_accuracy_bps;
        }
        if let Some(cache_hit_rate_bps) = cache_hit_rate_bps {
            final_chunk["reincore_cache_hit_rate_bps"] = cache_hit_rate_bps;
        }
        if let Some(cache_hit) = semantic_cache_hit {
            final_chunk["reincore_cache_hit"] = cache_hit;
        }
        if let Some(provenance) = semantic_cache_provenance {
            final_chunk["reincore_semantic_cache_provenance"] = provenance;
        }
        if let Some(candidates) = semantic_retrieval_candidates {
            final_chunk["reincore_semantic_retrieval_candidates"] = candidates;
        }
        yield Ok(Event::default().data(final_chunk.to_string()));

        // Send [DONE] marker.
        yield Ok(Event::default().data("[DONE]"));
    };

    stream
}

#[derive(Default)]
struct PartialStreamToolCall {
    id: String,
    call_type: String,
    name: String,
    arguments: String,
}

pub(super) struct LocalStreamContext {
    pub(super) claim_grounding: std::sync::Arc<engine_core::claim_grounding::ClaimGroundingConfig>,
    pub(super) action_policy: std::sync::Arc<engine_core::action_gate::ActionPolicy>,
    pub(super) detection: std::sync::Arc<engine_core::detection::DetectionConfig>,
    pub(super) detection_judge: Option<engine_core::detection::LlmJudge>,
    pub(super) db: std::sync::Arc<LocalDatabase>,
    pub(super) session_id: Option<Uuid>,
    pub(super) request_group_id: String,
    pub(super) task: engine_core::TaskAssignment,
    pub(super) task_context_messages_removed: usize,
    pub(super) original_request: ChatCompletionRequest,
    pub(super) provider_request: ChatCompletionRequest,
    pub(super) provider_label: String,
    pub(super) resolved_model: String,
    pub(super) model_tier: engine_core::ModelTier,
    pub(super) prefix_hash: String,
    pub(super) prediction: engine_core::CostPrediction,
    pub(super) cost_predictor: engine_core::CostPredictor,
    pub(super) classification: Option<engine_core::ClassificationTelemetry>,
    pub(super) batch_requested: bool,
    pub(super) semantic: Option<engine_core::SemanticCachePlan>,
    pub(super) semantic_candidates: Vec<engine_core::SemanticCacheCandidate>,
    pub(super) citation_attachments: Option<Vec<engine_core::FileContextAttachment>>,
    pub(super) initial_agent_health: engine_core::AgentHealthReport,
    pub(super) prewarm_expected: bool,
    pub(super) prewarm_provider: std::sync::Arc<dyn engine_core::Provider>,
    pub(super) prewarm_provider_config: engine_core::ProviderConfig,
    pub(super) provider_request_sent_at: chrono::DateTime<chrono::Utc>,
    pub(super) provider_request_started: std::time::Instant,
    pub(super) request_started: std::time::Instant,
    pub(super) provider_timer: engine_core::operational::Timer,
    pub(super) context_adjustment: engine_core::ContextAdjustment,
    /// Repeated read/search results this request replaced with a pointer.
    pub(super) deduped_tool_results: engine_core::ToolResultDedup,
    /// Loop configuration resolved for this request, reused at accounting so
    /// both stages agree on the thresholds and the off switch.
    pub(super) loop_breaker: CircuitBreaker,
    pub(super) provider_span: tracing::Span,
    pub(super) stream_span: tracing::Span,
}

fn attach_task_metadata(
    value: &mut serde_json::Value,
    task: &engine_core::TaskAssignment,
    context_messages_removed: usize,
) {
    value["reincore_task_id"] = json!(task.task_id);
    value["reincore_task_name"] = json!(task.task_name);
    value["reincore_task_type"] = json!(task.task_type.as_str());
    value["reincore_task_is_new"] = json!(task.is_new);
    value["reincore_task_context_messages_removed"] = json!(context_messages_removed);
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn record_batch_submission(
    db: &LocalDatabase,
    session_id: Option<Uuid>,
    request_group_id: &str,
    task: &engine_core::TaskAssignment,
    request: &ChatCompletionRequest,
    resolved_model: &str,
    model_tier: engine_core::ModelTier,
    prefix_hash: &str,
    batch_id: &str,
    predicted_cost_low_cents: i64,
    predicted_cost_high_cents: i64,
    classification: Option<engine_core::ClassificationTelemetry>,
    context_adjustment: engine_core::ContextAdjustment,
) -> Result<(), StatusCode> {
    let Some(session_id) = session_id else {
        return Ok(());
    };
    let now = chrono::Utc::now();
    let payload = engine_core::event::LlmCallPayload {
        tool_schema_hash: engine_core::tool_schema_hash(&request.tools).ok(),
        request_messages: engine_core::compact_audit_request_messages(&request.messages),
        response_message: ChatMessage {
            provider_state: None,
            role: "assistant".to_string(),
            content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        reasoning_content: None,
        provider_request_sent_at: Some(now),
        provider_response_received_at: Some(now),
        provider_duration_ms: Some(0),
        model: resolved_model.to_string(),
        model_tier: model_tier.as_str().to_string(),
        prefix_hash: Some(prefix_hash.to_string()),
        input_tokens: 0,
        output_tokens: 0,
        cached_prompt_tokens: 0,
        cache_creation_input_tokens: 0,
        batch_eligible: true,
        batch_submitted: true,
        batch_id: Some(batch_id.to_string()),
        predicted_cost_low_cents: Some(predicted_cost_low_cents),
        predicted_cost_high_cents: Some(predicted_cost_high_cents),
        cost_accuracy_bps: None,
        cache_hit_rate_bps: None,
        request_group_id: Some(request_group_id.to_string()),
        task_id: Some(task.task_id.clone()),
        task_name: Some(task.task_name.clone()),
        task_type: Some(task.task_type.as_str().to_string()),
        classification,
        context_adjustment: Some(context_adjustment),
        message_roles: Some(engine_core::user_content::annotate_message_roles(
            &request.messages,
        )),
        is_subagent: extra_bool(request, "reincore_is_subagent").unwrap_or(false),
        tool_results_deduped: None,
        exploration_redundancy: None,
        semantic_cache_similarity_bps: None,
        semantic_best_candidate_bps: None,
        semantic_best_candidate_entry_id: None,
    };
    db.append_event(
        session_id,
        SessionEventType::LlmCall,
        serde_json::to_value(payload).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
        Some(0),
    )
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(())
}

/// Build the Meta `muse-wonder` judge when any detection feature runs in
/// `llm` mode. Returns `None` - lexical checks everywhere - when no feature
/// opts in, the meta provider is not configured, or judge construction fails.
fn detection_judge(
    meta_provider: Option<std::sync::Arc<dyn engine_core::Provider>>,
    detection: &engine_core::detection::DetectionConfig,
) -> Option<engine_core::detection::LlmJudge> {
    if !detection.uses_llm() {
        return None;
    }
    let Some(provider) = meta_provider else {
        warn!("detection llm mode is configured but the meta provider is not available; using lexical checks");
        return None;
    };
    match engine_core::detection::LlmJudge::new(provider.config(), detection) {
        Ok(judge) => Some(judge),
        Err(error) => {
            warn!(%error, "detection llm judge is unavailable; using lexical checks");
            None
        }
    }
}

async fn resolve_and_scope_task(
    db: &LocalDatabase,
    session_id: Option<Uuid>,
    request_group_id: &str,
    request: &mut ChatCompletionRequest,
    task_decisions_enabled: bool,
    claim_grounding: &engine_core::claim_grounding::ClaimGroundingConfig,
    detection: &engine_core::detection::DetectionConfig,
    judge: Option<&engine_core::detection::LlmJudge>,
) -> Result<(engine_core::TaskAssignment, usize), StatusCode> {
    let mut events = match session_id {
        Some(id) => {
            const TASK_LOOKBACK_EVENTS: i64 = 4096;
            db.read_recent_task_events(id, TASK_LOOKBACK_EVENTS as usize)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
                .into_iter()
                .map(engine_core::TaskEventProjection::into_event)
                .collect()
        }
        None => Vec::new(),
    };
    if let Some(id) = session_id {
        if let Some(seed) = db
            .read_latest_continuation(id, "")
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .and_then(engine_core::continuation::task_resolution_seed)
        {
            events.push(seed);
        }
    }
    let task = engine_core::resolve_task_auto(
        session_id,
        request_group_id,
        request,
        &events,
        detection.task_boundary.mode,
        detection.task_type.mode,
        judge,
    )
    .await;
    let project_scope = match session_id.filter(|_| task_decisions_enabled) {
        Some(id) => {
            let session = db
                .get_session_async(id)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
                .ok_or(StatusCode::NOT_FOUND)?;
            serde_json::json!([session.project_id, session.project_path]).to_string()
        }
        None => "stateless".to_string(),
    };
    engine_core::task_decisions::prepare(
        db,
        session_id,
        &task,
        request_group_id,
        &project_scope,
        request,
        task_decisions_enabled,
    )
    .await
    .map_err(|error| match error {
        engine_core::EngineError::InvalidInput(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    })?;

    let removed = if session_id.is_some() {
        engine_core::scope_request_to_task(request, &task)
    } else {
        0
    };
    engine_core::continuation::prepare(db, session_id, &task, request_group_id, request)
        .await
        .map_err(|error| match error {
            engine_core::EngineError::InvalidInput(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        })?;
    engine_core::claim_grounding::prepare(db, session_id, request, claim_grounding).await;
    engine_core::task_decisions::attach(request);
    Ok((task, removed))
}

/// Forward normalized provider chunks immediately while retaining just enough
/// state to perform accounting and loop detection before the terminal event.
/// This changes streaming latency from full-generation time to provider
/// time-to-first-token without sacrificing the existing audit trail.
pub(super) fn live_chat_completion_to_sse_stream(
    upstream: engine_core::ProviderStream,
    context: LocalStreamContext,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let mut upstream = upstream.chunks;
    async_stream::stream! {
        let response_id = format!("rc-chatcmpl-{}", Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();
        let mut content = String::new();
        let mut reasoning_content = String::new();
        let mut provider_state = None;
        // Group streamed tool-call fragments by provider index because multiple calls
        // can arrive interleaved. Reconstruct complete calls for validation and audit
        // after the upstream stream finishes.
        let mut tool_calls = BTreeMap::<usize, PartialStreamToolCall>::new();
        let mut tool_buffer_bytes = 0usize;
        let mut finish_reason: Option<String> = None;
        let mut usage: Option<engine_core::TokenUsage> = None;
        let mut retry_usage = engine_core::TokenUsage::default();
        let mut all_retry_usage_reported = true;
        let mut stream_failed = false;
        let stream_started = std::time::Instant::now();
        let mut upstream_wait_duration = std::time::Duration::ZERO;
        let mut first_chunk_ms = None;
        let mut output_timing = engine_core::operational::OutputTiming::new(context.request_started, context.provider_request_started);
        let provider_timer = context.provider_timer;
        let mut chunk_count = 0_u64;
        let mut empty_response_retries = 0usize;

        // Emit the role as soon as the upstream has accepted the request.
        yield Ok(Event::default().data(json!({
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": context.resolved_model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
        }).to_string()));

        // Live deltas are client-bound: text is emitted promptly at
        // whitespace boundaries, but a trailing partial token (and any
        // in-progress PEM block) is held back so a secret split across
        // chunks is redacted whole rather than leaked in pieces.
        let privacy_policy = engine_core::privacy::current_policy();
        let mut pending_redaction = String::new();
        const STREAM_REDACTION_FORCE_FLUSH: usize = 8192;

        'provider_attempt: loop {
            'provider_stream: loop {
                let upstream_wait_started = std::time::Instant::now();
                let next = upstream.next().await;
                upstream_wait_duration = upstream_wait_duration.saturating_add(upstream_wait_started.elapsed());
                let Some(next) = next else { break };
                chunk_count = chunk_count.saturating_add(1);
                first_chunk_ms.get_or_insert_with(|| {
                    u64::try_from(context.provider_request_started.elapsed().as_millis())
                        .unwrap_or(u64::MAX)
                });
                let chunk = match next {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        engine_core::operational::increment(engine_core::operational::Counter::ProviderStreamErrors, 1);
                        warn!(provider = context.provider_label, %error, "provider stream failed");
                        engine_core::telemetry::record_error(
                            &context.provider_span,
                            "provider_stream_error",
                            &engine_core::privacy::current_policy().sanitize_for_logging(&error.to_string()),
                        );
                        engine_core::telemetry::record_error(
                            &context.stream_span,
                            "provider_stream_error",
                            &engine_core::privacy::current_policy().sanitize_for_logging(&error.to_string()),
                        );
                        yield Ok(Event::default().data(json!({
                            "error": {
                                "message": privacy_policy.sanitize_for_logging(&error.to_string()),
                                "type": "provider_stream_error"
                            }
                        }).to_string()));
                        stream_failed = true;
                        break;
                    }
                };

                output_timing.chunk(&chunk);
                if let Some(state) = chunk.provider_state { provider_state = Some(state); }
                if let Some(delta) = chunk.delta_content {
                    content.push_str(&delta);
                    pending_redaction.push_str(&delta);
                    let pem_in_progress = pending_redaction.contains("-----BEGIN")
                        && !pending_redaction.contains("-----END");
                    // Only a credential-shaped trailing token is held: prose
                    // streams through immediately, while a secret arriving
                    // in pieces is emitted only once complete and redacted.
                    let tail_start = pending_redaction
                        .rfind(|c: char| c.is_whitespace())
                        .map(|index| index + 1)
                        .unwrap_or(0);
                    let tail = &pending_redaction[tail_start..];
                    let tail_is_risky = !tail.is_empty()
                        && (tail.contains('=')
                            || tail.contains(':')
                            || tail.len() >= 20
                                && tail.bytes().any(|b| b.is_ascii_digit())
                                && tail.bytes().any(|b| b.is_ascii_alphabetic()));
                    let flush_end = if pem_in_progress {
                        None
                    } else if pending_redaction.len() > STREAM_REDACTION_FORCE_FLUSH {
                        let mut cut = pending_redaction.len() - 1024;
                        while !pending_redaction.is_char_boundary(cut) {
                            cut -= 1;
                        }
                        Some(cut)
                    } else if tail_is_risky {
                        Some(tail_start)
                    } else {
                        Some(pending_redaction.len())
                    };
                    if let Some(cut) = flush_end.filter(|cut| *cut > 0) {
                        let head: String = pending_redaction.drain(..cut).collect();
                        let safe_delta = privacy_policy.redact_text(&head).value;
                        if !safe_delta.is_empty() {
                            yield Ok(Event::default().data(json!({
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": context.resolved_model,
                                "choices": [{"index": 0, "delta": {"content": safe_delta}, "finish_reason": null}]
                            }).to_string()));
                        }
                    }
                }
                if let Some(delta) = chunk.delta_reasoning_content {
                    reasoning_content.push_str(&delta);
                }

                if let Some(deltas) = chunk.delta_tool_calls {
                    for delta in deltas {
                        tool_buffer_bytes = tool_buffer_bytes.saturating_add(delta.id.as_ref().map_or(0, String::len))
                            .saturating_add(delta.function_name.as_ref().map_or(0, String::len))
                            .saturating_add(delta.function_arguments.as_ref().map_or(0, String::len));
                        if tool_buffer_bytes > engine_core::action_gate::MAX_TOOL_ARGUMENT_BYTES
                            || (!tool_calls.contains_key(&delta.index) && tool_calls.len() >= engine_core::action_gate::MAX_TOOL_CALLS) {
                            yield Ok(Event::default().data(json!({"error":{"type":"action_validation_failed","message":"Tool call buffer limit exceeded"}}).to_string()));
                            stream_failed = true;
                            break 'provider_stream;
                        }

                        let accumulated = tool_calls.entry(delta.index).or_default();
                        if let Some(id) = delta.id.as_deref() {
                            accumulated.id = id.to_string();
                        }
                        if let Some(call_type) = delta.call_type.as_deref() {
                            accumulated.call_type = call_type.to_string();
                        }
                        delta.merge_function_fields(&mut accumulated.name, &mut accumulated.arguments);

                    }
                }
                if chunk.finish_reason.is_some() {
                    finish_reason = chunk.finish_reason;
                }
                if chunk.usage.is_some() {
                    usage = chunk.usage;
                }
            }

            tool_calls.retain(|_, call| {
                !engine_core::is_terminal_notification_tool(&call.name)
            });
            if !stream_failed
                && content.is_empty()
                && tool_calls.is_empty()
                && empty_response_retries < engine_core::EMPTY_PROVIDER_RESPONSE_RETRY_LIMIT
            {
                empty_response_retries += 1;
                warn!(
                    provider = context.provider_label,
                    model = %context.resolved_model,
                    attempt = empty_response_retries + 1,
                    "provider returned no visible output; retrying provider stream"
                );
                match context.prewarm_provider.complete_stream(&context.provider_request).await {
                    Ok(retry) => {
                        match usage.take() {
                            Some(attempt_usage) => retry_usage = retry_usage.saturating_add(attempt_usage),
                            None => all_retry_usage_reported = false,
                        }
                        upstream = retry.chunks;
                        reasoning_content.clear();
                        provider_state = None;
                        finish_reason = None;
                        tool_buffer_bytes = 0;
                        continue 'provider_attempt;
                    }
                    Err(error) => {
                        engine_core::operational::increment(engine_core::operational::Counter::ProviderStreamErrors, 1);
                        warn!(provider = context.provider_label, %error, "empty-response provider retry failed");
                        yield Ok(Event::default().data(json!({
                            "error": {"message": privacy_policy.sanitize_for_logging(&error.to_string()), "type": "provider_stream_error"}
                        }).to_string()));
                        stream_failed = true;
                    }
                }
            }
            break 'provider_attempt;
        }

        if !pending_redaction.is_empty() {
            let safe_tail = privacy_policy.redact_text(&pending_redaction).value;
            pending_redaction.clear();
            if !safe_tail.is_empty() {
                yield Ok(Event::default().data(json!({
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": context.resolved_model,
                    "choices": [{"index": 0, "delta": {"content": safe_tail}, "finish_reason": null}]
                }).to_string()));
            }
        }

        drop(provider_timer);
        let _finalize_timer = engine_core::operational::Timer::new(engine_core::operational::Metric::GatewayFinalize);
        let stream_duration = stream_started.elapsed();
        let stream_ms = u64::try_from(stream_duration.as_millis()).unwrap_or(u64::MAX);
        let upstream_wait_ms = u64::try_from(upstream_wait_duration.as_millis()).unwrap_or(u64::MAX);
        engine_core::operational::observe(engine_core::operational::Metric::ProviderStreamUpstreamWait, upstream_wait_duration);
        engine_core::operational::observe(engine_core::operational::Metric::ProviderStreamDownstreamWait, stream_duration.saturating_sub(upstream_wait_duration));
        context.stream_span.record(
            "reincore.provider_first_chunk_ms",
            first_chunk_ms.map(|value| value as i64).unwrap_or(-1),
        );
        context
            .stream_span
            .record("reincore.provider_stream_ms", stream_ms as i64);
        context.stream_span.record(
            "reincore.provider_stream_upstream_wait_ms",
            upstream_wait_ms as i64,
        );
        context.stream_span.record(
            "reincore.provider_stream_downstream_wait_ms",
            stream_ms.saturating_sub(upstream_wait_ms) as i64,
        );
        context
            .stream_span
            .record("reincore.provider_stream_chunks", chunk_count as i64);
        context.stream_span.record(
            "reincore.provider_stream_content_chars",
            content.chars().count() as i64,
        );
        context.stream_span.record(
            "reincore.provider_stream_reasoning_chars",
            reasoning_content.chars().count() as i64,
        );
        context.stream_span.record(
            "reincore.provider_stream_tool_calls",
            tool_calls.len() as i64,
        );
        drop(context.stream_span);

        // An upstream error has already been sent to the client. End the stream here
        // without recording normal completion accounting or caching a partial answer;
        // the [DONE] marker closes delivery and does not indicate a successful call.
        if !tool_calls.is_empty() && !matches!(finish_reason.as_deref(), Some("stop" | "tool_calls")) {
            yield Ok(Event::default().data(json!({"error":{"type":"action_validation_failed","message":"Incomplete tool call response was withheld"}}).to_string()));
            stream_failed = true;
        }
        if stream_failed {
            yield Ok(Event::default().data("[DONE]"));
            return;
        }

        // Warn when the provider returned a completely empty response - no
        // content, no tool calls.  This is the upstream cause of the VS Code
        // "Sorry, no response was returned" UX: the SSE stream completes
        // successfully but never emits any visible content.  Logging here
        // makes the silent failure observable on the server side.
        if content.is_empty() && tool_calls.is_empty() {
            warn!(
                provider = context.provider_label,
                model = %context.resolved_model,
                finish_reason = ?finish_reason,
                "provider returned an empty response (no content, no tool calls) - \
                 client will see no output",
            );
        }

        let completed_tool_calls = (!tool_calls.is_empty()).then(|| {
            tool_calls
                .into_values()
                .map(|call| engine_core::ToolCall {
                    id: call.id,
                    call_type: if call.call_type.is_empty() { "function".to_string() } else { call.call_type },
                    function: engine_core::FunctionCall { name: call.name, arguments: call.arguments },
                })
                .collect::<Vec<_>>()
        });
        let mut response_message = ChatMessage {
            provider_state: None,
            role: "assistant".to_string(),
            content: (!content.is_empty()).then_some(engine_core::MessageContent::text(content)),
            tool_calls: completed_tool_calls,
            tool_call_id: None,
            name: None,
        };
        response_message.provider_state = provider_state.map(|state| state.bind(&response_message).for_request_group(&context.request_group_id));
        let injection_report = engine_core::agent_health::scan_prompt_injection_auto(
            &context.provider_request,
            context.detection.prompt_injection.mode,
            context.detection_judge.as_ref(),
        )
        .await;
        let preflight_health = engine_core::action_gate::gate_response_with_injection_report(
            &context.provider_request,
            &mut response_message,
            &context.action_policy,
            injection_report,
        );
        let was_denied = preflight_health.should_stop;
        let (preflight_health, recovery_text) = engine_core::action_recovery::recover(
            &context.provider_request, &mut response_message, preflight_health, &context.action_policy,
            context.prewarm_provider.as_ref(), &context.request_group_id, context.session_id.map(|id| (
                context.db.as_ref() as &dyn engine_core::EventStore,
                context.db.as_ref() as &dyn engine_core::accounting::AccountingStore, id)),
        ).await;
        if was_denied && !preflight_health.should_stop {
            finish_reason = Some(engine_core::finish_reason_for(&response_message).into());
        }
        if let Some(text) = recovery_text {
            yield Ok(Event::default().data(json!({"id":response_id,"object":"chat.completion.chunk",
                "created":created,"model":context.resolved_model,
                "choices":[{"index":0,"delta":{"content":text},"finish_reason":null}]}).to_string()));
        }
        let action_blocked = preflight_health.should_stop;
        let effective_finish_reason = if action_blocked { "stop".to_string() } else { engine_core::effective_finish_reason(
            &response_message,
            finish_reason.as_deref(),
        ) };

        if let Some(key) = engine_core::request_group_key(context.session_id, &context.original_request) {
            engine_core::mark_group_continues(&key, effective_finish_reason == "tool_calls");
        }

        let health_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Health);
        let mut agent_health = context.initial_agent_health;
        agent_health.merge(preflight_health);
        agent_health.merge(engine_core::claim_grounding::evaluate(
            &context.provider_request, &response_message, &context.claim_grounding,
            context.prewarm_provider.as_ref(),
            context.session_id.map(|id| (context.db.as_ref() as &dyn engine_core::accounting::AccountingStore, id)),
        ).await);

        if let Some(attachments) = context.citation_attachments.as_deref() {
            agent_health.merge(engine_core::analyze_response_citations_with_request(
                &response_message,
                attachments,
                &context.provider_request.messages,
            ));
        }
        engine_core::apply_intervention_policy(&mut agent_health);
        health_span.record("reincore.health_findings", agent_health.findings.len() as i64);
        drop(health_span);
        let usage_reported = usage.is_some() && all_retry_usage_reported;
        let actual_usage = usage.unwrap_or(engine_core::TokenUsage {
            prompt_tokens: context.prediction.input_tokens,
            completion_tokens: context.prediction.output_tokens,
            total_tokens: context.prediction.input_tokens.saturating_add(context.prediction.output_tokens),
            cached_prompt_tokens: 0,
            cache_creation_input_tokens: 0,
            ..Default::default()
        }).saturating_add(retry_usage);
        let cost = context.cost_predictor.estimate_usage(&actual_usage);
        let cost = if usage_reported { cost } else { engine_core::pricing::CostEstimate::unknown(&cost.provider, &cost.model, "provider did not return usage") };
        let accuracy_bps = cost.accuracy_bps(context.prediction.high_cents);
        let cache_hit_bps = engine_core::cache_hit_rate_bps(
            actual_usage.cached_prompt_tokens,
            actual_usage.prompt_tokens,
        );

        let group_snapshot = record_request_group_step(
            &context.request_group_id,
            actual_usage.prompt_tokens,
            actual_usage.completion_tokens,
            cost.amount_nano_usd,
        ).await;
        let provider_response_received_at = chrono::Utc::now();
        let provider_duration_ms = u64::try_from(context.provider_request_started.elapsed().as_millis())
            .unwrap_or(u64::MAX);
        context.provider_span.record("reincore.provider_ms", provider_duration_ms as i64);
        context.provider_span.record("reincore.provider", context.provider_label.as_str());
        context.provider_span.record("reincore.model", context.resolved_model.as_str());
        context.provider_span.record("gen_ai.usage.input_tokens", actual_usage.prompt_tokens as i64);
        context.provider_span.record("gen_ai.usage.output_tokens", actual_usage.completion_tokens as i64);

        let payload = engine_core::event::LlmCallPayload {
            tool_schema_hash: engine_core::tool_schema_hash(&context.provider_request.tools).ok(),
            request_messages: engine_core::compact_audit_request_messages(
                &context.provider_request.messages,
            ),
            response_message: response_message.clone(),
            reasoning_content: (!reasoning_content.is_empty()).then_some(reasoning_content),
            provider_request_sent_at: Some(context.provider_request_sent_at),
            provider_response_received_at: Some(provider_response_received_at),
            provider_duration_ms: Some(provider_duration_ms),
            model: context.resolved_model.clone(),
            model_tier: context.model_tier.as_str().to_string(),
            prefix_hash: Some(context.prefix_hash),
            input_tokens: actual_usage.prompt_tokens,
            output_tokens: actual_usage.completion_tokens,
            cached_prompt_tokens: actual_usage.cached_prompt_tokens,
            cache_creation_input_tokens: actual_usage.cache_creation_input_tokens,
            batch_eligible: context.batch_requested,
            batch_submitted: false,
            batch_id: None,
            predicted_cost_low_cents: Some(context.prediction.low_cents),
            predicted_cost_high_cents: Some(context.prediction.high_cents),
            // Set by `record_completion_accounting` once the actual cost is known;
            // initialized to None here so the struct literal is exhaustive.
            cost_accuracy_bps: None,
            cache_hit_rate_bps: None,
            classification: context.classification.clone(),
            request_group_id: Some(context.request_group_id.clone()),
            task_id: Some(context.task.task_id.clone()),
            task_name: Some(context.task.task_name.clone()),
            task_type: Some(context.task.task_type.as_str().to_string()),
            context_adjustment: Some(context.context_adjustment.clone()),
            message_roles: Some(engine_core::user_content::annotate_message_roles(
                &context.provider_request.messages,
            )),
            is_subagent: extra_bool(&context.original_request, "reincore_is_subagent")
                .unwrap_or(false),
            tool_results_deduped: Some(context.deduped_tool_results.new_replacements),
            // Stamped by `record_completion_accounting`, which scores the window.
            exploration_redundancy: None,
            semantic_cache_similarity_bps: None,
            semantic_best_candidate_bps: engine_core::best_candidate_similarity_bps(&context.semantic_candidates),
            semantic_best_candidate_entry_id: engine_core::best_candidate_entry_id(&context.semantic_candidates),
        };

        let account_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Account);
        let accounting_result = record_completion_accounting(
            context.db.clone(),
            context.session_id,
            payload,
            &agent_health,
            cost.clone(),
            actual_usage.prompt_tokens,
            actual_usage.completion_tokens,
            accuracy_bps,
            cache_hit_bps,
            engine_core::task_decisions::completion_context(&context.provider_request),
            &context.loop_breaker,
        ).await;
        if accounting_result.is_err() {
            yield Ok(Event::default().event("error").data(json!({
                "error": {"type": "accounting_failed", "message": "Usage persistence failed; completion is not confirmed."}
            }).to_string()));
            return;
        }
        engine_core::telemetry::record_cost(&account_span, &cost);
        account_span.record(
            "reincore.actual_to_predicted_high_ratio_bps",
            accuracy_bps.map(|value| value as i64).unwrap_or(-1),
        );
        drop(account_span);

        trigger_prewarm_if_tier_changed(
            context.db.clone(),
            context.session_id,
            context.prewarm_expected,
            context.prewarm_provider,
            context.prewarm_provider_config,
        );

        let cached_message = serde_json::to_value(&response_message).unwrap_or_else(|_| json!({}));
        if let Some(plan) = context.semantic.filter(|_| engine_core::chat_message_has_visible_output(&response_message)) {
            store_semantic_in_background(context.db.clone(), plan, cached_message);
        }

        // Emit the final finish reason and usage only after required accounting has
        // completed. Earlier text deltas may already have reached the client.
        let mut final_chunk = json!({
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": context.resolved_model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": effective_finish_reason}],
            "usage": {
                "prompt_tokens": actual_usage.prompt_tokens,
                "completion_tokens": actual_usage.completion_tokens,
                "total_tokens": actual_usage.total_tokens,
                "prompt_tokens_details": {
                    "cached_tokens": actual_usage.cached_prompt_tokens
                }
            },
            "reincore_tool_calls_validated": !agent_health.should_stop,
            "reincore_agent_health": agent_health,
            "reincore_predicted_cost_high_cents": context.prediction.high_cents,
            "reincore_actual_to_predicted_high_ratio_bps": accuracy_bps,
            "reincore_cost": cost,
            "reincore_cache_hit_rate_bps": cache_hit_bps,
            "reincore_provider_retry_count": empty_response_retries,
        });
        if !context.semantic_candidates.is_empty() {
            final_chunk["reincore_semantic_retrieval_candidates"] =
                serde_json::to_value(&context.semantic_candidates).unwrap_or_default();
        }

        attach_request_group_metadata(&mut final_chunk, &group_snapshot);
        attach_task_metadata(
            &mut final_chunk,
            &context.task,
            context.task_context_messages_removed,
        );
        link_response_to_request_group(&response_id, &context.request_group_id).await;
        print_request_group_rollup(&group_snapshot);

        final_chunk["reincore_action_blocked"] = json!(action_blocked);
        if action_blocked {
            let blocked_message = format!(
                "\n{}",
                engine_core::action_gate::blocked_action_message(&agent_health)
            );
            yield Ok(Event::default().data(json!({
                "id":response_id,"object":"chat.completion.chunk","created":created,"model":context.resolved_model,
                "choices":[{"index":0,"delta":{"content":blocked_message},"finish_reason":null}],
                "reincore_action_blocked":true,"reincore_agent_health":agent_health
            }).to_string()));
        } else {
            if response_message.provider_state.is_some() {
                yield Ok(Event::default().data(json!({
                    "id":response_id,"object":"chat.completion.chunk","created":created,"model":context.resolved_model,
                    "choices":[{"index":0,"delta":{},"finish_reason":null}],"reincore_provider_state":response_message
                }).to_string()));
            }
            if let Some(calls) = &response_message.tool_calls {
                let calls: Vec<_> = calls.iter().enumerate().map(|(index, call)| {
                    let mut value = serde_json::to_value(call).unwrap(); value["index"] = json!(index); value
                }).collect();
                yield Ok(Event::default().data(json!({
                    "id":response_id,"object":"chat.completion.chunk","created":created,"model":context.resolved_model,
                    "reincore_tool_calls_validated":true,
                    "choices":[{"index":0,"delta":{"tool_calls":calls},"finish_reason":null}]
                }).to_string()));
            }
        }
        yield Ok(Event::default().data(final_chunk.to_string()));
        yield Ok(Event::default().data("[DONE]"));
    }
}

/// Read the server-side allow-list for narrowly scoped immutable workflows.
/// The cache plan below still requires a source-data version and fingerprints
/// the chosen workflow; a caller-provided embedding alone grants no exception.
fn immutable_semantic_workflows() -> Vec<String> {
    std::env::var("REINCORE_SEMANTIC_CACHE_IMMUTABLE_WORKFLOWS")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

fn build_semantic_cache_plan(
    req: &ChatCompletionRequest,
    caller_embedding: Option<Vec<f32>>,
    scope: Option<engine_core::SemanticCacheScope>,
    provider: &dyn engine_core::Provider,
    resolved_model: &str,
    prefix_hash: &str,
) -> Result<Option<engine_core::SemanticCachePlan>, StatusCode> {
    let app_config_version =
        std::env::var("REINCORE_CONFIG_VERSION").unwrap_or_else(|_| "default".to_string());
    let config_version =
        engine_core::semantic_provider_config_version(provider.config(), &app_config_version)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    engine_core::semantic_cache_plan(
        req,
        caller_embedding,
        scope,
        provider.label(),
        resolved_model,
        prefix_hash,
        env!("CARGO_PKG_VERSION"),
        &config_version,
        &immutable_semantic_workflows(),
    )
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

/// Build the OpenAI-shape completion response for a semantic-cache hit.
fn semantic_hit_response(
    response_id: String,
    model: &str,
    session_id: Option<Uuid>,
    cached_response: serde_json::Value,
) -> serde_json::Value {
    let finish_reason = engine_core::finish_reason_for_json(&cached_response);
    json!({
        "id": response_id,
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp(),
        "model": model,
        "reincore_cache_hit": "semantic",
        "reincore_session_id": session_id,
        "choices": [{
            "index": 0,
            "message": cached_response,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    })
}

/// Cache hits still represent completed agent work. Persist an LLM-shaped
/// event so task projection, history, and request-group accounting do not lose
/// the request merely because no provider call was necessary.
async fn record_semantic_cache_hit(
    db: &LocalDatabase,
    session_id: Option<Uuid>,
    request_group_id: &str,
    task: &engine_core::TaskAssignment,
    request: &ChatCompletionRequest,
    cached_response: &serde_json::Value,
    resolved_model: &str,
    prefix_hash: &str,
    similarity_bps: i64,
) -> Result<(), StatusCode> {
    let Some(session_id) = session_id else {
        return Ok(());
    };
    let response_message = serde_json::from_value(cached_response.clone())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let now = chrono::Utc::now();
    let payload = engine_core::event::LlmCallPayload {
        tool_schema_hash: engine_core::tool_schema_hash(&request.tools).ok(),
        request_messages: engine_core::compact_audit_request_messages(&request.messages),
        response_message,
        reasoning_content: None,
        provider_request_sent_at: Some(now),
        provider_response_received_at: Some(now),
        provider_duration_ms: Some(0),
        model: resolved_model.to_string(),
        model_tier: "semantic_cache".to_string(),
        prefix_hash: Some(prefix_hash.to_string()),
        input_tokens: 0,
        output_tokens: 0,
        cached_prompt_tokens: 0,
        cache_creation_input_tokens: 0,
        batch_eligible: false,
        batch_submitted: false,
        batch_id: None,
        predicted_cost_low_cents: Some(0),
        predicted_cost_high_cents: Some(0),
        cost_accuracy_bps: None,
        cache_hit_rate_bps: None,
        request_group_id: Some(request_group_id.to_string()),
        task_id: Some(task.task_id.clone()),
        task_name: Some(task.task_name.clone()),
        task_type: Some(task.task_type.as_str().to_string()),
        classification: None,
        context_adjustment: None,
        message_roles: Some(engine_core::user_content::annotate_message_roles(
            &request.messages,
        )),
        is_subagent: extra_bool(request, "reincore_is_subagent").unwrap_or(false),
        tool_results_deduped: None,
        exploration_redundancy: None,
        semantic_cache_similarity_bps: Some(similarity_bps),
        semantic_best_candidate_bps: None,
        semantic_best_candidate_entry_id: None,
    };
    db.append_event(
        session_id,
        SessionEventType::LlmCall,
        serde_json::to_value(payload).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
        Some(0),
    )
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(())
}

/// Store a completed response in the semantic cache without blocking the
/// caller: cache write-back is not needed to build the response.
fn store_semantic_in_background(
    db: std::sync::Arc<LocalDatabase>,
    plan: engine_core::SemanticCachePlan,
    message_json: serde_json::Value,
) {
    let entry = plan.entry(message_json, chrono::Utc::now());
    tokio::spawn(async move {
        let _ = db.put_semantic(entry).await;
    });
}

/// Persist post-response accounting for a completed LLM call and report
/// whether a loop intervention was scheduled. The event-log write and loop
/// inspection run synchronously because the *next* request reads them
/// (pending interventions). Durable accounting completes before success.
#[allow(clippy::too_many_arguments)]
async fn record_completion_accounting(
    db: std::sync::Arc<LocalDatabase>,
    session_id: Option<Uuid>,
    mut payload: engine_core::event::LlmCallPayload,
    agent_health: &engine_core::AgentHealthReport,
    cost: engine_core::pricing::CostEstimate,
    prompt_tokens: u64,
    completion_tokens: u64,
    accuracy_bps: Option<i64>,
    cache_hit_bps: Option<i64>,
    decision_context: Option<serde_json::Value>,
    breaker: &CircuitBreaker,
) -> Result<bool, StatusCode> {
    let Some(id) = session_id else {
        return Ok(false);
    };

    // Persist the accuracy ratio on the event payload for replay/audit. The
    // caller already computed it from the predicted high bound and the
    // actual billed cost so the helper stays a single source of truth.
    payload.cost_accuracy_bps = accuracy_bps;
    payload.cache_hit_rate_bps = cache_hit_bps;
    // Score the window before the payload is serialized into the accounting
    // record, so the per-turn redundancy is stamped onto the same `LlmCall`
    // event rather than needing a second read or a second event.
    let observation = breaker
        .observe_with_pending(db.as_ref(), id, Some(payload.pending_audit()))
        .await
        .unwrap_or(None);
    payload.exploration_redundancy = observation.as_ref().map(|found| found.redundancy());
    // Captured before `payload` is moved into the accounting record below.
    let payload_deduped = payload.tool_results_deduped;
    use engine_core::accounting::{AccountingStore, CompletionAccounting};
    // Commit recoverable usage before settlement; do not return success on a
    // persistence failure. Settlement owns only accounting columns, not lifecycle.
    payload.input_tokens = prompt_tokens;
    payload.output_tokens = completion_tokens;
    let mut record = CompletionAccounting {
        cost: Some(cost.clone()),
        purpose: None,
        recorded_at: chrono::Utc::now(),
        id: Uuid::new_v4(),
        session_id: id,
        payload: serde_json::to_value(payload).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
        cost_cents: cost.display_cents(),
    };
    record.payload["task_decision_context"] = decision_context.unwrap_or(serde_json::Value::Null);
    db.record_accounting(&record).await.map_err(|error| {
        tracing::error!(accounting_id=%record.id, "completion accounting failed: {error}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    if let Err(error) = engine_core::task_decisions::observe_completion(db.as_ref(), &record).await
    {
        tracing::warn!(%error, "task decision observation write failed after durable accounting");
    }
    // Both advisory signals ride on this turn's single health event: repetition
    // that dedup absorbed (which never reaches the detector) and repetition the
    // detector scored but did not act on.
    let mut health = agent_health.clone();
    health
        .findings
        .extend(engine_core::redundant_lookup_finding(
            payload_deduped.unwrap_or(0),
        ));
    health.findings.extend(
        observation
            .as_ref()
            .and_then(|found| found.health_finding()),
    );
    let _ = engine_core::log_agent_health(db.as_ref(), id, &health).await;
    // Reuses the observation scored above: at or above threshold this writes the
    // intervention the next turn acts on, below it an advisory health finding.
    let loop_intervention_scheduled = match observation {
        Some(observation) => breaker
            .log_observation(db.as_ref(), id, &observation)
            .await
            .unwrap_or(false),
        None => false,
    };
    Ok(loop_intervention_scheduled)
}

/// Trigger a predictive pre-warming pass in the background if the session
/// just transitioned into a degradation tier (≥ Tier 2). This is called
/// after `record_completion_accounting` has committed the spend/tier update.
///
/// Pre-warming runs in the background. Usage is durably accounted and
/// persistence failures are reported explicitly. The `PreWarmManager` issues a minimal
/// 1-token request to the Lite model (warming the HTTP connection pool
/// and prompt-cache prefix) and pre-fetches likely-needed tool results
/// from the RAG store based on session shape similarity.
fn trigger_prewarm_if_tier_changed(
    db: std::sync::Arc<LocalDatabase>,
    session_id: Option<Uuid>,
    prewarm_expected: bool,
    provider: std::sync::Arc<dyn engine_core::Provider>,
    provider_config: engine_core::ProviderConfig,
) {
    let Some(id) = session_id else {
        return;
    };

    // Only pre-warm if the session was below Tier 2 before this call -
    // we want to warm on the *transition*, not on every call while already
    // degraded (the connection is already warm by then).
    if !prewarm_expected {
        return;
    }

    tokio::spawn(async move {
        // Accounting is intentionally off the response hot path. Wait for its
        // session update with a short bounded retry so this task does not race
        // the tier transition it is meant to observe.
        let mut transitioned = None;
        for attempt in 0..8u64 {
            if let Ok(Some(session)) = db.get_session_async(id).await {
                if session.tier >= 2 {
                    transitioned = Some(session);
                    break;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(10 + attempt * 10)).await;
        }
        let Some(session) = transitioned else {
            return;
        };

        // Free providers (e.g. Ollama) have no per-token pricing, so the
        // degradation state machine never advances for them - but if a
        // transition somehow occurred, skip warming since there's no
        // cheaper model to warm.
        if provider.is_free() {
            return;
        }

        let manager = engine_core::PreWarmManager::new(engine_core::PreWarmConfig::default());
        let tool_state_store = db.clone() as std::sync::Arc<dyn engine_core::ToolStateStore>;
        let db_ref = db.clone();

        // Gather past session shapes from the same project for similarity
        // matching. We use the session list filtered by project path.
        let past_shapes = build_past_shapes(db_ref.as_ref(), &session).await;

        let result = manager
            .warm_on_tier_transition(
                &session,
                provider,
                &provider_config,
                Some(tool_state_store),
                db_ref.as_ref(),
                db_ref.as_ref(),
                &past_shapes,
            )
            .await;
        if let Err(error) = result {
            tracing::error!(%error, "prewarm or its accounting failed");
        }
    });
}

/// Build `SessionShape` entries for past sessions on the same project as
/// the given session. Used by the pre-warming manager to find similar
/// sessions whose tool results should be pre-fetched.
async fn build_past_shapes(
    db: &LocalDatabase,
    current: &Session,
) -> Vec<engine_core::SessionShape> {
    let path = current.project_path.clone();
    let Ok(past_sessions) = db
        .list_sessions_async(crate::db::SessionFilter {
            path: path.as_deref(),
            ..Default::default()
        })
        .await
    else {
        return Vec::new();
    };

    let candidates = past_sessions.into_iter().take(32).filter(|past| {
        past.id != current.id
            && !(past.root_session_id == current.root_session_id
                && current.root_session_id.is_some())
    });
    futures::stream::iter(candidates)
        .map(|past| async move {
            let tool_names = extract_tool_names_for_prewarm(db, past.id).await;
            engine_core::SessionShape::from_session(&past, tool_names)
        })
        .buffered(8)
        .filter_map(futures::future::ready)
        .collect()
        .await
}

/// Extract sorted, deduplicated tool names from a session's ToolCall events.
async fn extract_tool_names_for_prewarm(db: &LocalDatabase, session_id: Uuid) -> Vec<String> {
    const MAX_SHAPE_EVENTS: usize = 256;
    let Ok(events) = db
        .read_events(session_id, None, Some(MAX_SHAPE_EVENTS))
        .await
    else {
        return Vec::new();
    };
    let mut names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for event in &events {
        if event.event_type == SessionEventType::ToolCall {
            if let Some(name) = event
                .payload
                .get("tool_call")
                .and_then(|v| v.get("function"))
                .and_then(|v| v.get("name"))
                .and_then(|v| v.as_str())
            {
                names.insert(name.to_string());
            }
        }
    }
    names.into_iter().collect()
}

#[cfg(test)]
pub(crate) async fn record_external_tool_executions(
    db: &LocalDatabase,
    session_id: Uuid,
    request_messages: &[ChatMessage],
    fallback_request_group_id: &str,
    delegation_profile: Option<&str>,
    track_subagent_sessions: bool,
) -> Result<(), StatusCode> {
    if !engine_core::request_has_external_tool_result(request_messages) {
        return Ok(());
    }
    let outcome = engine_core::record_external_tool_executions_detailed_for_profile(
        db,
        session_id,
        request_messages,
        fallback_request_group_id,
        delegation_profile,
    )
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if track_subagent_sessions && !outcome.completed_subagents.is_empty() {
        let parent = db
            .get_session_async(session_id)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;
        reconcile_external_subagent_sessions(db, &parent, &outcome.completed_subagents).await?;
    }
    Ok(())
}

async fn record_external_tool_executions_on_request_path(
    db: std::sync::Arc<LocalDatabase>,
    parent: Option<Session>,
    session_id: Uuid,
    request_messages: &[ChatMessage],
    file_evidence: &[serde_json::Value],
    fallback_request_group_id: &str,
    delegation_profile: Option<&str>,
    track_subagent_sessions: bool,
) -> Result<(), StatusCode> {
    if !engine_core::request_has_external_tool_result(request_messages) {
        return Ok(());
    }
    let outcome = engine_core::external_tools::record_external_tool_executions_with_file_evidence(
        db.as_ref(),
        session_id,
        request_messages,
        fallback_request_group_id,
        delegation_profile,
        file_evidence,
    )
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    if track_subagent_sessions && !outcome.completed_subagents.is_empty() {
        if let Some(parent) = parent {
            tokio::spawn(async move {
                if let Err(status) = reconcile_external_subagent_sessions(
                    db.as_ref(),
                    &parent,
                    &outcome.completed_subagents,
                )
                .await
                {
                    warn!(session_id = %parent.id, %status, "failed to reconcile external subagent session");
                }
            });
        }
    }
    Ok(())
}

/// Materialize completed external delegation executions as deterministic
/// child sessions. The child contains the observable spawn/result boundary;
/// nested model calls remain explicitly opaque when the client does not pass
/// them through its provider protocol.
async fn reconcile_external_subagent_sessions(
    db: &LocalDatabase,
    parent: &Session,
    executions: &[engine_core::ExternalSubagentExecution],
) -> Result<(), StatusCode> {
    let parent_external_id = parent
        .external_session_id
        .clone()
        .unwrap_or_else(|| parent.id.to_string());
    let mut direct_children = db
        .list_subagent_sessions_async(parent_external_id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    for execution in executions {
        // A protocol-aware client can stream the nested model calls directly
        // and identify them with the originating delegation tool call. In that
        // case the real child already exists; do not add an opaque duplicate.
        if direct_children.iter().any(|child| {
            child.subagent_spawn_tool_call_id.as_deref() == Some(execution.tool_call.id.as_str())
        }) {
            continue;
        }
        let child = engine_core::external_subagent_session(parent, execution)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        if db
            .get_session_async(child.id)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .is_some()
        {
            continue;
        }

        db.insert_session_async(child.clone())
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        direct_children.push(child.clone());
        db.append_event(
            child.id,
            SessionEventType::ToolCall,
            json!({
                "tool_call": execution.tool_call,
                "requested_at": execution.requested_at,
                "execution_source": "external_client_opaque",
                "delegation_profile": execution.delegation_profile,
                "request_group_id": execution.request_group_id,
                "origin_llm_call_seq": execution.origin_llm_call_seq,
                "parent_session_id": parent.id,
                "is_subagent": true,
            }),
            None,
        )
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        db.append_event(
            child.id,
            SessionEventType::ToolResult,
            json!({
                "result": execution.result,
                "cache_hit": false,
                "execution_completed_at": execution.completed_at,
                "execution_source": "external_client_opaque",
                "delegation_profile": execution.delegation_profile,
                "request_group_id": child.subagent_spawn_request_group_id,
                "parent_session_id": parent.id,
                "is_subagent": true,
            }),
            None,
        )
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }
    Ok(())
}

/// Best-effort startup migration for executions already preserved inside
/// historical LLM request snapshots. The per-session recorder is idempotent,
/// so this is safe to run on every daemon start.
#[cfg(test)]
pub(crate) async fn backfill_external_tool_executions(db: &LocalDatabase) {
    let Ok(sessions) = db
        .list_sessions_async(crate::db::SessionFilter::default())
        .await
    else {
        return;
    };
    for session in sessions {
        let Ok(events) = db.read_events(session.id, None, None).await else {
            continue;
        };
        for event in events
            .iter()
            .filter(|event| event.event_type == SessionEventType::LlmCall)
        {
            let messages = event
                .payload
                .get("request_messages")
                .and_then(|value| serde_json::from_value::<Vec<ChatMessage>>(value.clone()).ok())
                .unwrap_or_default();
            if !engine_core::request_has_external_tool_result(&messages) {
                continue;
            }
            let group = event
                .payload
                .get("request_group_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("historical-external-tools");
            let profile = event
                .payload
                .get("delegation_profile")
                .and_then(serde_json::Value::as_str);
            let _ =
                record_external_tool_executions(db, session.id, &messages, group, profile, true)
                    .await;
        }
    }
}

pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, StatusCode> {
    let _handler_timer =
        engine_core::operational::Timer::new(engine_core::operational::Metric::ChatHandler);
    engine_core::operational::increment(engine_core::operational::Counter::ChatRequests, 1);
    let request_span = engine_core::telemetry::pipeline_root_span("chat_completions", None);
    request_span.record("http.request.method", "POST");
    request_span.record("http.route", "/v1/chat/completions");
    engine_core::telemetry::set_parent_from_trace_headers(
        &request_span,
        headers
            .get("traceparent")
            .and_then(|value| value.to_str().ok()),
        headers
            .get("tracestate")
            .and_then(|value| value.to_str().ok()),
    );

    let result = chat_completions_inner(state, headers, req)
        .instrument(request_span.clone())
        .await;
    let status = result
        .as_ref()
        .map(|response| response.status().as_u16())
        .unwrap_or_else(|status| status.as_u16());
    if status >= 400 {
        engine_core::operational::increment(engine_core::operational::Counter::ChatErrors, 1);
    }
    engine_core::telemetry::record_http_status(&request_span, status);
    result
}

async fn chat_completions_inner(
    state: AppState,
    headers: HeaderMap,
    mut req: ChatCompletionRequest,
) -> Result<axum::response::Response, StatusCode> {
    if engine_core::user_content::is_client_compaction_request(&req) {
        return Ok((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(engine_core::user_content::client_compaction_error()),
        )
            .into_response());
    }
    if engine_core::user_content::is_client_reminder_observer_request(&req) {
        return Ok((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(engine_core::user_content::client_reminder_observer_error()),
        )
            .into_response());
    }
    // Measure elapsed request processing from before normalization and session
    // resolution, so timing includes preparation as well as provider execution.
    let request_started = std::time::Instant::now();
    // Historical Copilot/editor messages repeat volatile context and reminder
    // scaffolding. Keep the current wrapper for the provider, but reduce older
    // user turns to their literal request before session logic sees them.
    engine_core::task_decisions::clear_untrusted_context(&mut req);
    engine_core::continuation::clear_untrusted_context(&mut req);
    engine_core::claim_grounding::clear_untrusted_context(&mut req);
    engine_core::normalize_correlated_terminal_notifications(&mut req);
    engine_core::normalize_previous_user_messages(&mut req);

    // STAGE: resolve
    // Session + file-attachment + request-group resolution. Runs before any
    // cache lookup or provider dispatch.
    let resolve_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Resolve);

    // Resolve local file paths (file://... or bare paths) in multi-part
    // content to base64 data URLs before any cache lookup or provider
    // dispatch. This lets clients send local file references that the
    // engine reads from disk and encodes inline.
    if let Err(e) = req.resolve_local_files() {
        warn!(error = %e, "failed to resolve local file attachments");
        return Err(StatusCode::BAD_REQUEST);
    }

    // Validate provider selection before cache lookup so multi-provider calls
    // can never bypass the explicit-header contract through a cache hit.
    let provider_override = provider_override_from_headers(&headers, &state.providers, &req.model)?;
    if let Some(label) = provider_override.as_deref() {
        if state.providers.get_by_label(label).is_none() {
            warn!(
                provider = label,
                "requested local provider is not configured"
            );
            return Err(StatusCode::BAD_REQUEST);
        }
    }

    let session_resolve_span = tracing::info_span!(
        parent: &resolve_span,
        "reincore.resolve.session"
    );
    // Resolve the active session and its persisted state before assigning the
    // request group, recording external tool results, or selecting task context.
    // Later budget checks and completion accounting use this session identity.
    let mut session_snapshot = resolve_local_session(
        state.db.as_ref(),
        &req,
        state.default_project_name.as_str(),
        state.default_project_path.as_str(),
    )
    .instrument(session_resolve_span)
    .await?;
    // Attach the resolved session id to the root span so downstream trace
    // explorers can correlate with the ReinCore session/event-log views.
    resolve_span.record(
        "reincore.session_id",
        session_snapshot
            .as_ref()
            .map(|s| s.id)
            .map(|id| id.to_string())
            .as_deref()
            .unwrap_or("")
            .to_string(),
    );
    let batch_requested = extra_bool(&req, "reincore_batch_eligible").unwrap_or_else(|| {
        session_snapshot
            .as_ref()
            .map(|session| session.mode == SessionMode::Background)
            .unwrap_or(false)
    });
    let mut session_id = session_snapshot.as_ref().map(|session| session.id);
    if let Some(session) = session_snapshot.as_ref() {
        engine_core::file_history::bind_request_roots(state.db.as_ref(), session, &req)
            .await
            .map_err(|error| match error {
                engine_core::EngineError::InvalidInput(_) => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            })?;
        state
            .sessions
            .lock()
            .await
            .insert(session.id, session.clone());
    }
    let group_resolve_span = tracing::info_span!(
        parent: &resolve_span,
        "reincore.resolve.request_group"
    );
    let request_group_id =
        resolve_request_group_id(state.db.as_ref(), Some(&headers), &req, session_id)
            .instrument(group_resolve_span)
            .await?;
    resolve_span.record("reincore.request_group_id", request_group_id.as_str());
    engine_core::provider::prepare_request_replay(
        &mut req,
        &request_group_id,
        state.provider_replay_enabled,
    );
    if let Some(id) = session_id {
        let delegation_profile = delegation_profile(Some(&headers), &req);
        let track_subagent_sessions =
            extra_bool(&req, "reincore_track_subagent_sessions").unwrap_or(true);
        let external_tools_span = tracing::info_span!(
            parent: &resolve_span,
            "reincore.resolve.external_tools",
            message_count = req.messages.len()
        );
        record_external_tool_executions_on_request_path(
            state.db.clone(),
            session_snapshot.clone(),
            id,
            &req.messages,
            req.extra
                .get("reincore_file_evidence")
                .and_then(serde_json::Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            &request_group_id,
            delegation_profile.as_deref(),
            track_subagent_sessions,
        )
        .instrument(external_tools_span)
        .await?;
    }
    let task_resolve_span = tracing::info_span!(
        parent: &resolve_span,
        "reincore.resolve.task"
    );
    let detection_judge = detection_judge(state.providers.get_by_label("meta"), &state.detection);
    let (task, task_context_messages_removed) = resolve_and_scope_task(
        state.db.as_ref(),
        session_id,
        &request_group_id,
        &mut req,
        state.task_decisions_enabled,
        &state.claim_grounding,
        &state.detection,
        detection_judge.as_ref(),
    )
    .instrument(task_resolve_span)
    .await?;
    drop(resolve_span);

    let caller_semantic_embedding = parse_embedding(&req).map_err(|_| StatusCode::BAD_REQUEST)?;
    let semantic_scope = session_snapshot
        .as_ref()
        .and_then(|session| session.project_id)
        .map(|project_id| engine_core::SemanticCacheScope {
            workspace_id: "local".to_string(),
            project_id: project_id.to_string(),
        });

    // STAGE: classify
    // Classify the request to pick a model tier.
    let classify_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Classify);
    let validation_signal = req
        .extra
        .get("reincore_validation_signal")
        .and_then(|v| v.as_str());
    let classifier = if let Some(id) = session_id {
        state
            .classifier
            .trained_for_session(state.db.as_ref(), id)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    } else {
        state.classifier.as_ref().clone()
    };
    let routing = classifier
        .classify(&req, validation_signal)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    // Retain the classifier's recommended model tier for initial pricing and
    // diagnostics. Budget degradation and routing overrides may select a different
    // effective tier later, without replacing this original recommendation.
    let classified_model_tier = routing.model_tier;
    classify_span.record("reincore.classified_tier", classified_model_tier.as_str());
    drop(classify_span);

    // STAGE: route
    // Provider selection + budget degradation + tier resolution.
    let route_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Route);
    // In true multi-provider mode the explicit header is mandatory unless the
    // requested model identifies its provider. With one configured provider
    // the header remains optional for backward compatibility.
    let provider = state
        .providers
        .select_with_override(&req, provider_override.as_deref())
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    super::titles::spawn_titles(
        state.db.clone(),
        provider.clone(),
        session_snapshot.as_ref(),
        &task,
        &req,
    );

    // Resolve the classified tier before budget evaluation so pre-flight
    // pricing is based on the concrete model rather than a tier average.
    let classified_model = provider
        .config()
        .model_for_tier(classified_model_tier)
        .map_err(|error| {
            warn!(
                provider = provider.label(),
                tier = classified_model_tier.as_str(),
                %error,
                "failed to resolve classified model from local provider configuration"
            );
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let initial_cost_predictor = (*state.cost_predictor)
        .clone()
        .with_provider(provider.label())
        .with_model(classified_model);
    drop(route_span);

    // STAGE: predict
    // Pre-flight cost prediction. Runs even when the request is subsequently
    // degraded - the range is computed against the classified model so
    // budget decisions use the best available price signal.
    let predict_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Predict);

    let initial_prediction = initial_cost_predictor
        .predict(&req, classified_model_tier)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    drop(predict_span);
    // Budget degradation / routing tier resolution is part of the route stage.
    let route_span_sec = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Route);
    // Compare recorded spend across the session lineage and the predicted call
    // cost with budget thresholds. Exhausted actual spend pauses at a checkpoint;
    // projected spend can lower the model tier without triggering that pause.
    let budget_routing = resolve_budget_routing(
        session_snapshot.as_ref(),
        classified_model_tier,
        initial_prediction.high_cents,
    )
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    route_span_sec.record(
        "reincore.budget_tier",
        budget_routing.projected_tier.as_u8() as i64,
    );

    // Capture the pre-call tier so we can detect a tier transition after
    // durable completion accounting applies the spend. Used by the
    // predictive pre-warming subsystem.
    let prewarm_expected =
        budget_routing.current_tier.as_u8() < 2 && budget_routing.projected_tier.as_u8() >= 2;
    // Capture the provider config clone for the pre-warm trigger.
    let prewarm_provider = provider.clone();
    let prewarm_provider_config = provider.config().clone();

    if budget_routing.should_checkpoint {
        let mut session = session_snapshot.ok_or(StatusCode::CONFLICT)?;
        let last_seq = state
            .db
            .latest_seq(session.id)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .unwrap_or(0);
        let token = state
            .checkpoints
            .create(&mut session, last_seq, "budget exhausted")
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        state
            .db
            .insert_checkpoint_async(token.clone())
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        state
            .db
            .insert_session_async(session.clone())
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        let group_snapshot = record_request_group_step(&request_group_id, 0, 0, Some(0)).await;

        let mut checkpoint_json = json!({
            "object": "reincore.checkpoint",
            "reincore_paused": true,
            "reason": "budget exhausted",
            "session_id": session.id,
            "checkpoint_token": token.token,
            "last_event_seq": token.last_event_seq,
            "expires_at": token.expires_at,
        });

        attach_request_group_metadata(&mut checkpoint_json, &group_snapshot);
        attach_task_metadata(&mut checkpoint_json, &task, task_context_messages_removed);
        print_request_group_rollup(&group_snapshot);

        return Ok(Json(checkpoint_json).into_response());
    }

    let budgeted_model_tier = budget_routing
        .effective_model_tier
        .ok_or(StatusCode::CONFLICT)?;

    // Apply the routing policy + per-request escape hatches
    // (reincore_routing_mode / reincore_minimum_tier / reincore_maximum_tier)
    // to the classified + budget-degraded tier.
    let routing_overrides = engine_core::RequestRoutingOverrides::from_extra(&req.extra);
    let resolved_tier = engine_core::resolve_routing_tier(
        provider.config().routing_policy(),
        &routing_overrides,
        classified_model_tier,
        budgeted_model_tier,
    )
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let model_tier = resolved_tier.tier;
    route_span_sec.record("reincore.final_tier", model_tier.as_str());
    drop(route_span_sec);

    // Loop-detection configuration, least to most specific: built-in defaults,
    // then `config.toml`, then `REINCORE_LOOP_*`, then stored settings, then
    // this request's `reincore_loop_*` overrides.
    let loop_breaker = CircuitBreaker::new(
        engine_core::LoopSettings::clone(&state.loop_detection)
            .overlay(engine_core::LoopSettings::from_env())
            .overlay(state.db.get_loop_settings().await.unwrap_or_default())
            .overlay(engine_core::LoopSettings::from_extra(&req.extra))
            .resolve(),
    );

    // Load any intervention scheduled by earlier loop-health checks. A hard stop
    // returns a refusal; a soft intervention adds a system nudge to the request.
    // Check before compaction and cache lookup so answer reuse cannot bypass it.
    let loop_nudge = if let Some(id) = session_id {
        loop_breaker
            .pending_intervention(state.db.as_ref(), id)
            .await
            .unwrap_or(None)
    } else {
        None
    };

    if let Some(intervention) = loop_nudge.as_ref().filter(|item| item.hard_stop) {
        let findings = match session_id {
            Some(id) => {
                let _ = loop_breaker.clear_intervention(state.db.as_ref(), id).await;
                engine_core::recent_health_findings(state.db.as_ref(), id, 32)
                    .await
                    .unwrap_or_default()
            }
            None => Vec::new(),
        };
        return Ok(loop_hard_stop_response(session_id, intervention, findings));
    }

    let loop_nudge_injected = loop_nudge.is_some();
    if let Some(intervention) = loop_nudge {
        inject_system_nudge(&mut req, intervention.message);
    }

    // Replace read/search results the transcript already carries with a pointer
    // to the earlier copy. This runs before compaction and the final prediction
    // so the saved tokens are real, and it gives a re-reading model something to
    // act on rather than only counting the repeat toward a refusal.
    let deduped_tool_results = loop_breaker.dedupe_tool_results(&mut req);
    if !deduped_tool_results.is_empty() {
        tracing::debug!(
            replaced = deduped_tool_results.replaced,
            saved_chars = deduped_tool_results.saved_chars,
            "replaced duplicate tool results with pointers to the earlier copy"
        );
    }

    let settings = state
        .db
        .get_compression_settings()
        .await
        .unwrap_or_default();

    // STAGE: compact
    // Context compression, context-window enforcement, session rollover /
    // handoff creation. Rollover compacts the conversation into a successor
    // session - the *detection* happens here, the actual compaction work
    // happens inside `prepare_context*` / `rollover_*_if_needed`.
    let compact_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Compact);

    // Preserve prompt-injection findings from tool results that compaction may
    // replace with a bounded digest, then scan the compiled request as well.
    // Retain these findings in agent_health and merge the later scan of prepared
    // context, so replacing suspicious tool output with a summary does not erase
    // the evidence included in completion health reporting.
    engine_core::task_decisions::detach(&mut req);
    engine_core::continuation::detach(&mut req);
    let mut agent_health = engine_core::scan_prompt_injection(&req);
    let context_adjustment = match session_id {
        Some(id) if !engine_core::compression_turn_is_new(id, &request_group_id) => {
            engine_core::prepare_context_continuation(
                &mut req,
                settings,
                budget_routing.projected_tier,
            )
        }
        _ => prepare_context(&mut req, settings, budget_routing.projected_tier),
    };

    agent_health.merge(engine_core::scan_prompt_injection(&req));

    log_compression(provider.label(), &context_adjustment);

    let resolved_model = provider
        .config()
        .model_for_tier(model_tier)
        .map_err(|error| {
            warn!(
                provider = provider.label(),
                tier = model_tier.as_str(),
                %error,
                "failed to resolve effective model from local provider configuration"
            );
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Budget degradation may have changed the tier/model. From this point on,
    // context checks, final prediction, and actual spend share that exact model.
    let cost_predictor = (*state.cost_predictor)
        .clone()
        .with_provider(provider.label())
        .with_model(resolved_model.clone());

    // Provider metadata and citation state are independent I/O. Resolve them
    // concurrently so neither adds serial latency before the upstream call.
    let citation_files = async {
        match session_snapshot.as_ref() {
            Some(session) => state.db.read_latest_files(session.id).await.map(Some),
            None => Ok(None),
        }
    };

    let (reported_context_limit, citation_attachments) =
        tokio::join!(provider.context_window(&resolved_model), citation_files);
    let citation_attachments =
        citation_attachments.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    // Prefer a context window the provider can report first-hand (Ollama
    // queries its own `/api/show`); fall back to the catalog, then the
    // REINCORE_CONTEXT_LIMIT env var (defaulting to 1_000_000). The old
    // hardcoded 128K fallback was too small - it caused premature session
    // rollovers whenever a model wasn't in the catalog, which is exactly
    // the behaviour we're trying to avoid.
    let context_limit = match reported_context_limit {
        Some(ctx) => ctx.max(1),
        None => find_model(provider.label(), &resolved_model)
            .map(|model| model.max_input_tokens.max(1) as u64)
            .unwrap_or_else(fallback_context_limit),
    };

    let (prepared_request, context_reset) =
        engine_core::continuation::enforce_context_window_async(
            req,
            cost_predictor.clone(),
            model_tier,
            context_limit,
        )
        .await
        .map_err(|error| match error {
            engine_core::EngineError::ContextWindowExceeded { .. } => StatusCode::PAYLOAD_TOO_LARGE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        })?;
    req = prepared_request;

    let mut rolled_over_from = None;
    let mut session_handoff: Option<SessionHandoff> = None;
    if let Some(session) = session_snapshot.take() {
        let (active, previous_id) =
            rollover_local_session_if_needed(state.db.as_ref(), session, &context_reset, &req)
                .await?;
        session_id = Some(active.id);
        engine_core::file_history::bind_request_roots(state.db.as_ref(), &active, &req)
            .await
            .map_err(|error| match error {
                engine_core::EngineError::InvalidInput(_) => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            })?;
        rolled_over_from = previous_id;
        if previous_id.is_some() {
            session_handoff = Some(context_reset.handoff.clone());
        }
        state
            .sessions
            .lock()
            .await
            .insert(active.id, active.clone());
    }
    compact_span.record("reincore.rolled_over", rolled_over_from.is_some());
    drop(compact_span);

    // STAGE: route (final model resolution)
    let route_span3 = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Route);

    // Canonicalize the prepared request's static prefix and tool schemas before
    // building the response-cache query. Combine the resulting prefix hash with
    // the effective provider/model and scope to avoid reusing incompatible answers.
    let prefix = CachePrefixNormalizer::new()
        .normalize_request(&mut req)
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    route_span3.record("reincore.prefix_hash", prefix.prefix_hash.as_str());
    drop(route_span3);
    // Hand the prefix identity to provider builders that can use it as a cache
    // routing key. The reincore_ prefix keeps it out of semantic-cache
    // fingerprints, so it never fragments otherwise identical rows.
    req.extra.insert(
        "reincore_prefix_hash".to_string(),
        serde_json::Value::String(prefix.prefix_hash.clone()),
    );

    let semantic = build_semantic_cache_plan(
        &req,
        caller_semantic_embedding,
        semantic_scope,
        provider.as_ref(),
        &resolved_model,
        &prefix.prefix_hash,
    )?;
    let semantic_lookup = if let Some(plan) = semantic.as_ref() {
        state
            .db
            .get_semantic(
                &plan
                    .query(chrono::Utc::now())
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
            )
            .await
            .unwrap_or_default()
    } else {
        engine_core::SemanticCacheLookup::default()
    };
    if let Some(mut hit) = semantic_lookup.reusable_answer.filter(|hit| {
        serde_json::from_value::<ChatMessage>(hit.response.clone()).is_ok_and(|message| {
            !message.full_text().is_empty()
                && !message
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
        })
    }) {
        if let Some(message) = hit.response.as_object_mut() {
            message.remove("provider_state");
        }
        if let Ok(message) = serde_json::from_value::<ChatMessage>(hit.response.clone()) {
            agent_health.merge(engine_core::claim_grounding::analyze_health(&req, &message));
            if let Some(id) = session_id {
                let _ = engine_core::log_agent_health(state.db.as_ref(), id, &agent_health).await;
            }
        }
        record_semantic_cache_hit(
            state.db.as_ref(),
            session_id,
            &request_group_id,
            &task,
            &req,
            &hit.response,
            &resolved_model,
            &prefix.prefix_hash,
            engine_core::similarity_to_bps(hit.provenance.similarity),
        )
        .await?;
        let response_id = Uuid::new_v4().to_string();
        let mut response_json = semantic_hit_response(
            response_id.clone(),
            &resolved_model,
            session_id,
            hit.response,
        );
        response_json["reincore_agent_health"] =
            serde_json::to_value(&agent_health).unwrap_or_default();
        response_json["reincore_semantic_cache_provenance"] =
            serde_json::to_value(hit.provenance).unwrap_or_default();
        let group_snapshot = record_request_group_step(&request_group_id, 0, 0, Some(0)).await;
        attach_request_group_metadata(&mut response_json, &group_snapshot);
        attach_task_metadata(&mut response_json, &task, task_context_messages_removed);
        link_response_to_request_group(&response_id, &request_group_id).await;
        print_request_group_rollup(&group_snapshot);
        if req.stream.unwrap_or(false) {
            return Ok(Sse::new(chat_completion_to_sse_stream(response_json))
                .keep_alive(KeepAlive::default())
                .into_response());
        }
        return Ok(Json(response_json).into_response());
    }
    let semantic_candidates = semantic_lookup.candidates;

    // STAGE: predict (final prediction on the effective model)
    let predict_span_sec =
        engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Predict);
    // Estimate cost for the effective model using the input-token count already
    // computed during context enforcement. This avoids recounting the prepared
    // input and supplies final diagnostics; it does not rerun budget routing.
    let prediction = cost_predictor
        .predict_with_input_tokens(&req, model_tier, context_reset.input_tokens_after)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    predict_span_sec.record("reincore.predicted_cost_high_cents", prediction.high_cents);
    predict_span_sec.record(
        "reincore.predicted_input_tokens",
        prediction.input_tokens as i64,
    );
    drop(predict_span_sec);

    log_provider(
        provider.label(),
        &format!(
            "{}  {} input tokens, tier: {}, model: {}, reasoning: {}",
            format!("Predicted cost: {}", prediction.range_dollars()).yellow(),
            prediction.input_tokens,
            prediction.model_tier.as_str().cyan(),
            resolved_model.green(),
            prediction.reasoning.dimmed(),
        ),
    );

    // This estimate is audit metadata and never shapes the current response.
    // Persist it off the provider hot path.
    {
        let db = state.db.clone();
        let group_id = request_group_id.clone();
        let predicted_tier = prediction.model_tier.as_str().to_string();
        let (input, output, low, high) = (
            prediction.input_tokens,
            prediction.output_tokens,
            prediction.low_cents,
            prediction.high_cents,
        );
        tokio::spawn(async move {
            let _ = db
                .insert_cost_estimate_async(
                    session_id,
                    Some(group_id),
                    "llm_call".to_string(),
                    predicted_tier,
                    input,
                    output,
                    low,
                    high,
                )
                .await;
        });
    }

    // Inject the chosen tier into the request so the provider adapter resolves
    // the concrete model name.
    let mut provider_request = req.clone();
    provider_request.extra.insert(
        "reincore_model_tier".to_string(),
        serde_json::json!(model_tier.as_str()),
    );

    // Submit eligible non-streaming requests as durable batch jobs and return
    // their submission status. Do not fall back to a live call after an uncertain
    // submission: the provider may already have accepted and billed the batch.
    if batch_requested && !req.stream.unwrap_or(false) {
        match engine_core::batch_jobs::submit_chat(
            state.db.clone(),
            provider.clone(),
            session_id,
            &provider_request,
        )
        .await
        {
            Ok(batch) => {
                let batch_factor = engine_core::batch_jobs::prediction_factor(
                    provider.label(),
                    &resolved_model,
                    prediction.input_tokens,
                );
                let batch_low_cents =
                    (prediction.low_cents as f64 * batch_factor.unwrap_or(1.0)).ceil() as i64;
                let batch_high_cents =
                    (prediction.high_cents as f64 * batch_factor.unwrap_or(1.0)).ceil() as i64;
                if batch.status != "submission_unknown" && batch.status != "submitting" {
                    record_batch_submission(
                        state.db.as_ref(),
                        session_id,
                        &request_group_id,
                        &task,
                        &provider_request,
                        &resolved_model,
                        model_tier,
                        &prefix.prefix_hash,
                        &batch.batch_id,
                        batch_low_cents,
                        batch_high_cents,
                        routing.classification.clone(),
                        context_adjustment.clone(),
                    )
                    .await?;
                }
                let group_snapshot =
                    record_request_group_step(&request_group_id, 0, 0, Some(0)).await;
                let mut batch_json = json!({
                    "id": batch.batch_id,
                    "object": "reincore.batch",
                    "created": chrono::Utc::now().timestamp(),
                    "model": resolved_model,
                    "reincore_batch_eligible": true,
                    "reincore_batch_submitted": batch.status != "submission_unknown" && batch.status != "submitting",
                    "reincore_batch_status": batch.status,
                    "reincore_batch_discount_factor": batch_factor,
                    "reincore_predicted_cost": cents_range_dollars(batch_low_cents, batch_high_cents),
                    "reincore_predicted_cost_low_cents": batch_low_cents,
                    "reincore_predicted_cost_high_cents": batch_high_cents,
                    "reincore_routing_reason": routing.reason,
                    "reincore_classification": routing.classification,
                    "reincore_routing_adjustment": resolved_tier.adjustment,
                    "reincore_budget_tier": budget_routing.projected_tier.as_u8(),
                    "reincore_model_degraded": budget_routing.model_was_degraded(),
                    "reincore_session_id": session_id,
                    "reincore_session_rolled_over_from": rolled_over_from,
                });
                attach_request_group_metadata(&mut batch_json, &group_snapshot);
                attach_task_metadata(&mut batch_json, &task, task_context_messages_removed);
                print_request_group_rollup(&group_snapshot);
                return Ok((StatusCode::ACCEPTED, Json(batch_json)).into_response());
            }
            Err(error) => {
                warn!(
                    provider = provider.label(),
                    %error,
                    "batch submission failed; synchronous fallback is prohibited"
                );
                return Err(StatusCode::BAD_GATEWAY);
            }
        }
    }

    // Record provider-call timing, then choose native streaming when supported
    // or a complete response otherwise. The live SSE body takes ownership of
    // response assembly and accounting, which run as that body is consumed.
    let provider_request_sent_at = chrono::Utc::now();
    let provider_request_started = std::time::Instant::now();
    engine_core::operational::observe(
        engine_core::operational::Metric::GatewayPrepare,
        request_started.elapsed(),
    );
    let provider_timer =
        engine_core::operational::Timer::new(engine_core::operational::Metric::ProviderOperation);

    // STAGE: provider
    // Upstream LLM call (sync or streaming first chunk). The returned guard
    // stays open until the response is received or the stream opens.
    let provider_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Provider);

    if req.stream.unwrap_or(false) && provider.supports_streaming() {
        let upstream = match provider
            .complete_stream(&provider_request)
            .instrument(provider_span.clone())
            .await
        {
            Ok(stream) => stream,
            Err(error) if error.is_context_overflow_error() => {
                warn!(
                    provider = provider.label(),
                    %error,
                    "provider reported a full context window while opening stream; rolling the session over"
                );
                let recovery = recover_from_provider_overflow(
                    state.db.as_ref(),
                    session_id,
                    &mut provider_request,
                    context_limit,
                )
                .await?;
                if let Some(active) = recovery.session {
                    session_id = Some(active.id);
                    engine_core::file_history::bind_request_roots(state.db.as_ref(), &active, &req)
                        .await
                        .map_err(|error| match error {
                            engine_core::EngineError::InvalidInput(_) => StatusCode::BAD_REQUEST,
                            _ => StatusCode::INTERNAL_SERVER_ERROR,
                        })?;
                    state.sessions.lock().await.insert(active.id, active);
                }
                provider
                    .complete_stream(&provider_request)
                    .instrument(provider_span.clone())
                    .await
                    .map_err(|error| provider_error_to_status(&error))?
            }
            Err(error) => {
                engine_core::telemetry::record_error(
                    &provider_span,
                    "provider_error",
                    &engine_core::privacy::current_policy()
                        .sanitize_for_logging(&error.to_string()),
                );
                return Err(provider_error_to_status(&error));
            }
        };

        let provider_response_headers = upstream.response_headers.clone();
        let stream_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Stream);
        stream_span.record("reincore.provider", provider.label());
        stream_span.record("reincore.model", resolved_model.as_str());
        let stream = live_chat_completion_to_sse_stream(
            upstream,
            LocalStreamContext {
                action_policy: state.action_policy.clone(),
                claim_grounding: state.claim_grounding.clone(),
                db: state.db.clone(),
                session_id,
                request_group_id,
                task,
                task_context_messages_removed,
                original_request: req,
                provider_request,
                provider_label: provider.label().to_string(),
                detection: state.detection.clone(),
                detection_judge,
                resolved_model,
                model_tier,
                prefix_hash: prefix.prefix_hash,
                prediction,
                cost_predictor,
                classification: routing.classification,
                batch_requested,
                semantic,
                semantic_candidates,
                citation_attachments,
                initial_agent_health: agent_health,
                prewarm_expected,
                prewarm_provider,
                prewarm_provider_config,
                provider_request_sent_at,
                provider_request_started,
                request_started,
                provider_timer,
                context_adjustment,
                deduped_tool_results,
                loop_breaker,
                provider_span: provider_span.clone(),
                stream_span,
            },
        );
        let stream = engine_core::telemetry::instrument_stream(
            stream,
            tracing::Span::current(),
            vec![provider_span],
        );

        let response = Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response();
        return Ok(with_provider_rate_limit_headers(
            response,
            &provider_response_headers,
        ));
    }

    // Call the real provider. A context-window rejection here means our
    // pre-flight estimate disagreed with the provider's own accounting, so the
    // conversation is compacted into a successor session and retried once.
    let provider_response = match provider
        .complete(&provider_request)
        .instrument(provider_span.clone())
        .await
    {
        Ok(response) => response,
        Err(error) if error.is_context_overflow_error() => {
            warn!(
                provider = provider.label(),
                %error,
                "provider reported a full context window; rolling the session over"
            );
            let recovery = recover_from_provider_overflow(
                state.db.as_ref(),
                session_id,
                &mut provider_request,
                context_limit,
            )
            .await?;
            if let Some(active) = recovery.session {
                session_id = Some(active.id);
                engine_core::file_history::bind_request_roots(state.db.as_ref(), &active, &req)
                    .await
                    .map_err(|error| match error {
                        engine_core::EngineError::InvalidInput(_) => StatusCode::BAD_REQUEST,
                        _ => StatusCode::INTERNAL_SERVER_ERROR,
                    })?;
                rolled_over_from = recovery.rolled_over_from;
                state
                    .sessions
                    .lock()
                    .await
                    .insert(active.id, active.clone());
            }
            session_handoff = Some(recovery.handoff);
            provider
                .complete(&provider_request)
                .instrument(provider_span.clone())
                .await
                .map_err(|e| provider_error_to_status(&e))?
        }
        Err(error) => {
            engine_core::telemetry::record_error(
                &provider_span,
                "provider_error",
                &engine_core::privacy::current_policy().sanitize_for_logging(&error.to_string()),
            );
            return Err(provider_error_to_status(&error));
        }
    };
    let engine_core::ProviderResponseAttempts {
        response: provider_response,
        prior_usage: retry_usage,
        all_usage_reported: all_retry_usage_reported,
        retry_count,
    } = engine_core::retry_empty_provider_response(
        provider.as_ref(),
        &provider_request,
        provider_response,
    )
    .instrument(provider_span.clone())
    .await
    .map_err(|error| provider_error_to_status(&error))?;
    if retry_count > 0 {
        warn!(
            provider = provider.label(),
            retry_count, "retried empty provider response"
        );
    }
    let provider_response_received_at = chrono::Utc::now();
    let provider_response_headers = provider_response.response_headers.clone();
    let provider_duration_ms =
        u64::try_from(provider_request_started.elapsed().as_millis()).unwrap_or(u64::MAX);
    drop(provider_timer);
    let _finalize_timer =
        engine_core::operational::Timer::new(engine_core::operational::Metric::GatewayFinalize);
    provider_span.record("reincore.provider_ms", provider_duration_ms as i64);
    provider_span.record("reincore.provider", provider.label());
    provider_span.record("reincore.model", provider_response.model.as_str());

    // STAGE: health
    let health_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Health);
    let reasoning_content = engine_core::extract_provider_reasoning(&provider_response.raw);

    let mut response_message = provider_response.message.clone();
    response_message.provider_state = response_message
        .provider_state
        .take()
        .map(|state| state.for_request_group(&request_group_id));
    // Warn when the provider returned a completely empty response - no
    // content and no tool calls.  This is the upstream cause of the VS Code
    // "Sorry, no response was returned" UX.  Logging here makes the silent
    // failure observable on the server side.
    if response_message.content.is_none() && response_message.tool_calls.is_none() {
        warn!(
            provider = provider.label(),
            model = %provider_response.model,
            "provider returned an empty response (no content, no tool calls) - \
             client will see no output",
        );
    }

    // The prompt is still running while the model keeps requesting tools.
    if let Some(key) = engine_core::request_group_key(session_id, &req) {
        engine_core::mark_group_continues(
            &key,
            engine_core::finish_reason_for(&response_message) == "tool_calls",
        );
    }

    let action_health = engine_core::action_gate::gate_provider_response(
        &provider_request,
        &mut response_message,
        &provider_response.raw,
        &state.action_policy,
    );
    let (action_health, _) = engine_core::action_recovery::recover(
        &provider_request,
        &mut response_message,
        action_health,
        &state.action_policy,
        provider.as_ref(),
        &request_group_id,
        session_id.map(|id| {
            (
                state.db.as_ref() as &dyn engine_core::EventStore,
                state.db.as_ref() as &dyn engine_core::accounting::AccountingStore,
                id,
            )
        }),
    )
    .await;
    if let Some(key) = engine_core::request_group_key(session_id, &req) {
        engine_core::mark_group_continues(
            &key,
            !action_health.should_stop
                && engine_core::finish_reason_for(&response_message) == "tool_calls",
        );
    }
    if action_health.should_stop {
        response_message.content = Some(engine_core::MessageContent::text(format!(
            "{}\n{}",
            response_message.full_text(),
            engine_core::action_gate::blocked_action_message(&action_health)
        )));
        if let Some(key) = engine_core::request_group_key(session_id, &req) {
            engine_core::mark_group_continues(&key, false);
        }
    }
    agent_health.merge(action_health);
    agent_health.merge(
        engine_core::claim_grounding::evaluate(
            &provider_request,
            &response_message,
            &state.claim_grounding,
            provider.as_ref(),
            session_id.map(|id| {
                (
                    state.db.as_ref() as &dyn engine_core::accounting::AccountingStore,
                    id,
                )
            }),
        )
        .await,
    );

    if let Some(attachments) = citation_attachments.as_deref() {
        agent_health.merge(engine_core::analyze_response_citations_with_request(
            &response_message,
            attachments,
            &provider_request.messages,
        ));
    }
    engine_core::apply_intervention_policy(&mut agent_health);
    health_span.record(
        "reincore.health_findings",
        agent_health.findings.len() as i64,
    );
    drop(health_span);

    // Use provider-reported token counts when available. Otherwise expose the
    // predicted counts as a fallback while keeping cost explicitly unknown, so
    // estimated tokens are not treated as observed billable usage.
    let usage_reported = provider_response.usage.is_some() && all_retry_usage_reported;
    let actual_usage = provider_response
        .usage
        .unwrap_or(engine_core::TokenUsage {
            prompt_tokens: prediction.input_tokens,
            completion_tokens: prediction.output_tokens,
            total_tokens: prediction.input_tokens + prediction.output_tokens,
            cached_prompt_tokens: 0,
            cache_creation_input_tokens: 0,
            ..Default::default()
        })
        .saturating_add(retry_usage);
    provider_span.record(
        "gen_ai.usage.input_tokens",
        actual_usage.prompt_tokens as i64,
    );
    provider_span.record(
        "gen_ai.usage.output_tokens",
        actual_usage.completion_tokens as i64,
    );
    drop(provider_span);

    let cost = cost_predictor
        .with_model(provider_response.model.clone())
        .estimate_usage(&actual_usage);
    let cost = if usage_reported {
        cost
    } else {
        engine_core::pricing::CostEstimate::unknown(
            &cost.provider,
            &cost.model,
            "provider did not return usage",
        )
    };

    log_provider(
        provider.label(),
        &format!(
            "{}  model={}, {} prompt / {} completion / {} total tokens (predicted: {})",
            "Actual cost:".green().bold(),
            provider_response.model.green(),
            actual_usage.prompt_tokens,
            actual_usage.completion_tokens,
            actual_usage.total_tokens,
            prediction.range_dollars().yellow(),
        ),
    );

    let group_snapshot = record_request_group_step(
        &request_group_id,
        actual_usage.prompt_tokens,
        actual_usage.completion_tokens,
        cost.amount_nano_usd,
    )
    .await;

    let payload = engine_core::event::LlmCallPayload {
        tool_schema_hash: engine_core::tool_schema_hash(&provider_request.tools).ok(),
        request_messages: engine_core::compact_audit_request_messages(&provider_request.messages),
        response_message: response_message.clone(),
        reasoning_content,
        provider_request_sent_at: Some(provider_request_sent_at),
        provider_response_received_at: Some(provider_response_received_at),
        provider_duration_ms: Some(provider_duration_ms),
        model: provider_response.model.clone(),
        model_tier: model_tier.as_str().to_string(),
        prefix_hash: Some(prefix.prefix_hash.clone()),
        input_tokens: actual_usage.prompt_tokens,
        output_tokens: actual_usage.completion_tokens,
        cached_prompt_tokens: actual_usage.cached_prompt_tokens,
        cache_creation_input_tokens: actual_usage.cache_creation_input_tokens,
        batch_eligible: batch_requested,
        batch_submitted: false,
        batch_id: None,
        predicted_cost_low_cents: Some(prediction.low_cents),
        predicted_cost_high_cents: Some(prediction.high_cents),
        // Set by `record_completion_accounting` once the actual cost is known;
        // initialized to None here so the struct literal is exhaustive.
        cost_accuracy_bps: None,
        cache_hit_rate_bps: None,
        classification: routing.classification.clone(),
        request_group_id: Some(request_group_id.clone()),
        task_id: Some(task.task_id.clone()),
        task_name: Some(task.task_name.clone()),
        task_type: Some(task.task_type.as_str().to_string()),
        context_adjustment: Some(context_adjustment.clone()),
        message_roles: Some(engine_core::user_content::annotate_message_roles(
            &provider_request.messages,
        )),
        is_subagent: extra_bool(&req, "reincore_is_subagent").unwrap_or(false),
        tool_results_deduped: Some(deduped_tool_results.new_replacements),
        // Stamped by `record_completion_accounting`, which scores the window.
        exploration_redundancy: None,
        semantic_cache_similarity_bps: None,
        semantic_best_candidate_bps: engine_core::best_candidate_similarity_bps(
            &semantic_candidates,
        ),
        semantic_best_candidate_entry_id: engine_core::best_candidate_entry_id(
            &semantic_candidates,
        ),
    };

    // Feedback loop: compare the predicted high bound against the actual
    // billed cost. Free providers (predicted high == 0) return None and are
    // not calibratable, so their session row and the cloud aggregation skip
    // them rather than recording a misleading infinity.
    let accuracy_bps = cost.accuracy_bps(prediction.high_cents);
    let cache_hit_bps = engine_core::cache_hit_rate_bps(
        actual_usage.cached_prompt_tokens,
        actual_usage.prompt_tokens,
    );

    // Persist the completion receipt and settle its event, ledger, and session
    // usage before returning success. If required accounting fails, propagate the
    // error instead of proceeding to response-cache write-back or prewarming.
    // STAGE: account
    let account_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Account);
    let loop_intervention_scheduled = record_completion_accounting(
        state.db.clone(),
        session_id,
        payload,
        &agent_health,
        cost.clone(),
        actual_usage.prompt_tokens,
        actual_usage.completion_tokens,
        accuracy_bps,
        cache_hit_bps,
        engine_core::task_decisions::completion_context(&provider_request),
        &loop_breaker,
    )
    .await?;
    engine_core::telemetry::record_cost(&account_span, &cost);
    account_span.record(
        "reincore.actual_to_predicted_high_ratio_bps",
        accuracy_bps.map(|v| v as i64).unwrap_or(-1),
    );
    drop(account_span);

    // Predictive pre-warming: if the session just transitioned into a
    // degradation tier, warm the Lite model connection and pre-fetch
    // likely-needed tool results in the background. Best-effort, never
    // blocks the response.
    trigger_prewarm_if_tier_changed(
        state.db.clone(),
        session_id,
        prewarm_expected,
        prewarm_provider,
        prewarm_provider_config,
    );

    let response_cacheable = engine_core::chat_message_has_visible_output(&response_message);
    let response_json = json!({
        "role": "assistant",
        "content": response_message.content,
        "tool_calls": response_message.tool_calls,
        "provider_state": response_message.provider_state,
    });

    if let Some(plan) = semantic.filter(|_| response_cacheable) {
        store_semantic_in_background(state.db.clone(), plan, response_json.clone());
    }

    let mut response_json = json!({
        "id": provider_response.id,
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp(),
        "model": provider_response.model,
        "reincore_predicted_cost": prediction.range_dollars(),
        "reincore_actual_cost_cents": cost.cents(),
        "reincore_predicted_cost_high_cents": prediction.high_cents,
        "reincore_actual_to_predicted_high_ratio_bps": accuracy_bps,
            "reincore_cost": cost,
        "reincore_cache_hit_rate_bps": cache_hit_bps,
        "reincore_provider_retry_count": retry_count,
        "reincore_routing_reason": routing.reason,
        "reincore_classification": routing.classification,
        "reincore_routing_adjustment": resolved_tier.adjustment,
        "reincore_budget_tier": budget_routing.projected_tier.as_u8(),
        "reincore_model_degraded": budget_routing.model_was_degraded(),
        "reincore_prefix_hash": prefix.prefix_hash,
        "reincore_batch_eligible": batch_requested,
        "reincore_batch_submitted": false,
        "reincore_cached_prompt_tokens": actual_usage.cached_prompt_tokens,
        "reincore_cache_creation_input_tokens": actual_usage.cache_creation_input_tokens,
        "reincore_context": context_adjustment,
        "reincore_context_reset": context_reset,
        "reincore_session_id": session_id,
        "reincore_session_rolled_over_from": rolled_over_from,
        "reincore_session_handoff": session_handoff,
        "reincore_tool_calls_validated": !agent_health.should_stop,
        "reincore_agent_health": agent_health,
        "reincore_loop_nudge_injected": loop_nudge_injected,
        "reincore_loop_intervention_scheduled": loop_intervention_scheduled,
        "reincore_tool_results_deduped": deduped_tool_results.replaced,
        "reincore_semantic_retrieval_candidates": semantic_candidates,
        "choices": [{
            "index": 0,
            "message": response_message,
            "finish_reason": engine_core::finish_reason_for(&response_message)
        }],
        "usage": {
            "prompt_tokens": actual_usage.prompt_tokens,
            "completion_tokens": actual_usage.completion_tokens,
            "total_tokens": actual_usage.total_tokens,
            "prompt_tokens_details": {
                "cached_tokens": actual_usage.cached_prompt_tokens
            }
        }
    });
    attach_request_group_metadata(&mut response_json, &group_snapshot);
    attach_task_metadata(&mut response_json, &task, task_context_messages_removed);
    if let Some(response_id) = response_json.get("id").and_then(|value| value.as_str()) {
        link_response_to_request_group(response_id, &request_group_id).await;
    }
    print_request_group_rollup(&group_snapshot);

    // If the client requested streaming, convert the response to SSE.
    if req.stream.unwrap_or(false) {
        let sse_stream = chat_completion_to_sse_stream(response_json);
        let response = Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response();
        return Ok(with_provider_rate_limit_headers(
            response,
            &provider_response_headers,
        ));
    }

    Ok(with_provider_rate_limit_headers(
        Json(response_json).into_response(),
        &provider_response_headers,
    ))
}

pub async fn chat_completions_single_provider(
    State(state): State<SingleProviderAppState>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, StatusCode> {
    let _handler_timer =
        engine_core::operational::Timer::new(engine_core::operational::Metric::ChatHandler);
    engine_core::operational::increment(engine_core::operational::Counter::ChatRequests, 1);
    let request_span =
        engine_core::telemetry::pipeline_root_span("chat_completions_single_provider", None);
    request_span.record("http.request.method", "POST");
    request_span.record("http.route", "/v1/chat/completions");
    engine_core::telemetry::set_parent_from_trace_headers(
        &request_span,
        headers
            .get("traceparent")
            .and_then(|value| value.to_str().ok()),
        headers
            .get("tracestate")
            .and_then(|value| value.to_str().ok()),
    );

    let result = chat_completions_single_provider_inner(state, headers, req)
        .instrument(request_span.clone())
        .await;
    let status = result
        .as_ref()
        .map(|response| response.status().as_u16())
        .unwrap_or_else(|status| status.as_u16());
    if status >= 400 {
        engine_core::operational::increment(engine_core::operational::Counter::ChatErrors, 1);
    }
    engine_core::telemetry::record_http_status(&request_span, status);
    result
}

async fn chat_completions_single_provider_inner(
    state: SingleProviderAppState,
    headers: HeaderMap,
    mut req: ChatCompletionRequest,
) -> Result<axum::response::Response, StatusCode> {
    if engine_core::user_content::is_client_compaction_request(&req) {
        return Ok((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(engine_core::user_content::client_compaction_error()),
        )
            .into_response());
    }
    if engine_core::user_content::is_client_reminder_observer_request(&req) {
        return Ok((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(engine_core::user_content::client_reminder_observer_error()),
        )
            .into_response());
    }
    // Measure elapsed request processing from before normalization and session
    // resolution, so timing includes preparation as well as provider execution.
    let request_started = std::time::Instant::now();
    // Historical Copilot/editor messages repeat volatile context and reminder
    // scaffolding. Keep the current wrapper for the provider, but reduce older
    // user turns to their literal request before session logic sees them.
    engine_core::task_decisions::clear_untrusted_context(&mut req);
    engine_core::claim_grounding::clear_untrusted_context(&mut req);
    engine_core::continuation::clear_untrusted_context(&mut req);
    engine_core::normalize_correlated_terminal_notifications(&mut req);
    engine_core::normalize_previous_user_messages(&mut req);

    // Resolve local file paths (file://... or bare paths) in multi-part
    // content to base64 data URLs before any cache lookup or provider
    // dispatch. This lets clients send local file references that the
    // engine reads from disk and encodes inline.
    if let Err(e) = req.resolve_local_files() {
        warn!(error = %e, "failed to resolve local file attachments");
        return Err(StatusCode::BAD_REQUEST);
    }

    // STAGE: resolve
    let resolve_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Resolve);

    let session_resolve_span = tracing::info_span!(
        parent: &resolve_span,
        "reincore.resolve.session"
    );
    // Resolve the active session and its persisted state before assigning the
    // request group, recording external tool results, or selecting task context.
    // Later budget checks and completion accounting use this session identity.
    let mut session_snapshot = resolve_local_session(
        state.db.as_ref(),
        &req,
        state.default_project_name.as_str(),
        state.default_project_path.as_str(),
    )
    .instrument(session_resolve_span)
    .await?;
    resolve_span.record(
        "reincore.session_id",
        session_snapshot
            .as_ref()
            .map(|s| s.id)
            .map(|id| id.to_string())
            .as_deref()
            .unwrap_or("")
            .to_string(),
    );

    let batch_requested = extra_bool(&req, "reincore_batch_eligible").unwrap_or_else(|| {
        session_snapshot
            .as_ref()
            .map(|session| session.mode == SessionMode::Background)
            .unwrap_or(false)
    });

    let mut session_id = session_snapshot.as_ref().map(|session| session.id);
    if let Some(session) = session_snapshot.as_ref() {
        engine_core::file_history::bind_request_roots(state.db.as_ref(), session, &req)
            .await
            .map_err(|error| match error {
                engine_core::EngineError::InvalidInput(_) => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            })?;
        state
            .sessions
            .lock()
            .await
            .insert(session.id, session.clone());
    }
    let group_resolve_span = tracing::info_span!(
        parent: &resolve_span,
        "reincore.resolve.request_group"
    );
    let request_group_id = resolve_request_group_id(state.db.as_ref(), None, &req, session_id)
        .instrument(group_resolve_span)
        .await?;
    resolve_span.record("reincore.request_group_id", request_group_id.as_str());
    engine_core::provider::prepare_request_replay(
        &mut req,
        &request_group_id,
        state.provider_replay_enabled,
    );
    if let Some(id) = session_id {
        let delegation_profile = delegation_profile(Some(&headers), &req);
        let track_subagent_sessions =
            extra_bool(&req, "reincore_track_subagent_sessions").unwrap_or(true);
        let external_tools_span = tracing::info_span!(
            parent: &resolve_span,
            "reincore.resolve.external_tools",
            message_count = req.messages.len()
        );
        record_external_tool_executions_on_request_path(
            state.db.clone(),
            session_snapshot.clone(),
            id,
            &req.messages,
            req.extra
                .get("reincore_file_evidence")
                .and_then(serde_json::Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            &request_group_id,
            delegation_profile.as_deref(),
            track_subagent_sessions,
        )
        .instrument(external_tools_span)
        .await?;
    }
    let task_resolve_span = tracing::info_span!(
        parent: &resolve_span,
        "reincore.resolve.task"
    );
    let meta_provider = (state.provider.label() == "meta").then(|| state.provider.clone());
    let detection_judge = detection_judge(meta_provider, &state.detection);
    let (task, task_context_messages_removed) = resolve_and_scope_task(
        state.db.as_ref(),
        session_id,
        &request_group_id,
        &mut req,
        state.task_decisions_enabled,
        &state.claim_grounding,
        &state.detection,
        detection_judge.as_ref(),
    )
    .instrument(task_resolve_span)
    .await?;
    super::titles::spawn_titles(
        state.db.clone(),
        state.provider.clone(),
        session_snapshot.as_ref(),
        &task,
        &req,
    );
    drop(resolve_span);

    let caller_semantic_embedding = parse_embedding(&req).map_err(|_| StatusCode::BAD_REQUEST)?;
    let semantic_scope = session_snapshot
        .as_ref()
        .and_then(|session| session.project_id)
        .map(|project_id| engine_core::SemanticCacheScope {
            workspace_id: "local".to_string(),
            project_id: project_id.to_string(),
        });

    let validation_signal = req
        .extra
        .get("reincore_validation_signal")
        .and_then(|v| v.as_str());

    // STAGE: classify
    let classify_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Classify);
    let classifier = if let Some(id) = session_id {
        state
            .classifier
            .trained_for_session(state.db.as_ref(), id)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    } else {
        state.classifier.as_ref().clone()
    };
    let routing = classifier
        .classify(&req, validation_signal)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    // Retain the classifier's recommended model tier for initial pricing and
    // diagnostics. Budget degradation and routing overrides may select a different
    // effective tier later, without replacing this original recommendation.
    let classified_model_tier = routing.model_tier;
    classify_span.record("reincore.classified_tier", classified_model_tier.as_str());
    drop(classify_span);

    // STAGE: route (provider/model selection)
    let route_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Route);

    let classified_model = state
        .provider
        .config()
        .model_for_tier(classified_model_tier)
        .map_err(|error| {
            warn!(
                provider = state.provider.label(),
                tier = classified_model_tier.as_str(),
                %error,
                "failed to resolve classified model from local provider configuration"
            );
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let initial_cost_predictor = (*state.cost_predictor).clone().with_model(classified_model);
    drop(route_span);

    // STAGE: predict (initial)
    let predict_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Predict);
    let initial_prediction = initial_cost_predictor
        .predict(&req, classified_model_tier)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    drop(predict_span);

    // STAGE: route (budget degradation)
    let route_span_sec = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Route);
    // Compare recorded spend across the session lineage and the predicted call
    // cost with budget thresholds. Exhausted actual spend pauses at a checkpoint;
    // projected spend can lower the model tier without triggering that pause.
    let budget_routing = resolve_budget_routing(
        session_snapshot.as_ref(),
        classified_model_tier,
        initial_prediction.high_cents,
    )
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    route_span_sec.record(
        "reincore.budget_tier",
        budget_routing.projected_tier.as_u8() as i64,
    );

    // Capture the pre-call tier for predictive pre-warming.
    let prewarm_expected =
        budget_routing.current_tier.as_u8() < 2 && budget_routing.projected_tier.as_u8() >= 2;
    let prewarm_provider = state.provider.clone();
    let prewarm_provider_config = state.provider.config().clone();

    if budget_routing.should_checkpoint {
        let mut session = session_snapshot.ok_or(StatusCode::CONFLICT)?;
        let last_seq = state
            .db
            .latest_seq(session.id)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .unwrap_or(0);
        let token = state
            .checkpoints
            .create(&mut session, last_seq, "budget exhausted")
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        state
            .db
            .insert_checkpoint_async(token.clone())
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        state
            .db
            .insert_session_async(session.clone())
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        let group_snapshot = record_request_group_step(&request_group_id, 0, 0, Some(0)).await;
        let mut checkpoint_json = json!({
            "object": "reincore.checkpoint",
            "reincore_paused": true,
            "reason": "budget exhausted",
            "session_id": session.id,
            "checkpoint_token": token.token,
            "last_event_seq": token.last_event_seq,
            "expires_at": token.expires_at,
        });
        attach_request_group_metadata(&mut checkpoint_json, &group_snapshot);
        attach_task_metadata(&mut checkpoint_json, &task, task_context_messages_removed);
        print_request_group_rollup(&group_snapshot);

        return Ok(Json(checkpoint_json).into_response());
    }

    let budgeted_model_tier = budget_routing
        .effective_model_tier
        .ok_or(StatusCode::CONFLICT)?;

    // Apply the routing policy + per-request escape hatches
    // (reincore_routing_mode / reincore_minimum_tier / reincore_maximum_tier)
    // to the classified + budget-degraded tier.
    let routing_overrides = engine_core::RequestRoutingOverrides::from_extra(&req.extra);
    let resolved_tier = engine_core::resolve_routing_tier(
        state.provider.config().routing_policy(),
        &routing_overrides,
        classified_model_tier,
        budgeted_model_tier,
    )
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let model_tier = resolved_tier.tier;
    route_span_sec.record("reincore.final_tier", model_tier.as_str());
    drop(route_span_sec);
    // Loop-detection configuration, least to most specific: built-in defaults,
    // then `config.toml`, then `REINCORE_LOOP_*`, then stored settings, then
    // this request's `reincore_loop_*` overrides.
    let loop_breaker = CircuitBreaker::new(
        engine_core::LoopSettings::clone(&state.loop_detection)
            .overlay(engine_core::LoopSettings::from_env())
            .overlay(state.db.get_loop_settings().await.unwrap_or_default())
            .overlay(engine_core::LoopSettings::from_extra(&req.extra))
            .resolve(),
    );

    // Load any intervention scheduled by earlier loop-health checks. A hard stop
    // returns a refusal; a soft intervention adds a system nudge to the request.
    // Check before compaction and cache lookup so answer reuse cannot bypass it.
    let loop_nudge = if let Some(id) = session_id {
        loop_breaker
            .pending_intervention(state.db.as_ref(), id)
            .await
            .unwrap_or(None)
    } else {
        None
    };

    if let Some(intervention) = loop_nudge.as_ref().filter(|item| item.hard_stop) {
        let findings = match session_id {
            Some(id) => {
                let _ = loop_breaker.clear_intervention(state.db.as_ref(), id).await;
                engine_core::recent_health_findings(state.db.as_ref(), id, 32)
                    .await
                    .unwrap_or_default()
            }
            None => Vec::new(),
        };
        return Ok(loop_hard_stop_response(session_id, intervention, findings));
    }

    let loop_nudge_injected = loop_nudge.is_some();
    if let Some(intervention) = loop_nudge {
        inject_system_nudge(&mut req, intervention.message);
    }

    // Replace read/search results the transcript already carries with a pointer
    // to the earlier copy. This runs before compaction and the final prediction
    // so the saved tokens are real, and it gives a re-reading model something to
    // act on rather than only counting the repeat toward a refusal.
    let deduped_tool_results = loop_breaker.dedupe_tool_results(&mut req);
    if !deduped_tool_results.is_empty() {
        tracing::debug!(
            replaced = deduped_tool_results.replaced,
            saved_chars = deduped_tool_results.saved_chars,
            "replaced duplicate tool results with pointers to the earlier copy"
        );
    }

    let settings = state
        .db
        .get_compression_settings()
        .await
        .unwrap_or_default();

    // STAGE: compact
    let compact_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Compact);

    // Preserve prompt-injection findings from tool results that compaction may
    // replace with a bounded digest, then scan the compiled request as well.
    // Retain these findings in agent_health and merge the later scan of prepared
    // context, so replacing suspicious tool output with a summary does not erase
    // the evidence included in completion health reporting.
    engine_core::task_decisions::detach(&mut req);
    let mut agent_health = engine_core::scan_prompt_injection(&req);
    engine_core::continuation::detach(&mut req);
    let context_adjustment = match session_id {
        Some(id) if !engine_core::compression_turn_is_new(id, &request_group_id) => {
            engine_core::prepare_context_continuation(
                &mut req,
                settings,
                budget_routing.projected_tier,
            )
        }
        _ => prepare_context(&mut req, settings, budget_routing.projected_tier),
    };
    agent_health.merge(engine_core::scan_prompt_injection(&req));
    log_compression(state.provider.label(), &context_adjustment);
    let resolved_model = state
        .provider
        .config()
        .model_for_tier(model_tier)
        .map_err(|error| {
            warn!(
                provider = state.provider.label(),
                tier = model_tier.as_str(),
                %error,
                "failed to resolve effective model from local provider configuration"
            );
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let cost_predictor = (*state.cost_predictor)
        .clone()
        .with_model(resolved_model.clone());

    let citation_files = async {
        match session_snapshot.as_ref() {
            Some(session) => state.db.read_latest_files(session.id).await.map(Some),
            None => Ok(None),
        }
    };
    let (reported_context_limit, citation_attachments) = tokio::join!(
        state.provider.context_window(&resolved_model),
        citation_files
    );
    let citation_attachments =
        citation_attachments.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    // Prefer a context window the provider can report first-hand (Ollama
    // queries its own `/api/show`); fall back to the catalog, then the
    // REINCORE_CONTEXT_LIMIT env var (defaulting to 1_000_000). The old
    // hardcoded 128K fallback was too small - it caused premature session
    // rollovers whenever a model wasn't in the catalog, which is exactly
    // the behaviour we're trying to avoid.
    let context_limit = match reported_context_limit {
        Some(ctx) => ctx.max(1),
        None => find_model(state.provider.label(), &resolved_model)
            .map(|model| model.max_input_tokens.max(1) as u64)
            .unwrap_or_else(fallback_context_limit),
    };

    let (prepared_request, context_reset) =
        engine_core::continuation::enforce_context_window_async(
            req,
            cost_predictor.clone(),
            model_tier,
            context_limit,
        )
        .await
        .map_err(|error| match error {
            engine_core::EngineError::ContextWindowExceeded { .. } => StatusCode::PAYLOAD_TOO_LARGE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        })?;
    req = prepared_request;

    let mut rolled_over_from = None;
    let mut session_handoff: Option<SessionHandoff> = None;
    if let Some(session) = session_snapshot.take() {
        let (active, previous_id) =
            rollover_local_session_if_needed(state.db.as_ref(), session, &context_reset, &req)
                .await?;
        session_id = Some(active.id);
        engine_core::file_history::bind_request_roots(state.db.as_ref(), &active, &req)
            .await
            .map_err(|error| match error {
                engine_core::EngineError::InvalidInput(_) => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            })?;
        rolled_over_from = previous_id;
        if previous_id.is_some() {
            session_handoff = Some(context_reset.handoff.clone());
        }
        state
            .sessions
            .lock()
            .await
            .insert(active.id, active.clone());
    }
    compact_span.record("reincore.rolled_over", rolled_over_from.is_some());
    drop(compact_span);

    // STAGE: route (final model resolution)
    let route_span3 = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Route);

    // Canonicalize the prepared request's static prefix and tool schemas before
    // building the response-cache query. Combine the resulting prefix hash with
    // the effective provider/model and scope to avoid reusing incompatible answers.
    let prefix = CachePrefixNormalizer::new()
        .normalize_request(&mut req)
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    route_span3.record("reincore.prefix_hash", prefix.prefix_hash.as_str());
    drop(route_span3);
    // Hand the prefix identity to provider builders that can use it as a cache
    // routing key. The reincore_ prefix keeps it out of semantic-cache
    // fingerprints, so it never fragments otherwise identical rows.
    req.extra.insert(
        "reincore_prefix_hash".to_string(),
        serde_json::Value::String(prefix.prefix_hash.clone()),
    );

    let semantic = build_semantic_cache_plan(
        &req,
        caller_semantic_embedding,
        semantic_scope,
        state.provider.as_ref(),
        &resolved_model,
        &prefix.prefix_hash,
    )?;
    let semantic_lookup = if let Some(plan) = semantic.as_ref() {
        state
            .db
            .get_semantic(
                &plan
                    .query(chrono::Utc::now())
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
            )
            .await
            .unwrap_or_default()
    } else {
        engine_core::SemanticCacheLookup::default()
    };
    if let Some(mut hit) = semantic_lookup.reusable_answer.filter(|hit| {
        serde_json::from_value::<ChatMessage>(hit.response.clone()).is_ok_and(|message| {
            !message.full_text().is_empty()
                && !message
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
        })
    }) {
        if let Some(message) = hit.response.as_object_mut() {
            message.remove("provider_state");
        }
        if let Ok(message) = serde_json::from_value::<ChatMessage>(hit.response.clone()) {
            agent_health.merge(engine_core::claim_grounding::analyze_health(&req, &message));
            if let Some(id) = session_id {
                let _ = engine_core::log_agent_health(state.db.as_ref(), id, &agent_health).await;
            }
        }
        record_semantic_cache_hit(
            state.db.as_ref(),
            session_id,
            &request_group_id,
            &task,
            &req,
            &hit.response,
            &resolved_model,
            &prefix.prefix_hash,
            engine_core::similarity_to_bps(hit.provenance.similarity),
        )
        .await?;
        let response_id = Uuid::new_v4().to_string();
        let mut response_json = semantic_hit_response(
            response_id.clone(),
            &resolved_model,
            session_id,
            hit.response,
        );
        response_json["reincore_agent_health"] =
            serde_json::to_value(&agent_health).unwrap_or_default();
        response_json["reincore_semantic_cache_provenance"] =
            serde_json::to_value(hit.provenance).unwrap_or_default();
        let group_snapshot = record_request_group_step(&request_group_id, 0, 0, Some(0)).await;
        attach_request_group_metadata(&mut response_json, &group_snapshot);
        attach_task_metadata(&mut response_json, &task, task_context_messages_removed);
        link_response_to_request_group(&response_id, &request_group_id).await;
        print_request_group_rollup(&group_snapshot);
        if req.stream.unwrap_or(false) {
            return Ok(Sse::new(chat_completion_to_sse_stream(response_json))
                .keep_alive(KeepAlive::default())
                .into_response());
        }
        return Ok(Json(response_json).into_response());
    }
    let semantic_candidates = semantic_lookup.candidates;

    // STAGE: predict (final prediction on the effective model)
    let predict_span_sec =
        engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Predict);
    // Estimate cost for the effective model using the input-token count already
    // computed during context enforcement. This avoids recounting the prepared
    // input and supplies final diagnostics; it does not rerun budget routing.
    let prediction = cost_predictor
        .predict_with_input_tokens(&req, model_tier, context_reset.input_tokens_after)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    predict_span_sec.record("reincore.predicted_cost_high_cents", prediction.high_cents);
    predict_span_sec.record(
        "reincore.predicted_input_tokens",
        prediction.input_tokens as i64,
    );
    drop(predict_span_sec);

    log_provider(
        state.provider.label(),
        &format!(
            "{}  {} input tokens, tier: {}, model: {}, reasoning: {}",
            format!("Predicted cost: {}", prediction.range_dollars()).yellow(),
            prediction.input_tokens,
            prediction.model_tier.as_str().cyan(),
            resolved_model.green(),
            prediction.reasoning.dimmed(),
        ),
    );

    {
        let db = state.db.clone();
        let group_id = request_group_id.clone();
        let predicted_tier = prediction.model_tier.as_str().to_string();
        let (input, output, low, high) = (
            prediction.input_tokens,
            prediction.output_tokens,
            prediction.low_cents,
            prediction.high_cents,
        );
        tokio::spawn(async move {
            let _ = db
                .insert_cost_estimate_async(
                    session_id,
                    Some(group_id),
                    "llm_call".to_string(),
                    predicted_tier,
                    input,
                    output,
                    low,
                    high,
                )
                .await;
        });
    }

    let mut provider_request = req.clone();
    provider_request.extra.insert(
        "reincore_model_tier".to_string(),
        serde_json::json!(model_tier.as_str()),
    );

    // Submit eligible non-streaming requests as durable batch jobs and return
    // their submission status. Do not fall back to a live call after an uncertain
    // submission: the provider may already have accepted and billed the batch.
    if batch_requested && !req.stream.unwrap_or(false) {
        match engine_core::batch_jobs::submit_chat(
            state.db.clone(),
            state.provider.clone(),
            session_id,
            &provider_request,
        )
        .await
        {
            Ok(batch) => {
                let batch_factor = engine_core::batch_jobs::prediction_factor(
                    state.provider.label(),
                    &resolved_model,
                    prediction.input_tokens,
                );
                let batch_low_cents =
                    (prediction.low_cents as f64 * batch_factor.unwrap_or(1.0)).ceil() as i64;
                let batch_high_cents =
                    (prediction.high_cents as f64 * batch_factor.unwrap_or(1.0)).ceil() as i64;
                if batch.status != "submission_unknown" && batch.status != "submitting" {
                    record_batch_submission(
                        state.db.as_ref(),
                        session_id,
                        &request_group_id,
                        &task,
                        &provider_request,
                        &resolved_model,
                        model_tier,
                        &prefix.prefix_hash,
                        &batch.batch_id,
                        batch_low_cents,
                        batch_high_cents,
                        routing.classification.clone(),
                        context_adjustment.clone(),
                    )
                    .await?;
                }
                let group_snapshot =
                    record_request_group_step(&request_group_id, 0, 0, Some(0)).await;
                let mut batch_json = json!({
                    "id": batch.batch_id,
                    "object": "reincore.batch",
                    "created": chrono::Utc::now().timestamp(),
                    "model": resolved_model,
                    "reincore_batch_eligible": true,
                    "reincore_batch_submitted": batch.status != "submission_unknown" && batch.status != "submitting",
                    "reincore_batch_status": batch.status,
                    "reincore_batch_discount_factor": batch_factor,
                    "reincore_predicted_cost": cents_range_dollars(batch_low_cents, batch_high_cents),
                    "reincore_predicted_cost_low_cents": batch_low_cents,
                    "reincore_predicted_cost_high_cents": batch_high_cents,
                    "reincore_routing_reason": routing.reason,
                    "reincore_classification": routing.classification,
                    "reincore_routing_adjustment": resolved_tier.adjustment,
                    "reincore_budget_tier": budget_routing.projected_tier.as_u8(),
                    "reincore_model_degraded": budget_routing.model_was_degraded(),
                    "reincore_session_id": session_id,
                    "reincore_session_rolled_over_from": rolled_over_from,
                });
                attach_request_group_metadata(&mut batch_json, &group_snapshot);
                attach_task_metadata(&mut batch_json, &task, task_context_messages_removed);
                print_request_group_rollup(&group_snapshot);
                return Ok((StatusCode::ACCEPTED, Json(batch_json)).into_response());
            }
            Err(error) => {
                warn!(
                    provider = state.provider.label(),
                    %error,
                    "batch submission failed; synchronous fallback is prohibited"
                );
                return Err(StatusCode::BAD_GATEWAY);
            }
        }
    }

    // Record provider-call timing, then choose native streaming when supported
    // or a complete response otherwise. The live SSE body takes ownership of
    // response assembly and accounting, which run as that body is consumed.
    let provider_request_sent_at = chrono::Utc::now();
    let provider_request_started = std::time::Instant::now();
    engine_core::operational::observe(
        engine_core::operational::Metric::GatewayPrepare,
        request_started.elapsed(),
    );
    let provider_timer =
        engine_core::operational::Timer::new(engine_core::operational::Metric::ProviderOperation);

    // STAGE: provider
    let provider_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Provider);

    if req.stream.unwrap_or(false) && state.provider.supports_streaming() {
        let upstream = match state
            .provider
            .complete_stream(&provider_request)
            .instrument(provider_span.clone())
            .await
        {
            Ok(stream) => stream,
            Err(error) if error.is_context_overflow_error() => {
                warn!(
                    provider = state.provider.label(),
                    %error,
                    "provider reported a full context window while opening stream; rolling the session over"
                );
                let recovery = recover_from_provider_overflow(
                    state.db.as_ref(),
                    session_id,
                    &mut provider_request,
                    context_limit,
                )
                .await?;
                if let Some(active) = recovery.session {
                    session_id = Some(active.id);
                    engine_core::file_history::bind_request_roots(state.db.as_ref(), &active, &req)
                        .await
                        .map_err(|error| match error {
                            engine_core::EngineError::InvalidInput(_) => StatusCode::BAD_REQUEST,
                            _ => StatusCode::INTERNAL_SERVER_ERROR,
                        })?;
                    state.sessions.lock().await.insert(active.id, active);
                }
                state
                    .provider
                    .complete_stream(&provider_request)
                    .instrument(provider_span.clone())
                    .await
                    .map_err(|error| provider_error_to_status(&error))?
            }
            Err(error) => {
                engine_core::telemetry::record_error(
                    &provider_span,
                    "provider_error",
                    &engine_core::privacy::current_policy()
                        .sanitize_for_logging(&error.to_string()),
                );
                return Err(provider_error_to_status(&error));
            }
        };

        let provider_response_headers = upstream.response_headers.clone();
        let stream_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Stream);
        stream_span.record("reincore.provider", state.provider.label());
        stream_span.record("reincore.model", resolved_model.as_str());
        let stream = live_chat_completion_to_sse_stream(
            upstream,
            LocalStreamContext {
                action_policy: state.action_policy.clone(),
                claim_grounding: state.claim_grounding.clone(),
                db: state.db.clone(),
                session_id,
                request_group_id,
                task,
                task_context_messages_removed,
                original_request: req,
                provider_request,
                provider_label: state.provider.label().to_string(),
                detection: state.detection.clone(),
                detection_judge,
                resolved_model,
                model_tier,
                prefix_hash: prefix.prefix_hash,
                prediction,
                cost_predictor,
                classification: routing.classification,
                batch_requested,
                semantic,
                semantic_candidates,
                citation_attachments,
                initial_agent_health: agent_health,
                prewarm_expected,
                prewarm_provider,
                prewarm_provider_config,
                provider_request_sent_at,
                provider_request_started,
                request_started,
                provider_timer,
                context_adjustment,
                deduped_tool_results,
                loop_breaker,
                provider_span: provider_span.clone(),
                stream_span,
            },
        );
        let stream = engine_core::telemetry::instrument_stream(
            stream,
            tracing::Span::current(),
            vec![provider_span],
        );
        let response = Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response();
        return Ok(with_provider_rate_limit_headers(
            response,
            &provider_response_headers,
        ));
    }

    let provider_response = match state
        .provider
        .complete(&provider_request)
        .instrument(provider_span.clone())
        .await
    {
        Ok(response) => response,
        Err(error) if error.is_context_overflow_error() => {
            warn!(
                provider = state.provider.label(),
                %error,
                "provider reported a full context window; rolling the session over"
            );
            let recovery = recover_from_provider_overflow(
                state.db.as_ref(),
                session_id,
                &mut provider_request,
                context_limit,
            )
            .await?;
            if let Some(active) = recovery.session {
                session_id = Some(active.id);
                engine_core::file_history::bind_request_roots(state.db.as_ref(), &active, &req)
                    .await
                    .map_err(|error| match error {
                        engine_core::EngineError::InvalidInput(_) => StatusCode::BAD_REQUEST,
                        _ => StatusCode::INTERNAL_SERVER_ERROR,
                    })?;
                rolled_over_from = recovery.rolled_over_from;
                state
                    .sessions
                    .lock()
                    .await
                    .insert(active.id, active.clone());
            }
            session_handoff = Some(recovery.handoff);
            state
                .provider
                .complete(&provider_request)
                .instrument(provider_span.clone())
                .await
                .map_err(|e| provider_error_to_status(&e))?
        }
        Err(error) => {
            engine_core::telemetry::record_error(
                &provider_span,
                "provider_error",
                &engine_core::privacy::current_policy().sanitize_for_logging(&error.to_string()),
            );
            return Err(provider_error_to_status(&error));
        }
    };
    let engine_core::ProviderResponseAttempts {
        response: provider_response,
        prior_usage: retry_usage,
        all_usage_reported: all_retry_usage_reported,
        retry_count,
    } = engine_core::retry_empty_provider_response(
        state.provider.as_ref(),
        &provider_request,
        provider_response,
    )
    .instrument(provider_span.clone())
    .await
    .map_err(|error| provider_error_to_status(&error))?;
    if retry_count > 0 {
        warn!(
            provider = state.provider.label(),
            retry_count, "retried empty provider response"
        );
    }
    let provider_response_received_at = chrono::Utc::now();
    let provider_response_headers = provider_response.response_headers.clone();
    let provider_duration_ms =
        u64::try_from(provider_request_started.elapsed().as_millis()).unwrap_or(u64::MAX);
    drop(provider_timer);
    let _finalize_timer =
        engine_core::operational::Timer::new(engine_core::operational::Metric::GatewayFinalize);
    provider_span.record("reincore.provider_ms", provider_duration_ms as i64);
    provider_span.record("reincore.provider", state.provider.label());
    provider_span.record("reincore.model", provider_response.model.as_str());

    // STAGE: health
    let health_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Health);
    let reasoning_content = engine_core::extract_provider_reasoning(&provider_response.raw);

    let mut response_message = provider_response.message.clone();
    response_message.provider_state = response_message
        .provider_state
        .take()
        .map(|state| state.for_request_group(&request_group_id));
    // Warn when the provider returned a completely empty response - no
    // content and no tool calls.  This is the upstream cause of the VS Code
    // "Sorry, no response was returned" UX.  Logging here makes the silent
    // failure observable on the server side.
    if response_message.content.is_none() && response_message.tool_calls.is_none() {
        warn!(
            provider = state.provider.label(),
            model = %provider_response.model,
            "provider returned an empty response (no content, no tool calls) - \
             client will see no output",
        );
    }

    // The prompt is still running while the model keeps requesting tools.
    if let Some(key) = engine_core::request_group_key(session_id, &req) {
        engine_core::mark_group_continues(
            &key,
            engine_core::finish_reason_for(&response_message) == "tool_calls",
        );
    }

    let action_health = engine_core::action_gate::gate_provider_response(
        &provider_request,
        &mut response_message,
        &provider_response.raw,
        &state.action_policy,
    );
    let (action_health, _) = engine_core::action_recovery::recover(
        &provider_request,
        &mut response_message,
        action_health,
        &state.action_policy,
        state.provider.as_ref(),
        &request_group_id,
        session_id.map(|id| {
            (
                state.db.as_ref() as &dyn engine_core::EventStore,
                state.db.as_ref() as &dyn engine_core::accounting::AccountingStore,
                id,
            )
        }),
    )
    .await;
    if let Some(key) = engine_core::request_group_key(session_id, &req) {
        engine_core::mark_group_continues(
            &key,
            !action_health.should_stop
                && engine_core::finish_reason_for(&response_message) == "tool_calls",
        );
    }
    if action_health.should_stop {
        response_message.content = Some(engine_core::MessageContent::text(format!(
            "{}\n{}",
            response_message.full_text(),
            engine_core::action_gate::blocked_action_message(&action_health)
        )));
        if let Some(key) = engine_core::request_group_key(session_id, &req) {
            engine_core::mark_group_continues(&key, false);
        }
    }
    agent_health.merge(action_health);
    agent_health.merge(
        engine_core::claim_grounding::evaluate(
            &provider_request,
            &response_message,
            &state.claim_grounding,
            state.provider.as_ref(),
            session_id.map(|id| {
                (
                    state.db.as_ref() as &dyn engine_core::accounting::AccountingStore,
                    id,
                )
            }),
        )
        .await,
    );

    if let Some(attachments) = citation_attachments.as_deref() {
        agent_health.merge(engine_core::analyze_response_citations_with_request(
            &response_message,
            attachments,
            &provider_request.messages,
        ));
    }
    engine_core::apply_intervention_policy(&mut agent_health);
    health_span.record(
        "reincore.health_findings",
        agent_health.findings.len() as i64,
    );
    drop(health_span);

    // Use provider-reported token counts when available. Otherwise expose the
    // predicted counts as a fallback while keeping cost explicitly unknown, so
    // estimated tokens are not treated as observed billable usage.
    let usage_reported = provider_response.usage.is_some() && all_retry_usage_reported;
    let actual_usage = provider_response
        .usage
        .unwrap_or(engine_core::TokenUsage {
            prompt_tokens: prediction.input_tokens,
            completion_tokens: prediction.output_tokens,
            total_tokens: prediction.input_tokens + prediction.output_tokens,
            cached_prompt_tokens: 0,
            cache_creation_input_tokens: 0,
            ..Default::default()
        })
        .saturating_add(retry_usage);
    provider_span.record(
        "gen_ai.usage.input_tokens",
        actual_usage.prompt_tokens as i64,
    );
    provider_span.record(
        "gen_ai.usage.output_tokens",
        actual_usage.completion_tokens as i64,
    );
    drop(provider_span);

    let cost = cost_predictor
        .with_model(provider_response.model.clone())
        .estimate_usage(&actual_usage);
    let cost = if usage_reported {
        cost
    } else {
        engine_core::pricing::CostEstimate::unknown(
            &cost.provider,
            &cost.model,
            "provider did not return usage",
        )
    };

    log_provider(
        state.provider.label(),
        &format!(
            "{}  model={}, {} prompt / {} completion / {} total tokens (predicted: {})",
            "Actual cost:".green().bold(),
            provider_response.model.green(),
            actual_usage.prompt_tokens,
            actual_usage.completion_tokens,
            actual_usage.total_tokens,
            prediction.range_dollars().yellow(),
        ),
    );

    let group_snapshot = record_request_group_step(
        &request_group_id,
        actual_usage.prompt_tokens,
        actual_usage.completion_tokens,
        cost.amount_nano_usd,
    )
    .await;

    let payload = engine_core::event::LlmCallPayload {
        tool_schema_hash: engine_core::tool_schema_hash(&provider_request.tools).ok(),
        request_messages: engine_core::compact_audit_request_messages(&provider_request.messages),
        response_message: response_message.clone(),
        reasoning_content,
        provider_request_sent_at: Some(provider_request_sent_at),
        provider_response_received_at: Some(provider_response_received_at),
        provider_duration_ms: Some(provider_duration_ms),
        model: provider_response.model.clone(),
        model_tier: model_tier.as_str().to_string(),
        prefix_hash: Some(prefix.prefix_hash.clone()),
        input_tokens: actual_usage.prompt_tokens,
        output_tokens: actual_usage.completion_tokens,
        cached_prompt_tokens: actual_usage.cached_prompt_tokens,
        cache_creation_input_tokens: actual_usage.cache_creation_input_tokens,
        batch_eligible: batch_requested,
        batch_submitted: false,
        batch_id: None,
        predicted_cost_low_cents: Some(prediction.low_cents),
        predicted_cost_high_cents: Some(prediction.high_cents),
        // Set by `record_completion_accounting` once the actual cost is known;
        // initialized to None here so the struct literal is exhaustive.
        cost_accuracy_bps: None,
        cache_hit_rate_bps: None,
        classification: routing.classification.clone(),
        request_group_id: Some(request_group_id.clone()),
        task_id: Some(task.task_id.clone()),
        task_name: Some(task.task_name.clone()),
        task_type: Some(task.task_type.as_str().to_string()),
        context_adjustment: Some(context_adjustment.clone()),
        message_roles: Some(engine_core::user_content::annotate_message_roles(
            &provider_request.messages,
        )),
        is_subagent: extra_bool(&req, "reincore_is_subagent").unwrap_or(false),
        tool_results_deduped: Some(deduped_tool_results.new_replacements),
        // Stamped by `record_completion_accounting`, which scores the window.
        exploration_redundancy: None,
        semantic_cache_similarity_bps: None,
        semantic_best_candidate_bps: engine_core::best_candidate_similarity_bps(
            &semantic_candidates,
        ),
        semantic_best_candidate_entry_id: engine_core::best_candidate_entry_id(
            &semantic_candidates,
        ),
    };

    // Feedback loop: compare the predicted high bound against the actual
    // billed cost. Free providers (predicted high == 0) return None and are
    // not calibratable, so their session row and the cloud aggregation skip
    // them rather than recording a misleading infinity.
    let accuracy_bps = cost.accuracy_bps(prediction.high_cents);
    let cache_hit_bps = engine_core::cache_hit_rate_bps(
        actual_usage.cached_prompt_tokens,
        actual_usage.prompt_tokens,
    );

    // Persist the completion receipt and settle its event, ledger, and session
    // usage before returning success. If required accounting fails, propagate the
    // error instead of proceeding to response-cache write-back or prewarming.
    // STAGE: account
    let account_span = engine_core::telemetry::stage_span(engine_core::telemetry::Stage::Account);
    let loop_intervention_scheduled = record_completion_accounting(
        state.db.clone(),
        session_id,
        payload,
        &agent_health,
        cost.clone(),
        actual_usage.prompt_tokens,
        actual_usage.completion_tokens,
        accuracy_bps,
        cache_hit_bps,
        engine_core::task_decisions::completion_context(&provider_request),
        &loop_breaker,
    )
    .await?;
    engine_core::telemetry::record_cost(&account_span, &cost);
    account_span.record(
        "reincore.actual_to_predicted_high_ratio_bps",
        accuracy_bps.map(|v| v as i64).unwrap_or(-1),
    );
    drop(account_span);

    // Predictive pre-warming: if the session just transitioned into a
    // degradation tier, warm the Lite model connection and pre-fetch
    // likely-needed tool results in the background. Best-effort, never
    // blocks the response.
    trigger_prewarm_if_tier_changed(
        state.db.clone(),
        session_id,
        prewarm_expected,
        prewarm_provider,
        prewarm_provider_config,
    );

    let response_cacheable = engine_core::chat_message_has_visible_output(&response_message);
    let response_json = json!({
        "role": "assistant",
        "content": response_message.content,
        "tool_calls": response_message.tool_calls,
        "provider_state": response_message.provider_state,
    });

    if let Some(plan) = semantic.filter(|_| response_cacheable) {
        store_semantic_in_background(state.db.clone(), plan, response_json.clone());
    }

    let mut response_json = json!({
        "id": provider_response.id,
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp(),
        "model": provider_response.model,
        "reincore_predicted_cost": prediction.range_dollars(),
        "reincore_actual_cost_cents": cost.cents(),
        "reincore_predicted_cost_high_cents": prediction.high_cents,
        "reincore_actual_to_predicted_high_ratio_bps": accuracy_bps,
            "reincore_cost": cost,
        "reincore_cache_hit_rate_bps": cache_hit_bps,
        "reincore_provider_retry_count": retry_count,
        "reincore_routing_reason": routing.reason,
        "reincore_classification": routing.classification,
        "reincore_routing_adjustment": resolved_tier.adjustment,
        "reincore_budget_tier": budget_routing.projected_tier.as_u8(),
        "reincore_model_degraded": budget_routing.model_was_degraded(),
        "reincore_prefix_hash": prefix.prefix_hash,
        "reincore_batch_eligible": batch_requested,
        "reincore_batch_submitted": false,
        "reincore_cached_prompt_tokens": actual_usage.cached_prompt_tokens,
        "reincore_cache_creation_input_tokens": actual_usage.cache_creation_input_tokens,
        "reincore_context": context_adjustment,
        "reincore_context_reset": context_reset,
        "reincore_session_id": session_id,
        "reincore_session_rolled_over_from": rolled_over_from,
        "reincore_session_handoff": session_handoff,
        "reincore_tool_calls_validated": !agent_health.should_stop,
        "reincore_agent_health": agent_health,
        "reincore_loop_nudge_injected": loop_nudge_injected,
        "reincore_loop_intervention_scheduled": loop_intervention_scheduled,
        "reincore_tool_results_deduped": deduped_tool_results.replaced,
        "reincore_semantic_retrieval_candidates": semantic_candidates,
        "choices": [{
            "index": 0,
            "message": response_message,
            "finish_reason": engine_core::finish_reason_for(&response_message)
        }],
        "usage": {
            "prompt_tokens": actual_usage.prompt_tokens,
            "completion_tokens": actual_usage.completion_tokens,
            "total_tokens": actual_usage.total_tokens,
            "prompt_tokens_details": {
                "cached_tokens": actual_usage.cached_prompt_tokens
            }
        }
    });
    attach_request_group_metadata(&mut response_json, &group_snapshot);
    attach_task_metadata(&mut response_json, &task, task_context_messages_removed);
    if let Some(response_id) = response_json.get("id").and_then(|value| value.as_str()) {
        link_response_to_request_group(response_id, &request_group_id).await;
    }
    print_request_group_rollup(&group_snapshot);

    // If the client requested streaming, convert the response to SSE.
    if req.stream.unwrap_or(false) {
        let sse_stream = chat_completion_to_sse_stream(response_json);
        let response = Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response();
        return Ok(with_provider_rate_limit_headers(
            response,
            &provider_response_headers,
        ));
    }

    Ok(with_provider_rate_limit_headers(
        Json(response_json).into_response(),
        &provider_response_headers,
    ))
}

pub async fn list_models_single_provider(
    State(state): State<SingleProviderAppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let models = state
        .provider
        .list_models()
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?;

    // Claude Code's gateway model discovery filters entries whose `id`
    // doesn't begin with "claude" or "anthropic". Add a synthetic
    // "claude-sonnet-4-5" entry so discovery succeeds and the model appears
    // in the /model picker labeled "From gateway". The proxy ignores the
    // client-supplied model and routes itself, so the ID is just for
    // passing Claude Code's validation.
    let mut data: Vec<serde_json::Value> = models
        .into_iter()
        .map(|m| {
            // Look up context window from the curated catalog so CLI tools
            // that read /v1/models can discover ReinCore's real context limits.
            let catalog_entry = find_model(&m.provider, &m.id);
            let context_window = catalog_entry
                .as_ref()
                .map(|c| c.max_input_tokens as u64)
                .unwrap_or_else(fallback_context_limit);
            let max_output_tokens = catalog_entry
                .as_ref()
                .map(|c| c.max_output_tokens as u64)
                .unwrap_or(65_536);
            json!({
                "id": m.id,
                "object": "model",
                "created": m.created_at.map(|dt| dt.timestamp()).unwrap_or(0),
                "owned_by": m.provider,
                "display_name": m.display_name.unwrap_or_else(|| m.id.clone()),
                "context_window": context_window,
                "max_input_tokens": context_window,
                "max_output_tokens": max_output_tokens,
            })
        })
        .collect();

    // Prepend a claude-prefixed entry for gateway discovery (Claude Code
    // filters entries whose `id` doesn't begin with "claude" or "anthropic").
    // Also prepend the ReinCore Auto entry - ReinCore handles model selection
    // and tier degradation automatically, so this is the recommended default.
    // Both entries advertise a 1M context window so Claude Code doesn't
    // trigger its own session compaction before ReinCore's engine does.
    data.insert(
        0,
        json!({
            "id": "claude-sonnet-4-5",
            "object": "model",
            "created": 0,
            "owned_by": "reincore",
            "display_name": "ReinCore Auto",
            "context_window": 1_000_000,
            "max_input_tokens": 1_000_000,
            "max_output_tokens": 65_536,
        }),
    );
    data.insert(
        0,
        json!({
            "id": "reincore-auto",
            "object": "model",
            "created": 0,
            "owned_by": "reincore",
            "display_name": "ReinCore Auto",
            "context_window": 1_000_000,
            "max_input_tokens": 1_000_000,
            "max_output_tokens": 65_536,
        }),
    );

    Ok(Json(json!({
        "object": "list",
        "data": data,
    })))
}

/// `GET /muse-code/models` - single-provider model catalog for Muse Code.
/// Same payload as the single-provider `/v1/models`.
pub async fn muse_code_models_single_provider(
    State(state): State<SingleProviderAppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    list_models_single_provider(State(state)).await
}
