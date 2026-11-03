use super::lifecycle::extra_text;
use super::logging::log_timestamp;
use super::*;

/// Header and extra field used to explicitly pin a request-group id.
pub(super) const REQUEST_GROUP_HEADER: &str = engine_core::REQUEST_GROUP_HEADER;
pub(super) const REQUEST_GROUP_EXTRA_KEY: &str = engine_core::REQUEST_GROUP_EXTRA_KEY;

/// Extra keys commonly used by Responses-style clients to refer to an
/// earlier response. When present, we reuse that response's group id.
pub(super) const PREVIOUS_RESPONSE_ID_KEYS: [&str; 2] =
    ["previous_response_id", "reincore_previous_response_id"];

#[derive(Clone, Debug, Default)]
pub(super) struct RequestGroupTotals {
    pub(super) steps: u64,
    pub(super) total_prompt_tokens: u64,
    pub(super) total_completion_tokens: u64,
    pub(super) total_actual_cost_nano_usd: i64,
    pub(super) unknown_cost_calls: u64,
}

#[derive(Clone, Debug)]
pub(super) struct RequestGroupSnapshot {
    pub(super) group_id: String,
    pub(super) totals: RequestGroupTotals,
}

/// Current active request-group for each session lineage.
pub(super) fn session_active_request_groups() -> &'static Mutex<HashMap<Uuid, String>> {
    static GROUPS: OnceLock<Mutex<HashMap<Uuid, String>>> = OnceLock::new();
    GROUPS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Running totals keyed by request-group id.
pub(super) fn request_group_totals() -> &'static Mutex<HashMap<String, RequestGroupTotals>> {
    static TOTALS: OnceLock<Mutex<HashMap<String, RequestGroupTotals>>> = OnceLock::new();
    TOTALS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Reverse links so follow-up requests that reference a previous response id
/// can be attached to the same request-group without relying on timers.
pub(super) fn response_to_request_group() -> &'static Mutex<HashMap<String, String>> {
    static RESPONSES: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    RESPONSES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Request correlation is best-effort process-local metadata. Keep it bounded
/// so long-running daemons cannot accumulate one entry per historical request.
pub(super) const MAX_TRACKED_REQUEST_GROUPS: usize = 4096;
pub(super) const MAX_TRACKED_RESPONSES: usize = 16_384;

pub(super) async fn remember_active_request_group(session_id: Uuid, group_id: &str) {
    let mut active = session_active_request_groups().lock().await;
    if active.len() >= MAX_TRACKED_REQUEST_GROUPS && !active.contains_key(&session_id) {
        active.clear();
    }
    active.insert(session_id, group_id.to_string());
}

/// HTTP header that can override the provider selection for a request. Used
/// by multi-provider deployments to route requests to a specific provider.
pub(super) const PROVIDER_HEADER: &str = "x-reincore-provider";

pub(super) fn request_group_from_headers(headers: &HeaderMap) -> Option<String> {
    headers
        .get(REQUEST_GROUP_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

pub(super) fn last_non_system_role(req: &ChatCompletionRequest) -> Option<&str> {
    req.messages
        .iter()
        .rev()
        .find(|message| message.role != "system")
        .map(|message| message.role.as_str())
}

pub(super) async fn resolve_request_group_id(
    db: &LocalDatabase,
    headers: Option<&HeaderMap>,
    req: &ChatCompletionRequest,
    session_id: Option<Uuid>,
) -> Result<String, StatusCode> {
    if let Some(headers) = headers {
        if let Some(group_id) = request_group_from_headers(headers) {
            if let Some(id) = session_id {
                remember_active_request_group(id, &group_id).await;
            }
            if let Some(key) = engine_core::request_group_key(session_id, req) {
                engine_core::bind_conversation_group(
                    &key,
                    &group_id,
                    engine_core::request_continues_conversation(req),
                );
            }
            return Ok(group_id);
        }
    }

    if let Some(group_id) = extra_text(req, REQUEST_GROUP_EXTRA_KEY) {
        if let Some(id) = session_id {
            remember_active_request_group(id, &group_id).await;
        }
        if let Some(key) = engine_core::request_group_key(session_id, req) {
            engine_core::bind_conversation_group(
                &key,
                &group_id,
                engine_core::request_continues_conversation(req),
            );
        }
        return Ok(group_id);
    }

    for key in PREVIOUS_RESPONSE_ID_KEYS {
        if let Some(previous_response_id) = extra_text(req, key) {
            if let Some(group_id) = response_to_request_group()
                .lock()
                .await
                .get(&previous_response_id)
                .cloned()
            {
                if let Some(id) = session_id {
                    remember_active_request_group(id, &group_id).await;
                }
                if let Some(key) = engine_core::request_group_key(session_id, req) {
                    engine_core::bind_conversation_group(&key, &group_id, true);
                }
                return Ok(group_id);
            }
        }
    }

    if engine_core::request_continues_conversation(req) {
        if let Some(key) = engine_core::request_group_key(session_id, req) {
            if let Some(group_id) = engine_core::resume_conversation_group(&key) {
                if let Some(id) = session_id {
                    remember_active_request_group(id, &group_id).await;
                }
                return Ok(group_id);
            }
        }
    }

    if let Some(id) = session_id {
        if let Some(group_id) = engine_core::request_group_from_tool_lineage(db, id, req)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        {
            remember_active_request_group(id, &group_id).await;
            if let Some(key) = engine_core::request_group_key(session_id, req) {
                engine_core::bind_conversation_group(&key, &group_id, true);
            }
            return Ok(group_id);
        }
    }

    // Continuation-based correlation: the agent is still on the same prompt for
    // as long as the model keeps asking for tool calls. This is the one signal
    // that does not depend on how a client encodes its transcript.
    if let Some(key) = engine_core::request_group_key(session_id, req) {
        let group_id = engine_core::resolve_request_group_id(None, session_id, req);
        tracing::debug!(
            group_id = %group_id,
            key = %key,
            messages = req.messages.len(),
            "request-group: resolved from conversation continuation"
        );
        if let Some(id) = session_id {
            remember_active_request_group(id, &group_id).await;
        }
        return Ok(group_id);
    }

    tracing::debug!(
        messages = req.messages.len(),
        roles = %req
            .messages
            .iter()
            .map(|message| message.role.as_str())
            .collect::<Vec<_>>()
            .join(","),
        has_session = session_id.is_some(),
        "request-group: no conversation key; falling back"
    );

    if let Some(id) = session_id {
        let should_reuse_active_group =
            matches!(last_non_system_role(req), Some("assistant") | Some("tool"));
        if should_reuse_active_group {
            if let Some(group_id) = session_active_request_groups()
                .lock()
                .await
                .get(&id)
                .cloned()
            {
                return Ok(group_id);
            }
        }

        let group_id = Uuid::new_v4().to_string();
        remember_active_request_group(id, &group_id).await;
        return Ok(group_id);
    }

    Ok(Uuid::new_v4().to_string())
}

pub(super) async fn record_request_group_step(
    group_id: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
    amount_nano_usd: Option<i64>,
) -> RequestGroupSnapshot {
    let mut totals = request_group_totals().lock().await;
    if totals.len() >= MAX_TRACKED_REQUEST_GROUPS && !totals.contains_key(group_id) {
        totals.clear();
    }
    let entry = totals.entry(group_id.to_string()).or_default();
    entry.steps = entry.steps.saturating_add(1);
    entry.total_prompt_tokens = entry.total_prompt_tokens.saturating_add(prompt_tokens);
    entry.total_completion_tokens = entry
        .total_completion_tokens
        .saturating_add(completion_tokens);
    entry.total_actual_cost_nano_usd = entry
        .total_actual_cost_nano_usd
        .saturating_add(amount_nano_usd.unwrap_or(0));
    if amount_nano_usd.is_none() {
        entry.unknown_cost_calls += 1;
    }
    RequestGroupSnapshot {
        group_id: group_id.to_string(),
        totals: entry.clone(),
    }
}

pub(super) async fn link_response_to_request_group(response_id: &str, group_id: &str) {
    let mut responses = response_to_request_group().lock().await;
    if responses.len() >= MAX_TRACKED_RESPONSES && !responses.contains_key(response_id) {
        responses.clear();
    }
    responses.insert(response_id.to_string(), group_id.to_string());
}

pub(super) fn attach_request_group_metadata(
    value: &mut serde_json::Value,
    group: &RequestGroupSnapshot,
) {
    value["reincore_request_group_id"] = json!(group.group_id);
    value["reincore_request_group_step_count"] = json!(group.totals.steps);
    value["reincore_request_group_total_prompt_tokens"] = json!(group.totals.total_prompt_tokens);
    value["reincore_request_group_total_completion_tokens"] =
        json!(group.totals.total_completion_tokens);
    value["reincore_request_group_total_tokens"] =
        json!(group.totals.total_prompt_tokens + group.totals.total_completion_tokens);
    value["reincore_request_group_total_actual_cost_nano_usd"] =
        json!(group.totals.total_actual_cost_nano_usd);
    value["reincore_request_group_unknown_cost_calls"] = json!(group.totals.unknown_cost_calls);
    value["reincore_request_group_total_actual_cost_cents"] =
        json!(group.totals.total_actual_cost_nano_usd as f64 / 10_000_000.0);
}

/// Refuse a turn whose session is provably looping, so a stuck agent stops
/// billing tokens and the client is told why instead of silently spinning.
pub(super) fn loop_hard_stop_response(
    session_id: Option<Uuid>,
    intervention: &engine_core::PendingIntervention,
    findings: Vec<engine_core::AgentHealthFinding>,
) -> axum::response::Response {
    println!(
        "{}  {} {}",
        log_timestamp().dimmed(),
        "LOOP HARD STOP".red().bold(),
        intervention.message,
    );
    let mut findings = findings;
    let retry_hint = if intervention.retryable {
        "Reincore stopped this turn to avoid burning tokens. Retry to give the agent one more attempt, ideally with a narrower or rephrased instruction."
    } else {
        "This request exhausted its bounded tool-loop budget. Start a new, narrower request after reviewing the evidence already collected."
    };
    if !findings
        .iter()
        .any(|finding| finding.kind == engine_core::AgentHealthKind::ToolLoop)
    {
        findings.insert(
            0,
            engine_core::AgentHealthFinding {
                kind: engine_core::AgentHealthKind::ToolLoop,
                severity: engine_core::AgentHealthSeverity::Critical,
                message: intervention.message.clone(),
                tool_name: None,
            },
        );
    }
    (
        StatusCode::TOO_MANY_REQUESTS,
        Json(json!({
            "error": {
                "message": intervention.message,
                "type": "reincore_loop_detected",
                "code": "loop_detected",
            },
            "blocked": true,
            "session_id": session_id.map(|id| id.to_string()),
            "reincore_agent_health": {
                "findings": findings,
                "should_stop": true,
            },
            "reincore_retryable": intervention.retryable,
            "reincore_retry_hint": retry_hint,
        })),
    )
        .into_response()
}

pub(super) fn print_request_group_rollup(group: &RequestGroupSnapshot) {
    println!(
        "{}  {} group={} steps={} prompt={} completion={} total={} estimated_cost=${:.6} unknown_calls={}",
        log_timestamp().dimmed(),
        "GROUP TOTAL".green().bold(),
        group.group_id,
        group.totals.steps,
        group.totals.total_prompt_tokens,
        group.totals.total_completion_tokens,
        group.totals.total_prompt_tokens + group.totals.total_completion_tokens,
        group.totals.total_actual_cost_nano_usd as f64 / 1_000_000_000.0,
        group.totals.unknown_cost_calls,
    );
}

#[cfg(test)]
mod cost_tests {
    use super::*;
    #[tokio::test]
    async fn request_group_keeps_subcent_spend_and_marks_unknown_calls() {
        let id = Uuid::new_v4().to_string();
        for _ in 0..100 {
            record_request_group_step(&id, 1, 0, Some(100_000)).await;
        }
        let group = record_request_group_step(&id, 0, 0, None).await;
        let mut metadata = json!({});
        attach_request_group_metadata(&mut metadata, &group);
        assert_eq!(
            metadata["reincore_request_group_total_actual_cost_nano_usd"],
            10_000_000
        );
        assert_eq!(
            metadata["reincore_request_group_total_actual_cost_cents"],
            1.0
        );
        assert_eq!(metadata["reincore_request_group_unknown_cost_calls"], 1);
    }
}
