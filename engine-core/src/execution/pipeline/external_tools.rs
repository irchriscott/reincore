//! Correlate tool executions performed outside ReinCore with the canonical
//! session event log.

use crate::{
    ChatMessage, EngineResult, EventStore, Session, SessionEvent, SessionEventType, SessionStatus,
    ToolCall, ToolResult,
};
use chrono::{DateTime, Utc};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Optional request header used by known client adapters to select a scoped
/// delegation-tool profile without adding provider-specific fields to the
/// OpenAI, Anthropic, or Gemini request bodies.
pub const DELEGATION_PROFILE_HEADER: &str = "x-reincore-client";

const PROFILED_SUBAGENT_TOOLS: &[(&[&str], &[&str])] = &[
    (
        &[
            "codex",
            "codexapp",
            "openairesponses",
            "responses",
            "reincore",
        ],
        &["spawn_agent", "spawnagent", "spawn_subagent"],
    ),
    (
        &["claude", "claudecode", "anthropicmessages"],
        &["agent", "task"],
    ),
    (
        &["copilot", "githubcopilot", "vscode", "vscodecopilot"],
        &["runsubagent", "run_subagent"],
    ),
    (
        &["copilotcli", "githubcopilotcli"],
        &["runsubagent", "run_subagent", "task"],
    ),
    (&["opencode", "opencodecli"], &["subagent", "task"]),
    (&["openclaw", "openclawcli"], &["sessions_spawn"]),
    (&["cursor", "cursorcli", "cursoragent"], &["task"]),
    (
        &["gemini", "geminicli", "googlegemini"],
        &[
            "codebase_investigator",
            "cli_help",
            "generalist",
            "browser_agent",
        ],
    ),
];

/// A completed delegated-agent execution observed in an external client's
/// replayed tool history. The nested model trace is opaque to ReinCore, but
/// the spawn boundary, prompt, result, and request-group correlation are
/// canonical and sufficient to represent the work as a child session.
#[derive(Clone, Debug)]
pub struct ExternalSubagentExecution {
    pub tool_call: ToolCall,
    pub result: ToolResult,
    pub delegation_profile: Option<String>,
    pub request_group_id: String,
    pub origin_llm_call_seq: Option<i64>,
    pub requested_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
}

/// Recognize the tool name used by VS Code Copilot for delegated work.
pub fn is_run_subagent_tool(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_lowercase().as_str(),
        "runsubagent" | "run_subagent"
    )
}

/// Return whether a tool call is a delegated-agent boundary for the client
/// protocol that produced it. Ambiguous names such as `Task`, `Agent`, and
/// `subagent` are deliberately scoped to known clients instead of being
/// treated as delegated work for every OpenAI-compatible client.
pub fn is_subagent_tool_for_profile(name: &str, profile: Option<&str>) -> bool {
    let name = name.trim().to_ascii_lowercase().replace(['-', ' '], "_");
    if matches!(name.as_str(), "runsubagent" | "run_subagent") {
        // Backward-compatible with Copilot events recorded before delegation
        // profiles were persisted.
        return true;
    }

    let profile = profile
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .replace(['-', '_', ' '], "");
    PROFILED_SUBAGENT_TOOLS.iter().any(|(profiles, tools)| {
        profiles.contains(&profile.as_str()) && tools.contains(&name.as_str())
    })
}

/// Stable external id for an opaque child execution. Replayed transcripts and
/// daemon restarts therefore converge on the same child session.
pub fn external_subagent_session_id(parent_external_id: &str, tool_call_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(parent_external_id.as_bytes());
    hasher.update([0x1f]);
    hasher.update(tool_call_id.as_bytes());
    format!("subagent:{:x}", hasher.finalize())
}

fn deterministic_subagent_uuid(parent_id: Uuid, tool_call_id: &str) -> Uuid {
    let mut hasher = Sha256::new();
    hasher.update(parent_id.as_bytes());
    hasher.update([0x1f]);
    hasher.update(tool_call_id.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    // RFC 4122 variant + version-5-style deterministic identifier.
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes)
}

/// Prefer common human-readable delegation arguments, then fall back to a
/// neutral title.
pub fn external_subagent_task_summary(execution: &ExternalSubagentExecution) -> String {
    let arguments =
        serde_json::from_str::<serde_json::Value>(&execution.tool_call.function.arguments)
            .unwrap_or(serde_json::Value::Null);
    for key in ["description", "prompt", "task", "message", "instructions"] {
        if let Some(value) = arguments
            .get(key)
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            let mut summary = value
                .chars()
                .take(crate::summary::MAX_TASK_SUMMARY_CHARS)
                .collect::<String>();
            if value.chars().count() > crate::summary::MAX_TASK_SUMMARY_CHARS {
                summary = summary.trim_end().to_string();
            }
            return summary;
        }
    }
    "Delegated subagent task".to_string()
}

/// Construct the child session while preserving the parent's ownership and
/// project scope. It is completed immediately because ReinCore only observes
/// the external tool once its matching result has returned.
pub fn external_subagent_session(
    parent: &Session,
    execution: &ExternalSubagentExecution,
) -> EngineResult<Session> {
    let parent_external_id = parent
        .external_session_id
        .clone()
        .unwrap_or_else(|| parent.id.to_string());
    let mut child = Session::new_tracked(
        external_subagent_task_summary(execution),
        0,
        parent.project_name.clone(),
        parent.project_path.clone(),
        parent.client_id.clone(),
        Some(external_subagent_session_id(
            &parent_external_id,
            &execution.tool_call.id,
        )),
    )?;
    child.id = deterministic_subagent_uuid(parent.id, &execution.tool_call.id);
    child.workspace_id = parent.workspace_id;
    child.project_id = parent.project_id;
    child.user_id = parent.user_id;
    child.mode = parent.mode;
    child.status = SessionStatus::Completed;
    child.is_subagent = true;
    child.subagent_parent_session_id = Some(parent_external_id);
    child.subagent_spawn_tool_call_id = Some(execution.tool_call.id.clone());
    child.subagent_spawn_request_group_id = Some(execution.request_group_id.clone());
    child.created_at = execution.requested_at;
    child.updated_at = execution.completed_at;
    Ok(child)
}

/// Find completed external subagent executions in canonical parent events.
/// Tool-history replay is already deduplicated by call id before these events
/// are written, so each returned item is a stable spawn boundary.
pub fn completed_external_subagent_executions(
    events: &[SessionEvent],
) -> Vec<ExternalSubagentExecution> {
    let mut calls: HashMap<String, (&SessionEvent, ToolCall)> = HashMap::new();
    let mut results: HashMap<String, (&SessionEvent, ToolResult)> = HashMap::new();

    for event in events {
        match event.event_type {
            SessionEventType::ToolCall => {
                let Some(raw) = event.payload.get("tool_call") else {
                    continue;
                };
                let Ok(call) = serde_json::from_value::<ToolCall>(raw.clone()) else {
                    continue;
                };
                let delegation_profile = event
                    .payload
                    .get("delegation_profile")
                    .and_then(|value| value.as_str());
                if is_subagent_tool_for_profile(&call.function.name, delegation_profile) {
                    calls.entry(call.id.clone()).or_insert((event, call));
                }
            }
            SessionEventType::ToolResult => {
                let Some(raw) = event.payload.get("result") else {
                    continue;
                };
                let Ok(result) = serde_json::from_value::<ToolResult>(raw.clone()) else {
                    continue;
                };
                results
                    .entry(result.tool_call_id.clone())
                    .or_insert((event, result));
            }
            _ => {}
        }
    }

    let mut executions = calls
        .into_iter()
        .filter_map(|(call_id, (call_event, tool_call))| {
            let (result_event, result) = results.remove(&call_id)?;
            let request_group_id = call_event
                .payload
                .get("request_group_id")
                .and_then(|value| value.as_str())
                .or_else(|| {
                    result_event
                        .payload
                        .get("request_group_id")
                        .and_then(|value| value.as_str())
                })?
                .to_string();
            Some(ExternalSubagentExecution {
                tool_call,
                result,
                delegation_profile: call_event
                    .payload
                    .get("delegation_profile")
                    .and_then(|value| value.as_str())
                    .map(str::to_string),
                request_group_id,
                origin_llm_call_seq: call_event
                    .payload
                    .get("origin_llm_call_seq")
                    .and_then(|value| value.as_i64()),
                requested_at: payload_datetime(&call_event.payload, "requested_at")
                    .unwrap_or(call_event.created_at),
                completed_at: payload_datetime(&result_event.payload, "execution_completed_at")
                    .unwrap_or(result_event.created_at),
            })
        })
        .collect::<Vec<_>>();
    executions.sort_by_key(|execution| execution.requested_at);
    executions
}

#[derive(Clone)]
struct ExternalToolCallObservation {
    call: ToolCall,
    delegation_profile: Option<String>,
    request_group_id: Option<String>,
    requested_at: DateTime<Utc>,
    /// The LLM event whose response originally requested this function.
    origin_llm_call_seq: Option<i64>,
}

#[derive(Clone)]
struct ExternalToolResultObservation {
    content: String,
    completed_at: DateTime<Utc>,
}

fn payload_datetime(payload: &serde_json::Value, key: &str) -> Option<DateTime<Utc>> {
    payload
        .get(key)
        .and_then(|value| value.as_str())
        .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
        .map(|value| value.with_timezone(&Utc))
}

fn observe_external_call(
    ordered_ids: &mut Vec<String>,
    calls: &mut HashMap<String, ExternalToolCallObservation>,
    observation: ExternalToolCallObservation,
) {
    if let Some(existing) = calls.get_mut(&observation.call.id) {
        // The request body is authoritative for the replayed tool definition,
        // while the durable link carries lineage that is intentionally absent
        // from OpenAI-compatible messages. Merge the two observations instead
        // of allowing request order to discard the indexed origin sequence.
        if existing.delegation_profile.is_none() {
            existing.delegation_profile = observation.delegation_profile;
        }
        if existing.request_group_id.is_none() {
            existing.request_group_id = observation.request_group_id;
        }
        if existing.origin_llm_call_seq.is_none() {
            existing.origin_llm_call_seq = observation.origin_llm_call_seq;
        }
        if observation.origin_llm_call_seq.is_some() {
            existing.requested_at = observation.requested_at;
        }
    } else {
        ordered_ids.push(observation.call.id.clone());
        calls.insert(observation.call.id.clone(), observation);
    }
}

/// Promote externally executed tool calls embedded in OpenAI-compatible chat
/// history into canonical `tool_call` and `tool_result` session events.
///
/// Clients replay assistant `tool_calls` and matching `role=tool` results on
/// later requests. Existing event IDs are therefore checked first, making the
/// operation idempotent across retries, daemon restarts, and historical
/// backfills. Only calls with a matching returned result are recorded as
/// executions; a function request by itself has not run yet.
pub async fn record_external_tool_executions<S: EventStore + ?Sized>(
    store: &S,
    session_id: Uuid,
    request_messages: &[ChatMessage],
    fallback_request_group_id: &str,
) -> EngineResult<usize> {
    Ok(record_external_tool_executions_detailed_for_profile(
        store,
        session_id,
        request_messages,
        fallback_request_group_id,
        None,
    )
    .await?
    .appended)
}

/// Result of incrementally recording externally executed tools.
#[derive(Clone, Debug, Default)]
pub struct ExternalToolRecordOutcome {
    pub appended: usize,
    pub completed_subagents: Vec<ExternalSubagentExecution>,
}

/// Cheap request-only guard used by HTTP paths to avoid any database work for
/// ordinary prompts. A returned tool result is the signal that an external
/// execution may need to be canonicalized.
pub fn request_has_external_tool_result(request_messages: &[ChatMessage]) -> bool {
    request_messages
        .iter()
        .any(|message| message.role.eq_ignore_ascii_case("tool") && message.tool_call_id.is_some())
}

/// Profile-aware variant used by protocol adapters. The profile is persisted
/// with canonical tool events so later endpoint reads and startup backfills
/// can classify delegation without relying on transient request context.
pub async fn record_external_tool_executions_for_profile<S: EventStore + ?Sized>(
    store: &S,
    session_id: Uuid,
    request_messages: &[ChatMessage],
    fallback_request_group_id: &str,
    delegation_profile: Option<&str>,
) -> EngineResult<usize> {
    Ok(record_external_tool_executions_detailed_for_profile(
        store,
        session_id,
        request_messages,
        fallback_request_group_id,
        delegation_profile,
    )
    .await?
    .appended)
}

/// Incremental profile-aware recorder. Only call IDs present in the current
/// request are looked up, so runtime is independent of session age.
pub async fn record_external_tool_executions_detailed_for_profile<S: EventStore + ?Sized>(
    store: &S,
    session_id: Uuid,
    request_messages: &[ChatMessage],
    fallback_request_group_id: &str,
    delegation_profile: Option<&str>,
) -> EngineResult<ExternalToolRecordOutcome> {
    record_external_tool_executions_with_file_evidence(
        store,
        session_id,
        request_messages,
        fallback_request_group_id,
        delegation_profile,
        &[],
    )
    .await
}

pub async fn record_external_tool_executions_with_file_evidence<S: EventStore + ?Sized>(
    store: &S,
    session_id: Uuid,
    request_messages: &[ChatMessage],
    fallback_request_group_id: &str,
    delegation_profile: Option<&str>,
    file_evidence: &[serde_json::Value],
) -> EngineResult<ExternalToolRecordOutcome> {
    if !request_has_external_tool_result(request_messages) {
        return Ok(ExternalToolRecordOutcome::default());
    }
    let replayed_calls: Vec<_> = request_messages
        .iter()
        .rev()
        .flat_map(|message| {
            message
                .tool_calls
                .as_deref()
                .unwrap_or_default()
                .iter()
                .rev()
        })
        .take(512)
        .cloned()
        .collect();
    let now = Utc::now();
    let mut ordered_ids = Vec::new();
    let mut calls: HashMap<String, ExternalToolCallObservation> = HashMap::new();
    let mut results: HashMap<String, ExternalToolResultObservation> = HashMap::new();

    // The current request has not been persisted yet and normally contains the
    // assistant call plus its returned tool message.
    for message in request_messages {
        if let Some(tool_calls) = message.tool_calls.as_deref() {
            for call in tool_calls {
                observe_external_call(
                    &mut ordered_ids,
                    &mut calls,
                    ExternalToolCallObservation {
                        call: call.clone(),
                        delegation_profile: delegation_profile.map(str::to_string),
                        request_group_id: Some(fallback_request_group_id.to_string()),
                        requested_at: now,
                        origin_llm_call_seq: None,
                    },
                );
            }
        }
        if message.role.eq_ignore_ascii_case("tool") {
            if let Some(id) = message.tool_call_id.as_deref() {
                results
                    .entry(id.to_string())
                    .or_insert_with(|| ExternalToolResultObservation {
                        content: message.content_text().unwrap_or_default(),
                        completed_at: now,
                    });
            }
        }
    }

    let result_ids = results.keys().cloned().collect::<Vec<_>>();
    if result_ids.is_empty() {
        return Ok(ExternalToolRecordOutcome::default());
    }

    // Recover call metadata from the indexed durable link when a client sends
    // only the result message rather than replaying the assistant call.
    for link in store.read_tool_call_links(session_id, &result_ids).await? {
        observe_external_call(
            &mut ordered_ids,
            &mut calls,
            ExternalToolCallObservation {
                call: link.tool_call,
                delegation_profile: link
                    .delegation_profile
                    .or_else(|| delegation_profile.map(str::to_string)),
                request_group_id: link.request_group_id,
                requested_at: link.requested_at,
                origin_llm_call_seq: link.origin_llm_call_seq,
            },
        );
    }

    let replayed_results: Vec<_> = ordered_ids
        .iter()
        .rev()
        .filter_map(|id| {
            let result = results.get(id)?;
            let call = calls.get(id)?;
            Some(ToolResult {
                tool_call_id: id.clone(),
                role: "tool".into(),
                name: call.call.function.name.clone(),
                content: result.content.clone(),
                is_error: None,
            })
        })
        .take(512)
        .collect();
    store
        .observe_file_history_call_identities(session_id, &replayed_calls, &replayed_results)
        .await?;
    let existing = store
        .read_tool_events_by_call_ids(session_id, &result_ids)
        .await?;
    let mut existing_calls = HashSet::new();
    let mut existing_results = HashSet::new();
    for event in existing {
        match event.event_type {
            SessionEventType::ToolCall => {
                if let Some(id) = event
                    .payload
                    .pointer("/tool_call/id")
                    .and_then(serde_json::Value::as_str)
                {
                    existing_calls.insert(id.to_string());
                }
            }
            SessionEventType::ToolResult => {
                if let Some(id) = event
                    .payload
                    .pointer("/result/tool_call_id")
                    .and_then(serde_json::Value::as_str)
                {
                    existing_results.insert(id.to_string());
                }
            }
            _ => {}
        }
    }

    let mut appended = 0;
    let mut completed_subagents = Vec::new();
    for call_id in ordered_ids {
        let Some(call) = calls.get(&call_id) else {
            continue;
        };
        let Some(result) = results.get(&call_id) else {
            continue;
        };
        let request_group_id = call
            .request_group_id
            .as_deref()
            .unwrap_or(fallback_request_group_id);

        if !existing_calls.contains(&call_id) {
            store
                .append_event(
                    session_id,
                    SessionEventType::ToolCall,
                    json!({
                        "tool_call": call.call,
                        "requested_at": call.requested_at,
                        "execution_source": "external_client",
                        "delegation_profile": call.delegation_profile,
                        "request_group_id": request_group_id,
                        "origin_llm_call_seq": call.origin_llm_call_seq,
                    }),
                    None,
                )
                .await?;
            existing_calls.insert(call_id.clone());
            appended += 1;
        }

        if !existing_results.contains(&call_id) {
            let tool_result = ToolResult {
                tool_call_id: call_id.clone(),
                role: "tool".to_string(),
                name: call.call.function.name.clone(),
                content: result.content.clone(),
                is_error: None,
            };
            let mut payload = json!({
                "result": tool_result,
                "cache_hit": false,
                "execution_started_at": null,
                "execution_completed_at": result.completed_at,
                "execution_duration_ms": null,
                "execution_source": "external_client",
                "delegation_profile": call.delegation_profile,
                "request_group_id": request_group_id,
                "origin_llm_call_seq": call.origin_llm_call_seq,
            });
            if let Some(file) =
                crate::file_history::execution_evidence(&call.call, &tool_result, file_evidence)
            {
                payload["file"] = serde_json::to_value(file)?;
            }
            store
                .append_event(session_id, SessionEventType::ToolResult, payload, None)
                .await?;
            existing_results.insert(call_id.clone());
            appended += 1;
        }

        if is_subagent_tool_for_profile(
            &call.call.function.name,
            call.delegation_profile.as_deref(),
        ) {
            completed_subagents.push(ExternalSubagentExecution {
                tool_call: call.call.clone(),
                result: ToolResult {
                    tool_call_id: call_id,
                    role: "tool".to_string(),
                    name: call.call.function.name.clone(),
                    content: result.content.clone(),
                    is_error: None,
                },
                delegation_profile: call.delegation_profile.clone(),
                request_group_id: request_group_id.to_string(),
                origin_llm_call_seq: call.origin_llm_call_seq,
                requested_at: call.requested_at,
                completed_at: result.completed_at,
            });
        }
    }

    Ok(ExternalToolRecordOutcome {
        appended,
        completed_subagents,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FunctionCall, InMemoryEventStore};

    #[test]
    fn delegation_registry_scopes_ambiguous_tool_names() {
        for (profile, tool) in [
            ("codex", "spawn_agent"),
            ("codex-app", "spawn-subagent"),
            ("reincore", "spawnAgent"),
            ("claude-code", "Agent"),
            ("anthropic-messages", "Task"),
            ("github-copilot-cli", "task"),
            ("opencode", "subagent"),
            ("open-code", "task"),
            ("openclaw", "sessions_spawn"),
            ("cursor-agent", "Task"),
            ("gemini-cli", "codebase_investigator"),
            ("gemini", "cli_help"),
            ("google-gemini", "generalist"),
            ("gemini", "browser_agent"),
        ] {
            assert!(
                is_subagent_tool_for_profile(tool, Some(profile)),
                "expected {profile}/{tool} to be a delegation tool"
            );
        }

        assert!(is_subagent_tool_for_profile("runSubagent", None));
        assert!(!is_subagent_tool_for_profile("spawn_agent", Some("claude")));
        assert!(!is_subagent_tool_for_profile("Task", None));
        assert!(!is_subagent_tool_for_profile("Task", Some("vscode")));
        assert!(!is_subagent_tool_for_profile("Task", Some("gemini")));
        assert!(!is_subagent_tool_for_profile(
            "sessions_spawn",
            Some("opencode")
        ));
        assert!(!is_subagent_tool_for_profile("Bash", Some("cursor")));
        assert!(!is_subagent_tool_for_profile("Read", Some("continue")));
        assert!(!is_subagent_tool_for_profile(
            "run_shell_command",
            Some("aider")
        ));
    }

    #[tokio::test]
    async fn profile_qualified_codex_spawn_is_detected() {
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        let messages = vec![
            ChatMessage {
                provider_state: None,
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call-codex-agent".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "spawn_agent".to_string(),
                        arguments: r#"{"task":"Inspect the API"}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                provider_state: None,
                role: "tool".to_string(),
                content: Some(crate::MessageContent::text("done")),
                tool_calls: None,
                tool_call_id: Some("call-codex-agent".to_string()),
                name: Some("spawn_agent".to_string()),
            },
        ];

        record_external_tool_executions_for_profile(
            &store,
            session_id,
            &messages,
            "group-codex",
            Some("codex"),
        )
        .await
        .unwrap();
        let events = store.read_events(session_id, None, None).await.unwrap();
        let executions = completed_external_subagent_executions(&events);
        assert_eq!(executions.len(), 1);
        assert_eq!(executions[0].delegation_profile.as_deref(), Some("codex"));
    }

    #[tokio::test]
    async fn records_a_matching_external_execution_once() {
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        let messages = vec![
            ChatMessage {
                provider_state: None,
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call-1".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "read_file".to_string(),
                        arguments: r#"{"path":"a.rs"}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                provider_state: None,
                role: "tool".to_string(),
                content: Some(crate::MessageContent::text("contents".to_string())),
                tool_calls: None,
                tool_call_id: Some("call-1".to_string()),
                name: Some("read_file".to_string()),
            },
        ];

        assert_eq!(
            record_external_tool_executions(&store, session_id, &messages, "group-1")
                .await
                .unwrap(),
            2
        );
        assert_eq!(
            record_external_tool_executions(&store, session_id, &messages, "group-1")
                .await
                .unwrap(),
            0
        );

        let events = store.read_events(session_id, None, None).await.unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, SessionEventType::ToolCall);
        assert_eq!(events[1].event_type, SessionEventType::ToolResult);
        assert_eq!(events[0].payload["execution_source"], "external_client");
    }

    #[tokio::test]
    async fn does_not_record_an_unexecuted_function_request() {
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        let messages = vec![ChatMessage {
            provider_state: None,
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call-1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            name: None,
        }];

        assert_eq!(
            record_external_tool_executions(&store, session_id, &messages, "group-1")
                .await
                .unwrap(),
            0
        );
    }

    #[tokio::test]
    async fn completed_run_subagent_becomes_a_deterministic_child_session() {
        let store = InMemoryEventStore::default();
        let mut parent = Session::new_tracked(
            "Parent task",
            100,
            Some("Demo"),
            Some("/workspace/demo"),
            Some("vscode"),
            Some("parent-external"),
        )
        .unwrap();
        parent.project_id = Some(Uuid::new_v4());
        let messages = vec![
            ChatMessage {
                provider_state: None,
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call-subagent-1".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "runSubagent".to_string(),
                        arguments: r#"{"description":"Explore filter architecture","prompt":"Inspect the filters."}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                provider_state: None,
                role: "tool".to_string(),
                content: Some(crate::MessageContent::text("report path".to_string())),
                tool_calls: None,
                tool_call_id: Some("call-subagent-1".to_string()),
                name: Some("runSubagent".to_string()),
            },
        ];

        record_external_tool_executions(&store, parent.id, &messages, "group-spawn")
            .await
            .unwrap();
        let events = store.read_events(parent.id, None, None).await.unwrap();
        let executions = completed_external_subagent_executions(&events);
        assert_eq!(executions.len(), 1);
        assert_eq!(executions[0].request_group_id, "group-spawn");

        let first = external_subagent_session(&parent, &executions[0]).unwrap();
        let replayed = external_subagent_session(&parent, &executions[0]).unwrap();
        assert_eq!(first.id, replayed.id);
        assert_eq!(first.external_session_id, replayed.external_session_id);
        assert_eq!(first.status, SessionStatus::Completed);
        assert!(first.is_subagent);
        assert_eq!(
            first.subagent_parent_session_id.as_deref(),
            Some("parent-external")
        );
        assert_eq!(
            first.subagent_spawn_tool_call_id.as_deref(),
            Some("call-subagent-1")
        );
        assert_eq!(
            first.subagent_spawn_request_group_id.as_deref(),
            Some("group-spawn")
        );
        assert_eq!(first.project_id, parent.project_id);
    }
}
