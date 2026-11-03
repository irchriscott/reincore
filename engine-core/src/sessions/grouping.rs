//! Automatic request-group correlation.
//!
//! Agentic clients (VS Code, Codex, Claude Code, the OpenAI SDK, or a bare
//! `curl` loop) run a tool loop as a series of *independent* HTTP requests.
//! Almost none of them send a correlation id, so the engine has to recover the
//! grouping itself.
//!
//! The signal every OpenAI-compatible client shares is that each turn resends
//! the whole conversation. Within one tool loop the transcript only ever grows
//! *after* the triggering user prompt, so the slice up to and including the
//! last user turn is byte-identical on every request of that loop and changes
//! the moment a new prompt arrives. Hashing that slice yields a stable group
//! key without any client cooperation.

use crate::models::ChatCompletionRequest;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// How long a conversation key keeps mapping to the same group id once the
/// client goes quiet. A later request that hashes the same (a client that does
/// not resend history repeating an identical prompt) starts a fresh group.
pub const CONVERSATION_GROUP_TTL: Duration = Duration::from_secs(30 * 60);

/// Upper bound on tracked conversations before expired entries are reclaimed.
const MAX_TRACKED_CONVERSATIONS: usize = 4096;

fn reclaim_conversation_capacity(groups: &mut HashMap<String, ConversationEntry>, now: Instant) {
    groups.retain(|_, entry| now.duration_since(entry.last_seen) <= CONVERSATION_GROUP_TTL);
    if groups.len() < MAX_TRACKED_CONVERSATIONS {
        return;
    }

    // A traffic spike must not discard every active tool loop. Evict only the
    // least-recently-seen entry to make room for the new conversation.
    if let Some(oldest) = groups
        .iter()
        .min_by_key(|(_, entry)| entry.last_seen)
        .map(|(key, _)| key.clone())
    {
        groups.remove(&oldest);
    }
}

struct ConversationEntry {
    group_id: String,
    /// The last response asked for tool calls, so the prompt is still running.
    continues: bool,
    last_seen: Instant,
}

/// Bind a known request group to a conversation key.
///
/// Explicit request-group ids and durable tool-call lineage can recover a
/// group without consulting the process-local continuation map. Recording the
/// recovered binding here lets the normal response lifecycle update it again.
pub fn bind_conversation_group(key: &str, group_id: &str, continues: bool) {
    let now = Instant::now();
    let mut groups = match conversation_groups().lock() {
        Ok(groups) => groups,
        Err(poisoned) => poisoned.into_inner(),
    };
    if groups.len() >= MAX_TRACKED_CONVERSATIONS && !groups.contains_key(key) {
        reclaim_conversation_capacity(&mut groups, now);
    }
    groups.insert(
        key.to_string(),
        ConversationEntry {
            group_id: group_id.to_string(),
            continues,
            last_seen: now,
        },
    );
}

fn conversation_groups() -> &'static Mutex<HashMap<String, ConversationEntry>> {
    static GROUPS: OnceLock<Mutex<HashMap<String, ConversationEntry>>> = OnceLock::new();
    GROUPS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A user turn is a genuine prompt, not a tool result wearing the user role.
fn is_user_turn(message: &crate::models::ChatMessage) -> bool {
    message.role.eq_ignore_ascii_case("user") && message.tool_call_id.is_none()
}

/// Fingerprint the conversation prefix that identifies the current tool loop.
///
/// System messages are excluded: agentic clients regenerate them each turn with
/// volatile context (timestamps, open editors, cwd), which would otherwise
/// break correlation within a single loop.
///
/// Normally the prefix runs up to and including the last user turn. When a
/// transcript carries no user turn at all - some clients fold everything into
/// the system prompt - the conversation head is used instead, which is still
/// stable while the loop appends. Returns `None` only when there is nothing
/// usable to hash.
pub fn conversation_fingerprint(request: &ChatCompletionRequest) -> Option<String> {
    let is_hashable = |message: &&crate::models::ChatMessage| {
        !crate::task_decisions::is_record_message(message)
            && !message.role.eq_ignore_ascii_case("system")
            && !(is_user_turn(message)
                && message
                    .content_str()
                    .is_some_and(crate::user_content::is_user_scaffolding_only))
    };

    let end = request
        .messages
        .iter()
        .rposition(|message| {
            !message.role.eq_ignore_ascii_case("system")
                && is_user_turn(message)
                && message
                    .content_str()
                    .and_then(crate::user_content::session_user_request)
                    .is_some()
        })
        .or_else(|| {
            // No user turn: anchor on the first non-system message instead.
            request
                .messages
                .iter()
                .position(|message| !message.role.eq_ignore_ascii_case("system"))
        })?;

    let mut hasher = Sha256::new();
    for message in request.messages[..=end].iter().filter(is_hashable) {
        hasher.update(message.role.as_bytes());
        hasher.update([0x1f]);
        if let Some(content) = message.content_str() {
            let content = if is_user_turn(message) {
                crate::user_content::session_user_request(content).unwrap_or_default()
            } else {
                content
            };
            hasher.update(content.as_bytes());
        }
        hasher.update([0x1f]);
        if let Some(tool_calls) = &message.tool_calls {
            for call in tool_calls {
                hasher.update(call.function.name.as_bytes());
                hasher.update([0x1e]);
                // Client reserialization (key order, spacing) must not mint
                // a new group: hash the canonical argument form.
                let canonical_arguments =
                    serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                        .ok()
                        .and_then(|parsed| crate::cache::canonicalize_json(&parsed).ok())
                        .unwrap_or_else(|| call.function.arguments.clone());
                hasher.update(canonical_arguments.as_bytes());
                hasher.update([0x1e]);
            }
        }
        if let Some(tool_call_id) = &message.tool_call_id {
            hasher.update(tool_call_id.as_bytes());
        }
        hasher.update([0x1d]);
    }
    Some(format!("{:x}", hasher.finalize()))
}

/// Resolve the group id for a conversation fingerprint, minting one on first
/// sight and reusing it for every subsequent turn of the same tool loop.
pub fn group_id_for_conversation(fingerprint: &str) -> String {
    let now = Instant::now();
    let mut groups = match conversation_groups().lock() {
        Ok(groups) => groups,
        // A poisoned lock must not take request handling down with it.
        Err(poisoned) => poisoned.into_inner(),
    };

    if let Some(entry) = groups.get_mut(fingerprint) {
        if entry.continues && now.duration_since(entry.last_seen) <= CONVERSATION_GROUP_TTL {
            entry.last_seen = now;
            return entry.group_id.clone();
        }
    }

    if groups.len() >= MAX_TRACKED_CONVERSATIONS {
        reclaim_conversation_capacity(&mut groups, now);
    }

    let group_id = Uuid::new_v4().to_string();
    groups.insert(
        fingerprint.to_string(),
        ConversationEntry {
            group_id: group_id.clone(),
            continues: false,
            last_seen: now,
        },
    );
    group_id
}

/// Resolve a continuation request against the current conversation group even
/// if an intervening nested response marked that group as stopped.
///
/// A request whose transcript ends in an assistant/tool message is still part
/// of the preceding user turn. In particular, a parent agent resumes with the
/// completed `runSubagent` tool result after the nested run has emitted its own
/// terminal response.
pub fn group_id_for_conversation_continuation(fingerprint: &str) -> String {
    if let Some(group_id) = resume_conversation_group(fingerprint) {
        return group_id;
    }

    let now = Instant::now();
    let mut groups = match conversation_groups().lock() {
        Ok(groups) => groups,
        Err(poisoned) => poisoned.into_inner(),
    };

    if groups.len() >= MAX_TRACKED_CONVERSATIONS {
        reclaim_conversation_capacity(&mut groups, now);
    }

    let group_id = Uuid::new_v4().to_string();
    groups.insert(
        fingerprint.to_string(),
        ConversationEntry {
            group_id: group_id.clone(),
            continues: true,
            last_seen: now,
        },
    );
    group_id
}

/// Resume an already-known conversation group without minting a replacement.
/// Callers use this as the inexpensive path before consulting durable event
/// lineage after a daemon restart.
pub fn resume_conversation_group(fingerprint: &str) -> Option<String> {
    let now = Instant::now();
    let mut groups = match conversation_groups().lock() {
        Ok(groups) => groups,
        Err(poisoned) => poisoned.into_inner(),
    };
    let entry = groups.get_mut(fingerprint)?;
    if now.duration_since(entry.last_seen) > CONVERSATION_GROUP_TTL {
        return None;
    }
    entry.continues = true;
    entry.last_seen = now;
    Some(entry.group_id.clone())
}

/// Record whether the model asked to keep working after this turn.
///
/// A response carrying `tool_calls` means the agent must run them and come
/// back, so the next request belongs to the same prompt. A response that stops
/// ends the prompt, so the next request opens a new group. This is the one
/// signal that does not depend on how a client encodes its transcript.
pub fn mark_group_continues(key: &str, continues: bool) {
    let mut groups = match conversation_groups().lock() {
        Ok(groups) => groups,
        Err(poisoned) => poisoned.into_inner(),
    };
    if let Some(entry) = groups.get_mut(key) {
        entry.continues = continues;
        entry.last_seen = Instant::now();
    }
}

/// Stable key for a conversation: the session when the client sends one,
/// otherwise the transcript head (which does not move as the loop appends).
pub fn conversation_key(
    session_id: Option<Uuid>,
    request: &ChatCompletionRequest,
) -> Option<String> {
    if let Some(id) = session_id {
        return Some(format!("session:{id}"));
    }
    conversation_head_fingerprint(request).map(|head| format!("head:{head}"))
}

/// Stable key for one user request and every tool-loop turn it produces.
///
/// `conversation_key` deliberately identifies the longer-lived chat. Request
/// group state needs a narrower key: if a response is cancelled while it is
/// still marked as continuing, the next user prompt in that chat must not
/// inherit the cancelled prompt's group. The request fingerprint is unchanged
/// while assistant/tool messages are appended, and changes when a new user
/// prompt is appended.
pub fn request_group_key(
    session_id: Option<Uuid>,
    request: &ChatCompletionRequest,
) -> Option<String> {
    let conversation = conversation_key(session_id, request)?;
    let request = conversation_fingerprint(request)?;
    Some(format!("{conversation}:request:{request}"))
}

/// Whether this HTTP request is continuing work after an assistant response or
/// a tool result rather than introducing a new user prompt.
pub fn request_continues_conversation(request: &ChatCompletionRequest) -> bool {
    request
        .messages
        .iter()
        .rev()
        .find(|message| !message.role.eq_ignore_ascii_case("system"))
        .is_some_and(|message| {
            message.tool_call_id.is_some()
                || message.role.eq_ignore_ascii_case("tool")
                || message.role.eq_ignore_ascii_case("assistant")
        })
}

fn continuation_tool_call_ids(request: &ChatCompletionRequest) -> Vec<&str> {
    let Some(last) = request
        .messages
        .iter()
        .rposition(|message| !message.role.eq_ignore_ascii_case("system"))
    else {
        return Vec::new();
    };

    let mut ids = Vec::new();
    for message in request.messages[..=last].iter().rev() {
        if message.role.eq_ignore_ascii_case("system") {
            continue;
        }
        if message.role.eq_ignore_ascii_case("user") && message.tool_call_id.is_none() {
            break;
        }
        if let Some(id) = message.tool_call_id.as_deref() {
            ids.push(id);
        }
        if let Some(tool_calls) = message.tool_calls.as_deref() {
            ids.extend(tool_calls.iter().map(|call| call.id.as_str()));
        }
    }
    ids
}

/// Recover a request group from the durable tool-call lineage in the session
/// event log. This is the authoritative fallback for parent-agent resumptions
/// and remains available after a daemon restart.
pub async fn request_group_from_tool_lineage<S: crate::event::EventStore + ?Sized>(
    store: &S,
    session_id: Uuid,
    request: &ChatCompletionRequest,
) -> crate::EngineResult<Option<String>> {
    let tool_call_ids = continuation_tool_call_ids(request);
    if tool_call_ids.is_empty() {
        return Ok(None);
    }

    let ids = tool_call_ids
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    Ok(store
        .read_tool_call_links(session_id, &ids)
        .await?
        .into_iter()
        .filter_map(|link| link.request_group_id)
        .find(|group| !group.trim().is_empty()))
}

/// Hash the first non-system message, which identifies a conversation without
/// moving as turns are appended.
pub fn conversation_head_fingerprint(request: &ChatCompletionRequest) -> Option<String> {
    let head = request.messages.iter().find(|message| {
        !message.role.eq_ignore_ascii_case("system")
            && !(is_user_turn(message)
                && message
                    .content_str()
                    .is_some_and(crate::user_content::is_user_scaffolding_only))
    })?;
    let mut hasher = Sha256::new();
    hasher.update(head.role.as_bytes());
    hasher.update([0x1f]);
    if let Some(content) = head.content_str() {
        let content = if is_user_turn(head) {
            crate::user_content::session_user_request(content).unwrap_or_default()
        } else {
            content
        };
        hasher.update(content.as_bytes());
    }
    Some(format!("{:x}", hasher.finalize()))
}

/// Derive the group id for a request directly from its transcript.
pub fn auto_request_group_id(request: &ChatCompletionRequest) -> Option<String> {
    conversation_fingerprint(request).map(|fingerprint| group_id_for_conversation(&fingerprint))
}

/// Header a client may set to pin a request-group id explicitly.
pub const REQUEST_GROUP_HEADER: &str = "x-reincore-request-group-id";

/// Request body field a client may set to pin a request-group id explicitly.
pub const REQUEST_GROUP_EXTRA_KEY: &str = "reincore_request_group_id";

/// Opaque client conversation identifier. The engine hashes and scopes this
/// value before persisting it; clients must not allocate ReinCore session ids.
pub const CONVERSATION_ID_EXTRA_KEY: &str = "reincore_conversation_id";
pub const PARENT_CONVERSATION_ID_EXTRA_KEY: &str = "reincore_parent_conversation_id";
pub const TRACE_ID_EXTRA_KEY: &str = "reincore_trace_id";
pub const SPAN_ID_EXTRA_KEY: &str = "reincore_span_id";

/// Select the opaque identity that the backend should bind to a session.
///
/// Current clients send an explicit conversation id. Trace id is a useful
/// compatibility identity for hosts that expose tracing before conversation
/// metadata. Older VS Code extension builds generated a new external-session
/// UUID for every tool-loop request; for those requests the stable transcript
/// head is deliberately preferred over that client-managed UUID.
pub fn client_conversation_id(request: &ChatCompletionRequest, client_id: &str) -> Option<String> {
    let extra_text = |key: &str| {
        request
            .extra
            .get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };

    extra_text(CONVERSATION_ID_EXTRA_KEY)
        .or_else(|| extra_text(TRACE_ID_EXTRA_KEY).map(|trace_id| format!("trace:{trace_id}")))
        .or_else(|| {
            client_id
                .eq_ignore_ascii_case("vscode")
                .then(|| {
                    conversation_head_fingerprint(request)
                        .map(|fingerprint| format!("transcript:{fingerprint}"))
                })
                .flatten()
        })
}

pub fn conversation_id_hash(conversation_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(conversation_id.trim().as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Durable, tenant/project-scoped lookup key for a client conversation.
pub fn conversation_binding_key(client_id: &str, scope_key: &str, conversation_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(client_id.as_bytes());
    hasher.update([0x1f]);
    hasher.update(scope_key.as_bytes());
    hasher.update([0x1f]);
    hasher.update(conversation_id.trim().as_bytes());
    format!("conversation:{:x}", hasher.finalize())
}

/// Resolve a group id: an explicit client-supplied id wins, otherwise the
/// conversation's continuation state decides, otherwise a fresh id is minted.
pub fn resolve_request_group_id(
    explicit: Option<String>,
    session_id: Option<Uuid>,
    request: &ChatCompletionRequest,
) -> String {
    if let Some(id) = explicit
        .map(|id| id.trim().to_string())
        .filter(|id| !id.is_empty())
    {
        if let Some(key) = request_group_key(session_id, request) {
            bind_conversation_group(&key, &id, request_continues_conversation(request));
        }
        return id;
    }
    request_group_key(session_id, request)
        .map(|key| {
            if request_continues_conversation(request) {
                group_id_for_conversation_continuation(&key)
            } else {
                group_id_for_conversation(&key)
            }
        })
        .unwrap_or_else(|| Uuid::new_v4().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ChatMessage, FunctionCall, ToolCall};
    use crate::MessageContent;
    use std::collections::HashMap;

    fn message(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            provider_state: None,
            role: role.to_string(),
            content: Some(MessageContent::text(content)),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn assistant_tool_call(id: &str, name: &str, arguments: &str) -> ChatMessage {
        ChatMessage {
            provider_state: None,
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: id.to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: arguments.to_string(),
                },
            }]),
            tool_call_id: None,
            name: None,
        }
    }

    fn tool_result(tool_call_id: &str, content: &str) -> ChatMessage {
        ChatMessage {
            provider_state: None,
            role: "tool".to_string(),
            content: Some(MessageContent::text(content)),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
            name: None,
        }
    }

    fn request(messages: Vec<ChatMessage>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "reincore-auto".to_string(),
            messages,
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: HashMap::new(),
        }
    }

    #[test]
    fn tool_loop_turns_share_one_fingerprint() {
        let first = request(vec![
            message("system", "You are a coding agent."),
            message("user", "Refactor the parser."),
        ]);
        let second = request(vec![
            message("system", "You are a coding agent."),
            message("user", "Refactor the parser."),
            assistant_tool_call("call_1", "read_file", r#"{"path":"a.rs"}"#),
            tool_result("call_1", "fn main() {}"),
        ]);
        let third = request(vec![
            message("system", "You are a coding agent."),
            message("user", "Refactor the parser."),
            assistant_tool_call("call_1", "read_file", r#"{"path":"a.rs"}"#),
            tool_result("call_1", "fn main() {}"),
            assistant_tool_call("call_2", "edit_file", r#"{"path":"a.rs"}"#),
            tool_result("call_2", "ok"),
        ]);

        let first_fingerprint = conversation_fingerprint(&first).expect("user turn present");
        assert_eq!(
            Some(first_fingerprint.clone()),
            conversation_fingerprint(&second)
        );
        assert_eq!(Some(first_fingerprint), conversation_fingerprint(&third));
    }

    #[test]
    fn a_new_user_prompt_starts_a_new_fingerprint() {
        let during_loop = request(vec![
            message("user", "Refactor the parser."),
            assistant_tool_call("call_1", "read_file", r#"{"path":"a.rs"}"#),
            tool_result("call_1", "fn main() {}"),
        ]);
        let next_prompt = request(vec![
            message("user", "Refactor the parser."),
            assistant_tool_call("call_1", "read_file", r#"{"path":"a.rs"}"#),
            tool_result("call_1", "fn main() {}"),
            message("assistant", "Done."),
            message("user", "Now add tests."),
        ]);

        assert_ne!(
            conversation_fingerprint(&during_loop),
            conversation_fingerprint(&next_prompt)
        );
    }

    #[test]
    fn volatile_system_prompts_do_not_break_correlation() {
        let first = request(vec![
            message("system", "Current time: 10:00:00. Open file: a.rs"),
            message("user", "Refactor the parser."),
        ]);
        let second = request(vec![
            message("system", "Current time: 10:00:07. Open file: b.rs"),
            message("user", "Refactor the parser."),
            assistant_tool_call("call_1", "read_file", r#"{"path":"a.rs"}"#),
            tool_result("call_1", "fn main() {}"),
        ]);

        assert_eq!(
            conversation_fingerprint(&first),
            conversation_fingerprint(&second)
        );
    }

    #[test]
    fn copilot_scaffolding_does_not_change_conversation_identity() {
        let plain = request(vec![message("user", "Refactor the parser.")]);
        let wrapped = request(vec![
            message(
                "user",
                "<environment_info>Windows</environment_info><workspace_info>a.rs</workspace_info>",
            ),
            message(
                "user",
                "<context>2026-08-18</context><reminderInstructions>Prefer replace edits.</reminderInstructions><userRequest>Refactor the parser.</userRequest>",
            ),
        ]);

        assert_eq!(
            conversation_fingerprint(&plain),
            conversation_fingerprint(&wrapped)
        );
        assert_eq!(
            conversation_head_fingerprint(&plain),
            conversation_head_fingerprint(&wrapped)
        );
    }

    #[test]
    fn distinct_conversations_do_not_collide() {
        let one = request(vec![message("user", "Refactor the parser.")]);
        let two = request(vec![message("user", "Write the changelog.")]);

        assert_ne!(
            conversation_fingerprint(&one),
            conversation_fingerprint(&two)
        );
    }

    #[test]
    fn a_tool_result_in_user_role_is_not_a_user_turn() {
        let mut disguised = tool_result("call_1", "fn main() {}");
        disguised.role = "user".to_string();
        let req = request(vec![message("user", "Refactor the parser."), disguised]);

        let baseline = request(vec![message("user", "Refactor the parser.")]);
        assert_eq!(
            conversation_fingerprint(&baseline),
            conversation_fingerprint(&req)
        );
    }

    #[test]
    fn a_request_without_a_user_turn_has_no_fingerprint() {
        let req = request(vec![message("system", "You are a coding agent.")]);
        assert!(conversation_fingerprint(&req).is_none());
    }

    #[test]
    fn a_request_without_a_user_turn_still_correlates() {
        // Some clients never send a `user` role (everything is folded into the
        // system prompt, or the turn opens with prior assistant context). The
        // conversation head is still stable across the loop.
        let first = request(vec![
            message("system", "You are a coding agent."),
            message("assistant", "Working on it."),
        ]);
        let second = request(vec![
            message("system", "You are a coding agent."),
            message("assistant", "Working on it."),
            assistant_tool_call("call_1", "read_file", r#"{"path":"a.rs"}"#),
            tool_result("call_1", "fn main() {}"),
        ]);

        let fingerprint = conversation_fingerprint(&first).expect("head fingerprint");
        assert_eq!(Some(fingerprint), conversation_fingerprint(&second));
    }

    #[test]
    fn a_request_with_no_usable_messages_has_no_fingerprint() {
        let req = request(vec![message("system", "You are a coding agent.")]);
        assert!(conversation_fingerprint(&req).is_none());
    }

    #[test]
    fn a_tool_loop_shares_one_group_and_a_finished_turn_starts_the_next() {
        let key = format!("session:{}", Uuid::new_v4());

        // First request of a prompt.
        let first = group_id_for_conversation(&key);
        // The model asked for tool calls, so the prompt is still running.
        mark_group_continues(&key, true);
        assert_eq!(group_id_for_conversation(&key), first);
        mark_group_continues(&key, true);
        assert_eq!(group_id_for_conversation(&key), first);

        // The model answered instead of calling a tool: the prompt is done.
        mark_group_continues(&key, false);
        let second = group_id_for_conversation(&key);
        assert_ne!(second, first);
    }

    #[test]
    fn a_tool_result_resumes_the_group_after_a_nested_terminal_response() {
        let session = Uuid::new_v4();
        let key = format!("session:{session}");
        let first_request = request(vec![message("user", "Implement content support")]);
        let first = resolve_request_group_id(None, Some(session), &first_request);

        // The parent delegated work, but the nested run's final text response
        // was observed on the same session key and marked the group stopped.
        mark_group_continues(&key, false);

        let parent_resume = request(vec![
            message("user", "Implement content support"),
            assistant_tool_call("call-subagent", "runSubagent", "{}"),
            tool_result("call-subagent", "research complete"),
        ]);
        assert_eq!(
            resolve_request_group_id(None, Some(session), &parent_resume),
            first
        );

        // A genuine new user turn still opens a different group.
        mark_group_continues(&key, false);
        let next_prompt = request(vec![
            message("user", "Implement content support"),
            assistant_tool_call("call-subagent", "runSubagent", "{}"),
            tool_result("call-subagent", "research complete"),
            message("assistant", "Done."),
            message("user", "Now update the README"),
        ]);
        assert_ne!(
            resolve_request_group_id(None, Some(session), &next_prompt),
            first
        );
    }

    #[tokio::test]
    async fn tool_lineage_recovers_the_group_after_process_state_is_lost() {
        use crate::{EventStore, InMemoryEventStore, SessionEventType};

        let store = InMemoryEventStore::default();
        let session = Uuid::new_v4();
        store
            .append_event(
                session,
                SessionEventType::LlmCall,
                serde_json::json!({
                    "request_group_id": "parent-group",
                    "response_message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call-subagent",
                            "type": "function",
                            "function": {"name": "runSubagent", "arguments": "{}"}
                        }]
                    }
                }),
                None,
            )
            .await
            .unwrap();

        let parent_resume = request(vec![
            message("user", "Implement content support"),
            assistant_tool_call("call-subagent", "runSubagent", "{}"),
            tool_result("call-subagent", "research complete"),
        ]);
        assert_eq!(
            request_group_from_tool_lineage(&store, session, &parent_resume)
                .await
                .unwrap()
                .as_deref(),
            Some("parent-group")
        );
    }

    #[test]
    fn conversations_are_keyed_independently() {
        let one = format!("session:{}", Uuid::new_v4());
        let two = format!("session:{}", Uuid::new_v4());
        assert_ne!(
            group_id_for_conversation(&one),
            group_id_for_conversation(&two)
        );
    }

    #[test]
    fn a_session_keys_a_conversation_even_when_the_transcript_shifts() {
        let session = Uuid::new_v4();
        let early = request(vec![message("user", "Refactor the parser.")]);
        let later = request(vec![
            message("system", "Current time: 11:35:36"),
            message("user", "Refactor the parser."),
            message("assistant", "working"),
            message("user", "tool output"),
        ]);

        assert_eq!(
            conversation_key(Some(session), &early),
            conversation_key(Some(session), &later)
        );
    }

    #[test]
    fn a_sessionless_conversation_is_keyed_by_its_head() {
        let early = request(vec![message("user", "Refactor the parser.")]);
        let later = request(vec![
            message("system", "Current time: 11:35:36"),
            message("user", "Refactor the parser."),
            message("assistant", "working"),
            message("user", "tool output"),
        ]);

        let key = conversation_key(None, &early).expect("head key");
        assert_eq!(Some(key), conversation_key(None, &later));
    }

    #[test]
    fn request_group_key_is_stable_during_a_tool_loop() {
        let session = Uuid::new_v4();
        let first = request(vec![message("user", "Refactor the parser.")]);
        let continuation = request(vec![
            message("user", "Refactor the parser."),
            assistant_tool_call("call_1", "read_file", r#"{"path":"a.rs"}"#),
            tool_result("call_1", "fn main() {}"),
        ]);

        assert_eq!(
            request_group_key(Some(session), &first),
            request_group_key(Some(session), &continuation)
        );
    }

    #[test]
    fn correlated_terminal_completion_remains_in_the_originating_request_group() {
        let session = Uuid::new_v4();
        let first = request(vec![message("user", "Review the request grouping bug")]);
        let first_group = resolve_request_group_id(None, Some(session), &first);
        let first_key = request_group_key(Some(session), &first).unwrap();
        mark_group_continues(&first_key, true);

        let continuation = request(vec![
            message("user", "Review the request grouping bug"),
            assistant_tool_call(
                "terminal-call",
                "run_in_terminal",
                r#"{"command":"cargo run"}"#,
            ),
            tool_result(
                "terminal-call",
                "Command is running in terminal 5a5a9e24-343a-47ee-9413-386b13defc27",
            ),
            assistant_tool_call(
                "reincore_terminal_notification",
                "reincore_terminal_notification",
                r#"{"terminal_id":"5a5a9e24-343a-47ee-9413-386b13defc27"}"#,
            ),
            tool_result(
                "reincore_terminal_notification",
                "Access is denied. (os error 5)",
            ),
        ]);

        assert_eq!(
            request_group_key(Some(session), &continuation),
            Some(first_key)
        );
        assert_eq!(
            resolve_request_group_id(None, Some(session), &continuation),
            first_group
        );
    }

    #[test]
    fn raw_vscode_terminal_steering_is_normalized_before_grouping() {
        let terminal_id = "5a5a9e24-343a-47ee-9413-386b13defc27";
        let session = Uuid::new_v4();
        let first = request(vec![message("user", "Review the request grouping bug")]);
        let first_group = resolve_request_group_id(None, Some(session), &first);
        let first_key = request_group_key(Some(session), &first).unwrap();
        mark_group_continues(&first_key, true);

        let mut continuation = request(vec![
            message("user", "Review the request grouping bug"),
            assistant_tool_call(
                "terminal-call",
                "run_in_terminal",
                r#"{"command":"npx tsc --noEmit","mode":"async"}"#,
            ),
            tool_result(
                "terminal-call",
                &format!("Command is running in terminal with ID={terminal_id}"),
            ),
            message(
                "user",
                &format!(
                    "[Terminal {terminal_id} notification: command completed. The terminal has been cleaned up.]\nTerminal output:\nTypeScript compiled successfully"
                ),
            ),
        ]);
        assert_eq!(
            crate::normalize_correlated_terminal_notifications(&mut continuation),
            1
        );
        assert_eq!(
            request_group_key(Some(session), &continuation),
            Some(first_key)
        );
        assert_eq!(
            resolve_request_group_id(None, Some(session), &continuation),
            first_group
        );
    }

    #[test]
    fn interrupted_request_does_not_capture_the_next_user_prompt() {
        let session = Uuid::new_v4();
        let first_request = request(vec![message("user", "Refactor the parser.")]);
        let first_key = request_group_key(Some(session), &first_request).unwrap();
        let first_group = resolve_request_group_id(None, Some(session), &first_request);
        mark_group_continues(&first_key, true);

        // The first response is cancelled before it can mark the group as
        // stopped. A later prompt in the same session must still start fresh.
        let second_request = request(vec![
            message("user", "Refactor the parser."),
            message("assistant", "I'll inspect the current implementation."),
            message("user", "Instead, update the documentation."),
        ]);
        let second_group = resolve_request_group_id(None, Some(session), &second_request);

        assert_ne!(second_group, first_group);

        // A late terminal update from the cancelled request is isolated to its
        // own key and cannot close or reopen the second prompt's group.
        mark_group_continues(&first_key, false);
        let second_key = request_group_key(Some(session), &second_request).unwrap();
        mark_group_continues(&second_key, true);
        let second_continuation = request(vec![
            message("user", "Refactor the parser."),
            message("assistant", "I'll inspect the current implementation."),
            message("user", "Instead, update the documentation."),
            assistant_tool_call("call_2", "read_file", r#"{"path":"README.md"}"#),
            tool_result("call_2", "# Project"),
        ]);
        assert_eq!(
            resolve_request_group_id(None, Some(session), &second_continuation),
            second_group
        );
    }

    #[test]
    fn conversation_bindings_are_stable_and_scope_isolated() {
        let key = conversation_binding_key("vscode", "project:one", "chat-123");
        assert_eq!(
            key,
            conversation_binding_key("vscode", "project:one", "chat-123")
        );
        assert_ne!(
            key,
            conversation_binding_key("vscode", "project:one", "chat-456")
        );
        assert_ne!(
            key,
            conversation_binding_key("vscode", "project:two", "chat-123")
        );
        assert_ne!(
            key,
            conversation_binding_key("another-client", "project:one", "chat-123")
        );
        assert!(!key.contains("chat-123"));
    }

    #[test]
    fn backend_conversation_identity_prefers_explicit_then_trace_then_vscode_transcript() {
        let mut explicit = request(vec![message("user", "Implement the SDK")]);
        explicit.extra.insert(
            CONVERSATION_ID_EXTRA_KEY.to_string(),
            serde_json::json!("chat-123"),
        );
        explicit.extra.insert(
            TRACE_ID_EXTRA_KEY.to_string(),
            serde_json::json!("trace-ignored"),
        );
        assert_eq!(
            client_conversation_id(&explicit, "vscode").as_deref(),
            Some("chat-123")
        );

        let mut traced = request(vec![message("user", "Implement the SDK")]);
        traced.extra.insert(
            TRACE_ID_EXTRA_KEY.to_string(),
            serde_json::json!("trace-123"),
        );
        assert_eq!(
            client_conversation_id(&traced, "vscode").as_deref(),
            Some("trace:trace-123")
        );

        let legacy = request(vec![message("user", "Implement the SDK")]);
        assert!(client_conversation_id(&legacy, "vscode")
            .expect("VS Code transcript fallback")
            .starts_with("transcript:"));
        assert_eq!(client_conversation_id(&legacy, "other-client"), None);
    }
}
