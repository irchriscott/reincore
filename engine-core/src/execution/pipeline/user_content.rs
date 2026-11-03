//! Normalize user messages wrapped in editor/agent scaffolding.
//!
//! VS Code Copilot and other agentic clients may encode the actual prompt in
//! tags such as `<userRequest>` while placing volatile context and operational
//! reminders around it. ReinCore forwards the current message unchanged, but
//! session intelligence and historical turns should use only the user's words.
//! Provably correlated system-initiated terminal notifications are repaired
//! into tool protocol before any request-boundary logic runs.

use crate::models::{
    ChatCompletionRequest, ChatMessage, ContentPart, FunctionCall, MessageContent, ToolCall,
};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

const REQUEST_TAGS: [&str; 4] = ["userrequest", "user_request", "user_query", "prompt"];

/// Blocks injected by editors/agents around the user's own words.
const SCAFFOLD_TAGS: [&str; 16] = [
    "environment_info",
    "workspace_info",
    "editorcontext",
    "context",
    "attachments",
    "attachment",
    "instructions",
    "reminderinstructions",
    "usermemory",
    "sessionmemory",
    "repomemory",
    "system-reminder",
    "tool_result",
    "files",
    "selection",
    "diff",
];

const EXTERNAL_SUMMARY_TAGS: [&str; 5] = [
    "reincore-file-history-data",
    "reincore-continuation-data",
    "conversation-summary",
    "conversation_summary",
    "context-summary",
];

const COMPACTION_PREAMBLE: &str =
    "the conversation has grown too large for the context window and must be compacted now.";
const SUMMARY_PREAMBLE: &str =
    "your task is to create a comprehensive, detailed summary of the entire conversation";
const SUMMARY_REQUEST: &str = "summarize the conversation history so far, paying special attention to the most recent agent commands and tool results that triggered this summarization.";

fn normalized_client_instruction(raw: &str) -> String {
    raw.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn starts_with_client_instruction(raw: &str, prefix: &str) -> bool {
    let mut words = raw.split_whitespace();
    let mut expected = prefix.split_whitespace().peekable();
    while let Some(token) = expected.next() {
        let Some(word) = words.next() else {
            return false;
        };
        let matches = if expected.peek().is_none() {
            word.get(..token.len())
                .is_some_and(|start| start.eq_ignore_ascii_case(token))
        } else {
            word.eq_ignore_ascii_case(token)
        };
        if !matches {
            return false;
        }
    }
    true
}

/// Recognize the client's inline maintenance prompt, not ordinary user
/// requests to summarize documents or a quoted description of compaction.
pub fn is_client_compaction_instruction(raw: &str) -> bool {
    if !starts_with_client_instruction(raw, COMPACTION_PREAMBLE)
        || wrapped_user_request(raw).is_some()
    {
        return false;
    }
    let text = normalized_client_instruction(raw);
    text.starts_with(COMPACTION_PREAMBLE)
        && text.contains(SUMMARY_PREAMBLE)
        && text.contains("this summary will be used to compact the conversation")
}

/// Separate-call and inline Copilot compaction must never select a task or
/// reach a provider. Existing summaries may still accompany a real request.
pub fn is_client_compaction_request(request: &ChatCompletionRequest) -> bool {
    let Some(message) = request.messages.iter().rev().find(|message| {
        message.role.eq_ignore_ascii_case("user") && message.tool_call_id.is_none()
    }) else {
        return false;
    };
    let text = message.full_text();
    is_client_compaction_instruction(&text)
        || (starts_with_client_instruction(&text, SUMMARY_REQUEST)
            && wrapped_user_request(&text).is_none()
            && request.messages.iter().any(|message| {
                (message.role.eq_ignore_ascii_case("system")
                    || message.role.eq_ignore_ascii_case("user"))
                    && starts_with_client_instruction(&message.full_text(), SUMMARY_PREAMBLE)
            }))
}

pub fn client_compaction_error() -> serde_json::Value {
    serde_json::json!({"error": {
        "type": "invalid_request_error",
        "code": "client_compaction_disabled",
        "message": "ReinCore manages conversation context. Disable automatic summarization and compaction in the client. In VS Code, run Reincore: Configure Copilot Context Management for This Workspace."
    }})
}

/// Recognize a client's internal reminder-observer check (e.g. Muse Code's
/// "You are a reminder observer ..." prompt driving a
/// `submit_reminder_decision` tool call). These checks are client-side
/// housekeeping, not user requests: they must never open sessions, run
/// providers, or appear in the timeline. Both markers are required so an
/// ordinary user message that merely mentions reminders is unaffected.
pub fn is_client_reminder_observer_request(request: &ChatCompletionRequest) -> bool {
    request.messages.iter().any(|message| {
        if message.tool_call_id.is_some() {
            return false;
        }
        if !(message.role.eq_ignore_ascii_case("system")
            || message.role.eq_ignore_ascii_case("user"))
        {
            return false;
        }
        let text = message.full_text().to_ascii_lowercase();
        text.contains("reminder observer") && text.contains("submit_reminder_decision")
    })
}

pub fn client_reminder_observer_error() -> serde_json::Value {
    serde_json::json!({"error": {
        "type": "invalid_request_error",
        "code": "client_reminder_check_disabled",
        "message": "ReinCore does not run client reminder-observer checks through the proxy. Treat the check as decision=none and continue the main conversation."
    }})
}

pub const TERMINAL_NOTIFICATION_TOOL: &str = "reincore_terminal_notification";

/// Synthetic protocol calls correlate VS Code terminal notifications with the
/// tool that launched them. They are history metadata, never executable tools.
pub fn is_terminal_notification_tool(name: &str) -> bool {
    name == TERMINAL_NOTIFICATION_TOOL
}

#[derive(Default)]
struct TerminalToolCandidate {
    name: String,
    input_text: String,
    result_text: String,
}

fn terminal_status_tail<'a>(status: &'a str, prefix: &str) -> Option<&'a str> {
    let rest = status.strip_prefix(prefix)?;
    if let Some(exit) = rest.strip_prefix(" with exit code ") {
        let (code, tail) = exit.split_once('.')?;
        code.parse::<i32>().ok()?;
        Some(tail)
    } else {
        rest.strip_prefix('.')
    }
}

fn is_terminal_notification_status(status: &str) -> bool {
    if let Some(tail) = terminal_status_tail(status, "command completed") {
        return matches!(
            tail,
            " the terminal has been cleaned up."
                | " use send_to_terminal to send another command or kill_terminal to stop it."
        );
    }
    if let Some(tail) = terminal_status_tail(status, "terminal exited") {
        return tail == " the terminal process ended before the command could complete normally; further commands cannot be sent to this terminal id.";
    }
    matches!(
        status,
        "command may be waiting for input — assess the output below."
            | "command may be waiting for input - assess the output below."
    )
}

fn terminal_notification_id(raw: &str) -> Option<String> {
    let raw = raw.trim_start();
    let first_line = raw.lines().next()?.trim_end_matches('\r');
    let lower = first_line.to_ascii_lowercase();
    const PREFIX: &str = "[terminal ";
    const MARKER: &str = " notification: ";
    if !lower.starts_with(PREFIX) || !lower.ends_with(']') {
        return None;
    }
    let marker_start = lower.find(MARKER)?;
    let terminal_id = first_line.get(PREFIX.len()..marker_start)?;
    uuid::Uuid::parse_str(terminal_id).ok()?;
    let status = lower.get(marker_start + MARKER.len()..lower.len() - 1)?;
    if !is_terminal_notification_status(status) {
        return None;
    }
    if !raw.lines().skip(1).any(|line| {
        line.trim_end_matches('\r')
            .eq_ignore_ascii_case("Terminal output:")
    }) {
        return None;
    }
    Some(terminal_id.to_ascii_lowercase())
}

fn terminal_tool_priority(name: &str) -> u8 {
    let normalized = name
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    if normalized == "reincoreterminalnotification" {
        0
    } else if normalized.contains("runinterminal") {
        2
    } else if normalized.contains("terminal") {
        1
    } else {
        0
    }
}

fn originating_terminal_call(
    terminal_id: &str,
    candidates: &HashMap<String, TerminalToolCandidate>,
) -> Option<String> {
    let mut matching = candidates
        .iter()
        .filter_map(|(call_id, candidate)| {
            let priority = terminal_tool_priority(&candidate.name);
            (priority > 0
                && format!("{}\n{}", candidate.input_text, candidate.result_text)
                    .to_ascii_lowercase()
                    .contains(terminal_id))
            .then_some((call_id, priority))
        })
        .collect::<Vec<_>>();
    let highest = matching.iter().map(|(_, priority)| *priority).max()?;
    matching.retain(|(_, priority)| *priority == highest);
    (matching.len() == 1).then(|| matching[0].0.clone())
}

/// Convert VS Code's system-initiated terminal steering messages from a user
/// role into a protocol-valid synthetic tool call/result pair.
///
/// Some VS Code versions send background terminal completion, exit, and input
/// notifications as a new user request. ReinCore only reclassifies an exact
/// notification shape when its terminal UUID can be tied to one earlier
/// terminal tool call in the same transcript. This keeps the request group
/// open without granting arbitrary notification-shaped text user authority.
pub fn normalize_correlated_terminal_notifications(request: &mut ChatCompletionRequest) -> usize {
    let mut candidates = HashMap::<String, TerminalToolCandidate>::new();
    let mut normalized = Vec::with_capacity(request.messages.len());
    let mut changed = 0;

    for message in std::mem::take(&mut request.messages) {
        if message.role.eq_ignore_ascii_case("assistant") {
            for call in message.tool_calls.iter().flatten() {
                candidates.insert(
                    call.id.clone(),
                    TerminalToolCandidate {
                        name: call.function.name.clone(),
                        input_text: call.function.arguments.clone(),
                        result_text: String::new(),
                    },
                );
            }
        }
        if let Some(call_id) = message.tool_call_id.as_deref() {
            if let Some(candidate) = candidates.get_mut(call_id) {
                candidate.result_text.push_str(&message.full_text());
            }
        }

        let notification_text = message.full_text();
        let correlation = (message.role.eq_ignore_ascii_case("user")
            && message.tool_call_id.is_none())
        .then(|| terminal_notification_id(&notification_text))
        .flatten()
        .and_then(|terminal_id| {
            originating_terminal_call(&terminal_id, &candidates)
                .map(|call_id| (terminal_id, call_id))
        });
        let Some((terminal_id, originating_call_id)) = correlation else {
            normalized.push(message);
            continue;
        };

        let mut hasher = Sha256::new();
        hasher.update(originating_call_id.as_bytes());
        hasher.update([0x1f]);
        hasher.update(terminal_id.as_bytes());
        hasher.update([0x1f]);
        hasher.update(notification_text.as_bytes());
        let synthetic_call_id = format!(
            "reincore_terminal_{}",
            &format!("{:x}", hasher.finalize())[..24]
        );
        normalized.push(ChatMessage {
            provider_state: None,
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: synthetic_call_id.clone(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: TERMINAL_NOTIFICATION_TOOL.to_string(),
                    arguments: serde_json::json!({
                        "terminal_id": terminal_id,
                        "originating_tool_call_id": originating_call_id,
                    })
                    .to_string(),
                },
            }]),
            tool_call_id: None,
            name: None,
        });
        normalized.push(ChatMessage {
            provider_state: None,
            role: "tool".to_string(),
            content: message.content,
            tool_calls: None,
            tool_call_id: Some(synthetic_call_id),
            name: Some(TERMINAL_NOTIFICATION_TOOL.to_string()),
        });
        changed += 1;
    }

    request.messages = normalized;
    changed
}

/// True when `raw` contains an external conversation summary injected by the
/// client/editor (for example a Copilot compaction `<conversation-summary>`
/// block). These are not user prompts and must never be treated as session
/// turns, but they are allowed to flow to the provider unchanged.
pub fn is_external_summary_message(raw: &str) -> bool {
    let lower = raw.to_ascii_lowercase();
    EXTERNAL_SUMMARY_TAGS
        .iter()
        .any(|tag| lower.contains(&format!("<{tag}")))
}

/// True when `content` contains an external context compaction summary that
/// came from the client/editor rather than the user.
pub fn message_is_external_summary(content: Option<&str>) -> bool {
    content.map_or(false, is_external_summary_message)
}

/// Label a message role as an external summary when it contains an injected
/// conversation summary. Useful for audit logging without changing provider
/// request shape.
pub fn external_summary_role_label(role: &str, content: Option<&str>) -> String {
    if message_is_external_summary(content) {
        format!("{role}:external_summary")
    } else {
        role.to_string()
    }
}

/// Return the literal user-request body when `raw` contains a recognized
/// wrapper. Matching is ASCII case-insensitive and preserves the body exactly
/// apart from leading/trailing whitespace. Tags like `<userRequest>`,
/// `<user_request>`, `<user_query>`, and `<prompt>` are recognized,
/// including any attributes on the opening tag.
pub fn wrapped_user_request(raw: &str) -> Option<&str> {
    let lower = raw.to_ascii_lowercase();
    let summary_ranges = external_summary_ranges(&lower);
    let mut selected: Option<(usize, &str)> = None;

    for tag in REQUEST_TAGS {
        let open = format!("<{tag}");
        let close = format!("</{tag}>");
        let mut cursor = 0;
        while let Some(relative_start) = lower[cursor..].find(&open) {
            let start = cursor + relative_start;
            let after_open = start + open.len();
            let valid_boundary = lower
                .as_bytes()
                .get(after_open)
                .is_some_and(|byte| *byte == b'>' || *byte == b'/' || byte.is_ascii_whitespace());
            if !valid_boundary {
                cursor = after_open;
                continue;
            }
            let Some(tag_end) = raw[after_open..].find('>') else {
                break;
            };
            let body_start = after_open + tag_end + 1;
            let end = lower[body_start..]
                .find(&close)
                .map(|index| body_start + index)
                .unwrap_or(raw.len());
            let inside_external_summary = summary_ranges
                .iter()
                .any(|range| start >= range.start && start < range.end);
            let body = raw[body_start..end].trim();
            if !inside_external_summary
                && !body.is_empty()
                && selected.map_or(true, |(selected_start, _)| start > selected_start)
            {
                selected = Some((start, body));
            }
            cursor = end.saturating_add(close.len()).min(raw.len());
        }
    }
    selected.map(|(_, body)| body)
}

/// Byte ranges occupied by external summaries. Request-looking markup inside
/// these ranges is quoted history, not the live user request.
fn external_summary_ranges(lower: &str) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    for tag in EXTERNAL_SUMMARY_TAGS {
        let open = format!("<{tag}");
        let close = format!("</{tag}>");
        let mut cursor = 0;
        while let Some(relative_start) = lower[cursor..].find(&open) {
            let start = cursor + relative_start;
            let after_open = start + open.len();
            let Some(tag_end) = lower[after_open..].find('>') else {
                ranges.push(start..lower.len());
                break;
            };
            let content_start = after_open + tag_end + 1;
            let end = lower[content_start..]
                .find(&close)
                .map(|relative| content_start + relative + close.len())
                .unwrap_or(lower.len());
            ranges.push(start..end);
            cursor = end;
        }
    }
    ranges
}

/// A fresh task cannot inherit a client's summary of some earlier task.
/// Preserve current editor scaffolding and non-text attachments.
pub(crate) fn discard_prior_task_summaries(
    request: &mut ChatCompletionRequest,
    task_start: usize,
) -> usize {
    fn strip(raw: &str) -> String {
        let literal = wrapped_user_request(raw);
        if literal.is_none() && crate::lexical::starts_new_task(raw) {
            return raw.to_string();
        }
        let mut ranges = external_summary_ranges(&raw.to_ascii_lowercase());
        if let Some(literal) = literal {
            // Do not remove markup the human explicitly wrote inside the
            // request body (for example a feature about summary tags).
            let start = literal.as_ptr() as usize - raw.as_ptr() as usize;
            let end = start + literal.len();
            ranges.retain(|range| range.end <= start || range.start >= end);
        }
        ranges.sort_by_key(|range| range.start);
        let mut text = String::new();
        let mut cursor = 0;
        for range in ranges {
            if range.start > cursor {
                text.push_str(&raw[cursor..range.start]);
            }
            cursor = cursor.max(range.end);
        }
        text.push_str(&raw[cursor..]);
        text
    }
    let before = request.messages.len();
    let mut index = 0;
    request.messages.retain_mut(|message| {
        let before_or_at_task_start = index <= task_start;
        index += 1;
        // A later summary belongs to this task. When an older client has
        // already compacted its history, preserve that execution evidence.
        if !before_or_at_task_start {
            return true;
        }
        if !message.role.eq_ignore_ascii_case("user") || message.tool_call_id.is_some() {
            return true;
        }
        if is_client_compaction_instruction(&message.full_text()) {
            return false;
        }
        match &mut message.content {
            Some(MessageContent::Text(text)) => *text = strip(text),
            Some(MessageContent::Parts(parts)) => {
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        *text = strip(text);
                    }
                }
            }
            None => {}
        }
        !message.full_text().trim().is_empty()
            || message
                .content
                .as_ref()
                .is_some_and(|content| content.has_images() || content.has_files())
    });
    before - request.messages.len()
}

/// The text ReinCore should use when interpreting a user turn.
pub fn effective_user_request(raw: &str) -> &str {
    wrapped_user_request(raw).unwrap_or(raw)
}

/// Remove recognized editor/agent scaffolding blocks while preserving any
/// ordinary text outside them.
pub fn strip_user_scaffolding(raw: &str) -> String {
    let mut out = raw.to_string();
    for tag in SCAFFOLD_TAGS {
        let open = format!("<{tag}");
        let close = format!("</{tag}>");
        loop {
            let lower = out.to_ascii_lowercase();
            let Some(start) = lower.find(&open) else {
                break;
            };
            let after_open = start + open.len();
            let Some(tag_end) = lower[after_open..].find('>') else {
                break;
            };
            let content_start = after_open + tag_end + 1;
            let end = lower[content_start..]
                .find(&close)
                .map(|index| content_start + index + close.len())
                .unwrap_or(out.len());
            out.replace_range(start..end, "");
        }
    }
    out
}

/// True when a user-role message contains only recognized client scaffolding
/// and no literal request.
pub fn is_user_scaffolding_only(raw: &str) -> bool {
    wrapped_user_request(raw).is_none()
        && strip_user_scaffolding(raw).trim().is_empty()
        && SCAFFOLD_TAGS
            .iter()
            .any(|tag| raw.to_ascii_lowercase().contains(&format!("<{tag}")))
}

/// User text for session intelligence. Scaffolding-only or external summary
/// messages are ignored.
pub fn session_user_request(raw: &str) -> Option<&str> {
    // Some clients place their generated conversation summary and the new
    // literal request in the same user-role message. The explicit request
    // must win; otherwise compaction mistakes an old user turn for the active
    // boundary and protects the wrong tool-loop tail.
    if let Some(request) = wrapped_user_request(raw) {
        return Some(request);
    }
    if is_client_compaction_instruction(raw)
        || (is_external_summary_message(raw) && !crate::lexical::starts_new_task(raw))
    {
        return None;
    }
    (!is_user_scaffolding_only(raw)).then_some(raw)
}

/// Replace a wrapped user message with its literal request body.
pub fn normalize_user_message(message: &mut ChatMessage) -> bool {
    if !message.role.eq_ignore_ascii_case("user") || message.tool_call_id.is_some() {
        return false;
    }

    // Handle multi-part content: normalize each text part in-place without
    // dropping image/file attachments.
    if let Some(MessageContent::Parts(parts)) = &mut message.content {
        let mut changed = false;
        for part in parts.iter_mut() {
            if let ContentPart::Text { text } = part {
                if let Some(request) = wrapped_user_request(text) {
                    *text = request.to_string();
                    changed = true;
                }
            }
        }
        return changed;
    }

    let Some(content) = message.content_str() else {
        return false;
    };
    let Some(request) = wrapped_user_request(content) else {
        return false;
    };
    message.content = Some(MessageContent::text(request.to_string()));
    true
}

/// Compact wrapped historical user turns while leaving the latest/current
/// user message untouched for the provider. Returns the number changed.
pub fn normalize_previous_user_messages(request: &mut ChatCompletionRequest) -> usize {
    normalize_previous_user_message_slice(&mut request.messages)
}

/// Slice variant used by compaction and other provider-independent paths.
pub fn normalize_previous_user_message_slice(messages: &mut [ChatMessage]) -> usize {
    let Some(current_user_index) = messages.iter().rposition(|message| {
        if !message.role.eq_ignore_ascii_case("user") || message.tool_call_id.is_some() {
            return false;
        }
        // For text-only messages, check via session_user_request.
        // For multipart messages, accept them as valid user turns if they
        // have any content (text or attachments).
        if let Some(text) = message.content_str() {
            session_user_request(text).is_some()
        } else {
            message.has_user_content()
        }
    }) else {
        return 0;
    };

    messages[..current_user_index]
        .iter_mut()
        .map(normalize_user_message)
        .filter(|changed| *changed)
        .count()
}

/// Build a role annotation vector for a request, marking external summary
/// messages so the audit log can distinguish them from actual user turns.
pub fn annotate_message_roles(messages: &[ChatMessage]) -> Vec<String> {
    messages
        .iter()
        .map(|message| {
            // For multipart messages, check concatenated text parts for
            // external summary content. Attachment-only messages (no text)
            // are treated as normal user turns.
            let text = if message.is_multipart() {
                message.full_text()
            } else {
                message
                    .content_str()
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            };
            external_summary_role_label(
                &message.role,
                if text.is_empty() { None } else { Some(&text) },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(content: &str) -> ChatMessage {
        ChatMessage {
            provider_state: None,
            role: "user".to_string(),
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

    fn tool_result(id: &str, content: &str) -> ChatMessage {
        ChatMessage {
            provider_state: None,
            role: "tool".to_string(),
            content: Some(MessageContent::text(content)),
            tool_calls: None,
            tool_call_id: Some(id.to_string()),
            name: None,
        }
    }

    fn request(messages: Vec<ChatMessage>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "model".to_string(),
            messages,
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn extracts_copilot_user_request_case_insensitively() {
        let wrapped = "<context>today</context>\n<UserRequest>  Fix the bug\nnow  </UserRequest>";
        assert_eq!(wrapped_user_request(wrapped), Some("Fix the bug\nnow"));
        assert_eq!(effective_user_request(wrapped), "Fix the bug\nnow");
    }

    #[test]
    fn leaves_unwrapped_messages_unchanged() {
        assert_eq!(wrapped_user_request("Fix the bug"), None);
        assert_eq!(effective_user_request("Fix the bug"), "Fix the bug");
    }

    #[test]
    fn identifies_scaffolding_only_user_messages() {
        let scaffolding =
            "<environment_info>Windows</environment_info>\n<workspace_info>repo</workspace_info>";
        // Both tags must be stripped; if any remains, the message is not
        // considered scaffolding-only.
        let stripped = strip_user_scaffolding(scaffolding);
        assert!(
            stripped.trim().is_empty(),
            "leftover after strip: {:?}",
            stripped
        );
        assert!(is_user_scaffolding_only(scaffolding));
        assert_eq!(session_user_request(scaffolding), None);
        assert!(!is_user_scaffolding_only("Use the <context> component"));
    }

    #[test]
    fn correlated_terminal_steering_becomes_tool_protocol() {
        let terminal_id = "5a5a9e24-343a-47ee-9413-386b13defc27";
        let mut req = request(vec![
            message("Review the request grouping bug"),
            assistant_tool_call(
                "terminal-call",
                "run_in_terminal",
                r#"{"command":"npx tsc --noEmit","mode":"async"}"#,
            ),
            tool_result(
                "terminal-call",
                &format!("Command is running in terminal with ID={terminal_id}"),
            ),
            message(&format!(
                "[Terminal {terminal_id} notification: command completed. The terminal has been cleaned up.]\nTerminal output:\nTypeScript compiled successfully"
            )),
        ]);

        assert_eq!(normalize_correlated_terminal_notifications(&mut req), 1);
        assert_eq!(
            req.messages
                .iter()
                .map(|message| message.role.as_str())
                .collect::<Vec<_>>(),
            vec!["user", "assistant", "tool", "assistant", "tool"]
        );
        let synthetic = &req.messages[3].tool_calls.as_ref().unwrap()[0];
        assert_eq!(synthetic.function.name, TERMINAL_NOTIFICATION_TOOL);
        assert_eq!(
            req.messages[4].tool_call_id.as_deref(),
            Some(synthetic.id.as_str())
        );
        assert!(req.messages[4]
            .full_text()
            .contains("TypeScript compiled successfully"));
    }

    #[test]
    fn recognizes_current_vscode_terminal_steering_variants() {
        let terminal_id = "5a5a9e24-343a-47ee-9413-386b13defc27";
        let notifications = [
            format!(
                "[Terminal {terminal_id} notification: command completed. The terminal has been cleaned up.]\nTerminal output:\nDone"
            ),
            format!(
                "[Terminal {terminal_id} notification: command completed with exit code 1. Use send_to_terminal to send another command or kill_terminal to stop it.]\nTerminal output:\nFailed"
            ),
            format!(
                "[Terminal {terminal_id} notification: terminal exited. The terminal process ended before the command could complete normally; further commands cannot be sent to this terminal ID.]\nTerminal output:\nStopped"
            ),
            format!(
                "[Terminal {terminal_id} notification: command may be waiting for input — assess the output below.]\nUse send_to_terminal to answer.\nTerminal output:\nContinue? [y/N]"
            ),
        ];
        for notification in notifications {
            assert_eq!(
                terminal_notification_id(&notification).as_deref(),
                Some(terminal_id)
            );
        }
        assert_eq!(
            terminal_notification_id(&format!(
                "[Terminal {terminal_id} notification: unrelated text.]\nTerminal output:\nDone"
            )),
            None
        );
    }

    #[test]
    fn terminal_steering_requires_unique_prior_tool_lineage() {
        let terminal_id = "5a5a9e24-343a-47ee-9413-386b13defc27";
        let notification = format!(
            "[Terminal {terminal_id} notification: command completed. The terminal has been cleaned up.]\nTerminal output:\nDone"
        );
        let mut uncorrelated = request(vec![message("Run the tests"), message(&notification)]);
        assert_eq!(
            normalize_correlated_terminal_notifications(&mut uncorrelated),
            0
        );
        assert_eq!(uncorrelated.messages.last().unwrap().role, "user");

        let mut ambiguous = request(vec![
            message("Run the tests"),
            assistant_tool_call("one", "run_in_terminal", "{}"),
            tool_result("one", &format!("terminal ID={terminal_id}")),
            assistant_tool_call("two", "run_in_terminal", "{}"),
            tool_result("two", &format!("terminal ID={terminal_id}")),
            message(&notification),
        ]);
        assert_eq!(
            normalize_correlated_terminal_notifications(&mut ambiguous),
            0
        );
        assert_eq!(ambiguous.messages.last().unwrap().role, "user");
    }

    #[test]
    fn explicit_request_wins_when_external_summary_shares_the_message() {
        let mixed = "<conversation-summary>old state</conversation-summary>\n<userRequest>Run the tests</userRequest>";
        assert_eq!(session_user_request(mixed), Some("Run the tests"));
    }

    #[test]
    fn request_markup_quoted_inside_external_summary_is_ignored() {
        let mixed = "<conversation-summary>Earlier input: <userRequest>old request</userRequest></conversation-summary>\n<userRequest>new request</userRequest>";
        assert_eq!(session_user_request(mixed), Some("new request"));
    }

    #[test]
    fn reminder_observer_check_is_detected_without_touching_normal_prompts() {
        let reminder = "You are a reminder observer for the main agent. Do not answer the user. Record exactly one decision this round by calling submit_reminder_decision. Use decision=\"remind\" only when the remind condition stated in your duty is met; otherwise use decision=\"none\".";
        assert!(is_client_reminder_observer_request(&request(vec![message(
            reminder
        )])));

        // Either marker alone is not enough: ordinary user text stays routable.
        assert!(!is_client_reminder_observer_request(&request(vec![
            message("Remind me to water the plants tomorrow"),
        ])));
        assert!(!is_client_reminder_observer_request(&request(vec![
            message("Call submit_reminder_decision when the build finishes"),
        ])));
        // Model output mentioning the mechanism is not a client check either.
        let mut assistant_echo = message(reminder);
        assistant_echo.role = "assistant".to_string();
        assert!(!is_client_reminder_observer_request(&request(vec![
            assistant_echo
        ])));
    }

    #[test]
    fn final_top_level_request_wrapper_wins() {
        let mixed =
            "<userRequest>draft request</userRequest>\n<userRequest>final request</userRequest>";
        assert_eq!(wrapped_user_request(mixed), Some("final request"));
        assert_eq!(
            wrapped_user_request("<promptly>not a wrapper</promptly>"),
            None
        );
    }

    #[test]
    fn compacts_only_previous_user_messages() {
        let previous = "<context>old</context><userRequest>First task</userRequest>";
        let current = "<context>new</context><userRequest>Follow up</userRequest>";
        let mut request = request(vec![message(previous), message(current)]);

        assert_eq!(normalize_previous_user_messages(&mut request), 1);
        assert_eq!(request.messages[0].content_str(), Some("First task"));
        assert_eq!(request.messages[1].content_str(), Some(current));
    }

    #[test]
    fn scaffolding_after_current_turn_does_not_make_current_historical() {
        let current = "<context>new</context><userRequest>Follow up</userRequest>";
        let scaffolding = "<environment_info>Windows</environment_info>";
        let mut request = request(vec![message(current), message(scaffolding)]);

        assert_eq!(normalize_previous_user_messages(&mut request), 0);
        assert_eq!(request.messages[0].content_str(), Some(current));
        assert_eq!(request.messages[1].content_str(), Some(scaffolding));
    }
}

#[cfg(test)]
mod client_compaction_tests {
    use crate::{
        models::ChatCompletionRequest,
        user_content::{
            is_client_compaction_instruction, is_client_compaction_request, session_user_request,
        },
    };
    use serde_json::json;

    const SUMMARY: &str = "Your task is to create a comprehensive, detailed summary of the entire conversation that captures all essential information needed to seamlessly continue the work without any loss of context. This summary will be used to compact the conversation.";
    const SUMMARY_REQUEST: &str = "Summarize the conversation history so far, paying special attention to the most recent agent commands and tool results that triggered this summarization. Structure your summary using the enhanced format provided in the system message.";

    fn inline() -> String {
        format!("The conversation has grown too large for the context window and must be compacted now.\n\n{SUMMARY}")
    }

    fn request(messages: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(json!({"model":"model","messages":messages})).unwrap()
    }

    #[test]
    fn inline_and_separate_client_compaction_are_recognized() {
        let prompt = inline();
        assert!(is_client_compaction_instruction(&prompt));
        assert!(session_user_request(&prompt).is_none());
        assert!(is_client_compaction_request(&request(json!([
            {"role":"user","content":"Implement portfolio events"},
            {"role":"user","content":[{"type":"text","text":prompt}]}
        ]))));
        assert!(is_client_compaction_request(&request(json!([
            {"role":"system","content":SUMMARY},
            {"role":"user","content":SUMMARY_REQUEST}
        ]))));
        assert!(is_client_compaction_request(&request(json!([
            {"role":"user","content":inline()},
            {"role":"assistant","content":"<summary>"}
        ]))));
    }

    #[test]
    fn user_summaries_quoted_examples_and_historical_maintenance_are_not_blocked() {
        for prompt in [
            "Summarize this report",
            "Summarize the conversation history so far",
            SUMMARY_REQUEST,
        ] {
            assert!(!is_client_compaction_request(&request(
                json!([{"role":"user","content":prompt}])
            )));
            assert_eq!(session_user_request(prompt), Some(prompt));
        }
        let quoted = format!("Explain why Copilot sent this instruction: {}", inline());
        assert!(!is_client_compaction_instruction(&quoted));
        let wrapped = format!(
            "<conversation-summary>old state</conversation-summary><userRequest>{}</userRequest>",
            inline()
        );
        assert!(!is_client_compaction_request(&request(
            json!([{"role":"user","content":wrapped}])
        )));
        assert_eq!(session_user_request(&wrapped), Some(inline().as_str()));
        assert!(!is_client_compaction_request(&request(json!([
            {"role":"user","content":inline()},
            {"role":"user","content":"New task: Build a dashboard"}
        ]))));
        assert!(!is_client_compaction_request(&request(json!([
            {"role":"tool","tool_call_id":"read-1","content":inline()}
        ]))));
    }
}
