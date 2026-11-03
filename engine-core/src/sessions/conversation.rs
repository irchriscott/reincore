//! Conversation projections used for provider context and session memory.
//!
//! The event log remains a complete audit trail. Context is intentionally
//! smaller: completed request groups contribute their user request and final
//! assistant outcome, while an unfinished tool loop retains its protocol
//! messages so a provider can continue it safely.

use crate::event::{SessionEvent, SessionEventType};
use crate::models::{ChatMessage, MessageContent, ToolCall, ToolResult};
use std::collections::{HashMap, HashSet};

#[derive(Default)]
struct EventGroup {
    first_user: Option<ChatMessage>,
    messages: Vec<ChatMessage>,
    final_response: Option<ChatMessage>,
    completed: bool,
}

fn has_tool_calls(message: &ChatMessage) -> bool {
    message
        .tool_calls
        .as_deref()
        .is_some_and(|calls| !calls.is_empty())
}

fn request_turn(event: &SessionEvent) -> Option<ChatMessage> {
    event
        .payload
        .get("request_messages")
        .and_then(|value| serde_json::from_value::<Vec<ChatMessage>>(value.clone()).ok())
        .and_then(|messages| {
            messages.into_iter().rev().find(|message| {
                message.role.eq_ignore_ascii_case("user")
                    && message.tool_call_id.is_none()
                    && message.has_user_content()
            })
        })
        .map(|mut message| {
            crate::user_content::normalize_user_message(&mut message);
            message
        })
}

fn response_message(event: &SessionEvent) -> Option<ChatMessage> {
    event
        .payload
        .get("response_message")
        .and_then(|value| serde_json::from_value::<ChatMessage>(value.clone()).ok())
}

fn explicit_group_id(event: &SessionEvent) -> Option<String> {
    event
        .payload
        .get("request_group_id")
        .and_then(serde_json::Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .map(|id| format!("group:{id}"))
}

fn normalized_user_text(message: Option<&ChatMessage>) -> Option<String> {
    message.and_then(|message| {
        if let Some(text) = message.content_str() {
            crate::user_content::session_user_request(text).map(|s| s.to_string())
        } else if message.is_multipart() {
            // For multipart messages, extract text parts and check if they
            // contain a valid user request. Attachment-only messages (no
            // text) produce None, which means they won't match legacy
            // groups by text - but explicit group IDs handle those cases.
            let text = message.full_text();
            if text.is_empty() {
                None
            } else {
                crate::user_content::session_user_request(&text).map(|s| s.to_string())
            }
        } else {
            None
        }
    })
}

/// Project session events into provider-safe conversational messages.
///
/// Explicit `request_group_id` values are preferred. Older events without a
/// group id are joined when the preceding response requested a tool and the
/// next LLM call repeats the same user request.
pub(crate) fn project_session_events(events: &[SessionEvent]) -> Vec<ChatMessage> {
    let mut group_first_seq = HashMap::<String, i64>::new();
    let mut llm_group_by_seq = HashMap::<i64, String>::new();
    let mut tool_group_by_call_id = HashMap::<String, String>::new();
    let mut legacy_open: Option<(String, String)> = None;

    // Establish group membership from LLM events before projecting canonical
    // tool events, which may link back by origin sequence or tool-call id.
    for event in events
        .iter()
        .filter(|event| event.event_type == SessionEventType::LlmCall)
    {
        let user = request_turn(event);
        let response = response_message(event);
        let explicit = explicit_group_id(event);
        let key = if let Some(key) = explicit {
            key
        } else {
            let user_text = normalized_user_text(user.as_ref()).unwrap_or_default();
            match legacy_open.as_ref() {
                Some((key, open_user)) if *open_user == user_text => key.clone(),
                _ => format!("legacy:{}", event.seq),
            }
        };

        group_first_seq.entry(key.clone()).or_insert(event.seq);
        llm_group_by_seq.insert(event.seq, key.clone());

        if let Some(response) = response.as_ref() {
            if let Some(calls) = response.tool_calls.as_deref() {
                for call in calls {
                    tool_group_by_call_id.insert(call.id.clone(), key.clone());
                }
            }
            if explicit_group_id(event).is_none() {
                if has_tool_calls(response) {
                    legacy_open =
                        Some((key, normalized_user_text(user.as_ref()).unwrap_or_default()));
                } else {
                    legacy_open = None;
                }
            }
        }
    }

    let available_llm_seqs: HashSet<i64> = llm_group_by_seq.keys().copied().collect();
    let mut groups = HashMap::<String, EventGroup>::new();

    for event in events {
        match event.event_type {
            SessionEventType::LlmCall => {
                let Some(key) = llm_group_by_seq.get(&event.seq).cloned() else {
                    continue;
                };
                let group = groups.entry(key).or_default();
                if group.first_user.is_none() {
                    group.first_user = request_turn(event);
                }
                if let Some(response) = response_message(event) {
                    group.completed = !has_tool_calls(&response);
                    if group.completed {
                        group.final_response = Some(response.clone());
                    }
                    group.messages.push(response);
                }
            }
            SessionEventType::ToolCall => {
                let origin_seq = event
                    .payload
                    .get("origin_llm_call_seq")
                    .and_then(serde_json::Value::as_i64);
                if origin_seq.is_some_and(|seq| available_llm_seqs.contains(&seq)) {
                    // The originating LLM response already carries this call.
                    continue;
                }
                let Some(call) = event
                    .payload
                    .get("tool_call")
                    .and_then(|value| serde_json::from_value::<ToolCall>(value.clone()).ok())
                else {
                    continue;
                };
                let key = explicit_group_id(event)
                    .or_else(|| origin_seq.and_then(|seq| llm_group_by_seq.get(&seq).cloned()))
                    .or_else(|| tool_group_by_call_id.get(&call.id).cloned())
                    .unwrap_or_else(|| format!("tool:{}", event.seq));
                group_first_seq.entry(key.clone()).or_insert(event.seq);
                tool_group_by_call_id.insert(call.id.clone(), key.clone());
                groups.entry(key).or_default().messages.push(ChatMessage {
                    provider_state: None,
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![call]),
                    tool_call_id: None,
                    name: None,
                });
            }
            SessionEventType::ToolResult => {
                let Some(result) = event
                    .payload
                    .get("result")
                    .and_then(|value| serde_json::from_value::<ToolResult>(value.clone()).ok())
                else {
                    continue;
                };
                let origin_seq = event
                    .payload
                    .get("origin_llm_call_seq")
                    .and_then(serde_json::Value::as_i64);
                let key = explicit_group_id(event)
                    .or_else(|| origin_seq.and_then(|seq| llm_group_by_seq.get(&seq).cloned()))
                    .or_else(|| tool_group_by_call_id.get(&result.tool_call_id).cloned())
                    .unwrap_or_else(|| format!("tool:{}", event.seq));
                group_first_seq.entry(key.clone()).or_insert(event.seq);
                let mut content = result.content;
                let redaction_count = crate::privacy::event_redaction_count(&event.payload);
                if redaction_count > 0 {
                    content.push_str("\n\n");
                    content.push_str(&crate::privacy::stored_redaction_notice(redaction_count));
                }
                groups.entry(key).or_default().messages.push(ChatMessage {
                    provider_state: None,
                    role: "tool".to_string(),
                    content: Some(MessageContent::text(content)),
                    tool_calls: None,
                    tool_call_id: Some(result.tool_call_id),
                    name: Some(result.name),
                });
            }
            _ => {}
        }
    }

    let mut ordered_keys: Vec<(String, i64)> = group_first_seq.into_iter().collect();
    ordered_keys.sort_by_key(|(_, seq)| *seq);
    let mut projected = Vec::new();
    for (key, _) in ordered_keys {
        let Some(mut group) = groups.remove(&key) else {
            continue;
        };
        if let Some(user) = group.first_user.take() {
            projected.push(user);
        }
        if group.completed {
            if let Some(final_response) = group.final_response.take() {
                projected.push(final_response);
            }
        } else {
            projected.extend(group.messages);
        }
    }
    projected
}

/// Collapse completed message-history groups while preserving the newest
/// user request and its entire in-progress tool protocol.
pub(crate) fn compact_completed_message_groups(messages: &mut Vec<ChatMessage>) -> usize {
    crate::user_content::normalize_previous_user_message_slice(messages);
    let user_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            if !message.role.eq_ignore_ascii_case("user") || message.tool_call_id.is_some() {
                return None;
            }
            // Check for valid user content, including multipart messages
            // with attachments that content_str() would miss.
            if message.has_user_content() {
                // For text-only messages, verify it's not scaffolding-only
                // or an external summary. Multipart messages with
                // attachments are always valid user turns.
                if let Some(text) = message.content_str() {
                    crate::user_content::session_user_request(text)
                        .is_some()
                        .then_some(index)
                } else {
                    Some(index)
                }
            } else {
                None
            }
        })
        .collect();
    if user_indices.len() < 2 {
        return 0;
    }

    let original_len = messages.len();
    let current_start = *user_indices.last().expect("at least two user messages");
    let mut compacted = messages[..user_indices[0]].to_vec();

    for (position, start) in user_indices[..user_indices.len() - 1]
        .iter()
        .copied()
        .enumerate()
    {
        let end = user_indices[position + 1];
        let mut user = messages[start].clone();
        crate::user_content::normalize_user_message(&mut user);
        compacted.push(user);

        if let Some(final_response) = messages[start + 1..end].iter().rev().find(|message| {
            message.role.eq_ignore_ascii_case("assistant")
                && !has_tool_calls(message)
                && message.content_len() > 0
        }) {
            compacted.push(final_response.clone());
        }
    }

    compacted.extend_from_slice(&messages[current_start..]);
    let removed = original_len.saturating_sub(compacted.len());
    *messages = compacted;
    removed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::FunctionCall;
    use chrono::Utc;
    use uuid::Uuid;

    fn message(role: &str, content: Option<&str>) -> ChatMessage {
        ChatMessage {
            provider_state: None,
            role: role.to_string(),
            content: content.map(|s| crate::MessageContent::text(s.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn tool_call_message(id: &str) -> ChatMessage {
        ChatMessage {
            provider_state: None,
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: id.to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            name: None,
        }
    }

    fn llm_event(
        seq: i64,
        group_id: &str,
        user: &str,
        response: ChatMessage,
        reasoning: Option<&str>,
    ) -> SessionEvent {
        SessionEvent {
            session_id: Uuid::nil(),
            seq,
            event_type: SessionEventType::LlmCall,
            payload: serde_json::json!({
                "request_messages": [{"role": "user", "content": user}],
                "response_message": response,
                "reasoning_content": reasoning,
                "request_group_id": group_id
            }),
            cost_cents: None,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn message_projection_keeps_only_prior_request_and_outcome() {
        let mut messages = vec![
            message("system", Some("system")),
            message("user", Some("old request")),
            tool_call_message("call-1"),
            ChatMessage {
                provider_state: None,
                role: "tool".to_string(),
                content: Some(crate::MessageContent::text("large tool output".to_string())),
                tool_calls: None,
                tool_call_id: Some("call-1".to_string()),
                name: Some("read_file".to_string()),
            },
            message("assistant", Some("Implemented the old request.")),
            message("user", Some("current request")),
            tool_call_message("call-2"),
        ];

        assert_eq!(compact_completed_message_groups(&mut messages), 2);
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[1].content_str(), Some("old request"));
        assert_eq!(
            messages[2].content_str(),
            Some("Implemented the old request.")
        );
        assert_eq!(messages[3].content_str(), Some("current request"));
        assert!(has_tool_calls(&messages[4]));
    }

    #[test]
    fn event_projection_collapses_completed_group_and_keeps_active_tool_loop() {
        let prior_call = llm_event(
            1,
            "prior",
            "build it",
            tool_call_message("call-1"),
            Some("private intermediate reasoning"),
        );
        let prior_result = SessionEvent {
            session_id: Uuid::nil(),
            seq: 2,
            event_type: SessionEventType::ToolResult,
            payload: serde_json::json!({
                "result": {
                    "tool_call_id": "call-1",
                    "name": "read_file",
                    "content": "large prior output",
                    "is_error": false
                },
                "request_group_id": "prior",
                "origin_llm_call_seq": 1
            }),
            cost_cents: None,
            created_at: Utc::now(),
        };
        let prior_final = llm_event(
            3,
            "prior",
            "build it",
            message("assistant", Some("Built the feature.")),
            Some("more private reasoning"),
        );
        let active_call = llm_event(
            4,
            "active",
            "test it",
            tool_call_message("call-2"),
            Some("active private reasoning"),
        );
        let active_result = SessionEvent {
            session_id: Uuid::nil(),
            seq: 5,
            event_type: SessionEventType::ToolResult,
            payload: serde_json::json!({
                "result": {
                    "tool_call_id": "call-2",
                    "name": "read_file",
                    "content": "active result",
                    "is_error": false
                },
                "request_group_id": "active",
                "origin_llm_call_seq": 4
            }),
            cost_cents: None,
            created_at: Utc::now(),
        };

        let projected = project_session_events(&[
            prior_call,
            prior_result,
            prior_final,
            active_call,
            active_result,
        ]);

        assert_eq!(projected.len(), 5);
        assert_eq!(projected[0].content_str(), Some("build it"));
        assert_eq!(projected[1].content_str(), Some("Built the feature."));
        assert_eq!(projected[2].content_str(), Some("test it"));
        assert!(has_tool_calls(&projected[3]));
        assert_eq!(projected[4].content_str(), Some("active result"));
        assert!(projected.iter().all(|message| {
            !message
                .content_text()
                .is_some_and(|content| content.contains("reasoning"))
        }));
        assert!(projected
            .iter()
            .all(|message| { message.content_str() != Some("large prior output") }));
    }

    #[test]
    fn projected_redacted_tool_result_explains_that_placeholder_is_not_source_content() {
        let call = llm_event(
            1,
            "active",
            "inspect auth",
            tool_call_message("call-1"),
            None,
        );
        let result = SessionEvent {
            session_id: Uuid::nil(),
            seq: 2,
            event_type: SessionEventType::ToolResult,
            payload: serde_json::json!({
                "result": {
                    "tool_call_id": "call-1",
                    "name": "read_file",
                    "content": "{\"access_token\":\"[REDACTED]\"}",
                    "is_error": false
                },
                "request_group_id": "active",
                "origin_llm_call_seq": 1,
                "_reincore_privacy_redaction": {
                    "redacted": true,
                    "redaction_count": 1,
                    "representation": "sanitized"
                }
            }),
            cost_cents: None,
            created_at: Utc::now(),
        };

        let projected = project_session_events(&[call, result]);
        let content = projected.last().unwrap().full_text();
        assert!(content.contains(r#""access_token":"[REDACTED]""#));
        assert!(content.contains("not literal workspace content"));
        assert!(content.contains("not usable credentials"));
        assert!(content.contains("Do not copy a placeholder into executable code"));
        assert!(content.contains("generate independent synthetic credentials at runtime"));
        assert!(content.contains("reuse the same generated variable for signup and login"));
        assert!(content.contains("existing refresh variable in the refresh_token field"));
        assert!(content.contains("Do not bypass redaction with fixed character-code arrays"));
        assert!(content.contains("existing environment variable or local credential reference"));
        assert!(content.contains("must not stop execution"));
        assert!(!content.contains("continue using it as-is"));
        assert!(content.contains("do not retry or switch tools"));
        assert!(content.contains("Do not ask the user to supply ordinary isolated test fixtures"));
        assert!(content
            .contains("Only if a genuinely unavailable existing external credential is required"));
    }
}
