//! Session handoff: what a successor session inherits when the current one
//! runs out of context window.
//!
//! Two things are decided here. First, whether the latest user turn stands on
//! its own - a fresh, fully-stated task starts clean, while a follow-up that
//! leans on earlier turns ("also do the same for X") needs carry-over. Second,
//! how much carry-over: a single bounded system message, deliberately small so
//! the successor session does not start out nearly full again.

use crate::models::{ChatCompletionRequest, ChatMessage, MessageContent};
use crate::summary::{clean_message, specificity_score, SPECIFIC_SCORE};
use serde::{Deserialize, Serialize};
/// `name` of the system message that carries the summary across.
pub const HANDOFF_MESSAGE_NAME: &str = "reincore_session_handoff";

/// How much of a filled session its successor inherits.
///
/// The defaults answer "how far back do we go" as a size budget rather than a
/// turn count: the last few exchanges verbatim plus a summary of the older
/// turns that are relevant to the new message - around 1.2k tokens. Small
/// enough that the successor does not fill up again immediately, detailed
/// enough to stay understandable.
#[derive(Clone, Copy, Debug)]
pub struct HandoffPolicy {
    /// Recent exchanges kept verbatim so the successor sees exactly where the
    /// work stopped.
    pub verbatim_turns: usize,
    /// Per-message cap applied to those verbatim turns.
    pub verbatim_chars: usize,
    /// Cap on the summary of everything older.
    pub summary_chars: usize,
    /// Older messages kept as summary input, ranked by relevance.
    pub summary_candidates: usize,
}

impl Default for HandoffPolicy {
    fn default() -> Self {
        Self {
            verbatim_turns: 3,
            verbatim_chars: 600,
            summary_chars: 1_200,
            summary_candidates: 24,
        }
    }
}

/// What a successor session inherited from the session it replaced.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionHandoff {
    /// The new turn leans on earlier turns, so context was carried across.
    pub carried_context: bool,
    /// Prior messages folded into the carried summary.
    pub summarized_messages: usize,
    /// Recent messages carried across verbatim.
    pub verbatim_messages: usize,
    /// Messages removed from the request while compacting it.
    pub dropped_messages: usize,
}

/// True when the latest user turn states a complete task by itself, so the
/// successor session can start with no history at all.
///
/// The bar is deliberately high: mistaking a follow-up for a fresh task loses
/// context the model needs, while the opposite only costs a short summary.
pub fn is_self_contained(request: &ChatCompletionRequest) -> bool {
    let Some(raw) = request
        .messages
        .iter()
        .rev()
        .filter(|message| message.role == "user")
        .filter_map(|message| {
            message
                .content_str()
                .and_then(crate::user_content::session_user_request)
        })
        .next()
    else {
        return false;
    };
    let cleaned = clean_message(raw);
    if specificity_score(&cleaned) < SPECIFIC_SCORE {
        return false;
    }
    !crate::lexical::refers_to_prior_context(&cleaned)
}

/// Compact `request` down to what a successor session should start from: the
/// system prefix, the last few exchanges verbatim, and a summary of the older
/// turns that relate to the current one. A self-contained turn keeps none of it.
///
/// The current turn is carried whole - including any assistant/tool messages
/// that belong to it - so tool-call pairing stays intact.
pub fn compact_for_new_session(
    request: &mut ChatCompletionRequest,
    policy: HandoffPolicy,
) -> SessionHandoff {
    crate::user_content::normalize_previous_user_messages(request);
    let original_count = request.messages.len();
    let leading_systems = request
        .messages
        .iter()
        .take_while(|message| {
            message.role == "system"
                && !matches!(
                    message.name.as_deref(),
                    Some("context_summary" | "untrusted_context_summary")
                )
        })
        .count();
    let Some(current_turn) = request
        .messages
        .iter()
        .rposition(|message| {
            message.role == "user"
                && message
                    .content_str()
                    .and_then(crate::user_content::session_user_request)
                    .is_some()
        })
        .filter(|index| *index >= leading_systems)
    else {
        return SessionHandoff::default();
    };

    let mut messages = request.messages[..leading_systems].to_vec();
    let mut handoff = SessionHandoff::default();

    if !is_self_contained(request) {
        let query = request.messages[current_turn]
            .content_str()
            .and_then(crate::user_content::session_user_request)
            .unwrap_or_default()
            .to_string();
        let earlier = &request.messages[leading_systems..current_turn];
        let (older, recent) = earlier.split_at(verbatim_start(
            earlier,
            policy.verbatim_turns.saturating_mul(2),
        ));

        if let Some(summary) = summarize_older_turns(older, &query, policy) {
            messages.push(summary);
            handoff.summarized_messages = older.len();
        }
        messages.extend(
            recent
                .iter()
                .map(|message| trim_message(message, policy.verbatim_chars)),
        );
        handoff.verbatim_messages = recent.len();
        handoff.carried_context = handoff.summarized_messages > 0 || handoff.verbatim_messages > 0;
    }

    messages.extend_from_slice(&request.messages[current_turn..]);
    request.messages = messages;
    handoff.dropped_messages = original_count.saturating_sub(request.messages.len());
    handoff
}

/// First index of the verbatim tail, moved forward past any tool result whose
/// originating assistant message would be left behind.
fn verbatim_start(earlier: &[ChatMessage], want: usize) -> usize {
    let mut start = earlier.len().saturating_sub(want);
    while start < earlier.len() && earlier[start].role == "tool" {
        start += 1;
    }
    start
}

fn summarize_older_turns(
    older: &[ChatMessage],
    query: &str,
    policy: HandoffPolicy,
) -> Option<ChatMessage> {
    const PREFIX: &str = "Earlier-session context relevant to the current turn:\n";

    if older.is_empty() || policy.summary_chars <= PREFIX.chars().count() {
        return None;
    }
    // One condensed line per turn rather than free prose: a digest keeps the
    // who-said-what structure the successor needs to resume, and keeps the
    // turns picked for relevance from being paraphrased away.
    let limit = policy.summary_candidates.min(older.len());
    if limit == 0 {
        return None;
    }
    let available = policy.summary_chars - PREFIX.chars().count();
    let per_turn = (available / limit.max(1)).clamp(24, 400).min(available);
    let mut digest = String::with_capacity(available.min(4096));
    for message in select_relevant(older, query, limit) {
        let Some(content) = message.content_str() else {
            continue;
        };
        let content = if message.role.eq_ignore_ascii_case("user") {
            let Some(content) = crate::user_content::session_user_request(content) else {
                continue;
            };
            content
        } else {
            content
        };
        let normalized = normalize_whitespace(content, per_turn);
        if normalized.is_empty() {
            continue;
        }
        let line = format!("[{}] {normalized}", message.role);
        let separator = usize::from(!digest.is_empty());
        let remaining = available.saturating_sub(digest.chars().count() + separator);
        if remaining == 0 {
            break;
        }
        if separator != 0 {
            digest.push('\n');
        }
        digest.push_str(&truncate_chars(&line, remaining));
    }
    if digest.is_empty() {
        return None;
    }
    Some(ChatMessage {
        provider_state: None,
        role: "system".to_string(),
        content: Some(MessageContent::text(format!("{PREFIX}{digest}"))),
        tool_calls: None,
        tool_call_id: None,
        name: Some(HANDOFF_MESSAGE_NAME.to_string()),
    })
}

/// Pick the older messages worth summarizing: those sharing vocabulary with the
/// current turn, with a nudge towards recency to break ties. Chronological order
/// is restored so the summary still reads as a narrative.
fn select_relevant<'a>(
    messages: &'a [ChatMessage],
    query: &str,
    limit: usize,
) -> Vec<&'a ChatMessage> {
    if messages.len() <= limit {
        return messages.iter().collect();
    }
    let terms = crate::lexical::relevance_terms(query);
    let total = messages.len();
    let mut ranked: Vec<(usize, usize)> = messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            let content = message.content_str().unwrap_or_default();
            let content = if message.role.eq_ignore_ascii_case("user") {
                crate::user_content::session_user_request(content)?
            } else {
                content
            };
            let overlap = crate::lexical::relevance_terms(content)
                .intersection(&terms)
                .count();
            Some((index, overlap * 4 + (index * 3) / total.max(1)))
        })
        .collect();
    ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));

    let mut chosen: Vec<usize> = ranked
        .into_iter()
        .take(limit)
        .map(|(index, _)| index)
        .collect();
    chosen.sort_unstable();
    chosen.into_iter().map(|index| &messages[index]).collect()
}

fn normalize_whitespace(text: &str, max_chars: usize) -> String {
    let mut normalized = String::with_capacity(text.len().min(max_chars));
    let mut pending_space = false;
    let mut written = 0usize;
    let scan_limit = max_chars.saturating_add(1);
    for character in text.chars() {
        if character.is_whitespace() {
            pending_space = !normalized.is_empty();
            continue;
        }
        if pending_space && written < scan_limit {
            normalized.push(' ');
            written += 1;
            pending_space = false;
        }
        if written >= scan_limit {
            break;
        }
        normalized.push(character);
        written += 1;
        if written >= scan_limit {
            break;
        }
    }
    truncate_chars(&normalized, max_chars)
}

fn trim_message(message: &ChatMessage, max_chars: usize) -> ChatMessage {
    let mut trimmed = message.clone();
    if let Some(content) = trimmed.content_str() {
        if content.chars().count() > max_chars {
            trimmed.content = Some(MessageContent::text(truncate_chars(content, max_chars)));
        }
    }
    trimmed
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let Some((byte_index, _)) = text.char_indices().nth(max_chars) else {
        return text.to_string();
    };
    if max_chars <= 3 {
        return ".".repeat(max_chars);
    }
    let clipped_chars = max_chars - 3;
    let clipped_byte = text
        .char_indices()
        .nth(clipped_chars)
        .map(|(index, _)| index)
        .unwrap_or(byte_index);
    format!("{}...", text[..clipped_byte].trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn request(messages: Vec<ChatMessage>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test".to_string(),
            messages,
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        }
    }

    fn long_history() -> Vec<ChatMessage> {
        let mut messages = vec![message("system", "You are a coding agent.")];
        for index in 0..10 {
            messages.push(message(
                "user",
                &format!("Migrate the provider registry step {index} to the new trait"),
            ));
            messages.push(message(
                "assistant",
                &format!("Step {index} migrated, provider registry now resolves by label"),
            ));
        }
        messages
    }

    #[test]
    fn follow_ups_carry_recent_turns_plus_a_summary() {
        let mut history = long_history();
        history.push(message("user", "now do the same for the cloud engine"));
        let mut req = request(history);

        let handoff = compact_for_new_session(&mut req, HandoffPolicy::default());

        assert!(handoff.carried_context);
        assert!(handoff.summarized_messages > 0);
        assert_eq!(handoff.verbatim_messages, 6, "three exchanges");
        assert!(handoff.dropped_messages > 0);

        assert_eq!(req.messages[0].role, "system");
        assert_eq!(
            req.messages[1].name.as_deref(),
            Some(HANDOFF_MESSAGE_NAME),
            "older turns must survive as a summary"
        );
        assert_eq!(
            req.messages.last().unwrap().content_str(),
            Some("now do the same for the cloud engine")
        );
        // system + summary + 6 verbatim + the current turn.
        assert_eq!(req.messages.len(), 9);
    }

    #[test]
    fn plural_implementation_request_carries_the_preceding_analysis() {
        let mut history = long_history();
        history.push(message("user", "Why does the project filter use a path?"));
        history.push(message(
            "assistant",
            "Add project_id to ProjectUsage and use it in the session filter.",
        ));
        history.push(message("user", "Please implement the chnges."));
        let mut req = request(history);

        let handoff = compact_for_new_session(&mut req, HandoffPolicy::default());

        assert!(handoff.carried_context);
        assert!(req.messages.iter().any(|message| {
            message.content_str()
                == Some("Add project_id to ProjectUsage and use it in the session filter.")
        }));
        assert_eq!(
            req.messages.last().and_then(ChatMessage::content_str),
            Some("Please implement the chnges.")
        );
    }

    #[test]
    fn handoff_compacts_historical_wrappers_but_preserves_the_current_one() {
        let old = "<context>old date</context><reminderInstructions>old reminder</reminderInstructions><userRequest>Migrate the provider registry</userRequest>";
        let current = "<context>current date</context><reminderInstructions>current reminder</reminderInstructions><userRequest>now do the same for cloud</userRequest>";
        let mut req = request(vec![
            message("system", "You are a coding agent."),
            message("user", old),
            message("assistant", "The local registry is migrated."),
            message("user", current),
        ]);

        compact_for_new_session(&mut req, HandoffPolicy::default());

        assert!(req
            .messages
            .iter()
            .all(|message| { message.content_str() != Some(old) }));
        assert_eq!(req.messages.last().unwrap().content_str(), Some(current));
        assert!(!req
            .messages
            .iter()
            .filter_map(|message| message.content_str())
            .any(|content| content.contains("old reminder")));
    }

    #[test]
    fn carried_context_stays_small() {
        let policy = HandoffPolicy::default();
        let mut history = long_history();
        for message in history.iter_mut() {
            message.content = Some(crate::MessageContent::text(
                "a very long assistant transcript. ".repeat(200),
            ));
        }
        history.push(message("user", "now do the same again"));
        let mut req = request(history);

        compact_for_new_session(&mut req, policy);

        let carried: usize = req.messages[1..req.messages.len() - 1]
            .iter()
            .filter_map(|message| message.content_str())
            .map(|content| content.chars().count())
            .sum();
        let ceiling =
            policy.summary_chars + policy.verbatim_turns * 2 * policy.verbatim_chars + 500;
        assert!(
            carried < ceiling,
            "carried {carried} chars, budget is about {ceiling}"
        );
    }

    #[test]
    fn a_fresh_task_starts_clean() {
        let mut history = long_history();
        history.push(message(
            "user",
            "Add a Prometheus metrics endpoint to the local engine and document the scrape config",
        ));
        let mut req = request(history);

        let handoff = compact_for_new_session(&mut req, HandoffPolicy::default());

        assert!(!handoff.carried_context);
        assert_eq!(handoff.summarized_messages, 0);
        assert_eq!(handoff.verbatim_messages, 0);
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, "system");
        assert_eq!(req.messages[1].role, "user");
    }

    #[test]
    fn fresh_non_coding_tasks_are_self_contained() {
        for turn in [
            "Schedule candidate interviews",
            "Draft the renewal proposal",
            "Draft a renewal proposal for Acme procurement",
            "Reconcile the July ledger for account ACCT-2048",
            "Research regional suppliers and recommend the best option",
        ] {
            assert!(
                is_self_contained(&request(vec![message("user", turn)])),
                "{turn} should stand alone without coding vocabulary"
            );
        }
    }

    #[test]
    fn relative_clauses_do_not_force_unrelated_handoff_context() {
        for turn in [
            "Draft a proposal that covers the renewal terms",
            "Prepare a report for your regional sales team",
        ] {
            assert!(
                is_self_contained(&request(vec![message("user", turn)])),
                "{turn} should be treated as a complete standalone task"
            );
        }
    }

    #[test]
    fn the_current_turn_keeps_its_tool_messages() {
        let mut history = long_history();
        history.push(message("user", "now finish the migration"));
        history.push(message("assistant", "calling a tool"));
        history.push(message("tool", "tool output"));
        let mut req = request(history);

        compact_for_new_session(&mut req, HandoffPolicy::default());

        let tail: Vec<&str> = req
            .messages
            .iter()
            .rev()
            .take(3)
            .map(|m| m.role.as_str())
            .collect();
        assert_eq!(tail, ["tool", "assistant", "user"]);
    }

    #[test]
    fn a_carried_tail_never_starts_with_an_orphan_tool_result() {
        let mut history = vec![message("system", "You are a coding agent.")];
        for index in 0..6 {
            history.push(message("user", &format!("step {index}")));
            history.push(message("assistant", &format!("running tool {index}")));
            history.push(message("tool", &format!("output {index}")));
        }
        history.push(message("user", "now do the same again"));
        let mut req = request(history);

        compact_for_new_session(&mut req, HandoffPolicy::default());

        let first_carried = req
            .messages
            .iter()
            .position(|m| m.name.as_deref() != Some(HANDOFF_MESSAGE_NAME) && m.role != "system")
            .unwrap();
        assert_ne!(req.messages[first_carried].role, "tool");
    }

    #[test]
    fn older_turns_are_chosen_by_relevance_to_the_new_message() {
        let mut history = vec![message("system", "You are a coding agent.")];
        for index in 0..30 {
            history.push(message(
                "user",
                &format!("unrelated bookkeeping chore number {index}"),
            ));
        }
        history.insert(
            5,
            message(
                "assistant",
                "the checkpoint token expires after fifteen minutes",
            ),
        );
        history.push(message("user", "why did the checkpoint token expire again"));
        let mut req = request(history);

        compact_for_new_session(&mut req, HandoffPolicy::default());

        let summary = req.messages[1].content_str().unwrap();
        assert!(
            summary.contains("checkpoint token"),
            "relevant older turn was dropped: {summary}"
        );
    }

    #[test]
    fn short_or_vague_turns_are_never_self_contained() {
        for turn in [
            "continue",
            "fix it",
            "why did that fail",
            "and the tests?",
            "Please implement the feature",
            "Please implement the changes",
            "Please implement the chnges",
            "Option 3 - Change docker postgres port mapping",
            "I choose option 3: change the Docker Postgres port mapping",
            "Choice B - Use the local database for migrations",
        ] {
            assert!(
                !is_self_contained(&request(vec![message("user", turn)])),
                "{turn} must not start a clean session"
            );
        }
    }

    #[test]
    fn zero_summary_candidates_is_safe_and_skips_the_summary() {
        let mut history = long_history();
        history.push(message("user", "now do the same again"));
        let mut req = request(history);
        let policy = HandoffPolicy {
            summary_candidates: 0,
            ..HandoffPolicy::default()
        };

        let handoff = compact_for_new_session(&mut req, policy);

        assert_eq!(handoff.summarized_messages, 0);
        assert!(!req
            .messages
            .iter()
            .any(|message| message.name.as_deref() == Some(HANDOFF_MESSAGE_NAME)));
    }

    #[test]
    fn handoff_summary_obeys_its_character_budget() {
        let mut history = long_history();
        history.push(message("user", "now do the same for the cloud engine"));
        let mut req = request(history);
        let policy = HandoffPolicy {
            summary_chars: 220,
            ..HandoffPolicy::default()
        };

        compact_for_new_session(&mut req, policy);

        let summary = req
            .messages
            .iter()
            .find(|message| message.name.as_deref() == Some(HANDOFF_MESSAGE_NAME))
            .and_then(|message| message.content_str())
            .expect("bounded summary");
        assert!(summary.chars().count() <= policy.summary_chars);
    }

    #[test]
    fn huge_verbatim_turn_count_does_not_overflow() {
        let mut history = long_history();
        history.push(message("user", "now do the same again"));
        let mut req = request(history);
        let policy = HandoffPolicy {
            verbatim_turns: usize::MAX,
            ..HandoffPolicy::default()
        };

        compact_for_new_session(&mut req, policy);
        assert_eq!(req.messages.last().unwrap().role, "user");
    }
}
