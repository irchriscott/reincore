//! End-to-end request workflow orchestration.
//!
//! Ties together the individual engine-core subsystems - routing,
//! cost prediction, budget degradation, compaction, context assembly, and
//! session handoff - into the shared decision pipeline of the local engine.
//! The central [`resolve_budget_routing`] function
//! consults the session's spend lineage and a pre-flight cost estimate to
//! decide whether to proceed at the classified tier, degrade to a cheaper
//! model, or emit a resumable checkpoint.

use crate::budget::{BudgetTier, ProgressiveBudget};
use crate::compression::{CompressionMode, CompressionSettings, ContextCompressor};
use crate::cost::CostPredictor;
use crate::error::{EngineError, EngineResult};
use crate::handoff::{
    compact_for_new_session, HandoffPolicy, SessionHandoff, HANDOFF_MESSAGE_NAME,
};
use crate::models::{ChatCompletionRequest, ChatMessage, MessageContent, ModelTier};
use crate::session::Session;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Shared budget decision used by both the local and cloud request paths.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BudgetRoutingDecision {
    pub current_tier: BudgetTier,
    pub projected_tier: BudgetTier,
    pub classified_model_tier: ModelTier,
    pub effective_model_tier: Option<ModelTier>,
    pub should_checkpoint: bool,
}

impl BudgetRoutingDecision {
    pub fn model_was_degraded(&self) -> bool {
        self.effective_model_tier
            .map(|tier| tier != self.classified_model_tier)
            .unwrap_or(false)
    }
}

/// Resolve budget degradation before the provider call. A prediction that
/// crosses 100% degrades as Tier 3; only already-exhausted sessions checkpoint.
/// This avoids pausing a session based solely on a conservative estimate.
pub fn resolve_budget_routing(
    session: Option<&Session>,
    classified_model_tier: ModelTier,
    predicted_high_cents: i64,
) -> EngineResult<BudgetRoutingDecision> {
    let Some(session) = session else {
        return Ok(BudgetRoutingDecision {
            current_tier: BudgetTier::Tier1,
            projected_tier: BudgetTier::Tier1,
            classified_model_tier,
            effective_model_tier: Some(classified_model_tier),
            should_checkpoint: false,
        });
    };

    let budget = ProgressiveBudget::new(session.budget_cents)?;
    let lineage_spend = session.lineage_spend_nano_usd();
    let current_tier = budget.current_tier_nano_usd(lineage_spend);
    let projected_tier = budget
        .current_tier_nano_usd(lineage_spend.saturating_add(
            predicted_high_cents.saturating_mul(crate::pricing::NANO_USD_PER_CENT),
        ));
    let should_checkpoint = current_tier == BudgetTier::Tier4;
    let degradation_tier = if projected_tier == BudgetTier::Tier4 && !should_checkpoint {
        BudgetTier::Tier3
    } else {
        projected_tier
    };

    Ok(BudgetRoutingDecision {
        current_tier,
        projected_tier,
        classified_model_tier,
        effective_model_tier: if should_checkpoint {
            None
        } else {
            degradation_tier.cap_model_tier(classified_model_tier)
        },
        should_checkpoint,
    })
}

/// Observable context changes returned with each completion.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextAdjustment {
    pub original_message_count: usize,
    pub final_message_count: usize,
    /// Messages removed by collapsing completed request groups to their user
    /// request and final assistant outcome.
    #[serde(default)]
    pub request_group_compacted_message_count: usize,
    /// Net messages removed from older completed tool exchanges in the active
    /// request group. Recent and unresolved tool protocol remains verbatim.
    #[serde(default)]
    pub active_tool_compacted_message_count: usize,
    /// Total net message reduction across request-group projection, semantic
    /// history summarization, and active tool-history compaction.
    pub compressed_message_count: usize,
    pub reset_message_count: usize,
    pub compression_mode: Option<CompressionMode>,
    /// Estimated token count before context compaction.
    /// Zero when neither operation changed the request.
    pub original_token_count: u64,
    /// Estimated token count after context compaction.
    /// Zero when neither operation changed the request.
    pub final_token_count: u64,
    /// Serialized byte size before context compaction.
    /// Zero when neither operation changed the request.
    pub original_byte_count: u64,
    /// Serialized byte size after context compaction.
    /// Zero when neither operation changed the request.
    pub final_byte_count: u64,
}

impl ContextAdjustment {
    /// The request was left exactly as the caller supplied it.
    pub fn unchanged(message_count: usize) -> Self {
        Self {
            original_message_count: message_count,
            final_message_count: message_count,
            ..Self::default()
        }
    }
}

/// Last request-group compressed for each session.
fn compressed_turns() -> &'static Mutex<HashMap<Uuid, (String, Instant)>> {
    static TURNS: OnceLock<Mutex<HashMap<Uuid, (String, Instant)>>> = OnceLock::new();
    TURNS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// How long a session's compression bookkeeping survives inactivity.
const COMPRESSION_TURN_TTL: Duration = Duration::from_secs(30 * 60);

/// Upper bound on tracked sessions before expired entries are reclaimed.
const MAX_TRACKED_COMPRESSION_TURNS: usize = 4096;

/// Whether proactive compression should run for this turn.
///
/// Inside one user prompt the transcript only ever grows, so the provider's
/// prompt-prefix cache stays warm and cached input is billed at a fraction of
/// the normal rate. Stable history is selected once per request group;
/// continuation calls reproduce that prefix deterministically. Only an active
/// loop that crosses its token budget rewrites older completed tool rounds.
///
/// Process-local: a second replica may compress the same turn once more. That
/// costs one extra cache miss, never correctness.
pub fn compression_turn_is_new(session_id: Uuid, group_id: &str) -> bool {
    let now = Instant::now();
    let mut turns = match compressed_turns().lock() {
        Ok(turns) => turns,
        Err(poisoned) => poisoned.into_inner(),
    };

    if let Some((last_group, last_seen)) = turns.get_mut(&session_id) {
        if last_group == group_id && now.duration_since(*last_seen) <= COMPRESSION_TURN_TTL {
            // TTL is based on inactivity, not the first request in a long tool
            // loop. Refresh it whenever the same group is observed.
            *last_seen = now;
            return false;
        }
    }

    if turns.len() >= MAX_TRACKED_COMPRESSION_TURNS {
        turns.retain(|_, (_, last_seen)| now.duration_since(*last_seen) <= COMPRESSION_TURN_TTL);
        if turns.len() >= MAX_TRACKED_COMPRESSION_TURNS {
            if let Some(oldest) = turns
                .iter()
                .min_by_key(|(_, (_, last_seen))| *last_seen)
                .map(|(session_id, _)| *session_id)
            {
                turns.remove(&oldest);
            }
        }
    }

    turns.insert(session_id, (group_id.to_string(), now));
    true
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextWindowReset {
    pub limit_tokens: u64,
    pub input_tokens_before: u64,
    pub input_tokens_after: u64,
    pub reset_tokens: u64,
    pub reset_message_count: usize,
    /// What a successor session inherits from the compacted request.
    pub handoff: SessionHandoff,
}

impl ContextWindowReset {
    /// Clients may resend the entire original transcript on every tool round.
    /// Trimming that replay is not evidence that the active physical session
    /// filled up again. A new handoff requires its last prepared context to
    /// have reached the threshold and to actually become smaller afterwards.
    pub fn requires_session_rollover(&self, prepared_context_tokens: u64) -> bool {
        self.reset_tokens > 0
            && prepared_context_tokens >= context_compaction_target(self.limit_tokens)
            && self.input_tokens_after < prepared_context_tokens
    }
}

pub(crate) fn context_compaction_target(limit_tokens: u64) -> u64 {
    let percent = std::env::var("REINCORE_CONTEXT_COMPACT_PERCENT")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| (50..=100).contains(value))
        .unwrap_or(80);
    limit_tokens.saturating_mul(percent) / 100
}

/// Apply configured compaction and the Tier-3 aggressive compaction rule to
/// the outbound request. The caller receives exact message counts for logging
/// and response metadata.
pub fn prepare_context(
    request: &mut ChatCompletionRequest,
    settings: CompressionSettings,
    budget_tier: BudgetTier,
) -> ContextAdjustment {
    let original_message_count = request.messages.len();
    let original_metrics = estimate_message_metrics(&request.messages);
    crate::user_content::normalize_previous_user_messages(request);
    // Deduplicate file/image attachments across the entire compiled session.
    // Historical messages and the current request are all present. Removing
    // duplicate data URLs here ensures each unique file is sent only once.
    request.deduplicate_attachments();
    // Completed user-request groups do not need their raw intermediate tool
    // transcript in future turns. Project each one to the normalized request
    // and final assistant outcome before applying semantic compaction.
    let group_removed =
        crate::conversation::compact_completed_message_groups(&mut request.messages);
    let mode = if budget_tier >= BudgetTier::Tier3 {
        Some(CompressionMode::Max)
    } else {
        settings.mode
    };

    let Some(mode) = mode else {
        let changed = group_removed > 0;
        let final_metrics = changed.then(|| estimate_message_metrics(&request.messages));
        return ContextAdjustment {
            original_message_count,
            final_message_count: request.messages.len(),
            request_group_compacted_message_count: group_removed,
            active_tool_compacted_message_count: 0,
            compressed_message_count: group_removed,
            reset_message_count: 0,
            // Even though full compaction is disabled, request-group
            // compaction may have removed messages. Record the configured
            // mode (if any) so the event log is accurate.
            compression_mode: settings.mode,
            original_token_count: changed.then_some(original_metrics.0).unwrap_or(0),
            final_token_count: final_metrics.map(|metrics| metrics.0).unwrap_or(0),
            original_byte_count: changed.then_some(original_metrics.1).unwrap_or(0),
            final_byte_count: final_metrics.map(|metrics| metrics.1).unwrap_or(0),
        };
    };

    let result = ContextCompressor::new().compress(mode, &request.messages, &[]);
    request.messages = result.retained_messages;
    let removed = original_message_count.saturating_sub(request.messages.len());

    let (final_token_count, final_byte_count) = estimate_message_metrics(&request.messages);

    ContextAdjustment {
        original_message_count,
        final_message_count: request.messages.len(),
        request_group_compacted_message_count: group_removed,
        active_tool_compacted_message_count: result.active_tool_compacted_message_count,
        compressed_message_count: removed,
        reset_message_count: if budget_tier >= BudgetTier::Tier3 {
            removed
        } else {
            0
        },
        compression_mode: Some(mode),
        original_token_count: original_metrics.0,
        final_token_count,
        original_byte_count: original_metrics.1,
        final_byte_count,
    }
}

/// Prepare a later call in the same tool loop. Enabled compaction is applied
/// against the stable history prefix. The active request remains verbatim
/// until its mode-specific token threshold is crossed, after which older
/// closed tool rounds are summarized at safe protocol boundaries.
pub fn prepare_context_continuation(
    request: &mut ChatCompletionRequest,
    settings: CompressionSettings,
    budget_tier: BudgetTier,
) -> ContextAdjustment {
    let original_message_count = request.messages.len();
    let original_metrics = estimate_message_metrics(&request.messages);
    crate::user_content::normalize_previous_user_messages(request);
    // Deduplicate attachments on continuation calls too - the active tool
    // loop may re-send the same file across multiple provider calls.
    request.deduplicate_attachments();
    let group_removed =
        crate::conversation::compact_completed_message_groups(&mut request.messages);

    let mode = if budget_tier >= BudgetTier::Tier3 {
        Some(CompressionMode::Max)
    } else {
        settings.mode
    };
    if let Some(mode) = mode {
        let result = ContextCompressor::new().compress(mode, &request.messages, &[]);
        request.messages = result.retained_messages;
        let removed = original_message_count.saturating_sub(request.messages.len());
        if removed == 0 {
            return ContextAdjustment {
                original_message_count,
                final_message_count: request.messages.len(),
                request_group_compacted_message_count: group_removed,
                active_tool_compacted_message_count: result.active_tool_compacted_message_count,
                compression_mode: Some(mode),
                ..ContextAdjustment::default()
            };
        }
        let final_metrics = estimate_message_metrics(&request.messages);
        return ContextAdjustment {
            original_message_count,
            final_message_count: request.messages.len(),
            request_group_compacted_message_count: group_removed,
            active_tool_compacted_message_count: result.active_tool_compacted_message_count,
            compressed_message_count: removed,
            reset_message_count: if budget_tier >= BudgetTier::Tier3 {
                removed
            } else {
                0
            },
            compression_mode: Some(mode),
            original_token_count: original_metrics.0,
            final_token_count: final_metrics.0,
            original_byte_count: original_metrics.1,
            final_byte_count: final_metrics.1,
        };
    }

    if group_removed == 0 {
        return ContextAdjustment::unchanged(original_message_count);
    }
    let final_metrics = estimate_message_metrics(&request.messages);
    ContextAdjustment {
        original_message_count,
        final_message_count: request.messages.len(),
        request_group_compacted_message_count: group_removed,
        active_tool_compacted_message_count: 0,
        compressed_message_count: group_removed,
        reset_message_count: 0,
        compression_mode: None,
        original_token_count: original_metrics.0,
        final_token_count: final_metrics.0,
        original_byte_count: original_metrics.1,
        final_byte_count: final_metrics.1,
    }
}

/// Rough token estimate for a slice of chat messages. Uses a 4-chars-per-token
/// heuristic (the same default as the cost predictor's fallback) so the
/// compaction report does not depend on a tokenizer being available.
fn estimate_message_metrics(messages: &[ChatMessage]) -> (u64, u64) {
    #[derive(Default)]
    struct ByteCounter(u64);

    impl Write for ByteCounter {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.saturating_add(buffer.len() as u64);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let mut chars = 0u64;
    let mut serialized = ByteCounter::default();
    for message in messages {
        chars = chars
            .saturating_add(message.role.len() as u64)
            // Use content_len() to account for multipart messages including
            // file/image data URL payloads, not just plain text.
            .saturating_add(message.content_len() as u64);
        chars =
            chars.saturating_add(crate::provider::replay::estimation_text(message).len() as u64);
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                chars = chars
                    .saturating_add(call.function.name.len() as u64)
                    .saturating_add(call.function.arguments.len() as u64);
            }
        }
        // Count serializer output without allocating a String per message.
        let _ = serde_json::to_writer(&mut serialized, message);
    }
    (chars.saturating_add(3) / 4, serialized.0)
}

/// Inject a hidden system nudge immediately before the latest user turn. This
/// keeps the user message last for providers that validate turn ordering and
/// keeps the nudge out of the leading cacheable system prefix.
pub fn inject_system_nudge(request: &mut ChatCompletionRequest, message: String) {
    let insertion_index = request
        .messages
        .iter()
        .rposition(|item| {
            if !item.role.eq_ignore_ascii_case("user") {
                return false;
            }
            // Handle both text-only and multipart user messages.
            if let Some(content) = item.content_str() {
                crate::user_content::session_user_request(content).is_some()
            } else {
                let text = item.full_text();
                !text.is_empty() && crate::user_content::session_user_request(&text).is_some()
            }
        })
        .unwrap_or(request.messages.len());
    request.messages.insert(
        insertion_index,
        ChatMessage {
            provider_state: None,
            role: "system".to_string(),
            content: Some(MessageContent::text(message)),
            tool_calls: None,
            tool_call_id: None,
            name: Some("reincore_loop_intervention".to_string()),
        },
    );
}

/// Enforce a provider model's input window.
///
/// Overflowing means the session is full, so the request is not merely squeezed
/// back under the limit - it is compacted to what a successor session should
/// start from (see [`crate::handoff`]), which leaves that successor room to work
/// instead of overflowing again on the next turn.
pub fn enforce_context_window(
    request: &mut ChatCompletionRequest,
    predictor: &CostPredictor,
    model_tier: ModelTier,
    limit_tokens: u64,
) -> EngineResult<ContextWindowReset> {
    let before = predictor.predict(request, model_tier)?.input_tokens;
    let target = context_compaction_target(limit_tokens);
    if before <= target {
        return Ok(ContextWindowReset {
            limit_tokens,
            input_tokens_before: before,
            input_tokens_after: before,
            ..ContextWindowReset::default()
        });
    }

    let original_message_count = request.messages.len();
    let mut handoff = compact_for_new_session(request, HandoffPolicy::default());
    // A long active tool loop can fill the window by itself. Compact only
    // closed rounds; pending calls and their pairing remain intact.
    ContextCompressor::new().compact_active_to_budget(&mut request.messages, (target / 2) as usize);
    let mut current = predictor.predict(request, model_tier)?.input_tokens;

    // Only the system prefix, the carried context and the current turn are left.
    // Shed the carried context before giving up on the request entirely.
    if current > limit_tokens {
        let before_retain = request.messages.len();
        request
            .messages
            .retain(|message| message.name.as_deref() != Some(HANDOFF_MESSAGE_NAME));
        if request.messages.len() != before_retain {
            handoff.summarized_messages = 0;
            handoff.carried_context = handoff.verbatim_messages > 0;
        }
        current = predictor.predict(request, model_tier)?.input_tokens;
    }
    if current > limit_tokens {
        return Err(EngineError::ContextWindowExceeded {
            required: current,
            limit: limit_tokens,
        });
    }

    Ok(ContextWindowReset {
        limit_tokens,
        input_tokens_before: before,
        input_tokens_after: current,
        reset_tokens: before.saturating_sub(current),
        reset_message_count: original_message_count.saturating_sub(request.messages.len()),
        handoff,
    })
}

/// Parse and validate a caller-supplied semantic-cache embedding.
pub fn parse_embedding(request: &ChatCompletionRequest) -> EngineResult<Option<Vec<f32>>> {
    let Some(value) = request.extra.get("reincore_embedding") else {
        return Ok(None);
    };
    let values = value.as_array().ok_or_else(|| {
        EngineError::InvalidInput("reincore_embedding must be an array".to_string())
    })?;
    if values.is_empty() {
        return Err(EngineError::InvalidInput(
            "reincore_embedding cannot be empty".to_string(),
        ));
    }
    if values.len() != 384 {
        return Err(EngineError::InvalidInput(format!(
            "reincore_embedding must contain exactly 384 values, got {}",
            values.len()
        )));
    }
    values
        .iter()
        .map(|value| {
            value
                .as_f64()
                .filter(|number| number.is_finite())
                .map(|number| number as f32)
                .ok_or_else(|| {
                    EngineError::InvalidInput(
                        "reincore_embedding must contain only finite numbers".to_string(),
                    )
                })
        })
        .collect::<EngineResult<Vec<_>>>()
        .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_include_replay_without_changing_visible_content() {
        let mut message: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant", "content": "visible answer"
        }))
        .unwrap();
        let before = estimate_message_metrics(&[message.clone()]);
        message.provider_state = crate::provider::replay::capture(
            "scope".into(),
            &serde_json::json!({"thinking":"r".repeat(400)}),
            false,
        )
        .map(|state| state.bind(&message).for_request_group("group"));
        let after = estimate_message_metrics(&[message.clone()]);
        assert_eq!(after.0, before.0 + 100);
        assert!(after.1 > before.1);
        assert_eq!(message.full_text(), "visible answer");
    }
    use crate::models::ChatMessage;

    #[test]
    fn replayed_uncompacted_history_does_not_fill_a_successor_session() {
        let reset = ContextWindowReset {
            limit_tokens: 128_000,
            input_tokens_before: 296_317,
            input_tokens_after: 66_720,
            reset_tokens: 229_597,
            ..Default::default()
        };
        assert!(reset.requires_session_rollover(296_317));
        assert!(!reset.requires_session_rollover(0));
        // The original transcript is replayed after reconnect and on later
        // user turns. Its size does not describe the successor's context.
        for prepared in [66_720, 70_188, 91_231] {
            let replay = ContextWindowReset {
                input_tokens_after: prepared + 1,
                reset_tokens: reset.input_tokens_before - (prepared + 1),
                ..reset.clone()
            };
            assert!(!replay.requires_session_rollover(prepared));
        }
        // A genuinely full successor can still hand off its active work.
        assert!(reset.requires_session_rollover(128_000));
        let growing = ContextWindowReset {
            input_tokens_after: 128_001,
            ..reset
        };
        assert!(!growing.requires_session_rollover(128_000));
    }

    fn request(message_count: usize) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "router".to_string(),
            messages: (0..message_count)
                .map(|index| ChatMessage {
                    provider_state: None,
                    role: if index == 0 { "system" } else { "user" }.to_string(),
                    content: Some(MessageContent::text(format!("message {index}"))),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                })
                .collect(),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn predicted_exhaustion_degrades_without_premature_checkpoint() {
        let mut session = Session::new("task", 100).unwrap();
        session.spend_cents = 95;
        let decision = resolve_budget_routing(Some(&session), ModelTier::Ultra, 10).unwrap();
        assert_eq!(decision.projected_tier, BudgetTier::Tier4);
        assert_eq!(decision.effective_model_tier, Some(ModelTier::Lite));
        assert!(!decision.should_checkpoint);
    }

    #[test]
    fn tier_three_forces_max_compression_and_reports_counts() {
        let mut request = request(10);
        let report = prepare_context(
            &mut request,
            CompressionSettings::disabled(),
            BudgetTier::Tier3,
        );
        assert_eq!(report.compression_mode, Some(CompressionMode::Max));
        assert!(report.reset_message_count > 0);
        assert_eq!(report.final_message_count, request.messages.len());
    }

    #[test]
    fn compression_reports_token_and_byte_reduction() {
        let mut request = request(10);
        let report = prepare_context(
            &mut request,
            CompressionSettings::default_enabled(),
            BudgetTier::Tier1,
        );
        // When compression removes messages, the before/after token and byte
        // counts must be populated and reflect the reduction.
        if report.compressed_message_count > 0 {
            assert!(report.original_token_count > report.final_token_count);
            assert!(report.original_byte_count > report.final_byte_count);
            assert_eq!(report.final_message_count, request.messages.len());
        } else {
            // If nothing was removed, before and after are equal and nonzero.
            assert_eq!(report.original_token_count, report.final_token_count);
            assert_eq!(report.original_byte_count, report.final_byte_count);
            assert!(report.original_token_count > 0);
            assert!(report.original_byte_count > 0);
        }
    }

    #[test]
    fn mid_compression_projects_completed_request_groups_first() {
        let message = |role: &str, content: String| ChatMessage {
            provider_state: None,
            role: role.to_string(),
            content: Some(MessageContent::text(content)),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let mut messages = vec![
            message("system", "You are a coding agent.".to_string()),
            message("user", "Help me design agent attribution.".to_string()),
        ];
        for index in 2..54 {
            let content = if index == 35 {
                "Use a delegation profile header and expose it in task metadata.".to_string()
            } else {
                format!("discussion message {index}")
            };
            messages.push(message("assistant", content));
        }
        messages.push(message(
            "user",
            "Please implement this feature.".to_string(),
        ));
        let mut request = ChatCompletionRequest {
            model: "router".to_string(),
            messages,
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        };

        let report = prepare_context(
            &mut request,
            CompressionSettings::new(CompressionMode::Mid),
            BudgetTier::Tier1,
        );

        // The completed first request is reduced to its request and final
        // assistant outcome before semantic compaction. Referential wording
        // keeps that preceding exchange alongside the current request.
        assert_eq!(report.original_message_count, 55);
        assert_eq!(report.request_group_compacted_message_count, 51);
        assert_eq!(report.final_message_count, 4);
        assert_eq!(
            request.messages[2].content_str(),
            Some("discussion message 53")
        );
        assert_eq!(
            request.messages.last().and_then(ChatMessage::content_str),
            Some("Please implement this feature.")
        );
    }

    #[test]
    fn second_request_drops_the_completed_long_tool_loop() {
        let message = |role: &str, content: Option<String>| ChatMessage {
            provider_state: None,
            role: role.to_string(),
            content: content.map(MessageContent::text),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let mut messages = vec![message(
            "user",
            Some("Complete the first task.".to_string()),
        )];
        for index in 0..193 {
            let call_id = format!("prior-{index}");
            messages.push(ChatMessage {
                provider_state: None,
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![crate::models::ToolCall {
                    id: call_id.clone(),
                    call_type: "function".to_string(),
                    function: crate::models::FunctionCall {
                        name: "inspect".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tool_call_id: None,
                name: None,
            });
            messages.push(ChatMessage {
                provider_state: None,
                role: "tool".to_string(),
                content: Some(MessageContent::text(format!("prior result {index}"))),
                tool_calls: None,
                tool_call_id: Some(call_id),
                name: Some("inspect".to_string()),
            });
        }
        messages.push(message(
            "assistant",
            Some("The first task is complete.".to_string()),
        ));
        messages.push(message("user", Some("Start the second task.".to_string())));
        assert_eq!(messages.len(), 389);

        let mut request = ChatCompletionRequest {
            model: "router".to_string(),
            messages,
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        };

        let report = prepare_context(
            &mut request,
            CompressionSettings::new(CompressionMode::Mid),
            BudgetTier::Tier1,
        );

        assert_eq!(report.original_message_count, 389);
        assert_eq!(report.request_group_compacted_message_count, 386);
        assert_eq!(report.final_message_count, 3);
        assert_eq!(report.compressed_message_count, 386);
        assert!(request
            .messages
            .iter()
            .all(|message| message.role != "tool"));
        assert_eq!(
            request.messages.last().and_then(ChatMessage::content_str),
            Some("Start the second task.")
        );
    }

    #[test]
    fn disabled_compression_leaves_token_and_byte_counts_at_zero() {
        let mut request = request(5);
        let report = prepare_context(
            &mut request,
            CompressionSettings::disabled(),
            BudgetTier::Tier1,
        );
        assert_eq!(report.compression_mode, None);
        assert_eq!(report.original_token_count, 0);
        assert_eq!(report.final_token_count, 0);
        assert_eq!(report.original_byte_count, 0);
        assert_eq!(report.final_byte_count, 0);
    }

    #[test]
    fn continuation_reuses_mid_compression_and_keeps_current_tool_protocol() {
        let message = |role: &str, content: Option<&str>| ChatMessage {
            provider_state: None,
            role: role.to_string(),
            content: content.map(|s| crate::MessageContent::text(s.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let mut request = ChatCompletionRequest {
            model: "router".to_string(),
            messages: vec![
                message("system", Some("system")),
                message(
                    "user",
                    Some("<context>old</context><userRequest>first task</userRequest>"),
                ),
                message("assistant", Some("intermediate")),
                message("tool", Some("large result")),
                message("assistant", Some("Finished first task.")),
                message(
                    "user",
                    Some("<context>current</context><userRequest>second task</userRequest>"),
                ),
                message("assistant", Some("working")),
                message("tool", Some("current result")),
            ],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        };

        let report = prepare_context_continuation(
            &mut request,
            CompressionSettings::new(CompressionMode::Mid),
            BudgetTier::Tier1,
        );

        assert_eq!(report.request_group_compacted_message_count, 2);
        assert_eq!(report.compression_mode, Some(CompressionMode::Mid));
        assert_eq!(request.messages.len(), 5);
        assert_eq!(request.messages[1].name.as_deref(), Some("context_summary"));
        assert!(request.messages[1].full_text().contains("first task"));
        assert_eq!(
            request.messages[2].content_str(),
            Some("<context>current</context><userRequest>second task</userRequest>")
        );
        assert_eq!(request.messages[4].content_str(), Some("current result"));
    }

    #[test]
    fn compression_runs_once_per_request_group() {
        let session = Uuid::new_v4();
        // First turn of a prompt selects a fresh stable-history boundary; the
        // tool-loop steps that follow reuse it so the prompt prefix is stable.
        assert!(compression_turn_is_new(session, "group-a"));
        assert!(!compression_turn_is_new(session, "group-a"));
        assert!(!compression_turn_is_new(session, "group-a"));

        // A new user prompt is a fresh turn boundary.
        assert!(compression_turn_is_new(session, "group-b"));
        assert!(!compression_turn_is_new(session, "group-b"));

        // Sessions are tracked independently.
        let other = Uuid::new_v4();
        assert!(compression_turn_is_new(other, "group-a"));
    }

    #[test]
    fn embedding_rejects_partial_or_non_numeric_vectors() {
        let mut request = request(1);
        request.extra.insert(
            "reincore_embedding".to_string(),
            serde_json::json!([0.1, "bad", 0.3]),
        );
        assert!(parse_embedding(&request).is_err());
    }

    #[test]
    fn context_window_reset_reports_removed_tokens() {
        let mut request = request(20);
        for message in request.messages.iter_mut() {
            message.content = Some(crate::MessageContent::text("long context ".repeat(200)));
        }
        let predictor = CostPredictor::new();
        let before = predictor
            .predict(&request, ModelTier::Lite)
            .unwrap()
            .input_tokens;
        let report =
            enforce_context_window(&mut request, &predictor, ModelTier::Lite, before / 2).unwrap();
        assert!(report.reset_tokens > 0);
        assert!(report.input_tokens_after <= report.limit_tokens);
    }

    #[test]
    fn proactive_compaction_leaves_headroom_before_the_hard_limit() {
        let mut request = request(20);
        for message in &mut request.messages {
            message.content = Some(MessageContent::text("earlier context ".repeat(200)));
        }
        let predictor = CostPredictor::new();
        let before = predictor
            .predict(&request, ModelTier::Lite)
            .unwrap()
            .input_tokens;
        let limit = before + before / 10;
        let report =
            enforce_context_window(&mut request, &predictor, ModelTier::Lite, limit).unwrap();
        assert!(before < limit);
        // At 100% compaction is explicitly disabled until the hard limit.
        let percent = std::env::var("REINCORE_CONTEXT_COMPACT_PERCENT")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|v| (50..=100).contains(v))
            .unwrap_or(80);
        if percent <= 90 {
            assert!(report.reset_tokens > 0);
        }
        assert!(report.input_tokens_after <= limit);
    }

    #[test]
    fn a_follow_up_keeps_its_summary_when_history_is_dropped() {
        let mut request = request(20);
        for message in request.messages.iter_mut() {
            message.content = Some(crate::MessageContent::text("long context ".repeat(200)));
        }
        request.messages.push(ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(crate::MessageContent::text(
                "now do the same again".to_string(),
            )),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
        let predictor = CostPredictor::new();
        let before = predictor
            .predict(&request, ModelTier::Lite)
            .unwrap()
            .input_tokens;

        enforce_context_window(&mut request, &predictor, ModelTier::Lite, before / 4).unwrap();

        assert!(
            request
                .messages
                .iter()
                .any(|message| message.name.as_deref() == Some(HANDOFF_MESSAGE_NAME)),
            "a follow-up turn must keep something to continue from"
        );
    }

    #[test]
    fn an_unsatisfiable_request_reports_the_window_it_exceeded() {
        let mut request = request(1);
        request.messages[0].role = "user".to_string();
        request.messages[0].content = Some(crate::MessageContent::text("x ".repeat(5_000)));
        let predictor = CostPredictor::new();
        let error = enforce_context_window(&mut request, &predictor, ModelTier::Lite, 10)
            .expect_err("a single oversized turn cannot be compacted");
        assert!(matches!(
            error,
            EngineError::ContextWindowExceeded { limit: 10, .. }
        ));
        assert!(error.is_context_overflow_error());
    }

    #[test]
    fn oversized_unresolved_tool_protocol_reports_context_window_error() {
        let mut request = request(1);
        request.messages[0].role = "user".to_string();
        request.messages[0].content = Some(crate::MessageContent::text("Continue safely."));
        request.messages.push(ChatMessage {
            provider_state: None,
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![crate::models::ToolCall {
                id: "pending".to_string(),
                call_type: "function".to_string(),
                function: crate::models::FunctionCall {
                    name: "inspect".to_string(),
                    arguments: format!("{{\"payload\":\"{}\"}}", "x ".repeat(10_000)),
                },
            }]),
            tool_call_id: None,
            name: None,
        });
        let predictor = CostPredictor::new();
        let error = enforce_context_window(&mut request, &predictor, ModelTier::Lite, 100)
            .expect_err("unresolved protocol cannot be truncated safely");
        assert!(matches!(
            error,
            EngineError::ContextWindowExceeded { limit: 100, .. }
        ));
    }
}
