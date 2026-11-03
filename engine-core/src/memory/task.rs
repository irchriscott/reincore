//! Durable task grouping inside a session.
//!
//! A request group is one user prompt and its tool loop. A task is the
//! longer-lived unit of work that may contain several related request groups.
//! Task metadata is stored on `LlmCall` events, so the append-only event log
//! remains the source of truth and older databases need no schema migration.

use crate::event::{SessionEvent, SessionEventType};
use crate::models::{ChatCompletionRequest, ChatMessage};
use crate::summary::derive_task_summary;
use crate::task_type::{classify_task_type, TaskType};
use crate::{is_effective_loop_intervention, CostOutcome};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use uuid::Uuid;

/// Request body field that explicitly selects a task.
pub const TASK_ID_EXTRA_KEY: &str = "reincore_task_id";

/// Optional request body field used as the name when an explicit task is new.
pub const TASK_NAME_EXTRA_KEY: &str = "reincore_task_name";

/// Optional request body field that explicitly classifies a new task.
pub const TASK_TYPE_EXTRA_KEY: &str = "reincore_task_type";

/// A task reconstructed from the session event log.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SessionTask {
    pub task_id: String,
    pub name: String,
    pub task_type: TaskType,
    pub first_prompt: String,
    pub request_group_ids: Vec<String>,
    pub first_event_seq: i64,
    pub last_event_seq: i64,
    pub request_count: usize,
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub cost_cents: i64,
    #[serde(default)]
    pub cost_nano_usd: i64,
    #[serde(default)]
    pub unknown_cost_calls: u64,
    #[serde(default)]
    pub completed: bool,
    #[serde(default)]
    pub clean_completion: bool,
    #[serde(default)]
    pub success_status: crate::cost_outcome::SuccessStatus,
    #[serde(default)]
    pub completed_after_intervention: bool,
    #[serde(default)]
    pub loop_intervention_count: u64,
    #[serde(default)]
    pub intervention_metrics: crate::InterventionMetrics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_quality_bps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_efficiency_score: Option<f64>,
    #[serde(default)]
    pub request_groups: Vec<RequestGroupOutcome>,
}

/// Cost/outcome projection for one user request and its tool loop.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RequestGroupOutcome {
    pub request_group_id: String,
    pub task_id: String,
    pub first_event_seq: i64,
    pub last_event_seq: i64,
    pub llm_call_count: usize,
    pub cost_cents: i64,
    #[serde(default)]
    pub cost_nano_usd: i64,
    #[serde(default)]
    pub unknown_cost_calls: u64,
    pub completed: bool,
    pub clean_completion: bool,
    #[serde(default)]
    pub success_status: crate::cost_outcome::SuccessStatus,
    #[serde(default)]
    pub completed_after_intervention: bool,
    pub loop_intervention_count: u64,
    #[serde(default)]
    pub intervention_metrics: crate::InterventionMetrics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_quality_bps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_efficiency_score: Option<f64>,
}

/// Lightweight session projection used by timeline clients. Raw event payloads
/// deliberately remain out of this response and are loaded by sequence range
/// only when the operator opens a request.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SessionTimeline {
    pub tasks: Vec<SessionTask>,
    pub request_groups: Vec<RequestGroupTimelineSummary>,
    pub total_event_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latest_seq: Option<i64>,
    /// PII/user-information catches masked by the privacy policy across all
    /// events in this timeline. Older payloads without the field count zero.
    #[serde(default)]
    pub pii_redaction_count: u64,
}

/// Display metadata for one user request and its contiguous event range.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RequestGroupTimelineSummary {
    pub request_group_id: String,
    pub task_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    pub first_event_seq: i64,
    pub last_event_seq: i64,
    pub event_count: usize,
    pub llm_call_count: usize,
    pub tool_call_count: usize,
    pub tool_error_count: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_cents: i64,
    #[serde(default)]
    pub cost_nano_usd: i64,
    #[serde(default)]
    pub unknown_cost_calls: u64,
    pub models: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<DateTime<Utc>>,
    pub completed: bool,
    pub clean_completion: bool,
    #[serde(default)]
    pub success_status: crate::cost_outcome::SuccessStatus,
    #[serde(default)]
    pub completed_after_intervention: bool,
    #[serde(default)]
    pub intervention_metrics: crate::InterventionMetrics,
    /// PII/user-information values the privacy policy masked in this request
    /// group. Older payloads without the field count zero.
    #[serde(default)]
    pub pii_redaction_count: u64,
}

/// The task selected for the current request.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskAssignment {
    pub task_id: String,
    pub task_name: String,
    pub task_type: TaskType,
    pub first_prompt: String,
    pub is_new: bool,
    /// User prompts and assistant context already known to belong to this
    /// task. This is kept out of API serialization and is used only to resolve
    /// follow-ups and scope the provider transcript.
    #[serde(skip)]
    related_prompts: Vec<String>,
}

impl TaskAssignment {
    pub fn related_prompts(&self) -> &[String] {
        &self.related_prompts
    }
}

#[derive(Clone, Debug)]
struct TaskCandidate {
    task: SessionTask,
    /// User prompts plus bounded assistant responses already associated with
    /// the task. Assistant context matters when a follow-up names an entity
    /// introduced by the answer rather than repeated in the original prompt.
    related_texts: Vec<String>,
}

const MAX_RELATED_RESPONSE_CHARS: usize = 8_192;

/// Reconstruct all task groups in first-seen order.
pub fn session_tasks(events: &[SessionEvent]) -> Vec<SessionTask> {
    let mut tasks: Vec<SessionTask> = task_candidates(events)
        .into_iter()
        .map(|candidate| candidate.task)
        .collect();
    project_cost_outcomes(events, &mut tasks);
    tasks
}

/// Build the small, task-aware projection used for the initial session page.
/// This is intentionally separate from the audit log response so very large
/// model transcripts never need to be serialized into the page bootstrap.
///
/// Durable `continuation` snapshots are execution state, not user-visible
/// activity: the settlement path rewrites them after nearly every tool call,
/// so counting them here turns every tool loop into timeline spam. They stay
/// in the full event log; the timeline only projects everything else.
pub fn session_timeline(events: &[SessionEvent]) -> SessionTimeline {
    let visible: Vec<SessionEvent> = events
        .iter()
        .filter(|event| event.event_type != SessionEventType::Continuation)
        .cloned()
        .collect();
    let events = &visible;
    let tasks = session_tasks(events);
    let mut request_groups = tasks
        .iter()
        .flat_map(|task| {
            task.request_groups
                .iter()
                .map(|group| RequestGroupTimelineSummary {
                    request_group_id: group.request_group_id.clone(),
                    task_id: group.task_id.clone(),
                    prompt: None,
                    first_event_seq: group.first_event_seq,
                    last_event_seq: group.last_event_seq,
                    event_count: 0,
                    llm_call_count: 0,
                    tool_call_count: 0,
                    tool_error_count: 0,
                    input_tokens: 0,
                    output_tokens: 0,
                    cost_cents: 0,
                    cost_nano_usd: 0,
                    unknown_cost_calls: 0,
                    models: Vec::new(),
                    started_at: None,
                    ended_at: None,
                    completed: group.completed,
                    clean_completion: group.clean_completion,
                    success_status: group.success_status.clone(),
                    completed_after_intervention: group.completed_after_intervention,
                    intervention_metrics: group.intervention_metrics.clone(),
                    pii_redaction_count: 0,
                })
        })
        .collect::<Vec<_>>();
    request_groups.sort_by_key(|group| group.first_event_seq);

    let mut group_index = 0_usize;
    for event in events {
        while group_index < request_groups.len()
            && event.seq > request_groups[group_index].last_event_seq
        {
            group_index += 1;
        }
        let Some(group) = request_groups.get_mut(group_index) else {
            break;
        };
        if event.seq < group.first_event_seq || event.seq > group.last_event_seq {
            continue;
        }

        group.event_count = group.event_count.saturating_add(1);
        group.started_at.get_or_insert(event.created_at);
        group.ended_at = Some(event.created_at);
        group.cost_nano_usd = group
            .cost_nano_usd
            .saturating_add(event.amount_nano_usd().unwrap_or(0));
        group.cost_cents = group.cost_nano_usd / crate::pricing::NANO_USD_PER_CENT;
        group.unknown_cost_calls += u64::from(event.amount_nano_usd().is_none());
        group.pii_redaction_count = group.pii_redaction_count.saturating_add(
            u64::try_from(crate::privacy::event_pii_redaction_count(&event.payload))
                .unwrap_or(u64::MAX),
        );

        match event.event_type {
            SessionEventType::LlmCall => {
                group.llm_call_count = group.llm_call_count.saturating_add(1);
                group.input_tokens = group.input_tokens.saturating_add(
                    event
                        .payload
                        .get("input_tokens")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0),
                );
                group.output_tokens = group.output_tokens.saturating_add(
                    event
                        .payload
                        .get("output_tokens")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0),
                );
                if group.prompt.is_none() {
                    group.prompt = event_prompt(event);
                }
                if let Some(model) = event
                    .payload
                    .get("model")
                    .and_then(serde_json::Value::as_str)
                {
                    if !group.models.iter().any(|known| known == model) {
                        group.models.push(model.to_string());
                    }
                }
            }
            SessionEventType::ToolCall => {
                group.tool_call_count = group.tool_call_count.saturating_add(1);
            }
            SessionEventType::ToolResult => {
                if event
                    .payload
                    .get("result")
                    .and_then(|result| result.get("is_error"))
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                {
                    group.tool_error_count = group.tool_error_count.saturating_add(1);
                }
            }
            _ => {}
        }
    }

    let pii_redaction_count = request_groups
        .iter()
        .map(|group| group.pii_redaction_count)
        .fold(0u64, u64::saturating_add);

    SessionTimeline {
        tasks,
        request_groups,
        total_event_count: events.len(),
        latest_seq: events.last().map(|event| event.seq),
        pii_redaction_count,
    }
}

/// Resolve the current request to an existing task or create a deterministic
/// new task. Explicit task ids win, then exact request-group continuity, then
/// prompt relatedness. Short/anaphoric follow-ups continue the latest task.
pub fn resolve_task(
    session_id: Option<Uuid>,
    request_group_id: &str,
    request: &ChatCompletionRequest,
    events: &[SessionEvent],
) -> TaskAssignment {
    resolve_task_inner(session_id, request_group_id, request, events, None, None)
}

/// Mode-dispatched task resolution. In `llm` mode the judge decides the
/// current prompt's boundary/type verdicts in one analysis call and the
/// deterministic assignment logic applies them; any judge failure falls back
/// to the lexical path with a warning. Inner transcript loops always keep
/// their lexical heuristics.
pub async fn resolve_task_auto(
    session_id: Option<Uuid>,
    request_group_id: &str,
    request: &ChatCompletionRequest,
    events: &[SessionEvent],
    boundary_mode: crate::detection::DetectionMode,
    type_mode: crate::detection::DetectionMode,
    judge: Option<&crate::detection::LlmJudge>,
) -> TaskAssignment {
    use crate::detection::DetectionMode::Llm;
    let judge = match (boundary_mode, type_mode, judge) {
        (Llm, _, Some(judge)) | (_, Llm, Some(judge)) => judge,
        _ => {
            if (boundary_mode.is_llm() || type_mode.is_llm()) && judge.is_none() {
                crate::detection::missing_judge_warning("task_assignment");
            }
            return resolve_task(session_id, request_group_id, request, events);
        }
    };
    // Explicit ids already in history and request-group continuity are
    // decided deterministically; asking the judge would only pay for a
    // discarded verdict.
    let candidates = task_candidates(events);
    let explicit_hit = request
        .extra
        .get(TASK_ID_EXTRA_KEY)
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .is_some_and(|explicit_id| {
            candidates
                .iter()
                .any(|candidate| candidate.task.task_id == explicit_id)
        });
    let has_continuity = candidates.iter().any(|candidate| {
        candidate
            .task
            .request_group_ids
            .iter()
            .any(|group| group == request_group_id)
    });
    if explicit_hit || has_continuity {
        return resolve_task(session_id, request_group_id, request, events);
    }
    let prompt = latest_user_prompt(request).unwrap_or_else(|| "Untitled task".to_string());
    // Most recent by activity (not creation order), with the opening prompt
    // so follow-ups to older tasks are recognizable to the judge.
    let mut by_recency: Vec<&TaskCandidate> = candidates.iter().collect();
    by_recency.sort_by_key(|candidate| candidate.task.last_event_seq);
    let recent: Vec<String> = by_recency
        .iter()
        .rev()
        .take(5)
        .map(|candidate| {
            let first: String = candidate.task.first_prompt.chars().take(160).collect();
            format!("{} | {first}", candidate.task.name)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    match judge.judge_task_assignment(&prompt, &recent).await {
        Ok((new_task, task_type)) => resolve_task_inner(
            session_id,
            request_group_id,
            request,
            events,
            (boundary_mode == Llm).then_some(new_task),
            (type_mode == Llm).then_some(task_type),
        ),
        Err(error) => {
            crate::detection::fallback_warning("task_assignment", &error);
            resolve_task(session_id, request_group_id, request, events)
        }
    }
}

fn resolve_task_inner(
    session_id: Option<Uuid>,
    request_group_id: &str,
    request: &ChatCompletionRequest,
    events: &[SessionEvent],
    boundary_override: Option<bool>,
    type_override: Option<TaskType>,
) -> TaskAssignment {
    let prompt = latest_user_prompt(request).unwrap_or_else(|| "Untitled task".to_string());
    let candidates = task_candidates(events);

    if let Some(explicit_id) = request
        .extra
        .get(TASK_ID_EXTRA_KEY)
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if let Some(candidate) = candidates
            .iter()
            .find(|candidate| candidate.task.task_id == explicit_id)
        {
            return assignment_from_candidate(candidate, false);
        }
        let name = request
            .extra
            .get(TASK_NAME_EXTRA_KEY)
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| name_from_prompt(request, &prompt));
        let task_type = explicit_task_type(request)
            .or(type_override)
            .unwrap_or_else(|| classify_task_type(&prompt));
        return TaskAssignment {
            task_id: explicit_id.to_string(),
            task_name: name,
            task_type,
            first_prompt: prompt.clone(),
            is_new: true,
            related_prompts: vec![prompt],
        };
    }

    if let Some(candidate) = candidates.iter().find(|candidate| {
        candidate
            .task
            .request_group_ids
            .iter()
            .any(|group| group == request_group_id)
    }) {
        return assignment_from_candidate(candidate, false);
    }

    let starts_new_task =
        boundary_override.unwrap_or_else(|| crate::lexical::starts_new_task(&prompt));
    // Greetings, thanks, and acknowledgements are not work: they ride along
    // with the current task instead of creating or extending one.
    if !starts_new_task && crate::lexical::is_non_task_chatter(&prompt) {
        if let Some(candidate) = most_recent_candidate(&candidates) {
            return assignment_from_candidate(candidate, false);
        }
    }
    if !starts_new_task && is_follow_up(&prompt) {
        if let Some(candidate) = most_recent_candidate(&candidates) {
            return assignment_from_candidate(candidate, false);
        }
    }

    let best = candidates
        .iter()
        .filter(|_| !starts_new_task)
        .filter_map(|candidate| {
            let score = candidate
                .related_texts
                .iter()
                .chain(std::iter::once(&candidate.task.name))
                .map(|known| relatedness(&prompt, known))
                .fold(0.0_f32, f32::max);
            (score >= 0.4).then_some((candidate, score))
        })
        .max_by(|(left_task, left_score), (right_task, right_score)| {
            left_score.total_cmp(right_score).then(
                left_task
                    .task
                    .last_event_seq
                    .cmp(&right_task.task.last_event_seq),
            )
        });
    if let Some((candidate, _)) = best {
        return assignment_from_candidate(candidate, false);
    }

    // An LLM verdict of "not a new task" is binding when work exists: even
    // if lexical signals found nothing to attach to, the prompt continues
    // the most recent task instead of spawning a fresh one.
    if boundary_override == Some(false) {
        if let Some(candidate) = most_recent_candidate(&candidates) {
            return assignment_from_candidate(candidate, false);
        }
    }

    let name = name_from_prompt(request, &prompt);
    TaskAssignment {
        task_id: deterministic_task_id(
            session_id,
            &prompt,
            starts_new_task.then_some(request_group_id),
        ),
        task_name: name,
        task_type: explicit_task_type(request)
            .or(type_override)
            .unwrap_or_else(|| classify_task_type(&prompt)),
        first_prompt: prompt.clone(),
        is_new: true,
        related_prompts: vec![prompt],
    }
}

/// Remove completed transcript segments belonging to other tasks. Leading
/// system messages and the current user segment are always retained. When a
/// referential follow-up is detected, retain its immediately preceding answer;
/// if it cannot be tied to persisted task history, retain a bounded wider tail
/// as its antecedent rather than leaving phrases such as "implement the
/// feature" without the preceding explanation. Returns the number of messages
/// removed.
pub fn scope_request_to_task(
    request: &mut ChatCompletionRequest,
    assignment: &TaskAssignment,
) -> usize {
    const FOLLOW_UP_CONTEXT_TURNS: usize = 3;

    // Extract each prompt once. Wrapped and multipart prompts can be large;
    // re-parsing them for every heuristic needlessly rescans the transcript.
    let user_turns: Vec<(usize, String)> = request
        .messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| user_prompt(message).map(|prompt| (index, prompt)))
        .collect();
    if user_turns.is_empty() {
        return 0;
    }

    let current_turn = user_turns.last().expect("at least one user turn").0;
    // A declared task change starts a fresh transcript even when two feature
    // specifications share domain vocabulary. Keep this boundary through the
    // tool loop and subsequent follow-ups, not just the first HTTP call.
    let task_boundary = crate::lexical::starts_new_task(&assignment.first_prompt)
        .then(|| {
            user_turns
                .iter()
                .rposition(|(_, prompt)| prompt == &assignment.first_prompt)
        })
        .flatten();
    let mut within_selected_boundary = task_boundary.is_none();
    let eligible: Vec<bool> = user_turns
        .iter()
        .enumerate()
        .map(|(position, (_, prompt))| {
            if crate::lexical::starts_new_task(prompt) {
                within_selected_boundary = prompt == &assignment.first_prompt
                    || assignment
                        .related_prompts
                        .iter()
                        .any(|known| known == prompt);
            }
            task_boundary.map_or(true, |boundary| position >= boundary) && within_selected_boundary
        })
        .collect();
    let known_terms: Vec<_> = assignment
        .related_prompts
        .iter()
        .chain(std::iter::once(&assignment.task_name))
        .map(|known| (known, std::cell::OnceCell::new()))
        .collect();
    let directly_related = user_turns
        .iter()
        .enumerate()
        .map(|(position, (_, prompt))| {
            // The current turn is retained unconditionally. Only older,
            // eligible turns need topic matching, with lazily cached terms.
            if !eligible[position] || position + 1 == user_turns.len() {
                return false;
            }
            let terms = crate::lexical::relevance_terms(prompt);
            known_terms.iter().any(|(known, cached)| {
                relatedness_terms(
                    &terms,
                    cached.get_or_init(|| crate::lexical::relevance_terms(known)),
                ) >= 0.4
            })
        })
        .collect::<Vec<_>>();
    let has_known_prior_context = directly_related[..directly_related.len() - 1]
        .iter()
        .any(|related| *related);
    let current_prompt = &user_turns.last().expect("at least one user turn").1;
    let fallback_start = is_follow_up(current_prompt).then(|| {
        // A referential request always needs its immediately preceding answer.
        // When persisted task metadata cannot identify any earlier context,
        // keep a slightly wider tail so nested follow-ups retain their anchor.
        let antecedent_turns = if has_known_prior_context {
            1
        } else {
            FOLLOW_UP_CONTEXT_TURNS
        };
        user_turns.len().saturating_sub(1 + antecedent_turns)
    });

    // Select in place instead of cloning retained messages and their possibly
    // large attachments/tool output on every step of an agent loop.
    let mut keep = vec![false; request.messages.len()];
    let mut kept_count = 0;
    for (index, message) in request.messages[..user_turns[0].0].iter().enumerate() {
        // An ignored/removed summary can leave old assistant/tool messages
        // before the first substantive turn. They are history, not global
        // instructions, and must not leak across a declared task boundary.
        let keep_leading = task_boundary.is_none()
            || message.role.eq_ignore_ascii_case("system")
            || message.role.eq_ignore_ascii_case("developer")
            || (message.role.eq_ignore_ascii_case("user")
                && !crate::user_content::is_external_summary_message(&message.full_text())
                && crate::user_content::is_user_scaffolding_only(&message.full_text()));
        keep[index] = keep_leading;
        kept_count += usize::from(keep_leading);
    }
    let mut previous_belongs = false;
    let mut retained_task_start = None;

    for (position, (start, prompt)) in user_turns.iter().enumerate() {
        let start = *start;
        let end = user_turns
            .get(position + 1)
            .map(|(start, _)| *start)
            .unwrap_or(request.messages.len());
        let is_fallback_antecedent =
            fallback_start.is_some_and(|fallback| position >= fallback && start < current_turn);
        let belongs = start == current_turn
            || (eligible[position]
                && (directly_related[position]
                    || is_fallback_antecedent
                    || (previous_belongs && is_follow_up(&prompt))));
        if belongs {
            if task_boundary == Some(position) {
                retained_task_start = Some(kept_count);
            }
            keep[start..end].fill(true);
            kept_count += end - start;
        }
        previous_belongs = belongs;
    }

    let removed = request.messages.len().saturating_sub(kept_count);
    let mut index = 0;
    request.messages.retain(|_| {
        let retained = keep[index];
        index += 1;
        retained
    });
    let summaries_removed = retained_task_start.map_or(0, |task_start| {
        crate::user_content::discard_prior_task_summaries(request, task_start)
    });
    summaries_removed + removed
}

fn task_candidates(events: &[SessionEvent]) -> Vec<TaskCandidate> {
    let mut by_id = BTreeMap::<String, TaskCandidate>::new();
    let mut order = Vec::<String>::new();

    for event in events
        .iter()
        .filter(|event| event.event_type == SessionEventType::LlmCall)
    {
        let Some(task_id) = event
            .payload
            .get("task_id")
            .and_then(serde_json::Value::as_str)
        else {
            continue;
        };
        let task_id = task_id.trim();
        if task_id.is_empty() {
            continue;
        }
        let prompt = event_prompt(event).unwrap_or_default();
        let name = event
            .payload
            .get("task_name")
            .and_then(serde_json::Value::as_str)
            .filter(|name| !name.trim().is_empty())
            .unwrap_or("Untitled task")
            .to_string();
        let task_type = event
            .payload
            .get("task_type")
            .and_then(serde_json::Value::as_str)
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| classify_task_type(if prompt.is_empty() { &name } else { &prompt }));
        let group = event
            .payload
            .get("request_group_id")
            .and_then(serde_json::Value::as_str)
            .filter(|group| !group.trim().is_empty())
            .map(str::to_string);

        let candidate = by_id.entry(task_id.to_string()).or_insert_with(|| {
            order.push(task_id.to_string());
            TaskCandidate {
                task: SessionTask {
                    task_id: task_id.to_string(),
                    name: name.clone(),
                    task_type,
                    first_prompt: prompt.clone(),
                    request_group_ids: Vec::new(),
                    first_event_seq: event.seq,
                    last_event_seq: event.seq,
                    request_count: 0,
                    input_tokens: 0,
                    output_tokens: 0,
                    cost_cents: 0,
                    cost_nano_usd: 0,
                    unknown_cost_calls: 0,
                    completed: false,
                    clean_completion: false,
                    success_status: Default::default(),
                    completed_after_intervention: false,
                    loop_intervention_count: 0,
                    intervention_metrics: Default::default(),
                    completion_quality_bps: None,
                    cost_efficiency_score: None,
                    request_groups: Vec::new(),
                },
                related_texts: Vec::new(),
            }
        });
        candidate.task.last_event_seq = event.seq;
        if !prompt.is_empty() && !candidate.related_texts.contains(&prompt) {
            candidate.related_texts.push(prompt);
        }
        if let Some(response) = event_response_text(event) {
            if !candidate.related_texts.contains(&response) {
                candidate.related_texts.push(response);
            }
        }
        if let Some(group) = group {
            if !candidate.task.request_group_ids.contains(&group) {
                candidate.task.request_group_ids.push(group);
                candidate.task.request_count += 1;
            }
        }
        candidate.task.input_tokens = candidate.task.input_tokens.saturating_add(
            event
                .payload
                .get("input_tokens")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        );
        candidate.task.output_tokens = candidate.task.output_tokens.saturating_add(
            event
                .payload
                .get("output_tokens")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        );
        candidate.task.cost_nano_usd = candidate
            .task
            .cost_nano_usd
            .saturating_add(event.amount_nano_usd().unwrap_or(0));
        candidate.task.cost_cents =
            candidate.task.cost_nano_usd / crate::pricing::NANO_USD_PER_CENT;
        candidate.task.unknown_cost_calls += u64::from(event.amount_nano_usd().is_none());
    }

    order
        .into_iter()
        .filter_map(|id| by_id.remove(&id))
        .collect()
}

fn project_cost_outcomes(events: &[SessionEvent], tasks: &mut [SessionTask]) {
    let task_by_id: std::collections::HashMap<String, usize> = tasks
        .iter()
        .enumerate()
        .map(|(index, task)| (task.task_id.clone(), index))
        .collect();
    let mut group_location = std::collections::HashMap::<String, (usize, usize)>::new();

    for event in events
        .iter()
        .filter(|event| event.event_type == SessionEventType::LlmCall)
    {
        let Some(task_id) = event
            .payload
            .get("task_id")
            .and_then(serde_json::Value::as_str)
        else {
            continue;
        };
        let Some(group_id) = event
            .payload
            .get("request_group_id")
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
        else {
            continue;
        };
        let Some(&task_index) = task_by_id.get(task_id) else {
            continue;
        };
        let group_index = if let Some((known_task, known_group)) = group_location.get(group_id) {
            if *known_task == task_index {
                *known_group
            } else {
                continue;
            }
        } else {
            let group_index = tasks[task_index].request_groups.len();
            tasks[task_index].request_groups.push(RequestGroupOutcome {
                request_group_id: group_id.to_string(),
                task_id: task_id.to_string(),
                first_event_seq: event.seq,
                last_event_seq: event.seq,
                llm_call_count: 0,
                cost_cents: 0,
                cost_nano_usd: 0,
                unknown_cost_calls: 0,
                completed: false,
                clean_completion: false,
                success_status: Default::default(),
                completed_after_intervention: false,
                loop_intervention_count: 0,
                intervention_metrics: Default::default(),
                completion_quality_bps: None,
                cost_efficiency_score: None,
            });
            group_location.insert(group_id.to_string(), (task_index, group_index));
            group_index
        };

        let group = &mut tasks[task_index].request_groups[group_index];
        group.last_event_seq = event.seq;
        group.llm_call_count = group.llm_call_count.saturating_add(1);
        group.cost_nano_usd = group
            .cost_nano_usd
            .saturating_add(event.amount_nano_usd().unwrap_or(0));
        group.cost_cents = group.cost_nano_usd / crate::pricing::NANO_USD_PER_CENT;
        group.unknown_cost_calls += u64::from(event.amount_nano_usd().is_none());
        group.completed = response_finishes_request_group(event);
    }

    // Interventions are emitted after an LLM step is inspected, so the most
    // recently observed request group is the correct owner until another
    // group begins.
    let mut current_group: Option<(usize, usize)> = None;
    for event in events {
        if let Some(explicit_group) = event
            .payload
            .get("request_group_id")
            .and_then(serde_json::Value::as_str)
            .and_then(|group| group_location.get(group).copied())
        {
            current_group = Some(explicit_group);
        }
        if event.event_type != SessionEventType::LlmCall {
            if let Some((task_index, group_index)) = current_group {
                let group = &mut tasks[task_index].request_groups[group_index];
                group.last_event_seq = group.last_event_seq.max(event.seq);
                group.cost_nano_usd = group
                    .cost_nano_usd
                    .saturating_add(event.amount_nano_usd().unwrap_or(0));
                group.cost_cents = group.cost_nano_usd / crate::pricing::NANO_USD_PER_CENT;
                group.unknown_cost_calls += u64::from(event.amount_nano_usd().is_none());
                if is_effective_loop_intervention(event) {
                    group.loop_intervention_count = group.loop_intervention_count.saturating_add(1);
                }
            }
        }
    }

    for task in tasks {
        for group in &mut task.request_groups {
            let mut outcome = CostOutcome::precise(
                group.completed,
                group.cost_nano_usd,
                group.unknown_cost_calls,
                group.loop_intervention_count,
            );
            outcome.intervention_metrics = crate::InterventionMetrics::from_events(
                events_in_seq_range(events, group.first_event_seq, group.last_event_seq),
                group.completed,
                group.cost_nano_usd,
                group.unknown_cost_calls,
            );
            outcome.completed_after_intervention =
                group.completed && outcome.intervention_metrics.actionable_count() > 0;
            outcome.clean_completion =
                group.completed && outcome.intervention_metrics.detections == 0;
            apply_group_outcome(group, outcome);
        }
        task.cost_nano_usd = task
            .request_groups
            .iter()
            .map(|group| group.cost_nano_usd)
            .fold(0_i64, i64::saturating_add);
        task.cost_cents = task.cost_nano_usd / crate::pricing::NANO_USD_PER_CENT;
        task.unknown_cost_calls = task
            .request_groups
            .iter()
            .map(|g| g.unknown_cost_calls)
            .sum();
        let completed = !task.request_groups.is_empty()
            && task.request_groups.iter().all(|group| group.completed);
        let loop_intervention_count = task
            .request_groups
            .iter()
            .map(|group| group.loop_intervention_count)
            .sum();
        let mut outcome = CostOutcome::precise(
            completed,
            task.cost_nano_usd,
            task.unknown_cost_calls,
            loop_intervention_count,
        );
        outcome.intervention_metrics = crate::InterventionMetrics::from_event_iter(
            task.request_groups.iter().flat_map(|group| {
                events_in_seq_range(events, group.first_event_seq, group.last_event_seq).iter()
            }),
            completed,
            task.cost_nano_usd,
            task.unknown_cost_calls,
        );
        outcome.completed_after_intervention =
            completed && outcome.intervention_metrics.actionable_count() > 0;
        outcome.clean_completion = completed && outcome.intervention_metrics.detections == 0;
        task.completed = outcome.completed;
        task.clean_completion = outcome.clean_completion;
        task.success_status = outcome.success_status;
        task.completed_after_intervention = outcome.completed_after_intervention;
        task.loop_intervention_count = outcome.loop_intervention_count;
        task.intervention_metrics = outcome.intervention_metrics;
        task.completion_quality_bps = outcome.completion_quality_bps;
        task.cost_efficiency_score = outcome.cost_efficiency_score;
    }
}

fn events_in_seq_range(events: &[SessionEvent], first_seq: i64, last_seq: i64) -> &[SessionEvent] {
    let start = events.partition_point(|event| event.seq < first_seq);
    let end = events.partition_point(|event| event.seq <= last_seq);
    &events[start.min(end)..end]
}

fn response_finishes_request_group(event: &SessionEvent) -> bool {
    if event
        .payload
        .get("batch_submitted")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return false;
    }
    event
        .payload
        .get("response_message")
        .and_then(|value| serde_json::from_value::<ChatMessage>(value.clone()).ok())
        .is_some_and(|message| {
            message
                .tool_calls
                .as_ref()
                .map_or(true, |tool_calls| tool_calls.is_empty())
        })
}

fn apply_group_outcome(group: &mut RequestGroupOutcome, outcome: CostOutcome) {
    group.completed = outcome.completed;
    group.clean_completion = outcome.clean_completion;
    group.success_status = outcome.success_status;
    group.completed_after_intervention = outcome.completed_after_intervention;
    group.loop_intervention_count = outcome.loop_intervention_count;
    group.intervention_metrics = outcome.intervention_metrics;
    group.completion_quality_bps = outcome.completion_quality_bps;
    group.cost_efficiency_score = outcome.cost_efficiency_score;
}

fn assignment_from_candidate(candidate: &TaskCandidate, is_new: bool) -> TaskAssignment {
    TaskAssignment {
        task_id: candidate.task.task_id.clone(),
        task_name: candidate.task.name.clone(),
        task_type: candidate.task.task_type,
        first_prompt: candidate.task.first_prompt.clone(),
        is_new,
        related_prompts: candidate.related_texts.clone(),
    }
}

fn event_prompt(event: &SessionEvent) -> Option<String> {
    event
        .payload
        .get("request_messages")
        .and_then(|value| serde_json::from_value::<Vec<ChatMessage>>(value.clone()).ok())
        .and_then(|messages| messages.iter().rev().find_map(user_prompt))
}

fn event_response_text(event: &SessionEvent) -> Option<String> {
    let message = event
        .payload
        .get("response_message")
        .and_then(|value| serde_json::from_value::<ChatMessage>(value.clone()).ok())?;
    let text = message.full_text();
    if text.trim().is_empty() {
        return None;
    }
    Some(text.chars().take(MAX_RELATED_RESPONSE_CHARS).collect())
}

fn latest_user_prompt(request: &ChatCompletionRequest) -> Option<String> {
    request.messages.iter().rev().find_map(user_prompt)
}

fn user_prompt(message: &ChatMessage) -> Option<String> {
    if !message.role.eq_ignore_ascii_case("user") || message.tool_call_id.is_some() {
        return None;
    }
    if let Some(text) = message.content_str() {
        return crate::user_content::session_user_request(text).map(str::to_string);
    }
    let text = message.full_text();
    crate::user_content::session_user_request(&text)
        .filter(|prompt| !prompt.trim().is_empty())
        .map(str::to_string)
}

fn name_from_prompt(request: &ChatCompletionRequest, prompt: &str) -> String {
    let mut first = request.clone();
    first.messages = request
        .messages
        .iter()
        .rev()
        .find(|message| user_prompt(message).as_deref() == Some(prompt))
        .cloned()
        .into_iter()
        .collect();
    derive_task_summary(&first)
}

fn explicit_task_type(request: &ChatCompletionRequest) -> Option<TaskType> {
    request
        .extra
        .get(TASK_TYPE_EXTRA_KEY)
        .and_then(serde_json::Value::as_str)
        .and_then(|value| value.parse().ok())
}

fn deterministic_task_id(
    session_id: Option<Uuid>,
    first_prompt: &str,
    declared_request_group: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    if let Some(session_id) = session_id {
        hasher.update(session_id.as_bytes());
    }
    hasher.update([0x1f]);
    hasher.update(first_prompt.trim().to_lowercase().as_bytes());
    // A repeated explicit declaration means fresh work. Correlate retries
    // by request group, but never reuse a previous task's continuation simply
    // because the user pasted the same specification again.
    if let Some(group) = declared_request_group {
        hasher.update([0x1f]);
        hasher.update(group.as_bytes());
    }
    let digest = hex::encode(hasher.finalize());
    format!("task_{}", &digest[..24])
}

/// The candidate the session most recently worked on. Candidate order is
/// creation order, which interleaved work makes stale; activity wins.
fn most_recent_candidate(candidates: &[TaskCandidate]) -> Option<&TaskCandidate> {
    candidates
        .iter()
        .max_by_key(|candidate| candidate.task.last_event_seq)
}

fn is_follow_up(prompt: &str) -> bool {
    if crate::lexical::starts_new_task(prompt) {
        return false;
    }
    // Strong signals (phrases, labeled answers, underspecified references)
    // always continue prior work.
    if crate::lexical::refers_to_prior_context_strong(prompt) {
        return true;
    }
    // A discourse-led imperative with a concrete topic ("Now schedule
    // candidate interviews", "Retry the payment migration") is self-contained
    // new work, not a follow-up; relatedness may still attach it below.
    // Questions about prior state are never self-contained work requests.
    let words = crate::lexical::words(prompt);
    let self_contained = words.len() >= 4
        && !crate::lexical::is_question(prompt)
        && !crate::lexical::relevance_terms(prompt).is_empty()
        && (crate::lexical::contains_action_verb(prompt)
            || crate::lexical::starts_with_prior_context_word(prompt));
    if self_contained {
        return false;
    }
    if crate::lexical::refers_to_prior_context(prompt) {
        return true;
    }
    words.len() < 5 && !crate::lexical::contains_action_verb(prompt)
}

fn relatedness(left: &str, right: &str) -> f32 {
    let left = crate::lexical::relevance_terms(left);
    let right = crate::lexical::relevance_terms(right);
    relatedness_terms(&left, &right)
}

fn relatedness_terms(
    left: &std::collections::HashSet<String>,
    right: &std::collections::HashSet<String>,
) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let overlap = left.intersection(&right).count();
    if overlap == 0 {
        return 0.0;
    }
    // One shared generic noun ("billing") must not merge distinct work:
    // larger vocabularies need at least two overlapping terms.
    if overlap < 2 && left.len().min(right.len()) >= 3 {
        return 0.0;
    }
    overlap as f32 / left.len().min(right.len()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::MessageContent;
    use chrono::Utc;
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

    fn request(prompts: &[&str]) -> ChatCompletionRequest {
        let mut messages = vec![message("system", "You are helpful")];
        for prompt in prompts {
            messages.push(message("user", prompt));
            messages.push(message("assistant", "done"));
        }
        ChatCompletionRequest {
            model: "model".to_string(),
            messages,
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: HashMap::new(),
        }
    }

    fn event(seq: i64, task_id: &str, task_name: &str, group: &str, prompt: &str) -> SessionEvent {
        SessionEvent {
            session_id: Uuid::nil(),
            seq,
            event_type: SessionEventType::LlmCall,
            payload: serde_json::json!({
                "request_messages": [message("user", prompt)],
                "response_message": message("assistant", "done"),
                "task_id": task_id,
                "task_name": task_name,
                "request_group_id": group,
            }),
            cost_cents: None,
            created_at: Utc::now(),
        }
    }

    fn event_with_usage(
        seq: i64,
        task_id: &str,
        task_name: &str,
        group: &str,
        prompt: &str,
        input_tokens: u64,
        output_tokens: u64,
        cost_cents: i64,
    ) -> SessionEvent {
        let mut event = event(seq, task_id, task_name, group, prompt);
        event.payload["input_tokens"] = serde_json::json!(input_tokens);
        event.payload["output_tokens"] = serde_json::json!(output_tokens);
        event.cost_cents = Some(cost_cents);
        event
    }

    #[test]
    fn timeline_projects_request_metadata_without_raw_payloads() {
        let mut llm = event_with_usage(
            1,
            "task-1",
            "Implement the timeline",
            "group-1",
            "Please implement the timeline",
            120,
            30,
            4,
        );
        llm.payload["model"] = serde_json::json!("test-model");
        let tool_call = SessionEvent {
            session_id: Uuid::nil(),
            seq: 2,
            event_type: SessionEventType::ToolCall,
            payload: serde_json::json!({
                "tool_call": {"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
            }),
            cost_cents: None,
            created_at: Utc::now(),
        };
        let tool_result = SessionEvent {
            session_id: Uuid::nil(),
            seq: 3,
            event_type: SessionEventType::ToolResult,
            payload: serde_json::json!({
                "result": {"tool_call_id": "call-1", "name": "read_file", "content": "ok", "is_error": false}
            }),
            cost_cents: None,
            created_at: Utc::now(),
        };

        let timeline = session_timeline(&[llm, tool_call, tool_result]);

        assert_eq!(timeline.total_event_count, 3);
        assert_eq!(timeline.latest_seq, Some(3));
        assert_eq!(timeline.tasks.len(), 1);
        assert_eq!(timeline.request_groups.len(), 1);
        let request = &timeline.request_groups[0];
        assert_eq!(
            request.prompt.as_deref(),
            Some("Please implement the timeline")
        );
        assert_eq!(request.event_count, 3);
        assert_eq!(request.tool_call_count, 1);
        assert_eq!(request.input_tokens, 120);
        assert_eq!(request.output_tokens, 30);
        assert_eq!(request.cost_cents, 4);
        assert_eq!(request.models, vec!["test-model"]);
        assert_eq!(request.pii_redaction_count, 0);
        assert_eq!(timeline.pii_redaction_count, 0);
    }

    #[test]
    fn timeline_ignores_durable_continuation_snapshots() {
        let llm = event(1, "task-1", "Do the work", "group-1", "Do the work");
        // Settlement rewrites a continuation snapshot after nearly every tool
        // call; none of them may surface as timeline activity.
        let continuations = (2..=6).map(|seq| SessionEvent {
            session_id: Uuid::nil(),
            seq,
            event_type: SessionEventType::Continuation,
            payload: serde_json::json!({
                "task_id": "task-1",
                "request_group_id": "group-1",
            }),
            cost_cents: None,
            created_at: Utc::now(),
        });
        let mut events = vec![llm];
        events.extend(continuations);

        let timeline = session_timeline(&events);

        assert_eq!(timeline.total_event_count, 1);
        assert_eq!(timeline.latest_seq, Some(1));
        assert_eq!(timeline.request_groups.len(), 1);
        assert_eq!(timeline.request_groups[0].event_count, 1);
    }

    #[test]
    fn timeline_surfaces_pii_redaction_activity_per_group() {
        let mut first = event_with_usage(
            1,
            "task-1",
            "Handle contacts",
            "group-1",
            "Email alice@example.com",
            10,
            5,
            1,
        );
        first.payload["_reincore_privacy_redaction"] =
            serde_json::json!({"redaction_count": 2, "pii_redaction_count": 1, "representation": "sanitized"});
        let mut second = event_with_usage(
            2,
            "task-1",
            "Handle contacts",
            "group-1",
            "Call (555) 123-4567",
            10,
            5,
            1,
        );
        second.payload["_reincore_privacy_redaction"] =
            serde_json::json!({"redaction_count": 3, "pii_redaction_count": 2, "representation": "sanitized"});
        let plain = event_with_usage(3, "task-1", "Handle contacts", "group-2", "Hello", 10, 5, 1);

        let timeline = session_timeline(&[first, second, plain]);

        assert_eq!(timeline.request_groups.len(), 2);
        assert_eq!(timeline.request_groups[0].pii_redaction_count, 3);
        assert_eq!(timeline.request_groups[1].pii_redaction_count, 0);
        assert_eq!(timeline.pii_redaction_count, 3);
    }

    #[test]
    fn groups_request_groups_under_distinct_tasks() {
        let events = vec![
            event(
                1,
                "auth",
                "Implement OAuth login",
                "g1",
                "Implement OAuth login",
            ),
            event(
                2,
                "billing",
                "Build billing dashboard",
                "g2",
                "Build billing dashboard",
            ),
            event(
                3,
                "auth",
                "Implement OAuth login",
                "g3",
                "Fix the OAuth callback",
            ),
        ];
        let tasks = session_tasks(&events);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].request_group_ids, vec!["g1", "g3"]);
        assert_eq!(tasks[0].request_count, 2);
        assert_eq!(tasks[0].task_type, TaskType::Feature);
        assert_eq!(tasks[1].request_group_ids, vec!["g2"]);
    }

    #[test]
    fn task_metrics_link_a_consumed_nudge_across_noncontiguous_request_groups() {
        let finding = crate::AgentHealthFinding {
            kind: crate::AgentHealthKind::ToolLoop,
            severity: crate::AgentHealthSeverity::Warning,
            message: "repeating".into(),
            tool_name: Some("read".into()),
        };
        let record = crate::InterventionRecord::from_finding(
            &finding,
            &crate::InterventionPolicy::default(),
            crate::InterventionAction::RepairRequested,
        );
        let intervention = SessionEvent {
            session_id: Uuid::nil(),
            seq: 2,
            event_type: SessionEventType::LoopIntervention,
            payload: serde_json::json!({
                "request_group_id": "g1",
                "repetition_count": 3,
                "injected_message": "try a different source",
                "intervention": record,
            }),
            cost_cents: None,
            created_at: Utc::now(),
        };
        let events = vec![
            event(
                1,
                "auth",
                "Implement OAuth login",
                "g1",
                "Implement OAuth login",
            ),
            intervention,
            event(
                3,
                "billing",
                "Build billing dashboard",
                "g2",
                "Build billing dashboard",
            ),
            event(
                4,
                "auth",
                "Implement OAuth login",
                "g3",
                "Continue OAuth login",
            ),
        ];

        let tasks = session_tasks(&events);
        let authentication = tasks.iter().find(|task| task.task_id == "auth").unwrap();
        let billing = tasks.iter().find(|task| task.task_id == "billing").unwrap();
        assert_eq!(authentication.intervention_metrics.detections, 1);
        assert_eq!(authentication.intervention_metrics.successful_recoveries, 1);
        assert!(authentication.intervention_metrics.added_input_tokens > 0);
        assert_eq!(
            authentication.request_groups[0]
                .intervention_metrics
                .successful_recoveries,
            0
        );
        assert_eq!(billing.intervention_metrics.detections, 0);
    }

    #[test]
    fn aggregates_tokens_and_cost_for_each_task() {
        let events = vec![
            event_with_usage(
                1,
                "auth",
                "Implement OAuth login",
                "g1",
                "Implement OAuth login",
                100,
                20,
                3,
            ),
            event_with_usage(
                2,
                "auth",
                "Implement OAuth login",
                "g2",
                "Update the login flow",
                80,
                15,
                2,
            ),
            event_with_usage(
                3,
                "research",
                "Research competitors",
                "g3",
                "Research competitors",
                200,
                40,
                7,
            ),
        ];

        let tasks = session_tasks(&events);
        let authentication = tasks
            .iter()
            .find(|task| task.task_id == "auth")
            .expect("authentication task");
        let research = tasks
            .iter()
            .find(|task| task.task_id == "research")
            .expect("research task");

        assert_eq!(authentication.input_tokens, 180);
        assert_eq!(authentication.output_tokens, 35);
        assert_eq!(authentication.cost_cents, 5);
        assert_eq!(research.input_tokens, 200);
        assert_eq!(research.output_tokens, 40);
        assert_eq!(research.cost_cents, 7);
    }

    #[test]
    fn projects_cost_outcome_for_request_group_and_task() {
        let mut first = event_with_usage(
            1,
            "auth",
            "Implement OAuth login",
            "g1",
            "Implement OAuth login",
            100,
            20,
            200,
        );
        first.payload["response_message"]["tool_calls"] = serde_json::json!([{
            "id": "call_1",
            "type": "function",
            "function": { "name": "edit", "arguments": "{}" }
        }]);
        let mut tool_cost = event(2, "auth", "Implement OAuth login", "g1", "tool");
        tool_cost.event_type = SessionEventType::ToolResult;
        tool_cost.payload = serde_json::json!({ "request_group_id": "g1" });
        tool_cost.cost_cents = Some(50);
        let mut intervention = event(3, "auth", "Implement OAuth login", "g1", "retry");
        intervention.event_type = SessionEventType::LoopIntervention;
        intervention.payload = serde_json::json!({
            "repetition_count": 3,
            "injected_message": "try a different approach"
        });
        intervention.cost_cents = None;
        let final_call = event_with_usage(
            4,
            "auth",
            "Implement OAuth login",
            "g1",
            "Implement OAuth login",
            50,
            10,
            100,
        );

        let tasks = session_tasks(&[first, tool_cost, intervention, final_call]);
        assert_eq!(tasks.len(), 1);
        assert!(tasks[0].completed);
        assert!(!tasks[0].clean_completion);
        assert_eq!(tasks[0].loop_intervention_count, 1);
        assert_eq!(tasks[0].completion_quality_bps, None);
        assert_eq!(tasks[0].request_groups.len(), 1);
        assert_eq!(tasks[0].cost_cents, 350);
        assert_eq!(tasks[0].request_groups[0].cost_cents, 350);
        assert_eq!(tasks[0].request_groups[0].cost_efficiency_score, None);
    }

    #[test]
    fn related_prompt_returns_to_an_older_task() {
        let events = vec![
            event(
                1,
                "auth",
                "Implement OAuth login",
                "g1",
                "Implement OAuth login",
            ),
            event(
                2,
                "billing",
                "Build billing dashboard",
                "g2",
                "Build billing dashboard",
            ),
        ];
        let assignment = resolve_task(
            Some(Uuid::nil()),
            "g3",
            &request(&["Fix the OAuth callback handling"]),
            &events,
        );
        assert_eq!(assignment.task_id, "auth");
        assert!(!assignment.is_new);
    }

    #[test]
    fn related_sales_prompt_returns_to_an_older_task() {
        let events = vec![
            event(
                1,
                "renewal",
                "Draft Acme renewal proposal",
                "g1",
                "Draft a renewal proposal for Acme procurement",
            ),
            event(
                2,
                "interviews",
                "Schedule regional interviews",
                "g2",
                "Schedule interviews for regional sales candidates",
            ),
        ];
        let assignment = resolve_task(
            Some(Uuid::nil()),
            "g3",
            &request(&["Review the Acme procurement renewal terms"]),
            &events,
        );
        assert_eq!(assignment.task_id, "renewal");
        assert!(!assignment.is_new);
    }

    #[test]
    fn labeled_option_answer_keeps_the_original_task_across_request_groups() {
        let events = vec![event(
            1,
            "portfolio",
            "Implement Portfolio CRUD API Endpoints",
            "original-request",
            "Implement portfolio CRUD API endpoints",
        )];
        let request = request(&["Option 3 - Change docker postgres port mapping"]);
        let task = resolve_task(Some(Uuid::new_v4()), "after-reconnect", &request, &events);
        assert_eq!(task.task_id, "portfolio");
        assert_eq!(task.task_name, "Implement Portfolio CRUD API Endpoints");
        assert!(!task.is_new);
    }

    #[test]
    fn short_follow_up_stays_on_the_latest_task() {
        let events = vec![event(
            1,
            "billing",
            "Build billing dashboard",
            "g1",
            "Build billing dashboard",
        )];
        let assignment = resolve_task(None, "g2", &request(&["Now add tests"]), &events);
        assert_eq!(assignment.task_id, "billing");
    }

    #[test]
    fn attribution_discussion_stays_in_one_task() {
        let task_id = "agent-attribution";
        let task_name = "Identify the requesting agent or tool";
        let mut first = event(
            1,
            task_id,
            task_name,
            "g1",
            "Perfect! Is there a way to also get the name of the agent or tool where the user made the request from?",
        );
        first.payload["response_message"] = serde_json::to_value(message(
            "assistant",
            "The delegation profile identifies the agent or tool. The VS Code extension records GitHub Copilot and the ReinCore client.",
        ))
        .unwrap();

        let second_prompt =
            "Will this value be manually set by the user on the agent side or will ReinCore capture it automatically?";
        let second_assignment = resolve_task(
            Some(Uuid::nil()),
            "g2",
            &request(&[second_prompt]),
            &[first.clone()],
        );
        assert_eq!(second_assignment.task_id, task_id);

        let mut second = event(2, task_id, task_name, "g2", second_prompt);
        second.payload["response_message"] = serde_json::to_value(message(
            "assistant",
            "ReinCore captures the client automatically. VS Code sends client metadata without manual user configuration.",
        ))
        .unwrap();
        let third_prompt =
            "What will happen if the user uses the ReinCore SDK in their project for an agent workflow?";
        let third_assignment = resolve_task(
            Some(Uuid::nil()),
            "g3",
            &request(&[third_prompt]),
            &[first.clone(), second.clone()],
        );
        assert_eq!(third_assignment.task_id, task_id);

        let mut third = event(3, task_id, task_name, "g3", third_prompt);
        third.payload["response_message"] = serde_json::to_value(message(
            "assistant",
            "SDK workflows can send a delegation profile for their framework or custom agent.",
        ))
        .unwrap();
        let fourth_prompt = "Doesn't VS Code use GitHub Copilot as the agent?";
        let fourth_assignment = resolve_task(
            Some(Uuid::nil()),
            "g4",
            &request(&[fourth_prompt]),
            &[first.clone(), second.clone(), third.clone()],
        );
        assert_eq!(fourth_assignment.task_id, task_id);

        let mut fourth = event(4, task_id, task_name, "g4", fourth_prompt);
        fourth.payload["response_message"] = serde_json::to_value(message(
            "assistant",
            "Yes, GitHub Copilot is the agent inside VS Code.",
        ))
        .unwrap();
        let final_assignment = resolve_task(
            Some(Uuid::nil()),
            "g5",
            &request(&["Please implement the feature."]),
            &[first, second, third, fourth],
        );
        assert_eq!(final_assignment.task_id, task_id);
        assert!(!final_assignment.is_new);
    }

    #[test]
    fn contrastive_revision_stays_on_the_latest_task() {
        let initial_prompt = "In my project, currently the SessionTimeline are taking quite a long time to load and it is a very long list as it lists all requests and their events. What is the best way to optimize that page?";
        let events = vec![event(
            1,
            "timeline",
            "Optimize SessionTimeline loading",
            "g1",
            initial_prompt,
        )];
        let assignment = resolve_task(
            Some(Uuid::nil()),
            "g2",
            &request(&[
                "But instead of having Collapse request outside the request section, please move it inside",
            ]),
            &events,
        );

        assert_eq!(assignment.task_id, "timeline");
        assert!(!assignment.is_new);
    }

    #[test]
    fn short_non_coding_action_starts_a_new_task() {
        let events = vec![event(
            1,
            "ledger",
            "Reconcile July ledger",
            "g1",
            "Reconcile the July ledger",
        )];
        let assignment = resolve_task(
            Some(Uuid::nil()),
            "g2",
            &request(&["Schedule candidate interviews"]),
            &events,
        );
        assert!(assignment.is_new);
        assert_ne!(assignment.task_id, "ledger");
    }

    #[test]
    fn terminal_completion_tool_data_cannot_select_an_unrelated_task() {
        let events = vec![
            event(
                1,
                "grouping",
                "Review request grouping",
                "origin-group",
                "Review the request grouping bug",
            ),
            event(
                2,
                "cloud-server",
                "Run cloud engine server",
                "other-group",
                "Run the cloud engine server",
            ),
        ];
        let request = ChatCompletionRequest {
            model: "model".to_string(),
            messages: vec![
                message("user", "Review the request grouping bug"),
                message(
                    "assistant",
                    "I will start the cloud engine to reproduce it.",
                ),
                {
                    let mut call = message("assistant", "");
                    call.tool_calls = Some(vec![serde_json::from_value(serde_json::json!({
                        "id": "reincore-terminal-notification",
                        "type": "function",
                        "function": {
                            "name": "reincore_terminal_notification",
                            "arguments": "{}"
                        }
                    }))
                    .unwrap()]);
                    call
                },
                {
                    let mut result = message("tool", "cargo run failed: Access is denied");
                    result.tool_call_id = Some("reincore-terminal-notification".into());
                    result
                },
            ],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: HashMap::new(),
        };

        let assignment = resolve_task(Some(Uuid::nil()), "origin-group", &request, &events);
        assert_eq!(assignment.task_id, "grouping");
        assert_eq!(assignment.first_prompt, "Review the request grouping bug");
        assert!(!assignment.is_new);
    }

    #[test]
    fn new_task_name_comes_from_its_first_prompt() {
        let req = request(&["Please implement session task tracking with request groups"]);
        let assignment = resolve_task(Some(Uuid::nil()), "g1", &req, &[]);
        assert!(assignment.is_new);
        assert_eq!(
            assignment.task_name,
            "Implement session task tracking with request groups"
        );
    }

    #[test]
    fn task_name_prefers_the_requested_outcome_over_problem_context() {
        let req = request(&[
            "In my project, currently the SessionTimeline are taking quite a long time to load and it is a very long list as it lists all requests and their events. What is the best way to optimize that page?",
        ]);

        let assignment = resolve_task(Some(Uuid::nil()), "g1", &req, &[]);

        assert_eq!(assignment.task_name, "Task: Optimize SessionTimeline page");
    }

    #[test]
    fn non_coding_task_name_comes_from_its_first_prompt() {
        let req = request(&["Please draft a renewal proposal for Acme procurement"]);
        let assignment = resolve_task(Some(Uuid::nil()), "g1", &req, &[]);
        assert!(assignment.is_new);
        assert_eq!(
            assignment.task_name,
            "Draft a renewal proposal for Acme procurement"
        );
        assert_eq!(assignment.task_type, TaskType::ContentCreation);
    }

    #[tokio::test]
    async fn resolve_task_auto_without_judge_matches_lexical_resolution() {
        use crate::detection::DetectionMode::Llm;
        let req = request(&["Please draft a renewal proposal for Acme procurement"]);
        let lexical = resolve_task(Some(Uuid::nil()), "g1", &req, &[]);
        // `llm` mode with no judge available degrades to the lexical path,
        // so the assignment is identical to the default resolution.
        let auto = resolve_task_auto(Some(Uuid::nil()), "g1", &req, &[], Llm, Llm, None).await;
        assert_eq!(auto, lexical);
        assert!(auto.is_new);
    }

    #[test]
    fn discourse_openers_do_not_merge_self_contained_new_work() {
        let events = vec![event(
            1,
            "events-task",
            "Implement portfolio events",
            "g1",
            "Implement portfolio events",
        )];
        for prompt in [
            "Now schedule candidate interviews",
            "Next, analyze Q3 churn",
            "Retry the payment migration",
        ] {
            let req = request(&["Implement portfolio events", prompt]);
            let assignment = resolve_task(Some(Uuid::nil()), "g2", &req, &events);
            assert!(assignment.is_new, "prompt: {prompt}");
        }
        // "OK, next feature:" declarations still start new tasks.
        let req = request(&[
            "Implement portfolio events",
            "OK, next feature: add CSV export",
        ]);
        let assignment = resolve_task(Some(Uuid::nil()), "g2", &req, &events);
        assert!(assignment.is_new);
    }

    #[test]
    fn chatter_rides_along_with_the_most_recent_task() {
        let events = vec![
            event(1, "auth-task", "Fix auth timeout", "g1", "Fix auth timeout"),
            event(
                2,
                "billing-task",
                "Build billing dashboard",
                "g2",
                "Build billing dashboard",
            ),
            event(3, "auth-task", "Fix auth timeout", "g3", "Fix auth timeout"),
        ];
        let req = request(&["Fix auth timeout", "Thanks!"]);
        let assignment = resolve_task(Some(Uuid::nil()), "g4", &req, &events);
        assert!(!assignment.is_new);
        assert_eq!(assignment.task_id, "auth-task");
    }

    #[test]
    fn follow_up_attaches_to_most_recently_active_task_not_latest_created() {
        let events = vec![
            event(1, "auth-task", "Fix auth timeout", "g1", "Fix auth timeout"),
            event(
                2,
                "billing-task",
                "Build billing dashboard",
                "g2",
                "Build billing dashboard",
            ),
            event(3, "auth-task", "Fix auth timeout", "g3", "Fix auth timeout"),
        ];
        let req = request(&["Fix auth timeout", "that one please"]);
        let assignment = resolve_task(Some(Uuid::nil()), "g4", &req, &events);
        assert!(!assignment.is_new);
        assert_eq!(assignment.task_id, "auth-task");
    }

    #[test]
    fn explicit_task_type_override_is_honored_for_a_new_task() {
        let mut req = request(&["Draft a renewal proposal for Acme procurement"]);
        req.extra.insert(
            TASK_TYPE_EXTRA_KEY.to_string(),
            serde_json::json!("planning"),
        );
        let assignment = resolve_task(Some(Uuid::nil()), "g1", &req, &[]);
        assert_eq!(assignment.task_type, TaskType::Planning);
    }

    #[test]
    fn persisted_task_type_wins_and_legacy_events_are_inferred() {
        let mut explicit = event(
            1,
            "policy",
            "Review procurement policy",
            "g1",
            "Review the procurement policy",
        );
        explicit.payload["task_type"] = serde_json::json!("operations");
        let legacy = event(
            2,
            "research",
            "Research regional suppliers",
            "g2",
            "Research regional suppliers",
        );

        let tasks = session_tasks(&[explicit, legacy]);
        assert_eq!(tasks[0].task_type, TaskType::Operations);
        assert_eq!(tasks[1].task_type, TaskType::Research);
    }

    #[test]
    fn scopes_context_to_the_selected_task() {
        let events = vec![event(
            1,
            "auth",
            "Implement OAuth login",
            "g1",
            "Implement OAuth login",
        )];
        let mut req = request(&[
            "Implement OAuth login",
            "Build billing dashboard",
            "Fix the OAuth callback handling",
        ]);
        let assignment = resolve_task(Some(Uuid::nil()), "g3", &req, &events);
        let removed = scope_request_to_task(&mut req, &assignment);
        let text = req
            .messages
            .iter()
            .filter_map(ChatMessage::content_str)
            .collect::<Vec<_>>()
            .join(" ");
        assert!(removed >= 2);
        assert!(text.contains("Implement OAuth login"));
        assert!(text.contains("Fix the OAuth callback"));
        assert!(!text.contains("Build billing dashboard"));
    }

    #[test]
    fn referential_follow_up_keeps_recent_antecedent_without_task_events() {
        let mut req = ChatCompletionRequest {
            model: "model".to_string(),
            messages: vec![
                message("system", "You are a coding agent"),
                message(
                    "user",
                    "I get an Ollama error when sending an image in the request",
                ),
                message("assistant", "Fixed Ollama multipart image serialization."),
                message("user", "Could you check the rest of the providers?"),
                message("assistant", "The other provider formats are correct."),
                message("user", "Yes, silently drop unsupported file inputs."),
                message("assistant", "Unsupported Ollama files are now dropped."),
                message(
                    "user",
                    "What happens when an Ollama model does not accept images?",
                ),
                message(
                    "assistant",
                    "Option C is best: query /api/show, cache vision support, and reject unsupported image input clearly.",
                ),
                message("user", "Please implement the feature"),
            ],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: HashMap::new(),
        };

        // This is the first request observed for the resolved session, so no
        // persisted task projections are available even though the client sent
        // the preceding conversation in full.
        let assignment = resolve_task(Some(Uuid::nil()), "g5", &req, &[]);
        assert!(assignment.is_new);

        let removed = scope_request_to_task(&mut req, &assignment);
        assert_eq!(
            removed, 2,
            "only the oldest exchange should be outside the tail"
        );
        assert!(req.messages.iter().any(|message| {
            message
                .content_str()
                .is_some_and(|content| content.contains("Option C is best"))
        }));

        // Task scoping feeds directly into configured compression. The
        // antecedent must survive every configured compression level.
        for mode in [
            crate::CompressionMode::Low,
            crate::CompressionMode::Mid,
            crate::CompressionMode::Max,
        ] {
            let mut compressed = req.clone();
            crate::prepare_context(
                &mut compressed,
                crate::CompressionSettings::new(mode),
                crate::BudgetTier::Tier1,
            );
            assert!(
                compressed.messages.iter().any(|message| {
                    message
                        .content_str()
                        .is_some_and(|content| content.contains("Option C is best"))
                }),
                "{mode:?} compression dropped the follow-up antecedent"
            );
            assert_eq!(
                compressed
                    .messages
                    .last()
                    .and_then(ChatMessage::content_str),
                Some("Please implement the feature")
            );
        }
    }

    #[test]
    fn self_contained_relative_clause_does_not_continue_latest_task() {
        let events = vec![event(
            1,
            "auth",
            "Implement OAuth login",
            "g1",
            "Implement OAuth login",
        )];
        let assignment = resolve_task(
            None,
            "g2",
            &request(&["Build a dashboard that shows billing totals"]),
            &events,
        );
        assert!(assignment.is_new);
        assert_ne!(assignment.task_id, "auth");
    }
}

#[cfg(test)]
mod task_boundaries_tests {
    use crate::{
        event::{SessionEvent, SessionEventType},
        models::ChatCompletionRequest,
        task::{resolve_task, scope_request_to_task, session_tasks},
    };
    use serde_json::json;
    use uuid::Uuid;

    const PIP: &str = "The python path it is using is at C:\\Python314\\python.exe and not the one in AppData. Plus, I am using python -m pip script";
    const TRANSACTIONS: &str = "Next feature: ### Implement Transaction Recording and History Priority: High Story Points: 13 Team: Backend Description: Build transaction tracking system recording all buy/sell operations with automatic holding updates and cost basis calculations.";
    const EVENTS: &str = "Next feature: ### Implement Portfolio Event Sourcing and History Reconstruction Priority: High Story Points: 12 Team: Backend Description: Build event sourcing system storing all portfolio state changes in portfolio_events table. Implement state reconstruction from events and provide audit trail API for compliance.";

    fn request(prompts: &[&str]) -> ChatCompletionRequest {
        let mut messages = vec![json!({"role":"system","content":"You are a coding agent"})];
        for prompt in prompts {
            messages.push(json!({"role":"user","content":prompt}));
            messages.push(json!({"role":"assistant","content":"Completed the requested work."}));
        }
        messages.pop();
        serde_json::from_value(json!({"model":"test-model","messages":messages})).unwrap()
    }

    fn event(seq: i64, task: &str, group: &str, prompt: &str) -> SessionEvent {
        SessionEvent {
            session_id: Uuid::nil(),
            seq,
            event_type: SessionEventType::LlmCall,
            payload: json!({"task_id":task, "task_name":prompt, "request_group_id":group,
            "request_messages":[{"role":"user","content":prompt}],
            "response_message":{"role":"assistant","content":"Completed the requested work."}}),
            cost_cents: None,
            created_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn next_features_start_distinct_tasks_and_drop_completed_pip_context() {
        let mut history = vec![event(1, "pip-task", "pip-request", PIP)];
        assert!(session_tasks(&history)[0].completed);
        let mut req = request(&[PIP, TRANSACTIONS]);
        let transactions = resolve_task(Some(Uuid::nil()), "transactions-request", &req, &history);
        assert!(transactions.is_new);
        assert_ne!(transactions.task_id, "pip-task");
        assert_eq!(transactions.first_prompt, TRANSACTIONS);
        assert_eq!(scope_request_to_task(&mut req, &transactions), 2);
        assert_eq!(req.messages[0].role, "system");
        assert_eq!(req.messages[1].content_str(), Some(TRANSACTIONS));

        history.push(event(
            2,
            &transactions.task_id,
            "transactions-request",
            TRANSACTIONS,
        ));
        let mut req = request(&[PIP, TRANSACTIONS, EVENTS]);
        let events = resolve_task(Some(Uuid::nil()), "events-request", &req, &history);
        assert!(events.is_new);
        assert_ne!(events.task_id, transactions.task_id);
        assert_eq!(scope_request_to_task(&mut req, &events), 4);
        assert_eq!(req.messages[1].content_str(), Some(EVENTS));
    }

    #[test]
    fn a_new_task_boundary_survives_tool_loops_and_referential_follow_ups() {
        let history = vec![
            event(1, "transactions", "transactions-request", TRANSACTIONS),
            event(2, "events", "events-request", EVENTS),
        ];
        let mut req = request(&[TRANSACTIONS, EVENTS]);
        req.messages.extend(serde_json::from_value::<Vec<crate::ChatMessage>>(json!([
        {"role":"assistant","tool_calls":[{"id":"edit-1","type":"function","function":{"name":"edit_file","arguments":"{}"}}]},
        {"role":"tool","tool_call_id":"edit-1","content":"Edited portfolio_events.rs"}
    ])).unwrap());
        let task = resolve_task(Some(Uuid::nil()), "events-request", &req, &history);
        assert_eq!(task.task_id, "events");
        assert!(!task.is_new);
        scope_request_to_task(&mut req, &task);
        assert_eq!(req.messages.len(), 4);
        assert_eq!(
            req.messages.last().unwrap().tool_call_id.as_deref(),
            Some("edit-1")
        );

        let mut follow_up = request(&[TRANSACTIONS, EVENTS, "Please proceed"]);
        let task = resolve_task(Some(Uuid::nil()), "follow-up", &follow_up, &history);
        assert_eq!(task.task_id, "events");
        scope_request_to_task(&mut follow_up, &task);
        assert!(!follow_up
            .messages
            .iter()
            .any(|m| m.content_str() == Some(TRANSACTIONS)));
        assert!(follow_up
            .messages
            .iter()
            .any(|m| m.content_str() == Some(EVENTS)));
    }

    #[test]
    fn explicit_task_and_request_group_continuity_remain_authoritative() {
        let history = vec![event(1, "pip-task", "pip-request", PIP)];
        let mut req = request(&[EVENTS]);
        req.extra
            .insert("reincore_task_id".into(), json!("pip-task"));
        assert_eq!(
            resolve_task(None, "events-request", &req, &history).task_id,
            "pip-task"
        );
        req.extra.clear();
        assert_eq!(
            resolve_task(None, "pip-request", &req, &history).task_id,
            "pip-task"
        );
    }

    #[test]
    fn repeated_new_task_declarations_get_new_ids_but_retries_keep_their_id() {
        let old_req = request(&[EVENTS]);
        let old = resolve_task(Some(Uuid::nil()), "first-request", &old_req, &[]);
        let history = vec![event(1, &old.task_id, "first-request", EVENTS)];
        let mut req = request(&[EVENTS, EVENTS]);
        req.messages[2].content = Some(crate::MessageContent::text("Old task completed"));
        let new = resolve_task(Some(Uuid::nil()), "second-request", &req, &history);
        assert_ne!(
        new.task_id, old.task_id,
        "an explicit fresh task must not load the prior continuation just because the text repeats"
    );
        assert_eq!(
            resolve_task(Some(Uuid::nil()), "second-request", &req, &history).task_id,
            new.task_id
        );
        scope_request_to_task(&mut req, &new);
        assert_eq!(
            req.messages.len(),
            2,
            "use the most recent occurrence of a repeated declaration"
        );
        let mut history = history;
        history.push(event(2, &new.task_id, "second-request", EVENTS));
        assert_eq!(
            resolve_task(Some(Uuid::nil()), "second-request", &req, &history).task_id,
            new.task_id
        );
    }

    #[test]
    fn task_changes_are_detected_in_wrapped_and_multipart_user_requests() {
        let history = vec![event(1, "pip-task", "pip-request", PIP)];
        let wrapped =
            format!("<context>Previous pip task</context><userRequest>{EVENTS}</userRequest>");
        let mut req = request(&[&wrapped]);
        req.messages.last_mut().unwrap().content = Some(
            serde_json::from_value(json!([
                {"type":"text","text":wrapped},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}
            ]))
            .unwrap(),
        );
        let task = resolve_task(Some(Uuid::nil()), "events-request", &req, &history);
        assert!(task.is_new);
        assert_eq!(task.first_prompt, EVENTS);
    }

    #[test]
    fn new_task_drops_external_summaries_and_preserves_editor_context_and_attachments() {
        let mut req: ChatCompletionRequest = serde_json::from_value(json!({"model":"model", "messages":[
        {"role":"system","content":"You are a coding agent"},
        {"role":"user","content":"<conversation-summary>Investigate pip</conversation-summary>"},
        {"role":"user","content":[
            {"type":"text","text":format!("<conversation-summary>Old pip state</conversation-summary><context>Current portfolio file</context><userRequest>{EVENTS}</userRequest>")},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}
        ]}
    ]})).unwrap();
        let task = resolve_task(Some(Uuid::nil()), "events-request", &req, &[]);
        assert!(task.is_new);
        assert_eq!(scope_request_to_task(&mut req, &task), 1);
        assert_eq!(req.messages.len(), 2);
        let text = req.messages[1].full_text();
        assert!(text.contains("Current portfolio file"));
        assert!(text.contains(EVENTS));
        assert!(!text.contains("pip"));
        assert!(req.messages[1].content.as_ref().unwrap().has_images());
    }

    #[test]
    fn new_task_drops_replies_and_tool_rounds_after_an_old_summary() {
        let mut req: ChatCompletionRequest = serde_json::from_value(json!({"model":"model", "messages":[
        {"role":"system","content":"You are a coding agent"},
        {"role":"user","content":"<conversation-summary>Investigate pip</conversation-summary>"},
        {"role":"assistant","content":"Investigating pip", "tool_calls":[{"id":"pip-1","type":"function","function":{"name":"run_in_terminal","arguments":"{}"}}]},
        {"role":"tool","tool_call_id":"pip-1","content":"Python314 path missing"},
        {"role":"assistant","content":"The pip configuration needs repair"},
        {"role":"user","content":EVENTS}
    ]})).unwrap();
        let task = resolve_task(Some(Uuid::nil()), "events-request", &req, &[]);
        scope_request_to_task(&mut req, &task);
        assert_eq!(
            req.messages.len(),
            2,
            "old replies must not become leading instructions after summary removal"
        );
        assert_eq!(req.messages[1].content_str(), Some(EVENTS));
    }

    #[test]
    fn follow_up_preserves_summary_and_tool_rounds_within_the_selected_task() {
        let history = vec![event(1, "events", "events-request", EVENTS)];
        let summary = "<conversation-summary>Portfolio event reconstruction uses ordered event versions. The next step is pagination.</conversation-summary>";
        let mut req: ChatCompletionRequest =
            serde_json::from_value(json!({"model":"model", "messages":[
                {"role":"system","content":"You are a coding agent"},
                {"role":"user","content":EVENTS},
                {"role":"assistant","content":"Implemented event reconstruction"},
                {"role":"user","content":summary},
                {"role":"assistant","content":"Pagination will use event versions"},
                {"role":"user","content":"Please proceed"}
            ]}))
            .unwrap();
        let task = resolve_task(Some(Uuid::nil()), "follow-up", &req, &history);
        assert_eq!(task.task_id, "events");
        scope_request_to_task(&mut req, &task);
        assert!(
            req.messages
                .iter()
                .any(|message| message.content_str() == Some(summary)),
            "same-task summaries must survive later follow-ups"
        );
    }

    #[test]
    fn literal_summary_markup_in_a_new_feature_is_preserved() {
        let prompt = "Next feature: Build a renderer for <conversation-summary> blocks";
        for raw in [
            prompt.to_string(),
            format!("<userRequest>{prompt}</userRequest>"),
        ] {
            let mut req = request(&[&raw]);
            let task = resolve_task(None, "new-feature", &req, &[]);
            assert_eq!(task.first_prompt, prompt);
            scope_request_to_task(&mut req, &task);
            assert!(req.messages.last().unwrap().full_text().contains(prompt));
        }
    }

    #[tokio::test]
    async fn new_feature_loads_its_own_continuation_goal() {
        let store = crate::in_memory_stores::InMemoryEventStore::default();
        let session = Uuid::nil();
        let mut old_req = request(&[PIP]);
        let old_task = resolve_task(Some(session), "pip-request", &old_req, &[]);
        crate::continuation::prepare(
            &store,
            Some(session),
            &old_task,
            "pip-request",
            &mut old_req,
        )
        .await
        .unwrap();
        let history = vec![event(1, &old_task.task_id, "pip-request", PIP)];
        let mut req = request(&[PIP, EVENTS]);
        let task = resolve_task(Some(session), "events-request", &req, &history);
        scope_request_to_task(&mut req, &task);
        crate::continuation::prepare(&store, Some(session), &task, "events-request", &mut req)
            .await
            .unwrap();
        let state = req
            .extra
            .values()
            .find(|value| value.get("goal").is_some())
            .unwrap();
        assert_eq!(state["goal"], EVENTS);
        assert_eq!(state["task_id"], task.task_id);
        assert!(!serde_json::to_string(state).unwrap().contains("pip script"));
    }
}
