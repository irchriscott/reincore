//! Measurable responses to agent-health detections.
//!
//! Detectors only emit evidence. This module resolves operator policy into an
//! explicit response, while keeping authorization and tenant isolation as
//! mandatory boundaries that cannot be downgraded by heuristic settings.

use crate::agent_health::{
    AgentHealthFinding, AgentHealthKind, AgentHealthReport, AgentHealthSeverity,
};
use crate::event::{EventStore, SessionEvent, SessionEventType};
use crate::EngineResult;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::{OnceLock, RwLock};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HeuristicMode {
    Observe,
    Warn,
    #[default]
    RepairOnce,
    Block,
    Pause,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InterventionAction {
    Observed,
    Warned,
    RepairRequested,
    ActionBlocked,
    Paused,
}

impl InterventionAction {
    pub fn is_actionable(self) -> bool {
        matches!(
            self,
            Self::RepairRequested | Self::ActionBlocked | Self::Paused
        )
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InterventionOutcome {
    #[default]
    Pending,
    NotApplicable,
    Recovered,
    Failed,
    FalseAlarm,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MandatoryBoundary {
    Authorization,
    Privacy,
    TenantIsolation,
}

/// These controls are deliberately not configurable. A heuristic can become
/// advisory; an authorization or tenant boundary cannot.
pub fn mandatory_boundary_action(_: MandatoryBoundary) -> InterventionAction {
    InterventionAction::ActionBlocked
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct InterventionPolicy {
    pub loop_detection: HeuristicMode,
    pub prompt_injection: HeuristicMode,
    pub claim_grounding: HeuristicMode,
    pub repeated_questions: HeuristicMode,
    pub action_validation: HeuristicMode,
}

impl Default for InterventionPolicy {
    fn default() -> Self {
        Self {
            loop_detection: HeuristicMode::RepairOnce,
            prompt_injection: HeuristicMode::Warn,
            claim_grounding: HeuristicMode::Warn,
            repeated_questions: HeuristicMode::Block,
            action_validation: HeuristicMode::RepairOnce,
        }
    }
}

impl InterventionPolicy {
    pub fn mode_for(&self, kind: AgentHealthKind) -> HeuristicMode {
        match kind {
            AgentHealthKind::ToolLoop | AgentHealthKind::RedundantLookup => self.loop_detection,
            AgentHealthKind::PromptInjection => self.prompt_injection,
            AgentHealthKind::UngroundedCitation
            | AgentHealthKind::UnverifiedCompletion
            | AgentHealthKind::ContradictedClaim
            | AgentHealthKind::PartialCompletion => self.claim_grounding,
            AgentHealthKind::RepeatedClarification => self.repeated_questions,
            AgentHealthKind::UnknownTool
            | AgentHealthKind::MalformedToolArguments
            | AgentHealthKind::DuplicateToolCall
            | AgentHealthKind::ActionDenied => self.action_validation,
            AgentHealthKind::AuthorizationDenied | AgentHealthKind::PrivacyBoundary => {
                HeuristicMode::Block
            }
            AgentHealthKind::RecoveryStatus | AgentHealthKind::ToolNamespaceAlias => {
                HeuristicMode::Warn
            }
        }
    }

    pub fn from_env() -> Self {
        std::env::var("REINCORE_INTERVENTION_POLICY")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .and_then(|value| serde_json::from_str(&value).ok())
            .unwrap_or_else(process_default_policy)
    }
}

static PROCESS_DEFAULT: OnceLock<RwLock<InterventionPolicy>> = OnceLock::new();

fn process_default_policy() -> InterventionPolicy {
    PROCESS_DEFAULT
        .get_or_init(|| RwLock::new(InterventionPolicy::default()))
        .read()
        .map(|policy| policy.clone())
        .unwrap_or_default()
}

pub fn set_process_default_policy(policy: InterventionPolicy) {
    if let Ok(mut current) = PROCESS_DEFAULT
        .get_or_init(|| RwLock::new(InterventionPolicy::default()))
        .write()
    {
        *current = policy;
    }
}

pub fn current_policy() -> InterventionPolicy {
    InterventionPolicy::from_env()
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct InterventionEvidence {
    pub kind: AgentHealthKind,
    pub severity: AgentHealthSeverity,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

impl From<&AgentHealthFinding> for InterventionEvidence {
    fn from(finding: &AgentHealthFinding) -> Self {
        Self {
            kind: finding.kind,
            severity: finding.severity,
            message: finding.message.clone(),
            tool_name: finding.tool_name.clone(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct InterventionRecord {
    pub id: Uuid,
    pub detector: String,
    pub configured_mode: HeuristicMode,
    pub action: InterventionAction,
    pub evidence: InterventionEvidence,
    pub outcome: InterventionOutcome,
    #[serde(default)]
    pub repair_attempts: u32,
    #[serde(default)]
    pub added_latency_ms: u64,
    #[serde(default)]
    pub added_input_tokens: u64,
    #[serde(default)]
    pub added_output_tokens: u64,
    pub recorded_at: DateTime<Utc>,
}

impl InterventionRecord {
    pub fn from_finding(
        finding: &AgentHealthFinding,
        policy: &InterventionPolicy,
        action: InterventionAction,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            detector: detector_name(finding.kind).into(),
            configured_mode: policy.mode_for(finding.kind),
            action,
            evidence: finding.into(),
            outcome: if action == InterventionAction::RepairRequested {
                InterventionOutcome::Pending
            } else {
                InterventionOutcome::NotApplicable
            },
            repair_attempts: 0,
            added_latency_ms: 0,
            added_input_tokens: 0,
            added_output_tokens: 0,
            recorded_at: Utc::now(),
        }
    }
}

fn detector_name(kind: AgentHealthKind) -> &'static str {
    match kind {
        AgentHealthKind::ToolLoop | AgentHealthKind::RedundantLookup => "loop_detection",
        AgentHealthKind::PromptInjection => "prompt_injection",
        AgentHealthKind::UngroundedCitation
        | AgentHealthKind::UnverifiedCompletion
        | AgentHealthKind::ContradictedClaim
        | AgentHealthKind::PartialCompletion => "claim_grounding",
        AgentHealthKind::RepeatedClarification => "repeated_questions",
        AgentHealthKind::AuthorizationDenied => "authorization",
        AgentHealthKind::PrivacyBoundary => "privacy_redaction",
        AgentHealthKind::RecoveryStatus => "action_recovery",
        _ => "action_validation",
    }
}

fn action_for(kind: AgentHealthKind, mode: HeuristicMode) -> InterventionAction {
    if matches!(
        kind,
        AgentHealthKind::AuthorizationDenied | AgentHealthKind::PrivacyBoundary
    ) {
        return mandatory_boundary_action(if kind == AgentHealthKind::PrivacyBoundary {
            MandatoryBoundary::Privacy
        } else {
            MandatoryBoundary::Authorization
        });
    }
    match mode {
        HeuristicMode::Observe => InterventionAction::Observed,
        HeuristicMode::Warn => InterventionAction::Warned,
        HeuristicMode::RepairOnce => InterventionAction::RepairRequested,
        HeuristicMode::Block => InterventionAction::ActionBlocked,
        HeuristicMode::Pause => InterventionAction::Paused,
    }
}

fn action_for_finding(finding: &AgentHealthFinding, mode: HeuristicMode) -> InterventionAction {
    // Advisory schema diagnostics remain advisory unless strict schema
    // enforcement emitted the separate critical ActionDenied finding.
    if finding.kind == AgentHealthKind::MalformedToolArguments
        && finding.severity == AgentHealthSeverity::Warning
    {
        return InterventionAction::Warned;
    }
    // A deterministic validation failure is never released. Observe/warn can
    // disable the correction request, but cannot convert invalid input into a
    // valid executable action.
    if finding.kind == AgentHealthKind::ActionDenied
        && matches!(mode, HeuristicMode::Observe | HeuristicMode::Warn)
    {
        return InterventionAction::ActionBlocked;
    }
    action_for(finding.kind, mode)
}

/// Apply response policy after detection and attach the exact decision that
/// will be persisted. Calling this more than once is idempotent.
pub fn apply_policy(report: &mut AgentHealthReport) {
    // Healthy reports are the common chat path. Avoid reading and parsing a
    // process policy when there is no new detector evidence to resolve.
    if !report
        .findings
        .iter()
        .any(|finding| finding_needs_policy(report, finding))
    {
        refresh_should_stop(report);
        return;
    }
    let policy = current_policy();
    apply_policy_with(report, &policy);
}

pub fn apply_policy_with(report: &mut AgentHealthReport, policy: &InterventionPolicy) {
    for finding in &report.findings {
        if !finding_needs_policy(report, finding) {
            continue;
        }
        let mode = policy.mode_for(finding.kind);
        report.interventions.push(InterventionRecord::from_finding(
            finding,
            policy,
            action_for_finding(finding, mode),
        ));
    }
    refresh_should_stop(report);
}

fn finding_needs_policy(report: &AgentHealthReport, finding: &AgentHealthFinding) -> bool {
    let summarized_by_denial = matches!(
        finding.kind,
        AgentHealthKind::UnknownTool
            | AgentHealthKind::MalformedToolArguments
            | AgentHealthKind::DuplicateToolCall
    ) && report.findings.iter().any(|other| {
        other.kind == AgentHealthKind::ActionDenied && other.tool_name == finding.tool_name
    });
    finding.kind != AgentHealthKind::RecoveryStatus
        && !summarized_by_denial
        && !report.interventions.iter().any(|record| {
            record.evidence.kind == finding.kind
                && record.evidence.message == finding.message
                && record.evidence.tool_name == finding.tool_name
        })
}

fn refresh_should_stop(report: &mut AgentHealthReport) {
    report.should_stop = report.interventions.iter().any(|record| {
        record.action.is_actionable()
            && !matches!(
                record.outcome,
                InterventionOutcome::Recovered | InterventionOutcome::FalseAlarm
            )
    });
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct InterventionMetrics {
    pub detections: u64,
    pub observations: u64,
    pub warnings: u64,
    pub repair_requests: u64,
    pub blocked_actions: u64,
    pub pauses: u64,
    pub false_alarms: u64,
    pub repeated_questions: u64,
    pub successful_recoveries: u64,
    pub task_completed: bool,
    pub added_latency_ms: u64,
    pub added_input_tokens: u64,
    pub added_output_tokens: u64,
    /// Known estimated cost divided by one completed task. Absent for an
    /// incomplete task or when any call has unknown cost.
    pub cost_per_completed_task_nano_usd: Option<i64>,
}

struct TrackedRecord {
    first_seq: i64,
    last_seq: i64,
    record: InterventionRecord,
    scheduled_input_tokens: u64,
}

impl InterventionMetrics {
    pub fn from_events(
        events: &[SessionEvent],
        completed: bool,
        cost_nano_usd: i64,
        unknown_cost_calls: u64,
    ) -> Self {
        Self::from_event_iter(events.iter(), completed, cost_nano_usd, unknown_cost_calls)
    }

    pub(crate) fn from_event_iter<'a>(
        events: impl IntoIterator<Item = &'a SessionEvent>,
        completed: bool,
        cost_nano_usd: i64,
        unknown_cost_calls: u64,
    ) -> Self {
        let mut metrics = Self {
            task_completed: completed,
            cost_per_completed_task_nano_usd: (completed && unknown_cost_calls == 0)
                .then_some(cost_nano_usd),
            ..Self::default()
        };
        let mut records = std::collections::HashMap::<Uuid, TrackedRecord>::new();
        let mut last_llm_call_seq = None;
        let mut legacy_loops = 0;
        for event in events {
            if event.event_type == SessionEventType::LlmCall {
                last_llm_call_seq =
                    Some(last_llm_call_seq.map_or(event.seq, |latest: i64| latest.max(event.seq)));
            }
            if event.event_type == SessionEventType::LoopIntervention
                && event
                    .payload
                    .get("repetition_count")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or_default()
                    > 0
                && event.payload.get("intervention").is_none()
            {
                legacy_loops += 1;
            }
            if let Some(values) = event
                .payload
                .get("interventions")
                .and_then(|v| v.as_array())
            {
                for value in values {
                    if let Ok(record) = serde_json::from_value::<InterventionRecord>(value.clone())
                    {
                        track_record(&mut records, event, record, 0);
                    }
                }
            }
            if let Some(value) = event.payload.get("intervention") {
                if let Ok(record) = serde_json::from_value::<InterventionRecord>(value.clone()) {
                    let scheduled_input_tokens = if event.event_type
                        == SessionEventType::LoopIntervention
                        && record.action == InterventionAction::RepairRequested
                    {
                        event
                            .payload
                            .get("injected_message")
                            .and_then(serde_json::Value::as_str)
                            .map(crate::cost::count_text_tokens)
                            .unwrap_or(0)
                    } else {
                        0
                    };
                    track_record(&mut records, event, record, scheduled_input_tokens);
                }
            }
        }
        metrics.detections = records.len() as u64 + legacy_loops;
        metrics.repair_requests += legacy_loops;
        for tracked in records.values() {
            let record = &tracked.record;
            let loop_nudge_was_consumed = record.detector == "loop_detection"
                && last_llm_call_seq.is_some_and(|seq| seq > tracked.first_seq);
            match record.action {
                InterventionAction::Observed => metrics.observations += 1,
                InterventionAction::Warned => metrics.warnings += 1,
                InterventionAction::RepairRequested => metrics.repair_requests += 1,
                InterventionAction::ActionBlocked => metrics.blocked_actions += 1,
                InterventionAction::Paused => metrics.pauses += 1,
            }
            metrics.false_alarms += u64::from(record.outcome == InterventionOutcome::FalseAlarm);
            metrics.successful_recoveries +=
                u64::from(record.outcome == InterventionOutcome::Recovered);
            if completed
                && record.detector == "loop_detection"
                && record.action == InterventionAction::RepairRequested
                && record.outcome == InterventionOutcome::Pending
                && loop_nudge_was_consumed
            {
                // A later completed lifecycle is the available durable proof
                // that the loop nudge recovered progress. It is not treated as
                // proof that the answer itself was correct.
                metrics.successful_recoveries += 1;
            }
            metrics.repeated_questions +=
                u64::from(record.evidence.kind == AgentHealthKind::RepeatedClarification);
            metrics.added_latency_ms = metrics
                .added_latency_ms
                .saturating_add(record.added_latency_ms);
            metrics.added_input_tokens = metrics
                .added_input_tokens
                .saturating_add(record.added_input_tokens)
                .saturating_add(if loop_nudge_was_consumed {
                    tracked.scheduled_input_tokens
                } else {
                    0
                });
            metrics.added_output_tokens = metrics
                .added_output_tokens
                .saturating_add(record.added_output_tokens);
        }
        metrics
    }

    pub fn actionable_count(&self) -> u64 {
        self.repair_requests + self.blocked_actions + self.pauses
    }
}

fn track_record(
    records: &mut std::collections::HashMap<Uuid, TrackedRecord>,
    event: &SessionEvent,
    record: InterventionRecord,
    scheduled_input_tokens: u64,
) {
    use std::collections::hash_map::Entry;
    match records.entry(record.id) {
        Entry::Vacant(slot) => {
            slot.insert(TrackedRecord {
                first_seq: event.seq,
                last_seq: event.seq,
                record,
                scheduled_input_tokens,
            });
        }
        Entry::Occupied(mut slot) => {
            let tracked = slot.get_mut();
            tracked.first_seq = tracked.first_seq.min(event.seq);
            tracked.scheduled_input_tokens =
                tracked.scheduled_input_tokens.max(scheduled_input_tokens);
            // Projections are normally ordered, but outcome metrics should not
            // regress if a caller supplies an unsorted event slice.
            if event.seq >= tracked.last_seq {
                tracked.last_seq = event.seq;
                tracked.record = record;
            }
        }
    }
}

/// Explicitly classify a prior intervention (for example, after operator or
/// validator review) without rewriting immutable evidence.
pub async fn record_intervention_outcome(
    store: &dyn EventStore,
    session_id: Uuid,
    intervention_id: Uuid,
    outcome: InterventionOutcome,
) -> EngineResult<SessionEvent> {
    if !matches!(
        outcome,
        InterventionOutcome::Recovered
            | InterventionOutcome::Failed
            | InterventionOutcome::FalseAlarm
    ) {
        return Err(crate::EngineError::InvalidInput(
            "intervention outcome must be recovered, failed, or false_alarm".into(),
        ));
    }
    let events = store.read_events(session_id, None, None).await?;
    let mut record = events
        .iter()
        .rev()
        .find_map(|event| intervention_records(event).find(|record| record.id == intervention_id))
        .ok_or_else(|| {
            crate::EngineError::InvalidInput(
                "intervention does not belong to the requested session".into(),
            )
        })?;
    record.outcome = outcome;
    store
        .append_event(
            session_id,
            SessionEventType::AgentHealth,
            serde_json::json!({"intervention": record}),
            None,
        )
        .await
}

fn intervention_records(event: &SessionEvent) -> impl Iterator<Item = InterventionRecord> + '_ {
    event
        .payload
        .get("interventions")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .chain(event.payload.get("intervention"))
        .filter_map(|value| serde_json::from_value(value.clone()).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mandatory_boundaries_cannot_be_downgraded() {
        for boundary in [
            MandatoryBoundary::Authorization,
            MandatoryBoundary::Privacy,
            MandatoryBoundary::TenantIsolation,
        ] {
            assert_eq!(
                mandatory_boundary_action(boundary),
                InterventionAction::ActionBlocked
            );
        }
        let policy = InterventionPolicy {
            action_validation: HeuristicMode::Observe,
            ..Default::default()
        };
        assert_eq!(
            policy.mode_for(AgentHealthKind::AuthorizationDenied),
            HeuristicMode::Block
        );
        assert_eq!(
            policy.mode_for(AgentHealthKind::PrivacyBoundary),
            HeuristicMode::Block
        );
        let mut authorization = AgentHealthReport {
            findings: vec![AgentHealthFinding {
                kind: AgentHealthKind::AuthorizationDenied,
                severity: AgentHealthSeverity::Critical,
                message: "outside tenant scope".into(),
                tool_name: Some("read".into()),
            }],
            ..Default::default()
        };
        apply_policy_with(&mut authorization, &policy);
        assert!(authorization.should_stop);
        assert_eq!(
            authorization.interventions[0].action,
            InterventionAction::ActionBlocked
        );

        let mut invalid = AgentHealthReport {
            findings: vec![AgentHealthFinding {
                kind: AgentHealthKind::ActionDenied,
                severity: AgentHealthSeverity::Critical,
                message: "invalid arguments".into(),
                tool_name: Some("read".into()),
            }],
            ..Default::default()
        };
        apply_policy_with(&mut invalid, &policy);
        assert!(invalid.should_stop);
        assert_eq!(
            invalid.interventions[0].action,
            InterventionAction::ActionBlocked
        );
    }

    #[test]
    fn every_heuristic_mode_has_a_distinct_response() {
        for (mode, action, stops) in [
            (HeuristicMode::Observe, InterventionAction::Observed, false),
            (HeuristicMode::Warn, InterventionAction::Warned, false),
            (
                HeuristicMode::RepairOnce,
                InterventionAction::RepairRequested,
                true,
            ),
            (
                HeuristicMode::Block,
                InterventionAction::ActionBlocked,
                true,
            ),
            (HeuristicMode::Pause, InterventionAction::Paused, true),
        ] {
            let policy = InterventionPolicy {
                prompt_injection: mode,
                ..Default::default()
            };
            let mut report = AgentHealthReport {
                findings: vec![AgentHealthFinding {
                    kind: AgentHealthKind::PromptInjection,
                    severity: AgentHealthSeverity::Critical,
                    message: "untrusted instruction".into(),
                    tool_name: None,
                }],
                ..Default::default()
            };
            apply_policy_with(&mut report, &policy);
            assert_eq!(report.interventions[0].action, action);
            assert_eq!(report.should_stop, stops);
        }
    }

    #[test]
    fn metrics_deduplicate_updates_and_measure_overhead() {
        let finding = AgentHealthFinding {
            kind: AgentHealthKind::RepeatedClarification,
            severity: AgentHealthSeverity::Critical,
            message: "already answered".into(),
            tool_name: None,
        };
        let mut record = InterventionRecord::from_finding(
            &finding,
            &InterventionPolicy::default(),
            InterventionAction::RepairRequested,
        );
        let first = SessionEvent {
            session_id: Uuid::nil(),
            seq: 1,
            event_type: SessionEventType::AgentHealth,
            payload: serde_json::json!({"intervention": record}),
            cost_cents: None,
            created_at: Utc::now(),
        };
        record.outcome = InterventionOutcome::Recovered;
        record.added_latency_ms = 12;
        record.added_input_tokens = 7;
        let second = SessionEvent {
            seq: 2,
            payload: serde_json::json!({"intervention": record}),
            ..first.clone()
        };
        let metrics = InterventionMetrics::from_events(&[first, second], true, 42, 0);
        assert_eq!(metrics.detections, 1);
        assert_eq!(metrics.successful_recoveries, 1);
        assert_eq!(metrics.repeated_questions, 1);
        assert_eq!(metrics.added_latency_ms, 12);
        assert_eq!(metrics.added_input_tokens, 7);
        assert_eq!(metrics.cost_per_completed_task_nano_usd, Some(42));
    }

    #[test]
    fn loop_recovery_and_token_overhead_require_a_later_provider_call() {
        let finding = AgentHealthFinding {
            kind: AgentHealthKind::ToolLoop,
            severity: AgentHealthSeverity::Warning,
            message: "repeating".into(),
            tool_name: Some("read".into()),
        };
        let record = InterventionRecord::from_finding(
            &finding,
            &InterventionPolicy::default(),
            InterventionAction::RepairRequested,
        );
        let scheduled = SessionEvent {
            session_id: Uuid::nil(),
            seq: 5,
            event_type: SessionEventType::LoopIntervention,
            payload: serde_json::json!({
                "injected_message": "Try a different source.",
                "repetition_count": 3,
                "intervention": record,
            }),
            cost_cents: None,
            created_at: Utc::now(),
        };

        let before_use =
            InterventionMetrics::from_events(std::slice::from_ref(&scheduled), true, 0, 0);
        assert_eq!(before_use.successful_recoveries, 0);
        assert_eq!(before_use.added_input_tokens, 0);

        let later_call = SessionEvent {
            seq: 6,
            event_type: SessionEventType::LlmCall,
            payload: serde_json::json!({}),
            ..scheduled.clone()
        };
        let after_use = InterventionMetrics::from_events(&[scheduled, later_call], true, 0, 0);
        assert_eq!(after_use.successful_recoveries, 1);
        assert!(after_use.added_input_tokens > 0);
    }

    #[tokio::test]
    async fn outcome_feedback_is_terminal_and_bound_to_the_session_record() {
        let store = crate::InMemoryEventStore::default();
        let session = Uuid::new_v4();
        let other_session = Uuid::new_v4();
        let finding = AgentHealthFinding {
            kind: AgentHealthKind::PromptInjection,
            severity: AgentHealthSeverity::Warning,
            message: "suspicious retrieved instruction".into(),
            tool_name: None,
        };
        let record = InterventionRecord::from_finding(
            &finding,
            &InterventionPolicy::default(),
            InterventionAction::Warned,
        );
        store
            .append_event(
                session,
                SessionEventType::AgentHealth,
                serde_json::json!({"intervention": record}),
                None,
            )
            .await
            .unwrap();

        assert!(record_intervention_outcome(
            &store,
            other_session,
            record.id,
            InterventionOutcome::FalseAlarm,
        )
        .await
        .is_err());
        assert!(record_intervention_outcome(
            &store,
            session,
            record.id,
            InterventionOutcome::Pending,
        )
        .await
        .is_err());

        let updated = record_intervention_outcome(
            &store,
            session,
            record.id,
            InterventionOutcome::FalseAlarm,
        )
        .await
        .unwrap();
        let saved: InterventionRecord =
            serde_json::from_value(updated.payload["intervention"].clone()).unwrap();
        assert_eq!(saved.evidence, record.evidence);
        assert_eq!(saved.outcome, InterventionOutcome::FalseAlarm);
    }
}
