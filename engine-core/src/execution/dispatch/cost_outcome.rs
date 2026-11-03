//! Cost/outcome correlation for request groups, tasks, and sessions.
//!
//! Completion is a lifecycle diagnostic, never evidence of answer quality.
//! Success requires an explicit validator or user report. These diagnostics
//! must not be used as autonomous policy rewards.

use crate::event::{SessionEvent, SessionEventType};
use crate::session::{Session, SessionStatus};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeSource {
    Validator,
    User,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeVerdict {
    Success,
    Failure,
    Unknown,
}

/// Caller-supplied evidence, not a verdict inferred by the engine.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OutcomeReport {
    pub source: OutcomeSource,
    pub verdict: OutcomeVerdict,
    pub evidence: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SuccessStatus {
    #[default]
    Unknown,
    ValidatedSuccess,
    UserReportedSuccess,
    RecoveredSuccess,
    ReportedFailure,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CostOutcome {
    pub completed: bool,
    pub clean_completion: bool,
    #[serde(default)]
    pub completed_after_intervention: bool,
    #[serde(default)]
    pub success_status: SuccessStatus,
    #[serde(default)]
    pub outcome_report: Option<OutcomeReport>,
    /// Lifecycle completions / known estimated USD; not a quality score.
    pub completions_per_estimated_dollar: Option<f64>,
    /// Absent when only the inexpensive lifecycle/outcome projection was loaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<crate::diagnostics::CallDiagnostics>,
    pub loop_intervention_count: u64,
    #[serde(default)]
    pub intervention_metrics: crate::intervention::InterventionMetrics,
    pub cost_cents: i64,
    pub cost_nano_usd: i64,
    pub unknown_cost_calls: u64,
    pub cost_status: String,
    /// Deprecated: always absent. Completion cannot establish quality.
    pub completion_quality_bps: Option<u32>,
    /// Deprecated: always absent. No quality-adjusted efficiency is measured.
    pub cost_efficiency_score: Option<f64>,
}

impl CostOutcome {
    pub fn new(completed: bool, cost_cents: i64, loop_intervention_count: u64) -> Self {
        Self::precise(
            completed,
            cost_cents.saturating_mul(crate::pricing::NANO_USD_PER_CENT),
            0,
            loop_intervention_count,
        )
    }
    pub fn precise(
        completed: bool,
        cost_nano_usd: i64,
        unknown_cost_calls: u64,
        loop_intervention_count: u64,
    ) -> Self {
        let cost_cents = cost_nano_usd / crate::pricing::NANO_USD_PER_CENT;
        let clean_completion = completed && loop_intervention_count == 0;
        let completion_quality_bps = None;
        let cost_efficiency_score = None;
        Self {
            completed,
            clean_completion,
            completed_after_intervention: completed && loop_intervention_count > 0,
            success_status: SuccessStatus::Unknown,
            outcome_report: None,
            completions_per_estimated_dollar: (completed
                && unknown_cost_calls == 0
                && cost_nano_usd > 0)
                .then(|| 1_000_000_000.0 / cost_nano_usd as f64),
            diagnostics: Default::default(),
            loop_intervention_count,
            intervention_metrics: crate::intervention::InterventionMetrics {
                task_completed: completed,
                cost_per_completed_task_nano_usd: (completed && unknown_cost_calls == 0)
                    .then_some(cost_nano_usd),
                ..Default::default()
            },
            cost_cents,
            cost_nano_usd,
            unknown_cost_calls,
            cost_status: if unknown_cost_calls > 0 {
                "unknown"
            } else {
                "estimated"
            }
            .into(),
            completion_quality_bps,
            cost_efficiency_score,
        }
    }

    pub fn with_report(mut self, report: Option<OutcomeReport>, recovered: bool) -> Self {
        let report = report.filter(|report| !report.evidence.trim().is_empty());
        self.success_status = match report.as_ref() {
            Some(report) if report.verdict == OutcomeVerdict::Failure => {
                SuccessStatus::ReportedFailure
            }
            Some(report) if self.completed && report.verdict == OutcomeVerdict::Success => {
                if recovered || self.completed_after_intervention {
                    SuccessStatus::RecoveredSuccess
                } else if report.source == OutcomeSource::Validator {
                    SuccessStatus::ValidatedSuccess
                } else {
                    SuccessStatus::UserReportedSuccess
                }
            }
            _ => SuccessStatus::Unknown,
        };
        self.outcome_report = report;
        self
    }
}

/// Tombstones written when a pending intervention is consumed are lifecycle
/// bookkeeping, not actual agent-health interventions.
pub fn is_effective_loop_intervention(event: &SessionEvent) -> bool {
    if event.event_type != SessionEventType::LoopIntervention {
        return false;
    }
    event
        .payload
        .get("repetition_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
        > 0
        || event
            .payload
            .get("injected_message")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|message| !message.trim().is_empty())
}

pub fn session_cost_outcome(session: &Session, events: &[SessionEvent]) -> CostOutcome {
    let mut outcome = session_cost_outcome_summary(session, events);
    outcome.diagnostics = Some(crate::diagnostics::CallDiagnostics::from_events(events));
    outcome
}

/// Build outcome labels from lifecycle events without replaying call telemetry.
pub fn session_cost_outcome_summary(session: &Session, events: &[SessionEvent]) -> CostOutcome {
    let interventions = events
        .iter()
        .filter(|event| is_effective_loop_intervention(event))
        .count() as u64;
    let cost_nano_usd = session.spend_nano_usd.unwrap_or_else(|| {
        session
            .spend_cents
            .saturating_mul(crate::pricing::NANO_USD_PER_CENT)
    });
    let completed = session.status == SessionStatus::Completed;
    let mut outcome = CostOutcome::precise(
        completed,
        cost_nano_usd,
        session.unknown_cost_calls,
        interventions,
    )
    .with_report(
        events
            .iter()
            .filter(|event| event.event_type == SessionEventType::SessionCompleted)
            .max_by_key(|event| event.seq)
            .and_then(|event| event.payload.get("outcome_report"))
            .and_then(|value| serde_json::from_value(value.clone()).ok()),
        events
            .iter()
            .any(|event| event.event_type == SessionEventType::CheckpointRedeemed),
    );
    outcome.intervention_metrics = crate::intervention::InterventionMetrics::from_events(
        events,
        completed,
        cost_nano_usd,
        session.unknown_cost_calls,
    );
    outcome.completed_after_intervention =
        completed && outcome.intervention_metrics.actionable_count() > 0;
    outcome.clean_completion = completed && outcome.intervention_metrics.detections == 0;
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn intervention(repetition_count: u64, message: &str) -> SessionEvent {
        SessionEvent {
            session_id: Uuid::nil(),
            seq: 1,
            event_type: SessionEventType::LoopIntervention,
            payload: serde_json::json!({
                "repetition_count": repetition_count,
                "injected_message": message,
            }),
            cost_cents: None,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn completion_does_not_establish_quality() {
        let outcome = CostOutcome::new(true, 200, 0);
        assert_eq!(outcome.completion_quality_bps, None);
        assert_eq!(outcome.cost_efficiency_score, None);
        assert!(outcome.clean_completion);
        assert_eq!(outcome.success_status, SuccessStatus::Unknown);
        assert_eq!(outcome.completions_per_estimated_dollar, Some(0.5));
    }

    #[test]
    fn explicit_reports_separate_failure_success_and_recovery() {
        let report = |source, verdict| {
            Some(OutcomeReport {
                source,
                verdict,
                evidence: "test result".into(),
            })
        };
        let wrong = CostOutcome::new(true, 200, 0).with_report(
            report(OutcomeSource::Validator, OutcomeVerdict::Failure),
            false,
        );
        assert_eq!(wrong.success_status, SuccessStatus::ReportedFailure);
        assert_eq!(wrong.cost_efficiency_score, None);
        let validated = CostOutcome::new(true, 200, 0).with_report(
            report(OutcomeSource::Validator, OutcomeVerdict::Success),
            false,
        );
        assert_eq!(validated.success_status, SuccessStatus::ValidatedSuccess);
        let recovered = CostOutcome::new(true, 200, 1)
            .with_report(report(OutcomeSource::User, OutcomeVerdict::Success), false);
        assert_eq!(recovered.success_status, SuccessStatus::RecoveredSuccess);
        assert_eq!(recovered.completions_per_estimated_dollar, Some(0.5));
        let checkpoint = CostOutcome::new(true, 200, 0).with_report(
            report(OutcomeSource::Validator, OutcomeVerdict::Success),
            true,
        );
        assert_eq!(checkpoint.success_status, SuccessStatus::RecoveredSuccess);
        let user = CostOutcome::new(true, 200, 0)
            .with_report(report(OutcomeSource::User, OutcomeVerdict::Success), false);
        assert_eq!(user.success_status, SuccessStatus::UserReportedSuccess);
        let unfinished = CostOutcome::new(false, 200, 0)
            .with_report(report(OutcomeSource::User, OutcomeVerdict::Success), false);
        assert_eq!(unfinished.success_status, SuccessStatus::Unknown);
    }

    #[test]
    fn incomplete_work_has_no_score() {
        let outcome = CostOutcome::new(false, 200, 0);
        assert_eq!(outcome.completion_quality_bps, None);
        assert_eq!(outcome.cost_efficiency_score, None);
        assert!(!outcome.clean_completion);
    }

    #[test]
    fn intervention_tombstones_do_not_reduce_health() {
        assert!(is_effective_loop_intervention(&intervention(
            3,
            "try again"
        )));
        assert!(!is_effective_loop_intervention(&intervention(0, "")));
    }
}

#[cfg(test)]
mod precise_tests {
    use super::*;
    #[test]
    fn efficiency_uses_fractional_cents_and_refuses_unknown_denominators() {
        assert_eq!(
            CostOutcome::precise(true, 100_000, 0, 0).completions_per_estimated_dollar,
            Some(10_000.0)
        );
        assert_eq!(
            CostOutcome::precise(true, 100_000, 1, 0).completions_per_estimated_dollar,
            None
        );
        assert_eq!(
            CostOutcome::precise(true, 0, 0, 0).completions_per_estimated_dollar,
            None
        );
    }
}
