//! Tier routing, pre-flight cost prediction, and outcome correlation.
//!
//! Everything that answers "which tier handles this turn and what will it
//! cost": the escalation classifier and routing policy, the cost predictor
//! the budget router consumes, and lifecycle diagnostics for request groups,
//! tasks, and sessions.

pub mod cost;
pub mod cost_outcome;
pub mod routing;
