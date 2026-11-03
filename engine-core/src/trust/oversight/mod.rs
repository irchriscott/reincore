//! Behavioral oversight: detection, loop breaking, and verification.
//!
//! Everything that answers "did the agent behave": health monitoring and
//! guard-rail detection, the progress-aware circuit breaker, call-level
//! diagnostics, and completion claims checked against correlated tool
//! results. Detectors emit evidence; responses live in enforcement.

pub mod agent_health;
pub mod circuit;
pub mod claim_grounding;
pub mod diagnostics;
