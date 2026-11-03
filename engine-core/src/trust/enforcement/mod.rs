//! Action enforcement: pre-call checks, evidence receipts, and recovery.
//!
//! Everything that constrains what the agent may do: deterministic gates
//! before tool calls leave the gateway, bounded execution receipts as
//! evidence contracts, one bounded provider correction after a rejection,
//! and the operator policy responses to health detections.

pub mod action_gate;
pub mod action_receipts;
pub mod action_recovery;
pub mod intervention;
