//! Agent session lifecycle and identity.
//!
//! Everything that answers "which session is this, where did it come from,
//! and what happened in it": the session record, the immutable event log,
//! live event broadcast, client conversation bindings, continuation lineage,
//! cross-session handoff, and request/task grouping.

pub mod continuation;
pub(crate) mod conversation;
pub mod event;
pub mod grouping;
pub mod handoff;
pub mod live_events;
pub mod session;
