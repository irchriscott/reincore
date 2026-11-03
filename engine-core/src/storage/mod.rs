//! Persistence, caching, and runtime infrastructure.
//!
//! Everything that answers "where is this stored, and how does the engine
//! run", organized by kind:
//!
//! - [`persistence`]: store contracts, implementations, and schema.
//! - [`settlement`]: durable batch and accounting records.
//! - [`caching`]: memoization, semantic cache, and prompt-cache support.
//! - [`observability`]: metrics, tracing, and logging.
//!
//! [`error`] and [`resilience`] stay here: they cut across every store and
//! serve path.

pub mod caching;
pub mod error;
pub mod observability;
pub mod persistence;
pub mod resilience;
pub mod settlement;
