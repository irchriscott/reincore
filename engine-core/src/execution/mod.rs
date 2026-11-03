//! Providers, routing, and cost.
//!
//! Everything that answers "which model answers this, and what will it
//! cost", organized by stage:
//!
//! - [`providers`]: adapters, model records, catalogs, and pricing.
//! - [`dispatch`]: tier routing, pre-flight prediction, and outcome
//!   correlation.
//! - [`pipeline`]: request orchestration, framework adapters, and content
//!   normalization.

pub mod dispatch;
pub mod pipeline;
pub mod providers;
