//! Durable work records: batch jobs and completion accounting.
//!
//! Persist intent before submission and immutable receipts before
//! settlement, so retries settle with the same record instead of
//! generating a new identity.

pub mod accounting;
pub mod batch_jobs;
