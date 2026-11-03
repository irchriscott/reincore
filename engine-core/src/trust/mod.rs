//! Budgets, privacy, safety, and policy.
//!
//! Everything that answers "is this allowed, and what does it cost the
//! budget", organized by role:
//!
//! - [`privacy`]: the data-leaving-the-boundary policy and its helpers.
//! - [`enforcement`]: pre-call checks, evidence receipts, and recovery.
//! - [`oversight`]: behavioral detection, loop breaking, and verification.
//!
//! [`budget`] and [`checkpoint`] stay here: the spend ledger and its
//! resumable continuation tokens.

pub mod budget;
pub mod checkpoint;
pub mod enforcement;
pub mod oversight;
pub mod privacy;
