//! Privacy boundary: the policy, source recognition, and fixture feedback.
//!
//! Everything that guards data leaving a trust boundary: the privacy
//! policy enforced by providers, tools, logging, and stores; bounded
//! syntax-aware source recognition; and non-blocking feedback for
//! repeated secret-associated test failures.

pub mod policy;
pub(crate) mod privacy_fixture_recovery;
pub mod privacy_source;
