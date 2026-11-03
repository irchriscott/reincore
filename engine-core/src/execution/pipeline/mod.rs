//! Request pipeline: orchestration, adapters, and content normalization.
//!
//! Everything that runs a turn end to end: the shared decision pipeline,
//! framework adapters for external orchestrators, correlation of
//! externally executed tools, user-message normalization, and the private
//! lexical and tool-name heuristics task intelligence builds on.

pub mod external_tools;
pub mod framework;
pub(crate) mod lexical;
pub(crate) mod tool_names;
pub mod user_content;
pub mod workflow;
