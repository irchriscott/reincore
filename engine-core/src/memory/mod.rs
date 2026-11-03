//! Context assembly, compaction, and recall.
//!
//! Everything that answers "what does the agent need to see this turn":
//! tiered memory management, context compression, task tracking and typing,
//! agent-made task decisions, summaries, display titles, and the bounded,
//! attributed history of structured file mutations.

pub mod compression;
pub mod context;
pub mod file_history;
pub mod summary;
pub mod task;
pub mod task_decisions;
pub mod task_type;
pub mod title_generation;
