//! Database handle and stable public types for the local engine.
//!
//! Implementations live under `db/`: connection/schema manage startup, sessions
//! and conversations manage identity/lifecycle, events own the event store and
//! transaction helpers, and accounting owns durable usage settlement. Cache,
//! tool-state, ledger, settings, and usage queries each have their own module.
//! Keep SQL and feature-specific behavior in those modules; retain public
//! handles, types, and compatibility re-exports here.

use chrono::{DateTime, Utc};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::sync::{atomic::AtomicUsize, Arc, Mutex};
use uuid::Uuid;

pub(crate) use projects::normalize_project_path;
#[cfg(test)]
mod acceptance;
mod accounting;
mod batch_jobs;
mod budgets;
mod cache;
mod checkpoints;
mod connection;
mod conversations;
pub mod data_lifecycle;
mod events;
mod ledger;
pub mod migrations;
mod operational;
mod outcomes;
mod projects;
mod providers;
mod schema;
mod sessions;
mod settings;
#[cfg(test)]
mod tests;
mod tool_state;
mod usage;

/// A logical project known to the local engine. Directory mappings live in a
/// separate table so one project can cover multiple checkouts or worktrees.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct LocalProject {
    pub id: Uuid,
    pub name: String,
    pub directories: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Owned version of `SessionFilter` for async/spawn_blocking contexts.
struct OwnedSessionFilter {
    limit: Option<usize>,
    before: Option<String>,
    path: Option<String>,
    status: Option<String>,
    mode: Option<String>,
    project_id: Option<String>,
    date_from: Option<String>,
    date_to: Option<String>,
}

/// Local SQLite database handle.
#[derive(Clone, Debug)]
pub struct LocalDatabase {
    /// Writes remain serialized so per-session sequence assignment is atomic.
    conn: Arc<Mutex<Connection>>,
    /// WAL permits independent readers to run without waiting behind a large
    /// audit-log query or an unrelated write transaction.
    readers: Arc<Vec<Mutex<Connection>>>,
    next_reader: Arc<AtomicUsize>,
    event_hub: engine_core::SessionEventHub,
}

/// Filter parameters for listing sessions. All fields are optional -
/// `None` means "no filter on this column".
pub struct SessionFilter<'a> {
    pub limit: Option<usize>,
    pub before: Option<&'a str>,
    pub path: Option<&'a str>,
    pub status: Option<&'a str>,
    pub mode: Option<&'a str>,
    pub project_id: Option<&'a str>,
    /// Inclusive lower bound on `updated_at` (ISO 8601, e.g. "2026-08-01T00:00:00Z").
    pub date_from: Option<&'a str>,
    /// Inclusive upper bound on `updated_at` (ISO 8601, e.g. "2026-08-31T23:59:59Z").
    pub date_to: Option<&'a str>,
}

impl<'a> Default for SessionFilter<'a> {
    fn default() -> Self {
        Self {
            limit: None,
            before: None,
            path: None,
            status: None,
            mode: None,
            project_id: None,
            date_from: None,
            date_to: None,
        }
    }
}
