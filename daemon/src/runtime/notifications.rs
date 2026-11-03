//! Desktop notifications for the local engine.
//!
//! There is one person and no membership, so every event is yours. What is
//! left is which categories are worth interrupting for, which lives in
//! `~/.reincore-io/config.toml`.
//!
//! The daemon runs in its own process, so the desktop app cannot share its
//! in-process event hub. It reads the log instead, paging by the monotonic
//! `session_events.id` - the column exists precisely so a second consumer can
//! do this without the ambiguity of a timestamp cursor.
//!
//! Nothing here shows a toast; it only decides what is worth showing. The
//! Tauri layer owns the OS call, because that is the only part that needs a
//! window.

use std::path::Path;

use anyhow::Result;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

/// One thing worth telling the person at the keyboard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalNotification {
    /// The `session_events.id` this came from. The caller stores the highest
    /// it has shown so the next poll resumes after it.
    pub id: i64,
    pub session_id: String,
    pub event_type: String,
    pub category: LocalCategory,
    pub title: String,
    pub body: String,
}

/// The categories a local user can switch on or off.
///
/// Deliberately fewer than the cloud's: without membership there is no
/// security or access story to report, and budget locally is a single
/// person's own limit rather than a shared one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalCategory {
    /// A run reached an end state.
    RunOutcomes,
    /// A grounded finding or a circuit-breaker intervention.
    AgentHealth,
    /// Spend crossed a degradation boundary.
    Budget,
}

impl LocalCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            LocalCategory::RunOutcomes => "run_outcomes",
            LocalCategory::AgentHealth => "agent_health",
            LocalCategory::Budget => "budget",
        }
    }
}

/// Which session event types are worth a toast, and how they read.
///
/// The per-call types (`llm_call`, `tool_call`, `tool_result`) are absent for
/// the same reason they are not webhook events: they fire constantly, and an
/// OS toast per model call would make the machine unusable.
fn classify(event_type: &str) -> Option<(LocalCategory, &'static str, &'static str)> {
    match event_type {
        "session_completed" => Some((
            LocalCategory::RunOutcomes,
            "Session completed",
            "A run you started finished.",
        )),
        "checkpoint" => Some((
            LocalCategory::RunOutcomes,
            "Session paused",
            "A run hit a hard stop and paused at a checkpoint it can resume from.",
        )),
        "checkpoint_redeemed" => Some((
            LocalCategory::RunOutcomes,
            "Session resumed",
            "A paused run picked up again from its checkpoint.",
        )),
        "batch" => Some((
            LocalCategory::RunOutcomes,
            "Batch finished",
            "A batch job finished.",
        )),
        "agent_health" => Some((
            LocalCategory::AgentHealth,
            "Agent health finding",
            "A run recorded a grounded finding - a hallucination or a lack of progress.",
        )),
        "loop_intervention" => Some((
            LocalCategory::AgentHealth,
            "Loop intervention",
            "The circuit breaker interrupted a repeating loop.",
        )),
        "tier_change" => Some((
            LocalCategory::Budget,
            "Budget tier changed",
            "Spend moved this session to a cheaper tier, which changes how it is routed.",
        )),
        _ => None,
    }
}

/// Which categories the person wants interrupting for.
///
/// Read from `[notifications]` in `~/.reincore-io/config.toml`, and written back
/// by the desktop app. Each field carries its own serde default rather than
/// the struct deriving one, because a table that sets a single key must leave
/// the other three at the values [`Default`] gives them - not at `false`,
/// which is where a struct-level `#[serde(default)]` would quietly put them.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LocalNotificationSettings {
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default = "default_run_outcomes")]
    pub run_outcomes: bool,
    #[serde(default = "default_agent_health")]
    pub agent_health: bool,
    #[serde(default = "default_budget")]
    pub budget: bool,
}

// One source of truth for both `Default` and the serde field defaults, so the
// two cannot drift apart.
const DEFAULT_ENABLED: bool = true;
/// Run outcomes and budget default on because they are why somebody walks
/// away from a long run.
const DEFAULT_RUN_OUTCOMES: bool = true;
/// Agent health defaults off: it is diagnostic, it fires more often, and it
/// is better read in the app than as a toast.
const DEFAULT_AGENT_HEALTH: bool = false;
const DEFAULT_BUDGET: bool = true;

fn default_enabled() -> bool {
    DEFAULT_ENABLED
}
fn default_run_outcomes() -> bool {
    DEFAULT_RUN_OUTCOMES
}
fn default_agent_health() -> bool {
    DEFAULT_AGENT_HEALTH
}
fn default_budget() -> bool {
    DEFAULT_BUDGET
}

impl Default for LocalNotificationSettings {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_ENABLED,
            run_outcomes: DEFAULT_RUN_OUTCOMES,
            agent_health: DEFAULT_AGENT_HEALTH,
            budget: DEFAULT_BUDGET,
        }
    }
}

impl LocalNotificationSettings {
    fn allows(&self, category: LocalCategory) -> bool {
        if !self.enabled {
            return false;
        }
        match category {
            LocalCategory::RunOutcomes => self.run_outcomes,
            LocalCategory::AgentHealth => self.agent_health,
            LocalCategory::Budget => self.budget,
        }
    }
}

/// Read events newer than `after_id` that the settings admit.
///
/// Returns at most `limit`, oldest first, so a caller that toasts them in
/// order shows them in the order they happened. An empty result leaves the
/// caller's cursor where it was.
///
/// The database is opened read-only: the daemon owns this file and is very
/// likely writing to it, and a reader must never take a write lock.
pub fn poll(
    db_path: impl AsRef<Path>,
    after_id: i64,
    limit: usize,
    settings: LocalNotificationSettings,
) -> Result<Vec<LocalNotification>> {
    if !settings.enabled {
        return Ok(Vec::new());
    }
    let conn = Connection::open_with_flags(
        db_path.as_ref(),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    )?;

    let mut statement = conn.prepare(
        "SELECT id, session_id, event_type
         FROM session_events
         WHERE id > ?1
         ORDER BY id
         LIMIT ?2",
    )?;

    let rows = statement.query_map(rusqlite::params![after_id, limit as i64], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;

    let mut out = Vec::new();
    for row in rows {
        let (id, session_id, event_type) = row?;
        let Some((category, title, body)) = classify(&event_type) else {
            continue;
        };
        if !settings.allows(category) {
            continue;
        }
        out.push(LocalNotification {
            id,
            session_id,
            event_type,
            category,
            title: title.to_string(),
            body: body.to_string(),
        });
    }
    Ok(out)
}

/// The highest `session_events.id` currently in the log.
///
/// A first run starts here rather than at zero, so switching notifications on
/// does not toast the entire history at once.
pub fn latest_event_id(db_path: impl AsRef<Path>) -> Result<i64> {
    let conn = Connection::open_with_flags(
        db_path.as_ref(),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    )?;
    let newest: Option<i64> =
        conn.query_row("SELECT MAX(id) FROM session_events", [], |row| row.get(0))?;
    Ok(newest.unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed(path: &Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            "CREATE TABLE session_events (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               session_id TEXT NOT NULL,
               seq INTEGER NOT NULL,
               event_type TEXT NOT NULL,
               payload TEXT NOT NULL,
               cost_cents INTEGER,
               created_at TEXT NOT NULL
             );",
        )
        .unwrap();
        for (seq, event_type) in [
            (1, "llm_call"),
            (2, "tool_call"),
            (3, "agent_health"),
            (4, "tier_change"),
            (5, "session_completed"),
        ] {
            conn.execute(
                "INSERT INTO session_events (session_id, seq, event_type, payload, created_at)
                 VALUES ('s1', ?1, ?2, '{}', '2026-09-12T00:00:00Z')",
                rusqlite::params![seq, event_type],
            )
            .unwrap();
        }
    }

    fn temp_db(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("reincore-notif-{name}.db"));
        let _ = std::fs::remove_file(&path);
        seed(&path);
        path
    }

    #[test]
    fn per_call_events_never_notify() {
        let path = temp_db("percall");
        let all = LocalNotificationSettings {
            enabled: true,
            run_outcomes: true,
            agent_health: true,
            budget: true,
        };
        let found = poll(&path, 0, 100, all).unwrap();
        assert!(
            found
                .iter()
                .all(|n| n.event_type != "llm_call" && n.event_type != "tool_call"),
            "a per-call event produced a toast"
        );
        assert_eq!(found.len(), 3, "expected the three deliverable events");
    }

    #[test]
    fn settings_filter_by_category() {
        let path = temp_db("filter");
        let runs_only = LocalNotificationSettings {
            enabled: true,
            run_outcomes: true,
            agent_health: false,
            budget: false,
        };
        let found = poll(&path, 0, 100, runs_only).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].event_type, "session_completed");
    }

    #[test]
    fn disabled_yields_nothing() {
        let path = temp_db("disabled");
        let off = LocalNotificationSettings {
            enabled: false,
            ..Default::default()
        };
        assert!(poll(&path, 0, 100, off).unwrap().is_empty());
    }

    #[test]
    fn the_cursor_advances_past_what_was_shown() {
        let path = temp_db("cursor");
        let all = LocalNotificationSettings {
            enabled: true,
            run_outcomes: true,
            agent_health: true,
            budget: true,
        };
        let first = poll(&path, 0, 100, all).unwrap();
        let highest = first.iter().map(|n| n.id).max().unwrap();
        assert!(
            poll(&path, highest, 100, all).unwrap().is_empty(),
            "polling past the cursor returned events again"
        );
    }

    #[test]
    fn latest_event_id_matches_the_log() {
        let path = temp_db("latest");
        assert_eq!(latest_event_id(&path).unwrap(), 5);
    }
}
