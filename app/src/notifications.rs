//! OS toasts for desktop notifications.
//!
//! There is one person and no membership, so there is nothing to resolve:
//! everything that happens is theirs. What remains is a background poll and
//! an OS toast.
//!
//! The daemon runs in its own process, so this cannot subscribe to its
//! in-process event hub. It reads the log directly instead, paging by the
//! monotonic `session_events.id`. `reincore::notifications` owns the
//! query and the classification; this module owns the window, the timer, and
//! the cursor.

use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use reincore::notifications::{self, LocalNotificationSettings};
use tauri::AppHandle;
use tauri_plugin_notification::NotificationExt;

/// The settings the poll loop reads, shared with the commands that edit them.
///
/// Shared rather than copied in at startup because these are toasts: a switch
/// that only takes effect after a restart is a switch people conclude is
/// broken. Turning a category off stops the next toast, not the next launch.
pub type SharedSettings = Arc<RwLock<LocalNotificationSettings>>;

/// Read the current settings, recovering from a poisoned lock.
///
/// A panic while someone held the write lock should cost a toast, not every
/// toast from here on: the value behind it is a plain copy of four booleans
/// and cannot have been left half-written.
fn current(settings: &SharedSettings) -> LocalNotificationSettings {
    match settings.read() {
        Ok(guard) => *guard,
        Err(poisoned) => *poisoned.into_inner(),
    }
}

/// How often the log is checked. Frequent enough that a finished run is
/// noticed while the person is still looking away, rare enough to be free.
const POLL_INTERVAL: Duration = Duration::from_secs(5);

/// How many events one pass will toast.
///
/// A burst larger than this is throttled rather than fired all at once: a
/// machine that wakes to forty toasts is worse than one that gets a few and
/// catches up. The cursor still advances past everything read, so nothing is
/// shown twice.
const MAX_PER_PASS: usize = 3;

/// Where the "already shown" watermark lives.
///
/// A file rather than the config, because it is state rather than
/// preference - rewriting the user's config every five seconds would be
/// rude, and losing this file only costs a quiet restart.
fn cursor_path() -> Option<PathBuf> {
    dirs_next_home().map(|home| home.join(".reincore-io").join("notification-cursor"))
}

fn dirs_next_home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn read_cursor() -> Option<i64> {
    let path = cursor_path()?;
    std::fs::read_to_string(path)
        .ok()?
        .trim()
        .parse::<i64>()
        .ok()
}

fn write_cursor(value: i64) {
    let Some(path) = cursor_path() else { return };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, value.to_string());
}

/// Start the background poll. Returns immediately; the work happens on a
/// Tauri async task for the lifetime of the app.
///
/// The loop runs even when notifications are switched off. It has to: there
/// would otherwise be nothing to switch back on without a restart, and a loop
/// that keeps running while disabled also keeps the cursor at the newest
/// event, so turning toasts on later shows what happens next rather than
/// replaying everything that happened while they were off.
///
/// A failure to read the database is not fatal and is not reported to the
/// user: the daemon may simply not have created it yet, and a toast about
/// missing toasts helps nobody.
pub fn spawn(app: AppHandle, db_path: PathBuf, settings: SharedSettings) {
    if !current(&settings).enabled {
        tracing::info!("notifications: desktop toasts are switched off");
    }

    // A first run starts at the newest event, so switching notifications on
    // does not toast the whole history at once.
    let start =
        read_cursor().unwrap_or_else(|| notifications::latest_event_id(&db_path).unwrap_or(0));
    let cursor = Arc::new(AtomicI64::new(start));
    write_cursor(start);

    tauri::async_runtime::spawn(async move {
        let mut ticker = tokio::time::interval(POLL_INTERVAL);
        loop {
            ticker.tick().await;

            let after = cursor.load(Ordering::Relaxed);
            // Re-read every tick rather than capturing once, so a change made
            // in Settings applies to the next poll.
            let live = current(&settings);
            let found = match notifications::poll(&db_path, after, MAX_PER_PASS, live) {
                Ok(found) => found,
                Err(err) => {
                    tracing::debug!(error = %err, "notifications: could not read the event log");
                    continue;
                }
            };
            if found.is_empty() {
                // Nothing to show, but the log may still have advanced past
                // events this person filtered out - or past everything, while
                // notifications were off. Move the cursor to the newest so the
                // next poll is not re-reading them forever, and so switching a
                // category on does not arrive with a backlog.
                if let Ok(newest) = notifications::latest_event_id(&db_path) {
                    if newest > after {
                        cursor.store(newest, Ordering::Relaxed);
                        write_cursor(newest);
                    }
                }
                continue;
            }

            let mut highest = after;
            for item in &found {
                if let Err(err) = app
                    .notification()
                    .builder()
                    .title(&item.title)
                    .body(&item.body)
                    .show()
                {
                    tracing::warn!(error = %err, "notifications: could not show a toast");
                }
                if item.id > highest {
                    highest = item.id;
                }
            }
            cursor.store(highest, Ordering::Relaxed);
            write_cursor(highest);
        }
    });
}
