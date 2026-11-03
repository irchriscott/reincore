//! Shared terminal logging helpers for ReinCore.
//!
//! Provides colored, timestamped log line formatting used by the
//! local daemon and the provider adapters (e.g. Copilot OAuth flow).
//!
//! Format: `[HH:MM:SS] [provider] message`
//! - Timestamp: dimmed
//! - Provider tag: magenta + bold
//! - Message: caller-colored (the caller passes a pre-colored string)

use colored::Colorize;

/// Format a timestamp as `HH:MM:SS` (local time) for log line prefixes.
pub fn timestamp() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}

/// Format a provider-tagged log line: `[time] [provider] message`.
///
/// The provider tag is colored magenta + bold, the timestamp is dimmed.
/// The `message` string is used as-is (the caller is responsible for
/// coloring individual parts of it).
pub fn provider_line(provider: &str, message: &str) -> String {
    format!(
        "{}  {} {}",
        timestamp().dimmed(),
        format!("[{}]", provider).magenta().bold(),
        message,
    )
}

/// Format a general (non-provider) log line: `[time] message`.
///
/// The timestamp is dimmed. The `message` string is used as-is.
pub fn line(message: &str) -> String {
    format!("{}  {}", timestamp().dimmed(), message)
}

/// Format a success message with a green checkmark prefix.
pub fn success(message: &str) -> String {
    format!(
        "{}  {}",
        timestamp().dimmed(),
        format!("✓ {}", message).green()
    )
}

/// Format an error message with a red cross prefix.
pub fn error(message: &str) -> String {
    format!(
        "{}  {}",
        timestamp().dimmed(),
        format!("✗ {}", message).red().bold()
    )
}

/// Format a warning message with a yellow exclamation prefix.
pub fn warning(message: &str) -> String {
    format!(
        "{}  {}",
        timestamp().dimmed(),
        format!("⚠ {}", message).yellow().bold(),
    )
}

/// Format an info message with a blue arrow prefix.
pub fn info(message: &str) -> String {
    format!(
        "{}  {}",
        timestamp().dimmed(),
        format!("→ {}", message).blue(),
    )
}
