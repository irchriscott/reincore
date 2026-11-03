//! CLI command handlers.
//!
//! One module per command family: each handler takes parsed CLI arguments,
//! talks to the database or local configuration, and prints human-readable
//! output. Shared building blocks the daemon also needs (provider registry,
//! proxy startup) live alongside the `provider` command that owns them.

pub mod budget;
pub mod config_cmd;
pub mod provider;
pub mod session;
pub mod telemetry_cmd;
