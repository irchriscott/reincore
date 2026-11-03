// ReinCore engine - library entry point.
//
// The local engine ships as both a binary (the `reincore` / `rc` CLI) and a
// library. The binary is the primary entry point for command-line users; the
// library is what the ReinCore desktop app (Tauri) links against so it can
// run the daemon in-process instead of spawning a separate binary.
//
// Every module is re-exported as `pub` here so the binary (in `main.rs`) and
// the desktop app reach the exact same code through one source tree. The
// CLI's behavior is unchanged by this split - `main.rs` simply imports these
// symbols from `reincore::*` instead of private `mod` declarations.

// Feature modules. Each directory groups one daemon concern; the
// aliases below preserve every pre-existing `reincore::<module>` path,
// so moving a module into a group never changes how callers reach it.
pub mod commands;
pub mod config;
pub mod runtime;

pub mod cli;
pub mod db;
pub mod http;

// Module aliases: `reincore::<module>` keeps resolving exactly as before.
pub use commands::{budget, config_cmd, provider, session, telemetry_cmd};
pub use config::{credentials, local_config, mcp_config, mcp_credentials, mcp_sign_in};
pub use db::migrations;
pub use runtime::{app, daemon, launch, launch_routing, notifications};
