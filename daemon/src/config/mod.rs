//! Local configuration and credentials under `~/.reincore-io`.
//!
//! Everything the engine persists about this machine: `config.toml`
//! management, the provider credential store, MCP server configuration,
//! and OAuth sign-in flows for local MCP servers.

pub mod credentials;
pub mod local_config;
pub mod mcp_config;
pub mod mcp_credentials;
pub mod mcp_sign_in;
