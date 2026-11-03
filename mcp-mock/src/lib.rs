//! Mock MCP HTTP server shared by the engine integration tests.
//!
//! Engine crates consume this library through dev-dependencies only.

mod mcp;

pub use mcp::McpFixture;
