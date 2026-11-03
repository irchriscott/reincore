use crate::app::{AppState, SingleProviderAppState};
use crate::db::LocalDatabase;
use axum::{
    body::to_bytes,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
};
use colored::Colorize;
use engine_core::{
    derive_task_summary, find_model, improved_task_summary, inject_system_nudge, is_self_contained,
    parse_embedding, prepare_context, resolve_budget_routing, CachePrefixNormalizer, CacheStore,
    ChatCompletionRequest, ChatMessage, CircuitBreaker, EventStore, FrameworkCheckpoint,
    FrameworkMemorySnapshot, Session, SessionEventType, SessionHandoff, SessionMode, SessionStatus,
    SettingsStore,
};
use futures::{stream::Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::convert::Infallible;
use std::sync::OnceLock;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::warn;
use uuid::Uuid;

mod base;
mod batches;
mod budget_admin;
mod chat;
mod compatibility;
mod config_admin;
mod framework;
mod lifecycle;
mod logging;
mod mcp;
mod provider_admin;
mod request_group;
mod sessions;
mod titles;
mod tool_loop;
mod tools;

pub use base::*;
pub use batches::*;
pub use budget_admin::*;
pub use chat::*;
pub use compatibility::*;
pub use config_admin::*;
pub use framework::*;
pub use logging::print_cumulative_cost;
pub use mcp::{mcp, mcp_single_provider};
pub use provider_admin::*;
pub use sessions::*;
pub use tools::*;

#[cfg(test)]
mod tests;
