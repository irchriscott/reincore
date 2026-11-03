//! Framework adapters for external agent orchestrators.
//!
//! This module exposes ReinCore's resumable checkpoints and tiered memory
//! manager in formats that LangGraph, CrewAI, AutoGen, and similar frameworks
//! can consume without knowing ReinCore internals.
//!
//! # Concepts
//!
//! - `FrameworkCheckpoint`: a serializable checkpoint containing the session
//!   state, the last event sequence number, and a resumable token. Frameworks
//!   can store this object as their graph state and later pass the token back
//!   to resume execution from the same point.
//! - `FrameworkMemorySnapshot`: a framework-friendly view of the three memory
//!   tiers (working, episodic, tool-state) produced by `MemoryManager`.
//! - `FrameworkAdapter`: a thin wrapper that converts between engine-core types
//!   and framework types, and can optionally rehydrate a `Session` from a
//!   stored checkpoint.

use crate::checkpoint::{CheckpointManager, ResumableToken};
use crate::context::{MemoryManager, MemoryTier};
use crate::error::EngineResult;
use crate::event::{EventStore, SessionEvent, SessionEventType};
use crate::models::{ChatMessage, FileContextAttachment, MessageContent};
use crate::session::{Session, SessionStatus};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A framework-consumable checkpoint.
///
/// Store this object in your graph state (LangGraph `State`, CrewAI
/// `Memory`, etc.) and call `reincore` with `checkpoint_token` to resume.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameworkCheckpoint {
    pub version: u32,
    pub checkpoint_token: String,
    pub session_id: String,
    pub session_status: String,
    pub task_summary: String,
    pub project_path: Option<String>,
    pub budget_cents: i64,
    pub spend_cents: i64,
    #[serde(default)]
    pub spend_nano_usd: Option<i64>,
    #[serde(default)]
    pub unknown_cost_calls: u64,
    pub tier: u8,
    pub last_event_seq: i64,
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

/// A framework-consumable memory snapshot.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FrameworkMemorySnapshot {
    pub version: u32,
    pub session_id: String,
    pub working_messages: Vec<ChatMessage>,
    pub episodic_summary: Option<String>,
    pub tool_state_messages: Vec<ChatMessage>,
    pub attached_files: Vec<FileContextAttachment>,
    #[serde(default)]
    pub provenance: Vec<crate::ContentProvenance>,
    pub captured_at: DateTime<Utc>,
}

/// A single memory entry for frameworks that prefer a flat list.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameworkMemoryEntry {
    pub tier: String,
    #[serde(default)]
    pub sources: Vec<crate::ContentSource>,
    #[serde(default)]
    pub trust: crate::ContentTrust,
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<crate::models::ToolCall>>,
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
}

/// Adapter configuration.
#[derive(Clone, Debug)]
pub struct FrameworkAdapterConfig {
    /// How many recent events to include in the working window when
    /// frameworks ask for a flat memory list.
    pub working_window_size: usize,
    /// When the event log exceeds this length, emit a placeholder episodic
    /// summary if none exists yet.
    pub summary_threshold: usize,
}

impl Default for FrameworkAdapterConfig {
    fn default() -> Self {
        Self {
            working_window_size: 6,
            summary_threshold: 12,
        }
    }
}

/// Framework adapter state.
#[derive(Clone, Debug)]
pub struct FrameworkAdapter {
    #[allow(dead_code)]
    config: FrameworkAdapterConfig,
    checkpoint_manager: CheckpointManager,
    memory_manager: MemoryManager,
}

impl FrameworkAdapter {
    /// Create a new adapter with the given checkpoint manager and default
    /// memory configuration.
    pub fn new(checkpoint_manager: CheckpointManager) -> Self {
        Self::with_config(checkpoint_manager, FrameworkAdapterConfig::default())
    }

    /// Create a new adapter with explicit memory configuration.
    pub fn with_config(
        checkpoint_manager: CheckpointManager,
        config: FrameworkAdapterConfig,
    ) -> Self {
        let memory_manager =
            MemoryManager::new(config.working_window_size, config.summary_threshold);
        Self {
            config,
            checkpoint_manager,
            memory_manager,
        }
    }

    /// Attach a RAG tool-state store, enabling the third memory tier so
    /// `retrieved_tool_states` is populated from past `ToolResult` events
    /// semantically similar to the current turn. See
    /// [`MemoryManager::with_tool_state_store`].
    pub fn with_tool_state_store(
        mut self,
        store: std::sync::Arc<dyn crate::store::ToolStateStore>,
        config: crate::context::ToolStateRetrievalConfig,
    ) -> Self {
        self.memory_manager = self.memory_manager.with_tool_state_store(store, config);
        self
    }

    /// Pause a session and produce a framework checkpoint.
    ///
    /// The session is marked `Checkpointed` and a signed token is returned.
    pub async fn checkpoint<E: EventStore>(
        &self,
        session: &mut Session,
        reason: impl Into<String>,
        event_store: &E,
        metadata: Option<serde_json::Value>,
    ) -> EngineResult<FrameworkCheckpoint> {
        let latest_seq = event_store.latest_seq(session.id).await?.unwrap_or(0);
        let token = self
            .checkpoint_manager
            .create(session, latest_seq, reason.into())?;

        Ok(FrameworkCheckpoint {
            version: 1,
            checkpoint_token: token.token.clone(),
            session_id: session.id.to_string(),
            session_status: session.status.as_str().to_string(),
            task_summary: session.task_summary.clone(),
            project_path: session.project_path.clone(),
            budget_cents: session.budget_cents,
            spend_cents: session.spend_cents,
            spend_nano_usd: session.spend_nano_usd,
            unknown_cost_calls: session.unknown_cost_calls,
            tier: session.tier,
            last_event_seq: token.last_event_seq,
            expires_at: Some(token.expires_at),
            created_at: Utc::now(),
            metadata: metadata.unwrap_or(serde_json::Value::Object(Default::default())),
        })
    }

    /// Validate a checkpoint token and return the decoded token.
    pub fn validate_token(&self, token: &str) -> EngineResult<ResumableToken> {
        self.checkpoint_manager.validate(token)
    }

    /// Resume a session from a framework checkpoint.
    ///
    /// The caller must supply the current session object; the adapter verifies
    /// the token, checks the event log reaches the checkpointed sequence,
    /// and transitions the session back to `Active`.
    pub async fn resume<E: EventStore>(
        &self,
        checkpoint: &FrameworkCheckpoint,
        session: &mut Session,
        event_store: &E,
    ) -> EngineResult<()> {
        let token = self
            .checkpoint_manager
            .validate(&checkpoint.checkpoint_token)?;
        self.checkpoint_manager
            .resume(&token, session, event_store)
            .await
    }

    /// Resume a session from a raw checkpoint token.
    pub async fn resume_from_token<E: EventStore>(
        &self,
        token: &str,
        session: &mut Session,
        event_store: &E,
    ) -> EngineResult<()> {
        let token = self.checkpoint_manager.validate(token)?;
        self.checkpoint_manager
            .resume(&token, session, event_store)
            .await
    }

    /// Build a memory snapshot for a session from its event log.
    pub async fn memory_snapshot<E: EventStore>(
        &self,
        session_id: Uuid,
        system_prompt: Option<&str>,
        event_store: &E,
    ) -> EngineResult<FrameworkMemorySnapshot> {
        let ctx = self
            .memory_manager
            .assemble(event_store, session_id, system_prompt)
            .await?;

        Ok(FrameworkMemorySnapshot {
            version: 1,
            session_id: session_id.to_string(),
            working_messages: ctx.working_messages,
            episodic_summary: ctx.episodic_summary,
            tool_state_messages: ctx.retrieved_tool_states,
            attached_files: ctx.attached_files,
            provenance: ctx.provenance,
            captured_at: Utc::now(),
        })
    }

    /// Build a flat list of memory entries for frameworks that do not use
    /// tiered message lists.
    pub async fn memory_entries<E: EventStore>(
        &self,
        session_id: Uuid,
        system_prompt: Option<&str>,
        event_store: &E,
    ) -> EngineResult<Vec<FrameworkMemoryEntry>> {
        let snapshot = self
            .memory_snapshot(session_id, system_prompt, event_store)
            .await?;
        let mut entries = Vec::new();

        for msg in &snapshot.working_messages {
            let sources = crate::message_content_sources(msg);
            let trust = aggregate_trust(&sources);
            entries.push(FrameworkMemoryEntry {
                tier: MemoryTier::Working.as_str().to_string(),
                sources,
                trust,
                role: msg.role.clone(),
                content: msg.content_text(),
                tool_calls: msg.tool_calls.clone(),
                tool_call_id: msg.tool_call_id.clone(),
                name: msg.name.clone(),
            });
        }

        if let Some(summary) = &snapshot.episodic_summary {
            let source = snapshot
                .provenance
                .iter()
                .find(|item| item.location == "episodic_summary")
                .map(|item| item.source)
                .unwrap_or(crate::ContentSource::CompactedContext);
            entries.push(FrameworkMemoryEntry {
                tier: MemoryTier::Episodic.as_str().to_string(),
                sources: vec![source],
                trust: source.trust(),
                role: "system".to_string(),
                content: Some(summary.clone()),
                tool_calls: None,
                tool_call_id: None,
                name: Some("episodic_summary".to_string()),
            });
        }

        for msg in &snapshot.tool_state_messages {
            let sources = crate::message_content_sources(msg);
            let trust = aggregate_trust(&sources);
            entries.push(FrameworkMemoryEntry {
                tier: MemoryTier::ToolState.as_str().to_string(),
                sources,
                trust,
                role: msg.role.clone(),
                content: msg.content_text(),
                tool_calls: msg.tool_calls.clone(),
                tool_call_id: msg.tool_call_id.clone(),
                name: msg.name.clone(),
            });
        }

        Ok(entries)
    }

    /// Append a framework-provided message to the session event log as an
    /// `LlmCall` event. This lets frameworks push their own turns into
    /// ReinCore memory without going through the chat-completion endpoint.
    pub async fn record_framework_turn<E: EventStore>(
        &self,
        session_id: Uuid,
        role: &str,
        content: &str,
        event_store: &E,
    ) -> EngineResult<SessionEvent> {
        let payload = serde_json::json!({
            "response_message": ChatMessage {
                provider_state: None,
                role: role.to_string(),
                content: Some(MessageContent::text(content)),
                tool_calls: None,
                tool_call_id: None,
                name: Some("framework".to_string()),
            }
        });
        event_store
            .append_event(session_id, SessionEventType::LlmCall, payload, None)
            .await
    }
}

fn aggregate_trust(sources: &[crate::ContentSource]) -> crate::ContentTrust {
    if sources
        .iter()
        .any(|source| source.trust() == crate::ContentTrust::UntrustedData)
    {
        crate::ContentTrust::UntrustedData
    } else {
        sources
            .first()
            .copied()
            .unwrap_or(crate::ContentSource::Unknown)
            .trust()
    }
}

impl MemoryTier {
    fn as_str(&self) -> &'static str {
        match self {
            MemoryTier::Working => "working",
            MemoryTier::Episodic => "episodic",
            MemoryTier::ToolState => "tool_state",
        }
    }
}

impl SessionStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            SessionStatus::Active => "active",
            SessionStatus::Paused => "paused",
            SessionStatus::Completed => "completed",
            SessionStatus::Checkpointed => "checkpointed",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::in_memory_stores::InMemoryEventStore;
    use crate::models::{ChatMessage, MessageContent};

    #[tokio::test]
    async fn checkpoint_round_trip() {
        let manager = CheckpointManager::new("test-secret", 60);
        let adapter = FrameworkAdapter::new(manager);
        let mut session = Session::new("framework test", 1000).unwrap();
        let store = InMemoryEventStore::default();

        store
            .append_event(
                session.id,
                SessionEventType::LlmCall,
                serde_json::json!({
                    "response_message": ChatMessage {
                        provider_state: None,
                        role: "assistant".to_string(),
                        content: Some(crate::MessageContent::text("hello".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    }
                }),
                Some(1),
            )
            .await
            .unwrap();

        let checkpoint = adapter
            .checkpoint(&mut session, "test-checkpoint", &store, None)
            .await
            .unwrap();

        assert_eq!(checkpoint.session_id, session.id.to_string());
        assert!(!checkpoint.checkpoint_token.is_empty());
        assert_eq!(session.status, SessionStatus::Checkpointed);

        adapter
            .resume(&checkpoint, &mut session, &store)
            .await
            .unwrap();
        assert_eq!(session.status, SessionStatus::Active);
    }

    #[tokio::test]
    async fn memory_snapshot_has_messages() {
        let manager = CheckpointManager::new("test-secret", 60);
        let adapter = FrameworkAdapter::new(manager);
        let session = Session::new("framework test", 1000).unwrap();
        let store = InMemoryEventStore::default();

        store
            .append_event(
                session.id,
                SessionEventType::LlmCall,
                serde_json::json!({
                    "response_message": ChatMessage {
                        provider_state: None,
                        role: "assistant".to_string(),
                        content: Some(crate::MessageContent::text("first".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    }
                }),
                Some(1),
            )
            .await
            .unwrap();

        let snapshot = adapter
            .memory_snapshot(session.id, Some("You are helpful"), &store)
            .await
            .unwrap();

        assert!(!snapshot.working_messages.is_empty());
        assert!(!snapshot.provenance.is_empty());
        assert_eq!(snapshot.session_id, session.id.to_string());
    }

    /// Integration test simulating a LangGraph-style multi-node workflow:
    ///
    /// 1. Create a session and record a "research" node turn.
    /// 2. Checkpoint the session (simulating a node boundary).
    /// 3. Record a "synthesize" node turn.
    /// 4. Checkpoint again.
    /// 5. Verify the memory snapshot reflects both turns.
    /// 6. Resume from the first checkpoint.
    /// 7. Verify the session is active again.
    /// 8. Record a final "respond" turn.
    /// 9. Verify the final memory snapshot includes all three turns.
    ///
    /// This mirrors the exact flow the `ReinCoreCheckpointSaver` SDK
    /// adapter uses when running a LangGraph through ReinCore.
    #[tokio::test]
    async fn langgraph_style_multi_node_checkpoint_flow() {
        let manager = CheckpointManager::new("test-secret", 60);
        let adapter = FrameworkAdapter::new(manager);
        let mut session = Session::new("langgraph demo", 1000).unwrap();
        let store = InMemoryEventStore::default();

        // Helper: record a node turn (like framework_record_turn).
        async fn record_turn(
            adapter: &FrameworkAdapter,
            session_id: Uuid,
            role: &str,
            content: &str,
            store: &InMemoryEventStore,
        ) {
            adapter
                .record_framework_turn(session_id, role, content, store)
                .await
                .unwrap();
        }

        // 1. Research node - record the turn.
        record_turn(
            &adapter,
            session.id,
            "assistant",
            "Researched: Rust. Found 3 sources.",
            &store,
        )
        .await;

        // 2. Checkpoint after the research node.
        let cp1 = adapter
            .checkpoint(
                &mut session,
                "langgraph:research",
                &store,
                Some(serde_json::json!({"node": "research", "checkpoint_id": "cp-1"})),
            )
            .await
            .unwrap();
        assert_eq!(session.status, SessionStatus::Checkpointed);
        assert_eq!(cp1.metadata["node"], "research");
        assert!(cp1.checkpoint_token.contains(&session.id.to_string()));

        // 3. Resume from cp1 to continue to the synthesize node.
        adapter.resume(&cp1, &mut session, &store).await.unwrap();
        assert_eq!(session.status, SessionStatus::Active);

        // 4. Synthesize node - record the turn.
        record_turn(
            &adapter,
            session.id,
            "assistant",
            "Synthesized: Rust is well-supported.",
            &store,
        )
        .await;

        // 5. Checkpoint after the synthesize node.
        let cp2 = adapter
            .checkpoint(
                &mut session,
                "langgraph:synthesize",
                &store,
                Some(serde_json::json!({"node": "synthesize", "checkpoint_id": "cp-2"})),
            )
            .await
            .unwrap();
        assert_eq!(cp2.metadata["node"], "synthesize");
        // cp2 should have a higher seq than cp1.
        assert!(cp2.last_event_seq > cp1.last_event_seq);

        // 6. Verify the memory snapshot reflects both turns.
        let snapshot = adapter
            .memory_snapshot(session.id, Some("You are a research assistant"), &store)
            .await
            .unwrap();
        assert!(!snapshot.working_messages.is_empty());
        // The working window should contain at least one of the recorded turns.
        let working_content: Vec<String> = snapshot
            .working_messages
            .iter()
            .filter_map(|m| m.content_text())
            .collect();
        assert!(
            working_content.iter().any(|c| c.contains("Synthesized")),
            "working messages should include the synthesize turn: {working_content:?}"
        );

        // 7. Resume from cp2 (the synthesize checkpoint) to simulate
        //    re-entering the graph at that node. Note: cp1 was already
        //    redeemed in step 3 - checkpoint tokens are single-use, so
        //    we must resume from the latest unredeemed checkpoint.
        adapter.resume(&cp2, &mut session, &store).await.unwrap();
        assert_eq!(session.status, SessionStatus::Active);

        // 8. Record a final "respond" turn.
        record_turn(
            &adapter,
            session.id,
            "assistant",
            "Final answer: Rust is a great choice.",
            &store,
        )
        .await;

        // 9. Verify the final memory snapshot includes the respond turn.
        let final_snapshot = adapter
            .memory_snapshot(session.id, None, &store)
            .await
            .unwrap();
        let final_content: Vec<String> = final_snapshot
            .working_messages
            .iter()
            .filter_map(|m| m.content_text())
            .collect();
        assert!(
            final_content.iter().any(|c| c.contains("Final answer")),
            "final working messages should include the respond turn: {final_content:?}"
        );

        // 10. Verify the session has accumulated events.
        let events = store.read_events(session.id, None, None).await.unwrap();
        assert!(
            events.len() >= 3,
            "should have at least 3 events (3 turns), got {}",
            events.len()
        );

        // 11. Verify checkpoint metadata is preserved.
        assert_eq!(cp1.metadata["node"], "research");
        assert_eq!(cp2.metadata["node"], "synthesize");
        assert_eq!(cp1.metadata["checkpoint_id"], "cp-1");
        assert_eq!(cp2.metadata["checkpoint_id"], "cp-2");
    }

    /// Test that a checkpoint with rich metadata (simulating a LangGraph
    /// checkpoint dict embedded in metadata) round-trips correctly.
    #[tokio::test]
    async fn checkpoint_preserves_rich_langgraph_metadata() {
        let manager = CheckpointManager::new("test-secret", 60);
        let adapter = FrameworkAdapter::new(manager);
        let mut session = Session::new("metadata test", 1000).unwrap();
        let store = InMemoryEventStore::default();

        store
            .append_event(
                session.id,
                SessionEventType::LlmCall,
                serde_json::json!({"response_message": ChatMessage {
                    provider_state: None,
                    role: "assistant".to_string(),
                    content: Some(crate::MessageContent::text("step 1".to_string())),
                    tool_calls: None, tool_call_id: None, name: None,
                }}),
                Some(1),
            )
            .await
            .unwrap();

        // Simulate the metadata shape the LangGraph adapter sends:
        // { "langgraph_checkpoint": {...}, "langgraph_metadata": {...} }
        let meta = serde_json::json!({
            "langgraph_checkpoint": {
                "v": 1,
                "id": "cp-1",
                "channel_values": {
                    "messages": [{"role": "user", "content": "hello"}],
                    "node": "research"
                }
            },
            "langgraph_metadata": {"step": 1, "source": "loop"},
            "langgraph_new_versions": {"messages": 2}
        });

        let checkpoint = adapter
            .checkpoint(
                &mut session,
                "langgraph-superstep",
                &store,
                Some(meta.clone()),
            )
            .await
            .unwrap();

        // The metadata should be preserved verbatim.
        assert_eq!(checkpoint.metadata, meta);
        assert_eq!(checkpoint.metadata["langgraph_checkpoint"]["id"], "cp-1");
        assert_eq!(checkpoint.metadata["langgraph_metadata"]["step"], 1);
    }

    /// Test that `memory_entries` (the flat list format) correctly
    /// partitions messages by tier.
    #[tokio::test]
    async fn memory_entries_partition_by_tier() {
        let manager = CheckpointManager::new("test-secret", 60);
        let adapter = FrameworkAdapter::new(manager);
        let session = Session::new("entries test", 1000).unwrap();
        let store = InMemoryEventStore::default();

        // Seed a few LLM call events.
        for i in 0..3 {
            store
                .append_event(
                    session.id,
                    SessionEventType::LlmCall,
                    serde_json::json!({
                        "response_message": ChatMessage {
                            provider_state: None,
                            role: "assistant".to_string(),
                            content: Some(MessageContent::text(format!("turn {i}"))),
                            tool_calls: None, tool_call_id: None, name: None,
                        }
                    }),
                    Some(i + 1),
                )
                .await
                .unwrap();
        }

        let entries = adapter
            .memory_entries(session.id, Some("You are helpful"), &store)
            .await
            .unwrap();

        // All entries should have a tier field.
        assert!(!entries.is_empty());
        for entry in &entries {
            assert!(!entry.tier.is_empty());
            assert!(!entry.sources.is_empty());
            assert_ne!(entry.trust, crate::ContentTrust::Unknown);
            // Tier should be one of the known values.
            assert!(
                entry.tier == "working" || entry.tier == "episodic" || entry.tier == "tool_state",
                "unexpected tier: {}",
                entry.tier
            );
        }

        // At least some entries should be in the "working" tier.
        let working_count = entries.iter().filter(|e| e.tier == "working").count();
        assert!(working_count > 0, "should have working-tier entries");
    }
}
