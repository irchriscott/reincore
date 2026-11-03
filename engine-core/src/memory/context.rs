//! Three-tier stateful context manager (working / episodic / tool-state).
//!
//! [`MemoryManager`] assembles the prompt for each LLM turn from three tiers:
//!
//! - **Working** - system prompt, original goal, and the last few turns,
//!   sent verbatim.
//! - **Episodic** - older turns auto-summarized into a compact paragraph.
//! - **Tool-state** - raw tool/API results retrieved via RAG using an embedding
//!   of the current turn, so large outputs stay out of the context window
//!   until they are actually relevant.
//!
//! In the cloud edition the tool-state tier is team-scoped: results cached
//! by one member are retrievable by teammates working on similar subtasks.

use crate::error::EngineResult;
use crate::event::{EventStore, SessionEvent, SessionEventType};
use crate::models::{ChatMessage, FileContextAttachment, MessageContent, ToolResult};
use crate::store::ToolStateStore;
use crate::{ContentProvenance, ContentSource};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Memory tiers for the stateful context manager.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTier {
    /// System prompt, original goal, last 2-3 steps (verbatim, always sent).
    Working,
    /// Older steps auto-summarized into a compact paragraph.
    Episodic,
    /// Raw tool/API results stored externally, retrievable via RAG lookup.
    ToolState,
}

/// Assembled context for a single LLM turn.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AssembledContext {
    pub working_messages: Vec<ChatMessage>,
    pub episodic_summary: Option<String>,
    /// RAG-retrieved `ToolResult` messages from earlier in the session
    /// (or, for the cloud edition, across the team) that are semantically
    /// similar to the current turn. Populated by querying the optional
    /// [`ToolStateStore`] with an embedding of the current turn. When no
    /// store is configured this is empty (the 3-tier memory still works,
    /// just without the RAG tier).
    pub retrieved_tool_states: Vec<ChatMessage>,
    /// Latest versions of files referenced in the context. Older versions
    /// are discarded during assembly so the LLM only sees current state.
    /// Sorted by path for deterministic output across identical calls.
    pub attached_files: Vec<FileContextAttachment>,
    /// Origin/trust sidecar for every assembled item. This remains separate
    /// from provider roles so compaction and framework adapters cannot turn
    /// retrieved data into trusted instructions accidentally.
    #[serde(default)]
    pub provenance: Vec<ContentProvenance>,
}

/// Configuration for the RAG tool-state tier.
#[derive(Clone, Debug)]
pub struct ToolStateRetrievalConfig {
    /// Number of past `ToolResult` messages to retrieve per turn.
    pub top_k: usize,
    /// Minimum cosine similarity to include a result.
    pub min_similarity: f32,
}

impl Default for ToolStateRetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 4,
            min_similarity: 0.35,
        }
    }
}

/// Stateful context manager / memory manager.
#[derive(Clone)]
pub struct MemoryManager {
    working_window_size: usize,
    summary_threshold: usize,
    /// Optional RAG store backing the `ToolState` memory tier. When
    /// `None`, `retrieved_tool_states` is left empty and the 3-tier
    /// memory degrades gracefully to a 2-tier working+episodic model.
    tool_state_store: Option<std::sync::Arc<dyn ToolStateStore>>,
    tool_state_config: ToolStateRetrievalConfig,
}

impl std::fmt::Debug for MemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryManager")
            .field("working_window_size", &self.working_window_size)
            .field("summary_threshold", &self.summary_threshold)
            .field(
                "tool_state_store",
                &self.tool_state_store.as_ref().map(|_| "<ToolStateStore>"),
            )
            .field("tool_state_config", &self.tool_state_config)
            .finish()
    }
}

impl MemoryManager {
    pub fn new(working_window_size: usize, summary_threshold: usize) -> Self {
        Self {
            working_window_size,
            summary_threshold,
            tool_state_store: None,
            tool_state_config: ToolStateRetrievalConfig::default(),
        }
    }

    /// Attach a RAG tool-state store and retrieval config, enabling the
    /// third memory tier. The store indexes `ToolResult` events as they
    /// are observed during assembly and returns the top-k most similar
    /// past results for the current turn's embedding.
    pub fn with_tool_state_store(
        mut self,
        store: std::sync::Arc<dyn ToolStateStore>,
        mut config: ToolStateRetrievalConfig,
    ) -> Self {
        config.top_k = config.top_k.min(8);
        config.min_similarity = if config.min_similarity.is_finite() {
            config.min_similarity.clamp(-1.0, 1.0)
        } else {
            ToolStateRetrievalConfig::default().min_similarity
        };
        self.tool_state_store = Some(store);
        self.tool_state_config = config;
        self
    }

    pub fn default_config() -> Self {
        Self::new(3, 8)
    }

    /// Assemble the context for the current turn by querying the event log.
    ///
    /// This uses **bounded reads** so per-turn assembly cost stays constant
    /// regardless of total session length, fixing the O(n²) scan that arose
    /// when the entire event log was read on every turn.
    ///
    /// The strategy:
    /// 1. Fetch the latest event seq and the most recent `Summary` event via
    ///    two cheap, index-backed queries (`latest_seq` /
    ///    `read_latest_summary`). The summary's `summarized_seq_end` is the
    ///    cursor past which events are still "live" (older events are already
    ///    folded into the episodic summary).
    /// 2. Read only the recent tail - from `max(summarized_seq_end,
    ///    latest_seq - window + 1)` onward, capped at a small `limit` - for
    ///    the working window and tool-state indexing of new results.
    /// 3. Fetch attached files via `read_latest_files` (server-side
    ///    deduplication by path), avoiding a full-log scan for citation
    ///    grounding.
    ///
    /// Tool-state indexing is bounded to the same tail: events already
    /// covered by a summary have been folded into the episodic tier, so
    /// re-indexing them every turn was pure waste. The `ToolStateStore` is
    /// idempotent (upsert by `(session_id, seq)`), so the first turn after a
    /// summary indexes only the post-summary tail; older tool results remain
    /// retrievable only if they were indexed in a prior turn. This is the
    /// intended tradeoff - the RAG tier is a recency-biased augmentation of
    /// the working window, not a full-history search.
    pub async fn assemble<E: EventStore>(
        &self,
        event_store: &E,
        session_id: Uuid,
        system_prompt: Option<&str>,
    ) -> EngineResult<AssembledContext> {
        // These independent index-backed lookups share one database round-trip
        // instead of serially extending completion preflight latency.
        let (latest_seq, latest_summary) = tokio::try_join!(
            event_store.latest_seq(session_id),
            event_store.read_latest_summary(session_id)
        )?;
        let latest_seq = latest_seq.unwrap_or(0);

        // Cursor: events at or before this seq are already summarized into
        // the episodic tier. The working window only needs events *after*
        // this point. When there's no summary, start from the beginning
        // (seq 1) but the bounded `limit` below still caps the read.
        let summarized_up_to = latest_summary
            .as_ref()
            .map(|s| s.summarized_seq_end)
            .unwrap_or(0);

        // Bounded tail read. The working window counts *conversational
        // messages* (each event expands to 1-2 messages), so read a generous
        // multiple of the window to guarantee we recover at least
        // `working_window_size` messages. `max(1, ...)` keeps the limit
        // positive for tiny windows, and the `saturating_*` math keeps
        // `since_seq` within the live range.
        const MAX_TAIL_EVENTS: usize = 4096;
        let read_limit = self
            .working_window_size
            .saturating_mul(3)
            .clamp(1, MAX_TAIL_EVENTS);
        let since_seq = if latest_seq > 0 {
            // Start far enough back to capture the whole working window,
            // but never earlier than the summary cursor (older events are
            // already in the episodic summary and would be redundant here).
            let tail_start = latest_seq.saturating_sub(read_limit as i64 - 1).max(1);
            tail_start.max(summarized_up_to.saturating_add(1))
        } else {
            1
        };

        // Read only the bounded tail. `since_seq`/`limit` paginate forward
        // (oldest N matching events), which is what we want here: the tail
        // slice in ascending seq order, ready for `event_to_messages`.
        let read_tail = async {
            if latest_seq == 0 {
                Ok(Vec::new())
            } else {
                event_store
                    .read_events(session_id, Some(since_seq), Some(read_limit))
                    .await
            }
        };
        let (events, mut attached_files) =
            tokio::try_join!(read_tail, event_store.read_latest_files(session_id))?;

        let mut working_messages = Vec::new();
        if let Some(system) = system_prompt {
            working_messages.push(ChatMessage {
                provider_state: None,
                role: "system".to_string(),
                content: Some(MessageContent::text(system)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }

        // The event log is a complete audit trail, but provider context is
        // grouped by user request. Completed groups contribute only their
        // normalized request and final outcome; an unfinished tool loop keeps
        // its complete assistant/tool protocol so it can safely continue.
        let conversational = crate::conversation::project_session_events(&events);
        let recent_start = conversational
            .len()
            .saturating_sub(self.working_window_size);
        working_messages.extend_from_slice(&conversational[recent_start..]);

        // Episodic summary: prefer the real summary text from the targeted
        // lookup; fall back to a placeholder when the session has outgrown
        // the working window without any summary yet. The placeholder count
        // is derived from `latest_seq` (total events) rather than the
        // bounded slice's length, so it reflects how much history is
        // truncated, not how much we happened to read.
        let episodic_summary = latest_summary
            .as_ref()
            .map(|s| s.summary_text.clone())
            .or_else(|| {
                // Only emit a placeholder if there are older, un-summarized
                // events beyond the working window. The count has two parts:
                //   - `since_seq - 1`: events before the bounded tail read
                //     (≈ one conversational message each for LLM-call events)
                //   - `recent_start`: messages in the tail but before the
                //     working window
                // Together these approximate `total_conversational_messages
                // - working_window_size`, matching the pre-bounded-read
                // semantics without scanning the full log.
                let older_count = (since_seq - 1).max(0) + recent_start as i64;
                if older_count > self.summary_threshold as i64 {
                    Some(format!(
                        "[{older_count} older events truncated; async summarization pending]"
                    ))
                } else {
                    None
                }
            });

        // Attached files: server-side latest-version-per-path lookup. This
        // replaces the previous full-log scan + in-Rust dedup, which was the
        // other O(n) per-turn cost. Sorted by path for deterministic output.
        attached_files.sort_by(|a, b| a.path.cmp(&b.path));

        let paths = crate::file_history::requested_paths(&working_messages, &attached_files);
        let changes = event_store.read_file_history(session_id, &paths).await?;
        // Attached events are historical snapshots, not fresh version reports.
        if let Some(message) = crate::file_history::context_message(&changes, &[]) {
            let at = usize::from(system_prompt.is_some());
            working_messages.insert(at, message);
        }

        // RAG tool-state tier: index only the `ToolResult` events in the
        // bounded tail (new results since the last summary), then retrieve
        // the top-k past results whose embeddings are most similar to the
        // current turn's embedding. The `ToolStateStore` upsert is
        // idempotent by `(session_id, seq)`, so re-indexing the tail on
        // every turn is safe. When no store is configured this tier is
        // skipped and `retrieved_tool_states` stays empty.
        let retrieved_tool_states = match &self.tool_state_store {
            Some(store) => {
                let tool_results: Vec<(&SessionEvent, ToolResult)> = events
                    .iter()
                    .filter(|e| e.event_type == SessionEventType::ToolResult)
                    .filter_map(|e| {
                        e.payload
                            .get("result")
                            .and_then(|v| serde_json::from_value::<ToolResult>(v.clone()).ok())
                            .map(|tr| (e, tr))
                    })
                    .collect();

                // Index each tool result keyed by the embedding of the turn
                // that triggered it. The triggering turn is the most recent
                // user message (or assistant tool-call name, if the tool
                // result follows a tool call) at or before the result's
                // sequence number. We pass the bounded tail as the search
                // window; the trigger is always within the tail for any
                // tool result in the tail.
                let mut entries = Vec::with_capacity(tool_results.len());
                for (event, tr) in &tool_results {
                    let trigger_text = triggering_turn_text(&events, event.seq, tr);
                    if let Some(embedding) = crate::embedding::embed_text(&trigger_text) {
                        let message = ChatMessage {
                            provider_state: None,
                            role: "tool".to_string(),
                            content: Some(MessageContent::text(tr.content.clone())),
                            tool_calls: None,
                            tool_call_id: Some(tr.tool_call_id.clone()),
                            name: Some(tr.name.clone()),
                        };
                        entries.push(crate::store::IndexedToolState {
                            session_id,
                            seq: event.seq,
                            embedding,
                            message,
                        });
                    }
                }
                let results = futures::stream::iter(entries)
                    .map(|entry| {
                        let store = std::sync::Arc::clone(store);
                        async move { store.index_tool_result(entry).await }
                    })
                    .buffer_unordered(4)
                    .collect::<Vec<_>>()
                    .await;
                for result in results {
                    result?;
                }

                // Query with the embedding of the current turn: the latest
                // user message plus any pending tool-call names, so RAG
                // surfaces results from earlier turns doing similar work.
                let query_text = current_turn_query(&working_messages);
                let query_embedding = crate::embedding::embed_text(&query_text);
                match query_embedding {
                    Some(emb) => {
                        store
                            .retrieve_tool_states(
                                session_id,
                                &emb,
                                self.tool_state_config.top_k,
                                self.tool_state_config.min_similarity,
                            )
                            .await?
                    }
                    None => Vec::new(),
                }
            }
            None => Vec::new(),
        };

        let mut provenance = Vec::new();
        for (message_index, message) in working_messages.iter().enumerate() {
            provenance.extend(
                crate::agent_health::message_content_sources(message)
                    .into_iter()
                    .map(|source| ContentProvenance {
                        source,
                        trust: source.trust(),
                        location: "working_message".to_string(),
                        message_index: Some(message_index),
                        source_name: message.name.clone(),
                    }),
            );
        }
        if let Some(summary) = episodic_summary.as_deref() {
            let source = if summary.contains("untrusted data")
                || summary.contains("<untrusted_tool_history>")
            {
                ContentSource::CompactedUntrustedContext
            } else {
                ContentSource::CompactedContext
            };
            provenance.push(ContentProvenance {
                source,
                trust: source.trust(),
                location: "episodic_summary".to_string(),
                message_index: None,
                source_name: Some("episodic_summary".to_string()),
            });
        }
        for (message_index, message) in retrieved_tool_states.iter().enumerate() {
            let source = crate::agent_health::message_content_sources(message)
                .into_iter()
                .find(|source| source.trust() == crate::ContentTrust::UntrustedData)
                .unwrap_or(ContentSource::ToolResult);
            provenance.push(ContentProvenance {
                source,
                trust: source.trust(),
                location: "retrieved_tool_state".to_string(),
                message_index: Some(message_index),
                source_name: message.name.clone(),
            });
        }
        for file in &attached_files {
            provenance.push(ContentProvenance {
                source: ContentSource::RetrievedDocument,
                trust: ContentSource::RetrievedDocument.trust(),
                location: "attached_file".to_string(),
                message_index: None,
                source_name: Some(file.path.clone()),
            });
        }

        Ok(AssembledContext {
            working_messages,
            episodic_summary,
            retrieved_tool_states,
            attached_files,
            provenance,
        })
    }
}

/// Build the text used to embed a `ToolResult` event for indexing: the
/// most recent user message at or before the result's sequence, plus the
/// tool name from the result itself. Including the tool name keeps results
/// from different tools distinguishable even when the surrounding turn text
/// is identical.
fn triggering_turn_text(events: &[SessionEvent], result_seq: i64, result: &ToolResult) -> String {
    let mut text = String::new();
    // Walk backward from the result to find the latest user message that
    // precedes it. `events` is in ascending seq order, so iterate in
    // reverse and take the first `user` message with seq <= result_seq.
    for event in events.iter().rev().take_while(|e| e.seq >= 1) {
        if event.seq > result_seq {
            continue;
        }
        if event.event_type == SessionEventType::LlmCall {
            if let Some(content) = event
                .payload
                .get("request_messages")
                .and_then(serde_json::Value::as_array)
                .and_then(|messages| {
                    messages.iter().rev().find(|message| {
                        message
                            .get("role")
                            .and_then(serde_json::Value::as_str)
                            .is_some_and(|role| role.eq_ignore_ascii_case("user"))
                            && message
                                .get("content")
                                .and_then(serde_json::Value::as_str)
                                .and_then(crate::user_content::session_user_request)
                                .is_some()
                    })
                })
                .and_then(|message| message.get("content"))
                .and_then(serde_json::Value::as_str)
            {
                if let Some(request) = crate::user_content::session_user_request(content) {
                    text.push_str(request);
                }
                text.push('\n');
                break;
            }
        }
    }
    text.push_str(&result.name);
    text
}

/// Build the query text for the current turn from the assembled working
/// messages: the latest user message plus any tool-call names the model
/// has already issued this turn (so RAG can surface prior results for the
/// same tools).
fn current_turn_query(working_messages: &[ChatMessage]) -> String {
    let mut text = String::new();
    let current_turn_start = working_messages.iter().rposition(|message| {
        message.role.eq_ignore_ascii_case("user")
            && message
                .content_str()
                .and_then(crate::user_content::session_user_request)
                .is_some()
    });
    if let Some(message) = current_turn_start.and_then(|index| working_messages.get(index)) {
        if let Some(content) = message.content_str() {
            if let Some(request) = crate::user_content::session_user_request(content) {
                text.push_str(request);
            }
        }
    }
    let turn_messages = current_turn_start
        .map(|index| &working_messages[index..])
        .unwrap_or(working_messages);
    for message in turn_messages {
        if let Some(calls) = message.tool_calls.as_deref() {
            for call in calls {
                text.push('\n');
                text.push_str(&call.function.name);
            }
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::SummaryPayload;
    use crate::in_memory_stores::InMemoryEventStore;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            provider_state: None,
            role: role.to_string(),
            content: Some(MessageContent::text(content)),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn file(path: &str, version: u64) -> FileContextAttachment {
        FileContextAttachment {
            path: path.to_string(),
            version,
            content_hash: None,
            content: None,
        }
    }

    async fn push_llm_call(
        store: &InMemoryEventStore,
        session_id: Uuid,
        role: &str,
        content: &str,
    ) {
        store
            .append_event(
                session_id,
                SessionEventType::LlmCall,
                serde_json::json!({ "response_message": msg(role, content) }),
                None,
            )
            .await
            .unwrap();
    }

    async fn push_tool_result(
        store: &InMemoryEventStore,
        session_id: Uuid,
        tool_call_id: &str,
        name: &str,
        content: &str,
        file_attachment: Option<FileContextAttachment>,
    ) {
        let mut payload = serde_json::json!({
            "result": {
                "content": content,
                "tool_call_id": tool_call_id,
                "name": name,
            }
        });
        if let Some(f) = file_attachment {
            payload["file"] = serde_json::to_value(f).unwrap();
        }
        store
            .append_event(session_id, SessionEventType::ToolResult, payload, None)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn assemble_includes_system_prompt_and_recent_messages() {
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        push_llm_call(&store, session_id, "user", "Hello").await;
        push_llm_call(&store, session_id, "assistant", "Hi!").await;
        push_llm_call(&store, session_id, "user", "What's Rust?").await;

        let manager = MemoryManager::new(2, 8);
        let context = manager
            .assemble(&store, session_id, Some("You are helpful."))
            .await
            .unwrap();

        assert_eq!(context.working_messages[0].role, "system");
        assert_eq!(
            context.working_messages[0].content_str(),
            Some("You are helpful.")
        );
        // working_window_size = 2, so only the last 2 conversational
        // events should be included alongside the system prompt.
        assert_eq!(context.working_messages.len(), 3);
        assert_eq!(context.working_messages[1].content_str(), Some("Hi!"));
        assert_eq!(
            context.working_messages[2].content_str(),
            Some("What's Rust?")
        );
    }

    #[tokio::test]
    async fn assemble_returns_no_summary_when_history_is_short() {
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        push_llm_call(&store, session_id, "user", "Hello").await;
        push_llm_call(&store, session_id, "assistant", "Hi!").await;

        let manager = MemoryManager::new(3, 8);
        let context = manager.assemble(&store, session_id, None).await.unwrap();
        assert!(context.episodic_summary.is_none());
    }

    #[tokio::test]
    async fn assemble_emits_placeholder_reflecting_only_older_untracked_events() {
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        // 15 conversational events, no summary ever recorded.
        for i in 0..15 {
            push_llm_call(&store, session_id, "user", &format!("message {i}")).await;
        }

        let manager = MemoryManager::new(3, 5); // window=3, threshold=5
        let context = manager.assemble(&store, session_id, None).await.unwrap();

        // 15 conversational events - 3 in the working window = 12 older.
        // 12 > threshold(5), so a placeholder should appear naming 12, not
        // the raw total event count.
        let summary = context
            .episodic_summary
            .expect("expected placeholder summary");
        assert!(
            summary.contains("12"),
            "expected count of 12 older events: {summary}"
        );
    }

    #[tokio::test]
    async fn assemble_prefers_real_summary_over_placeholder() {
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        for i in 0..15 {
            push_llm_call(&store, session_id, "user", &format!("message {i}")).await;
        }
        store
            .append_event(
                session_id,
                SessionEventType::Summary,
                serde_json::json!({
                    "summary_text": "User asked several questions about Rust.",
                    "summarized_seq_start": 0,
                    "summarized_seq_end": 10,
                }),
                None,
            )
            .await
            .unwrap();

        let manager = MemoryManager::new(3, 5);
        let context = manager.assemble(&store, session_id, None).await.unwrap();
        assert_eq!(
            context.episodic_summary.as_deref(),
            Some("User asked several questions about Rust.")
        );
    }

    #[tokio::test]
    async fn assemble_deduplicates_files_to_latest_version() {
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        push_tool_result(
            &store,
            session_id,
            "call_1",
            "read_file",
            "contents v1",
            Some(file("src/main.rs", 1)),
        )
        .await;
        push_tool_result(
            &store,
            session_id,
            "call_2",
            "read_file",
            "contents v2",
            Some(file("src/main.rs", 2)),
        )
        .await;
        push_tool_result(
            &store,
            session_id,
            "call_3",
            "read_file",
            "contents v1",
            Some(file("src/lib.rs", 1)),
        )
        .await;

        let manager = MemoryManager::default_config();
        let context = manager.assemble(&store, session_id, None).await.unwrap();

        assert_eq!(context.attached_files.len(), 2);
        let main = context
            .attached_files
            .iter()
            .find(|f| f.path == "src/main.rs")
            .unwrap();
        assert_eq!(main.version, 2);
        let file_provenance: Vec<_> = context
            .provenance
            .iter()
            .filter(|item| item.location == "attached_file")
            .collect();
        assert_eq!(file_provenance.len(), 2);
        assert!(file_provenance.iter().all(|item| {
            item.source == ContentSource::RetrievedDocument
                && item.trust == crate::ContentTrust::UntrustedData
        }));
    }

    #[tokio::test]
    async fn attached_files_are_deterministically_ordered() {
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        for (path, version) in [("z.rs", 1), ("a.rs", 1), ("m.rs", 1)] {
            push_tool_result(
                &store,
                session_id,
                "call",
                "read_file",
                "contents",
                Some(file(path, version)),
            )
            .await;
        }

        let manager = MemoryManager::default_config();
        let context_a = manager.assemble(&store, session_id, None).await.unwrap();
        let context_b = manager.assemble(&store, session_id, None).await.unwrap();

        let paths_a: Vec<&str> = context_a
            .attached_files
            .iter()
            .map(|f| f.path.as_str())
            .collect();
        let paths_b: Vec<&str> = context_b
            .attached_files
            .iter()
            .map(|f| f.path.as_str())
            .collect();
        assert_eq!(paths_a, paths_b);
        assert_eq!(paths_a, vec!["a.rs", "m.rs", "z.rs"]);
    }

    #[tokio::test]
    async fn assemble_handles_empty_session() {
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        let manager = MemoryManager::default_config();
        let context = manager
            .assemble(&store, session_id, Some("sys"))
            .await
            .unwrap();
        assert_eq!(context.working_messages.len(), 1); // just the system prompt
        assert!(context.episodic_summary.is_none());
        assert!(context.attached_files.is_empty());
    }

    #[tokio::test]
    async fn assemble_reconstructs_user_turn_from_llm_call_payload() {
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        store
            .append_event(
                session_id,
                SessionEventType::LlmCall,
                serde_json::json!({
                    "request_messages": [msg("user", "Please inspect the bug")],
                    "response_message": msg("assistant", "I found it")
                }),
                None,
            )
            .await
            .unwrap();

        let context = MemoryManager::new(4, 8)
            .assemble(&store, session_id, None)
            .await
            .unwrap();
        assert_eq!(context.working_messages.len(), 2);
        assert_eq!(
            context.working_messages[0].content_str(),
            Some("Please inspect the bug")
        );
    }

    #[tokio::test]
    async fn assemble_does_not_duplicate_a_mirrored_external_tool_call() {
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        store
            .append_event(
                session_id,
                SessionEventType::LlmCall,
                serde_json::json!({
                    "request_messages": [msg("user", "Read the file")],
                    "response_message": tool_call_msg("read_file", "call-1")
                }),
                None,
            )
            .await
            .unwrap();
        store
            .append_event(
                session_id,
                SessionEventType::ToolCall,
                serde_json::json!({
                    "tool_call": {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"}
                    },
                    "execution_source": "external_client",
                    "origin_llm_call_seq": 1
                }),
                None,
            )
            .await
            .unwrap();
        push_tool_result(&store, session_id, "call-1", "read_file", "contents", None).await;

        let context = MemoryManager::new(8, 16)
            .assemble(&store, session_id, None)
            .await
            .unwrap();
        let assistant_tool_calls = context
            .working_messages
            .iter()
            .filter(|message| {
                message.role == "assistant"
                    && message
                        .tool_calls
                        .as_deref()
                        .is_some_and(|calls| !calls.is_empty())
            })
            .count();

        assert_eq!(assistant_tool_calls, 1);
        assert_eq!(context.working_messages.len(), 3);
    }

    // --- RAG tool-state tier tests ---

    use crate::in_memory_stores::InMemoryToolStateStore;

    fn tool_call_msg(name: &str, id: &str) -> ChatMessage {
        ChatMessage {
            provider_state: None,
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![crate::models::ToolCall {
                id: id.to_string(),
                call_type: "function".to_string(),
                function: crate::models::FunctionCall {
                    name: name.to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            name: None,
        }
    }

    #[tokio::test]
    async fn assemble_without_tool_state_store_leaves_retrieved_empty() {
        // Default MemoryManager has no ToolStateStore, so the RAG tier
        // must degrade to empty rather than panic.
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        push_llm_call(&store, session_id, "user", "hello").await;
        push_tool_result(&store, session_id, "c1", "read_file", "file contents", None).await;

        let context = MemoryManager::default_config()
            .assemble(&store, session_id, None)
            .await
            .unwrap();
        assert!(context.retrieved_tool_states.is_empty());
    }

    #[tokio::test]
    async fn assemble_retrieves_similar_tool_state_for_current_turn() {
        let event_store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();

        // Earlier turn: user asks to search the codebase, tool returns results.
        event_store
            .append_event(
                session_id,
                SessionEventType::LlmCall,
                serde_json::json!({
                    "request_messages": [msg("user", "search the codebase for authentication helpers")],
                    "response_message": tool_call_msg("grep_search", "call_1")
                }),
                None,
            )
            .await
            .unwrap();
        push_tool_result(
            &event_store,
            session_id,
            "call_1",
            "grep_search",
            "found auth.rs, oauth.rs, token.rs",
            None,
        )
        .await;

        // Later turn: user asks a similar question. The RAG tier should
        // surface the earlier grep_search result.
        event_store
            .append_event(
                session_id,
                SessionEventType::LlmCall,
                serde_json::json!({
                    "request_messages": [msg("user", "search the codebase for authentication utilities again")],
                    "response_message": msg("assistant", "let me look")
                }),
                None,
            )
            .await
            .unwrap();

        let tool_state_store = Arc::new(InMemoryToolStateStore::default());
        let manager = MemoryManager::default_config()
            .with_tool_state_store(tool_state_store, ToolStateRetrievalConfig::default());
        let context = manager
            .assemble(&event_store, session_id, None)
            .await
            .unwrap();

        assert!(
            !context.retrieved_tool_states.is_empty(),
            "expected at least one retrieved tool state for a similar turn"
        );
        let retrieved = &context.retrieved_tool_states[0];
        assert_eq!(retrieved.role, "tool");
        assert_eq!(retrieved.name.as_deref(), Some("grep_search"));
        assert_eq!(
            retrieved.content_str(),
            Some("found auth.rs, oauth.rs, token.rs")
        );
        assert!(context.provenance.iter().any(|item| {
            item.location == "retrieved_tool_state"
                && item.source == ContentSource::ToolResult
                && item.trust == crate::ContentTrust::UntrustedData
        }));
    }

    #[tokio::test]
    async fn assemble_does_not_retrieve_dissimilar_tool_state() {
        let event_store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();

        // Earlier turn about a completely different topic.
        event_store
            .append_event(
                session_id,
                SessionEventType::LlmCall,
                serde_json::json!({
                    "request_messages": [msg("user", "write a haiku about winter snowfall drifting")],
                    "response_message": tool_call_msg("write_poem", "call_1")
                }),
                None,
            )
            .await
            .unwrap();
        push_tool_result(
            &event_store,
            session_id,
            "call_1",
            "write_poem",
            "silent white fields",
            None,
        )
        .await;

        // Current turn about database migrations - unrelated text.
        event_store
            .append_event(
                session_id,
                SessionEventType::LlmCall,
                serde_json::json!({
                    "request_messages": [msg("user", "run the postgres schema migrations now")],
                    "response_message": msg("assistant", "on it")
                }),
                None,
            )
            .await
            .unwrap();

        let tool_state_store = Arc::new(InMemoryToolStateStore::default());
        // Use a high similarity floor so unrelated results are excluded.
        let manager = MemoryManager::default_config().with_tool_state_store(
            tool_state_store,
            ToolStateRetrievalConfig {
                top_k: 4,
                min_similarity: 0.6,
            },
        );
        let context = manager
            .assemble(&event_store, session_id, None)
            .await
            .unwrap();
        assert!(
            context.retrieved_tool_states.is_empty(),
            "dissimilar tool results should not be retrieved"
        );
    }

    #[tokio::test]
    async fn indexing_is_idempotent_across_assemblies() {
        // Re-assembling the same session should not duplicate indexed
        // entries, so retrieved_tool_states should not grow on repeat calls.
        let event_store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        event_store
            .append_event(
                session_id,
                SessionEventType::LlmCall,
                serde_json::json!({
                    "request_messages": [msg("user", "find the database connection helper")],
                    "response_message": tool_call_msg("grep_search", "call_1")
                }),
                None,
            )
            .await
            .unwrap();
        push_tool_result(
            &event_store,
            session_id,
            "call_1",
            "grep_search",
            "found db.rs",
            None,
        )
        .await;
        event_store
            .append_event(
                session_id,
                SessionEventType::LlmCall,
                serde_json::json!({
                    "request_messages": [msg("user", "find the database connection helper again")],
                    "response_message": msg("assistant", "ok")
                }),
                None,
            )
            .await
            .unwrap();

        let tool_state_store = Arc::new(InMemoryToolStateStore::default());
        let manager = MemoryManager::default_config()
            .with_tool_state_store(tool_state_store, ToolStateRetrievalConfig::default());

        let ctx1 = manager
            .assemble(&event_store, session_id, None)
            .await
            .unwrap();
        let ctx2 = manager
            .assemble(&event_store, session_id, None)
            .await
            .unwrap();

        assert_eq!(
            ctx1.retrieved_tool_states.len(),
            ctx2.retrieved_tool_states.len(),
            "repeat assembly must not duplicate retrieved tool states"
        );
        assert!(!ctx1.retrieved_tool_states.is_empty());
    }

    // Bounded-read regression tests (O(n²) → O(n) fix)

    /// A store wrapper that counts how many events `read_events` returns so
    /// we can assert `assemble` no longer pulls the entire log on every turn.
    #[derive(Clone, Default)]
    struct CountingEventStore {
        inner: InMemoryEventStore,
        read_events_calls: Arc<AtomicUsize>,
        last_read_count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl EventStore for CountingEventStore {
        async fn append_event(
            &self,
            session_id: Uuid,
            event_type: SessionEventType,
            payload: serde_json::Value,
            cost_cents: Option<i64>,
        ) -> EngineResult<SessionEvent> {
            self.inner
                .append_event(session_id, event_type, payload, cost_cents)
                .await
        }

        async fn read_events(
            &self,
            session_id: Uuid,
            since_seq: Option<i64>,
            limit: Option<usize>,
        ) -> EngineResult<Vec<SessionEvent>> {
            let events = self.inner.read_events(session_id, since_seq, limit).await?;
            self.read_events_calls.fetch_add(1, Ordering::SeqCst);
            self.last_read_count.store(events.len(), Ordering::SeqCst);
            Ok(events)
        }

        async fn latest_seq(&self, session_id: Uuid) -> EngineResult<Option<i64>> {
            self.inner.latest_seq(session_id).await
        }

        async fn read_latest_summary(
            &self,
            session_id: Uuid,
        ) -> EngineResult<Option<SummaryPayload>> {
            self.inner.read_latest_summary(session_id).await
        }

        async fn read_latest_files(
            &self,
            session_id: Uuid,
        ) -> EngineResult<Vec<FileContextAttachment>> {
            self.inner.read_latest_files(session_id).await
        }

        async fn redeem_checkpoint(
            &self,
            session_id: Uuid,
            token_fingerprint: &str,
            payload: serde_json::Value,
        ) -> EngineResult<SessionEvent> {
            self.inner
                .redeem_checkpoint(session_id, token_fingerprint, payload)
                .await
        }
    }

    #[tokio::test]
    async fn assemble_reads_a_bounded_tail_not_the_whole_log() {
        // A long session: 100 LLM-call events. The old `read_events(..,
        // None, None)` would have returned all 100 every turn. The bounded
        // read should return only `working_window_size * 3` (default
        // window=3 → read_limit=9) events regardless of total length.
        let store = CountingEventStore::default();
        let session_id = Uuid::new_v4();
        for i in 0..100 {
            push_llm_call(&store.inner, session_id, "user", &format!("turn {i}")).await;
        }

        let manager = MemoryManager::default_config(); // window=3 → read_limit=9
        let context = manager
            .assemble(&store, session_id, Some("sys"))
            .await
            .unwrap();

        // The single `read_events` call must have returned a bounded slice,
        // not the full 100 events.
        let read = store.last_read_count.load(Ordering::SeqCst);
        assert!(
            read <= 9,
            "bounded read should be ≤ 9 events for window=3, got {read}"
        );
        assert!(
            read > 0,
            "bounded read should return the tail, not an empty slice"
        );

        // The working window still has the system prompt + the last 3
        // conversational messages (each push_llm_call yields one
        // response_message → 3 messages).
        assert_eq!(context.working_messages.len(), 4);
        assert_eq!(context.working_messages[0].role, "system");
        assert_eq!(context.working_messages[3].content_str(), Some("turn 99"));
    }

    #[tokio::test]
    async fn assemble_surfaces_files_attached_outside_the_bounded_tail() {
        // A file attached early in the session (well before the bounded
        // tail) must still appear in `attached_files`, since
        // `read_latest_files` does its own server-side dedup across the
        // whole log. This guards the citation-grounding regression that a
        // naive bounded `read_events` would introduce.
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        // Early file attachment (seq 1), then 60 LLM-call events to push it
        // far outside the default window=3 → read_limit=9 tail.
        push_tool_result(
            &store,
            session_id,
            "call_1",
            "read_file",
            "early contents",
            Some(file("src/early.rs", 1)),
        )
        .await;
        for i in 0..60 {
            push_llm_call(&store, session_id, "user", &format!("turn {i}")).await;
        }

        let manager = MemoryManager::default_config();
        let context = manager.assemble(&store, session_id, None).await.unwrap();
        assert_eq!(context.attached_files.len(), 1);
        assert_eq!(context.attached_files[0].path, "src/early.rs");
        assert_eq!(context.attached_files[0].version, 1);
    }

    #[tokio::test]
    async fn assemble_uses_latest_summary_cursor_to_bound_the_tail() {
        // After a summary covers seqs 1..=50, the working window only
        // needs the post-summary tail. With 60 total events and a summary
        // at seq 51 (summarized_seq_end=50), the bounded read should start
        // at seq 51 and return only the last ~10 events, not all 60.
        let store = CountingEventStore::default();
        let session_id = Uuid::new_v4();
        for i in 0..50 {
            push_llm_call(&store.inner, session_id, "user", &format!("old {i}")).await;
        }
        store
            .append_event(
                session_id,
                SessionEventType::Summary,
                serde_json::json!({
                    "summary_text": "Earlier turns about an old topic.",
                    "summarized_seq_start": 0,
                    "summarized_seq_end": 50,
                }),
                None,
            )
            .await
            .unwrap();
        for i in 0..10 {
            push_llm_call(&store.inner, session_id, "user", &format!("new {i}")).await;
        }

        let manager = MemoryManager::default_config(); // window=3 → read_limit=9
        let context = manager
            .assemble(&store, session_id, Some("sys"))
            .await
            .unwrap();

        // The episodic summary text comes from the targeted lookup, not
        // the bounded slice.
        assert_eq!(
            context.episodic_summary.as_deref(),
            Some("Earlier turns about an old topic.")
        );
        // The read happened and was bounded by the post-summary tail +
        // window (≤ read_limit=9, since the tail is only ~10 events and
        // the since_seq cursor is 51).
        let read = store.last_read_count.load(Ordering::SeqCst);
        assert!(
            read <= 9,
            "post-summary bounded read should be ≤ 9, got {read}"
        );
        // The working window shows the *new* turns, not the summarized
        // ones.
        assert_eq!(context.working_messages.len(), 4); // system + last 3
        assert_eq!(context.working_messages[3].content_str(), Some("new 9"));
    }
}
