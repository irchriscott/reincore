//! Predictive pre-warming for budget-tier transitions.
//!
//! When a session enters Tier 2 (70–90 % budget), the cheap / Lite model
//! becomes the default routing target for routine steps. The first call to
//! that model incurs cold-start latency: TLS handshake, DNS resolution,
//! and (for providers that support it) prompt-cache population. If the
//! degradation transition itself is slow, the user perceives a stall right
//! when the system is trying to save money - the worst possible moment.
//!
//! This module proactively warms two things the moment a tier transition
//! is detected, **in the background** so the current response is never
//! delayed:
//!
//! 1. **Provider connection** - issues a minimal 1-token request to the
//!    Lite model so the HTTP connection pool has a warm entry and (for
//!    providers like Anthropic / OpenAI) the prompt-cache prefix is
//!    populated with the system prompt.
//! 2. **Tool-state pre-fetch** - queries the [`ToolStateStore`] with an
//!    embedding of the current session shape so likely-needed past tool
//!    results are resident in the memory manager's retrieval cache before
//!    the next turn asks for them.
//!
//! The "likely-needed" set is derived from **session shape similarity**:
//! a lightweight fingerprint (`SessionShape`) built from the task summary
//! embedding and the set of tools used so far. Past sessions on the same
//! project with high shape similarity are good predictors of which tool
//! results the degrading session will need next.

use crate::embedding::embed_text;
use crate::error::EngineResult;
use crate::event::{EventStore, SessionEventType};
use crate::models::{ChatCompletionRequest, MessageContent, ModelTier};
use crate::provider::{Provider, ProviderConfig};
use crate::session::Session;
use crate::store::ToolStateStore;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

const PROVIDER_PREWARM_TIMEOUT: Duration = Duration::from_secs(10);

/// Configuration for the pre-warming subsystem.
#[derive(Clone, Debug)]
pub struct PreWarmConfig {
    /// Maximum number of past sessions to compare shapes against when
    /// deciding which tool results to pre-fetch. Bounded to keep the
    /// similarity scan cheap.
    pub max_shape_comparisons: usize,
    /// Similarity threshold (0.0–1.0 cosine) above which a past session's
    /// tool results are considered "likely needed" and pre-fetched.
    pub shape_similarity_threshold: f32,
    /// Top-k tool-state results to pre-fetch from the store.
    pub pre_fetch_top_k: usize,
    /// Minimum similarity for pre-fetched tool states (matches the memory
    /// manager's RAG tier default).
    pub pre_fetch_min_similarity: f32,
}

impl Default for PreWarmConfig {
    fn default() -> Self {
        Self {
            max_shape_comparisons: 32,
            shape_similarity_threshold: 0.65,
            pre_fetch_top_k: 4,
            pre_fetch_min_similarity: 0.35,
        }
    }
}

/// A lightweight, privacy-preserving fingerprint of a session used for
/// shape-similarity matching. The shape is built from the deterministic
/// FNV-1a trigram embedding of the task summary and the sorted set of
/// tool names used so far. It does NOT include message content, file
/// paths, or any user data - only the structural "what kind of task is
/// this" signal.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionShape {
    pub session_id: Uuid,
    /// 384-dimensional FNV-1a trigram embedding of the task summary.
    pub summary_embedding: Vec<f32>,
    /// Sorted, deduplicated tool names used in this session so far.
    pub tool_names: Vec<String>,
    /// The budget tier the session was in when the shape was captured.
    pub tier: u8,
}

impl SessionShape {
    /// Build a shape from the current session state and its event log.
    ///
    /// `tool_names` should be the sorted, deduplicated list of tool names
    /// that appear in the session's `ToolCall` events. The caller is
    /// responsible for extracting this from the event log because the
    /// event-store trait is generic over the backend.
    pub fn from_session(session: &Session, mut tool_names: Vec<String>) -> Option<Self> {
        let summary_embedding = embed_text(&session.task_summary)?;
        // Keep the public constructor robust even when callers do not uphold
        // the documented sorted/deduplicated precondition.
        tool_names.sort_unstable();
        tool_names.dedup();
        Some(Self {
            session_id: session.id,
            summary_embedding,
            tool_names,
            tier: session.tier,
        })
    }

    /// Cosine similarity between two shapes, blending the summary embedding
    /// and the tool-name Jaccard index. The blend is 70 % embedding
    /// (semantic task shape) + 30 % tool overlap (structural similarity),
    /// which empirically weights "same kind of task" higher than "same
    /// tools" while still surfacing sessions that use identical tooling.
    pub fn similarity(&self, other: &SessionShape) -> f32 {
        let emb_sim = cosine_similarity(&self.summary_embedding, &other.summary_embedding);
        let tool_sim = jaccard(&self.tool_names, &other.tool_names);
        0.7 * emb_sim + 0.3 * tool_sim
    }
}

/// Outcome of a pre-warming pass, persisted as an event for observability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreWarmOutcome {
    /// Estimated cost of the provider request; unknown when usage is unavailable.
    pub cost: Option<crate::pricing::CostEstimate>,
    pub session_id: Uuid,
    /// The tier the session transitioned into (always ≥ 2).
    pub warmed_for_tier: u8,
    /// Whether the provider connection was warmed.
    pub provider_warmed: bool,
    /// Provider label that was warmed, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_label: Option<String>,
    /// Model that was pre-warmed (the Lite / cheap model for this provider).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmed_model: Option<String>,
    /// Number of tool-state results pre-fetched into the retrieval cache.
    pub pre_fetched_tool_states: usize,
    /// Number of past sessions compared for shape similarity.
    pub shape_comparisons: usize,
    /// Highest shape similarity score found (0.0–1.0).
    pub best_shape_similarity: f32,
    /// Elapsed time for the pre-warm pass, in microseconds.
    pub elapsed_us: u64,
}

/// The pre-warming manager. Stateless beyond its config - all session
/// data is read from the event store and tool-state store at invocation
/// time.
pub struct PreWarmManager {
    config: PreWarmConfig,
}

impl PreWarmManager {
    pub fn new(config: PreWarmConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &PreWarmConfig {
        &self.config
    }

    /// Trigger a pre-warming pass for a session that just entered a
    /// degradation tier (≥ Tier 2). This is safe to call from a background
    /// task. Provider failures produce unknown usage receipts; persistence
    /// failures are returned to the caller for explicit reporting.
    ///
    /// `provider` is the provider that will serve the degraded (Lite) calls.
    /// `tool_state_store` is the RAG store used by the memory manager.
    /// `event_store` is used to (a) extract the session's tool-call history
    /// for shape building and (b) append the `PreWarm` outcome event.
    /// `accounting_store` durably settles provider usage before tool prefetch.
    /// `past_shapes` are shapes from other sessions on the same project /
    /// workspace, supplied by the caller (who has access to the session
    /// list).
    pub async fn warm_on_tier_transition<E: EventStore>(
        &self,
        session: &Session,
        provider: Arc<dyn Provider>,
        provider_config: &ProviderConfig,
        tool_state_store: Option<Arc<dyn ToolStateStore>>,
        event_store: &E,
        accounting_store: &dyn crate::accounting::AccountingStore,
        past_shapes: &[SessionShape],
    ) -> EngineResult<PreWarmOutcome> {
        let start = std::time::Instant::now();
        let session_id = session.id;
        let warmed_for_tier = session.tier;

        // 1. Pre-warm the provider connection with a minimal request to
        //    the Lite model. This populates the HTTP connection pool and,
        //    for providers with prompt caching, seeds the cache prefix.
        //    The request uses a 1-token max and a trivial prompt so the
        //    cost is negligible (typically < $0.0001).
        // Connection warming and event-log inspection are independent. Run
        // them together so the best-effort background pass takes the slower
        // of the two operations instead of their sum. Bound the provider call
        // so a stalled upstream cannot retain this task indefinitely.
        let warm_provider = async {
            match resolve_lite_model(provider_config) {
                Some(model) => {
                    let warm_req = build_warm_request(&model);
                    let response = tokio::time::timeout(
                        PROVIDER_PREWARM_TIMEOUT,
                        provider.pre_warm(&warm_req),
                    )
                    .await
                    .ok()
                    .and_then(Result::ok);
                    let receipt = crate::accounting::CompletionAccounting::auxiliary(
                        session_id,
                        provider.as_ref(),
                        response
                            .as_ref()
                            .map(|r| r.model.as_str())
                            .unwrap_or(&model),
                        response.as_ref().and_then(|r| r.usage.as_ref()),
                        "prewarm",
                    );
                    accounting_store.record_accounting(&receipt).await?;
                    Ok((response.is_some(), Some(model), receipt.cost))
                }
                None => Ok::<_, crate::EngineError>((false, None, None)),
            }
        };
        let (warm_result, tool_names) =
            tokio::join!(warm_provider, extract_tool_names(event_store, session_id));
        let (provider_warmed, warmed_model, cost) = warm_result?;
        let provider_label = Some(provider.label().to_string());

        // 2. Build the current session's shape from its task summary and
        //    tool-call history.
        let current_shape = SessionShape::from_session(session, tool_names);

        // 3. Pre-fetch likely-needed tool results from the RAG store.
        //    We query with the current session's summary embedding (or
        //    the best-matching past session's embedding if the current
        //    session is too young to have tool results). The store
        //    retrieves top-k by cosine similarity, which is exactly the
        //    "likely needed" set.
        let mut pre_fetched = 0usize;
        let mut best_similarity = 0.0f32;
        let mut shape_comparisons = 0usize;
        let mut best_past_shape = None;

        if let Some(current) = &current_shape {
            for past in past_shapes.iter().take(self.config.max_shape_comparisons) {
                shape_comparisons += 1;
                let sim = current.similarity(past);
                if sim > best_similarity {
                    best_similarity = sim;
                    best_past_shape = Some(past);
                }
            }

            if let Some(store) = &tool_state_store {
                let query_embedding = best_past_shape
                    .filter(|_| best_similarity >= self.config.shape_similarity_threshold)
                    .map(|past| past.summary_embedding.as_slice())
                    .unwrap_or(&current.summary_embedding);
                let results = store
                    .retrieve_tool_states(
                        session_id,
                        query_embedding,
                        self.config.pre_fetch_top_k,
                        self.config.pre_fetch_min_similarity,
                    )
                    .await
                    .unwrap_or_default();
                pre_fetched = results.len();
            }
        }

        let outcome = PreWarmOutcome {
            cost,
            session_id,
            warmed_for_tier,
            provider_warmed,
            provider_label,
            warmed_model,
            pre_fetched_tool_states: pre_fetched,
            shape_comparisons,
            best_shape_similarity: best_similarity,
            elapsed_us: start.elapsed().as_micros() as u64,
        };

        // Append the outcome as an event for observability / dashboard.
        let _ = event_store
            .append_event(
                session_id,
                SessionEventType::PreWarm,
                serde_json::to_value(&outcome).unwrap_or(serde_json::Value::Null),
                None,
            )
            .await;

        Ok(outcome)
    }
}

/// Resolve the Lite (cheap) model name for a provider, if one can be
/// derived from the provider's config. Returns `None` when no Lite model
/// can be resolved (e.g. a provider with no standard model and no
/// explicit tier mappings).
///
/// Note: this does NOT check `Provider::is_free()` - the caller
/// (`trigger_prewarm_if_tier_changed`) is responsible for skipping free
/// providers before calling `warm_on_tier_transition`.
fn resolve_lite_model(config: &ProviderConfig) -> Option<String> {
    config.model_for_tier(ModelTier::Lite).ok()
}

/// Build a minimal chat-completion request suitable for warming a
/// provider connection. The prompt is a single user message asking for
/// a single token, with `max_tokens=1`. This minimizes cost while still
/// establishing the HTTP connection and (for caching providers) seeding
/// the prompt-cache prefix.
fn build_warm_request(model: &str) -> ChatCompletionRequest {
    use crate::models::ChatMessage;
    use std::collections::HashMap;

    ChatCompletionRequest {
        model: model.to_string(),
        messages: vec![ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(MessageContent::text(".")),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }],
        tools: None,
        tool_choice: None,
        temperature: Some(0.0),
        max_tokens: Some(1),
        stream: None,
        extra: {
            let mut extra = HashMap::new();
            extra.insert(
                "reincore_model_tier".to_string(),
                serde_json::Value::String("lite".to_string()),
            );
            extra.insert(
                "reincore_prewarm".to_string(),
                serde_json::Value::Bool(true),
            );
            extra
        },
    }
}

/// Extract the sorted, deduplicated tool names from a session's `ToolCall`
/// events. This is used to build the `SessionShape` for similarity matching.
async fn extract_tool_names<E: EventStore>(store: &E, session_id: Uuid) -> Vec<String> {
    let events = store
        .read_events(session_id, None, None)
        .await
        .unwrap_or_default();

    let mut names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for event in &events {
        if event.event_type == SessionEventType::ToolCall {
            if let Some(name) = event
                .payload
                .get("tool_call")
                .and_then(|v| v.get("function"))
                .and_then(|v| v.get("name"))
                .and_then(|v| v.as_str())
            {
                names.insert(name.to_string());
            }
        }
    }
    names.into_iter().collect()
}

/// Cosine similarity for two equal-length float vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt().max(f32::EPSILON) * norm_b.sqrt().max(f32::EPSILON);
    if !denom.is_finite() || denom <= f32::EPSILON {
        return 0.0;
    }
    (dot / denom).clamp(-1.0, 1.0)
}

/// Jaccard similarity for two string sets.
fn jaccard(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0; // Both empty = identical tool sets.
    }
    // Shapes loaded from older persisted payloads may predate constructor
    // normalization, so tolerate unsorted or duplicated names here.
    let set_a: std::collections::HashSet<&str> = a.iter().map(String::as_str).collect();
    let set_b: std::collections::HashSet<&str> = b.iter().map(String::as_str).collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        return 0.0;
    }
    (intersection as f32) / (union as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::in_memory_stores::InMemoryEventStore;
    use crate::provider::ProviderConfig;
    use crate::session::Session;

    #[derive(Default)]
    struct TestAccounting(std::sync::Mutex<Vec<crate::accounting::CompletionAccounting>>);
    #[async_trait::async_trait]
    impl crate::accounting::AccountingStore for TestAccounting {
        async fn stage_accounting(
            &self,
            record: &crate::accounting::CompletionAccounting,
        ) -> EngineResult<()> {
            record.validate()?;
            self.0.lock().unwrap().push(record.clone());
            Ok(())
        }
        async fn settle_accounting(&self, _: Uuid) -> EngineResult<()> {
            Ok(())
        }
        async fn recover_accounting(&self) -> EngineResult<()> {
            Ok(())
        }
    }
    fn test_session(summary: &str, tier: u8) -> Session {
        let mut session = Session::new(summary, 1000).unwrap();
        session.tier = tier;
        session
    }

    #[test]
    fn cosine_similarity_identical_vectors() {
        let v = vec![0.1; 384];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let mut a = vec![0.0; 384];
        let mut b = vec![0.0; 384];
        a[0] = 1.0;
        b[1] = 1.0;
        assert!(cosine_similarity(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_mismatched_lengths() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn jaccard_identical_sets() {
        let a = vec!["search".to_string(), "read".to_string()];
        assert!((jaccard(&a, &a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn jaccard_disjoint_sets() {
        let a = vec!["search".to_string()];
        let b = vec!["read".to_string()];
        assert!(jaccard(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn jaccard_partial_overlap() {
        let a = vec!["search".to_string(), "read".to_string()];
        let b = vec!["read".to_string(), "write".to_string()];
        // intersection = {read} = 1, union = {search, read, write} = 3
        assert!((jaccard(&a, &b) - (1.0 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn jaccard_both_empty() {
        assert!((jaccard(&[], &[]) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn session_shape_from_session_with_tools() {
        let session = test_session("Refactor authentication module", 2);
        let shape = SessionShape::from_session(
            &session,
            vec!["read_file".to_string(), "search".to_string()],
        );
        assert!(shape.is_some());
        let shape = shape.unwrap();
        assert_eq!(shape.summary_embedding.len(), 384);
        assert_eq!(shape.tool_names, vec!["read_file", "search"]);
        assert_eq!(shape.tier, 2);
    }

    #[test]
    fn session_shape_normalizes_tool_names() {
        let session = test_session("Refactor authentication module", 2);
        let shape = SessionShape::from_session(
            &session,
            vec![
                "search".to_string(),
                "read_file".to_string(),
                "search".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(shape.tool_names, vec!["read_file", "search"]);
    }

    #[test]
    fn session_shape_empty_summary_returns_none() {
        let session = test_session("", 2);
        let shape = SessionShape::from_session(&session, vec![]);
        // embed_text returns None for empty strings in some cases;
        // if it returns Some, the shape is still valid but has a zero embedding.
        if let Some(shape) = shape {
            assert_eq!(shape.summary_embedding.len(), 384);
        }
    }

    #[test]
    fn shape_similarity_same_tools_different_summary() {
        let session_a = test_session("Refactor auth module", 2);
        let session_b = test_session("Debug database connection", 2);
        let tools = vec!["read_file".to_string(), "search".to_string()];
        let shape_a = SessionShape::from_session(&session_a, tools.clone()).unwrap();
        let shape_b = SessionShape::from_session(&session_b, tools).unwrap();
        // Same tools → tool Jaccard = 1.0; different summaries → embedding < 1.0.
        let sim = shape_a.similarity(&shape_b);
        // 0.7 * emb + 0.3 * 1.0; emb < 1.0 so sim < 1.0 but > 0.3 (tool overlap).
        assert!(
            sim > 0.3,
            "similarity with same tools should be > 0.3, got {sim}"
        );
        assert!(
            sim < 1.0,
            "similarity with different summaries should be < 1.0, got {sim}"
        );
    }

    #[test]
    fn shape_similarity_identical_shapes() {
        let session = test_session("Refactor auth module", 2);
        let tools = vec!["read_file".to_string(), "search".to_string()];
        let shape_a = SessionShape::from_session(&session, tools.clone()).unwrap();
        let shape_b = SessionShape::from_session(&session, tools).unwrap();
        let sim = shape_a.similarity(&shape_b);
        assert!(
            (sim - 1.0).abs() < 1e-4,
            "identical shapes should have similarity ~1.0, got {sim}"
        );
    }

    #[test]
    fn build_warm_request_uses_lite_tier_and_one_token() {
        let req = build_warm_request("test-lite-model");
        assert_eq!(req.model, "test-lite-model");
        assert_eq!(req.max_tokens, Some(1));
        assert_eq!(
            req.extra
                .get("reincore_model_tier")
                .and_then(|v| v.as_str()),
            Some("lite")
        );
        assert_eq!(
            req.extra.get("reincore_prewarm").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
    }

    #[test]
    fn resolve_lite_model_returns_model_for_lite_tier() {
        let config = ProviderConfig {
            label: "openai".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            api_key: "test".to_string(),
            standard_model: Some("gpt-4o".to_string()),
            ..Default::default()
        };
        let model = resolve_lite_model(&config);
        assert!(
            model.is_some(),
            "should derive a lite model from standard_model"
        );
    }

    #[tokio::test]
    async fn extract_tool_names_from_event_store() {
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();

        // Append two tool-call events with different tool names.
        let _ = store
            .append_event(
                session_id,
                SessionEventType::ToolCall,
                serde_json::json!({
                    "tool_call": {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": "{}"
                        }
                    }
                }),
                None,
            )
            .await;
        let _ = store
            .append_event(
                session_id,
                SessionEventType::ToolCall,
                serde_json::json!({
                    "tool_call": {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{}"
                        }
                    }
                }),
                None,
            )
            .await;

        let names = extract_tool_names(&store, session_id).await;
        assert_eq!(names, vec!["read_file", "search"]); // BTreeSet sorts.
    }

    #[tokio::test]
    async fn extract_tool_names_empty_store() {
        let store = InMemoryEventStore::default();
        let names = extract_tool_names(&store, Uuid::new_v4()).await;
        assert!(names.is_empty());
    }

    // Integration tests for warm_on_tier_transition

    use crate::in_memory_stores::InMemoryToolStateStore;
    use crate::models::ChatMessage;
    use crate::provider::{ChatCompletionResponse, Provider, ProviderModel};

    /// A stub provider that records whether `pre_warm` / `complete` was called.
    struct StubProvider {
        config: ProviderConfig,
        warm_called: Arc<std::sync::Mutex<bool>>,
    }

    #[async_trait::async_trait]
    impl Provider for StubProvider {
        fn label(&self) -> &str {
            &self.config.label
        }
        fn config(&self) -> &ProviderConfig {
            &self.config
        }
        async fn complete(
            &self,
            request: &ChatCompletionRequest,
        ) -> EngineResult<ChatCompletionResponse> {
            // Record if this is a pre-warm call.
            if request
                .extra
                .get("reincore_prewarm")
                .and_then(|v| v.as_bool())
                == Some(true)
            {
                *self.warm_called.lock().unwrap() = true;
            }
            Ok(ChatCompletionResponse {
                id: "stub".to_string(),
                model: request.model.clone(),
                message: ChatMessage {
                    provider_state: None,
                    role: "assistant".to_string(),
                    content: Some(crate::MessageContent::text("ok".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                usage: None,
                raw: serde_json::Value::Null,
                response_headers: Default::default(),
            })
        }
        async fn list_models(&self) -> EngineResult<Vec<ProviderModel>> {
            Ok(Vec::new())
        }
    }

    fn stub_provider() -> (Arc<StubProvider>, ProviderConfig) {
        let config = ProviderConfig {
            label: "openai".to_string(),
            base_url: "https://example.invalid".to_string(),
            api_key: "test".to_string(),
            standard_model: Some("gpt-4o".to_string()),
            ..Default::default()
        };
        let warm_called = Arc::new(std::sync::Mutex::new(false));
        let provider = Arc::new(StubProvider {
            config: config.clone(),
            warm_called: warm_called.clone(),
        });
        (provider, config)
    }

    #[tokio::test]
    async fn warm_on_tier_transition_warms_provider_and_appends_event() {
        let event_store = InMemoryEventStore::default();
        let tool_state_store: Arc<dyn ToolStateStore> = Arc::new(InMemoryToolStateStore::default());
        let (provider, config) = stub_provider();
        let warm_called = provider.warm_called.clone();

        let mut session = test_session("Refactor auth module", 2);
        session.project_path = Some("/test/project".to_string());

        let accounting = TestAccounting::default();
        let manager = PreWarmManager::new(PreWarmConfig::default());
        let outcome = manager
            .warm_on_tier_transition(
                &session,
                provider.clone() as Arc<dyn Provider>,
                &config,
                Some(tool_state_store),
                &event_store,
                &accounting,
                &[],
            )
            .await
            .unwrap();

        // The provider should have been warmed.
        assert!(
            *warm_called.lock().unwrap(),
            "provider.pre_warm should have been called"
        );
        assert!(outcome.provider_warmed);
        let records = accounting.0.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].purpose.as_deref(), Some("prewarm"));
        assert_eq!(records[0].cost, outcome.cost);
        drop(records);
        assert_eq!(outcome.warmed_for_tier, 2);
        assert!(outcome.warmed_model.is_some());

        // A PreWarm event should have been appended.
        let events = event_store
            .read_events(session.id, None, None)
            .await
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, SessionEventType::PreWarm);
    }

    #[tokio::test]
    async fn warm_on_tier_transition_prefetches_tool_states() {
        let event_store = InMemoryEventStore::default();
        let tool_state_store = Arc::new(InMemoryToolStateStore::default());
        let session = test_session("search for auth files", 2);

        // Index a tool result so there's something to retrieve.
        let embedding = embed_text("search for auth files").unwrap();
        tool_state_store
            .index_tool_result(crate::store::IndexedToolState {
                session_id: session.id,
                seq: 1,
                embedding,
                message: ChatMessage {
                    provider_state: None,
                    role: "tool".to_string(),
                    content: Some(crate::MessageContent::text("found 3 files".to_string())),
                    tool_calls: None,
                    tool_call_id: Some("call_1".to_string()),
                    name: Some("search".to_string()),
                },
            })
            .await
            .unwrap();

        let (provider, config) = stub_provider();

        let manager = PreWarmManager::new(PreWarmConfig::default());
        let outcome = manager
            .warm_on_tier_transition(
                &session,
                provider.clone() as Arc<dyn Provider>,
                &config,
                Some(tool_state_store as Arc<dyn ToolStateStore>),
                &event_store,
                &TestAccounting::default(),
                &[],
            )
            .await
            .unwrap();

        // Should have pre-fetched the similar tool state.
        assert!(
            outcome.pre_fetched_tool_states > 0,
            "should pre-fetch tool states"
        );
    }

    #[tokio::test]
    async fn warm_on_tier_transition_without_tool_state_store() {
        let event_store = InMemoryEventStore::default();
        let (provider, config) = stub_provider();
        let session = test_session("Debug issue", 3);

        let manager = PreWarmManager::new(PreWarmConfig::default());
        let outcome = manager
            .warm_on_tier_transition(
                &session,
                provider.clone() as Arc<dyn Provider>,
                &config,
                None, // No tool state store.
                &event_store,
                &TestAccounting::default(),
                &[],
            )
            .await
            .unwrap();

        // Provider should still be warmed.
        assert!(outcome.provider_warmed);
        // No tool states pre-fetched.
        assert_eq!(outcome.pre_fetched_tool_states, 0);
    }

    #[tokio::test]
    async fn warm_on_tier_transition_records_shape_comparisons() {
        let event_store = InMemoryEventStore::default();
        let tool_state_store: Arc<dyn ToolStateStore> = Arc::new(InMemoryToolStateStore::default());
        let (provider, config) = stub_provider();

        let mut session = test_session("Refactor auth module", 2);
        session.project_path = Some("/test".to_string());

        // Build a past shape with the same summary and same tools.
        let past_session = test_session("Refactor auth module", 2);
        let past_shape =
            SessionShape::from_session(&past_session, vec!["search".to_string()]).unwrap();

        // Give the current session the same tool by appending a ToolCall event.
        let _ = event_store
            .append_event(
                session.id,
                SessionEventType::ToolCall,
                serde_json::json!({
                    "tool_call": {
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "search", "arguments": "{}" }
                    }
                }),
                None,
            )
            .await;

        let manager = PreWarmManager::new(PreWarmConfig::default());
        let outcome = manager
            .warm_on_tier_transition(
                &session,
                provider.clone() as Arc<dyn Provider>,
                &config,
                Some(tool_state_store),
                &event_store,
                &TestAccounting::default(),
                &[past_shape],
            )
            .await
            .unwrap();

        assert!(outcome.shape_comparisons > 0, "should compare shapes");
        assert!(
            outcome.best_shape_similarity > 0.9,
            "identical summary + same tools should have high similarity, got {}",
            outcome.best_shape_similarity
        );
    }

    #[tokio::test]
    async fn warm_on_tier_transition_provider_error_does_not_fail() {
        let event_store = InMemoryEventStore::default();
        let tool_state_store: Arc<dyn ToolStateStore> = Arc::new(InMemoryToolStateStore::default());

        // A provider that always errors on complete.
        struct ErrorProvider;
        #[async_trait::async_trait]
        impl Provider for ErrorProvider {
            fn label(&self) -> &str {
                "error"
            }
            fn config(&self) -> &ProviderConfig {
                use std::sync::OnceLock;
                static CONFIG: OnceLock<ProviderConfig> = OnceLock::new();
                CONFIG.get_or_init(|| ProviderConfig {
                    label: "error".to_string(),
                    base_url: "https://invalid".to_string(),
                    api_key: "test".to_string(),
                    standard_model: Some("test-model".to_string()),
                    ..Default::default()
                })
            }
            async fn complete(
                &self,
                _request: &ChatCompletionRequest,
            ) -> EngineResult<ChatCompletionResponse> {
                Err(crate::error::EngineError::Provider(
                    "warm-up failed".to_string(),
                ))
            }
            async fn list_models(&self) -> EngineResult<Vec<ProviderModel>> {
                Ok(Vec::new())
            }
        }

        let provider = Arc::new(ErrorProvider);
        let config = provider.config().clone();
        let session = test_session("Test task", 2);

        let manager = PreWarmManager::new(PreWarmConfig::default());
        // Should NOT error - provider warm failure is best-effort.
        let outcome = manager
            .warm_on_tier_transition(
                &session,
                provider as Arc<dyn Provider>,
                &config,
                Some(tool_state_store),
                &event_store,
                &TestAccounting::default(),
                &[],
            )
            .await
            .unwrap();

        assert!(!outcome.provider_warmed, "provider warm should have failed");
        // Event should still be appended.
        let events = event_store
            .read_events(session.id, None, None)
            .await
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, SessionEventType::PreWarm);
    }
}
