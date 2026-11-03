//! Storage contracts for memoization, semantic cache, and tool-state RAG.
//!
//! Defines the async traits that persistence backends (SQLite, Postgres,
//! in-memory) must implement:
//!
//! - [`CacheStore`] - content-addressed tool-result memoization and
//!   embedding-based semantic response cache.
//! - [`TeamScopedCacheStore`] - cloud persistence with workspace partitions and
//!   mandatory semantic scope fingerprints. Consumers also enforce server-owned
//!   policies, exact-key scopes and current source authorization.
//! - [`ToolStateStore`] - stores and retrieves raw tool results by embedding
//!   for the RAG tool-state tier of the memory manager.

use crate::cache::{
    CacheEntry, CacheKey, SemanticCacheEntry, SemanticCacheLookup, SemanticCacheQuery,
    SemanticCacheScope,
};
use crate::error::EngineResult;
use crate::models::{ChatMessage, FileContextAttachment};
use async_trait::async_trait;
use uuid::Uuid;

/// A safe tool result indexed by the request that produced it so another
/// agent in the same workspace and request group can reuse it.
#[derive(Clone, Debug)]
pub struct TeamSemanticToolCacheEntry {
    /// Cloud tenant boundary. Semantic tool results are never unscoped.
    pub workspace_id: Uuid,
    /// Coordination boundary shared by agents working on related subtasks.
    pub request_group_id: String,
    /// Mandatory server-computed authorization/policy/embedding partition.
    pub scope_fingerprint: String,
    /// Session that executed the original tool call.
    pub source_session_id: Uuid,
    /// Exact content-addressed key, retained for invalidation and upserts.
    pub cache_key: CacheKey,
    /// Server-owned, versioned embedding of the actual tool name + arguments.
    pub request_embedding: Vec<f32>,
    /// Cached result and its integrity metadata.
    pub cache_entry: CacheEntry,
}

/// A semantically matched tool result, including provenance for response
/// metadata and observability.
#[derive(Clone, Debug)]
pub struct TeamSemanticToolCacheHit {
    pub cache_entry: CacheEntry,
    pub source_session_id: Uuid,
    pub similarity: f32,
}

/// Storage contract for memoization and semantic caches.
#[async_trait]
pub trait CacheStore: Send + Sync {
    /// Look up a memoized tool result by its content-addressed key.
    async fn get_memoized(&self, key: &CacheKey) -> EngineResult<Option<CacheEntry>>;

    /// Store a memoized tool result.
    async fn put_memoized(&self, key: CacheKey, entry: CacheEntry) -> EngineResult<()>;

    /// Look up scoped, unexpired semantic responses with an exact request
    /// fingerprint. Only `reusable_answer` may short-circuit generation;
    /// lower-confidence matches are provenance-only retrieval candidates.
    async fn get_semantic(&self, query: &SemanticCacheQuery) -> EngineResult<SemanticCacheLookup>;

    /// Store a response with its embedding for semantic retrieval.
    async fn put_semantic(&self, entry: SemanticCacheEntry) -> EngineResult<()>;

    /// Invalidate every semantic response within the exact workspace/project
    /// partition. An optional fingerprint narrows invalidation to one request
    /// configuration/version boundary.
    async fn invalidate_semantic(
        &self,
        scope: &SemanticCacheScope,
        fingerprint_hash: Option<&str>,
    ) -> EngineResult<u64>;

    /// Invalidate cache entries matching a tool name or key pattern.
    async fn invalidate(
        &self,
        tool_name: Option<&str>,
        key_pattern: Option<&str>,
    ) -> EngineResult<u64>;
}

/// Low-level team-scoped persistence contract for cloud editions.
/// Workspace alone is NOT authorization. Authenticated HTTP consumers must
/// namespace exact keys by full policy scope, persist verified provenance, and
/// re-check source access on each hit. Legacy raw entries are not reusable.
///
/// The workspace partition is a storage boundary, not a reuse authorization.
/// Exact consumers must derive keys from full policy scope and validate source
/// provenance. Semantic queries additionally require the identical server-owned
/// scope fingerprint (project/principal/credential/data/policy/embedding version).
/// NULL/legacy semantic partitions must never match a scoped query.
///
/// `workspace_id = None` remains a separate legacy/local partition. Cloud HTTP
/// reuse refuses unscoped requests and does not consume legacy raw results.
#[async_trait]
pub trait TeamScopedCacheStore: CacheStore {
    /// Look up a memoized tool result within `workspace_id`'s partition.
    async fn get_memoized_scoped(
        &self,
        key: &CacheKey,
        workspace_id: Option<Uuid>,
    ) -> EngineResult<Option<CacheEntry>>;

    /// Store a memoized tool result in `workspace_id`'s partition.
    async fn put_memoized_scoped(
        &self,
        key: CacheKey,
        entry: CacheEntry,
        workspace_id: Option<Uuid>,
    ) -> EngineResult<()>;

    /// Find a semantically similar safe tool result produced by another
    /// session in the same workspace, request group and policy fingerprint.
    async fn get_semantic_tool_scoped(
        &self,
        workspace_id: Uuid,
        request_group_id: &str,
        scope_fingerprint: &str,
        current_session_id: Uuid,
        tool_name: &str,
        query_embedding: &[f32],
        threshold: f32,
    ) -> EngineResult<Option<TeamSemanticToolCacheHit>>;

    /// Index a safe tool result for cross-agent semantic reuse.
    async fn put_semantic_tool_scoped(&self, entry: TeamSemanticToolCacheEntry)
        -> EngineResult<()>;

    /// Invalidate cached entries within a workspace matching a tool name
    /// or key pattern.
    async fn invalidate_scoped(
        &self,
        workspace_id: Option<Uuid>,
        tool_name: Option<&str>,
        key_pattern: Option<&str>,
    ) -> EngineResult<u64>;

    /// Invalidate cached entries within `workspace_id`'s partition whose
    /// result payload references the given file path. This is the
    /// workspace-scoped version of [`MemoizationCache::invalidate_for_file_update`].
    /// A file version change in one workspace must not invalidate cached
    /// results in another workspace's partition.
    async fn invalidate_for_file_update_scoped(
        &self,
        workspace_id: Option<Uuid>,
        file: &FileContextAttachment,
    ) -> EngineResult<u64>;
}

/// A `ToolResult` event indexed for RAG retrieval, plus the embedding
/// computed from the turn that produced it. Concrete stores (sqlite-vec,
/// pgvector) persist this; the in-memory double keeps it in RAM.
#[derive(Clone, Debug)]
pub struct IndexedToolState {
    /// The session that produced the tool result.
    pub session_id: Uuid,
    /// Monotonic event sequence number, used as a stable primary key.
    pub seq: i64,
    /// Embedding of the triggering turn (user message + tool call name).
    pub embedding: Vec<f32>,
    /// The reconstructed `tool`-role chat message for the result.
    pub message: ChatMessage,
}

/// Storage contract for the RAG-backed tool-state tier of the memory
/// manager.
///
/// Concrete implementations are expected to delegate the vector lookup to a
/// native ANN index - `sqlite-vec` for the local edition, `pgvector` for
/// the cloud edition. The trait is intentionally narrow: it indexes
/// `ToolResult` events as they are appended and retrieves the top-k
/// semantically similar past results for the current turn's embedding.
#[async_trait]
pub trait ToolStateStore: Send + Sync {
    /// Index a `ToolResult` event for later semantic retrieval.
    ///
    /// Implementations should treat `(session_id, seq)` as a composite
    /// primary key and upsert idempotently - the memory manager re-indexes
    /// any events it observes on each assembly, so duplicate calls for
    /// the same key must not produce duplicates.
    async fn index_tool_result(&self, entry: IndexedToolState) -> EngineResult<()>;

    /// Retrieve only this exact authorized session's top-k `ToolResult` messages whose embeddings are
    /// most similar to `query_embedding`, subject to `min_similarity`.
    ///
    /// Never perform a global/team lookup here. Explicit team caches use a
    /// separate permission-scoped contract. Results are in descending similarity order. Concrete
    /// stores should cap `k` internally to a sane maximum (e.g. 8) to
    /// bound context growth.
    async fn retrieve_tool_states(
        &self,
        session_id: uuid::Uuid,
        query_embedding: &[f32],
        k: usize,
        min_similarity: f32,
    ) -> EngineResult<Vec<ChatMessage>>;
}
