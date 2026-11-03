//! Content-addressed tool-call memoization.
//!
//! Provides the cache key, policy, and entry types used to short-circuit
//! repeated tool calls with identical arguments. A [`CacheKey`] is derived
//! from a SHA-256 of the tool name plus canonicalized JSON arguments, making
//! it deterministic across processes. Each entry carries a
//! [`MemoizationPolicy`] (TTL, content-hash, or disabled) that decides when
//! the cached result is stale and should be re-fetched.

use crate::error::EngineResult;
use crate::models::{ChatCompletionRequest, FileContextAttachment, ToolCall, ToolResult};
use crate::provider::ProviderConfig;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Default freshness window for tool results when an endpoint does not expose
/// a tool-specific policy.
pub const DEFAULT_MEMOIZATION_TTL_SECONDS: i64 = 600;

/// Default freshness window for reusable model responses.
pub const DEFAULT_SEMANTIC_CACHE_TTL_SECONDS: i64 = 600;

/// Lowest similarity exposed as a retrieval candidate. Candidates are never
/// returned as answers unless they also clear the answer-reuse threshold.
pub const DEFAULT_SEMANTIC_RETRIEVAL_SIMILARITY: f32 = 0.80;

/// Hard upper bound for request-selected TTLs. Applications can shorten the
/// window, but cannot silently turn the response cache into permanent state.
pub const MAX_SEMANTIC_CACHE_TTL_SECONDS: i64 = 3_600;

/// Hash the non-secret provider/application configuration that can change a
/// generated answer. API credentials are intentionally excluded.
pub fn semantic_provider_config_version(
    config: &ProviderConfig,
    app_config_version: &str,
) -> EngineResult<String> {
    let value = serde_json::json!({
        "application_config_version": app_config_version,
        "provider": config.label,
        "base_url": config.base_url,
        "standard_model": config.standard_model,
        "models": config.models,
        "routing": config.routing,
        "timeout_seconds": config.timeout_seconds,
        "organization_id": config.organization_id,
    });
    let canonical = canonicalize_json(&value)?;
    Ok(hex::encode(Sha256::digest(canonical.as_bytes())))
}

/// A content-addressed cache key for tool-call memoization.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CacheKey(pub String);

impl CacheKey {
    /// Build a deterministic key from tool name + canonicalized arguments.
    pub fn for_tool_call(tool_name: &str, arguments: &serde_json::Value) -> EngineResult<Self> {
        let canonical = canonicalize_json(arguments)?;
        let mut hasher = Sha256::new();
        hasher.update(tool_name.as_bytes());
        hasher.update(b":");
        hasher.update(canonical.as_bytes());
        Ok(Self(hex::encode(hasher.finalize())))
    }
}

/// Memoization policy per tool type.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum MemoizationPolicy {
    /// Time-to-live in seconds.
    TimeTtl { ttl_seconds: i64 },
    /// Invalidated when the underlying content hash changes.
    ContentHash { content_hash: String },
    /// Never memoized (default for mutating calls).
    Disabled,
}

impl Default for MemoizationPolicy {
    fn default() -> Self {
        MemoizationPolicy::TimeTtl {
            ttl_seconds: DEFAULT_MEMOIZATION_TTL_SECONDS,
        }
    }
}

impl MemoizationPolicy {
    pub fn is_expired(&self, entry: &CacheEntry, now: DateTime<Utc>) -> bool {
        match self {
            MemoizationPolicy::TimeTtl { ttl_seconds } => {
                let ttl = Duration::seconds(*ttl_seconds);
                now > entry.created_at + ttl
            }
            MemoizationPolicy::ContentHash { content_hash } => &entry.content_hash != content_hash,
            MemoizationPolicy::Disabled => true,
        }
    }
}

/// A cached tool result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheEntry {
    pub tool_name: String,
    pub result_payload: serde_json::Value,
    pub content_hash: String,
    pub created_at: DateTime<Utc>,
}

/// Mandatory tenant/project boundary for every semantic response-cache row.
/// Unscoped requests are deliberately ineligible for response reuse.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SemanticCacheScope {
    pub workspace_id: String,
    pub project_id: String,
}

/// Output-affecting identity used to prevent reuse across materially different
/// provider requests. The individual fields are persisted for auditability;
/// [`digest`](Self::digest) is used by storage indexes.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SemanticCacheFingerprint {
    pub provider: String,
    pub resolved_model: String,
    pub system_prefix_hash: String,
    pub tool_schema_hash: String,
    pub generation_params_hash: String,
    pub application_version: String,
    pub config_version: String,
    pub source_data_version: String,
    pub immutable_workflow: Option<String>,
}

impl SemanticCacheFingerprint {
    pub fn for_request(
        request: &ChatCompletionRequest,
        provider: &str,
        resolved_model: &str,
        system_prefix_hash: &str,
        application_version: &str,
        config_version: &str,
        source_data_version: &str,
        immutable_workflow: Option<String>,
    ) -> EngineResult<Self> {
        let tool_schema_hash = tool_schema_hash(&request.tools)?;

        // Only parameters that can affect generated content belong here.
        // ReinCore bookkeeping keys (session ids, embeddings, cache controls,
        // telemetry) intentionally do not fragment otherwise identical rows.
        let mut generation = serde_json::Map::new();
        if let Some(identity) = crate::task_decisions::cache_identity(request) {
            generation.insert("task_decision_context".into(), serde_json::json!(identity));
        }
        generation.insert("temperature".into(), serde_json::json!(request.temperature));
        generation.insert("max_tokens".into(), serde_json::json!(request.max_tokens));
        generation.insert("tool_choice".into(), serde_json::json!(request.tool_choice));
        for (key, value) in &request.extra {
            // Provider-specific fields are output-affecting by default. Only
            // ReinCore's transport/telemetry/cache controls are excluded.
            if !key.starts_with("reincore_") {
                generation.insert(key.clone(), value.clone());
            }
        }
        let generation = canonicalize_json(&serde_json::Value::Object(generation))?;
        let generation_params_hash = hex::encode(Sha256::digest(generation.as_bytes()));

        Ok(Self {
            provider: provider.to_string(),
            resolved_model: resolved_model.to_string(),
            system_prefix_hash: system_prefix_hash.to_string(),
            tool_schema_hash,
            generation_params_hash,
            application_version: application_version.to_string(),
            config_version: config_version.to_string(),
            source_data_version: source_data_version.to_string(),
            immutable_workflow,
        })
    }

    pub fn digest(&self) -> EngineResult<String> {
        let canonical = canonicalize_json(&serde_json::to_value(self)?)?;
        Ok(hex::encode(Sha256::digest(canonical.as_bytes())))
    }
}

/// Stable hash of a request's declared tool schema.
///
/// Tools render before system and messages, so a change here invalidates the
/// entire cached prefix, system prompt included. A gateway aggregating several
/// MCP servers sees this change whenever one connects or drops, and recording
/// it per call is what makes that visible as churn rather than as an
/// unexplained collapse in cache reads.
pub fn tool_schema_hash(
    tools: &Option<Vec<crate::models::ToolDefinition>>,
) -> EngineResult<String> {
    let canonical = canonicalize_json(&serde_json::to_value(tools)?)?;
    Ok(hex::encode(Sha256::digest(canonical.as_bytes())))
}

/// Convert a cosine similarity to basis points, matching the bps convention
/// the rest of the diagnostics surface already uses.
pub fn similarity_to_bps(similarity: f32) -> i64 {
    (similarity as f64 * 10_000.0).round() as i64
}

/// Best candidate similarity in basis points, if anything was retrieved.
///
/// Recorded on calls that went to the provider anyway. Candidates are fetched
/// well below the reuse threshold, so this is the near-miss signal: how close
/// the threshold came to firing. It is what lowering that threshold would
/// trade against, and without it a threshold change is a guess.
pub fn best_candidate_similarity_bps(candidates: &[SemanticCacheCandidate]) -> Option<i64> {
    best_candidate(candidates).map(|candidate| similarity_to_bps(candidate.provenance.similarity))
}

/// Entry id of the candidate [`best_candidate_similarity_bps`] describes.
///
/// A shadow evaluation compares that stored answer against the one the provider
/// actually returned for the same request - ground truth about whether the reuse
/// would have been correct, taken from traffic already paid for and needing no
/// human labelling. The similarity alone cannot support that: without the id
/// there is no way to fetch the answer back. Both read the same selection, so
/// the recorded score and the recorded entry always describe one candidate.
pub fn best_candidate_entry_id(candidates: &[SemanticCacheCandidate]) -> Option<String> {
    best_candidate(candidates).map(|candidate| candidate.provenance.entry_id.clone())
}

fn best_candidate(candidates: &[SemanticCacheCandidate]) -> Option<&SemanticCacheCandidate> {
    candidates.iter().max_by(|left, right| {
        left.provenance
            .similarity
            .total_cmp(&right.provenance.similarity)
    })
}

/// Explicit reuse decision attached to every entry.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SemanticCachePolicy {
    pub safe_to_reuse: bool,
    pub ttl_seconds: i64,
    pub immutable_workflow: Option<String>,
}

impl SemanticCachePolicy {
    pub fn expires_at(&self, created_at: DateTime<Utc>) -> DateTime<Utc> {
        created_at + Duration::seconds(self.ttl_seconds.clamp(1, MAX_SEMANTIC_CACHE_TTL_SECONDS))
    }
}

/// Fully scoped response-cache row.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticCacheEntry {
    pub embedding: Vec<f32>,
    pub prompt: String,
    pub response: serde_json::Value,
    pub scope: SemanticCacheScope,
    pub fingerprint: SemanticCacheFingerprint,
    pub policy: SemanticCachePolicy,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

/// Provenance returned for both reusable answers and lower-confidence
/// retrieval candidates.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SemanticCacheProvenance {
    pub entry_id: String,
    pub prompt_hash: String,
    pub scope: SemanticCacheScope,
    pub fingerprint_hash: String,
    pub similarity: f32,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub provider: String,
    pub resolved_model: String,
    pub source_data_version: String,
    pub immutable_workflow: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticCacheHit {
    pub response: serde_json::Value,
    pub provenance: SemanticCacheProvenance,
}

/// A lower-confidence match returned for explicit retrieval/RAG use. It is a
/// distinct type so callers cannot accidentally treat it as a reusable hit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticCacheCandidate {
    pub response: serde_json::Value,
    pub provenance: SemanticCacheProvenance,
}

#[derive(Clone, Debug)]
pub struct SemanticCacheQuery {
    pub embedding: Vec<f32>,
    pub scope: SemanticCacheScope,
    pub fingerprint_hash: String,
    pub answer_threshold: f32,
    pub candidate_threshold: f32,
    pub candidate_limit: usize,
    pub now: DateTime<Utc>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SemanticCacheLookup {
    pub reusable_answer: Option<SemanticCacheHit>,
    pub candidates: Vec<SemanticCacheCandidate>,
}

/// Request-local cache plan assembled only after routing and prefix
/// normalization have resolved every output-affecting field.
#[derive(Clone, Debug)]
pub struct SemanticCachePlan {
    pub embedding: Vec<f32>,
    pub scope: SemanticCacheScope,
    pub fingerprint: SemanticCacheFingerprint,
    pub policy: SemanticCachePolicy,
    pub answer_threshold: f32,
    pub prompt: String,
}

impl SemanticCachePlan {
    pub fn query(&self, now: DateTime<Utc>) -> EngineResult<SemanticCacheQuery> {
        Ok(SemanticCacheQuery {
            embedding: self.embedding.clone(),
            scope: self.scope.clone(),
            fingerprint_hash: self.fingerprint.digest()?,
            answer_threshold: self.answer_threshold,
            candidate_threshold: DEFAULT_SEMANTIC_RETRIEVAL_SIMILARITY,
            candidate_limit: 3,
            now,
        })
    }

    pub fn entry(&self, response: serde_json::Value, now: DateTime<Utc>) -> SemanticCacheEntry {
        SemanticCacheEntry {
            embedding: self.embedding.clone(),
            prompt: self.prompt.clone(),
            response,
            scope: self.scope.clone(),
            fingerprint: self.fingerprint.clone(),
            policy: self.policy.clone(),
            created_at: now,
            expires_at: self.policy.expires_at(now),
        }
    }
}

impl CacheEntry {
    pub fn from_tool_result(tool_name: &str, result: &ToolResult) -> EngineResult<Self> {
        Ok(Self {
            tool_name: tool_name.to_string(),
            result_payload: serde_json::to_value(result)?,
            content_hash: hex::encode(Sha256::digest(result.content.as_bytes())),
            created_at: Utc::now(),
        })
    }
}

/// Trait for tool-call / MCP result memoization.
#[async_trait::async_trait]
pub trait MemoizationCache: Send + Sync {
    /// Look up a cached result for a tool call, respecting the tool's policy.
    async fn lookup(
        &self,
        tool: &ToolDefinitionForCache,
        call: &ToolCall,
    ) -> EngineResult<Option<ToolResult>>;

    /// Store a tool result if the tool is eligible for memoization.
    async fn store(
        &self,
        tool: &ToolDefinitionForCache,
        call: &ToolCall,
        result: &ToolResult,
    ) -> EngineResult<()>;

    /// Invalidate cached entries whose arguments reference a file path that
    /// has since been updated to a newer version.
    async fn invalidate_for_file_update(&self, file: &FileContextAttachment) -> EngineResult<u64>;
}

/// Minimal tool definition needed by the cache layer.
#[derive(Clone, Debug)]
pub struct ToolDefinitionForCache {
    pub name: String,
    pub safe_to_memoize: bool,
    pub policy: MemoizationPolicy,
}

/// Trait for semantic / response caching.
#[async_trait::async_trait]
pub trait SemanticCache: Send + Sync {
    /// Find scoped, fingerprint-matched, unexpired responses. Implementations
    /// must never promote `candidates` to an answer below `answer_threshold`.
    async fn lookup(&self, query: &SemanticCacheQuery) -> EngineResult<SemanticCacheLookup>;

    /// Store a fully scoped response and its explicit reuse policy.
    async fn store(&self, entry: SemanticCacheEntry) -> EngineResult<()>;

    /// Invalidate an exact workspace/project partition, optionally narrowed
    /// to one request fingerprint.
    async fn invalidate(
        &self,
        scope: &SemanticCacheScope,
        fingerprint_hash: Option<&str>,
    ) -> EngineResult<u64>;
}

/// Canonicalize a JSON value: sort object keys, compact arrays.
pub(crate) fn canonicalize_json(value: &serde_json::Value) -> EngineResult<String> {
    fn canonical_value(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                let mut entries: Vec<_> = map.iter().collect();
                entries.sort_unstable_by_key(|(key, _)| *key);
                let canonical = entries
                    .into_iter()
                    .map(|(key, value)| (key.clone(), canonical_value(value)))
                    .collect();
                serde_json::Value::Object(canonical)
            }
            serde_json::Value::Array(items) => {
                serde_json::Value::Array(items.iter().map(canonical_value).collect())
            }
            scalar => scalar.clone(),
        }
    }

    Ok(serde_json::to_string(&canonical_value(value))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ChatMessage, MessageContent, ToolDefinition};
    use chrono::Utc;

    fn request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "router".to_string(),
            messages: vec![ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(MessageContent::text("Explain cache isolation")),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            tool_choice: None,
            temperature: Some(0.2),
            max_tokens: Some(200),
            stream: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn cache_key_is_deterministic() {
        let args = serde_json::json!({"z": 1, "a": 2, "nested": {"b": 1, "a": 2}});
        let key1 = CacheKey::for_tool_call("read_file", &args).unwrap();
        let key2 = CacheKey::for_tool_call("read_file", &args).unwrap();
        assert_eq!(key1, key2);
    }

    #[test]
    fn cache_key_ignores_nested_object_insertion_order_without_stringifying_values() {
        let left = serde_json::json!({"nested": {"b": 1, "a": [true, 2]}});
        let right = serde_json::json!({"nested": {"a": [true, 2], "b": 1}});
        assert_eq!(
            CacheKey::for_tool_call("read_file", &left).unwrap(),
            CacheKey::for_tool_call("read_file", &right).unwrap()
        );
        assert_ne!(
            CacheKey::for_tool_call("read_file", &serde_json::json!({"value": 1})).unwrap(),
            CacheKey::for_tool_call("read_file", &serde_json::json!({"value": "1"})).unwrap()
        );
    }

    #[test]
    fn memoization_policy_expiry() {
        let entry = CacheEntry {
            tool_name: "weather".to_string(),
            result_payload: serde_json::json!({"temp": 72}),
            content_hash: "abc".to_string(),
            created_at: Utc::now(),
        };

        let policy = MemoizationPolicy::TimeTtl { ttl_seconds: 10 };
        assert!(!policy.is_expired(&entry, Utc::now()));
        assert!(policy.is_expired(&entry, Utc::now() + chrono::Duration::seconds(11)));
    }

    #[test]
    fn semantic_fingerprint_changes_at_every_correctness_boundary() {
        let request = request();
        let baseline = SemanticCacheFingerprint::for_request(
            &request, "openai", "gpt-a", "prefix-a", "app-a", "config-a", "source-a", None,
        )
        .unwrap();
        let baseline_digest = baseline.digest().unwrap();

        for changed in [
            SemanticCacheFingerprint::for_request(
                &request,
                "anthropic",
                "gpt-a",
                "prefix-a",
                "app-a",
                "config-a",
                "source-a",
                None,
            )
            .unwrap(),
            SemanticCacheFingerprint::for_request(
                &request, "openai", "gpt-b", "prefix-a", "app-a", "config-a", "source-a", None,
            )
            .unwrap(),
            SemanticCacheFingerprint::for_request(
                &request, "openai", "gpt-a", "prefix-b", "app-a", "config-a", "source-a", None,
            )
            .unwrap(),
            SemanticCacheFingerprint::for_request(
                &request, "openai", "gpt-a", "prefix-a", "app-b", "config-a", "source-a", None,
            )
            .unwrap(),
            SemanticCacheFingerprint::for_request(
                &request, "openai", "gpt-a", "prefix-a", "app-a", "config-b", "source-a", None,
            )
            .unwrap(),
            SemanticCacheFingerprint::for_request(
                &request, "openai", "gpt-a", "prefix-a", "app-a", "config-a", "source-b", None,
            )
            .unwrap(),
        ] {
            assert_ne!(baseline_digest, changed.digest().unwrap());
        }

        let mut generation_changed = request.clone();
        generation_changed.temperature = Some(0.8);
        assert_ne!(
            baseline_digest,
            SemanticCacheFingerprint::for_request(
                &generation_changed,
                "openai",
                "gpt-a",
                "prefix-a",
                "app-a",
                "config-a",
                "source-a",
                None,
            )
            .unwrap()
            .digest()
            .unwrap()
        );

        let mut tools_changed = request;
        tools_changed.tools = Some(vec![ToolDefinition {
            action_contract: None,
            name: "lookup".to_string(),
            description: "Lookup immutable data".to_string(),
            parameters: serde_json::json!({"type": "object"}),
            safe_to_memoize: true,
            memoization_policy: None,
        }]);
        assert_ne!(
            baseline_digest,
            SemanticCacheFingerprint::for_request(
                &tools_changed,
                "openai",
                "gpt-a",
                "prefix-a",
                "app-a",
                "config-a",
                "source-a",
                Some("immutable".to_string()),
            )
            .unwrap()
            .digest()
            .unwrap()
        );
    }

    #[test]
    fn semantic_ttl_is_bounded() {
        let created = Utc::now();
        let policy = SemanticCachePolicy {
            safe_to_reuse: true,
            ttl_seconds: i64::MAX,
            immutable_workflow: None,
        };
        assert_eq!(
            policy.expires_at(created),
            created + Duration::seconds(MAX_SEMANTIC_CACHE_TTL_SECONDS)
        );
    }
}
