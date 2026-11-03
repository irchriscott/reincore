//! Deterministic local text embeddings for the semantic duplicate cache.
//!
//! ReinCore's semantic cache historically only fired when the caller supplied
//! a `reincore_embedding` vector, which most OpenAI-compatible clients never
//! do - leaving the "is this a duplicate request?" stage of the pipeline dead
//! in practice. This module provides a zero-dependency fallback: a hashed
//! character-trigram frequency vector (the classic "hashing trick"),
//! L2-normalized into the same 384-dimensional space the cache stores.
//!
//! Properties:
//! - **Deterministic** across processes and editions (FNV-1a hashing, no
//!   random state), so local SQLite and cloud Postgres agree.
//! - **Lexical**, not semantic: two prompts only score high when their
//!   surface text overlaps heavily. That is exactly what we want for
//!   duplicate detection - paraphrases should MISS, resends should HIT.
//! - **Safe to mix** with caller-supplied model embeddings in one table:
//!   a sparse signed trigram vector and a dense model embedding are
//!   near-orthogonal with overwhelming probability, so cross-matches above
//!   the similarity thresholds are statistically implausible.

use crate::models::ChatCompletionRequest;
use crate::{
    SemanticCacheFingerprint, SemanticCachePlan, SemanticCachePolicy, SemanticCacheScope,
    DEFAULT_SEMANTIC_CACHE_TTL_SECONDS, MAX_SEMANTIC_CACHE_TTL_SECONDS,
};

/// Dimensionality of the semantic-cache vector space.
pub const EMBEDDING_DIMENSIONS: usize = 384;

/// Similarity threshold for caller-supplied embeddings used by semantic tool
/// retrieval. Response reuse has a separate, stricter correctness threshold.
pub const CLIENT_EMBEDDING_SIMILARITY: f32 = 0.92;

/// Near-exact threshold for automatic response reuse, regardless of whether
/// the vector came from the caller or the local lexical embedder.
pub const RESPONSE_CACHE_SIMILARITY: f32 = 0.97;

/// Stricter threshold for engine-generated lexical embeddings. Hashed
/// trigram vectors capture surface text only, so a cached response is
/// returned only for near-identical resends, never for loose paraphrases.
pub const AUTO_EMBEDDING_SIMILARITY: f32 = 0.97;

/// Threshold for engine-generated tool-request embeddings. The workspace,
/// request-group, and tool-name filters provide strong additional guards, so
/// this can admit modest natural-language rewording while still rejecting
/// unrelated calls.
pub const TEAM_TOOL_SEMANTIC_SIMILARITY: f32 = 0.85;

/// Result of applying the response-cache safety policy to one request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticReuseDecision {
    pub policy: SemanticCachePolicy,
    pub source_data_version: String,
}

/// Conservatively identify transcript state that can change an answer even
/// when no tools are present. A system prefix plus one substantive user turn
/// is stateless; assistant history, unknown conversational roles, or multiple
/// real user turns require fresh generation. Copilot environment scaffolding
/// is ignored because it is transport metadata rather than conversation state.
fn has_conversation_state(request: &ChatCompletionRequest) -> bool {
    let mut substantive_user_turns = 0usize;
    for message in &request.messages {
        if message.role.eq_ignore_ascii_case("system")
            || message.role.eq_ignore_ascii_case("developer")
        {
            continue;
        }
        if message.role.eq_ignore_ascii_case("user") {
            if message
                .content_str()
                .is_some_and(crate::user_content::is_user_scaffolding_only)
            {
                continue;
            }
            substantive_user_turns += 1;
            if substantive_user_turns > 1 {
                return true;
            }
            continue;
        }
        return true;
    }
    substantive_user_turns != 1
}

/// Apply the explicit `safe_to_reuse` policy. Caller-supplied embeddings are
/// deliberately irrelevant here: tools, tool traffic, multipart/live content,
/// or an explicit live-state marker remain ineligible. An application may opt
/// a named immutable workflow in through a server-side allow-list, but must
/// also provide a source-data version so changes invalidate the fingerprint.
pub fn semantic_reuse_decision(
    request: &ChatCompletionRequest,
    allowed_immutable_workflows: &[String],
) -> Option<SemanticReuseDecision> {
    if request
        .extra
        .get("reincore_semantic_cache")
        .and_then(|value| value.as_bool())
        == Some(false)
        || request
            .extra
            .get("reincore_safe_to_reuse")
            .and_then(|value| value.as_bool())
            == Some(false)
    {
        return None;
    }

    let has_tools = request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty());
    let has_tool_traffic = request.messages.iter().any(|message| {
        message.role.eq_ignore_ascii_case("tool")
            || message
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty())
    });
    let has_non_text_content = request.messages.iter().any(|message| {
        message
            .content
            .as_ref()
            .is_some_and(|content| !content.is_text())
    });
    let live_state = request
        .extra
        .get("reincore_live_state")
        .and_then(|value| value.as_bool())
        == Some(true);
    let normally_safe = !has_tools
        && !has_tool_traffic
        && !has_non_text_content
        && !live_state
        && !has_conversation_state(request);

    let workflow = request
        .extra
        .get("reincore_immutable_workflow")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let source_version = request
        .extra
        .get("reincore_source_data_version")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let immutable_opt_in = workflow.is_some_and(|name| {
        source_version.is_some() && allowed_immutable_workflows.iter().any(|item| item == name)
    });

    if !normally_safe && !immutable_opt_in {
        return None;
    }

    let ttl_seconds = request
        .extra
        .get("reincore_semantic_cache_ttl_seconds")
        .and_then(|value| value.as_i64())
        .unwrap_or(DEFAULT_SEMANTIC_CACHE_TTL_SECONDS)
        .clamp(1, MAX_SEMANTIC_CACHE_TTL_SECONDS);
    let immutable_workflow = immutable_opt_in.then(|| workflow.unwrap().to_string());
    Some(SemanticReuseDecision {
        policy: SemanticCachePolicy {
            safe_to_reuse: true,
            ttl_seconds,
            immutable_workflow,
        },
        source_data_version: source_version.unwrap_or("none").to_string(),
    })
}

/// Assemble a fully scoped semantic-cache plan. This must be called after the
/// concrete provider/model and stable prefix are known.
#[allow(clippy::too_many_arguments)]
pub fn semantic_cache_plan(
    request: &ChatCompletionRequest,
    caller_embedding: Option<Vec<f32>>,
    scope: Option<SemanticCacheScope>,
    provider: &str,
    resolved_model: &str,
    system_prefix_hash: &str,
    application_version: &str,
    config_version: &str,
    allowed_immutable_workflows: &[String],
) -> crate::EngineResult<Option<SemanticCachePlan>> {
    let Some(scope) = scope else {
        return Ok(None);
    };
    let Some(decision) = semantic_reuse_decision(request, allowed_immutable_workflows) else {
        return Ok(None);
    };
    let embedding = caller_embedding.or_else(|| embed_text(&request_embedding_text(request)));
    let Some(embedding) = embedding else {
        return Ok(None);
    };
    let answer_threshold = RESPONSE_CACHE_SIMILARITY;
    let fingerprint = SemanticCacheFingerprint::for_request(
        request,
        provider,
        resolved_model,
        system_prefix_hash,
        application_version,
        config_version,
        &decision.source_data_version,
        decision.policy.immutable_workflow.clone(),
    )?;
    let prompt = request
        .messages
        .iter()
        .rev()
        .filter(|message| message.role.eq_ignore_ascii_case("user"))
        .filter_map(|message| message.content_str().and_then(crate::session_user_request))
        .next()
        .unwrap_or_default()
        .to_string();
    Ok(Some(SemanticCachePlan {
        embedding,
        scope,
        fingerprint,
        policy: decision.policy,
        answer_threshold,
        prompt,
    }))
}

/// FNV-1a 64-bit hash - stable across platforms and processes.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Embed text as a signed hashed character-trigram frequency vector,
/// L2-normalized to unit length. Returns `None` for effectively empty text.
pub fn embed_text(text: &str) -> Option<Vec<f32>> {
    let normalized = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let chars: Vec<char> = normalized.chars().collect();
    if chars.len() < 3 {
        return None;
    }

    let mut vector = vec![0.0f32; EMBEDDING_DIMENSIONS];
    let mut buffer = String::with_capacity(12);
    for window in chars.windows(3) {
        buffer.clear();
        buffer.extend(window.iter());
        let hash = fnv1a(buffer.as_bytes());
        let bucket = (hash % EMBEDDING_DIMENSIONS as u64) as usize;
        // Signed hashing: use one hash bit as the sign so collisions cancel
        // rather than accumulate, which keeps unrelated texts near-orthogonal.
        let sign = if hash & (1 << 63) == 0 { 1.0 } else { -1.0 };
        vector[bucket] += sign;
    }

    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm == 0.0 {
        return None;
    }
    for value in &mut vector {
        *value /= norm;
    }
    Some(vector)
}

/// Build a deterministic fallback embedding for a tool request. At least one
/// argument must contain a natural-language phrase; identifier-like arguments
/// such as file paths and record ids require exact matching because lexical
/// similarity does not imply they identify the same resource. Tool names are
/// included to reinforce the database's exact tool-name filter.
pub fn embed_tool_request(tool_name: &str, arguments: &serde_json::Value) -> Option<Vec<f32>> {
    fn has_natural_language_phrase(value: &serde_json::Value) -> bool {
        match value {
            serde_json::Value::String(text) => {
                text.len() >= 12 && text.split_whitespace().count() >= 3
            }
            serde_json::Value::Array(values) => values.iter().any(has_natural_language_phrase),
            serde_json::Value::Object(values) => values.values().any(has_natural_language_phrase),
            _ => false,
        }
    }

    if !has_natural_language_phrase(arguments) {
        return None;
    }
    embed_text(&format!("tool:{tool_name}\narguments:{arguments}"))
}

/// Build the canonical text a request is embedded from: every message's role
/// and content, in order. Two requests embed identically only when their
/// full conversations match.
fn request_embedding_text(request: &ChatCompletionRequest) -> String {
    let mut text = String::new();
    for message in &request.messages {
        if message.role.eq_ignore_ascii_case("user")
            && message
                .content_str()
                .is_some_and(crate::user_content::is_user_scaffolding_only)
        {
            continue;
        }
        text.push_str(&message.role);
        text.push(':');
        if let Some(content) = message.content_str() {
            let content = if message.role.eq_ignore_ascii_case("user") {
                crate::user_content::session_user_request(content).unwrap_or_default()
            } else {
                content
            };
            text.push_str(content);
        }
        text.push('\n');
    }
    text
}

/// Generate a fallback embedding for a request when the caller did not
/// supply one. Returns `None` when the request is not safe to memoize:
/// tool-using conversations depend on live tool results and must never be
/// served from the duplicate cache.
pub fn auto_embedding_for_request(request: &ChatCompletionRequest) -> Option<Vec<f32>> {
    if semantic_reuse_decision(request, &[]).is_none() {
        return None;
    }
    embed_text(&request_embedding_text(request))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ChatMessage;
    use crate::MessageContent;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn request_with_content(content: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "router".to_string(),
            messages: vec![ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(MessageContent::text(content)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn identical_text_embeds_identically() {
        let a = embed_text("Summarize the quarterly revenue report for me").unwrap();
        let b = embed_text("Summarize the quarterly  revenue report for me").unwrap();
        assert!(cosine(&a, &b) > 0.999);
    }

    #[test]
    fn unrelated_text_scores_low() {
        let a = embed_text("Summarize the quarterly revenue report for me").unwrap();
        let b = embed_text("Write a haiku about distributed consensus algorithms").unwrap();
        assert!(cosine(&a, &b) < AUTO_EMBEDDING_SIMILARITY);
    }

    #[test]
    fn embeddings_are_unit_length() {
        let v = embed_text("hello semantic cache world").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn similar_tool_requests_have_similar_embeddings() {
        let a = embed_tool_request(
            "search_code",
            &serde_json::json!({"query": "find authentication middleware in the codebase"}),
        )
        .unwrap();
        let b = embed_tool_request(
            "search_code",
            &serde_json::json!({"query": "search the codebase for authentication middleware"}),
        )
        .unwrap();
        let similarity = cosine(&a, &b);
        assert!(
            similarity >= TEAM_TOOL_SEMANTIC_SIMILARITY,
            "expected related tool requests to match, got {similarity}"
        );
    }

    #[test]
    fn identifier_only_tool_requests_require_exact_matching() {
        assert!(
            embed_tool_request("read_file", &serde_json::json!({"path": "src/auth.rs"})).is_none()
        );
        assert!(embed_tool_request(
            "get_record",
            &serde_json::json!({"record_id": "customer_12345"})
        )
        .is_none());
    }

    #[test]
    fn tool_using_requests_are_not_auto_embedded() {
        let mut request = request_with_content("run the build");
        request.tools = Some(vec![crate::models::ToolDefinition {
            action_contract: None,
            name: "run_build".to_string(),
            description: "Runs the build".to_string(),
            parameters: serde_json::json!({"type": "object"}),
            safe_to_memoize: false,
            memoization_policy: None,
        }]);
        assert!(auto_embedding_for_request(&request).is_none());
    }

    #[test]
    fn caller_embedding_does_not_make_tool_request_reusable() {
        let mut request = request_with_content("run the build");
        request.tools = Some(vec![crate::models::ToolDefinition {
            action_contract: None,
            name: "run_build".to_string(),
            description: "Runs the build".to_string(),
            parameters: serde_json::json!({"type": "object"}),
            safe_to_memoize: false,
            memoization_policy: None,
        }]);
        request.extra.insert(
            "reincore_embedding".to_string(),
            serde_json::json!(vec![0.0; EMBEDDING_DIMENSIONS]),
        );
        assert!(semantic_reuse_decision(&request, &[]).is_none());
    }

    #[test]
    fn caller_embedding_does_not_make_conversation_state_reusable() {
        let mut request = request_with_content("Can you explain that again?");
        request.messages.insert(
            0,
            ChatMessage {
                provider_state: None,
                role: "assistant".to_string(),
                content: Some(MessageContent::text("Earlier answer")),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        );
        request.extra.insert(
            "reincore_embedding".to_string(),
            serde_json::json!(vec![0.0; EMBEDDING_DIMENSIONS]),
        );

        assert!(semantic_reuse_decision(&request, &[]).is_none());
    }

    #[test]
    fn multiple_substantive_user_turns_are_live_state() {
        let mut request = request_with_content("Now compare it to the second option");
        request.messages.insert(
            0,
            ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(MessageContent::text("Evaluate the first option")),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        );

        assert!(semantic_reuse_decision(&request, &[]).is_none());
    }

    #[test]
    fn allowlisted_immutable_workflow_requires_source_version() {
        let mut request = request_with_content("run the immutable report");
        request.tools = Some(vec![crate::models::ToolDefinition {
            action_contract: None,
            name: "read_snapshot".to_string(),
            description: "Reads an immutable snapshot".to_string(),
            parameters: serde_json::json!({"type": "object"}),
            safe_to_memoize: true,
            memoization_policy: None,
        }]);
        request.extra.insert(
            "reincore_immutable_workflow".to_string(),
            serde_json::json!("monthly-report"),
        );
        let allowlist = vec!["monthly-report".to_string()];
        assert!(semantic_reuse_decision(&request, &allowlist).is_none());

        request.extra.insert(
            "reincore_source_data_version".to_string(),
            serde_json::json!("snapshot-42"),
        );
        let decision = semantic_reuse_decision(&request, &allowlist).unwrap();
        assert_eq!(
            decision.policy.immutable_workflow.as_deref(),
            Some("monthly-report")
        );
        assert_eq!(decision.source_data_version, "snapshot-42");
    }

    #[test]
    fn opt_out_disables_auto_embedding() {
        let mut request = request_with_content("hello there, general assistant");
        request.extra.insert(
            "reincore_semantic_cache".to_string(),
            serde_json::json!(false),
        );
        assert!(auto_embedding_for_request(&request).is_none());
    }

    #[test]
    fn plain_requests_are_auto_embedded_with_correct_dimensions() {
        let request = request_with_content("what is the capital of France?");
        let embedding = auto_embedding_for_request(&request).unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIMENSIONS);
    }

    #[test]
    fn copilot_scaffolding_does_not_change_the_cache_embedding() {
        let plain = request_with_content("Build the Watoto Church home page");
        let mut wrapped = request_with_content(
            "<context>2026-08-18</context><reminderInstructions>Use replace edits.</reminderInstructions><userRequest>Build the Watoto Church home page</userRequest>",
        );
        wrapped.messages.insert(
            0,
            ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(crate::MessageContent::text(
                    "<environment_info>Windows</environment_info><workspace_info>app</workspace_info>"
                        .to_string(),
                )),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        );

        assert_eq!(
            auto_embedding_for_request(&plain),
            auto_embedding_for_request(&wrapped)
        );
    }
}
