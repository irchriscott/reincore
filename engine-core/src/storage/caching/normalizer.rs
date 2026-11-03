//! Cache-prefix normalization for provider prompt caching.
//!
//! [`CachePrefixNormalizer`] reorders and canonicalizes the static portion
//! of a request (system instructions, tool schemas, few-shot examples) so
//! that it is byte-identical across calls and positioned first in the message
//! list. This lets provider-side prompt caching (Anthropic, OpenAI, etc.)
//! recognize and reuse the prefix, eliminating redundant token billing for
//! the stable context that accompanies every turn.

use crate::error::{EngineError, EngineResult};
use crate::models::{ChatCompletionRequest, ChatMessage, MessageContent, ToolDefinition};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A prompt after cache-prefix normalization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NormalizedPrompt {
    /// The stable, byte-identical prefix that should hit provider-side prompt
    /// caching (system instructions, tool schemas, few-shot examples).
    pub static_segment: Vec<ChatMessage>,
    /// Dynamic content appended after the static segment (conversation
    /// history, tool results).
    pub dynamic_segment: Vec<ChatMessage>,
    /// SHA-256 of the serialized static segment. Exposed as a metric.
    pub prefix_hash: String,
}

/// Request-shaping middleware that enforces a stable cache prefix.
#[derive(Clone, Debug, Default)]
pub struct CachePrefixNormalizer;

impl CachePrefixNormalizer {
    pub fn new() -> Self {
        Self
    }

    /// Normalize a completion request so the static portion of the prompt is
    /// deterministic and positioned first.
    pub fn normalize(
        &self,
        request: &ChatCompletionRequest,
        system_message: Option<&str>,
        few_shots: &[ChatMessage],
    ) -> EngineResult<NormalizedPrompt> {
        let mut static_segment = Vec::new();

        if let Some(system) = system_message {
            static_segment.push(ChatMessage {
                provider_state: None,
                role: "system".to_string(),
                content: Some(MessageContent::text(system)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }

        // Tool schemas are part of the static segment. We do not include them
        // as ChatMessage entries here; instead they travel in the request's
        // `tools` field, which is already deterministic if serialized
        // canonically. We record a placeholder so the prefix hash reflects
        // that tools are present.
        if let Some(tools) = request.tools.as_ref() {
            let tool_schema_text = serialize_tool_schemas(tools)?;
            static_segment.push(ChatMessage {
                provider_state: None,
                role: "system".to_string(),
                content: Some(MessageContent::text(format!(
                    "[tool_schemas:{}]",
                    tool_schema_text
                ))),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }

        static_segment.extend_from_slice(few_shots);

        // Everything else from the original messages is dynamic.
        let dynamic_segment = request.messages.clone();

        let prefix_hash = hash_messages(&static_segment)?;

        Ok(NormalizedPrompt {
            static_segment,
            dynamic_segment,
            prefix_hash,
        })
    }

    /// Canonicalize an actual provider request in place and return the
    /// resulting stable-prefix metadata.
    ///
    /// Only leading system messages are treated as static. Conversation order
    /// is otherwise preserved, while tools and nested schema keys are sorted
    /// so semantically identical definitions serialize identically.
    pub fn normalize_request(
        &self,
        request: &mut ChatCompletionRequest,
    ) -> EngineResult<NormalizedPrompt> {
        if let Some(tools) = request.tools.as_mut() {
            for tool in tools.iter_mut() {
                tool.parameters = canonicalize_value(&tool.parameters);
            }
            tools.sort_by(|left, right| {
                left.name
                    .cmp(&right.name)
                    .then_with(|| left.description.cmp(&right.description))
            });
        }

        let static_len = request
            .messages
            .iter()
            .take_while(|message| {
                message.role == "system" && !is_dynamic_system_message(message.name.as_deref())
            })
            .count();
        let static_segment = request.messages[..static_len].to_vec();
        let dynamic_segment = request.messages[static_len..].to_vec();
        let prefix = serde_json::json!({
            "messages": &static_segment,
            "tools": &request.tools,
        });
        let prefix_hash = hex::encode(Sha256::digest(serde_json::to_vec(&prefix)?));

        Ok(NormalizedPrompt {
            static_segment,
            dynamic_segment,
            prefix_hash,
        })
    }
}

/// System messages ReinCore injects that are rewritten between turns.
///
/// These must stay out of the stable cache prefix: a compaction summary or a
/// loop intervention that changes per turn would invalidate every provider
/// cache entry positioned after it. Both the prefix hash and the Anthropic
/// request builder consume this, so the two cannot drift apart.
pub fn is_dynamic_system_message(name: Option<&str>) -> bool {
    matches!(
        name,
        Some("context_summary" | "untrusted_context_summary" | "reincore_loop_intervention")
    )
}

fn serialize_tool_schemas(tools: &[ToolDefinition]) -> EngineResult<String> {
    let mut canonical = tools.to_vec();
    for tool in canonical.iter_mut() {
        tool.parameters = canonicalize_value(&tool.parameters);
    }
    canonical.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.description.cmp(&right.description))
    });
    Ok(serde_json::to_string(&canonical)?)
}

fn canonicalize_value(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_unstable_by_key(|(key, _)| *key);
            serde_json::Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key.clone(), canonicalize_value(value)))
                    .collect(),
            )
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(canonicalize_value).collect())
        }
        scalar => scalar.clone(),
    }
}

fn hash_messages(messages: &[ChatMessage]) -> EngineResult<String> {
    let canonical =
        serde_json::to_string(messages).map_err(|e| EngineError::Normalization(e.to_string()))?;
    Ok(hex::encode(Sha256::digest(canonical.as_bytes())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ChatCompletionRequest, ChatMessage};

    #[test]
    fn normalizer_produces_stable_prefix_hash() {
        let normalizer = CachePrefixNormalizer::new();
        let req = ChatCompletionRequest {
            model: "claude-3-5-sonnet".to_string(),
            messages: vec![ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(crate::MessageContent::text("hello".to_string())),
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
        };

        let n1 = normalizer
            .normalize(&req, Some("You are a helpful assistant"), &[])
            .unwrap();
        let n2 = normalizer
            .normalize(&req, Some("You are a helpful assistant"), &[])
            .unwrap();
        assert_eq!(n1.prefix_hash, n2.prefix_hash);
    }

    #[test]
    fn normalize_request_stabilizes_tool_order_and_preserves_conversation_order() {
        let mut left = ChatCompletionRequest {
            model: "router".to_string(),
            messages: vec![
                ChatMessage {
                    provider_state: None,
                    role: "system".to_string(),
                    content: Some(crate::MessageContent::text("be helpful".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                ChatMessage {
                    provider_state: None,
                    role: "user".to_string(),
                    content: Some(crate::MessageContent::text("hello".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
            ],
            tools: Some(vec![
                ToolDefinition {
                    action_contract: None,
                    name: "z_tool".to_string(),
                    description: "z".to_string(),
                    parameters: serde_json::json!({"type": "object", "properties": {"b": {"type": "string"}, "a": {"type": "number"}}}),
                    safe_to_memoize: true,
                    memoization_policy: None,
                },
                ToolDefinition {
                    action_contract: None,
                    name: "a_tool".to_string(),
                    description: "a".to_string(),
                    parameters: serde_json::json!({"type": "object"}),
                    safe_to_memoize: true,
                    memoization_policy: None,
                },
            ]),
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        };
        let mut right = left.clone();
        right.tools.as_mut().unwrap().reverse();

        let normalizer = CachePrefixNormalizer::new();
        let left_metadata = normalizer.normalize_request(&mut left).unwrap();
        let right_metadata = normalizer.normalize_request(&mut right).unwrap();

        assert_eq!(left_metadata.prefix_hash, right_metadata.prefix_hash);
        assert_eq!(left.messages[1].content_str(), Some("hello"));
        assert_eq!(left.tools.as_ref().unwrap()[0].name, "a_tool");
    }
}
