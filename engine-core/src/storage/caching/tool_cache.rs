//! Server-owned tool reuse policy. HTTP caller flags can opt out, never grant reuse.
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolCacheSharing {
    #[default]
    Private,
    /// Explicit permission to share across authorized users in the same project.
    /// Never relaxes project boundaries or source-session authorization.
    Workspace,
}

/// Trusted identity supplied by the gateway, not deserialized from tool arguments.
#[derive(Clone, Debug)]
pub struct ToolCacheContext {
    pub workspace_id: Uuid,
    pub project_id: Uuid,
    pub user_id: Uuid,
    pub session_id: Uuid,
}

/// Returned by the executor that actually owns/executes the tool. Default is
/// no policy (no caching). Versions must describe its effective configuration,
/// data snapshot and credentials, never unverified caller-provided strings.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCachePolicy {
    pub read_only: bool,
    pub sharing: ToolCacheSharing,
    pub semantic_reuse: bool,
    pub ttl_seconds: i64,
    pub policy_version: String,
    pub tool_version: String,
    pub data_namespace: String,
    pub data_version: String,
    /// Opaque non-secret identifier, not the credential itself or its raw hash.
    pub credential_identity: String,
    pub credential_version: String,
}

impl ToolCachePolicy {
    pub fn is_valid(&self) -> bool {
        self.read_only
            && (1..=600).contains(&self.ttl_seconds)
            && [
                &self.policy_version,
                &self.tool_version,
                &self.data_namespace,
                &self.data_version,
                &self.credential_identity,
                &self.credential_version,
            ]
            .iter()
            .all(|s| !s.trim().is_empty() && s.len() <= 256)
    }
}

/// Local execution context resolved from persisted session metadata. The path is
/// canonical and absolute when available; missing/unresolvable paths disable reuse.
/// Local caches stay session-private, including when policy requests sharing.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LocalToolCacheContext {
    pub session_id: Uuid,
    pub project_path: Option<String>,
    pub workspace_id: Option<Uuid>,
    pub project_id: Option<Uuid>,
    pub user_id: Option<Uuid>,
    pub client_id: Option<String>,
}

/// Exact reuse envelope shared by adapters. Scope must be server-resolved;
/// adapters must separately revalidate current source access before replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExactToolCacheEnvelope {
    version: u32,
    scope: serde_json::Value,
    policy: ToolCachePolicy,
    arguments: serde_json::Value,
    result: crate::ToolResult,
}

impl ExactToolCacheEnvelope {
    pub fn key(
        scope: &serde_json::Value,
        policy: &ToolCachePolicy,
        tool: &str,
        arguments: &serde_json::Value,
    ) -> crate::EngineResult<crate::CacheKey> {
        crate::CacheKey::for_tool_call(
            "authorized-exact-tool-v1",
            &serde_json::json!({
                "scope": scope, "policy": policy, "tool": tool, "arguments": arguments
            }),
        )
    }

    pub fn entry(
        scope: serde_json::Value,
        policy: ToolCachePolicy,
        arguments: serde_json::Value,
        result: &crate::ToolResult,
    ) -> crate::EngineResult<Option<crate::cache::CacheEntry>> {
        if !policy.is_valid() || result.role != "tool" || result.is_error == Some(true) {
            return Ok(None);
        }
        let payload = serde_json::to_value(Self {
            version: 1,
            scope,
            policy,
            arguments,
            result: result.clone(),
        })?;
        Ok(Some(crate::cache::CacheEntry {
            tool_name: result.name.clone(),
            content_hash: crate::CacheKey::for_tool_call("exact-tool-envelope-v1", &payload)?.0,
            result_payload: payload,
            created_at: chrono::Utc::now(),
        }))
    }

    pub fn validate(
        entry: &crate::cache::CacheEntry,
        scope: &serde_json::Value,
        policy: &ToolCachePolicy,
        tool: &str,
        arguments: &serde_json::Value,
        call_id: &str,
    ) -> crate::EngineResult<Option<crate::ToolResult>> {
        let now = chrono::Utc::now();
        if !policy.is_valid()
            || entry.created_at > now
            || now.signed_duration_since(entry.created_at).num_seconds() >= policy.ttl_seconds
            || entry.tool_name != tool
            || entry.content_hash
                != crate::CacheKey::for_tool_call("exact-tool-envelope-v1", &entry.result_payload)?
                    .0
        {
            return Ok(None);
        }
        let Ok(mut envelope) = serde_json::from_value::<Self>(entry.result_payload.clone()) else {
            return Ok(None);
        };
        if envelope.version != 1
            || &envelope.scope != scope
            || &envelope.policy != policy
            || &envelope.arguments != arguments
            || envelope.result.name != tool
            || envelope.result.role != "tool"
            || envelope.result.is_error == Some(true)
        {
            return Ok(None);
        }
        envelope.result.tool_call_id = call_id.to_owned();
        Ok(Some(envelope.result))
    }
}
