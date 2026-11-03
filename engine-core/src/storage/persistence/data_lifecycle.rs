//! Conservative operator-only session retention contract.
/// Tables with session-owned payloads. Ledger rows deliberately survive retention.
pub const SESSION_TABLES: &[&str] = &[
    "session_file_roots",
    "session_task_events",
    "session_file_history",
    "session_tool_call_links",
    "tool_state_index",
    "checkpoints",
    "session_conversation_bindings",
    "batch_jobs",
    "cost_estimates",
    "completion_accounting",
    "session_events",
];

/// Complete the lineage lookup indexes so retention does not scan every session
/// for each candidate when checking the parent/root/subagent OR predicate.
pub fn retention_indexes(version: i64) -> crate::migration::Migration {
    crate::migration::Migration::new(
        version,
        "Index retention parent lookups",
        vec![
            crate::migration::SchemaChange::create_index_if_not_exists(
                "idx_sessions_parent_session_id",
                "sessions",
                vec!["parent_session_id"],
            ),
            crate::migration::SchemaChange::create_index_if_not_exists(
                "idx_conversation_bindings_parent",
                "session_conversation_bindings",
                vec!["parent_binding_key"],
            ),
        ],
    )
}

/// Refuse linked sessions rather than silently cascading across a conversation.
pub fn eligibility(postgres: bool) -> String {
    let id = if postgres { "s.id::text" } else { "s.id" };
    let now = if postgres {
        "CURRENT_TIMESTAMP"
    } else {
        "strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')"
    };
    let active = if postgres { "TRUE" } else { "1" };
    format!("s.status = 'completed' AND s.is_subagent = 0
      AND s.parent_session_id IS NULL AND (s.root_session_id IS NULL OR s.root_session_id = s.id)
      AND s.subagent_parent_session_id IS NULL
      AND NOT EXISTS (SELECT 1 FROM sessions child WHERE child.id <> s.id AND
          (child.parent_session_id = s.id OR child.root_session_id = s.id OR child.subagent_parent_session_id = {id} OR child.subagent_parent_session_id = s.external_session_id))
      AND NOT EXISTS (SELECT 1 FROM completion_accounting a WHERE a.session_id = {id} AND a.settled = 0)
      AND NOT EXISTS (SELECT 1 FROM batch_jobs b WHERE b.session_id = s.id AND b.active = {active})
      AND NOT EXISTS (SELECT 1 FROM checkpoints c WHERE c.session_id = s.id AND c.expires_at > {now})
      AND NOT EXISTS (SELECT 1 FROM session_conversation_bindings parent JOIN session_conversation_bindings child ON child.parent_binding_key = parent.binding_key WHERE parent.session_id = s.id)")
}
