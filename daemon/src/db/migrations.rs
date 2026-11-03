//! SQLite schema migrations for the local ReinCore engine.
//!
//! The schema is five squashed versions (`all_migrations`). Later migrations'
//! effects on earlier tables are folded directly into the `CREATE TABLE`
//! statements (provider pricing columns, session lineage columns, fixed-point
//! cost columns, file-identity columns), and superseded steps are reduced to
//! their net effect (the semantic-cache rebuilds become "drop legacy, create
//! canonical"; the short-lived `_v2` table is never created).
//!
//! Upgrade story: databases created by older releases already contain the
//! final schema and record a higher `MAX(version)`, so `pending_sql` yields
//! nothing for them. Fresh databases apply versions 0-4 and converge on the
//! identical schema; the default database path also moved to
//! `~/.reincore-io/reincore.db`, so existing installs start fresh there.
//! Add new migrations by appending to `all_migrations()`.

use engine_core::migration::{ColumnDef, DataType, Migration, SchemaChange};

/// Version 0: core tables. Later migrations' column additions are folded in
/// here so a fresh database is created at the final shape in one step:
///
/// - `sessions`: project/lineage tracking, interaction mode, cost
///   calibration, cache-hit rate, project link, cumulative tokens, subagent
///   tracking, fixed-point costs.
/// - `cost_estimates`: request-group tracking.
/// - `ledger_events`: accounting link and fixed-point cost provenance.
pub fn baseline_migration() -> Migration {
    use DataType::*;

    Migration::new(
        0,
        "Baseline local ReinCore schema",
        vec![
            SchemaChange::create_table_if_not_exists(
                "sessions",
                vec![
                    ColumnDef::new("id", Uuid).primary_key(),
                    ColumnDef::new("task_summary", Text),
                    ColumnDef::new("project_path", Text).nullable(),
                    ColumnDef::new("status", VarChar { max_chars: 32 }),
                    ColumnDef::new("budget_cents", Integer).default("0"),
                    ColumnDef::new("spend_cents", Integer).default("0"),
                    ColumnDef::new("tier", Integer).default("1"),
                    ColumnDef::new("created_at", Timestamp),
                    ColumnDef::new("updated_at", Timestamp),
                    // Project grouping, external-app identity, and
                    // physical-session lineage for context-window rollover.
                    ColumnDef::new("project_name", Text).nullable(),
                    ColumnDef::new("client_id", Text).nullable(),
                    ColumnDef::new("external_session_id", Text).nullable(),
                    ColumnDef::new("parent_session_id", Text).nullable(),
                    ColumnDef::new("root_session_id", Text).nullable(),
                    ColumnDef::new("rollover_index", Integer).default("0"),
                    ColumnDef::new("inherited_spend_cents", Integer).default("0"),
                    ColumnDef::new("context_tokens", Integer).default("0"),
                    // Batch-eligible routing mode.
                    ColumnDef::new("mode", VarChar { max_chars: 32 }).default("interactive"),
                    // Pre-flight cost-prediction calibration feedback.
                    ColumnDef::new("predicted_high_cents", Integer).nullable(),
                    ColumnDef::new("cost_accuracy_bps", Integer).nullable(),
                    ColumnDef::new("cost_accuracy_samples", Integer).nullable(),
                    // Running cache-prefix hit rate for the dashboard gauge.
                    ColumnDef::new("cache_hit_rate_bps", Integer).nullable(),
                    ColumnDef::new("cache_hit_rate_samples", Integer).nullable(),
                    // Durable local project registry identity.
                    ColumnDef::new("project_id", Text).nullable(),
                    // Cumulative token totals for the session detail view.
                    ColumnDef::new("total_input_tokens", Integer).default("0"),
                    ColumnDef::new("total_output_tokens", Integer).default("0"),
                    // Subagent origin and spawn correlation.
                    ColumnDef::new("is_subagent", Integer).default("0"),
                    ColumnDef::new("subagent_parent_session_id", Text)
                        .nullable()
                        .default("NULL"),
                    ColumnDef::new("subagent_spawn_tool_call_id", Text)
                        .nullable()
                        .default("NULL"),
                    ColumnDef::new("subagent_spawn_request_group_id", Text)
                        .nullable()
                        .default("NULL"),
                    // Fixed-point usage cost and pricing provenance.
                    ColumnDef::new("spend_nano_usd", Integer).nullable(),
                    ColumnDef::new("inherited_spend_nano_usd", Integer).nullable(),
                    ColumnDef::new("unknown_cost_calls", Integer).default("0"),
                ],
            ),
            SchemaChange::create_table_if_not_exists(
                "session_events",
                vec![
                    // A monotonic id, so the log has a total order.
                    //
                    // (session_id, seq) only orders within one session, and
                    // created_at ties and cannot be paged safely. Anything
                    // that consumes the whole log needs a single column it
                    // can page by.
                    ColumnDef::new("id", Integer).primary_key().auto_increment(),
                    ColumnDef::new("session_id", Uuid),
                    ColumnDef::new("seq", Integer),
                    ColumnDef::new("event_type", VarChar { max_chars: 32 }),
                    ColumnDef::new("payload", Json),
                    ColumnDef::new("cost_cents", Integer).nullable(),
                    ColumnDef::new("created_at", Timestamp),
                ],
            ),
            SchemaChange::create_table_if_not_exists(
                "cost_estimates",
                vec![
                    ColumnDef::new("id", Integer).primary_key().auto_increment(),
                    ColumnDef::new("session_id", Uuid).nullable(),
                    ColumnDef::new("request_type", VarChar { max_chars: 64 }),
                    ColumnDef::new("model_tier", VarChar { max_chars: 64 }),
                    ColumnDef::new("input_tokens", Integer),
                    ColumnDef::new("output_tokens", Integer),
                    ColumnDef::new("low_cents", Integer),
                    ColumnDef::new("high_cents", Integer),
                    ColumnDef::new("created_at", Timestamp),
                    // Groups every LLM sub-call of one user request so they
                    // can be totaled deterministically.
                    ColumnDef::new("request_group_id", VarChar { max_chars: 128 }).nullable(),
                ],
            ),
            SchemaChange::create_table_if_not_exists(
                "ledger_events",
                vec![
                    ColumnDef::new("id", Integer).primary_key().auto_increment(),
                    ColumnDef::new("session_id", Uuid),
                    ColumnDef::new("amount_cents", Integer),
                    ColumnDef::new("reason", VarChar { max_chars: 64 }),
                    ColumnDef::new("created_at", Timestamp),
                    // Durable idempotent completion accounting link.
                    ColumnDef::new("accounting_id", Text).nullable(),
                    // Fixed-point cost and pricing provenance.
                    ColumnDef::new("amount_nano_usd", Integer).nullable(),
                    ColumnDef::new("cost", Text).nullable(),
                ],
            ),
            SchemaChange::create_table_if_not_exists(
                "tool_cache",
                vec![
                    ColumnDef::new("cache_key", VarChar { max_chars: 512 }).primary_key(),
                    ColumnDef::new("tool_name", VarChar { max_chars: 128 }),
                    ColumnDef::new("result_payload", Json),
                    ColumnDef::new("content_hash", VarChar { max_chars: 128 }),
                    ColumnDef::new("created_at", Timestamp),
                ],
            ),
            SchemaChange::create_table_if_not_exists(
                "checkpoints",
                vec![
                    ColumnDef::new("token", VarChar { max_chars: 512 }).primary_key(),
                    ColumnDef::new("session_id", Uuid),
                    ColumnDef::new("last_event_seq", Integer),
                    ColumnDef::new("expires_at", Timestamp),
                    ColumnDef::new("created_at", Timestamp),
                ],
            ),
            SchemaChange::create_table_if_not_exists(
                "budget_policies",
                vec![
                    ColumnDef::new("id", Uuid).primary_key(),
                    ColumnDef::new("scope", VarChar { max_chars: 32 }),
                    ColumnDef::new("scope_key", VarChar { max_chars: 512 }),
                    ColumnDef::new("budget_cents", Integer),
                    ColumnDef::new("created_at", Timestamp),
                    ColumnDef::new("updated_at", Timestamp),
                ],
            ),
            SchemaChange::create_table_if_not_exists(
                "budget_history",
                vec![
                    ColumnDef::new("id", Integer).primary_key().auto_increment(),
                    ColumnDef::new("policy_id", Uuid),
                    ColumnDef::new("scope", VarChar { max_chars: 32 }),
                    ColumnDef::new("scope_key", VarChar { max_chars: 512 }),
                    ColumnDef::new("previous_budget_cents", Integer),
                    ColumnDef::new("new_budget_cents", Integer),
                    ColumnDef::new("updated_at", Timestamp),
                ],
            ),
            SchemaChange::create_table_if_not_exists(
                "engine_settings",
                vec![
                    ColumnDef::new("key", VarChar { max_chars: 128 }).primary_key(),
                    ColumnDef::new("value", Json),
                ],
            ),
            SchemaChange::create_index(
                "idx_session_events_session_id",
                "session_events",
                vec!["session_id"],
            ),
            SchemaChange::create_index(
                "idx_ledger_session_id",
                "ledger_events",
                vec!["session_id"],
            ),
            SchemaChange::create_index("idx_tool_cache_tool_name", "tool_cache", vec!["tool_name"]),
            SchemaChange::create_index(
                "idx_cost_estimates_session_id",
                "cost_estimates",
                vec!["session_id"],
            ),
            SchemaChange::create_unique_index_if_not_exists(
                "idx_budget_policies_scope_key",
                "budget_policies",
                vec!["scope", "scope_key"],
            ),
        ],
    )
}

/// Version 1: provider model catalog and durable completion accounting.
///
/// The versioned pricing catalog columns (`pricing`, `aliases`,
/// `auto_select`, `notes`) are folded into the table definition: existing
/// rows have no catalog data and the upgrade preserved them untouched, so a
/// fresh table with nullable catalog columns is the identical end state.
pub fn provider_accounting_migration() -> Migration {
    use DataType::*;

    Migration::new(
        1,
        "Provider model catalog and completion accounting",
        vec![
            SchemaChange::create_table_if_not_exists(
                "provider_models",
                vec![
                    ColumnDef::new("id", Uuid).primary_key(),
                    ColumnDef::new("provider", VarChar { max_chars: 64 }),
                    ColumnDef::new("model_name", VarChar { max_chars: 128 }),
                    ColumnDef::new("model_id", VarChar { max_chars: 128 }),
                    ColumnDef::new("input_price", Float),
                    ColumnDef::new("output_price", Float),
                    ColumnDef::new("rank", Integer),
                    ColumnDef::new("max_input_tokens", Integer),
                    ColumnDef::new("max_output_tokens", Integer),
                    ColumnDef::new("reasoning", VarChar { max_chars: 32 }).default("none"),
                    ColumnDef::new("speed", VarChar { max_chars: 32 }),
                    ColumnDef::new("supported_input_data", Json),
                    ColumnDef::new("supported_output_data", Json),
                    ColumnDef::new("created_at", Timestamp),
                    ColumnDef::new("updated_at", Timestamp),
                    ColumnDef::new("pricing", Json).nullable(),
                    ColumnDef::new("aliases", Json).nullable(),
                    ColumnDef::new("auto_select", Boolean).default("TRUE"),
                    ColumnDef::new("notes", Text).nullable(),
                ],
            ),
            SchemaChange::create_unique_index_if_not_exists(
                "idx_provider_models_provider_model_id",
                "provider_models",
                vec!["provider", "model_id"],
            ),
            SchemaChange::create_table_if_not_exists(
                "completion_accounting",
                vec![
                    ColumnDef::new("id", Text).primary_key(),
                    ColumnDef::new("session_id", Text),
                    ColumnDef::new("record", Text),
                    ColumnDef::new("settled", Integer).default("0"),
                ],
            ),
            SchemaChange::create_index_if_not_exists(
                "idx_completion_accounting_session",
                "completion_accounting",
                vec!["session_id", "settled"],
            ),
            SchemaChange::create_index_if_not_exists(
                "idx_completion_accounting_pending",
                "completion_accounting",
                vec!["settled", "id"],
            ),
            SchemaChange::create_unique_index_if_not_exists(
                "idx_ledger_accounting_id",
                "ledger_events",
                vec!["accounting_id"],
            ),
            SchemaChange::create_index_if_not_exists(
                "idx_cost_estimates_request_group_id",
                "cost_estimates",
                vec!["request_group_id"],
            ),
        ],
    )
}

/// Version 2: local project registry plus every lookup index on `sessions`.
///
/// The registry table with its composite primary key keeps its original raw
/// definition verbatim. The session-page indexes and the continuation-parent
/// index are split out of their original multi-statement raw migrations so
/// each version only touches one concern.
pub fn session_lineage_migration() -> Migration {
    use DataType::*;

    Migration::new(
        2,
        "Local project registry and session lookup indexes",
        vec![
            SchemaChange::create_table_if_not_exists(
                "local_projects",
                vec![
                    ColumnDef::new("id", Text).primary_key(),
                    ColumnDef::new("name", Text),
                    ColumnDef::new("created_at", Timestamp),
                    ColumnDef::new("updated_at", Timestamp),
                ],
            ),
            // The local_project_directories table has a composite primary key
            // (project_id, normalized_path) which the DSL doesn't support, so
            // it keeps its original raw definition.
            SchemaChange::Raw {
                sqlite: "CREATE TABLE IF NOT EXISTS local_project_directories (\n  project_id TEXT NOT NULL,\n  project_path TEXT NOT NULL,\n  normalized_path TEXT NOT NULL UNIQUE,\n  created_at TEXT NOT NULL,\n  PRIMARY KEY (project_id, normalized_path),\n  FOREIGN KEY (project_id) REFERENCES local_projects(id) ON DELETE CASCADE\n);\nCREATE INDEX IF NOT EXISTS idx_local_project_directories_project_id ON local_project_directories(project_id);".to_string(),
                postgres: "".to_string(),
            },
            SchemaChange::create_index_if_not_exists(
                "idx_sessions_external_tracking",
                "sessions",
                vec!["client_id", "project_name", "external_session_id", "status"],
            ),
            SchemaChange::create_index_if_not_exists(
                "idx_sessions_project_name",
                "sessions",
                vec!["project_name"],
            ),
            SchemaChange::create_index_if_not_exists(
                "idx_sessions_root_session_id",
                "sessions",
                vec!["root_session_id"],
            ),
            SchemaChange::create_index_if_not_exists("idx_sessions_mode", "sessions", vec!["mode"]),
            SchemaChange::create_index_if_not_exists(
                "idx_sessions_cost_accuracy_samples",
                "sessions",
                vec!["cost_accuracy_samples"],
            ),
            SchemaChange::create_index_if_not_exists(
                "idx_sessions_cache_hit_rate_samples",
                "sessions",
                vec!["cache_hit_rate_samples"],
            ),
            SchemaChange::create_index_if_not_exists(
                "idx_sessions_project_id",
                "sessions",
                vec!["project_id"],
            ),
            SchemaChange::create_index_if_not_exists(
                "idx_sessions_subagent_parent",
                "sessions",
                vec!["subagent_parent_session_id"],
            ),
            SchemaChange::create_index_if_not_exists(
                "idx_sessions_subagent_spawn_tool_call",
                "sessions",
                vec!["subagent_spawn_tool_call_id"],
            ),
            SchemaChange::create_index_if_not_exists(
                "idx_sessions_parent_session_id",
                "sessions",
                vec!["parent_session_id"],
            ),
            // Session half of the query-performance and continuation indexes;
            // the event-log halves live in version 4.
            SchemaChange::Raw {
                sqlite: "CREATE INDEX IF NOT EXISTS idx_sessions_page ON sessions(updated_at DESC, id DESC) WHERE is_subagent = 0; CREATE INDEX IF NOT EXISTS idx_sessions_project_page ON sessions(project_id, updated_at DESC, id DESC) WHERE is_subagent = 0;".into(),
                postgres: "CREATE INDEX IF NOT EXISTS idx_sessions_page ON sessions(updated_at DESC, id DESC) WHERE is_subagent = 0; CREATE INDEX IF NOT EXISTS idx_sessions_project_page ON sessions(project_id, updated_at DESC, id DESC) WHERE is_subagent = 0; CREATE INDEX IF NOT EXISTS idx_sessions_workspace_page ON sessions(workspace_id, updated_at DESC, id DESC) WHERE is_subagent = 0; CREATE INDEX IF NOT EXISTS idx_sessions_user_page ON sessions(user_id, updated_at DESC, id DESC) WHERE is_subagent = 0;".into(),
            },
            SchemaChange::Raw {
                sqlite: "CREATE INDEX IF NOT EXISTS idx_sessions_continuation_parent ON sessions(parent_session_id,rollover_index);".into(),
                postgres: "CREATE INDEX IF NOT EXISTS idx_sessions_continuation_parent ON sessions(parent_session_id,rollover_index);".into(),
            },
        ],
    )
}

fn canonical_semantic_response_cache_sql() -> String {
    "CREATE VIRTUAL TABLE IF NOT EXISTS semantic_response_cache USING vec0(\n  entry_id TEXT PRIMARY KEY,\n  workspace_id TEXT PARTITION KEY,\n  project_id TEXT PARTITION KEY,\n  fingerprint_hash TEXT,\n  safe_to_reuse BOOLEAN,\n  expires_at INTEGER,\n  embedding float[384] distance_metric=cosine,\n  +fingerprint_json TEXT,\n  +prompt TEXT,\n  +response TEXT,\n  +created_at TEXT\n);".to_string()
}

/// Version 3: conversation bindings, event projections, durable titles and
/// batch jobs, the tool-state vector tier, and the hardened semantic cache.
///
/// Later migrations' effects are folded in:
/// - `session_tool_call_links` is created with the file-history identity
///   columns instead of adding them afterwards.
/// - The two semantic-cache rebuilds collapse to "drop the legacy table and
///   the short-lived `_v2` table, create the canonical table". Rows from the
///   legacy table were never carried over (they lacked scope, fingerprint,
///   policy, and expiry data), so dropping first is the identical end state.
pub fn conversation_memory_migration() -> Migration {
    use DataType::*;

    Migration::new(
        3,
        "Conversation bindings, projections, and semantic memory",
        vec![
            SchemaChange::create_table_if_not_exists(
                "session_conversation_bindings",
                vec![
                    ColumnDef::new("binding_key", Text).primary_key(),
                    ColumnDef::new("client_id", Text),
                    ColumnDef::new("scope_key", Text),
                    ColumnDef::new("conversation_id_hash", Text),
                    ColumnDef::new("session_id", Uuid),
                    ColumnDef::new("is_subagent", Boolean).default("false"),
                    ColumnDef::new("parent_binding_key", Text).nullable(),
                    ColumnDef::new("spawn_tool_call_id", Text).nullable(),
                    ColumnDef::new("created_at", Timestamp),
                    ColumnDef::new("updated_at", Timestamp),
                ],
            ),
            SchemaChange::create_index_if_not_exists(
                "idx_conversation_bindings_session",
                "session_conversation_bindings",
                vec!["session_id"],
            ),
            SchemaChange::create_index_if_not_exists(
                "idx_conversation_bindings_spawn",
                "session_conversation_bindings",
                vec!["spawn_tool_call_id"],
            ),
            SchemaChange::create_index_if_not_exists(
                "idx_conversation_bindings_parent",
                "session_conversation_bindings",
                vec!["parent_binding_key"],
            ),
            // Compact request-time projection tables. Historical events are
            // intentionally not backfilled; the tables populate as new events
            // are appended.
            SchemaChange::create_table_if_not_exists(
                "session_task_events",
                vec![
                    ColumnDef::new("session_id", Uuid),
                    ColumnDef::new("seq", Integer),
                    ColumnDef::new("request_group_id", Text).nullable(),
                    ColumnDef::new("task_id", Text).nullable(),
                    ColumnDef::new("task_name", Text).nullable(),
                    ColumnDef::new("task_type", Text).nullable(),
                    ColumnDef::new("user_prompt", Text).nullable(),
                    ColumnDef::new("response_text", Text).nullable(),
                    ColumnDef::new("response_has_tool_calls", Boolean).default("0"),
                    ColumnDef::new("input_tokens", Integer).default("0"),
                    ColumnDef::new("output_tokens", Integer).default("0"),
                    ColumnDef::new("cost_cents", Integer).nullable(),
                    ColumnDef::new("created_at", Timestamp),
                ],
            ),
            SchemaChange::create_unique_index_if_not_exists(
                "idx_session_task_events_identity",
                "session_task_events",
                vec!["session_id", "seq"],
            ),
            SchemaChange::create_index_if_not_exists(
                "idx_session_task_events_recent",
                "session_task_events",
                vec!["session_id", "seq"],
            ),
            SchemaChange::create_table_if_not_exists(
                "session_tool_call_links",
                vec![
                    ColumnDef::new("session_id", Uuid),
                    ColumnDef::new("tool_call_id", Text),
                    ColumnDef::new("event_seq", Integer),
                    ColumnDef::new("request_group_id", Text).nullable(),
                    ColumnDef::new("tool_call", Json),
                    ColumnDef::new("delegation_profile", Text).nullable(),
                    ColumnDef::new("origin_llm_call_seq", Integer).nullable(),
                    ColumnDef::new("result_event_seq", Integer).nullable(),
                    ColumnDef::new("requested_at", Timestamp),
                    ColumnDef::new("call_identity", Text).default("''"),
                    ColumnDef::new("call_ambiguous", Integer).default("0"),
                ],
            ),
            SchemaChange::create_unique_index_if_not_exists(
                "idx_session_tool_call_links_identity",
                "session_tool_call_links",
                vec!["session_id", "tool_call_id"],
            ),
            SchemaChange::create_index_if_not_exists(
                "idx_session_tool_call_links_group",
                "session_tool_call_links",
                vec!["session_id", "request_group_id"],
            ),
            SchemaChange::create_table_if_not_exists(
                "generated_titles",
                vec![
                    ColumnDef::new("title_key", Text).primary_key(),
                    ColumnDef::new("title", Text),
                    ColumnDef::new("original_title", Text),
                ],
            ),
            SchemaChange::create_table_if_not_exists(
                "batch_jobs",
                vec![
                    ColumnDef::new("id", Uuid).primary_key(),
                    ColumnDef::new("session_id", Uuid),
                    ColumnDef::new("revision", Integer).default("0"),
                    ColumnDef::new("active", Boolean).default("TRUE"),
                    ColumnDef::new("record", Text),
                ],
            ),
            // RAG tool-state memory tier (sqlite-vec `vec0`, cosine
            // distance), keyed by `(session_id, seq)` for idempotent upserts.
            SchemaChange::Raw {
                sqlite: "CREATE VIRTUAL TABLE IF NOT EXISTS tool_state_index USING vec0(\n  embedding float[384] distance_metric=cosine,\n  +session_id TEXT,\n  +seq INTEGER,\n  +role TEXT,\n  +content TEXT,\n  +tool_call_id TEXT,\n  +name TEXT,\n  +created_at TEXT\n);".to_string(),
                postgres: "".to_string(),
            },
            SchemaChange::Raw {
                sqlite: "DROP TABLE IF EXISTS semantic_cache;".to_string(),
                postgres: String::new(),
            },
            SchemaChange::Raw {
                sqlite: "DROP TABLE IF EXISTS semantic_response_cache_v2;".to_string(),
                postgres: String::new(),
            },
            SchemaChange::Raw {
                sqlite: canonical_semantic_response_cache_sql(),
                postgres: String::new(),
            },
        ],
    )
}

/// Version 4: structured file-mutation history plus every query index on the
/// event log.
///
/// Later migrations' effects are folded in:
/// - `session_file_history` is created with the portable `file_key` column
///   instead of adding it afterwards.
/// - The portable-identity index's intermediate shape is skipped: the lookup
///   upgrade dropped and recreated it with the final column list, so only the
///   final shape is created here. The drop uses `IF EXISTS` (the original
///   relied on the intermediate index existing, which a squashed chain never
///   creates), making the version idempotent on re-apply.
///
/// Every statement here is `IF NOT EXISTS` / `IF EXISTS` guarded, so deleting
/// this version's `schema_migrations` row and re-running `migrate` is safe.
pub fn history_and_query_migration() -> Migration {
    Migration::new(
        4,
        "File history and event-log query indexes",
        vec![
            // Composite primary key (session_id, tool_call_id, path), with the
            // portable `file_key` column folded in.
            SchemaChange::Raw {
                sqlite: "CREATE TABLE IF NOT EXISTS session_file_history (session_id TEXT NOT NULL, tool_call_id TEXT NOT NULL, path TEXT NOT NULL, seq INTEGER NOT NULL, payload TEXT NOT NULL, file_key TEXT, PRIMARY KEY(session_id,tool_call_id,path));".into(),
                postgres: "CREATE TABLE IF NOT EXISTS session_file_history (session_id UUID NOT NULL, tool_call_id TEXT NOT NULL, path TEXT NOT NULL, seq BIGINT NOT NULL, payload JSONB NOT NULL, file_key TEXT, PRIMARY KEY(session_id,tool_call_id,path));".into(),
            },
            // Portable session-scoped file-identity roots. The identity
            // column and index above already include the portable upgrade's
            // effect, so only the roots table is taken from it verbatim.
            SchemaChange::Raw {
                sqlite: "CREATE TABLE IF NOT EXISTS session_file_roots (session_id TEXT NOT NULL, project_id TEXT NOT NULL, root_id TEXT NOT NULL, local_path TEXT NOT NULL, PRIMARY KEY(session_id,local_path));".into(),
                postgres: "CREATE TABLE IF NOT EXISTS session_file_roots (session_id UUID NOT NULL, project_id UUID NOT NULL, root_id TEXT NOT NULL, local_path TEXT NOT NULL, PRIMARY KEY(session_id,local_path));".into(),
            },
            SchemaChange::Raw {
                sqlite: "CREATE INDEX IF NOT EXISTS idx_session_file_history_path ON session_file_history(session_id,path,seq DESC);".into(),
                postgres: "CREATE INDEX IF NOT EXISTS idx_session_file_history_path ON session_file_history(session_id,path,seq DESC);".into(),
            },
            SchemaChange::Raw {
                sqlite: "DROP INDEX IF EXISTS idx_session_file_history_identity; CREATE INDEX IF NOT EXISTS idx_session_file_history_identity ON session_file_history(session_id,file_key,seq DESC,tool_call_id DESC); CREATE INDEX IF NOT EXISTS idx_session_file_history_legacy ON session_file_history(session_id,path,seq DESC,tool_call_id DESC) WHERE file_key IS NULL;".into(),
                postgres: "DROP INDEX IF EXISTS idx_session_file_history_identity; CREATE INDEX IF NOT EXISTS idx_session_file_history_identity ON session_file_history(session_id,file_key,seq DESC,tool_call_id DESC); CREATE INDEX IF NOT EXISTS idx_session_file_history_legacy ON session_file_history(session_id,path,seq DESC,tool_call_id DESC) WHERE file_key IS NULL;".into(),
            },
            // Sequence-cursor event reads without a sort pass.
            SchemaChange::create_index_if_not_exists(
                "idx_session_events_session_seq",
                "session_events",
                vec!["session_id", "seq"],
            ),
            // Cross-session checkpoint redemption aggregates.
            SchemaChange::create_partial_index_if_not_exists(
                "idx_session_events_checkpoint_redeemed",
                "session_events",
                vec!["session_id", "created_at"],
                "event_type = 'checkpoint_redeemed'",
            ),
            // Lifecycle outcomes without scanning call telemetry.
            SchemaChange::create_partial_index_if_not_exists(
                "idx_session_events_outcomes",
                "session_events",
                vec!["session_id", "seq"],
                "event_type IN ('session_completed', 'loop_intervention', 'checkpoint_redeemed')",
            ),
            // Event-log halves of the query-performance and continuation
            // indexes; the session-table halves live in version 2.
            SchemaChange::Raw {
                sqlite: r#"CREATE INDEX IF NOT EXISTS idx_diagnostic_event_page ON session_events(session_id, seq) WHERE event_type IN ('llm_call', 'auxiliary_call', 'session_completed', 'loop_intervention', 'checkpoint_redeemed'); CREATE INDEX IF NOT EXISTS idx_latest_file_path ON session_events(session_id, json_extract(payload, '$.file.path'), seq DESC) WHERE event_type = 'tool_result' AND json_extract(payload, '$.file.path') IS NOT NULL;"#.into(),
                postgres: r#"CREATE INDEX IF NOT EXISTS idx_diagnostic_event_page ON session_events(session_id, seq) WHERE event_type IN ('llm_call', 'auxiliary_call', 'session_completed', 'loop_intervention', 'checkpoint_redeemed'); CREATE INDEX IF NOT EXISTS idx_latest_file_path ON session_events(session_id, (payload->'file'->>'path'), seq DESC) WHERE event_type = 'tool_result' AND payload ? 'file';"#.into(),
            },
            SchemaChange::Raw {
                sqlite: "CREATE INDEX IF NOT EXISTS idx_continuation_task ON session_events(session_id, json_extract(payload, '$.task_id'), seq DESC) WHERE event_type='continuation';".into(),
                postgres: "CREATE INDEX IF NOT EXISTS idx_continuation_task ON session_events(session_id, (payload->>'task_id'), seq DESC) WHERE event_type='continuation';".into(),
            },
            SchemaChange::Raw {
                sqlite: "CREATE INDEX IF NOT EXISTS idx_continuation_latest ON session_events(session_id,seq DESC) WHERE event_type='continuation';".into(),
                postgres: "CREATE INDEX IF NOT EXISTS idx_continuation_latest ON session_events(session_id,seq DESC) WHERE event_type='continuation';".into(),
            },
        ],
    )
}

/// Return the full list of local migrations. Add new migrations here.
pub fn all_migrations() -> Vec<Migration> {
    vec![
        baseline_migration(),
        provider_accounting_migration(),
        session_lineage_migration(),
        conversation_memory_migration(),
        history_and_query_migration(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::migration::{Dialect, MigrationRunner};
    use rusqlite::Connection;

    #[test]
    fn local_schema_is_five_squashed_versions() {
        let migrations = all_migrations();
        assert_eq!(
            migrations.len(),
            5,
            "expected 5 squashed migrations, found {}",
            migrations.len()
        );
        let versions: Vec<i64> = migrations.iter().map(|m| m.version).collect();
        assert_eq!(versions, vec![0, 1, 2, 3, 4]);
        assert!(engine_core::migration::validate_migrations(&migrations).is_ok());
    }

    /// The squashed chain must stay a faithful fold of the engine-level
    /// migration helpers it inlines. Every `CREATE` they emit (except the
    /// superseded intermediate identity index, which the lookup upgrade
    /// dropped and recreated) must appear verbatim in the squashed SQL.
    #[test]
    fn squashed_versions_contain_engine_migration_statements() {
        // The squash adds `IF [NOT] EXISTS` guards the originals relied on
        // ordering for; normalize those away before comparing statements.
        let normalize = |sql: &str| sql.replace("IF NOT EXISTS ", "").replace("IF EXISTS ", "");
        let squashed: String = normalize(
            &MigrationRunner::new(Dialect::SQLite, &all_migrations())
                .pending_sql(-1)
                .iter()
                .flat_map(|(_, _, statements)| statements.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let engine_sql: String = [
            engine_core::file_history::migration(0),
            engine_core::file_history::portable_migration(0),
            engine_core::file_history::lookup_indexes_migration(0),
            engine_core::data_lifecycle::retention_indexes(0),
        ]
        .iter()
        .flat_map(|migration| {
            MigrationRunner::new(Dialect::SQLite, std::slice::from_ref(migration))
                .pending_sql(-1)
                .into_iter()
                .flat_map(|(_, _, statements)| statements)
        })
        .collect::<Vec<_>>()
        .join("\n");
        let engine_sql = normalize(&engine_sql);

        for fragment in engine_sql.split(';').map(str::trim).filter(|s| {
            s.starts_with("CREATE TABLE session_file_history")
                || s.starts_with("CREATE TABLE session_file_roots")
                || s.starts_with("CREATE INDEX idx_session_file_history_path")
                || s.starts_with("CREATE INDEX idx_session_file_history_identity ON")
                || s.starts_with("CREATE INDEX idx_session_file_history_legacy")
                || s.starts_with("CREATE INDEX \"idx_sessions_parent_session_id\"")
                || s.starts_with("CREATE INDEX \"idx_conversation_bindings_parent\"")
                || s.starts_with("ALTER TABLE session_tool_call_links")
        }) {
            // Folded statements need no verbatim match: identity columns and
            // the file key arrive via the table definitions (covered by the
            // schema-shape test, including the composite primary key), and
            // the intermediate identity index is superseded by the final
            // shape.
            if fragment.starts_with("ALTER TABLE session_tool_call_links")
                || fragment.starts_with("ALTER TABLE session_file_history")
                || fragment.starts_with("CREATE TABLE session_file_history")
                || fragment
                    == "CREATE INDEX idx_session_file_history_identity ON session_file_history(session_id,file_key,seq DESC)"
            {
                continue;
            }
            assert!(
                squashed.contains(fragment),
                "squashed chain lost engine statement: {fragment}"
            );
        }
        // The lookup upgrade's drop is `IF EXISTS` in the squash (the
        // intermediate index never exists there) while staying a plain drop
        // in the engine helper.
        assert!(squashed.contains("DROP INDEX idx_session_file_history_identity"));
        assert!(engine_sql.contains("DROP INDEX idx_session_file_history_identity;"));
    }

    /// Characterization guard for the migration squash: the full chain must
    /// converge on the same final schema shape no matter how the versions are
    /// grouped. Every entry here traces back to a later migration's effect on
    /// an earlier table (pricing columns, lineage columns, file-key column,
    /// canonical semantic cache, final lookup indexes).
    #[test]
    fn squashed_chain_preserves_final_schema_shape() {
        let path = std::env::temp_dir().join(format!(
            "reincore-migration-shape-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let db = crate::db::LocalDatabase::open(&path).unwrap();
        drop(db);
        let conn = Connection::open(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let tables: std::collections::HashSet<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual_table')")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(|name| name.unwrap())
            .collect();
        for expected in [
            "sessions",
            "session_events",
            "cost_estimates",
            "ledger_events",
            "tool_cache",
            "checkpoints",
            "budget_policies",
            "budget_history",
            "engine_settings",
            "provider_models",
            "local_projects",
            "local_project_directories",
            "session_conversation_bindings",
            "session_task_events",
            "session_tool_call_links",
            "semantic_response_cache",
            "generated_titles",
            "completion_accounting",
            "batch_jobs",
            "session_file_history",
            "session_file_roots",
        ] {
            assert!(tables.contains(expected), "missing table {expected}");
        }
        for gone in ["semantic_cache", "semantic_response_cache_v2"] {
            assert!(!tables.contains(gone), "legacy table {gone} must be gone");
        }

        let columns_of = |table: &str| -> std::collections::HashSet<String> {
            conn.prepare(&format!("PRAGMA table_info({table})"))
                .unwrap()
                .query_map([], |row| row.get::<_, String>(1))
                .unwrap()
                .map(|name| name.unwrap())
                .collect()
        };
        let sessions = columns_of("sessions");
        for expected in [
            "mode",
            "project_id",
            "project_name",
            "client_id",
            "external_session_id",
            "parent_session_id",
            "root_session_id",
            "rollover_index",
            "inherited_spend_cents",
            "context_tokens",
            "predicted_high_cents",
            "cost_accuracy_bps",
            "cost_accuracy_samples",
            "cache_hit_rate_bps",
            "cache_hit_rate_samples",
            "total_input_tokens",
            "total_output_tokens",
            "is_subagent",
            "subagent_parent_session_id",
            "subagent_spawn_tool_call_id",
            "subagent_spawn_request_group_id",
            "spend_nano_usd",
            "inherited_spend_nano_usd",
            "unknown_cost_calls",
        ] {
            assert!(sessions.contains(expected), "sessions missing {expected}");
        }
        let cost_estimates = columns_of("cost_estimates");
        assert!(cost_estimates.contains("request_group_id"));
        let ledger = columns_of("ledger_events");
        for expected in ["accounting_id", "amount_nano_usd", "cost"] {
            assert!(
                ledger.contains(expected),
                "ledger_events missing {expected}"
            );
        }
        let provider_models = columns_of("provider_models");
        for expected in ["pricing", "aliases", "auto_select", "notes"] {
            assert!(
                provider_models.contains(expected),
                "provider_models missing {expected}"
            );
        }
        let links = columns_of("session_tool_call_links");
        for expected in ["call_identity", "call_ambiguous"] {
            assert!(
                links.contains(expected),
                "session_tool_call_links missing {expected}"
            );
        }
        assert!(columns_of("session_file_history").contains("file_key"));
        let file_history_ddl: String = conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE name = 'session_file_history'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(
            file_history_ddl.contains("PRIMARY KEY(session_id,tool_call_id,path)"),
            "session_file_history lost its composite primary key: {file_history_ddl}"
        );

        let indexes: std::collections::HashSet<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type = 'index'")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(|name| name.unwrap())
            .collect();
        for expected in [
            "idx_session_events_session_seq",
            "idx_session_events_checkpoint_redeemed",
            "idx_session_events_outcomes",
            "idx_sessions_page",
            "idx_continuation_task",
            "idx_continuation_latest",
            "idx_session_file_history_identity",
            "idx_session_file_history_legacy",
            "idx_ledger_accounting_id",
            "idx_completion_accounting_session",
            "idx_provider_models_provider_model_id",
            "idx_cost_estimates_request_group_id",
        ] {
            assert!(indexes.contains(expected), "missing index {expected}");
        }
    }
}
