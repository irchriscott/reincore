//! # engine-core
//!
//! Local core engine for ReinCore. Implements the product intelligence:
//!
//! - Cache-prefix normalization
//! - Tool-call / MCP result memoization
//! - Session event log / memory manager
//! - Cascading model routing / escalation classifier
//! - Pre-flight cost prediction
//! - Progressive budget degradation / tier state machine
//! - Stateful checkpointing & resumable tokens
//! - Progress-aware circuit breaker
//! - Semantic / response caching
//! - Batch-eligible routing
//!
//! Storage is abstracted behind narrow async traits; the local engine runs on
//! SQLite (the migration DSL still renders Postgres for tests).

// Feature modules. Each directory groups one engine concern; the aliases
// below preserve every pre-existing `engine_core::<module>` path, so moving a
// module into a group never changes how callers reach it.
pub mod detection;
pub mod execution;
pub mod mcp;
pub mod memory;
pub mod sessions;
pub mod storage;
pub mod trust;

// Module aliases: `engine_core::<module>` keeps resolving exactly as before,
// no matter how deep the module lives inside its feature group.
use execution::pipeline::{lexical, tool_names};
pub use execution::{
    dispatch::{cost, cost_outcome, routing},
    pipeline::{external_tools, framework, user_content, workflow},
    providers::{model_catalog, model_library, models, pricing, provider},
};
pub use memory::{
    compression, context, file_history, summary, task, task_decisions, task_type, title_generation,
};
use sessions::conversation;
pub use sessions::{continuation, event, grouping, handoff, live_events, session};
pub use storage::{
    caching::{cache, embedding, normalizer, prewarm, tool_cache},
    error,
    observability::{log, operational, otel_health, telemetry},
    persistence::{data_lifecycle, in_memory_stores, migration, store},
    resilience,
    settlement::{accounting, batch_jobs},
};
use trust::privacy::privacy_fixture_recovery;
pub use trust::{
    budget, checkpoint,
    enforcement::{action_gate, action_receipts, action_recovery, intervention},
    oversight::{agent_health, circuit, claim_grounding, diagnostics},
    privacy::{policy as privacy, privacy_source},
};

pub use agent_health::{
    analyze_provider_response, analyze_response_citations, analyze_response_citations_with_request,
    detect_tool_loop, log_agent_health, message_content_sources, recent_health_findings,
    request_content_provenance, scan_prompt_injection, AgentHealthFinding, AgentHealthKind,
    AgentHealthReport, AgentHealthSeverity, ContentProvenance, ContentSource, ContentTrust,
};
pub use budget::SettingsStore;
pub use budget::{
    BudgetHistoryEntry, BudgetPolicy, BudgetResolver, BudgetScope, BudgetStore, BudgetTier,
    LedgerEvent, LedgerReason, LedgerStore, ProgressiveBudget,
};
pub use cache::{
    best_candidate_entry_id, best_candidate_similarity_bps, semantic_provider_config_version,
    similarity_to_bps, tool_schema_hash, CacheKey, MemoizationCache, MemoizationPolicy,
    SemanticCache, SemanticCacheCandidate, SemanticCacheEntry, SemanticCacheFingerprint,
    SemanticCacheHit, SemanticCacheLookup, SemanticCachePlan, SemanticCachePolicy,
    SemanticCacheProvenance, SemanticCacheQuery, SemanticCacheScope,
    DEFAULT_SEMANTIC_CACHE_TTL_SECONDS, DEFAULT_SEMANTIC_RETRIEVAL_SIMILARITY,
    MAX_SEMANTIC_CACHE_TTL_SECONDS,
};
pub use checkpoint::{Checkpoint, CheckpointManager, ResumableToken};
pub use circuit::{
    dedupe_repeated_tool_results, is_exploration_tool, redundant_lookup_finding, CircuitBreaker,
    CircuitBreakerConfig, LoopObservation, LoopSettings, PendingIntervention, StepHash,
    ToolResultDedup,
};
pub use compression::{
    CompressionMode, CompressionResult, CompressionSettings, ContextCompressor,
    ExtractiveSummarizer, Summarizer, TemplateSummarizer,
};
pub use context::{AssembledContext, MemoryManager, MemoryTier, ToolStateRetrievalConfig};
pub use cost::{
    actual_to_predicted_high_ratio_bps, cache_hit_rate_bps, cost_accuracy_bps, count_text_tokens,
    CostPrediction, CostPredictor, ModelPricing, TaskCategory, CACHE_HIT_RATE_BPS_SCALE,
    COST_ACCURACY_BPS_SCALE,
};
pub use cost_outcome::{is_effective_loop_intervention, session_cost_outcome, CostOutcome};
pub use embedding::{
    auto_embedding_for_request, embed_text, embed_tool_request, semantic_cache_plan,
    semantic_reuse_decision, SemanticReuseDecision, AUTO_EMBEDDING_SIMILARITY,
    CLIENT_EMBEDDING_SIMILARITY, EMBEDDING_DIMENSIONS, RESPONSE_CACHE_SIMILARITY,
    TEAM_TOOL_SEMANTIC_SIMILARITY,
};
pub use error::{describe_request_error, EngineError, EngineResult};
pub use event::{
    compact_audit_request_messages, compact_session_event, tool_call_links_from_event, EventStore,
    SessionEvent, SessionEventType, TaskEventProjection, ToolCallLink,
};
pub use external_tools::{
    completed_external_subagent_executions, external_subagent_session,
    external_subagent_session_id, external_subagent_task_summary, is_run_subagent_tool,
    is_subagent_tool_for_profile, record_external_tool_executions,
    record_external_tool_executions_detailed_for_profile,
    record_external_tool_executions_for_profile, request_has_external_tool_result,
    ExternalSubagentExecution, ExternalToolRecordOutcome, DELEGATION_PROFILE_HEADER,
};
pub use framework::{
    FrameworkAdapter, FrameworkAdapterConfig, FrameworkCheckpoint, FrameworkMemoryEntry,
    FrameworkMemorySnapshot,
};
pub use grouping::{
    auto_request_group_id, bind_conversation_group, client_conversation_id,
    conversation_binding_key, conversation_fingerprint, conversation_head_fingerprint,
    conversation_id_hash, conversation_key, group_id_for_conversation,
    group_id_for_conversation_continuation, mark_group_continues, request_continues_conversation,
    request_group_from_tool_lineage, request_group_key, resolve_request_group_id,
    resume_conversation_group, CONVERSATION_GROUP_TTL, CONVERSATION_ID_EXTRA_KEY,
    PARENT_CONVERSATION_ID_EXTRA_KEY, REQUEST_GROUP_EXTRA_KEY, REQUEST_GROUP_HEADER,
    SPAN_ID_EXTRA_KEY, TRACE_ID_EXTRA_KEY,
};
pub use handoff::{
    compact_for_new_session, is_self_contained, HandoffPolicy, SessionHandoff, HANDOFF_MESSAGE_NAME,
};
pub use in_memory_stores::{
    InMemoryBudgetStore, InMemoryCacheStore, InMemoryEventStore, InMemoryLedgerStore,
    InMemorySettingsStore, InMemoryTeamScopedCacheStore, InMemoryToolStateStore,
};
pub use intervention::{
    apply_policy as apply_intervention_policy, apply_policy_with as apply_intervention_policy_with,
    current_policy as current_intervention_policy, mandatory_boundary_action,
    record_intervention_outcome,
    set_process_default_policy as set_process_default_intervention_policy, HeuristicMode,
    InterventionAction, InterventionEvidence, InterventionMetrics, InterventionOutcome,
    InterventionPolicy, InterventionRecord, MandatoryBoundary,
};
pub use live_events::{live_session_events, SessionEventHub};
pub use migration::{
    validate_migrations, ColumnDef, DataType, Dialect, Migration, MigrationRunner, SchemaChange,
};
pub use model_library::{
    default_model_for_provider_and_tier, derive_model_from_standard, find_model, model_catalog,
    Modality, ProviderModelRecord,
};
pub use models::{
    effective_finish_reason, finish_reason_for, finish_reason_for_json, ChatCompletionRequest,
    ChatMessage, ContentPart, FileContextAttachment, FileUrl, FunctionCall, ImageUrl,
    MessageContent, ModelTier, ToolCall, ToolDefinition, ToolResult,
};
pub use normalizer::{is_dynamic_system_message, CachePrefixNormalizer, NormalizedPrompt};
pub use prewarm::{PreWarmConfig, PreWarmManager, PreWarmOutcome, SessionShape};
pub use privacy::{
    PrivacyBoundary, PrivacyPolicy, PrivacyRedactionDisposition, RedactionResult, SecretAction,
};
pub use provider::{
    chat_message_has_visible_output, extract_provider_reasoning, is_rate_limit_response_header,
    provider_response_has_visible_output, retry_empty_provider_response, BatchSubmission,
    ChatCompletionResponse, ChatCompletionStreamChunk, Provider, ProviderConfig, ProviderModel,
    ProviderModelMapping, ProviderRegistry, ProviderResponseAttempts, ProviderResponseHeaders,
    ProviderStream, RateLimitRemaining, TokenUsage, ToolCallDelta, ToolExecutor,
    ToolExecutorRegistry, EMPTY_PROVIDER_RESPONSE_RETRY_LIMIT,
};
pub use routing::{
    resolve_routing_tier, ClassificationFeatures, ClassificationTelemetry, EscalationClassifier,
    LogisticRoutingModel, MinimumQuality, ProviderStrategy, RequestRoutingOverrides, ResolvedTier,
    RoutingAdjustment, RoutingDecision, RoutingMode, RoutingPolicy,
};
pub use session::{
    CacheHitRateBreakdown, CheckpointRedemptionBreakdown, CostAccuracyBreakdown, ProjectUsage,
    Session, SessionMode, SessionStatus,
};
pub use store::{
    CacheStore, IndexedToolState, TeamScopedCacheStore, TeamSemanticToolCacheEntry,
    TeamSemanticToolCacheHit, ToolStateStore,
};
pub use summary::{
    derive_task_summary, improved_task_summary, FALLBACK_TASK_SUMMARY, MAX_TASK_SUMMARY_CHARS,
};
pub use task::{
    resolve_task, resolve_task_auto, scope_request_to_task, session_tasks, session_timeline,
    RequestGroupOutcome, RequestGroupTimelineSummary, SessionTask, SessionTimeline, TaskAssignment,
    TASK_ID_EXTRA_KEY, TASK_NAME_EXTRA_KEY, TASK_TYPE_EXTRA_KEY,
};
pub use task_type::{classify_task_type, TaskType};
pub use user_content::{
    effective_user_request, is_external_summary_message, is_terminal_notification_tool,
    is_user_scaffolding_only, normalize_correlated_terminal_notifications,
    normalize_previous_user_message_slice, normalize_previous_user_messages,
    normalize_user_message, session_user_request, strip_user_scaffolding, wrapped_user_request,
    TERMINAL_NOTIFICATION_TOOL,
};
pub use workflow::{
    compression_turn_is_new, enforce_context_window, inject_system_nudge, parse_embedding,
    prepare_context, prepare_context_continuation, resolve_budget_routing, BudgetRoutingDecision,
    ContextAdjustment, ContextWindowReset,
};
