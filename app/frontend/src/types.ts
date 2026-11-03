// Types for the daemon's HTTP API. These mirror the shapes returned
// by the engine's /v1 endpoints (see daemon/src/http/).

export interface HealthStatus {
  status: string
  edition?: string
  version?: string
  mode?: 'multi_provider' | 'single_provider'
  api_mode?: 'agentic' | 'chat'
  providers?: string[]
}

export interface SessionSummary {
  session_id: string
  task_summary: string
  status: string
  mode: string
  project_path?: string
  project_name?: string
  external_session_id?: string
  parent_session_id?: string
  root_session_id?: string
  rollover_index?: number
  context_tokens?: number
  total_input_tokens?: number
  total_output_tokens?: number
  spend_cents?: number
  cost_outcome?: CostOutcome
  checkpoint_token?: string
  is_subagent?: boolean
  subagent_parent_session_id?: string
  subagent_spawn_tool_call_id?: string
  subagent_spawn_request_group_id?: string
  created_at?: string
  updated_at?: string
}

export type SuccessStatus = 'unknown' | 'validated_success' | 'user_reported_success' | 'recovered_success' | 'reported_failure'

export interface InterventionMetrics {
  detections: number
  observations: number
  warnings: number
  repair_requests: number
  blocked_actions: number
  pauses: number
  false_alarms: number
  repeated_questions: number
  successful_recoveries: number
  task_completed: boolean
  added_latency_ms: number
  added_input_tokens: number
  added_output_tokens: number
  cost_per_completed_task_nano_usd?: number | null
}

export interface CostOutcome {
  success_status?: SuccessStatus
  completed_after_intervention?: boolean
  outcome_report?: { source: 'validator' | 'user'; verdict: 'success' | 'failure' | 'unknown'; evidence: string } | null
  completions_per_estimated_dollar?: number | null
  diagnostics?: {
    prediction: {
      sample_count: number
      interval_coverage_bps: number | null
      mean_absolute_midpoint_error_nano_usd: number | null
      mean_actual_to_predicted_high_ratio_bps: number | null
    }
    cache: { token_weighted_utilization_bps: number | null; sample_count: number }
    /** ReinCore's own caches. Absent on diagnostics snapshots stored before
     *  these counters existed, so every read must tolerate undefined. */
    reincore_cache?: {
      llm_call_count: number
      response_cache_hit_count: number
      response_hit_rate_bps: number | null
      tool_result_count: number
      tool_cache_hit_count: number
      tool_hit_rate_bps: number | null
      hit_similarity_median_bps: number | null
      near_miss_count: number
      near_miss_median_bps: number | null
      near_miss_p90_bps: number | null
      would_reuse_at_9500_bps: number
      would_reuse_at_9000_bps: number
      would_reuse_at_8500_bps: number
      tool_schema_changes: number
      tool_schema_observations: number
    }
  }
  completed: boolean
  clean_completion: boolean
  loop_intervention_count: number
  intervention_metrics?: InterventionMetrics
  cost_cents: number
  completion_quality_bps?: number
  /** Deprecated: no quality score is inferred from completion. */
  cost_efficiency_score?: number
}

export interface RequestGroupOutcome extends CostOutcome {
  request_group_id: string
  task_id: string
  first_event_seq: number
  last_event_seq: number
  llm_call_count: number
}

export interface RequestGroupTimelineSummary {
  request_group_id: string
  task_id: string
  prompt?: string
  first_event_seq: number
  last_event_seq: number
  event_count: number
  llm_call_count: number
  tool_call_count: number
  tool_error_count: number
  pii_redaction_count?: number
  input_tokens: number
  output_tokens: number
  cost_cents: number
  models: string[]
  started_at?: string
  ended_at?: string
  completed: boolean
  clean_completion: boolean
  success_status?: SuccessStatus
  completed_after_intervention?: boolean
  intervention_metrics?: InterventionMetrics
}

export interface SessionTimelineSummary {
  tasks: SessionTask[]
  request_groups: RequestGroupTimelineSummary[]
  events: SessionEvent[]
  total_event_count: number
  latest_seq?: number
  pii_redaction_count?: number
}

export interface SessionEventPage {
  events: SessionEvent[]
  next_since_seq?: number
}

export interface SessionEvent {
  session_id: string
  seq: number
  event_type: string
  payload: Record<string, unknown>
  cost_cents?: number
  created_at: string
}

export type TaskType =
  | 'feature'
  | 'bug_fix'
  | 'research'
  | 'analysis'
  | 'planning'
  | 'content_creation'
  | 'communication'
  | 'operations'
  | 'review'
  | 'learning'
  | 'support'
  | 'general'

export interface SessionTask {
  task_id: string
  name: string
  task_type: TaskType
  first_prompt: string
  request_group_ids: string[]
  first_event_seq: number
  last_event_seq: number
  request_count: number
  input_tokens: number
  output_tokens: number
  cost_cents: number
  completed: boolean
  clean_completion: boolean
  success_status?: SuccessStatus
  completed_after_intervention?: boolean
  loop_intervention_count: number
  intervention_metrics?: InterventionMetrics
  completion_quality_bps?: number
  cost_efficiency_score?: number
  request_groups: RequestGroupOutcome[]
}

// Detail, rename, and completion endpoints serialize Session with `id`.
// Only list endpoints use the SessionSummary `session_id` field.
export interface SessionDetail extends Omit<SessionSummary, 'session_id'> {
  id: string
  events?: SessionEvent[]
}

export interface ProjectSummary {
  project_name: string
  project_path: string | null
  project_id?: string
  session_count: number
  active_session_count: number
  total_spend_cents: number
  active_context_tokens: number
  recorded_context_tokens: number
  total_input_tokens: number
  total_output_tokens: number
}

export interface LocalProject {
  id: string
  name: string
  directories: string[]
  created_at: string
  updated_at: string
}

export interface ModelInfo {
  id: string
  label?: string
  provider: string
  tier?: string
}

export interface RoutingPolicy {
  mode: string
  provider_strategy: string
  minimum_quality: string
  allow_upgrade: boolean
  allow_degradation: boolean
}

export interface ModelTierInfo {
  tier: string
  model: string | null
}

export interface ProviderStatus {
  id: string
  label: string
  name: string | null
  base_url: string | null
  configured: boolean
  source: string
  env_var: string
  models: ModelTierInfo[]
  routing: RoutingPolicy
}

// Legacy type kept for backward compatibility.
export interface ProviderInfo {
  id: string
  label: string
  type: string
  configured: boolean
  models?: ModelInfo[]
}

export interface CredentialResponse {
  provider: string
  success: boolean
  message: string
}

export interface RoutingConfigResponse {
  routing: RoutingPolicy
}

export interface ProviderConfigResponse {
  provider: string
  name: string | null
  base_url: string | null
  standard: string | null
  lite: string | null
  pro: string | null
  ultra: string | null
  routing: RoutingPolicy
}

// Model catalog entry from GET /v1/providers/:provider/models
export interface CatalogModel {
  provider: string
  model_name: string
  model_id: string
  input_price: number
  output_price: number
  reasoning: string
  speed: string
}

export interface ProviderCatalogResponse {
  provider: string
  models: CatalogModel[]
}

// Ollama live model list from POST /v1/providers/ollama/models
export interface OllamaModel {
  id: string
  name: string
  parameter_size: string | null
}

export interface CustomProviderModel {
  id: string
  name: string
}

export interface CustomProviderModelsResponse {
  models: CustomProviderModel[]
}

export interface OllamaModelsResponse {
  models: OllamaModel[]
}

export interface BudgetInfo {
  scope: string
  key?: string
  limit_cents: number
  spent_cents: number
  remaining_cents: number
  tier: number
  history?: BudgetHistoryEntry[]
}

export interface BudgetHistoryEntry {
  timestamp: string
  event: string
  scope: string
  key?: string
  limit_cents?: number
  spent_cents?: number
}

export interface PauseResponse {
  checkpoint_token: string
  expires_at: string
}

export interface ResumeResponse {
  session_id: string
  resumed_at_seq: number
}

export interface CompactionSettings {
  enabled: boolean
  mode: string | null
}
