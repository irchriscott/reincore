// HTTP client for the in-process ReinCore daemon.
//
// The daemon is started by the Tauri Rust backend (app/src/lib.rs) and
// listens on 127.0.0.1:8989. In dev (Vite only, outside Tauri) the proxy in
// vite.config.ts forwards /v1, /v1beta and /health to that same port, so the
// frontend code is identical in both modes.

import axios from 'axios'
import type {
  CostOutcome,
  HealthStatus,
  SessionSummary,
  SessionDetail,
  SessionEvent,
  SessionTask,
  SessionTimelineSummary,
  SessionEventPage,
  ProjectSummary,
  LocalProject,
  ModelInfo,
  ProviderStatus,
  CredentialResponse,
  RoutingConfigResponse,
  RoutingPolicy,
  ProviderConfigResponse,
	CustomProviderModel,
	CustomProviderModelsResponse,
  CatalogModel,
  ProviderCatalogResponse,
  OllamaModel,
  OllamaModelsResponse,
  BudgetInfo,
  PauseResponse,
  ResumeResponse,
  CompactionSettings,
} from './types'

// The daemon is always on localhost:8989. We use a relative baseURL in dev
// so the Vite proxy kicks in; in the Tauri build (served from the webview)
// we fall back to an absolute URL.
const isDev = import.meta.env.DEV
export const DAEMON_BASE = isDev ? '' : 'http://127.0.0.1:8989'

export const api = axios.create({
  baseURL: DAEMON_BASE,
  headers: { 'Content-Type': 'application/json' },
  timeout: 15000,
})

export function extractErrorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const data = err.response?.data as { error?: string; message?: string } | undefined
    return data?.error || data?.message || err.message
  }
  if (err instanceof Error) return err.message
  return 'An unexpected error occurred'
}

// Health / daemon status

export async function getHealth(): Promise<HealthStatus> {
  const res = await api.get<HealthStatus>('/health')
  return res.data
}

// Sessions

export interface SessionListFilter {
  path?: string
  status?: string
  mode?: string
  project_id?: string
  /** ISO 8601 lower bound on updated_at (e.g. "2026-08-01T00:00:00Z"). */
  date_from?: string
  /** ISO 8601 upper bound on updated_at (e.g. "2026-08-31T23:59:59Z"). */
  date_to?: string
}

export async function listSessionsPage(filter?: SessionListFilter, before?: string, signal?: AbortSignal): Promise<{ sessions: SessionSummary[]; next_cursor?: string | null }> {
  const params: Record<string, string> = {}
  if (filter?.path) params.path = filter.path
  if (filter?.status) params.status = filter.status
  if (filter?.mode) params.mode = filter.mode
  if (filter?.project_id) params.project_id = filter.project_id
  if (filter?.date_from) params.date_from = filter.date_from
  if (filter?.date_to) params.date_to = filter.date_to
  params.limit = "100"
  if (before) params.before = before
  const res = await api.get<{ sessions: SessionSummary[]; next_cursor?: string | null }>('/v1/sessions', {
    params: Object.keys(params).length > 0 ? params : undefined,
    signal,
  })
  return { ...res.data, sessions: res.data.sessions ?? [] }
}

/** Explicit full enumeration for screens requiring all sessions. */
export async function listSessions(filter?: SessionListFilter): Promise<SessionSummary[]> {
  const sessions = new Map<string, SessionSummary>()
  const cursors = new Set<string>()
  let before: string | undefined
  do {
    const page = await listSessionsPage(filter, before)
    for (const session of page.sessions) sessions.set(session.session_id, session)
    before = page.next_cursor ?? undefined
    if (before && cursors.has(before)) throw new Error('Session pagination did not advance')
    if (before) cursors.add(before)
  } while (before)
  return [...sessions.values()]
}

export async function getSession(id: string): Promise<SessionDetail> {
  const res = await api.get<SessionDetail>(`/v1/sessions/${id}`)
  return res.data
}

export async function getSessionEvents(id: string): Promise<SessionEvent[]> {
  const events: SessionEvent[] = []
  let sinceSeq = 1
  while (true) {
    const page = await getSessionEventPage(id, {
      sinceSeq,
      untilSeq: Number.MAX_SAFE_INTEGER,
      limit: 500,
    })
    events.push(...page.events)
    if (page.next_since_seq === undefined) return events
    sinceSeq = page.next_since_seq
  }
}

export async function getSessionTasks(id: string): Promise<SessionTask[]> {
  const res = await api.get<SessionTask[]>(`/v1/sessions/${id}/tasks`)
  return Array.isArray(res.data) ? res.data : []
}

export async function listSubagentSessions(id: string): Promise<SessionSummary[]> {
  const res = await api.get<SessionSummary[]>(`/v1/sessions/${id}/subagents`)
  return Array.isArray(res.data) ? res.data : []
}

export async function pauseSession(id: string, reason?: string): Promise<PauseResponse> {
  const res = await api.post<PauseResponse>(`/v1/sessions/${id}/pause`, { reason })
  return res.data
}

export async function getSessionTimeline(id: string): Promise<SessionTimelineSummary> {
  const res = await api.get<SessionTimelineSummary>(`/v1/sessions/${id}/timeline`, { timeout: 120_000 })
  return res.data
}

export async function getSessionEventPage(
  id: string,
  options: { sinceSeq: number; untilSeq: number; limit?: number },
): Promise<SessionEventPage> {
  const res = await api.get<SessionEventPage>(`/v1/sessions/${id}/events/page`, {
    params: {
      since_seq: options.sinceSeq,
      until_seq: options.untilSeq,
      limit: options.limit ?? 100,
    },
    timeout: 60_000,
  })
  return res.data
}

export async function getSessionEvent(id: string, seq: number): Promise<SessionEvent> {
  const res = await api.get<SessionEvent>(`/v1/sessions/${id}/event/${seq}`, { timeout: 60_000 })
  return res.data
}

export async function getSessionDiagnostics(id: string): Promise<NonNullable<CostOutcome['diagnostics']>> {
  const res = await api.get<NonNullable<CostOutcome['diagnostics']>>(`/v1/sessions/${id}/diagnostics`)
  return res.data
}

export async function completeSession(id: string, reason?: string, outcome_report?: {
  source: 'validator' | 'user'; verdict: 'success' | 'failure' | 'unknown'; evidence: string
}): Promise<SessionDetail> {
  const res = await api.post<SessionDetail>(`/v1/sessions/${id}/complete`, { reason, outcome_report })
  return res.data
}

export async function renameSession(id: string, taskSummary: string): Promise<SessionDetail> {
  const res = await api.patch<SessionDetail>(`/v1/sessions/${id}`, { task_summary: taskSummary })
  return res.data
}

export async function resumeSession(token: string): Promise<ResumeResponse> {
  const res = await api.post<ResumeResponse>(`/v1/checkpoints/${encodeURIComponent(token)}/resume`)
  return res.data
}

// Projects

export async function listProjects(): Promise<ProjectSummary[]> {
  const res = await api.get<{ projects: ProjectSummary[] } | ProjectSummary[]>('/v1/projects')
  return Array.isArray(res.data) ? res.data : (res.data.projects ?? [])
}

// Local projects (desktop edition)

export async function listLocalProjects(): Promise<LocalProject[]> {
  const res = await api.get<{ projects: LocalProject[] }>('/v1/local-projects')
  return res.data.projects ?? []
}

export async function renameLocalProject(id: string, name: string): Promise<LocalProject> {
  const res = await api.patch<LocalProject>(`/v1/local-projects/${encodeURIComponent(id)}`, { name })
  return res.data
}

/** Attach another working directory to an existing local project.
 *  The daemon validates the directory exists and re-attributes any
 *  sessions recorded under that path. */
export async function assignLocalProjectDirectory(id: string, projectPath: string): Promise<LocalProject> {
  const res = await api.post<LocalProject>(
    `/v1/local-projects/${encodeURIComponent(id)}/directories`,
    { project_path: projectPath },
  )
  return res.data
}

// Models / Providers

export async function listModels(): Promise<ModelInfo[]> {
  const res = await api.get<{ models: ModelInfo[] }>('/v1/models')
  return res.data.models ?? []
}

// Providers (management)

// Fetch all supported providers with configured status, credentials source,
// model tier mappings, and routing policy from the daemon's /v1/providers.
export async function listProvidersStatus(): Promise<ProviderStatus[]> {
  const res = await api.get<{ providers: ProviderStatus[] }>('/v1/providers')
  return res.data.providers ?? []
}

// Store an API key credential for a provider.
export async function addProviderCredential(provider: string, apiKey: string): Promise<CredentialResponse> {
  const res = await api.post<CredentialResponse>(`/v1/providers/${encodeURIComponent(provider)}/credentials`, {
    api_key: apiKey,
  })
  return res.data
}

// Remove stored credentials for a provider.
export async function removeProviderCredential(provider: string): Promise<CredentialResponse> {
  const res = await api.delete<CredentialResponse>(`/v1/providers/${encodeURIComponent(provider)}/credentials`)
  return res.data
}

// Config (routing)

export async function getRoutingConfig(): Promise<RoutingPolicy> {
  const res = await api.get<RoutingConfigResponse>('/v1/config/routing')
  return res.data.routing
}

export async function updateRoutingConfig(fields: {
  mode?: string
  provider_strategy?: string
  minimum_quality?: string
  allow_upgrade?: string
  allow_degradation?: string
}): Promise<RoutingPolicy> {
  const res = await api.put<RoutingConfigResponse>('/v1/config/routing', fields)
  return res.data.routing
}

// Config (per-provider model tiers)

export async function getProviderConfig(provider: string): Promise<ProviderConfigResponse> {
  const res = await api.get<ProviderConfigResponse>(`/v1/config/providers/${encodeURIComponent(provider)}`)
  return res.data
}

export async function updateProviderConfig(provider: string, fields: {
  name?: string
  base_url?: string
  standard?: string
  lite?: string
  pro?: string
  ultra?: string
  routing_mode?: string
  routing_minimum_quality?: string
  routing_allow_upgrade?: string
  routing_allow_degradation?: string
}): Promise<ProviderConfigResponse> {
  const res = await api.put<ProviderConfigResponse>(`/v1/config/providers/${encodeURIComponent(provider)}`, fields)
  return res.data
}

// Provider model catalog

// Fetch the curated model catalog for a specific provider. For Copilot, pass
// the model family ('openai' or 'anthropic') instead of 'copilot'.
export async function getProviderCatalog(provider: string): Promise<CatalogModel[]> {
  const res = await api.get<ProviderCatalogResponse>(`/v1/providers/${encodeURIComponent(provider)}/models`)
  return res.data.models ?? []
}

// Fetch live models from a running Ollama server. The API key is optional
// (for Ollama Cloud) and is not persisted by this call.
export async function listOllamaModels(apiKey?: string, baseUrl?: string): Promise<OllamaModel[]> {
  const res = await api.post<OllamaModelsResponse>('/v1/providers/ollama/models', {
    api_key: apiKey ?? null,
    base_url: baseUrl ?? null,
  })
  return (res.data.models ?? []).toSorted((a, b) =>
    a.name.localeCompare(b.name, 'en', { sensitivity: 'base', numeric: true }) ||
    a.id.localeCompare(b.id, 'en'),
  )
}

// Probe a custom OpenAI-compatible provider's {base_url}/models endpoint.
// Empty results and request failures are handled by the form as manual entry.
export async function listCustomProviderModels(apiKey?: string, baseUrl?: string): Promise<CustomProviderModel[]> {
  const res = await api.post<CustomProviderModelsResponse>('/v1/providers/custom/models', {
    api_key: apiKey ?? null,
    base_url: baseUrl ?? null,
  })
  return res.data.models ?? []
}

// Budget

export async function getBudget(scope: string, key?: string): Promise<BudgetInfo> {
  const res = await api.get<BudgetInfo>('/v1/budget', { params: { scope, key } })
  return res.data
}

export async function setBudget(scope: string, limit: string, key?: string): Promise<BudgetInfo> {
  const res = await api.post<BudgetInfo>('/v1/budget', { scope, key, limit })
  return res.data
}

export async function getBudgetHistory(scope: string, key?: string): Promise<BudgetInfo> {
  const res = await api.get<BudgetInfo>('/v1/budget/history', { params: { scope, key } })
  return res.data
}

// Compaction settings (wire type keeps its compatibility name).

export async function getCompactionSettings(): Promise<CompactionSettings> {
  const res = await api.get<{ enabled: boolean; mode: string }>('/v1/config/compaction')
  return { enabled: res.data.enabled, mode: res.data.enabled ? res.data.mode : null }
}

export async function updateCompactionSettings(
  enabled: boolean,
  mode: string,
): Promise<CompactionSettings> {
  const res = await api.put<{ enabled: boolean; mode: string }>('/v1/config/compaction', {
    enabled,
    mode,
  })
  return { enabled: res.data.enabled, mode: res.data.enabled ? res.data.mode : null }
}
