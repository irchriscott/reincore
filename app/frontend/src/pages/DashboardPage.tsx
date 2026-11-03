// Dashboard: overview of the engine - daemon health, session/project/
// provider counts, and a quick "what is ReinCore" blurb.

import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Activity, FolderKanban, Pause, CheckCircle2, Circle, Cpu, DollarSign, Clock } from 'lucide-react'
import { api, extractErrorMessage, getHealth, listProjects, listProvidersStatus } from '../api'
import { Card, Badge, Skeleton, StatusDot } from '../components/ui'
import { ProviderIcon } from '../components/ProviderIcon'
import { sessionOriginState } from '../navigation/sessionOrigin'
import type { SessionSummary, ProjectSummary, ModelInfo, HealthStatus, ProviderStatus } from '../types'
import { capitalizeStatus } from '../utils/status'

/** Format a timestamp as a relative "time ago" string. */
function timeAgo(iso?: string): string | null {
  if (!iso) return null
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return null
  const diff = Date.now() - then
  if (diff < 0) return null
  const sec = Math.floor(diff / 1000)
  if (sec < 60) return 'just now'
  const min = Math.floor(sec / 60)
  if (min < 60) return `${min}m ago`
  const hr = Math.floor(min / 60)
  if (hr < 24) return `${hr}h ago`
  const day = Math.floor(hr / 24)
  if (day < 7) return `${day}d ago`
  const wk = Math.floor(day / 7)
  if (wk < 4) return `${wk}w ago`
  const mo = Math.floor(day / 30)
  return `${mo}mo ago`
}

/** Map a session status to a color variable and icon. */
function statusVisual(status: string): { color: string; bg: string; Icon: typeof Activity } {
  switch (status) {
    case 'active':
      return { color: 'var(--color-success)', bg: 'rgba(16,185,129,0.10)', Icon: Activity }
    case 'paused':
      return { color: 'var(--color-warning)', bg: 'rgba(245,158,11,0.10)', Icon: Pause }
    case 'completed':
      return { color: 'var(--color-info)', bg: 'rgba(56,189,248,0.10)', Icon: CheckCircle2 }
    case 'checkpointed':
      return { color: 'var(--color-accent)', bg: 'var(--color-accent-soft)', Icon: CheckCircle2 }
    default:
      return { color: 'var(--color-text-muted)', bg: 'var(--color-surface-highlight)', Icon: Circle }
  }
}

/** Format token count compactly (e.g. 12.4k, 1.2M). */
function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}

export function DashboardPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [, setModels] = useState<ModelInfo[]>([])
  const [providers, setProviders] = useState<ProviderStatus[]>([])
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.allSettled([
      getHealth(),
      api.get<{ sessions: SessionSummary[] }>('/v1/sessions'),
      listProjects(),
      api.get<{ models: ModelInfo[] }>('/v1/models'),
      listProvidersStatus(),
    ]).then(([healthRes, sessionsRes, projectsRes, modelsRes, providersRes]) => {
      if (healthRes.status === 'fulfilled') setHealth(healthRes.value)
      if (sessionsRes.status === 'fulfilled') setSessions(sessionsRes.value.data.sessions ?? [])
      if (projectsRes.status === 'fulfilled') setProjects(projectsRes.value)
      if (modelsRes.status === 'fulfilled') setModels(modelsRes.value.data.models ?? [])
      if (providersRes.status === 'fulfilled') setProviders(providersRes.value)
      const failed = [sessionsRes, projectsRes, modelsRes].find((r) => r.status === 'rejected') as
        | PromiseRejectedResult
        | undefined
      if (failed && healthRes.status !== 'fulfilled') {
        setError(extractErrorMessage(failed.reason))
      }
      setLoading(false)
    })
  }, [])

  const totalCostCents = sessions.reduce((sum, s) => sum + (s.spend_cents ?? 0), 0)

  return (
    <div className="mx-auto w-full max-w-[1000px]">
      <h1 className="heading-display mb-2 text-3xl font-semibold text-[var(--color-text-heading)]">
        ReinCore
      </h1>
      <p className="mb-8 text-[var(--color-text-muted)]">
        A session-aware gateway for your AI agents. This is your ReinCore dashboard.
      </p>

      {error && (
        <div className="mb-6 rounded-lg border border-[var(--color-danger)] bg-[rgba(244,63,94,0.08)] px-4 py-3 text-sm text-[var(--color-danger)]">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* Daemon status */}
        <Card className="p-6">
          <div className="mb-3 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
              <Activity className="h-5 w-5 text-[var(--color-accent)]" />
            </div>
            <span className="text-sm font-medium text-[var(--color-text)]">Daemon</span>
          </div>
          {loading ? (
            <Skeleton className="h-9 w-20" />
          ) : (
            <div className="flex flex-col gap-1">
              <StatusDot status={health?.status === 'ok' ? 'running' : 'stopped'} />
              <span className="text-xs text-[var(--color-text-muted)]">
                {health?.mode === 'single_provider' && health.providers?.[0]
                  ? `Provider: ${health.providers[0]}`
                  : 'Multi-provider'}
              </span>
            </div>
          )}
        </Card>

        {/* Sessions stat */}
        <Card className="p-6">
          <div className="mb-3 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
              <Activity className="h-5 w-5 text-[var(--color-accent)]" />
            </div>
            <span className="text-sm font-medium text-[var(--color-text)]">Sessions</span>
          </div>
          {loading ? (
            <Skeleton className="h-9 w-12" />
          ) : (
            <p className="font-mono text-3xl font-bold text-[var(--color-text-heading)]">{sessions.length}</p>
          )}
        </Card>

        {/* Projects stat */}
        <Card className="p-6">
          <div className="mb-3 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
              <FolderKanban className="h-5 w-5 text-[var(--color-accent)]" />
            </div>
            <span className="text-sm font-medium text-[var(--color-text)]">Projects</span>
          </div>
          {loading ? (
            <Skeleton className="h-9 w-12" />
          ) : (
            <p className="font-mono text-3xl font-bold text-[var(--color-text-heading)]">{projects.length}</p>
          )}
        </Card>

        {/* Cost stat */}
        <Card className="p-6">
          <div className="mb-3 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
              <DollarSign className="h-5 w-5 text-[var(--color-accent)]" />
            </div>
            <span className="text-sm font-medium text-[var(--color-text)]">Total Spend</span>
          </div>
          {loading ? (
            <Skeleton className="h-9 w-24" />
          ) : (
            <p className="font-mono text-3xl font-bold text-[var(--color-text-heading)]">
              ${(totalCostCents / 100).toFixed(2)}
            </p>
          )}
        </Card>
      </div>

      {/* Recent sessions */}
      <div className="mt-8">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-[var(--color-text-heading)]">Recent sessions</h2>
          <Link to="/sessions" className="text-sm text-[var(--color-primary-light)] hover:underline">
            View all →
          </Link>
        </div>
        {loading ? (
          <div className="flex flex-col gap-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-[72px] w-full rounded-xl" />
            ))}
          </div>
        ) : sessions.length === 0 ? (
          <Card className="p-8 text-center text-sm text-[var(--color-text-muted)]">
            No sessions yet. Start an agent through ReinCore and it will appear here.
          </Card>
        ) : (
          <div className="flex flex-col gap-1.5">
            {sessions
              .filter((s): s is SessionSummary & { session_id: string } => Boolean(s?.session_id))
              .slice(0, 6)
              .map((s) => {
                const { color: statusColor, bg: statusBg, Icon: StatusIcon } = statusVisual(s.status)
                const rel = timeAgo(s.updated_at ?? s.created_at)
                return (
                  <Link
                    key={s.session_id}
                    to={`/sessions/${s.session_id}`}
                    state={sessionOriginState({ kind: 'dashboard' })}
                    className="block"
                  >
                    <Card className="group flex items-center gap-3 p-3.5 transition-all hover:border-[var(--color-border-strong)] hover:bg-[var(--color-surface-highlight)]">
                      {/* Status-colored left accent bar */}
                      <div className="h-10 w-1 shrink-0 rounded-full" style={{ backgroundColor: statusColor }} />
                      {/* Status icon */}
                      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg" style={{ backgroundColor: statusBg }}>
                        <StatusIcon className="h-4.5 w-4.5" style={{ color: statusColor }} />
                      </div>
                      {/* Main content */}
                      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                        <span className="truncate text-sm font-medium text-[var(--color-text-heading)]">
                          {s.task_summary || 'Untitled session'}
                        </span>
                        <div className="flex items-center gap-2 text-xs text-[var(--color-text-muted)]">
                          <span className="font-mono">{s.session_id.slice(0, 8)}</span>
                          <span className="text-[var(--color-text-muted)]/40">·</span>
                          <span className="truncate">{s.project_name || s.project_path || '-'}</span>
                          {rel && (
                            <>
                              <span className="text-[var(--color-text-muted)]/40">·</span>
                              <span className="flex items-center gap-1 whitespace-nowrap">
                                <Clock className="h-3 w-3" />
                                {rel}
                              </span>
                            </>
                          )}
                        </div>
                      </div>
                      {/* Right-side metrics */}
                      <div className="flex shrink-0 items-center gap-3">
                        <div className="hidden items-center gap-3 text-xs text-[var(--color-text-muted)] sm:flex">
                          <span className="flex items-center gap-1">
                            <DollarSign className="h-3 w-3" />
                            {((s.spend_cents ?? 0) / 100).toFixed(2)}
                          </span>
                          <span className="flex items-center gap-1">
                            <Cpu className="h-3 w-3" />
                            {fmtTokens(s.context_tokens ?? 0)}
                          </span>
                        </div>
                        <Badge variant={s.status === 'active' ? 'success' : s.status === 'paused' ? 'warning' : 'default'}>
                          {capitalizeStatus(s.status)}
                        </Badge>
                      </div>
                    </Card>
                  </Link>
                )
              })}
          </div>
        )}
      </div>

      <div className="mt-8">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-[var(--color-text-heading)]">Providers</h2>
          <Link to="/providers" className="text-sm text-[var(--color-primary-light)] hover:underline">
            Manage →
          </Link>
        </div>
        {loading ? (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-14 rounded-xl" />
            ))}
          </div>
        ) : providers.length === 0 ? (
          <span className="text-sm text-[var(--color-text-muted)]">No providers configured.</span>
        ) : (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
            {providers.map((p) => (
              <div
                key={p.id}
                className={`flex items-center gap-2.5 rounded-xl border px-3.5 py-2.5 ${
                  p.configured
                    ? 'border-[var(--color-border-strong)] bg-[var(--color-surface)]'
                    : 'border-[var(--color-border)] bg-[var(--color-surface-highlight)] opacity-60'
                }`}
              >
                <ProviderIcon id={p.id} className="h-5 w-5 shrink-0" />
                <div className="flex min-w-0 flex-col">
                  <span className="truncate text-sm font-medium text-[var(--color-text-heading)]">{p.label}</span>
                  <span className="text-xs text-[var(--color-text-muted)]">
                    {p.configured ? 'Configured' : 'Not configured'}
                  </span>
                </div>
                {p.configured && (
                  <div className="ml-auto h-2 w-2 shrink-0 rounded-full bg-[var(--color-success)]" title="Configured" />
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
