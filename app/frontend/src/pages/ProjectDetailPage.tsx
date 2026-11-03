// import { OutcomeLabel } from '../components/OutcomeLabel'
// Project detail page: the workspace-centric view of a single local project.
// Shows the project's aggregate usage (sessions, spend, tokens), its budget
// policy, the working directories assigned to it, and the sessions recorded
// under those directories. Mirrors the card/row language used by the
// Dashboard, Sessions, and Budget pages.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { invoke } from '@tauri-apps/api/core'
import {
  Activity,
  Check,
  Clock,
  DollarSign,
  Download,
  FolderKanban,
  FolderOpen,
  FolderPlus,
  Pencil,
  Plus,
  RefreshCw,
  Upload,
  Wallet,
  X,
} from 'lucide-react'
import {
  assignLocalProjectDirectory,
  extractErrorMessage,
  getBudget,
  listLocalProjects,
  listSessions,
  renameLocalProject,
  setBudget,
} from '../api'
import { Badge, Button, Card, EmptyState, Input, ListSelect, Skeleton } from '../components/ui'
import { sessionOriginState } from '../navigation/sessionOrigin'
import type { BudgetInfo, LocalProject, SessionSummary } from '../types'
import { capitalizeStatus } from '../utils/status'

/** Format a timestamp as a relative "time ago" string. */
function timeAgo(iso?: string | null): string | null {
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
      return { color: 'var(--color-warning)', bg: 'rgba(245,158,11,0.10)', Icon: Clock }
    case 'completed':
      return { color: 'var(--color-info)', bg: 'rgba(56,189,248,0.10)', Icon: Check }
    default:
      return { color: 'var(--color-text-muted)', bg: 'var(--color-surface-highlight)', Icon: Activity }
  }
}

/** Format token counts compactly (e.g. 12.4k, 1.2M). */
function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}



/** Budget tiers with thresholds used by the engine (1-4). */
const BUDGET_TIERS = [
  { tier: 1, label: 'Normal', threshold: 0 },
  { tier: 2, label: 'Warning', threshold: 0.7 },
  { tier: 3, label: 'Critical', threshold: 0.9 },
  { tier: 4, label: 'Exceeded', threshold: 1.0 },
]

export function ProjectDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  // Project record + aggregate usage
  const [project, setProject] = useState<LocalProject | null>(null)
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Inline rename
  const [editingName, setEditingName] = useState(false)
  const [nameDraft, setNameDraft] = useState('')
  const [savingName, setSavingName] = useState(false)

  // Directory assignment
  const [assigning, setAssigning] = useState(false)

  // Budget
  const [budget, setBudgetState] = useState<BudgetInfo | null>(null)
  const [budgetInput, setBudgetInput] = useState('')
  const [savingBudget, setSavingBudget] = useState(false)

  // Session status filter
  const [statusFilter, setStatusFilter] = useState('')
  const [search, setSearch] = useState('')
  const searchDebounce = useRef<number | null>(null)
  const [debouncedSearch, setDebouncedSearch] = useState('')

  useEffect(() => {
    if (searchDebounce.current) window.clearTimeout(searchDebounce.current)
    searchDebounce.current = window.setTimeout(() => setDebouncedSearch(search), 250)
    return () => {
      if (searchDebounce.current) window.clearTimeout(searchDebounce.current)
    }
  }, [search])

  const load = useCallback(async () => {
    if (!id) return
    setRefreshing(true)
    try {
      let failed: string | null = null
      // The project record must load first: the budget scope for local
      // projects is keyed by project *name* (see BudgetPage), and the
      // sessions filter is keyed by project id.
      const projectRes = await listLocalProjects()
      const found = projectRes.find((p) => p.id === id) ?? null
      setProject(found)
      if (found) setNameDraft(found.name)

      const [sessionsRes, budgetRes] = await Promise.allSettled([
        listSessions({ project_id: id }),
        found ? getBudget('project', found.name) : Promise.resolve(null),
      ])
      if (sessionsRes.status === 'fulfilled') {
        setSessions(sessionsRes.value)
      } else if (!failed) {
        failed = extractErrorMessage(sessionsRes.reason)
      }
      if (budgetRes.status === 'fulfilled' && budgetRes.value) {
        setBudgetState(budgetRes.value)
        if (budgetRes.value.limit_cents > 0) {
          setBudgetInput((budgetRes.value.limit_cents / 100).toFixed(2))
        }
      }
      setError(failed)
    } catch (err) {
      setError(extractErrorMessage(err))
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [id])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load()
  }, [load])

  // If the project disappeared (e.g. deleted), bounce back to the list.
  useEffect(() => {
    if (!loading && !project && !error) navigate('/projects', { replace: true })
  }, [loading, project, error, navigate])

  const startRename = () => {
    setNameDraft(project?.name ?? '')
    setEditingName(true)
  }

  const handleRename = async () => {
    if (!id || !nameDraft.trim()) return
    setSavingName(true)
    try {
      const updated = await renameLocalProject(id, nameDraft.trim())
      setProject(updated)
      setEditingName(false)
      setError(null)
    } catch (err) {
      setError(extractErrorMessage(err))
    } finally {
      setSavingName(false)
    }
  }

  /** Assign a directory chosen via the native OS folder picker. */
  const handleAssignDirectory = async () => {
    if (!id) return
    setAssigning(true)
    setError(null)
    try {
      const picked = await invoke<string | null>('pick_directory')
      if (!picked) return // user cancelled
      const updated = await assignLocalProjectDirectory(id, picked)
      setProject(updated)
    } catch (err: unknown) {
      setError(typeof err === 'string' ? err : extractErrorMessage(err))
    } finally {
      setAssigning(false)
    }
  }

  const handleSaveBudget = async () => {
    if (!project) return
    setSavingBudget(true)
    try {
      const updated = await setBudget('project', budgetInput || '0', project.name)
      setBudgetState(updated)
      setError(null)
    } catch (err) {
      setError(extractErrorMessage(err))
    } finally {
      setSavingBudget(false)
    }
  }

  // Derived aggregates

  const aggregate = useMemo(() => {
    let spend = 0
    let inputTokens = 0
    let outputTokens = 0
    let activeCount = 0
    let contextTokens = 0
    for (const s of sessions) {
      spend += s.spend_cents ?? 0
      inputTokens += s.total_input_tokens ?? 0
      outputTokens += s.total_output_tokens ?? 0
      if (s.status === 'active') {
        activeCount += 1
        contextTokens += s.context_tokens ?? 0
      }
    }
    return { spend, inputTokens, outputTokens, activeCount, contextTokens }
  }, [sessions])

  const filteredSessions = useMemo(() => {
    let list = sessions
    if (statusFilter) list = list.filter((s) => s.status === statusFilter)
    if (debouncedSearch.trim()) {
      const q = debouncedSearch.toLowerCase()
      list = list.filter(
        (s) =>
          s.task_summary?.toLowerCase().includes(q) ||
          s.session_id.toLowerCase().includes(q),
      )
    }
    return list
  }, [sessions, statusFilter, debouncedSearch])

  // The daemon's project-scope ledger spend is not populated in the local
  // edition (ledger events don't carry project names), so the progress bar
  // is driven by the session spend aggregated client-side. The limit itself
  // comes from the budget policy endpoint.
  const spentCents = aggregate.spend
  const budgetRatio =
    budget && budget.limit_cents > 0 ? spentCents / budget.limit_cents : 0
  const currentTier =
    budgetRatio >= 1 ? 4 : budgetRatio >= 0.9 ? 3 : budgetRatio >= 0.7 ? 2 : 1
  const remainingCents = budget ? budget.limit_cents - spentCents : 0

  // Render

  return (
    <div className="mx-auto w-full max-w-[1000px]">
      {/* Header */}
      {error && (
        <div className="mb-6 rounded-lg border border-[var(--color-danger)] bg-[rgba(244,63,94,0.08)] px-4 py-3 text-sm text-[var(--color-danger)]">
          {error}
        </div>
      )}

      {/* Title row */}
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0 flex-1">
          {loading && !project ? (
            <Skeleton className="h-9 w-56" />
          ) : editingName ? (
            <div className="flex items-center gap-2">
              <input
                autoFocus
                value={nameDraft}
                onChange={(e) => setNameDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleRename()
                  if (e.key === 'Escape') setEditingName(false)
                }}
                className="w-full max-w-sm rounded-lg border border-[var(--color-border-strong)] bg-[var(--color-surface)] px-3 py-1.5 text-2xl font-semibold text-[var(--color-text-heading)] outline-none focus:border-[var(--color-accent)]"
                disabled={savingName}
              />
              <button
                onClick={handleRename}
                disabled={savingName || !nameDraft.trim()}
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-[var(--color-accent)] hover:bg-[var(--color-surface-highlight)] disabled:opacity-50"
                title="Save"
              >
                <Check className="h-4 w-4" />
              </button>
              <button
                onClick={() => setEditingName(false)}
                disabled={savingName}
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-[var(--color-text-muted)] hover:bg-[var(--color-surface-highlight)]"
                title="Cancel"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <h1 className="heading-display truncate text-3xl font-semibold text-[var(--color-text-heading)]">
                {project?.name ?? 'Project'}
              </h1>
              {project && (
                <button
                  onClick={startRename}
                  className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-surface-highlight)] hover:text-[var(--color-text-heading)]"
                  title="Rename project"
                >
                  <Pencil className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
          )}
          <div className="mt-2 flex flex-wrap items-center gap-2 text-sm text-[var(--color-text-muted)]">
            <span className="font-mono text-xs" title={id}>
              {id?.slice(0, 8)}
            </span>
            <span className="text-[var(--color-text-muted)]/40">·</span>
            <span>{project?.directories.length ?? 0} directories</span>
            <span className="text-[var(--color-text-muted)]/40">·</span>
            <span>
              created{' '}
              {project?.created_at
                ? new Date(project.created_at).toLocaleDateString(undefined, {
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric',
                  })
                : '-'}
            </span>
          </div>
        </div>
        <Button variant="secondary" loading={refreshing} onClick={load}>
          <RefreshCw className="h-4 w-4" /> Refresh
        </Button>
      </div>

      {/* Stat cards */}
      <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          icon={<Activity className="h-5 w-5 text-[var(--color-accent)]" />}
          label="Sessions"
          value={loading && !project ? undefined : String(sessions.length)}
          sub={`${aggregate.activeCount} active`}
        />
        <StatCard
          icon={<DollarSign className="h-5 w-5 text-[var(--color-accent)]" />}
          label="Total spend"
          value={loading && !project ? undefined : `$${(aggregate.spend / 100).toFixed(2)}`}
          sub={`${fmtTokens(aggregate.inputTokens)} in · ${fmtTokens(aggregate.outputTokens)} out`}
        />
        <StatCard
          icon={<Upload className="h-5 w-5 text-[var(--color-accent)]" />}
          label="Input tokens"
          value={loading && !project ? undefined : fmtTokens(aggregate.inputTokens)}
          sub="across all sessions"
        />
        <StatCard
          icon={<Download className="h-5 w-5 text-[var(--color-accent)]" />}
          label="Output tokens"
          value={loading && !project ? undefined : fmtTokens(aggregate.outputTokens)}
          sub={`${fmtTokens(aggregate.contextTokens)} live context`}
        />
      </div>

      {/* Budget card */}
      <Card className="mb-6 p-6">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
            <Wallet className="h-5 w-5 text-[var(--color-accent)]" />
          </div>
          <div className="flex-1">
            <h2 className="text-base font-semibold text-[var(--color-text-heading)]">
              Project budget
            </h2>
            <p className="text-xs text-[var(--color-text-muted)]">
              Applies to every session recorded under this project's directories.
            </p>
          </div>
          {budget && budget.limit_cents > 0 && (
            <Badge
              variant={
                currentTier === 4
                  ? 'danger'
                  : currentTier === 3
                    ? 'warning'
                    : currentTier === 2
                      ? 'warning'
                      : 'success'
              }
            >
              {BUDGET_TIERS.find((t) => t.tier === currentTier)?.label}
            </Badge>
          )}
        </div>
        {loading && !budget ? (
          <Skeleton className="h-10 w-full" />
        ) : budget && budget.limit_cents > 0 ? (
          <div className="space-y-4">
            {/* Progress bar */}
            <div>
              <div className="mb-1.5 flex items-baseline justify-between text-sm">
                <span className="text-[var(--color-text)]">
                  ${(spentCents / 100).toFixed(2)} spent
                </span>
                <span className="text-[var(--color-text-muted)]">
                  of ${(budget.limit_cents / 100).toFixed(2)}
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-[var(--color-surface-highlight)]">
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${Math.min(100, budgetRatio * 100).toFixed(1)}%`,
                    backgroundColor:
                      currentTier === 4
                        ? 'var(--color-danger)'
                        : currentTier === 3
                          ? 'var(--color-warning)'
                          : 'var(--color-accent)',
                  }}
                />
              </div>
              <p className="mt-1.5 text-xs text-[var(--color-text-muted)]">
                {remainingCents < 0
                  ? `Over budget by $${(Math.abs(remainingCents) / 100).toFixed(2)}`
                  : `$${(remainingCents / 100).toFixed(2)} remaining`}
              </p>
            </div>
            {/* Tier strip */}
            <div className="flex gap-2">
              {BUDGET_TIERS.map((t) => (
                <div
                  key={t.tier}
                  className={`flex flex-1 flex-col items-center rounded-lg border p-2 text-xs ${
                    currentTier === t.tier
                      ? 'border-[var(--color-border-strong)] bg-[var(--color-surface-highlight)]'
                      : 'border-[var(--color-border)]'
                  }`}
                >
                  <span className="font-medium text-[var(--color-text-heading)]">{t.label}</span>
                  <span className="text-[var(--color-text-muted)]">
                    {Math.round(t.threshold * 100)}%
                  </span>
                </div>
              ))}
            </div>
            {/* Update limit */}
            <div className="flex items-end gap-2">
              <Input
                label="Monthly limit (USD)"
                type="number"
                min="0"
                step="0.01"
                value={budgetInput}
                onChange={(e) => setBudgetInput(e.target.value)}
                className="max-w-[160px]"
              />
              <Button variant="secondary" loading={savingBudget} onClick={handleSaveBudget}>
                <Check className="h-4 w-4" /> Update
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex items-end gap-2">
            <Input
              label="Set a monthly limit (USD)"
              type="number"
              min="0"
              step="0.01"
              placeholder="e.g. 25.00"
              value={budgetInput}
              onChange={(e) => setBudgetInput(e.target.value)}
              className="max-w-[220px]"
            />
            <Button loading={savingBudget} onClick={handleSaveBudget}>
              <Plus className="h-4 w-4" /> Set budget
            </Button>
          </div>
        )}
      </Card>

      {/* Directories card */}
      <Card className="mb-6 p-6">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
            <FolderOpen className="h-5 w-5 text-[var(--color-accent)]" />
          </div>
          <div className="flex-1">
            <h2 className="text-base font-semibold text-[var(--color-text-heading)]">
              Working directories
            </h2>
            <p className="text-xs text-[var(--color-text-muted)]">
              Sessions started in these directories are attributed to this project.
            </p>
          </div>
          <Button variant="secondary" loading={assigning} onClick={handleAssignDirectory}>
            <FolderPlus className="h-4 w-4" /> Assign
          </Button>
        </div>
        {project && project.directories.length > 0 ? (
          <div className="space-y-2">
            {project.directories.map((dir) => (
              <div
                key={dir}
                className="flex items-center gap-2.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-3.5 py-2.5"
              >
                <FolderKanban className="h-4 w-4 shrink-0 text-[var(--color-text-muted)]" />
                <span className="truncate font-mono text-xs text-[var(--color-text)]" title={dir}>
                  {dir}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-[var(--color-text-muted)]">
            No directories assigned yet. Use "Assign" to pick a folder - sessions recorded under it
            will be attributed to this project automatically.
          </p>
        )}
      </Card>

      {/* Sessions section */}
      <div>
        <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <h2 className="heading-display text-xl font-semibold text-[var(--color-text-heading)]">
            Sessions
          </h2>
          <div className="flex items-center gap-3">
            <div className="w-44">
              <Input
                placeholder="Search sessions..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
            <div className="w-40">
              <ListSelect
                options={[
                  { value: '', title: 'All statuses' },
                  { value: 'active', title: 'Active' },
                  { value: 'paused', title: 'Paused' },
                  { value: 'completed', title: 'Completed' },
                ]}
                value={statusFilter}
                onChange={setStatusFilter}
                placeholder="All statuses"
              />
            </div>
          </div>
        </div>

        {loading ? (
          <div className="flex flex-col gap-1.5">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-[72px] w-full rounded-xl" />
            ))}
          </div>
        ) : filteredSessions.length === 0 ? (
          <EmptyState
            icon={<Activity className="h-10 w-10" />}
            title="No sessions yet"
            description={
              search || statusFilter
                ? 'Try adjusting your search or filters.'
                : 'Start an agent in one of the project directories and its sessions will appear here.'
            }
          />
        ) : (
          <div className="flex flex-col gap-1.5">
            {filteredSessions.map((s) => (
              <SessionRow key={s.session_id} session={s} projectId={id} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function StatCard({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode
  label: string
  value?: string
  sub?: string
}) {
  return (
    <Card className="p-6">
      <div className="mb-3 flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
          {icon}
        </div>
        <span className="text-sm font-medium text-[var(--color-text)]">{label}</span>
      </div>
      {value === undefined ? (
        <Skeleton className="h-9 w-20" />
      ) : (
        <div className="flex flex-col gap-1">
          <span className="font-mono text-2xl font-semibold text-[var(--color-text-heading)]">{value}</span>
          {sub && <span className="text-xs text-[var(--color-text-muted)]">{sub}</span>}
        </div>
      )}
    </Card>
  )
}

function SessionRow({ session, projectId }: { session: SessionSummary; projectId?: string }) {
  const navigate = useNavigate()
  const { color: statusColor, bg: statusBg, Icon: StatusIcon } = statusVisual(session.status)
  const rel = timeAgo(session.updated_at ?? session.created_at)

  return (
    <Card
      className="group flex items-center gap-3 p-3.5 transition-all hover:border-[var(--color-border-strong)] hover:bg-[var(--color-surface-highlight)] cursor-pointer"
      onClick={() =>
        navigate(`/sessions/${session.session_id}`, {
          state: sessionOriginState(
            projectId ? { kind: 'project', projectId } : { kind: 'sessions' },
          ),
        })
      }
    >
      {/* Status-colored left accent bar */}
      <div className="h-10 w-1 shrink-0 rounded-full" style={{ backgroundColor: statusColor }} />
      {/* Status icon */}
      <div
        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg"
        style={{ backgroundColor: statusBg }}
      >
        <StatusIcon className="h-4.5 w-4.5" style={{ color: statusColor }} />
      </div>
      {/* Main content */}
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="truncate text-sm font-medium text-[var(--color-text-heading)]">
          {session.task_summary || 'Untitled session'}
        </span>
        <div className="flex items-center gap-2 text-xs text-[var(--color-text-muted)]">
          <span className="font-mono">{session.session_id.slice(0, 8)}</span>
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
          <span className="flex items-center gap-1" title="Total spend">
            <DollarSign className="h-3 w-3" />
            {((session.spend_cents ?? 0) / 100).toFixed(2)}
          </span>
          <span className="flex items-center gap-1" title="Input tokens">
            <Download className="h-3 w-3" style={{ color: 'var(--color-info)' }} />
            {fmtTokens(session.total_input_tokens ?? 0)}
          </span>
          <span className="flex items-center gap-1" title="Output tokens">
            <Upload className="h-3 w-3" style={{ color: 'var(--color-success)' }} />
            {fmtTokens(session.total_output_tokens ?? 0)}
          </span>
          {/*<OutcomeLabel outcome={session.cost_outcome} />*/}
        </div>
        <Badge
          variant={
            session.status === 'active'
              ? 'success'
              : session.status === 'paused'
                ? 'warning'
                : 'default'
          }
        >
          {capitalizeStatus(session.status)}
        </Badge>
      </div>
    </Card>
  )
}
