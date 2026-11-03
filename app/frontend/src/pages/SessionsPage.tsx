import { presetToDateRange } from '../utils/helpers'
import { OutcomeLabel } from '../components/OutcomeLabel'
import { OutcomeDiagnostics } from '../components/OutcomeDiagnostics'
// Sessions: list of all agent sessions, with a detail drawer for events.
// Replaces `reincore sessions` and `reincore sessions --checkpoints`.

import { useEffect, useState, useCallback, useMemo, useRef } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  Activity,
  Pause,
  Play,
  Search,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  GitBranch,
  Pencil,
  Check,
  X,
  Filter,
  XCircle,
  CheckCircle2,
  Circle,
  Cpu,
  DollarSign,
  Clock,
  ShieldCheck,
  TriangleAlert
} from 'lucide-react'
import {
  listSessionsPage,
  listSubagentSessions,
  getSession,
  getSessionTimeline,
  pauseSession,
  completeSession,
  resumeSession,
  renameSession,
  listProjects,
  extractErrorMessage
} from '../api'
import type { SessionListFilter } from '../api'
import { Card, Badge, Skeleton, EmptyState, Button, Input, ListSelect } from '../components/ui'
import type { ListSelectOption } from '../components/ui'
import { SessionEventTimeline } from '../components/SessionEventTimeline'
import type { ProcessKind } from '../components/SessionEventTimeline'
import { sessionOriginState } from '../navigation/sessionOrigin'
import type { SessionSummary, SessionDetail, SessionTimelineSummary, ProjectSummary } from '../types'
import { useSessionEventStream } from '../hooks/useSessionEventStream'
import { capitalizeStatus } from '../utils/status'

const STATUS_OPTIONS: ListSelectOption[] = [
  { value: '', title: 'All statuses' },
  { value: 'active', title: 'Active' },
  { value: 'paused', title: 'Paused' },
  { value: 'completed', title: 'Completed' },
  { value: 'checkpointed', title: 'Checkpointed' },
]



const MODE_OPTIONS: ListSelectOption[] = [
  { value: '', title: 'All modes' },
  { value: 'interactive', title: 'Interactive' },
  { value: 'background', title: 'Background' },
]

const DATE_PRESET_OPTIONS: ListSelectOption[] = [
  { value: '', title: 'All time' },
  { value: 'today', title: 'Today' },
  { value: 'this_month', title: 'This month' },
  { value: 'last_month', title: 'Last month' },
  { value: 'custom', title: 'Custom range' },
]

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

export function SessionsPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [refreshing, setRefreshing] = useState(false)
  const [statusFilter, setStatusFilter] = useState('')
  const [modeFilter, setModeFilter] = useState('')
  const [projectFilter, setProjectFilter] = useState('')
  const [datePreset, setDatePreset] = useState('')
  const [customDateFrom, setCustomDateFrom] = useState('')
  const [customDateTo, setCustomDateTo] = useState('')
  const [showFilters, setShowFilters] = useState(false)
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [nextCursor, setNextCursor] = useState<string | null>(null)
  const requestGeneration = useRef(0)
  const activeRequest = useRef<AbortController | null>(null)
  const searchDebounce = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Load projects once for the project filter dropdown
  useEffect(() => {
    listProjects().then(setProjects).catch(() => {})
  }, [])

  // Build project options from the projects list
  const projectOptions = useMemo<ListSelectOption[]>(() => {
    const opts: ListSelectOption[] = [{ value: '', title: 'All projects' }]
    for (const p of projects) {
      const value = p.project_id ?? p.project_path ?? p.project_name ?? ''
      if (!value) continue
      opts.push({ value, title: p.project_name || p.project_path || value })
    }
    return opts
  }, [projects])

  // Compute the ISO 8601 date range from the preset / custom dates
  const [dateFrom, dateTo] = useMemo(
    () => presetToDateRange(datePreset, customDateFrom, customDateTo),
    [datePreset, customDateFrom, customDateTo],
  )

  // Build the backend filter from the current state
  const backendFilter = useMemo<SessionListFilter>(() => {
    const filter: SessionListFilter = {}
    if (statusFilter) filter.status = statusFilter
    if (modeFilter) filter.mode = modeFilter
    if (projectFilter) {
      const project = projects.find((candidate) => candidate.project_id === projectFilter)
      if (project) filter.project_id = projectFilter
      else filter.path = projectFilter
    }
    if (dateFrom) filter.date_from = dateFrom
    if (dateTo) filter.date_to = dateTo
    return filter
  }, [statusFilter, modeFilter, projectFilter, projects, dateFrom, dateTo])

  useEffect(() => {
    const generation = ++requestGeneration.current
    activeRequest.current?.abort()
    const controller = new AbortController()
    activeRequest.current = controller
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true)
    listSessionsPage(backendFilter, undefined, controller.signal)
      .then((page) => {
        if (generation !== requestGeneration.current) return
        setSessions(page.sessions)
        setNextCursor(page.next_cursor ?? null)
        setError(null)
      })
      .catch((err) => {
        if (generation === requestGeneration.current) {
          setSessions([])
          setNextCursor(null)
          setError(extractErrorMessage(err))
        }
      })
      .finally(() => {
        if (generation === requestGeneration.current) { setLoading(false); setRefreshing(false) }
      })
    // The counter intentionally invalidates every outstanding request on cleanup.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    return () => { ++requestGeneration.current; activeRequest.current?.abort() }
  }, [backendFilter])

  const fetchPage = useCallback(async (before?: string) => {
    const generation = ++requestGeneration.current
    activeRequest.current?.abort()
    const controller = new AbortController()
    activeRequest.current = controller
    setRefreshing(true)
    try {
      const page = await listSessionsPage(backendFilter, before, controller.signal)
      if (generation !== requestGeneration.current) return
      setSessions(previous => before
        ? [...new Map([...previous, ...page.sessions].map(s => [s.session_id, s])).values()]
        : page.sessions)
      setNextCursor(page.next_cursor ?? null)
      setError(null)
    } catch (err) {
      if (generation === requestGeneration.current) {
        if (!before) { setSessions([]); setNextCursor(null) }
        setError(extractErrorMessage(err))
      }
    } finally {
      if (generation === requestGeneration.current) { setRefreshing(false); setLoading(false) }
    }
  }, [backendFilter])
  const handleRefresh = useCallback(() => fetchPage(), [fetchPage])

  // Debounced search: update the search term after the user stops typing.
  // The search itself is client-side (within the already backend-filtered results).
  const [debouncedSearch, setDebouncedSearch] = useState('')
  useEffect(() => {
    if (searchDebounce.current) clearTimeout(searchDebounce.current)
    searchDebounce.current = setTimeout(() => setDebouncedSearch(search), 250)
    return () => { if (searchDebounce.current) clearTimeout(searchDebounce.current) }
  }, [search])

  const hasActiveFilters = statusFilter || modeFilter || projectFilter || datePreset || undefined

  const clearFilters = () => {
    setStatusFilter('')
    setModeFilter('')
    setProjectFilter('')
    setDatePreset('')
    setCustomDateFrom('')
    setCustomDateTo('')
    setSearch('')
  }

  // Client-side text search within the backend-filtered results
  const filtered = useMemo(() => {
    if (!debouncedSearch) return sessions
    const q = debouncedSearch.toLowerCase()
    return sessions.filter((s) =>
      s.session_id.toLowerCase().includes(q) ||
      (s.project_name ?? '').toLowerCase().includes(q) ||
      (s.task_summary ?? '').toLowerCase().includes(q),
    )
  }, [sessions, debouncedSearch])


  return (
    <div className="mx-auto w-full max-w-[1000px]">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="heading-display text-3xl font-semibold text-[var(--color-text-heading)]">Sessions</h1>
        <Button variant="secondary" onClick={handleRefresh} loading={refreshing}>
          <RefreshCw className="h-4 w-4" /> Refresh
        </Button>
      </div>

      {error && (
        <div className="mb-6 rounded-lg border border-[var(--color-danger)] bg-[rgba(244,63,94,0.08)] px-4 py-3 text-sm text-[var(--color-danger)]">
          {error}
        </div>
      )}

      {/* Search + Filters */}
      <div className="mb-4 space-y-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
          <div className="flex-1 max-w-sm">
            <Input
              placeholder="Search loaded sessions by ID, project, or task..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <button
            onClick={() => setShowFilters((v) => !v)}
            className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-sm font-medium transition-colors ${
              hasActiveFilters
                ? 'border-[var(--color-accent)] text-[var(--color-accent)]'
                : 'border-[var(--color-border)] text-[var(--color-text-secondary)] hover:border-[var(--color-border-strong)] hover:text-[var(--color-text-heading)]'
            }`}
          >
            <Filter className="h-4 w-4" />
            Filters
            {hasActiveFilters && (
              <span className="flex h-5 min-w-5 items-center justify-center rounded-full bg-[var(--color-accent)] px-1.5 text-xs font-semibold text-white">
                {[statusFilter, modeFilter, projectFilter, datePreset].filter(Boolean).length}
              </span>
            )}
          </button>
        </div>

        {showFilters && (
          <Card className="p-4">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <ListSelect
                label="Project"
                options={projectOptions}
                value={projectFilter}
                onChange={setProjectFilter}
                placeholder="All projects"
              />
              <ListSelect
                label="Status"
                options={STATUS_OPTIONS}
                value={statusFilter}
                onChange={setStatusFilter}
              />
              <ListSelect
                label="Mode"
                options={MODE_OPTIONS}
                value={modeFilter}
                onChange={setModeFilter}
              />
              <ListSelect
                label="Date"
                options={DATE_PRESET_OPTIONS}
                value={datePreset}
                onChange={setDatePreset}
              />
              {datePreset === 'custom' && (
                <div className="flex flex-col gap-1.5 sm:col-span-1 lg:col-span-2">
                  <label className="text-sm font-medium text-[var(--color-text)]">Custom date range</label>
                  <div className="flex items-center gap-2">
                    <input
                      type="date"
                      value={customDateFrom}
                      onChange={(e) => setCustomDateFrom(e.target.value)}
                      className="flex-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-accent)]"
                    />
                    <span className="text-xs text-[var(--color-text-muted)]">to</span>
                    <input
                      type="date"
                      value={customDateTo}
                      onChange={(e) => setCustomDateTo(e.target.value)}
                      className="flex-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-accent)]"
                    />
                  </div>
                </div>
              )}
            </div>
            {hasActiveFilters && (
              <div className="mt-3 flex items-center justify-end">
                <button
                  onClick={clearFilters}
                  className="flex items-center gap-1.5 text-xs font-medium text-[var(--color-text-muted)] transition-colors hover:text-[var(--color-danger)]"
                >
                  <XCircle className="h-3.5 w-3.5" />
                  Clear all filters
                </button>
              </div>
            )}
          </Card>
        )}
      </div>

      {loading ? (
        <div className="flex flex-col gap-1.5">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-[72px] w-full rounded-xl" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <EmptyState
          icon={<Activity className="h-10 w-10" />}
          title="No sessions found"
          description={search || hasActiveFilters ? 'Try adjusting your search or filters.' : 'Start an agent through ReinCore and its sessions will appear here.'}
        />
      ) : (
        <>
          <p className="mb-2 text-xs text-[var(--color-text-muted)]">
            Showing {filtered.length} of {sessions.length} loaded sessions
          </p>
          <div className="flex flex-col gap-1.5">
            {filtered.map((s) => (
              <SessionRow
                key={s.session_id}
                session={s}
              />
            ))}
          </div>
        </>
      )}
      {!loading && nextCursor && (
        <Button className="mt-4" disabled={refreshing} onClick={() => fetchPage(nextCursor)}>
          {refreshing ? 'Loading...' : 'Load more sessions'}
        </Button>
      )}
    </div>
  )
}

function SessionRow({ session }: { session: SessionSummary }) {
  const navigate = useNavigate()
  const [expanded, setExpanded] = useState(false)
  const [subagents, setSubagents] = useState<SessionSummary[]>([])
  const [subagentLoading, setSubagentLoading] = useState(false)

  const toggleSubagents = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!expanded && subagents.length === 0 && !subagentLoading) {
      setSubagentLoading(true)
      try {
        const data = await listSubagentSessions(session.session_id)
        setSubagents(data)
      } catch {
        // silently fail - the expand just shows nothing
      } finally {
        setSubagentLoading(false)
      }
    }
    setExpanded(!expanded)
  }

  const { color: statusColor, bg: statusBg, Icon: StatusIcon } = statusVisual(session.status)
  const rel = timeAgo(session.updated_at ?? session.created_at)

  return (
    <div>
      <Card
        className="group flex items-center gap-3 p-3.5 transition-all hover:border-[var(--color-border-strong)] hover:bg-[var(--color-surface-highlight)] cursor-pointer"
        onClick={() =>
          navigate(`/sessions/${session.session_id}`, {
            state: sessionOriginState({ kind: 'sessions' }),
          })
        }
      >
        {/* Subagent toggle - sits before the accent bar so it doesn't shift content */}
        <button
          onClick={toggleSubagents}
          className="flex h-5 w-5 shrink-0 items-center justify-center rounded text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-surface-highlight)] hover:text-[var(--color-text-heading)] cursor-pointer"
          title="Toggle subagent sessions"
        >
          {subagentLoading ? (
            <RefreshCw className="h-3.5 w-3.5 animate-spin" />
          ) : expanded ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </button>
        {/* Status-colored left accent bar */}
        <div className="h-10 w-1 shrink-0 rounded-full" style={{ backgroundColor: statusColor }} />
        {/* Status icon */}
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg" style={{ backgroundColor: statusBg }}>
          <StatusIcon className="h-4.5 w-4.5" style={{ color: statusColor }} />
        </div>
        {/* Main content */}
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="truncate text-sm font-medium text-[var(--color-text-heading)]">
            {session.task_summary || 'Untitled session'}
          </span>
          <div className="flex items-center gap-2 text-xs text-[var(--color-text-muted)]">
            <span className="font-mono">{session.session_id.slice(0, 8)}</span>
            <span className="text-[var(--color-text-muted)]/40">·</span>
            <span className="truncate">{session.project_name || session.project_path || '-'}</span>
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
              {((session.spend_cents ?? 0) / 100).toFixed(2)}
            </span>
            <span className="flex items-center gap-1">
              <Cpu className="h-3 w-3" />
              {fmtTokens(session.context_tokens ?? 0)}
            </span>
            {/*<OutcomeLabel outcome={session.cost_outcome} />*/}

          </div>
          <Badge variant={session.status === 'active' ? 'success' : session.status === 'paused' ? 'warning' : 'default'}>
            {capitalizeStatus(session.status)}
          </Badge>
        </div>
      </Card>
      {expanded && subagents.length > 0 && (
        <div className="ml-10 mt-1 flex flex-col gap-1.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] p-2">
          {subagents.map((sa) => {
            const { color: saColor, bg: saBg, Icon: SaIcon } = statusVisual(sa.status)
            const saRel = timeAgo(sa.updated_at ?? sa.created_at)
            return (
            <div
              key={sa.session_id}
              className="group flex items-center gap-2.5 rounded-md px-3 py-2 transition-colors hover:bg-[var(--color-surface-highlight)] cursor-pointer"
              onClick={() =>
                navigate(`/sessions/${sa.session_id}`, {
                  state: sessionOriginState({ kind: 'sessions' }),
                })
              }
            >
              <GitBranch className="h-3.5 w-3.5 shrink-0 text-[var(--color-text-muted)]" />
              <div className="h-7 w-0.5 shrink-0 rounded-full" style={{ backgroundColor: saColor }} />
              <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded" style={{ backgroundColor: saBg }}>
                <SaIcon className="h-3 w-3" style={{ color: saColor }} />
              </div>
              <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                <span className="truncate text-xs font-medium text-[var(--color-text-secondary)]">
                  {sa.task_summary || 'Untitled subagent'}
                </span>
                <div className="flex items-center gap-1.5 text-[10px] text-[var(--color-text-muted)]">
                  <span className="font-mono">{sa.session_id.slice(0, 8)}</span>
                  {saRel && (
                    <>
                      <span className="text-[var(--color-text-muted)]/40">·</span>
                      <span className="flex items-center gap-0.5 whitespace-nowrap">
                        <Clock className="h-2.5 w-2.5" />
                        {saRel}
                      </span>
                    </>
                  )}
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <span className="flex items-center gap-0.5 text-[10px] text-[var(--color-text-muted)]">
                  <DollarSign className="h-2.5 w-2.5" />
                  {((sa.spend_cents ?? 0) / 100).toFixed(2)}
                </span>
                <Badge variant={sa.status === 'active' ? 'success' : sa.status === 'paused' ? 'warning' : 'default'}>
                  {capitalizeStatus(sa.status)}
                </Badge>
              </div>
            </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

export function SessionDetailPage() {
  const { id } = useParams<{ id: string }>()
  const [session, setSession] = useState<SessionDetail | null>(null)
  const [timeline, setTimeline] = useState<SessionTimelineSummary | null>(null)
  const [selectedKinds, setSelectedKinds] = useState<Set<ProcessKind>>(new Set())
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [pausing, setPausing] = useState(false)
  const [completing, setCompleting] = useState(false)
  const [pauseResult, setPauseResult] = useState<{ token: string; expires: string } | null>(null)
  const [resumeToken, setResumeToken] = useState('')
  const [resumeMsg, setResumeMsg] = useState<string | null>(null)
  const [subagentSessions, setSubagentSessions] = useState<SessionSummary[]>([])
  const [editingName, setEditingName] = useState(false)
  const [nameDraft, setNameDraft] = useState('')
  const [savingName, setSavingName] = useState(false)

  const startRename = () => {
    setNameDraft(session?.task_summary ?? '')
    setEditingName(true)
  }

  const handleRename = async () => {
    if (!id || !nameDraft.trim()) return
    setSavingName(true)
    try {
      const updated = await renameSession(id, nameDraft.trim())
      setSession(updated)
      setEditingName(false)
      setError(null)
    } catch (err) {
      setError(extractErrorMessage(err))
    } finally {
      setSavingName(false)
    }
  }

  const refreshSnapshot = useCallback(async () => {
    if (!id) return
    const [detail, timelineSummary, subs] = await Promise.all([
      getSession(id).catch(() => null),
      getSessionTimeline(id),
      listSubagentSessions(id).catch(() => [] as SessionSummary[]),
    ])
    if (detail) setSession(detail)
    setTimeline(timelineSummary)
    setSubagentSessions(subs)
    setError(null)
  }, [id])

  const load = useCallback(async () => {
    setLoading(true)
    try {
      await refreshSnapshot()
    } catch (err) {
      setError(extractErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }, [refreshSnapshot])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load()
  }, [load])

  const liveStream = useSessionEventStream(
    id,
    timeline?.latest_seq ?? 0,
    Boolean(id && timeline && !loading),
  )

  useEffect(() => {
    if (!timeline || liveStream.lastSeq <= (timeline.latest_seq ?? 0)) return
    const timer = window.setTimeout(() => {
      void refreshSnapshot().catch((err) => setError(extractErrorMessage(err)))
    }, 350)
    return () => window.clearTimeout(timer)
  }, [liveStream.lastSeq, refreshSnapshot, timeline])

  const handlePause = async () => {
    if (!id) return
    setPausing(true)
    try {
      const res = await pauseSession(id, 'user-pause')
      setPauseResult({ token: res.checkpoint_token, expires: res.expires_at })
    } catch (err) {
      setError(extractErrorMessage(err))
    } finally {
      setPausing(false)
    }
  }

  const handleComplete = async () => {
    if (!id) return
    setCompleting(true)
    try {
      const completed = await completeSession(id, 'completed in desktop app')
      setSession(completed)
      await load()
    } catch (err) {
      setError(extractErrorMessage(err))
    } finally {
      setCompleting(false)
    }
  }

  const handleResume = async () => {
    if (!resumeToken) return
    try {
      const res = await resumeSession(resumeToken)
      setResumeMsg(`Resumed session ${res.session_id} at step ${res.resumed_at_seq}`)
      setResumeToken('')
      load()
    } catch (err) {
      setError(extractErrorMessage(err))
    }
  }

  return (
    <div className="mx-auto w-full max-w-[1000px]">
      {/* Session header card */}
      <Card className="mb-6 overflow-hidden">
        <div className="p-5 sm:p-6">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0 flex-1">
              {editingName ? (
                <div className="flex items-center gap-2">
                  <input
                    autoFocus
                    value={nameDraft}
                    onChange={(e) => setNameDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleRename()
                      if (e.key === 'Escape') setEditingName(false)
                    }}
                    className="w-full rounded-lg border border-[var(--color-border-strong)] bg-[var(--color-surface)] px-3 py-1.5 text-xl font-semibold text-[var(--color-text-heading)] outline-none focus:border-[var(--color-accent)]"
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
                  <h1 className="heading-display line-clamp-2 text-2xl font-semibold leading-tight text-[var(--color-text-heading)]">
                    {session?.task_summary || `Session ${id?.slice(0, 8)}`}
                  </h1>
                  {session && (
                    <button
                      onClick={startRename}
                      className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-surface-highlight)] hover:text-[var(--color-text-heading)]"
                      title="Rename session"
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              )}
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <span className="font-mono text-xs text-[var(--color-text-muted)]" title={id}>
                  {id?.slice(0, 8)}
                </span>
                {session && (
                  <>
                    <Badge variant={session.status === 'active' ? 'success' : session.status === 'paused' ? 'warning' : 'default'}>
                      {capitalizeStatus(session.status)}
                    </Badge>
                    <Badge variant="info">{capitalizeStatus(session.mode)}</Badge>
                  </>
                )}
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              {session?.status !== 'completed' && (
                <Button loading={completing} onClick={handleComplete}>
                  <CheckCircle2 className="h-4 w-4" /> Complete
                </Button>
              )}
              <Button variant="secondary" loading={pausing} onClick={handlePause} disabled={session?.status === 'completed'}>
                <Pause className="h-4 w-4" /> Pause
              </Button>
            </div>
          </div>

          {/* Stat chips */}
          {session && (
            <div className="mt-4 flex flex-wrap gap-2">
              {(session.project_name || session.project_path) && (
                <span className="inline-flex items-center gap-1.5 rounded-lg bg-[var(--color-bg-subtle)] px-3 py-1.5 text-xs font-medium text-[var(--color-text-secondary)]">
                  <span className="text-[var(--color-text-muted)]">Project</span>
                  <span>{session.project_name || session.project_path}</span>
                </span>
              )}
              <span className="inline-flex items-center gap-1.5 rounded-lg bg-[var(--color-bg-subtle)] px-3 py-1.5 text-xs font-medium text-[var(--color-text-secondary)]">
                <span className="text-[var(--color-text-muted)]">Spend</span>
                <span>${((session.spend_cents ?? 0) / 100).toFixed(2)}</span>
              </span>
              <OutcomeLabel outcome={session.cost_outcome} appearance="stat" />
              {session.cost_outcome?.completed && (
                <span className="inline-flex items-center gap-1.5 rounded-lg bg-[var(--color-bg-subtle)] px-3 py-1.5 text-xs font-medium text-[var(--color-text-secondary)]">
                  {session.cost_outcome.clean_completion ? (
                    <ShieldCheck className="h-3.5 w-3.5 text-[var(--color-success)]" />
                  ) : (
                    <TriangleAlert className="h-3.5 w-3.5 text-[var(--color-warning)]" />
                  )}
                  <span>{session.cost_outcome.clean_completion
                    ? 'Completed without intervention'
                    : `${session.cost_outcome.intervention_metrics?.detections ?? session.cost_outcome.loop_intervention_count} intervention${(session.cost_outcome.intervention_metrics?.detections ?? session.cost_outcome.loop_intervention_count) === 1 ? '' : 's'} detected`}</span>
                </span>
              )}
              <span className="inline-flex items-center gap-1.5 rounded-lg bg-[var(--color-bg-subtle)] px-3 py-1.5 text-xs font-medium text-[var(--color-text-secondary)]">
                <span className="text-[var(--color-text-muted)]">Context</span>
                <span>{(session.context_tokens ?? 0).toLocaleString()} tok</span>
              </span>
              <span className="inline-flex items-center gap-1.5 rounded-lg bg-[var(--color-bg-subtle)] px-3 py-1.5 text-xs font-medium text-[var(--color-text-secondary)]">
                <span className="text-[var(--color-text-muted)]">Input</span>
                <span>{(session.total_input_tokens ?? 0).toLocaleString()} tok</span>
              </span>
              <span className="inline-flex items-center gap-1.5 rounded-lg bg-[var(--color-bg-subtle)] px-3 py-1.5 text-xs font-medium text-[var(--color-text-secondary)]">
                <span className="text-[var(--color-text-muted)]">Output</span>
                <span>{(session.total_output_tokens ?? 0).toLocaleString()} tok</span>
              </span>
            </div>
          )}
        </div>
      </Card>

      {session && <Card className="mb-6 p-4"><OutcomeDiagnostics key={session.id} session={session} onUpdate={setSession} /></Card>}

      {pauseResult && (
        <Card className="mb-6 p-4">
          <p className="text-sm font-medium text-[var(--color-text-heading)]">Checkpoint created</p>
          <p className="mt-1 font-mono text-xs text-[var(--color-text)] break-all">{pauseResult.token}</p>
          <p className="mt-1 text-xs text-[var(--color-text-muted)]">Expires: {pauseResult.expires}</p>
        </Card>
      )}

      {/* Resume from token - collapsed by default */}
      <details className="mb-6">
        <summary className="flex cursor-pointer list-none items-center gap-2 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3 text-sm font-medium text-[var(--color-text-secondary)] transition-colors hover:bg-[var(--color-surface-highlight)]/40 [&::-webkit-details-marker]:hidden">
          <Play className="h-4 w-4 text-[var(--color-text-muted)]" />
          Resume from checkpoint token
        </summary>
        <Card className="mt-2 p-4">
          <div className="flex gap-2">
            <Input
              placeholder="Paste checkpoint token..."
              value={resumeToken}
              onChange={(e) => setResumeToken(e.target.value)}
              className="flex-1"
            />
            <Button onClick={handleResume} disabled={!resumeToken}>
              <Play className="h-4 w-4" /> Resume
            </Button>
          </div>
          {resumeMsg && <p className="mt-2 text-xs text-[var(--color-success)]">{resumeMsg}</p>}
        </Card>
      </details>

      {error && (
        <div className="mb-6 rounded-lg border border-[var(--color-danger)] bg-[rgba(244,63,94,0.08)] px-4 py-3 text-sm text-[var(--color-danger)]">
          {error}
        </div>
      )}

      <div data-activity-scroll-region>
        <section data-activity-workspace className="sticky top-16 flex h-[calc(100dvh-4rem)] min-h-0 flex-col bg-[var(--color-bg)] py-4" aria-label="Activity timeline workspace">
        <div className="mb-3 shrink-0">
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-lg font-semibold text-[var(--color-text-heading)]">Activity timeline</h2>
          </div>
          <p className="mt-0.5 text-xs text-[var(--color-text-muted)]">
            Each request is grouped with its reasoning, tool calls, and response. Click a step to inspect details.
          </p>
        </div>
        {loading ? (
          <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto overscroll-contain">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-12 w-full rounded-lg" />
            ))}
          </div>
        ) : !timeline ? (
          <EmptyState icon={<Search className="h-8 w-8" />} title="No events" description="This session has no recorded events yet." />
        ) : (
          <SessionEventTimeline
            key={id}
            sessionId={id!}
            timeline={timeline}
            selectedKinds={selectedKinds}
            setSelectedKinds={setSelectedKinds}
            subagentSessions={subagentSessions}
            liveEvents={liveStream.events}
            connectionState={liveStream.connectionState}
          />
        )}
        </section>
        <div data-activity-scroll-range aria-hidden="true" />
      </div>
    </div>
  )
}
