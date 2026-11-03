// Budget: shows the per-session budget ledger and tier transitions. This is
// the visual surface for PROJECT.md §3.6 (Progressive Budget Degradation).
// Replaces `reincore budget --usage` and `reincore budget --history`.
//
// For session and project scopes, the user picks from a dropdown of
// available sessions/projects rather than typing a key manually.
// The session scope also has a "General" option that sets a default budget
// for all sessions (overridable per-session).
// For day and month scopes, there is a "General" default and a date/month
// picker for setting specific date/month overrides.

import { useEffect, useState, useMemo } from 'react'
import { Wallet, TrendingDown, AlertTriangle, Info } from 'lucide-react'
import {
  getBudget,
  getBudgetHistory,
  setBudget,
  listSessions,
  listProjects,
  extractErrorMessage,
} from '../api'
import { Card, Badge, Skeleton, Button, Input, EmptyState, ListSelect, DatePicker } from '../components/ui'
import type { ListSelectOption } from '../components/ui'
import type { BudgetInfo, SessionSummary, ProjectSummary } from '../types'

const TIERS = [
  { tier: 1, label: 'Normal', description: 'Strong model, full context', color: 'success' as const },
  { tier: 2, label: 'Economy', description: 'Cheap model for routine steps', color: 'info' as const },
  { tier: 3, label: 'Compact', description: 'Aggressive context compaction', color: 'warning' as const },
  { tier: 4, label: 'Paused', description: 'Resumable checkpoint', color: 'danger' as const },
]

const SCOPES = ['session', 'project', 'day', 'month'] as const
type Scope = (typeof SCOPES)[number]

export function BudgetPage() {
  const [scope, setScope] = useState<Scope>('session')
  const [key, setKey] = useState('') // '' = general/default
  const [budget, setBudgetState] = useState<BudgetInfo | null>(null)
  const [history, setHistory] = useState<BudgetInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [limitInput, setLimitInput] = useState('')

  // Dropdown data
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [projects, setProjects] = useState<ProjectSummary[]>([])

  // Load sessions and projects for dropdowns (once on mount)
  useEffect(() => {
    Promise.allSettled([listSessions(), listProjects()]).then(([s, p]) => {
      if (s.status === 'fulfilled') setSessions(s.value)
      if (p.status === 'fulfilled') setProjects(p.value)
    })
  }, [])

  // Build dropdown options based on scope
  const sessionOptions: ListSelectOption[] = useMemo(() => {
    return [
      { value: '', title: 'General (all sessions)', subtitle: 'Default budget for every session' },
      ...sessions.map((s) => ({
        value: s.session_id,
        title: s.task_summary?.slice(0, 60) || s.session_id.slice(0, 8),
        subtitle: s.project_name || s.project_path || 'No project',
      })),
    ]
  }, [sessions])

  const projectOptions: ListSelectOption[] = useMemo(() => {
    return projects.map((p) => ({
      value: p.project_name,
      title: p.project_name,
      subtitle: p.project_path || undefined,
    }))
  }, [projects])

  const isGeneralSession = scope === 'session' && key === ''
  const isGeneralDay = scope === 'day' && key === ''
  const isGeneralMonth = scope === 'month' && key === ''

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [b, h] = await Promise.allSettled([
        getBudget(scope, key || undefined),
        getBudgetHistory(scope, key || undefined),
      ])
      if (b.status === 'fulfilled') {
        setBudgetState(b.value)
        setLimitInput((b.value.limit_cents / 100).toFixed(2))
      } else {
        setError(extractErrorMessage(b.reason))
      }
      if (h.status === 'fulfilled') setHistory(h.value)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scope, key])

  // Reset key when scope changes
  const handleScopeChange = (s: Scope) => {
    setScope(s)
    setKey('') // reset to general/empty on scope switch
  }

  const handleSetLimit = async () => {
    if (!limitInput) return
    try {
      const updated = await setBudget(scope, limitInput, key || undefined)
      setBudgetState(updated)
    } catch (err) {
      setError(extractErrorMessage(err))
    }
  }

  const pct = budget && budget.limit_cents > 0 ? (budget.spent_cents / budget.limit_cents) * 100 : 0
  const currentTier = budget?.tier ?? 1
  const remaining = budget ? budget.remaining_cents / 100 : 0

  return (
    <div className="mx-auto w-full max-w-[1000px]">
      <h1 className="heading-display mb-6 text-3xl font-semibold text-[var(--color-text-heading)]">Budget</h1>

      {error && (
        <div className="mb-6 rounded-lg border border-[var(--color-danger)] bg-[rgba(244,63,94,0.08)] px-4 py-3 text-sm text-[var(--color-danger)]">
          {error}
        </div>
      )}

      {/* Scope selector */}
      <div className="mb-6 flex flex-wrap gap-3">
        <div className="flex gap-2">
          {SCOPES.map((s) => (
            <button
              key={s}
              onClick={() => handleScopeChange(s)}
              className={`rounded-lg border px-3 py-2 text-sm font-medium transition-colors cursor-pointer ${
                scope === s
                  ? 'border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-text-heading)]'
                  : 'border-transparent text-[var(--color-text)] hover:bg-[var(--color-surface)]'
              }`}
            >
              {s.charAt(0).toUpperCase() + s.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Dropdown selector for session/project scopes */}
      {scope === 'session' && (
        <div className="mb-6 max-w-md">
          <ListSelect
            label="Session"
            options={sessionOptions}
            value={key}
            onChange={setKey}
            placeholder="Select a session..."
          />
        </div>
      )}

      {scope === 'project' && (
        <div className="mb-6 max-w-md">
          <ListSelect
            label="Project"
            options={projectOptions}
            value={key}
            onChange={setKey}
            placeholder="Select a project..."
          />
        </div>
      )}

      {/* Date picker for day/month scopes */}
      {(scope === 'day' || scope === 'month') && (
        <div className="mb-6 max-w-md">
          <DatePicker
            label={scope === 'day' ? 'Day budget' : 'Month budget'}
            mode={scope}
            value={key}
            onChange={setKey}
          />
        </div>
      )}

      {/* General budget info banner */}
      {isGeneralSession && (
        <div className="mb-6 flex items-start gap-3 rounded-lg border border-[var(--color-info)] bg-[rgba(56,189,248,0.08)] px-4 py-3 text-sm text-[var(--color-text)]">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-[var(--color-info)]" />
          <span>
            This sets the <strong>default budget</strong> for all sessions. Individual sessions
            can override this by selecting them in the dropdown above and setting a specific limit.
          </span>
        </div>
      )}

      {isGeneralDay && (
        <div className="mb-6 flex items-start gap-3 rounded-lg border border-[var(--color-info)] bg-[rgba(56,189,248,0.08)] px-4 py-3 text-sm text-[var(--color-text)]">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-[var(--color-info)]" />
          <span>
            This sets the <strong>default daily budget</strong> applied to every day. You can
            set a specific override for a particular date by switching to "Specific date" above.
          </span>
        </div>
      )}

      {isGeneralMonth && (
        <div className="mb-6 flex items-start gap-3 rounded-lg border border-[var(--color-info)] bg-[rgba(56,189,248,0.08)] px-4 py-3 text-sm text-[var(--color-text)]">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-[var(--color-info)]" />
          <span>
            This sets the <strong>default monthly budget</strong> applied to every month. You can
            set a specific override for a particular month by switching to "Specific month" above.
          </span>
        </div>
      )}

      {/* Override indicator */}
      {scope === 'session' && key !== '' && (
        <div className="mb-6 flex items-start gap-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3 text-sm text-[var(--color-text-muted)]">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-[var(--color-text-muted)]" />
          <span>
            This is a <strong>per-session override</strong>. If unset, the session falls back to the
            general session budget (or project/day/month budget).
          </span>
        </div>
      )}

      {(scope === 'day' && key !== '') && (
        <div className="mb-6 flex items-start gap-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3 text-sm text-[var(--color-text-muted)]">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-[var(--color-text-muted)]" />
          <span>
            This is a <strong>date-specific override</strong> for {key}. If unset, the day falls back
            to the general daily budget.
          </span>
        </div>
      )}

      {(scope === 'month' && key !== '') && (
        <div className="mb-6 flex items-start gap-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3 text-sm text-[var(--color-text-muted)]">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-[var(--color-text-muted)]" />
          <span>
            This is a <strong>month-specific override</strong> for {key}. If unset, the month falls
            back to the general monthly budget.
          </span>
        </div>
      )}

      {/* Overview cards */}
      <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Card className="p-6">
          <div className="mb-3 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
              <Wallet className="h-5 w-5 text-[var(--color-accent)]" />
            </div>
            <span className="text-sm font-medium text-[var(--color-text)]">Spent</span>
          </div>
          {loading ? (
            <Skeleton className="h-9 w-24" />
          ) : (
            <p className="font-mono text-3xl font-bold text-[var(--color-text-heading)]">
              ${((budget?.spent_cents ?? 0) / 100).toFixed(2)}
            </p>
          )}
        </Card>

        <Card className="p-6">
          <div className="mb-3 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
              <TrendingDown className="h-5 w-5 text-[var(--color-accent)]" />
            </div>
            <span className="text-sm font-medium text-[var(--color-text)]">Remaining</span>
          </div>
          {loading ? (
            <Skeleton className="h-9 w-24" />
          ) : (
            <p className={`font-mono text-3xl font-bold ${remaining < 0 ? 'text-[var(--color-danger)]' : 'text-[var(--color-text-heading)]'}`}>
              ${remaining.toFixed(2)}
            </p>
          )}
        </Card>

        <Card className="p-6">
          <div className="mb-3 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
              <AlertTriangle className="h-5 w-5 text-[var(--color-accent)]" />
            </div>
            <span className="text-sm font-medium text-[var(--color-text)]">Current Tier</span>
          </div>
          {loading ? (
            <Skeleton className="h-9 w-20" />
          ) : (
            <div className="flex items-center gap-2">
              <Badge variant={TIERS[currentTier - 1]?.color ?? 'default'}>
                Tier {currentTier}
              </Badge>
              <span className="text-xs text-[var(--color-text-muted)]">{TIERS[currentTier - 1]?.label}</span>
            </div>
          )}
        </Card>
      </div>

      {/* Budget bar */}
      <Card className="mb-6 p-6">
        <div className="mb-2 flex items-center justify-between text-sm">
          <span className="font-medium text-[var(--color-text)]">Progress</span>
          <span className="font-mono text-[var(--color-text-muted)]">{pct.toFixed(1)}%</span>
        </div>
        <div className="h-3 overflow-hidden rounded-full bg-[var(--color-bg-subtle)]">
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${Math.min(pct, 100)}%`,
              backgroundColor: pct >= 100 ? 'var(--color-danger)' : pct >= 90 ? 'var(--color-warning)' : 'var(--color-primary)',
            }}
          />
        </div>
        {/* Tier markers */}
        <div className="mt-4 grid grid-cols-4 gap-2">
          {TIERS.map((t) => (
            <div key={t.tier} className={`flex flex-col rounded-lg border p-2 ${currentTier === t.tier ? 'border-[var(--color-border-strong)] bg-[var(--color-surface-highlight)]' : 'border-[var(--color-border)]'}`}>
              <span className="text-xs font-medium text-[var(--color-text-heading)]">Tier {t.tier}: {t.label}</span>
              <span className="text-xs text-[var(--color-text-muted)]">{t.description}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* Set limit */}
      <Card className="mb-6 p-6">
        <p className="mb-3 text-sm font-medium text-[var(--color-text-heading)]">Set limit</p>
        <div className="flex gap-2 max-w-sm">
          <Input
            type="number"
            step="0.01"
            placeholder="Amount in USD"
            value={limitInput}
            onChange={(e) => setLimitInput(e.target.value)}
            className="flex-1"
          />
          <Button onClick={handleSetLimit} disabled={!limitInput}>Set</Button>
        </div>
      </Card>

      {/* History */}
      <h2 className="mb-3 text-lg font-semibold text-[var(--color-text-heading)]">History</h2>
      {loading ? (
        <Skeleton className="h-32 w-full rounded-xl" />
      ) : history?.history && history.history.length > 0 ? (
        <div className="flex flex-col gap-2">
          {history.history.map((h, i) => (
            <Card key={i} className="flex items-center justify-between p-3">
              <div className="flex flex-col">
                <span className="text-sm text-[var(--color-text-heading)]">{h.event}</span>
                <span className="text-xs text-[var(--color-text-muted)]">{h.timestamp}</span>
              </div>
              <div className="flex items-center gap-3 text-xs text-[var(--color-text-muted)]">
                {h.limit_cents != null && <span>Limit: ${(h.limit_cents / 100).toFixed(2)}</span>}
                {h.spent_cents != null && <span>Spent: ${(h.spent_cents / 100).toFixed(2)}</span>}
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <EmptyState title="No history" description="Budget events will appear here as sessions accrue spend." />
      )}
    </div>
  )
}