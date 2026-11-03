// Projects page: list local projects, view their working directories, and
// rename them. This is the desktop edition's project registry, not the cloud
// workspace-scoped projects. Each card links to the project detail page.

import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { FolderKanban, FolderOpen, Pencil, Save, X, Search, DollarSign, Download, Upload, Activity, ChevronRight } from 'lucide-react'
import { Button, Card, EmptyState, Input, Badge } from '../components/ui'
import { listLocalProjects, renameLocalProject, listProjects, extractErrorMessage } from '../api'
import type { LocalProject, ProjectSummary } from '../types'

/** Format token count compactly (e.g. 12.4k, 1.2M). */
function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}

export function ProjectsPage() {
  const navigate = useNavigate()
  const [projects, setProjects] = useState<LocalProject[]>([])
  const [usage, setUsage] = useState<ProjectSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editName, setEditName] = useState('')
  const [saving, setSaving] = useState(false)

  const loadProjects = async () => {
    setLoading(true)
    setError(null)
    try {
      const [localData, usageData] = await Promise.allSettled([
        listLocalProjects(),
        listProjects(),
      ])
      if (localData.status === 'fulfilled') setProjects(localData.value)
      if (usageData.status === 'fulfilled') setUsage(usageData.value)
      if (localData.status === 'rejected') setError(extractErrorMessage(localData.reason))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadProjects()
  }, [])

  // Build a lookup from project name to usage stats.
  const usageByName = useMemo(() => {
    const map = new Map<string, ProjectSummary>()
    for (const u of usage) map.set(u.project_name, u)
    return map
  }, [usage])

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return projects
    return projects.filter(
      (p) =>
        p.name.toLowerCase().includes(q) ||
        p.directories.some((d) => d.toLowerCase().includes(q))
    )
  }, [projects, search])

  const startEdit = (project: LocalProject) => {
    setEditingId(project.id)
    setEditName(project.name)
  }

  const cancelEdit = () => {
    setEditingId(null)
    setEditName('')
  }

  const submitRename = async (id: string) => {
    const name = editName.trim()
    if (!name) return
    setSaving(true)
    try {
      const updated = await renameLocalProject(id, name)
      setProjects((prev) =>
        prev
          .map((p) => (p.id === id ? updated : p))
          .sort((a, b) => a.name.localeCompare(b.name))
      )
      setEditingId(null)
    } catch (err) {
      setError(extractErrorMessage(err))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="mx-auto w-full max-w-[1000px]">
      <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="heading-display text-3xl font-semibold text-[var(--color-text-heading)]">
            Projects
          </h1>
          <p className="text-[var(--color-text-muted)]">
            Local projects and the directories assigned to them.
          </p>
        </div>
        <Button variant="secondary" loading={loading} onClick={loadProjects}>
          Refresh
        </Button>
      </div>

      {error && (
        <div className="mb-6 rounded-lg border border-[var(--color-danger)] bg-[rgba(244,63,94,0.08)] px-4 py-3 text-sm text-[var(--color-danger)]">
          {error}
        </div>
      )}

      <div className="mb-6">
        <div className="relative">
          <Search className="absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--color-text-muted)]" />
          <Input
            className="pl-10"
            placeholder="Search by project name or path..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      {loading && projects.length === 0 ? (
        <div className="space-y-2">
          {[...Array(4)].map((_, i) => (
            <Card key={i} className="h-[88px] animate-pulse bg-[var(--color-surface-highlight)]">
              <div />
            </Card>
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <EmptyState
          icon={<FolderKanban className="h-10 w-10" />}
          title={search ? 'No matching projects' : 'No projects yet'}
          description={
            search
              ? 'Try a different search term.'
              : 'Start a ReinCore session in a directory and it will appear here automatically.'
          }
        />
      ) : (
        <div className="space-y-2">
          {filtered.map((project) => {
            const stats = usageByName.get(project.name)
            const sessionCount = stats?.session_count ?? 0
            const activeCount = stats?.active_session_count ?? 0
            const spendCents = stats?.total_spend_cents ?? 0
            const inputTokens = stats?.total_input_tokens ?? 0
            const outputTokens = stats?.total_output_tokens ?? 0
            return (
              <Card
                key={project.id}
                className="group cursor-pointer p-3.5 transition-all hover:border-[var(--color-border-strong)] hover:bg-[var(--color-surface-highlight)]"
                onClick={() => navigate(`/projects/${project.id}`)}
              >
                <div className="flex items-start gap-3">
                  {/* Accent bar */}
                  <div className="mt-0.5 h-10 w-1 shrink-0 rounded-full bg-[var(--color-accent)]" />
                  {/* Project icon */}
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
                    <FolderKanban className="h-4.5 w-4.5 text-[var(--color-accent)]" />
                  </div>

                  {/* Main content */}
                  <div className="flex min-w-0 flex-1 flex-col gap-1.5">
                    {/* Name + edit */}
                    <div className="flex items-center gap-2">
                      {editingId === project.id ? (
                        <div className="flex items-center gap-2">
                          <Input
                            value={editName}
                            onChange={(e) => setEditName(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') submitRename(project.id)
                              if (e.key === 'Escape') cancelEdit()
                            }}
                            disabled={saving}
                            className="max-w-sm"
                          />
                          <Button
                            variant="primary"
                            loading={saving}
                            onClick={() => submitRename(project.id)}
                            title="Save"
                          >
                            <Save className="h-4 w-4" />
                          </Button>
                          <Button variant="secondary" onClick={cancelEdit} disabled={saving} title="Cancel">
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      ) : (
                        <>
                          <h3 className="truncate text-sm font-semibold text-[var(--color-text-heading)]">
                            {project.name}
                          </h3>
                          <button
                            onClick={() => startEdit(project)}
                            className="rounded p-1 text-[var(--color-text-muted)] hover:bg-[var(--color-surface-highlight)] hover:text-[var(--color-text-heading)]"
                            title="Rename project"
                          >
                            <Pencil className="h-3.5 w-3.5" />
                          </button>
                        </>
                      )}
                    </div>

                    {/* Directories */}
                    <div className="space-y-0.5">
                      {project.directories.length === 0 ? (
                        <p className="text-xs text-[var(--color-text-muted)]">No directories assigned.</p>
                      ) : (
                        project.directories.map((dir) => (
                          <div
                            key={dir}
                            className="flex items-center gap-1.5 text-xs text-[var(--color-text-muted)]"
                          >
                            <FolderOpen className="h-3.5 w-3.5 shrink-0" />
                            <span className="truncate" title={dir}>
                              {dir}
                            </span>
                          </div>
                        ))
                      )}
                    </div>
                  </div>

                  {/* Right-side stats */}
                  <div className="flex shrink-0 flex-col items-end gap-1.5">
                    <div className="flex items-center gap-2">
                      <Badge variant="primary">
                        <Activity className="mr-1 h-3 w-3" />
                        {sessionCount} {sessionCount === 1 ? 'session' : 'sessions'}
                      </Badge>
                      {activeCount > 0 && (
                        <Badge variant="success">{activeCount} active</Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-3 text-xs text-[var(--color-text-muted)]">
                      <span className="flex items-center gap-1" title="Total spend">
                        <DollarSign className="h-3 w-3" />
                        {(spendCents / 100).toFixed(2)}
                      </span>
                      <span className="flex items-center gap-1" title="Input tokens" style={{ color: 'var(--color-info)' }}>
                        <Download className="h-3 w-3" />
                        {fmtTokens(inputTokens)}
                      </span>
                      <span className="flex items-center gap-1" title="Output tokens" style={{ color: 'var(--color-success)' }}>
                        <Upload className="h-3 w-3" />
                        {fmtTokens(outputTokens)}
                      </span>
                    </div>
                  </div>
                  {/* Chevron affordance to the detail page */}
                  <ChevronRight className="h-4 w-4 shrink-0 text-[var(--color-text-muted)] transition-all group-hover:translate-x-0.5 group-hover:text-[var(--color-text-heading)]" />
                </div>
              </Card>
            )
          })}
        </div>
      )}
    </div>
  )
}
