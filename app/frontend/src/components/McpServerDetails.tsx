import { ChevronDown, Pencil, Plug, Trash2 } from 'lucide-react'
import type { LocalProject } from '../types'
import type { Server } from '../utils/mcp-server-form'
import { describeServerAuth } from '../utils/mcp-auth'
import { Badge, Button } from './ui'

export function McpServerDetails({ server, projects, busy, signedIn, onEdit, onDelete }: {
  server: Server
  // Whether a usable OAuth sign-in is stored for the server.
  signedIn: boolean
  projects: LocalProject[]
  busy: boolean
  onEdit: () => void
  onDelete: () => void
}) {
  const projectIds = server.access.project_ids ?? (server.access.project_id ? [server.access.project_id] : [])
  const tools = Object.entries(server.tools)
  const readOnly = tools.filter(([, policy]) => policy.read_only).length

  return (
    <article className="overflow-hidden rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-subtle)]">
      <div className="flex flex-wrap items-start justify-between gap-3 p-4">
        <div className="flex min-w-0 flex-1 items-start gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
            <Plug aria-hidden="true" className="h-5 w-5 text-[var(--color-accent)]" />
          </div>
          <div className="min-w-0">
            <h4 className="break-all text-sm font-semibold text-[var(--color-text-heading)]">{server.name}</h4>
            <p className="mt-1 break-all font-mono text-xs text-[var(--color-text-muted)]">{server.url}</p>
          </div>
        </div>
        <div className="flex shrink-0 gap-1">
          <Button type="button" variant="secondary" disabled={busy} onClick={onEdit} aria-label={`Edit ${server.name}`}>
            <Pencil aria-hidden="true" className="h-4 w-4" /> Edit
          </Button>
          <Button type="button" variant="ghost" disabled={busy} onClick={onDelete} aria-label={`Delete ${server.name}`} title="Delete server">
            <Trash2 aria-hidden="true" className="h-4 w-4 text-[var(--color-danger)]" />
          </Button>
        </div>
      </div>
      <dl className="grid gap-4 border-t border-[var(--color-border)] p-4 text-xs sm:grid-cols-2">
        <div className="min-w-0 sm:col-span-2">
          <dt className="mb-2 font-medium text-[var(--color-text-muted)]">{projectIds.length ? `Projects (${projectIds.length})` : 'Local directory'}</dt>
          <dd className="flex max-h-32 flex-wrap gap-2 overflow-y-auto">
            {projectIds.length ? projectIds.map(id => (
              <span key={id} title={id} className="max-w-full break-all rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] px-2 py-1">
                {projects.find(project => project.id === id)?.name ?? id}
              </span>
            )) : <span className="break-all font-mono">{server.access.project_path || 'Not configured'}</span>}
          </dd>
        </div>
        <div>
          <dt className="mb-1 text-[var(--color-text-muted)]">Authentication</dt>
          <dd className="break-words">
            {describeServerAuth(server, signedIn)}
            {server.headers && Object.keys(server.headers).length > 0 && (
              <span className="block text-[var(--color-text-muted)]">Extra headers: {Object.keys(server.headers).join(', ')}</span>
            )}
          </dd>
        </div>
        <div>
          <dt className="mb-1 text-[var(--color-text-muted)]">Request timeout</dt>
          <dd>{server.timeout_ms / 1000} seconds</dd>
        </div>
      </dl>
      <details className="group border-t border-[var(--color-border)]">
        <summary className="flex cursor-pointer list-none flex-wrap items-center gap-2 px-4 py-3 text-xs [&::-webkit-details-marker]:hidden">
          <ChevronDown aria-hidden="true" className="h-4 w-4 transition-transform group-open:rotate-180" />
          <span className="font-medium">Configured tools ({tools.length})</span>
          <span className="text-[var(--color-text-muted)]">{readOnly} read-only · {tools.length - readOnly} may modify data</span>
        </summary>
        <ul className="max-h-64 overflow-y-auto border-t border-[var(--color-border)] px-4 divide-y divide-[var(--color-border)]">
          {tools.map(([name, policy]) => <li key={name} className="flex flex-wrap items-center justify-between gap-2 py-2.5 text-xs">
            <span className="min-w-0 break-all font-mono">{name}</span>
            <span className="flex shrink-0 gap-2">
              <Badge>{policy.read_only ? 'Read-only' : 'May modify data'}</Badge>
              {policy.cache != null && <Badge>Cache configured</Badge>}
            </span>
          </li>)}
          {tools.length === 0 && <li className="py-3 text-xs text-[var(--color-text-muted)]">No tools configured.</li>}
        </ul>
      </details>
    </article>
  )
}
