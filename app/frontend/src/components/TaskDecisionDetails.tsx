import { decisionEntries, decisionLabel, decisionText, objectValue, taskDecisionSummary } from '../utils/task-decisions'
import { MarkdownContent } from './MarkdownContent'

export function TaskDecisionDetails({ payload }: { payload: Record<string, unknown> }) {
  const entries = decisionEntries(payload)
  const field = (label: string, value: unknown) => {
    const text = decisionText(value)
    return text ? <div><dt className="text-xs text-[var(--color-text-muted)]">{label}</dt><dd className="mt-1 whitespace-pre-wrap break-words text-sm">{text}</dd></div> : null
  }
  return <section aria-label="Task decision details" className="space-y-3 text-[var(--color-text)]">
    <div>
      <h3 className="text-sm font-semibold">Task decisions · {taskDecisionSummary(payload)}</h3>
      {decisionText(payload.reason) && <MarkdownContent className="mt-1" content={decisionText(payload.reason)} />}
      <p className="mt-1 text-xs text-[var(--color-text-muted)]">Statuses below are recorded at the time of this event. Later events may resolve, replace, or revoke them. Permission records do not authorize tool execution.</p>
    </div>
    <dl className="grid gap-3 sm:grid-cols-2">
      {field('Recorded by', decisionLabel(payload.operation))}
      {field('Task', payload.task_id)}
      {field('Request group', payload.request_group_id)}
      {field('Previous session', payload.previous_session_id)}
      {field('Completion accounting ID', payload.completion_accounting_id)}
    </dl>
    {entries.map((entry, index) => {
      const source = objectValue(entry.source)
      const expires = decisionText(entry.expires_at)
      return <article key={`${decisionText(entry.id)}-${index}`} className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-subtle)] p-4">
        <div className="flex flex-wrap items-center gap-2 text-xs font-medium">
          <span className="capitalize">{decisionLabel(entry.kind) || 'Decision'}</span>
          <span className="rounded-full border border-[var(--color-border)] px-2 py-0.5 capitalize">{decisionLabel(entry.status) || 'Status not recorded'}</span>
          {entry.group_only === true && <span>Request group only</span>}
          {source.truncated === true && <span className="text-[var(--color-warning)]">Evidence truncated</span>}
        </div>
        <MarkdownContent className="mt-3" content={decisionText(entry.value) || 'No decision text was saved.'} />
        <dl className="mt-3 grid gap-3 sm:grid-cols-2">
          {field('Subject', entry.key)}
          {field('Question', entry.question)}
          {field('Scope', entry.scope)}
          {field('Source', decisionLabel(source.role))}
          {field('Recorded at', entry.recorded_at)}
          {field('Expires at', expires)}
          {field('Original request group', entry.request_group_id)}
          {field('Tool call ID', source.tool_call_id)}
        </dl>
        {decisionText(source.quote) && <div className="mt-3"><p className="text-xs text-[var(--color-text-muted)]">Source evidence</p><blockquote className="mt-1 border-l-2 border-[var(--color-border)] pl-3"><MarkdownContent content={decisionText(source.quote)} /></blockquote></div>}
        <details className="mt-3 text-xs"><summary className="cursor-pointer text-[var(--color-text-muted)]">Decision references</summary><dl className="mt-2 grid gap-2">
          {field('Decision ID', entry.id)}
          {field('Source message hash', source.message_hash)}
          {field('Supersedes', Array.isArray(entry.supersedes) ? entry.supersedes.filter((value): value is string => typeof value === 'string').join(', ') : '')}
        </dl></details>
      </article>
    })}
    {!entries.length && <p className="text-sm">No decision entries are available. The raw audit payload below may contain details from an older engine version.</p>}
  </section>
}
