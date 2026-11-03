import { AlertTriangle, Ban, OctagonX, Repeat, ShieldCheck } from 'lucide-react'
import { MarkdownContent } from './MarkdownContent'
import {
  detectorLabel,
  healthKindLabel,
  healthSeverityLabel,
  interventionActionLabel,
  type LoopInterventionView,
} from '../utils/interventions'

/** What the circuit breaker did, in the operator's language rather than the engine's. */
function explanation(view: LoopInterventionView): string {
  if (view.cleared) {
    return 'The circuit breaker consumed the pending intervention so the agent may take another turn. A clearing event carries no evidence by design; the detection it resolves is the preceding loop intervention.'
  }
  if (view.hardStop) {
    return view.retryable
      ? 'Nudging had already failed, so the next turn was refused instead of billed. An explicit retry is still allowed: if the agent is still stuck the next inspection re-arms the stop, so each retry costs at most one call.'
      : 'Nudging had already failed, so the next turn was refused instead of billed. This refusal is marked non-retryable, so a client must not retry it automatically.'
  }
  return 'The repeated step was detected before the next provider call. The message below was injected into that call as a bounded correction attempt; the turn itself was not withheld.'
}

export function LoopInterventionDetails({ view }: { view: LoopInterventionView }) {
  const detector = detectorLabel(view.record?.detector) ?? 'Loop detection'
  const action = interventionActionLabel(view.record?.action)
  const evidence = view.record?.evidence
  const critical = evidence?.severity === 'critical'
  const Icon = view.cleared ? ShieldCheck : view.hardStop ? OctagonX : Repeat

  return (
    <section aria-label="Loop intervention details" className="space-y-3 text-[var(--color-text)]">
      <div>
        <h3 className="flex items-center gap-2 text-sm font-semibold">
          <Icon className="h-4 w-4 shrink-0 text-[var(--color-text-muted)]" />
          <span>{view.cleared ? 'Loop guard cleared' : `${detector}${action ? ` · ${action}` : ''}`}</span>
        </h3>
        <p className="mt-1 text-sm text-[var(--color-text-muted)]">{explanation(view)}</p>
      </div>

      {evidence && (evidence.kind || evidence.severity || evidence.toolName || view.evidenceMessage) && (
        <article className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-subtle)] p-4">
          <div className="flex flex-wrap items-center gap-2 text-xs font-medium">
            <span className="inline-flex items-center gap-1.5">
              <AlertTriangle className={`h-3.5 w-3.5 ${critical ? 'text-[var(--color-danger)]' : 'text-[var(--color-warning)]'}`} />
              {healthKindLabel(evidence.kind) ?? 'Detection'}
            </span>
            {evidence.severity && (
              <span className={`rounded-full border px-2 py-0.5 ${critical ? 'border-[var(--color-danger)]/40 text-[var(--color-danger)]' : 'border-[var(--color-border)] text-[var(--color-text-muted)]'}`}>
                {healthSeverityLabel(evidence.severity)}
              </span>
            )}
            {evidence.toolName && (
              <span className="rounded-full border border-[var(--color-border)] px-2 py-0.5 font-mono text-[10px] text-[var(--color-text-muted)]">
                {evidence.toolName}
              </span>
            )}
          </div>
          {view.evidenceMessage && <MarkdownContent className="mt-3" content={view.evidenceMessage} />}
        </article>
      )}

      {view.hardStop && !view.retryable && (
        <p className="flex items-start gap-2 text-xs text-[var(--color-danger)]">
          <Ban className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>Marked non-retryable. A client that retries this turn anyway is billed for it.</span>
        </p>
      )}

      {!view.cleared && !view.record && (
        <p className="text-sm text-[var(--color-text-muted)]">
          This event predates structured intervention records, so no detector, policy, or evidence was saved with it.
          The repetition count and injected message are everything the engine recorded at the time.
        </p>
      )}

      {!view.cleared && (
        <p className="text-xs text-[var(--color-text-muted)]">
          The outcome shown above is the one written when the loop was detected. Recoveries and false alarms are
          recorded only when a validator or operator classifies them, and they land on a later agent-health event
          carrying the same intervention ID rather than rewriting this one.
        </p>
      )}
    </section>
  )
}
