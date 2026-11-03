import type { CostOutcome } from '../types'
import { Badge } from './ui'

export function OutcomeLabel({ outcome, appearance = 'badge' }: { outcome?: CostOutcome | null; appearance?: 'badge' | 'stat' }) {
  const labels = {
    unknown: 'Success unknown',
    validated_success: 'Validated success',
    user_reported_success: 'User-reported success',
    recovered_success: 'Recovered success',
    reported_failure: 'Reported failure',
  }
  const Container = appearance === 'stat' ? 'span' : Badge
  const className = appearance === 'stat'
    ? 'inline-flex items-center gap-1.5 rounded-lg bg-[var(--color-bg-subtle)] px-3 py-1.5 text-xs font-medium text-[var(--color-text-secondary)]'
    : undefined
  return <Container className={className} title={outcome?.outcome_report ? outcome.outcome_report.source + ': ' + outcome.outcome_report.evidence : 'No success or failure assessment has been recorded. Completion alone does not establish correctness.'}>
    {labels[outcome?.success_status ?? 'unknown']}
    {outcome?.completed_after_intervention && outcome.success_status !== 'recovered_success' ? ' · completed after intervention' : ''}
  </Container>
}
