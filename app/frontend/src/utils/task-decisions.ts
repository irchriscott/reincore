type RecordValue = Record<string, unknown>
export const objectValue = (value: unknown): RecordValue => value !== null && typeof value === 'object' && !Array.isArray(value) ? value as RecordValue : {}
export const decisionText = (value: unknown): string => typeof value === 'string' ? value : ''
export const decisionLabel = (value: unknown): string => decisionText(value).replaceAll('_', ' ')
export const decisionEntries = (payload: RecordValue): RecordValue[] => Array.isArray(payload.entries)
  ? payload.entries.filter((entry): entry is RecordValue => entry !== null && typeof entry === 'object' && !Array.isArray(entry)) : []

export function taskDecisionSummary(payload: RecordValue): string {
  const entries = decisionEntries(payload)
  const supplied = decisionText(payload.summary)
  if (supplied) return supplied
  if (!entries.length) return 'No decision details were saved in this event.'
  const counts = new Map<string, number>()
  for (const entry of entries) {
    const kind = decisionLabel(entry.kind) || 'decision'
    counts.set(kind, (counts.get(kind) ?? 0) + 1)
  }
  return [...counts].map(([kind, count]) => `${count} ${kind}`).join(', ')
}

export function taskDecisionPreview(payload: RecordValue): string {
  return decisionEntries(payload).slice(0, 3).map(entry => {
    const status = decisionLabel(entry.status)
    return `${decisionLabel(entry.kind) || 'Decision'}${status ? ` (${status})` : ''}: ${decisionText(entry.value) || decisionText(entry.question) || decisionText(entry.key)}`
  }).join(' · ') || decisionText(payload.reason) || taskDecisionSummary(payload)
}
