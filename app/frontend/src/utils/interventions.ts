// Read-side helpers for `loop_intervention` audit events.
//
// The engine writes one payload shape (`LoopInterventionPayload` in
// engine-core/src/event.rs) for three situations that must not look alike in
// the timeline:
//
//   * a repair nudge injected into the next request,
//   * a hard stop that refuses the next turn instead of billing it,
//   * a tombstone written by `CircuitBreaker::clear_intervention` once a
//     pending nudge has been consumed (empty message, zero repetitions).
//
// Everything is read defensively from the raw payload: `intervention`,
// `hard_stop`, and `retryable` were all added after the event type shipped, so
// older sessions still render with whatever fields they do carry.

type RecordValue = Record<string, unknown>

const objectValue = (value: unknown): RecordValue =>
  value !== null && typeof value === 'object' && !Array.isArray(value) ? value as RecordValue : {}
const textValue = (value: unknown): string | undefined =>
  typeof value === 'string' && value !== '' ? value : undefined
const countValue = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined

/** Sentence-case a snake_case enum variant that has no explicit label. */
const humanize = (value: string): string => {
  const spaced = value.replaceAll('_', ' ')
  return spaced.charAt(0).toUpperCase() + spaced.slice(1)
}

const labelFrom = (labels: Record<string, string>) => (value: unknown): string | undefined => {
  const text = textValue(value)
  if (!text) return undefined
  return labels[text] ?? humanize(text)
}

/** `detector_name` in engine-core/src/intervention.rs. */
export const detectorLabel = labelFrom({
  loop_detection: 'Loop detection',
  prompt_injection: 'Prompt injection',
  claim_grounding: 'Claim grounding',
  repeated_questions: 'Repeated questions',
  action_validation: 'Action validation',
  authorization: 'Authorization',
  action_recovery: 'Action recovery',
})

/** `HeuristicMode` — the operator policy that selected the response. */
export const heuristicModeLabel = labelFrom({
  observe: 'Observe',
  warn: 'Warn',
  repair_once: 'Repair once',
  block: 'Block',
  pause: 'Pause',
})

/** `InterventionAction` — the response actually taken. */
export const interventionActionLabel = labelFrom({
  observed: 'Observed',
  warned: 'Warned',
  repair_requested: 'Repair requested',
  action_blocked: 'Action blocked',
  paused: 'Paused',
})

/** `InterventionOutcome` as written on this event. Later resolutions arrive on
 *  a separate `agent_health` event, so this is never retroactively updated. */
export const interventionOutcomeLabel = labelFrom({
  pending: 'Pending',
  not_applicable: 'Not applicable',
  recovered: 'Recovered',
  failed: 'Failed',
  false_alarm: 'False alarm',
})

/** `AgentHealthKind` — the detection that produced the evidence. */
export const healthKindLabel = labelFrom({
  tool_loop: 'Tool loop',
  redundant_lookup: 'Redundant lookup',
  unknown_tool: 'Unknown tool',
  malformed_tool_arguments: 'Malformed tool arguments',
  duplicate_tool_call: 'Duplicate tool call',
  prompt_injection: 'Prompt injection',
  ungrounded_citation: 'Ungrounded citation',
  action_denied: 'Action denied',
  authorization_denied: 'Authorization denied',
  recovery_status: 'Recovery status',
  repeated_clarification: 'Repeated clarification',
  unverified_completion: 'Unverified completion',
  contradicted_claim: 'Contradicted claim',
  partial_completion: 'Partial completion',
})

export const healthSeverityLabel = labelFrom({ warning: 'Warning', critical: 'Critical' })

export interface InterventionEvidenceView {
  kind?: string
  severity?: string
  toolName?: string
}

export interface InterventionRecordView {
  id?: string
  detector?: string
  configuredMode?: string
  action?: string
  outcome?: string
  recordedAt?: string
  evidence: InterventionEvidenceView
}

export interface LoopInterventionView {
  /** Tombstone marking a consumed nudge. Carries no evidence by design. */
  cleared: boolean
  /** The next turn is refused rather than billed. */
  hardStop: boolean
  /** Whether a client may retry after a refusal. Defaults to the historical `true`. */
  retryable: boolean
  repetitionCount?: number
  repeatedHash?: string
  injectedMessage?: string
  /** Evidence text, but only when it says more than the injected message.
   *  `CircuitBreaker::intervention_record` builds it as the injected message
   *  plus the repetition count, both of which are already shown on their own. */
  evidenceMessage?: string
  record?: InterventionRecordView
}

/** Project a `loop_intervention` payload into everything the UI renders. */
export function readLoopIntervention(payload: RecordValue): LoopInterventionView {
  const injectedMessage = textValue(payload.injected_message)?.trim() || undefined
  const repetitionCount = countValue(payload.repetition_count)
  const raw = objectValue(payload.intervention)
  const hasRecord = Object.keys(raw).length > 0
  const evidence = objectValue(raw.evidence)
  const evidenceMessage = textValue(evidence.message)
  return {
    // Mirrors `is_effective_loop_intervention` in engine-core/src/cost_outcome.rs:
    // an event counts only when it repeated something or injected something.
    cleared: !injectedMessage && (repetitionCount ?? 0) === 0,
    hardStop: payload.hard_stop === true,
    retryable: payload.retryable !== false,
    repetitionCount,
    repeatedHash: textValue(payload.repeated_hash),
    injectedMessage,
    evidenceMessage: evidenceMessage && injectedMessage && evidenceMessage.includes(injectedMessage)
      ? undefined
      : evidenceMessage,
    record: hasRecord ? {
      id: textValue(raw.id),
      detector: textValue(raw.detector),
      configuredMode: textValue(raw.configured_mode),
      action: textValue(raw.action),
      outcome: textValue(raw.outcome),
      recordedAt: textValue(raw.recorded_at),
      evidence: {
        kind: textValue(evidence.kind),
        severity: textValue(evidence.severity),
        toolName: textValue(evidence.tool_name),
      },
    } : undefined,
  }
}

/**
 * Headline for the timeline row. The badge rendered beside it already says
 * "Loop intervention", so this carries only what the badge does not: which
 * response was taken, and how deep the loop had gone.
 */
export function loopInterventionTitle(view: LoopInterventionView): string {
  if (view.cleared) return 'Pending nudge cleared'
  const repeats = view.repetitionCount
  const suffix = repeats !== undefined && repeats > 0
    ? ` · ${repeats} repeat${repeats === 1 ? '' : 's'}`
    : ''
  return `${view.hardStop ? 'Turn refused' : 'Repair nudge'}${suffix}`
}

/** Two-line summary under the timeline row. */
export function loopInterventionPreview(view: LoopInterventionView): string | undefined {
  if (view.cleared) {
    return 'The nudge was consumed. The next turn may proceed without re-injecting it.'
  }
  const parts: string[] = []
  if (view.injectedMessage) parts.push(view.injectedMessage)
  if (view.evidenceMessage) parts.push(view.evidenceMessage)
  if (view.hardStop) parts.push(view.retryable ? 'Retry allowed' : 'Retry not allowed')
  return parts.length > 0 ? parts.join(' · ') : undefined
}
