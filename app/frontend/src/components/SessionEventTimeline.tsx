import {
  AlertTriangle,
  Bot,
  Brain,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Code2,
  FileText,
  Filter,
  GitBranch,
  HeartPulse,
  Loader2,
  OctagonX,
  Repeat,
  RotateCcw,
  Sparkles,
  TrendingDown,
  User,
  Wrench,
  X,
  Zap,
} from 'lucide-react'
import { useState, useMemo, memo, useEffect, useLayoutEffect, useCallback, useRef } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import { createPortal } from 'react-dom'
import { useActivityScroll } from '../hooks/useActivityScroll'
import { Badge, Card } from './ui'
import type { RequestGroupTimelineSummary, SessionEvent, SessionSummary, SessionTask, SessionTimelineSummary, TaskType } from '../types'
import { getSessionEvent, getSessionEventPage, getSessionEvents } from '../api'
import { capitalizeStatus } from '../utils/status'
import { MarkdownContent } from './MarkdownContent'
import { parseRequestContent, requestMessage } from '../utils/timelineText'
import { TaggedFiles } from './TaggedFiles'
import { TaskDecisionDetails } from './TaskDecisionDetails'
import { LoopInterventionDetails } from './LoopInterventionDetails'
import { taskDecisionPreview, taskDecisionSummary } from '../utils/task-decisions'
import {
  detectorLabel,
  healthKindLabel,
  healthSeverityLabel,
  heuristicModeLabel,
  interventionActionLabel,
  interventionOutcomeLabel,
  loopInterventionPreview,
  loopInterventionTitle,
  readLoopIntervention,
} from '../utils/interventions'
import type { SessionStreamState } from '../hooks/useSessionEventStream'

type JsonObject = Record<string, unknown>
export type ProcessKind =
  | 'reasoning'
  | 'function_call'
  | 'model_response'
  | 'model_call'
  | 'tool_call'
  | 'tool_result'
  | 'tool_error'
  | 'agent_health'
  | 'agent_health_stop'
  | 'task_decision'
  | 'prewarm'
  | 'compaction'
  | 'handoff'
  | 'tier_change'
  | 'loop_intervention'
  | 'loop_hard_stop'
  | 'subagent'
  | 'system'

interface ChatMessage extends JsonObject {
  role?: string
  content?: string | null
  tool_calls?: ToolCall[] | null
}

interface ToolCall extends JsonObject {
  id?: string
  type?: string
  function?: { name?: string; arguments?: string }
}

interface RequestGroup {
  id: string
  taskId?: string
  taskName?: string
  taskType?: TaskType
  userPrompt?: string
  externalSummary?: string
  events: SessionEvent[]
}

interface ProcessItem {
  id: string
  event: SessionEvent
  /** The provider event that requested this canonical tool execution. */
  modelEvent?: SessionEvent
  /** Matching response for a tool request, correlated by tool-call id. */
  resultEvent?: SessionEvent
  /** Seq numbers of provider-request and tool-result audit events merged into
   *  this item because they belong to the same canonical tool call. */
  mergedFromSeqs?: number[]
  /** Per-request-group sequence number (1-based within the group). */
  localSeq?: number
  /** Per-request-group seq numbers of related merged audit events. */
  mergedFromLocalSeqs?: number[]
  /** Map from global event seq → per-request-group local seq, used for
   *  translating payload fields like summarized_seq_start/end. */
  localSeqByGlobalSeq?: Map<number, number>
  kind: ProcessKind
  label: string
  title: string
  preview?: string
  content?: string
  toolCall?: ToolCall
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function asObject(value: unknown): JsonObject {
  return isObject(value) ? value : {}
}

function asMessage(value: unknown): ChatMessage {
  return asObject(value) as ChatMessage
}

function asToolCall(value: unknown): ToolCall | null {
  return isObject(value) ? (value as ToolCall) : null
}

function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined
}

function taskTypeValue(value: unknown): TaskType | undefined {
  const taskType = stringValue(value)
  return taskType as TaskType | undefined
}

function taskTypeLabel(value: TaskType): string {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function numberValue(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function piiRedactions(payload: unknown): number {
  const meta = asObject(asObject(payload)._reincore_privacy_redaction)
  const count = numberValue(meta.pii_redaction_count)
  return count !== undefined && count > 0 ? Math.floor(count) : 0
}

function effectiveUserPrompt(content: string): string | null {
  const match = content.match(/<(userRequest|user_request|user_query|prompt)>([\s\S]*?)(?:<\/\1>|$)/i)
  return match?.[2]?.trim() || null
}

const EXTERNAL_SUMMARY_TAGS = [
  'conversation-summary',
  'conversation_summary',
  'context-summary',
]

function isExternalSummaryMessage(message: ChatMessage): boolean {
  if (message.role?.toLowerCase() !== 'user') return false
  const content = message.content
  if (typeof content !== 'string') return false
  const lower = content.toLowerCase()
  return EXTERNAL_SUMMARY_TAGS.some((tag) => lower.includes(`<${tag}`))
}

function isScaffoldingOnly(content: string): boolean {
  const tags = 'environment_info|workspace_info|editorcontext|context|attachments?|instructions|reminderinstructions|usermemory|sessionmemory|repomemory|system-reminder|tool_result|files|selection|diff'
  const containsScaffolding = new RegExp(`<(?:${tags})\\b`, 'i').test(content)
  const withoutScaffolding = content.replace(
    new RegExp(`<(?:${tags})\\b[^>]*>[\\s\\S]*?(?:<\\/(?:${tags})>|$)`, 'gi'),
    '',
  )
  return containsScaffolding && withoutScaffolding.trim().length === 0
}

function latestUserPrompt(payload: JsonObject): { prompt?: string; externalSummary?: string } {
  const messages = Array.isArray(payload.request_messages) ? payload.request_messages : []
  const roles = Array.isArray(payload.message_roles) ? payload.message_roles : []
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = asMessage(messages[index])
    if (message.role?.toLowerCase() !== 'user' || typeof message.content !== 'string') continue
    const roleLabel = typeof roles[index] === 'string' ? (roles[index] as string) : message.role
    if (roleLabel?.includes('external_summary') || isExternalSummaryMessage(message)) {
      return { externalSummary: message.content }
    }
    if (isScaffoldingOnly(message.content)) continue
    const prompt = effectiveUserPrompt(message.content)
    return { prompt: prompt ?? message.content }
  }
  return {}
}

function completesRequestGroup(event: SessionEvent): boolean {
  if (event.event_type !== 'llm_call' || event.payload.batch_submitted === true) return false
  const response = asMessage(event.payload.response_message)
  return isObject(event.payload.response_message)
    && (!Array.isArray(response.tool_calls) || response.tool_calls.length === 0)
}

/**
 * Overlay SSE events that are newer than the durable timeline snapshot. This
 * lets a brand-new task/request render on the first realtime frame instead of
 * waiting for the debounced snapshot refresh. Once the API snapshot catches
 * up, the same sequence range drops out of this overlay without changing the
 * stable task/request keys.
 */
function projectLiveTimeline(
  timeline: SessionTimelineSummary,
  liveEvents: SessionEvent[],
): { timeline: SessionTimelineSummary; activeRequestId?: string; activeTaskId?: string; activeSeq?: number } {
  const pending = liveEvents
    .filter((event) => event.seq > (timeline.latest_seq ?? 0))
    // Durable continuation snapshots are execution state, not user-visible
    // activity; the backend timeline projection excludes them too.
    .filter((event) => event.event_type !== 'continuation')
    .sort((left, right) => left.seq - right.seq)
  if (pending.length === 0) return { timeline }

  const tasks = timeline.tasks.map((task) => ({
    ...task,
    request_group_ids: [...task.request_group_ids],
    request_groups: [...task.request_groups],
  }))
  const requests = timeline.request_groups.map((request) => ({
    ...request,
    models: [...request.models],
  }))
  const taskById = new Map(tasks.map((task) => [task.task_id, task]))
  const requestById = new Map(requests.map((request) => [request.request_group_id, request]))
  let currentRequest = requests.reduce<RequestGroupTimelineSummary | undefined>(
    (latest, request) => !latest || request.last_event_seq > latest.last_event_seq ? request : latest,
    undefined,
  )
  let currentTask = currentRequest ? taskById.get(currentRequest.task_id) : tasks.at(-1)
  let pendingPiiTotal = 0
  let activeSeq: number | undefined

  for (const event of pending) {
    const explicitRequestId = stringValue(event.payload.request_group_id)
    const explicitTaskId = stringValue(event.payload.task_id)
    const requestId = explicitRequestId ?? currentRequest?.request_group_id
    const taskId = explicitTaskId ?? (requestId ? requestById.get(requestId)?.task_id : undefined) ?? currentTask?.task_id
    if (!requestId || !taskId) continue

    let task = taskById.get(taskId)
    if (!task) {
      const prompt = latestUserPrompt(event.payload).prompt
      task = {
        task_id: taskId,
        name: stringValue(event.payload.task_name) ?? prompt ?? 'Untitled task',
        task_type: taskTypeValue(event.payload.task_type) ?? 'general',
        first_prompt: prompt ?? '',
        request_group_ids: [],
        first_event_seq: event.seq,
        last_event_seq: event.seq,
        request_count: 0,
        input_tokens: 0,
        output_tokens: 0,
        cost_cents: 0,
        completed: false,
        clean_completion: false,
        loop_intervention_count: 0,
        request_groups: [],
      }
      tasks.push(task)
      taskById.set(taskId, task)
    }

    let request = requestById.get(requestId)
    if (!request) {
      request = {
        request_group_id: requestId,
        task_id: taskId,
        prompt: latestUserPrompt(event.payload).prompt,
        first_event_seq: event.seq,
        last_event_seq: event.seq,
        event_count: 0,
        llm_call_count: 0,
        tool_call_count: 0,
        tool_error_count: 0,
        pii_redaction_count: 0,
        input_tokens: 0,
        output_tokens: 0,
        cost_cents: 0,
        models: [],
        started_at: event.created_at,
        ended_at: event.created_at,
        completed: false,
        clean_completion: false,
      }
      requests.push(request)
      requestById.set(requestId, request)
      if (!task.request_group_ids.includes(requestId)) {
        task.request_group_ids.push(requestId)
        task.request_count += 1
      }
    }

    request.last_event_seq = Math.max(request.last_event_seq, event.seq)
    request.event_count += 1
    request.ended_at = event.created_at
    const eventPii = piiRedactions(event.payload)
    request.pii_redaction_count = (request.pii_redaction_count ?? 0) + eventPii
    pendingPiiTotal += eventPii
    request.cost_cents += event.cost_cents ?? 0
    task.last_event_seq = Math.max(task.last_event_seq, event.seq)
    task.cost_cents += event.cost_cents ?? 0

    if (event.event_type === 'llm_call') {
      const inputTokens = numberValue(event.payload.input_tokens) ?? 0
      const outputTokens = numberValue(event.payload.output_tokens) ?? 0
      const prompt = latestUserPrompt(event.payload).prompt
      const model = stringValue(event.payload.model)
      request.llm_call_count += 1
      request.input_tokens += inputTokens
      request.output_tokens += outputTokens
      request.prompt ??= prompt
      if (model && !request.models.includes(model)) request.models.push(model)
      request.completed = completesRequestGroup(event)
      request.clean_completion = request.completed && request.tool_error_count === 0
      task.input_tokens += inputTokens
      task.output_tokens += outputTokens
    } else if (event.event_type === 'tool_call') {
      request.tool_call_count += 1
    } else if (event.event_type === 'tool_result') {
      const result = asObject(event.payload.result)
      if (result.is_error === true) request.tool_error_count += 1
    }

    currentRequest = request
    currentTask = task
    activeSeq = event.seq
  }

  return {
    timeline: {
      tasks: tasks.sort((left, right) => left.first_event_seq - right.first_event_seq),
      request_groups: requests.sort((left, right) => left.first_event_seq - right.first_event_seq),
      total_event_count: timeline.total_event_count + pending.length,
      pii_redaction_count: (timeline.pii_redaction_count ?? 0) + pendingPiiTotal,
      latest_seq: Math.max(timeline.latest_seq ?? 0, pending.at(-1)?.seq ?? 0),
    },
    activeRequestId: currentRequest?.request_group_id,
    activeTaskId: currentTask?.task_id,
    activeSeq,
  }
}

/** Project the event log into the user request that initiated each process. */
function buildRequestGroups(events: SessionEvent[]): RequestGroup[] {
  const groups: RequestGroup[] = []
  const byRequestGroupId = new Map<string, RequestGroup>()
  let current: RequestGroup | undefined
  let legacyIndex = 0
  let previousLlmPrompt: string | undefined
  let previousLlmRequestedFunctions = false

  for (const event of [...events].sort((left, right) => left.seq - right.seq)) {
    if (event.event_type === 'continuation') continue
    if (event.event_type !== 'llm_call') {
      const requestGroupId = stringValue(event.payload.request_group_id)
      const target = requestGroupId ? byRequestGroupId.get(requestGroupId) : undefined
      if (target) {
        target.events.push(event)
        continue
      }
    }
    if (event.event_type === 'llm_call') {
      const requestGroupId = stringValue(event.payload.request_group_id)
      const taskId = stringValue(event.payload.task_id)
      const taskName = stringValue(event.payload.task_name)
      const taskType = taskTypeValue(event.payload.task_type)
      const promptResult = latestUserPrompt(event.payload)
      const prompt = promptResult.prompt
      const externalSummary = promptResult.externalSummary
      const response = asMessage(event.payload.response_message)
      const requestedFunctions = Array.isArray(response.tool_calls) && response.tool_calls.length > 0
      const continuesToolLoop = Boolean(current && previousLlmRequestedFunctions && prompt === previousLlmPrompt)

      if (continuesToolLoop && current) {
        if (requestGroupId) byRequestGroupId.set(requestGroupId, current)
      } else if (requestGroupId) {
        current = byRequestGroupId.get(requestGroupId)
        if (!current) {
          current = { id: requestGroupId, taskId, taskName, taskType, userPrompt: prompt, externalSummary, events: [] }
          byRequestGroupId.set(requestGroupId, current)
          groups.push(current)
        } else if (!current.userPrompt && prompt) {
          current.userPrompt = prompt
        } else if (!current.externalSummary && externalSummary) {
          current.externalSummary = externalSummary
        }
      } else if (!current || (prompt && prompt !== current.userPrompt)) {
        legacyIndex += 1
        current = { id: `legacy-${legacyIndex}`, taskId, taskName, taskType, userPrompt: prompt, externalSummary, events: [] }
        groups.push(current)
      }
      previousLlmPrompt = prompt
      previousLlmRequestedFunctions = requestedFunctions
    }
    if (!current) {
      current = { id: 'session-activity', events: [] }
      groups.push(current)
    }
    current.events.push(event)
  }
  return groups
}

function formatTimestamp(value: string): string {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

function formatClock(value: string): string {
  const date = new Date(value)
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function formatDuration(value: number): string {
  if (value < 1_000) return `${Math.round(value)} ms`
  if (value < 60_000) return `${(value / 1_000).toFixed(2)} s`
  return `${(value / 60_000).toFixed(2)} min`
}

function formatTokens(value: number): string {
  return new Intl.NumberFormat().format(value)
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / (1024 * 1024)).toFixed(2)} MB`
}

function formatCost(cents: number): string {
  return `$${(cents / 100).toFixed(2)}`
}

function formatArguments(value: unknown): string {
  if (typeof value !== 'string') return ''
  try {
    return JSON.stringify(JSON.parse(value), null, 2)
  } catch {
    return value
  }
}

/** True when the value parses as a JSON object or array document. */
function isJsonDocument(value: string): boolean {
  const trimmed = value.trim()
  if (!trimmed || !'[{"'.includes(trimmed[0])) return false
  try {
    JSON.parse(trimmed)
    return true
  } catch {
    return false
  }
}

function compact(value: string | undefined, maxLength = 180): string | undefined {
  if (!value) return undefined
  const normalized = value.replace(/\s+/g, ' ').trim()
  return normalized.length > maxLength ? `${normalized.slice(0, maxLength)}...` : normalized
}

function eventTiming(event: SessionEvent): { start: string; end: string; duration?: number } {
  const payload = event.payload
  const explicitStart = stringValue(payload.provider_request_sent_at)
    ?? stringValue(payload.execution_started_at)
    ?? stringValue(payload.requested_at)
  const explicitEnd = stringValue(payload.provider_response_received_at)
    ?? stringValue(payload.execution_completed_at)
  const start = explicitStart ?? event.created_at
  const end = explicitEnd ?? event.created_at
  const directDuration = numberValue(payload.provider_duration_ms) ?? numberValue(payload.execution_duration_ms)
  if (directDuration !== undefined) return { start, end, duration: directDuration }
  if (!explicitStart || !explicitEnd) return { start, end }
  const measured = new Date(end).getTime() - new Date(start).getTime()
  return { start, end, duration: Number.isFinite(measured) && measured >= 0 ? measured : undefined }
}

function tierLabel(tier: number | undefined): string | undefined {
  if (tier === undefined) return undefined
  const names: Record<number, string> = { 1: 'Premium', 2: 'Standard', 3: 'Lite' }
  return names[tier] ?? `Tier ${tier}`
}

function genericLabel(eventType: string): string {
  const labels: Record<string, string> = {
    summary: 'Context summary',
    tier_change: 'Model tier changed',
    checkpoint: 'Checkpoint created',
    checkpoint_redeemed: 'Checkpoint resumed',
    session_rollover: 'Session rollover',
    pre_warm: 'Provider pre-warm',
  }
  return labels[eventType] ?? eventType.replaceAll('_', ' ')
}

function genericPreview(event: SessionEvent): string | undefined {
  return compact(
    stringValue(event.payload.summary_text)
      ?? stringValue(event.payload.reason)
      ?? stringValue(event.payload.message),
  )
}

function eventToProcessItems(event: SessionEvent): ProcessItem[] {
  if (event.event_type === 'llm_call') {
    const response = asMessage(event.payload.response_message)
    const reasoning = stringValue(event.payload.reasoning_content)
    const responseContent = stringValue(response.content)
    const calls = Array.isArray(response.tool_calls)
      ? response.tool_calls.map(asToolCall).filter((call): call is ToolCall => call !== null)
      : []
    const items: ProcessItem[] = []

    // Extract inline compaction from the LlmCall payload's context_adjustment.
    const adjustment = asObject(event.payload.context_adjustment)
    const totalCompacted = numberValue(adjustment.compressed_message_count) ?? 0
    const compactedGroups = numberValue(adjustment.request_group_compacted_message_count) ?? 0
    const compactedActiveTools = numberValue(adjustment.active_tool_compacted_message_count) ?? 0
    const resetMessages = numberValue(adjustment.reset_message_count) ?? 0
    // compressed_message_count is the total net reduction; its component
    // counters must not be added to it again.
    const totalRemoved = totalCompacted
    if (totalRemoved > 0) {
      const originalTokens = numberValue(adjustment.original_token_count)
      const finalTokens = numberValue(adjustment.final_token_count)
      const mode = stringValue(adjustment.compression_mode)
      const parts: string[] = []
      if (totalCompacted > 0) parts.push(`${totalCompacted} total compacted`)
      if (compactedGroups > 0) parts.push(`${compactedGroups} from completed groups`)
      if (compactedActiveTools > 0) parts.push(`${compactedActiveTools} from active tools`)
      if (resetMessages > 0) parts.push(`${resetMessages} reset`)
      if (originalTokens !== undefined && finalTokens !== undefined && originalTokens > 0) {
        const delta = originalTokens - finalTokens
        parts.push(`${formatTokens(originalTokens)} → ${formatTokens(finalTokens)} tokens`)
        if (delta > 0) parts.push(`${formatTokens(delta)} saved (${Math.round(delta / originalTokens * 100)}%)`)
      }
      if (mode) parts.push(`${mode} mode`)
      items.push({
        id: `${event.seq}-compaction`,
        event,
        kind: 'compaction',
        label: 'Context compaction',
        title: 'Context compacted before request',
        preview: compact(parts.join(' · ')),
      })
    }

    if (reasoning) {
      items.push({ id: `${event.seq}-reasoning`, event, kind: 'reasoning', label: 'Reasoning', title: 'Model reasoning', preview: compact(reasoning), content: reasoning })
    }
    if (responseContent) {
      items.push({ id: `${event.seq}-response`, event, kind: 'model_response', label: 'Model response', title: 'Assistant response', preview: compact(responseContent), content: responseContent })
    }
    calls.forEach((call, index) => {
      const name = call.function?.name || 'Unknown function'
      items.push({ id: `${event.seq}-function-${index}`, event, kind: 'function_call', label: 'Function call', title: name, preview: compact(formatArguments(call.function?.arguments)), toolCall: call })
    })
    if (items.length === 0) {
      items.push({ id: `${event.seq}-model`, event, kind: 'model_call', label: 'Model call', title: 'Provider returned an empty message' })
    }
    return items
  }

  if (event.event_type === 'tool_call') {
    const call = asToolCall(event.payload.tool_call)
    const external = event.payload.execution_source === 'external_client'
    return [{ id: `${event.seq}-tool`, event, kind: 'tool_call', label: external ? 'External tool call' : 'Tool call', title: call?.function?.name || 'Unknown tool', preview: compact(formatArguments(call?.function?.arguments)), toolCall: call ?? undefined }]
  }

  if (event.event_type === 'tool_result') {
    const result = asObject(event.payload.result)
    const isError = result.is_error === true
    const external = event.payload.execution_source === 'external_client'
    const content = stringValue(result.content)
    return [{ id: `${event.seq}-result`, event, kind: isError ? 'tool_error' : 'tool_result', label: isError ? 'Tool error' : external ? 'External tool result' : 'Tool result', title: stringValue(result.name) || 'Tool result', preview: compact(content), content }]
  }

  if (event.event_type === 'task_decision') {
    return [{ id: `${event.seq}-decision`, event, kind: 'task_decision', label: 'Task decisions', title: taskDecisionSummary(event.payload), preview: compact(taskDecisionPreview(event.payload)) }]
  }

  if (event.event_type === 'agent_health') {
    const findings = Array.isArray(event.payload.findings) ? event.payload.findings.map(asObject) : []
    const preview = findings.map((finding) => stringValue(finding.message)).filter((message): message is string => Boolean(message)).join(' · ')
    const shouldStop = event.payload.should_stop === true
    return [{ id: `${event.seq}-health`, event, kind: shouldStop ? 'agent_health_stop' : 'agent_health', label: shouldStop ? 'Agent stopped' : 'Agent health', title: shouldStop ? 'Agent stopped' : findings.length ? 'Health findings' : 'Health check passed', preview: compact(preview) }]
  }

  if (event.event_type === 'pre_warm') {
    const providerLabel = stringValue(event.payload.provider_label)
    const warmedModel = stringValue(event.payload.warmed_model)
    const providerWarmed = event.payload.provider_warmed === true
    const preFetched = numberValue(event.payload.pre_fetched_tool_states)
    const shapeComparisons = numberValue(event.payload.shape_comparisons)
    const bestSimilarity = numberValue(event.payload.best_shape_similarity)
    const tier = numberValue(event.payload.warmed_for_tier)
    const parts: string[] = []
    if (providerWarmed && providerLabel) parts.push(`Warmed ${providerLabel}`)
    if (preFetched !== undefined && preFetched > 0) parts.push(`${preFetched} tool state${preFetched === 1 ? '' : 's'} pre-fetched`)
    if (shapeComparisons !== undefined && shapeComparisons > 0) parts.push(`${shapeComparisons} shape comparison${shapeComparisons === 1 ? '' : 's'}`)
    if (bestSimilarity !== undefined && bestSimilarity > 0) parts.push(`${(bestSimilarity * 100).toFixed(0)}% best match`)
    const preview = parts.length > 0 ? parts.join(' · ') : undefined
    const title = providerWarmed
      ? `Pre-warmed ${tierLabel(tier) ?? `tier ${tier ?? '?'}`} provider`
      : `Pre-warm for ${tierLabel(tier) ?? `tier ${tier ?? '?'}`}`
    return [{ id: `${event.seq}-prewarm`, event, kind: 'prewarm', label: 'Pre-warm', title, preview, content: warmedModel ? `Model: ${warmedModel}` : undefined }]
  }

  if (event.event_type === 'summary') {
    const summaryText = stringValue(event.payload.summary_text)
    const seqStart = numberValue(event.payload.summarized_seq_start)
    const seqEnd = numberValue(event.payload.summarized_seq_end)
    const rangePart = seqStart !== undefined && seqEnd !== undefined ? `Events #${seqStart}–#${seqEnd}` : undefined
    const preview = compact(summaryText) ?? rangePart
    return [{ id: `${event.seq}-summary`, event, kind: 'compaction', label: 'Context compaction', title: 'Context compacted', preview, content: summaryText }]
  }

  if (event.event_type === 'session_rollover') {
    const reason = stringValue(event.payload.reason)
    const tokensBefore = numberValue(event.payload.context_tokens_before)
    const tokensAfter = numberValue(event.payload.context_tokens_after)
    const parts: string[] = []
    if (tokensBefore !== undefined) parts.push(`${formatTokens(tokensBefore)} → ${formatTokens(tokensAfter ?? 0)} tokens`)
    if (reason) parts.push(reason)
    return [{ id: `${event.seq}-rollover`, event, kind: 'handoff', label: 'Session handoff', title: 'Session handed off to successor', preview: compact(parts.join(' · ')) }]
  }

  if (event.event_type === 'tier_change') {
    const previousTier = numberValue(event.payload.previous_tier)
    const newTier = numberValue(event.payload.new_tier)
    const spendCents = numberValue(event.payload.spend_cents)
    const budgetCents = numberValue(event.payload.budget_cents)
    const title = `${tierLabel(previousTier) ?? `Tier ${previousTier}`} → ${tierLabel(newTier) ?? `Tier ${newTier}`}`
    const parts: string[] = []
    if (spendCents !== undefined) parts.push(`${formatCost(spendCents)} spent`)
    if (budgetCents !== undefined) parts.push(`${formatCost(budgetCents)} budget`)
    return [{ id: `${event.seq}-tier`, event, kind: 'tier_change', label: 'Tier change', title, preview: compact(parts.join(' · ')) }]
  }

  if (event.event_type === 'loop_intervention') {
    const view = readLoopIntervention(event.payload)
    // A hard stop refuses the next turn, so it reads as a stop rather than a
    // nudge. The tombstone `clear_intervention` writes keeps the nudge kind so
    // filtering for interventions still surfaces the whole lifecycle.
    const kind: ProcessKind = view.hardStop ? 'loop_hard_stop' : 'loop_intervention'
    const label = view.cleared ? 'Loop guard' : view.hardStop ? 'Loop hard stop' : 'Loop intervention'
    return [{
      id: `${event.seq}-loop`,
      event,
      kind,
      label,
      title: loopInterventionTitle(view),
      preview: compact(loopInterventionPreview(view)),
      content: view.injectedMessage,
    }]
  }

  return [{ id: `${event.seq}-system`, event, kind: 'system', label: genericLabel(event.event_type), title: genericLabel(event.event_type), preview: genericPreview(event) }]
}

/**
 * An LLM function request and its canonical `tool_call` event are two audit
 * records for one visible action. Merge them by call ID, retaining the model
 * event for timing/token details. Requests without a matching execution stay
 * visible as function calls because the client never ran them.
 */
function mergeToolLifecycleItems(items: ProcessItem[]): ProcessItem[] {
  const executionsByCallId = new Map<string, ProcessItem>()
  for (const item of items) {
    if (item.kind !== 'tool_call') continue
    const callId = stringValue(item.toolCall?.id)
    if (callId) executionsByCallId.set(callId, item)
  }

  const modelEventByCallId = new Map<string, SessionEvent>()
  const mergedSeqsByCallId = new Map<string, number[]>()
  for (const item of items) {
    if (item.kind !== 'function_call') continue
    const callId = stringValue(item.toolCall?.id)
    if (callId && executionsByCallId.has(callId)) {
      modelEventByCallId.set(callId, item.event)
      // Track the llm_call seq that was merged away
      const seqs = mergedSeqsByCallId.get(callId) ?? []
      seqs.push(item.event.seq)
      mergedSeqsByCallId.set(callId, seqs)
    }
  }

  const resultEventByCallId = new Map<string, SessionEvent>()
  for (const item of items) {
    if (item.kind !== 'tool_result' && item.kind !== 'tool_error') continue
    const result = asObject(item.event.payload.result)
    const callId = stringValue(result.tool_call_id)
    if (callId) resultEventByCallId.set(callId, item.event)
  }

  return items.flatMap((item) => {
    const result = asObject(item.event.payload.result)
    const callId = stringValue(item.toolCall?.id) ?? stringValue(result.tool_call_id)
    if (!callId) return [item]
    if (item.kind === 'function_call' && executionsByCallId.has(callId)) return []
    if ((item.kind === 'tool_result' || item.kind === 'tool_error') && executionsByCallId.has(callId)) return []
    if (item.kind === 'tool_call') {
      const modelEvent = modelEventByCallId.get(callId)
      const resultEvent = resultEventByCallId.get(callId)
      const mergedFromSeqs = [
        ...(mergedSeqsByCallId.get(callId) ?? []),
        ...(resultEvent ? [resultEvent.seq] : []),
      ].sort((a, b) => a - b)
      const resultPayload = asObject(resultEvent?.payload.result)
      const isError = resultPayload.is_error === true
      return [{
        ...item,
        modelEvent,
        resultEvent,
        mergedFromSeqs,
        label: isError ? 'Tool error' : 'Tool call',
        kind: isError ? 'tool_error' : item.kind,
        content: stringValue(resultPayload.content),
      }]
    }
    return [item]
  })
}

function ProcessIcon({ kind }: { kind: ProcessKind }) {
  const className = 'h-4 w-4'
  if (kind === 'reasoning') return <Brain className={className} />
  if (kind === 'function_call') return <Code2 className={className} />
  if (kind === 'model_response' || kind === 'model_call') return <Bot className={className} />
  if (kind === 'tool_call') return <Wrench className={className} />
  if (kind === 'tool_error') return <AlertTriangle className={className} />
  if (kind === 'tool_result') return <CheckCircle2 className={className} />
  if (kind === 'agent_health' || kind === 'agent_health_stop') return <HeartPulse className={className} />
  if (kind === 'task_decision') return <Brain className={className} />
  if (kind === 'prewarm') return <Zap className={className} />
  if (kind === 'compaction') return <Sparkles className={className} />
  if (kind === 'handoff') return <RotateCcw className={className} />
  if (kind === 'tier_change') return <TrendingDown className={className} />
  if (kind === 'loop_intervention') return <Repeat className={className} />
  if (kind === 'loop_hard_stop') return <OctagonX className={className} />
  if (kind === 'subagent') return <GitBranch className={className} />
  return <RotateCcw className={className} />
}

/** A health check with no findings is a pass, not a warning. Kept on the
 *  `agent_health` kind so the filter and sort order still group it. */
function isHealthPass(item: ProcessItem): boolean {
  return item.kind === 'agent_health' && !(Array.isArray(item.event.payload.findings) && item.event.payload.findings.length > 0)
}

function iconStyle(item: ProcessItem): string {
  const { kind } = item
  if (kind === 'tool_error' || kind === 'agent_health_stop') return 'bg-[rgba(244,63,94,0.12)] text-[var(--color-danger)]'
  if (kind === 'tool_result' || isHealthPass(item)) return 'bg-[rgba(16,185,129,0.12)] text-[var(--color-success)]'
  if (kind === 'reasoning') return 'bg-[var(--color-accent-soft)] text-[var(--color-accent)]'
  if (kind === 'function_call' || kind === 'tool_call') return 'bg-[rgba(56,189,248,0.12)] text-[var(--color-info)]'
  if (kind === 'agent_health') return 'bg-[rgba(245,158,11,0.12)] text-[var(--color-warning)]'
  if (kind === 'prewarm') return 'bg-[rgba(168,85,247,0.12)] text-[#a855f7]'
  if (kind === 'compaction') return 'bg-[rgba(99,102,241,0.12)] text-[#6366f1]'
  if (kind === 'handoff') return 'bg-[rgba(14,165,233,0.12)] text-[#0ea5e9]'
  if (kind === 'tier_change') return 'bg-[rgba(245,158,11,0.12)] text-[var(--color-warning)]'
  if (kind === 'loop_intervention') return 'bg-[rgba(245,158,11,0.12)] text-[var(--color-warning)]'
  if (kind === 'loop_hard_stop') return 'bg-[rgba(244,63,94,0.12)] text-[var(--color-danger)]'
  if (kind === 'subagent') return 'bg-[rgba(168,85,247,0.12)] text-[#a855f7]'
  return 'bg-[var(--color-surface-highlight)] text-[var(--color-text)]'
}

function badgeVariant(item: ProcessItem): 'default' | 'primary' | 'warning' | 'danger' | 'info' | 'success' {
  const { kind } = item
  if (kind === 'tool_error' || kind === 'agent_health_stop') return 'danger'
  if (kind === 'tool_result' || isHealthPass(item)) return 'success'
  if (kind === 'reasoning') return 'primary'
  if (kind === 'function_call' || kind === 'tool_call') return 'info'
  if (kind === 'agent_health') return 'warning'
  if (kind === 'prewarm') return 'primary'
  if (kind === 'compaction') return 'info'
  if (kind === 'handoff') return 'info'
  if (kind === 'tier_change') return 'warning'
  if (kind === 'loop_intervention') return 'warning'
  if (kind === 'loop_hard_stop') return 'danger'
  if (kind === 'subagent') return 'primary'
  return 'default'
}

function DetailValue({ label, value }: { label: string; value: string | number | undefined }) {
  if (value === undefined || value === '') return null
  const useMono = /(^event$|id$|request group|merged events|session$|summarized)/i.test(label)
  return (
    <div className="group relative min-w-0 overflow-hidden rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] px-3.5 py-3 shadow-sm transition-colors hover:border-[var(--color-border-strong)]">
      <div className="absolute inset-y-3 left-0 w-0.5 rounded-r-full bg-[var(--color-accent)]/45 opacity-0 transition-opacity group-hover:opacity-100" />
      <dt className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--color-text-muted)]">
        <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-border-strong)]" />
        {label}
      </dt>
      <dd className={`mt-1.5 break-words text-sm font-medium leading-5 text-[var(--color-text-heading)] ${useMono ? 'font-mono text-xs' : ''}`} title={String(value)}>{value}</dd>
    </div>
  )
}

function ContentPanel({ title, content, mono = false, markdown = false }: { title: string; content: string; mono?: boolean; markdown?: boolean }) {
  const showMarkdown = markdown && !isJsonDocument(content)
  return (
    <div>
      <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">{title}</p>
      {showMarkdown ? (
        // No height cap here: the content fits its natural height and the
        // inspector panel's own scroll container scrolls the whole thing.
        <MarkdownContent className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] p-3" content={content} />
      ) : (
        <pre className={`whitespace-pre-wrap break-words rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] p-3 text-sm text-[var(--color-text-heading)] ${mono ? 'font-mono text-xs' : 'font-sans'}`}>
          {content}
        </pre>
      )}
    </div>
  )
}

const LazyActivityDetails = memo(function ActivityDetails({ item }: { item: ProcessItem }) {
  const { event } = item
  const payload = event.payload
  const modelPayload = item.modelEvent?.payload
  const resultEvent = item.resultEvent
  const resultEventPayload = resultEvent?.payload
  const timing = eventTiming(resultEvent ?? event)
  const inputTokens = numberValue(modelPayload?.input_tokens) ?? numberValue(payload.input_tokens)
  const outputTokens = numberValue(modelPayload?.output_tokens) ?? numberValue(payload.output_tokens)
  const totalTokens = inputTokens !== undefined || outputTokens !== undefined ? (inputTokens ?? 0) + (outputTokens ?? 0) : undefined
  const cachedTokens = numberValue(modelPayload?.cached_prompt_tokens) ?? numberValue(payload.cached_prompt_tokens)
  const result = asObject(resultEventPayload?.result ?? payload.result)
  const toolName = item.toolCall?.function?.name ?? stringValue(result.name)
  const toolCallId = item.toolCall?.id ?? stringValue(result.tool_call_id)
  const argumentsText = formatArguments(item.toolCall?.function?.arguments)
  const findings = Array.isArray(payload.findings) ? payload.findings.map(asObject) : []
  const loopView = event.event_type === 'loop_intervention' ? readLoopIntervention(payload) : undefined
  const loopRecord = loopView?.record
  const loopRecordedAt = loopRecord?.recordedAt
  const executionPayload = resultEventPayload ?? payload
  const executionSource = executionPayload.execution_source === 'external_client'
    ? 'External client'
    : executionPayload.execution_source === 'reincore'
      ? 'ReinCore daemon'
      : undefined

  return (
    <div className="border-t border-[var(--color-border)] bg-[var(--color-bg-subtle)]/50 px-4 py-4 sm:px-5">
      <section>
        <div className="mb-3 flex items-end justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--color-text-heading)]">Event metadata</p>
            <p className="mt-0.5 text-xs text-[var(--color-text-muted)]">Timing, usage, execution, and audit identifiers</p>
          </div>
          <span className="rounded-full border border-[var(--color-border)] bg-[var(--color-surface)] px-2 py-1 font-mono text-[10px] text-[var(--color-text-muted)]">Global #{event.seq}</span>
        </div>
        <dl className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <DetailValue label="Event" value={`#${item.localSeq ?? event.seq} · ${event.event_type.replaceAll('_', ' ')}`} />
        {(item.mergedFromLocalSeqs?.length ?? item.mergedFromSeqs?.length ?? 0) > 0 && (
          <DetailValue label="Merged events" value={(item.mergedFromLocalSeqs ?? item.mergedFromSeqs!).map(s => `#${s}`).join(', ')} />
        )}
        <DetailValue label="Recorded" value={formatTimestamp(event.created_at)} />
        <DetailValue label="Result recorded" value={resultEvent ? formatTimestamp(resultEvent.created_at) : undefined} />
        <DetailValue label="Started" value={formatTimestamp(timing.start)} />
        <DetailValue label="Ended" value={formatTimestamp(timing.end)} />
        <DetailValue label="Duration" value={timing.duration !== undefined ? formatDuration(timing.duration) : undefined} />
        <DetailValue label="Model" value={stringValue(modelPayload?.model) ?? stringValue(payload.model)} />
        <DetailValue label="Model tier" value={stringValue(modelPayload?.model_tier) ?? stringValue(payload.model_tier)} />
        <DetailValue label="Cost" value={typeof item.modelEvent?.cost_cents === 'number' ? formatCost(item.modelEvent.cost_cents) : typeof event.cost_cents === 'number' ? formatCost(event.cost_cents) : undefined} />
        <DetailValue label="Input tokens" value={inputTokens !== undefined ? formatTokens(inputTokens) : undefined} />
        <DetailValue label="Output tokens" value={outputTokens !== undefined ? formatTokens(outputTokens) : undefined} />
        <DetailValue label="Total tokens" value={totalTokens !== undefined ? formatTokens(totalTokens) : undefined} />
        <DetailValue label="Cached tokens" value={cachedTokens !== undefined ? formatTokens(cachedTokens) : undefined} />
        <DetailValue label="Function / tool" value={toolName} />
        <DetailValue label="Call ID" value={toolCallId} />
        <DetailValue label="Executed by" value={executionSource} />
        <DetailValue label="Request group" value={stringValue(payload.request_group_id)} />
        <DetailValue label="Cache" value={payload.cache_hit === true ? 'Hit' : payload.cache_hit === false ? 'Miss' : undefined} />
        <DetailValue label="Batch" value={payload.batch_submitted === true ? `Submitted${stringValue(payload.batch_id) ? ` · ${stringValue(payload.batch_id)}` : ''}` : undefined} />

        {/* Pre-warm details */}
        {item.kind === 'prewarm' && <>
          <DetailValue label="Warmed for tier" value={tierLabel(numberValue(payload.warmed_for_tier))} />
          <DetailValue label="Provider warmed" value={payload.provider_warmed === true ? 'Yes' : 'No'} />
          <DetailValue label="Provider" value={stringValue(payload.provider_label)} />
          <DetailValue label="Warmed model" value={stringValue(payload.warmed_model)} />
          <DetailValue label="Pre-fetched tool states" value={numberValue(payload.pre_fetched_tool_states)} />
          <DetailValue label="Shape comparisons" value={numberValue(payload.shape_comparisons)} />
          <DetailValue label="Best similarity" value={(() => { const v = numberValue(payload.best_shape_similarity); return v !== undefined ? `${(v * 100).toFixed(1)}%` : undefined })()} />
          <DetailValue label="Elapsed" value={(() => { const v = numberValue(payload.elapsed_us); return v !== undefined ? formatDuration(v / 1000) : undefined })()} />
        </>}

        {/* Compaction details - standalone Summary event or inline from LlmCall context_adjustment */}
        {item.kind === 'compaction' && (() => {
          const adj = asObject(payload.context_adjustment)
          const hasAdj = numberValue(adj.compressed_message_count) !== undefined || numberValue(adj.original_token_count) !== undefined
          if (hasAdj) {
            const origTok = numberValue(adj.original_token_count)
            const finalTok = numberValue(adj.final_token_count)
            const tokensCompacted = origTok !== undefined && finalTok !== undefined ? origTok - finalTok : undefined
            return <>
              <DetailValue label="Total compacted" value={numberValue(adj.compressed_message_count)} />
              <DetailValue label="Completed groups" value={numberValue(adj.request_group_compacted_message_count)} />
              <DetailValue label="Active tools" value={numberValue(adj.active_tool_compacted_message_count)} />
              <DetailValue label="Reset" value={numberValue(adj.reset_message_count)} />
              <DetailValue label="Mode" value={stringValue(adj.compression_mode)} />
              <DetailValue label="Tokens before" value={origTok !== undefined ? formatTokens(origTok) : undefined} />
              <DetailValue label="Tokens after" value={finalTok !== undefined ? formatTokens(finalTok) : undefined} />
              <DetailValue label="Tokens compacted" value={tokensCompacted !== undefined && tokensCompacted > 0 ? `${formatTokens(tokensCompacted)} (${Math.round(tokensCompacted / (origTok ?? 1) * 100)}%)` : undefined} />
              <DetailValue label="Bytes before" value={numberValue(adj.original_byte_count) !== undefined ? formatBytes(numberValue(adj.original_byte_count)!) : undefined} />
              <DetailValue label="Bytes after" value={numberValue(adj.final_byte_count) !== undefined ? formatBytes(numberValue(adj.final_byte_count)!) : undefined} />
            </>
          }
          return <>
            <DetailValue label="Summarized from" value={(() => { const v = numberValue(payload.summarized_seq_start); if (v === undefined) return undefined; const local = item.localSeqByGlobalSeq?.get(v); return local !== undefined ? `#${local}` : `#${v}` })()} />
            <DetailValue label="Summarized to" value={(() => { const v = numberValue(payload.summarized_seq_end); if (v === undefined) return undefined; const local = item.localSeqByGlobalSeq?.get(v); return local !== undefined ? `#${local}` : `#${v}` })()} />
          </>
        })()}

        {/* Session handoff details */}
        {item.kind === 'handoff' && <>
          <DetailValue label="Previous session" value={stringValue(payload.previous_session_id)} />
          <DetailValue label="Next session" value={stringValue(payload.next_session_id)} />
          <DetailValue label="Root session" value={stringValue(payload.root_session_id)} />
          <DetailValue label="Reason" value={stringValue(payload.reason)} />
          <DetailValue label="Tokens before" value={numberValue(payload.context_tokens_before) !== undefined ? formatTokens(numberValue(payload.context_tokens_before)!) : undefined} />
          <DetailValue label="Tokens after" value={numberValue(payload.context_tokens_after) !== undefined ? formatTokens(numberValue(payload.context_tokens_after)!) : undefined} />
        </>}

        {/* Tier change details */}
        {item.kind === 'tier_change' && <>
          <DetailValue label="Previous tier" value={tierLabel(numberValue(payload.previous_tier))} />
          <DetailValue label="New tier" value={tierLabel(numberValue(payload.new_tier))} />
          <DetailValue label="Spend" value={numberValue(payload.spend_cents) !== undefined ? formatCost(numberValue(payload.spend_cents)!) : undefined} />
          <DetailValue label="Budget" value={numberValue(payload.budget_cents) !== undefined ? formatCost(numberValue(payload.budget_cents)!) : undefined} />
        </>}

        {/* Loop intervention details */}
        {loopView && <>
          <DetailValue label="Repeated steps" value={loopView.cleared ? undefined : loopView.repetitionCount} />
          <DetailValue label="Response" value={loopView.cleared ? 'Cleared' : interventionActionLabel(loopRecord?.action) ?? (loopView.hardStop ? 'Paused' : 'Repair requested')} />
          <DetailValue label="Detector" value={detectorLabel(loopRecord?.detector)} />
          <DetailValue label="Configured mode" value={heuristicModeLabel(loopRecord?.configuredMode)} />
          {/* Written at detection time. A later agent_health event carries any resolution. */}
          <DetailValue label="Outcome when recorded" value={interventionOutcomeLabel(loopRecord?.outcome)} />
          <DetailValue label="Detection" value={healthKindLabel(loopRecord?.evidence.kind)} />
          <DetailValue label="Severity" value={healthSeverityLabel(loopRecord?.evidence.severity)} />
          <DetailValue label="Detected on tool" value={loopRecord?.evidence.toolName} />
          <DetailValue label="Next turn" value={loopView.cleared ? 'Allowed to proceed' : loopView.hardStop ? (loopView.retryable ? 'Refused · retry allowed' : 'Refused · retry not allowed') : 'Allowed with the injected message'} />
          <DetailValue label="Detected at" value={loopRecordedAt ? formatTimestamp(loopRecordedAt) : undefined} />
          <DetailValue label="Repeated step ID" value={loopView.repeatedHash} />
          <DetailValue label="Intervention ID" value={loopRecord?.id} />
        </>}
        </dl>
      </section>

      <div className="mt-4 flex flex-col gap-4">
        {item.kind === 'task_decision' && <TaskDecisionDetails payload={payload} />}
        {loopView && <LoopInterventionDetails view={loopView} />}
        {argumentsText && <ContentPanel title={resultEvent ? 'Tool request' : 'Arguments'} content={argumentsText} mono />}
        {item.content && (
          <ContentPanel
            title={resultEvent ? 'Tool result' : item.kind === 'reasoning' ? 'Reasoning' : item.kind === 'tool_result' || item.kind === 'tool_error' ? 'Result' : item.kind === 'compaction' ? 'Summary' : loopView ? 'Injected message' : 'Content'}
            content={item.content}
            mono={Boolean(resultEvent) || item.kind === 'tool_result' || item.kind === 'tool_error'}
            markdown
          />
        )}
        {findings.length > 0 && (
          <div>
            <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">Findings</p>
            <div className="space-y-2">
              {findings.map((finding, index) => (
                <div key={index} className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] p-3 text-sm text-[var(--color-text)]">
                  <span className="font-medium text-[var(--color-text-heading)]">{stringValue(finding.kind)?.replaceAll('_', ' ') ?? 'Finding'}</span>
                  {stringValue(finding.message) && <MarkdownContent className="mt-1" content={stringValue(finding.message)!} />}
                </div>
              ))}
            </div>
          </div>
        )}
        {item.modelEvent && (
          <details>
            <summary className="cursor-pointer text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text)]">View provider function-request payload</summary>
            <pre className="mt-2 whitespace-pre-wrap break-words rounded-lg bg-[var(--color-bg-subtle)] p-3 font-mono text-xs text-[var(--color-text)]">{JSON.stringify(item.modelEvent.payload, null, 2)}</pre>
          </details>
        )}
        {resultEvent && (
          <details>
            <summary className="cursor-pointer text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text)]">View raw tool-result payload</summary>
            <pre className="mt-2 whitespace-pre-wrap break-words rounded-lg bg-[var(--color-bg-subtle)] p-3 font-mono text-xs text-[var(--color-text)]">{JSON.stringify(resultEvent.payload, null, 2)}</pre>
          </details>
        )}
        <details>
          <summary className="cursor-pointer text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text)]">View raw {item.modelEvent ? 'tool-event' : 'audit'} payload</summary>
          <pre className="mt-2 whitespace-pre-wrap break-words rounded-lg bg-[var(--color-bg-subtle)] p-3 font-mono text-xs text-[var(--color-text)]">{JSON.stringify(payload, null, 2)}</pre>
        </details>
      </div>
    </div>
  )
})

function seqLabel(item: ProcessItem): string {
  const seqs = [item.localSeq ?? item.event.seq, ...(item.mergedFromLocalSeqs ?? item.mergedFromSeqs ?? [])].sort((a, b) => a - b)
  if (seqs.length === 1) return `#${seqs[0]}`
  // Show compact range if contiguous, otherwise list them
  const isContiguous = seqs.every((s, i) => i === 0 || s === seqs[i - 1] + 1)
  if (isContiguous) return `#${seqs[0]}–#${seqs[seqs.length - 1]}`
  return seqs.map(s => `#${s}`).join(' + ')
}

const ProcessRow = memo(function ProcessRow({ item, onInspect, selected = false }: { item: ProcessItem; onInspect: (item: ProcessItem) => void; selected?: boolean }) {
  const timing = eventTiming(item.event)
  const metricPayload = item.modelEvent?.payload ?? item.event.payload
  const tokens = (numberValue(metricPayload.input_tokens) ?? 0) + (numberValue(metricPayload.output_tokens) ?? 0)
  const costCents = item.modelEvent?.cost_cents ?? item.event.cost_cents
  const hasMerged = ((item.mergedFromLocalSeqs ?? item.mergedFromSeqs)?.length ?? 0) > 0
  const outcome = item.kind === 'tool_error' ? 'Failed' : item.resultEvent || item.kind === 'tool_result' ? 'Completed' : item.kind === 'tool_call' ? 'Executing' : undefined
  return (
    <div className={`border-b border-[var(--color-border)] transition-colors ${selected ? 'bg-[var(--color-accent-soft)]' : 'bg-[var(--color-surface)]'}`}>
      <button
        type="button"
        onClick={() => onInspect(item)}
        aria-pressed={selected}
        className="flex w-full cursor-pointer items-start gap-3 px-4 py-3 text-left hover:bg-[var(--color-surface-highlight)]/40 focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-[var(--color-accent)]"
      >
        <span className={`mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center ${iconStyle(item)}`}><ProcessIcon kind={item.kind} /></span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
            <span className="break-words text-sm font-medium text-[var(--color-text-heading)]">{item.title}</span>
            <span className="text-[10px] text-[var(--color-text-muted)]">{item.label}</span>
          </div>
          {item.preview && (item.kind === 'function_call' || item.kind === 'tool_call' || isJsonDocument(item.preview)
            ? <p className="mt-0.5 line-clamp-1 text-xs leading-5 text-[var(--color-text-muted)]">{item.preview}</p>
            : <MarkdownContent preview className="rc-md-event-preview mt-0.5" content={item.content?.slice(0, 300) || item.preview} />)}
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-[var(--color-text-muted)]">
            {outcome && <span className={item.kind === 'tool_error' ? 'text-[var(--color-danger)]' : item.resultEvent || item.kind === 'tool_result' ? 'text-[var(--color-success)]' : 'text-[var(--color-accent)]'}>{outcome}</span>}
            {timing.duration !== undefined && <span>{formatDuration(timing.duration)}</span>}
            {tokens > 0 && <span>{formatTokens(tokens)} tokens</span>}
            {typeof costCents === 'number' && <span>{formatCost(costCents)}</span>}
          </div>
        </div>
        <div className="shrink-0 text-right font-mono text-[10px] leading-5 text-[var(--color-text-muted)]">
          <span className="block">{formatClock(timing.start)}</span>
          <span className="block" title={hasMerged ? `Combines ${(item.mergedFromLocalSeqs ?? item.mergedFromSeqs)!.length + 1} related audit events` : undefined}>{seqLabel(item)}{hasMerged && <span className="ml-1 text-[var(--color-accent)]">⟡</span>}</span>
        </div>
      </button>
    </div>
  )
})

function groupMetrics(group: RequestGroup) {
  let inputTokens = 0
  let outputTokens = 0
  let costCents = 0
  const models = new Set<string>()
  for (const event of group.events) {
    if (event.event_type === 'llm_call') {
      inputTokens += numberValue(event.payload.input_tokens) ?? 0
      outputTokens += numberValue(event.payload.output_tokens) ?? 0
      const model = stringValue(event.payload.model)
      if (model) models.add(model)
    }
    costCents += event.cost_cents ?? 0
  }
  const firstTiming = eventTiming(group.events[0])
  const lastTiming = eventTiming(group.events[group.events.length - 1])
  const elapsed = new Date(lastTiming.end).getTime() - new Date(firstTiming.start).getTime()
  return { inputTokens, outputTokens, costCents, models: [...models], start: firstTiming.start, duration: Number.isFinite(elapsed) && elapsed >= 0 ? elapsed : undefined }
}

const ALL_PROCESS_KINDS: { kind: ProcessKind; label: string }[] = [
  { kind: 'model_call', label: 'Model call' },
  { kind: 'reasoning', label: 'Reasoning' },
  { kind: 'model_response', label: 'Response' },
  { kind: 'tool_call', label: 'Tool call' },
  { kind: 'tool_result', label: 'Tool result' },
  { kind: 'tool_error', label: 'Tool error' },
  { kind: 'prewarm', label: 'Pre-warm' },
  { kind: 'compaction', label: 'Compaction' },
  { kind: 'handoff', label: 'Handoff' },
  { kind: 'tier_change', label: 'Tier change' },
  { kind: 'loop_intervention', label: 'Loop intervention' },
  { kind: 'loop_hard_stop', label: 'Loop hard stop' },
  { kind: 'subagent', label: 'Subagent' },
  { kind: 'agent_health', label: 'Agent health' },
  { kind: 'task_decision', label: 'Task decisions' },
  { kind: 'agent_health_stop', label: 'Agent stopped' },
]

export function ProcessKindFilter({ selectedKinds, setSelectedKinds }: { selectedKinds: Set<ProcessKind>; setSelectedKinds: Dispatch<SetStateAction<Set<ProcessKind>>> }) {
  const [open, setOpen] = useState(false)
  const activeCount = selectedKinds.size
  const toggle = (kind: ProcessKind) => {
    setSelectedKinds(prev => {
      const next = new Set(prev)
      if (next.has(kind)) next.delete(kind)
      else next.add(kind)
      return next
    })
  }
  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen(v => !v)}
        className="flex items-center gap-1.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1.5 text-xs font-medium text-[var(--color-text-secondary)] transition-colors hover:bg-[var(--color-surface-highlight)]/40 cursor-pointer"
      >
        <Filter className="h-3.5 w-3.5" />
        <span>Filter</span>
        {activeCount > 0 && <span className="rounded-full bg-[var(--color-accent)] px-1.5 py-0.5 text-[10px] font-semibold text-white">{activeCount}</span>}
        <ChevronDown className={`h-3 w-3 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute right-0 z-50 mt-1 w-56 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-2 shadow-lg">
            <div className="mb-2 flex items-center justify-between px-1">
              <span className="text-[11px] font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">Filter by type</span>
              {activeCount > 0 && (
                <button
                  type="button"
                  onClick={() => { selectedKinds.clear(); setSelectedKinds(new Set()) }}
                  className="text-[11px] font-medium text-[var(--color-accent)] hover:underline"
                >
                  Clear all
                </button>
              )}
            </div>
            <div className="max-h-64 space-y-0.5 overflow-y-auto">
              {ALL_PROCESS_KINDS.map(({ kind, label }) => {
                const checked = activeCount === 0 || selectedKinds.has(kind)
                return (
                  <button
                    key={kind}
                    type="button"
                    onClick={() => toggle(kind)}
                    className="flex w-full items-center gap-2.5 rounded-lg px-2 py-1.5 text-left text-sm transition-colors hover:bg-[var(--color-surface-highlight)]/40"
                  >
                    <span className={`flex h-4 w-4 items-center justify-center rounded border ${checked ? 'border-[var(--color-accent)] bg-[var(--color-accent)]' : 'border-[var(--color-border)] bg-transparent'}`}>
                      {checked && <Check className="h-3 w-3 text-white" />}
                    </span>
                    <span className="flex h-5 w-5 items-center justify-center rounded text-[var(--color-text-muted)]"><ProcessIcon kind={kind} /></span>
                    <span className="text-[var(--color-text-secondary)]">{label}</span>
                  </button>
                )
              })}
            </div>
            {activeCount > 0 && (
              <p className="mt-2 border-t border-[var(--color-border)] px-1 pt-2 text-[11px] text-[var(--color-text-muted)]">{activeCount} type{activeCount === 1 ? '' : 's'} selected · others hidden</p>
            )}
          </div>
        </>
      )}
    </div>
  )
}

/**
 * Renders a compact card with a GitBranch badge showing a subagent session's
 * task summary, status, and spend. Clicking the expand button lazy-loads the
 * subagent's events and renders them inline as a mini-timeline.
 */
function SubagentInlineCard({ subagent }: { subagent: SessionSummary }) {
  const [expanded, setExpanded] = useState(false)
  const [loadingEvents, setLoadingEvents] = useState(false)
  const [subEvents, setSubEvents] = useState<SessionEvent[]>([])
  const [error, setError] = useState<string | null>(null)

  const handleExpand = async () => {
    const next = !expanded
    setExpanded(next)
    if (next && subEvents.length === 0 && !loadingEvents) {
      setLoadingEvents(true)
      setError(null)
      try {
        const evs = await getSessionEvents(subagent.session_id)
        setSubEvents(evs)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load subagent events')
      } finally {
        setLoadingEvents(false)
      }
    }
  }

  const subGroups = expanded && subEvents.length > 0 ? buildRequestGroups(subEvents) : []

  return (
    <div className="mt-3 rounded-xl border border-[rgba(168,85,247,0.25)] bg-[rgba(168,85,247,0.04)] p-3">
      <div className="flex items-center gap-3">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-[rgba(168,85,247,0.12)] text-[#a855f7]">
          <GitBranch className="h-3.5 w-3.5" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="primary">Subagent</Badge>
            <span className="truncate text-sm font-medium text-[var(--color-text-heading)]">
              {subagent.task_summary || `Session ${subagent.session_id.slice(0, 8)}`}
            </span>
          </div>
          <div className="mt-0.5 flex flex-wrap items-center gap-x-3 text-xs text-[var(--color-text-muted)]">
            <span className="font-mono">{subagent.session_id.slice(0, 8)}</span>
            <Badge variant={subagent.status === 'active' ? 'success' : subagent.status === 'paused' ? 'warning' : 'default'}>{capitalizeStatus(subagent.status)}</Badge>
            {typeof subagent.spend_cents === 'number' && subagent.spend_cents > 0 && <span>{formatCost(subagent.spend_cents)}</span>}
            {typeof subagent.total_input_tokens === 'number' && typeof subagent.total_output_tokens === 'number' && (subagent.total_input_tokens + subagent.total_output_tokens) > 0 && (
              <span>{formatTokens(subagent.total_input_tokens + subagent.total_output_tokens)} tokens</span>
            )}
          </div>
        </div>
        <button
          type="button"
          onClick={handleExpand}
          className="flex shrink-0 items-center gap-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1.5 text-xs font-medium text-[var(--color-text-secondary)] transition-colors hover:bg-[var(--color-surface-highlight)]/40"
        >
          {loadingEvents ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : expanded ? <CheckCircle2 className="h-3.5 w-3.5 text-[var(--color-success)]" /> : <ChevronDown className="h-3.5 w-3.5" />}
          {expanded ? 'Hide' : 'View'} session
        </button>
      </div>
      {error && (
        <p className="mt-2 text-xs text-[var(--color-danger)]">{error}</p>
      )}
      {expanded && (
        <div className="mt-3 border-t border-[rgba(168,85,247,0.15)] pt-3">
          {loadingEvents ? (
            <p className="text-sm text-[var(--color-text-muted)]">Loading subagent events...</p>
          ) : subEvents.length === 0 ? (
            <p className="text-sm text-[var(--color-text-muted)]">No events recorded for this subagent session.</p>
          ) : (
            <div className="space-y-3">
              {subGroups.map((group, gi) => {
                const allItems = mergeToolLifecycleItems(group.events.flatMap(eventToProcessItems))
                const localSeqByGlobalSeq = new Map<number, number>()
                group.events.forEach((event, i) => localSeqByGlobalSeq.set(event.seq, i + 1))
                const stampLocalSeq = (item: ProcessItem): ProcessItem => ({ ...item, localSeq: localSeqByGlobalSeq.get(item.event.seq) ?? item.event.seq, localSeqByGlobalSeq })
                const stampedItems = allItems.map(stampLocalSeq)
                const subItems = stampedItems.sort((a, b) => (a.localSeq ?? a.event.seq) - (b.localSeq ?? b.event.seq))
                return (
                  <div key={group.id} className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
                    <div className="mb-2 flex items-center gap-2">
                      <span className="font-mono text-[10px] font-semibold uppercase tracking-widest text-[var(--color-accent)]">Subagent · {group.userPrompt ? `Request ${String(gi + 1).padStart(2, '0')}` : 'Activity'}</span>
                    </div>
                    {group.userPrompt && (
                      <MarkdownContent preview className="rc-md-navigation-preview mb-2" content={requestMessage(group.userPrompt)} />
                    )}
                    <TaggedFiles files={parseRequestContent(group.userPrompt).taggedFiles} className="mb-2" />
                    <div className="space-y-1.5">
                      {subItems.map(item => (
                        <div key={item.id} className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)]/30 px-3 py-2">
                          <div className="flex items-center gap-2">
                            <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded ${iconStyle(item)}`}><ProcessIcon kind={item.kind} /></span>
                            <Badge variant={badgeVariant(item)}>{item.label}</Badge>
                            <span className="truncate text-xs font-medium text-[var(--color-text-heading)]">{item.title}</span>
                            <span className="font-mono text-[10px] text-[var(--color-text-muted)]">{seqLabel(item)}</span>
                          </div>
                          {item.preview && <MarkdownContent preview className="rc-md-event-preview mt-1 pl-7" content={item.content?.slice(0, 300) || item.preview} />}
                        </div>
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

const RequestGroupCard = memo(function RequestGroupCard({ group, index, selectedKinds, subagentSessions, groupsLength, collapsed, onToggle }: { group: RequestGroup; index: number; selectedKinds: Set<ProcessKind>; subagentSessions: SessionSummary[]; groupsLength: number; collapsed: boolean; onToggle: () => void }) {
  const allItems = useMemo(() => mergeToolLifecycleItems(group.events.flatMap(eventToProcessItems)), [group.events])

  // Build a per-request-group sequence map: each event gets a 1-based local
  // sequence number so that the timeline numbering restarts at #1 for every
  // request group instead of continuing the global session-level counter.
  const localSeqByGlobalSeq = new Map<number, number>()
  group.events.forEach((event, i) => localSeqByGlobalSeq.set(event.seq, i + 1))
  const stampLocalSeq = (item: ProcessItem): ProcessItem => {
    const localSeq = localSeqByGlobalSeq.get(item.event.seq) ?? item.event.seq
    const mergedFromLocalSeqs = item.mergedFromSeqs?.map(s => localSeqByGlobalSeq.get(s) ?? s)
    return { ...item, localSeq, mergedFromLocalSeqs, localSeqByGlobalSeq }
  }
  const stampedItems = allItems.map(stampLocalSeq)

  const items = useMemo(() => (selectedKinds.size === 0 ? stampedItems : stampedItems.filter(item => selectedKinds.has(item.kind))).sort((left, right) => {
    // Primary sort: local sequence number - per-request-group ordering.
    const seqDiff = (left.localSeq ?? left.event.seq) - (right.localSeq ?? right.event.seq)
    if (seqDiff !== 0) return seqDiff

    // Secondary sort: kind-based ordering for items derived from the same event
    // (e.g. compaction → reasoning → response → function calls from one llm_call).
    const kindOrder: Partial<Record<ProcessKind, number>> = { compaction: 0, tool_result: 1, tool_error: 1, reasoning: 2, model_response: 3, function_call: 4, tool_call: 5, prewarm: 6, tier_change: 7, handoff: 8 }
    return (kindOrder[left.kind] ?? 5) - (kindOrder[right.kind] ?? 5)
  }), [stampedItems, selectedKinds])

  const metrics = useMemo(() => groupMetrics(group), [group])

  // Place each child beside the exact parent request that spawned it. Older
  // rows created before spawn correlation existed retain the previous
  // last-group fallback instead of disappearing from the timeline.
  const isLastGroup = index === groupsLength - 1
  const visibleSubagents = subagentSessions.filter((subagent) =>
    subagent.subagent_spawn_request_group_id === group.id ||
    (!subagent.subagent_spawn_request_group_id && isLastGroup),
  )

  return (
    <Card className="overflow-hidden">
      <div className="border-b border-[var(--color-border)] bg-[var(--color-bg-subtle)]/45 p-4 sm:p-5">
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <span className="font-mono text-xs font-semibold uppercase tracking-widest text-[var(--color-accent)]">
            {group.userPrompt ? `Request ${String(index + 1).padStart(2, '0')}` : 'Session activity'}
          </span>
          {group.taskName && <Badge variant="info">{group.taskName}</Badge>}
          {group.taskType && <Badge>{taskTypeLabel(group.taskType)}</Badge>}
          <span className="text-xs text-[var(--color-text-muted)]">{formatTimestamp(metrics.start)}</span>
          <div className="ml-auto flex flex-wrap items-center gap-2 text-xs text-[var(--color-text-muted)]">
            <span>{items.length} {items.length === 1 ? 'step' : 'steps'}</span>
            {metrics.duration !== undefined && <span>· {formatDuration(metrics.duration)}</span>}
            {metrics.inputTokens + metrics.outputTokens > 0 && <span>· {formatTokens(metrics.inputTokens + metrics.outputTokens)} tokens</span>}
            {metrics.costCents > 0 && <span>· {formatCost(metrics.costCents)}</span>}
          </div>
        </div>
        {group.userPrompt ? (
          <>
          <div className="flex gap-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--color-accent-soft)] text-[var(--color-accent)]"><User className="h-4 w-4" /></span>
            <div className="min-w-0">
              <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">User request</p>
              <MarkdownContent content={requestMessage(group.userPrompt)} />
            </div>
          </div>
          <TaggedFiles files={parseRequestContent(group.userPrompt).taggedFiles} className="mt-3" />
          </>
        ) : group.externalSummary ? (
          <div className="flex gap-3 rounded-xl border border-[var(--color-warning)]/40 bg-[var(--color-warning)]/5 p-4">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--color-warning)]/15 text-[var(--color-warning)]"><FileText className="h-4 w-4" /></span>
            <div className="min-w-0">
              <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-[var(--color-warning)]">External context summary · not a user message</p>
              <MarkdownContent content={group.externalSummary} />
            </div>
          </div>
        ) : (
          <p className="text-sm text-[var(--color-text-muted)]">Lifecycle events recorded outside a model request.</p>
        )}
        {metrics.models.length > 0 && (
          <div className="mt-3 flex flex-wrap items-center gap-2">
            {metrics.models.map((model) => <Badge key={model} variant="default">{model}</Badge>)}
            <Badge variant="info">{formatTokens(metrics.inputTokens)} in</Badge>
            <Badge variant="info">{formatTokens(metrics.outputTokens)} out</Badge>
            <button
              type="button"
              onClick={onToggle}
              className="ml-auto flex items-center gap-1 text-xs font-medium text-[var(--color-text-muted)] transition-colors hover:text-[var(--color-text-heading)]"
            >
              {collapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
              {collapsed ? 'Expand' : 'Collapse'}
            </button>
          </div>
        )}
      </div>
      {!collapsed && (
        <div className="p-4 sm:p-5">
          <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-[var(--color-text-muted)]"><Zap className="h-3.5 w-3.5 text-[var(--color-accent)]" /> Process</div>
          {items.length === 0 ? (
            <p className="py-6 text-center text-sm text-[var(--color-text-muted)]">No items match the active filter.</p>
          ) : (
            <div className="relative space-y-2 before:absolute before:bottom-4 before:left-[15px] before:top-4 before:w-px before:bg-[var(--color-border)]">
              {items.map((item) => <div key={item.id} className="relative"><ProcessRow item={item} onInspect={() => {}} /></div>)}
            </div>
          )}
        </div>
      )}
      {!collapsed && visibleSubagents.length > 0 && (
        <div className="border-t border-[rgba(168,85,247,0.15)] px-4 pb-4 pt-3 sm:px-5">
          <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-[#a855f7]">
            <GitBranch className="h-3.5 w-3.5" />
            {visibleSubagents.length === 1 ? 'Subagent session' : `${visibleSubagents.length} subagent sessions`}
          </div>
          {visibleSubagents.map(sa => <SubagentInlineCard key={sa.session_id} subagent={sa} />)}
        </div>
      )}
    </Card>
  )
})

/** Lazy-render wrapper that only mounts children when scrolled near the viewport. */
function LazyMount({ children, fallbackHeight = 80 }: { children: React.ReactNode; fallbackHeight?: number }) {
  const [shouldRender, setShouldRender] = useState(false)
  const ref = useRef<HTMLElement | null>(null)
  useEffect(() => {
    if (shouldRender) return
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setShouldRender(true)
          observer.disconnect()
        }
      },
      { rootMargin: '200px' },
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [shouldRender])
  if (shouldRender) return <>{children}</>
  return <div ref={ref as React.RefObject<HTMLDivElement>} style={{ minHeight: fallbackHeight }} />
}

export function LegacySessionEventTimeline({ events, selectedKinds, subagentSessions }: { events: SessionEvent[]; selectedKinds: Set<ProcessKind>; subagentSessions?: SessionSummary[] }) {
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set())
  const groups = useMemo(() => buildRequestGroups(events), [events])

  if (events.length === 0 && (!subagentSessions || subagentSessions.length === 0)) {
    return (
      <Card className="flex flex-col items-center justify-center p-10 text-center">
        <FileText className="mb-3 h-8 w-8 text-[var(--color-text-muted)]" />
        <p className="font-medium text-[var(--color-text-heading)]">No activity recorded</p>
        <p className="mt-1 text-sm text-[var(--color-text-muted)]">Events will appear here as this session runs.</p>
      </Card>
    )
  }
  const subs = subagentSessions ?? []

  const toggleGroup = (id: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const allCollapsed = groups.length > 1 && collapsedGroups.size === groups.length

  return (
    <div className="flex flex-col gap-5">
      {groups.length > 3 && (
        <div className="flex items-center justify-end gap-2 text-xs">
          <button
            type="button"
            onClick={() => {
              if (allCollapsed) setCollapsedGroups(new Set())
              else setCollapsedGroups(new Set(groups.map((g) => g.id)))
            }}
            className="flex items-center gap-1.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1.5 font-medium text-[var(--color-text-secondary)] transition-colors hover:bg-[var(--color-surface-highlight)]/40"
          >
            {allCollapsed ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
            {allCollapsed ? 'Expand all' : 'Collapse all'}
          </button>
        </div>
      )}
      {groups.map((group, index) => {
        const collapsed = collapsedGroups.has(group.id)
        return (
          <LazyMount key={group.id} fallbackHeight={120}>
            <RequestGroupCard
              group={group}
              index={index}
              selectedKinds={selectedKinds}
              subagentSessions={subs}
              groupsLength={groups.length}
              collapsed={collapsed}
              onToggle={() => toggleGroup(group.id)}
            />
          </LazyMount>
        )
      })}
    </div>
  )
}

function useTimelineDialog(onClose: () => void) {
  const panelRef = useRef<HTMLElement>(null)
  const triggerRef = useRef<HTMLButtonElement | null>(null)
  const closeDialog = useCallback(() => {
    const trigger = triggerRef.current
    onClose()
    if (trigger) window.requestAnimationFrame(() => {
      if (trigger.isConnected && document.activeElement === document.body) trigger.focus({ preventScroll: true })
    })
  }, [onClose])
  useEffect(() => {
    if (!triggerRef.current && document.activeElement instanceof HTMLButtonElement) triggerRef.current = document.activeElement
    const previousDocumentOverflow = document.documentElement.style.overflow
    const previousScrollbarGutter = document.documentElement.style.scrollbarGutter
    // Keep the sticky timeline attached to the root scroll container while
    // the modal is open, and avoid resizing it when the scrollbar is locked.
    document.documentElement.style.scrollbarGutter = 'stable'
    document.documentElement.style.overflow = 'hidden'
    panelRef.current?.focus({ preventScroll: true })
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        closeDialog()
        return
      }
      const panel = panelRef.current
      if (event.key !== 'Tab' || !panel) return
      const focusable = [...panel.querySelectorAll<HTMLElement>('button, [href], input, select, textarea, summary, [tabindex]:not([tabindex="-1"])')].filter((element) => !element.hasAttribute('disabled') && element.getClientRects().length > 0)
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (!first || !last) {
        event.preventDefault()
        panel.focus({ preventScroll: true })
      } else if (event.shiftKey && (document.activeElement === first || document.activeElement === panel || !panel.contains(document.activeElement))) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && (document.activeElement === last || document.activeElement === panel || !panel.contains(document.activeElement))) {
        event.preventDefault()
        first.focus()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      document.documentElement.style.overflow = previousDocumentOverflow
      document.documentElement.style.scrollbarGutter = previousScrollbarGutter
    }
  }, [closeDialog])
  return { panelRef, closeDialog }
}

function EventInspector({ item, onClose }: { item: ProcessItem; onClose: () => void }) {
  const [toolTab, setToolTab] = useState<'request' | 'response'>('request')
  const { panelRef, closeDialog } = useTimelineDialog(onClose)

  const displayedItem = useMemo((): ProcessItem => {
    if (!item.resultEvent) return item
    if (toolTab === 'request') {
      return {
        ...item,
        kind: 'tool_call',
        label: 'Tool request',
        resultEvent: undefined,
        content: undefined,
        mergedFromSeqs: item.modelEvent ? [item.modelEvent.seq] : undefined,
        mergedFromLocalSeqs: item.modelEvent
          ? [item.localSeqByGlobalSeq?.get(item.modelEvent.seq) ?? item.modelEvent.seq]
          : undefined,
      }
    }
    const responseItem = eventToProcessItems(item.resultEvent)[0]
    return {
      ...responseItem,
      localSeq: item.localSeqByGlobalSeq?.get(item.resultEvent.seq) ?? item.resultEvent.seq,
      localSeqByGlobalSeq: item.localSeqByGlobalSeq,
    }
  }, [item, toolTab])

  return createPortal(
    <div className="fixed inset-0 z-[100]" role="dialog" aria-modal="true" aria-label={`${item.label} details`}>
      <button type="button" aria-label="Close event details" className="absolute inset-0 bg-black/45 backdrop-blur-[1px] cursor-pointer" onClick={closeDialog} />
      <aside ref={panelRef} tabIndex={-1} className="absolute inset-y-0 right-0 flex w-[min(680px,94vw)] flex-col border-l border-[var(--color-border)] bg-[var(--color-surface)] shadow-2xl outline-none">
        <div className="flex items-start gap-3 border-b border-[var(--color-border)] px-5 py-4">
          <span className={`mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${iconStyle(displayedItem)}`}>
            <ProcessIcon kind={displayedItem.kind} />
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant={badgeVariant(displayedItem)}>{displayedItem.label}</Badge>
              <span className="font-mono text-[11px] text-[var(--color-text-muted)]">{seqLabel(displayedItem)}</span>
            </div>
            <h3 className="mt-1 text-base font-semibold text-[var(--color-text-heading)]">{displayedItem.title}</h3>
          </div>
          <button type="button" onClick={closeDialog} className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-[var(--color-text-muted)] hover:bg-[var(--color-surface-highlight)] hover:text-[var(--color-text-heading)] cursor-pointer transition-colors" aria-label="Close">
            <X className="h-4 w-4" />
          </button>
        </div>
        {item.resultEvent && (
          <div className="border-b border-[var(--color-border)] px-5 py-3">
            <div className="flex gap-2" role="tablist" aria-label="Tool event details">
              <button
                type="button"
                role="tab"
                aria-selected={toolTab === 'request'}
                onClick={() => setToolTab('request')}
                className={`rounded-lg border px-3 py-2 text-sm font-medium cursor-pointer transition-colors ${toolTab === 'request' ? 'border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-text-heading)]' : 'border-transparent text-[var(--color-text)] hover:bg-[var(--color-surface)]'}`}
              >
                Tool request
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={toolTab === 'response'}
                onClick={() => setToolTab('response')}
                className={`rounded-lg border px-3 py-2 text-sm font-medium cursor-pointer transition-colors ${toolTab === 'response' ? 'border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-text-heading)]' : 'border-transparent text-[var(--color-text)] hover:bg-[var(--color-surface)]'}`}
              >
                Tool result
              </button>
            </div>
          </div>
        )}
        <div className="min-h-0 flex-1 overscroll-contain overflow-y-auto">
          <LazyActivityDetails item={displayedItem} />
        </div>
      </aside>
    </div>,
    document.body,
  )
}

function RequestPromptDialog({ request, task, requestNumber, onClose }: { request: RequestGroupTimelineSummary; task?: SessionTask; requestNumber: number; onClose: () => void }) {
  const { panelRef, closeDialog } = useTimelineDialog(onClose)
  const { message, taggedFiles } = parseRequestContent(request.prompt)
  const duration = request.started_at && request.ended_at ? new Date(request.ended_at).getTime() - new Date(request.started_at).getTime() : undefined
  return createPortal(
    <div className="fixed inset-0 z-[100]" role="dialog" aria-modal="true" aria-label={`Request ${requestNumber} details`}>
      <button type="button" aria-label="Close request details" className="absolute inset-0 bg-black/45 backdrop-blur-[1px] cursor-pointer" onClick={closeDialog} />
      <aside ref={panelRef} tabIndex={-1} className="absolute inset-y-0 right-0 flex w-[min(680px,94vw)] flex-col border-l border-[var(--color-border)] bg-[var(--color-surface)] shadow-2xl outline-none">
        <div className="flex items-start gap-3 border-b border-[var(--color-border)] px-5 py-4">
          <div className="min-w-0 flex-1"><h3 className="text-base font-semibold text-[var(--color-text-heading)]">Request {String(requestNumber).padStart(2, '0')}</h3>{task && <p className="mt-1 text-xs text-[var(--color-text-muted)]">{task.name}</p>}</div>
          <button type="button" onClick={closeDialog} className="flex h-9 w-9 shrink-0 cursor-pointer items-center justify-center text-[var(--color-text-muted)] hover:bg-[var(--color-surface-highlight)] hover:text-[var(--color-text-heading)]" aria-label="Close"><X className="h-4 w-4" /></button>
        </div>
        <div className="min-h-0 flex-1 space-y-5 overflow-y-auto overscroll-contain p-5">
          {message && <ContentPanel title="User request" content={message} markdown />}
          {!message && taggedFiles.length === 0 && <p>Request details unavailable</p>}
          <TaggedFiles files={taggedFiles} />
          <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <DetailValue label="Status" value={request.completed ? 'Completed' : 'In progress'} />
            <DetailValue label="Success status" value={request.success_status?.replaceAll('_', ' ') ?? 'Unknown'} />
            <DetailValue label="Started" value={request.started_at ? formatTimestamp(request.started_at) : undefined} />
            <DetailValue label="Duration" value={duration !== undefined && Number.isFinite(duration) && duration >= 0 ? formatDuration(duration) : undefined} />
            <DetailValue label="Events" value={request.event_count.toLocaleString()} />
            <DetailValue label="Model calls" value={request.llm_call_count} />
            <DetailValue label="Tool calls" value={request.tool_call_count} />
            <DetailValue label="Tool errors" value={request.tool_error_count} />
            <DetailValue label="Input tokens" value={request.input_tokens.toLocaleString()} />
            <DetailValue label="Output tokens" value={request.output_tokens.toLocaleString()} />
            <DetailValue label="Cost" value={formatCost(request.cost_cents)} />
            <DetailValue label="Models" value={request.models.join(', ') || undefined} />
          </dl>
        </div>
      </aside>
    </div>,
    document.body,
  )
}

function RequestEventList({
  sessionId,
  request,
  selectedKinds,
  liveEvents,
  onInspect,
  selectedItemId,
}: {
  sessionId: string
  request: RequestGroupTimelineSummary
  selectedKinds: Set<ProcessKind>
  liveEvents: SessionEvent[]
  onInspect: (item: ProcessItem) => void
  selectedItemId?: string
}) {
  const [events, setEvents] = useState<SessionEvent[]>([])
  const [nextSeq, setNextSeq] = useState<number | undefined>(request.first_event_seq)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadMore = useCallback(async () => {
    if (loading || nextSeq === undefined) return
    setLoading(true)
    setError(null)
    try {
      const page = await getSessionEventPage(sessionId, {
        sinceSeq: nextSeq,
        untilSeq: request.last_event_seq,
        limit: 100,
      })
      setEvents((current) => {
        const known = new Set(current.map((event) => event.seq))
        return [...current, ...page.events.filter((event) => !known.has(event.seq))]
          .sort((left, right) => left.seq - right.seq)
      })
      setNextSeq(page.next_since_seq)
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Unable to load request events')
    } finally {
      setLoading(false)
    }
  }, [loading, nextSeq, request.last_event_seq, sessionId])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void loadMore()
    // The keyed component mounts when a request is selected.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const additions = liveEvents.filter((event) => (
      event.seq >= request.first_event_seq && event.seq <= request.last_event_seq
    ))
    if (additions.length === 0) return
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setEvents((current) => {
      const bySeq = new Map(current.map((event) => [event.seq, event]))
      additions.forEach((event) => bySeq.set(event.seq, event))
      return [...bySeq.values()].sort((left, right) => left.seq - right.seq)
    })
  }, [liveEvents, request.first_event_seq, request.last_event_seq])

  const items = useMemo(() => {
    const localSeqByGlobalSeq = new Map<number, number>()
    events.forEach((event) => localSeqByGlobalSeq.set(event.seq, event.seq - request.first_event_seq + 1))
    return mergeToolLifecycleItems(events.flatMap(eventToProcessItems))
      .map((item): ProcessItem => ({
        ...item,
        localSeq: localSeqByGlobalSeq.get(item.event.seq) ?? item.event.seq,
        mergedFromLocalSeqs: item.mergedFromSeqs?.map((seq) => localSeqByGlobalSeq.get(seq) ?? seq),
        localSeqByGlobalSeq,
      }))
      .filter((item) => selectedKinds.size === 0 || selectedKinds.has(item.kind))
      .sort((left, right) => (left.localSeq ?? left.event.seq) - (right.localSeq ?? right.event.seq))
  }, [events, request.first_event_seq, selectedKinds])

  if (loading && events.length === 0) {
    return <div className="flex items-center justify-center gap-2 py-8 text-sm text-[var(--color-text-muted)]"><Loader2 className="h-4 w-4 animate-spin" /> Loading request events...</div>
  }

  return (
    <div className="border-t border-[var(--color-border)]">
      {error && <div className="flex flex-wrap items-center justify-between gap-2 bg-[rgba(244,63,94,0.08)] px-4 py-3 text-sm text-[var(--color-danger)]"><p role="alert">{error}</p><button type="button" onClick={() => void loadMore()} disabled={loading} className="cursor-pointer font-medium underline underline-offset-2 disabled:opacity-50">Retry</button></div>}
      {items.length === 0 ? (
        !error && <p className="py-5 text-center text-sm text-[var(--color-text-muted)]">{selectedKinds.size > 0 ? 'No events match the active filter.' : 'No events recorded for this request.'}</p>
      ) : (
        <div aria-label="Request events">
          {items.map((item) => <ProcessRow key={item.id} item={item} onInspect={onInspect} selected={item.id === selectedItemId} />)}
        </div>
      )}
      {nextSeq !== undefined && (
        <button type="button" onClick={() => void loadMore()} disabled={loading} className="mt-3 mx-auto flex items-center gap-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-3 py-2 text-sm font-medium text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-highlight)] disabled:opacity-50 cursor-pointer transition-colors">
          {loading && <Loader2 className="h-4 w-4 animate-spin" />}
          Load more events
        </button>
      )}
      <p className="px-4 py-3 text-center text-[11px] text-[var(--color-text-muted)]">
        Loaded {events.length.toLocaleString()} of {request.event_count.toLocaleString()} events
      </p>
    </div>
  )
}

function RequestActivityPanel({
  sessionId,
  request,
  task,
  requestNumber,
  selectedKinds,
  liveEvents,
  subagents,
  onInspect,
  selectedItemId,
}: {
  sessionId: string
  request: RequestGroupTimelineSummary
  task?: SessionTask
  requestNumber: number
  selectedKinds: Set<ProcessKind>
  liveEvents: SessionEvent[]
  subagents: SessionSummary[]
  onInspect: (item: ProcessItem) => void
  selectedItemId?: string
}) {
  const [promptOpen, setPromptOpen] = useState(false)
  const { message, taggedFiles } = parseRequestContent(request.prompt)
  const closePrompt = useCallback(() => setPromptOpen(false), [])
  const started = request.started_at ? formatTimestamp(request.started_at) : undefined
  const duration = request.started_at && request.ended_at
    ? new Date(request.ended_at).getTime() - new Date(request.started_at).getTime()
    : undefined
  const metrics = [
    started,
    `${request.event_count.toLocaleString()} events`,
    `${request.llm_call_count} model call${request.llm_call_count === 1 ? '' : 's'}`,
    request.tool_call_count > 0 ? `${request.tool_call_count} tool call${request.tool_call_count === 1 ? '' : 's'}` : undefined,
    (request.input_tokens + request.output_tokens) > 0 ? `${formatTokens(request.input_tokens + request.output_tokens)} tokens` : undefined,
    request.cost_cents > 0 ? formatCost(request.cost_cents) : undefined,
    duration !== undefined && Number.isFinite(duration) && duration >= 0 ? formatDuration(duration) : undefined,
  ].filter(Boolean).join(' · ')

  return (
    <section className="flex h-full min-h-0 min-w-0 flex-col border border-[var(--color-border)] bg-[var(--color-surface)]" aria-label={`Request ${requestNumber} activity`}>
      <div className="max-h-[55%] shrink-0 overflow-hidden px-4 py-3 sm:px-5">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-[var(--color-accent)]">Request {String(requestNumber).padStart(2, '0')}</span>
            {task && <span className="max-w-full truncate text-xs text-[var(--color-text-secondary)]">{task.name}</span>}
            <Badge variant={request.completed ? 'success' : 'default'}>{request.completed ? 'Completed' : 'In progress'}</Badge>
            {request.tool_error_count > 0 && <Badge variant="danger">{request.tool_error_count} error{request.tool_error_count === 1 ? '' : 's'}</Badge>}
            {request.prompt && <button type="button" onClick={() => setPromptOpen(true)} aria-haspopup="dialog" className="ml-auto shrink-0 cursor-pointer text-[11px] font-medium text-[var(--color-accent)] hover:underline">Show full request</button>}
          </div>
          <div className="rc-request-preview mt-2"><MarkdownContent preview content={message || (taggedFiles.length > 0 ? 'Files attached' : 'Request details unavailable')} /></div>
          <p className="mt-2 line-clamp-2 text-[11px] leading-4 text-[var(--color-text-muted)]" title={metrics}>{metrics}</p>
          {request.models.length > 0 && <p className="mt-1 truncate text-[10px] text-[var(--color-text-muted)]" title={request.models.join(', ')}>{request.models.join(' · ')}</p>}
        </div>
      </div>
      <TaggedFiles files={taggedFiles} compact className="mx-4 mb-3 sm:mx-5" />
      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain" data-event-scroll tabIndex={0} aria-label="Request activity">
      <RequestEventList key={request.request_group_id} sessionId={sessionId} request={request} selectedKinds={selectedKinds} liveEvents={liveEvents} onInspect={onInspect} selectedItemId={selectedItemId} />
      {subagents.length > 0 && (
        <div className="border-t border-[rgba(168,85,247,0.15)] px-4 pb-4 pt-3 sm:px-5">
          <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-[#a855f7]"><GitBranch className="h-3.5 w-3.5" /> Subagent sessions</div>
          {subagents.map((subagent) => <SubagentInlineCard key={subagent.session_id} subagent={subagent} />)}
        </div>
      )}
      </div>
      {promptOpen && <RequestPromptDialog request={request} task={task} requestNumber={requestNumber} onClose={closePrompt} />}
    </section>
  )
}

export function SessionEventTimeline({
  sessionId,
  timeline,
  selectedKinds,
  setSelectedKinds,
  subagentSessions = [],
  liveEvents = [],
  connectionState,
}: {
  sessionId: string
  timeline: SessionTimelineSummary
  selectedKinds: Set<ProcessKind>
  setSelectedKinds: Dispatch<SetStateAction<Set<ProcessKind>>>
  subagentSessions?: SessionSummary[]
  liveEvents?: SessionEvent[]
  connectionState: SessionStreamState
}) {
  const activityScrollRef = useActivityScroll()
  const [viewMode, setViewMode] = useState<'tasks' | 'chronological'>('tasks')
  const [selectedRequestId, setSelectedRequestId] = useState<string | null>(() => [...timeline.request_groups].sort((left, right) => right.first_event_seq - left.first_event_seq)[0]?.request_group_id ?? null)
  const requestRailRef = useRef<HTMLDivElement>(null)
  const selectedRequestButtonRef = useRef<HTMLButtonElement>(null)
  const [inspectedItem, setInspectedItem] = useState<ProcessItem | null>(null)
  const closeInspector = useCallback(() => setInspectedItem(null), [])
  const inspectItem = useCallback((item: ProcessItem) => {
    setInspectedItem(item)
    const entries = ([
      ['event', item.event],
      ['modelEvent', item.modelEvent],
      ['resultEvent', item.resultEvent],
    ] as const).filter((entry): entry is readonly ['event' | 'modelEvent' | 'resultEvent', SessionEvent] => {
      const event = entry[1]
      return Boolean(event && Array.isArray(event.payload._reincore_omitted_fields))
    })
    if (entries.length === 0) return
    void Promise.all(entries.map(async ([key, event]) => [key, await getSessionEvent(sessionId, event.seq)] as const))
      .then((hydrated) => {
        setInspectedItem((current) => {
          if (!current || current.id !== item.id) return current
          return hydrated.reduce<ProcessItem>((updated, [key, event]) => ({ ...updated, [key]: event }), current)
        })
      })
      .catch(() => {
        // The compact event still contains all display fields; leave it open
        // if the optional full-payload hydration fails.
      })
  }, [sessionId])
  const liveProjection = useMemo(() => projectLiveTimeline(timeline, liveEvents), [timeline, liveEvents])
  const projectedTimeline = liveProjection.timeline
  const requests = useMemo(() => [...projectedTimeline.request_groups].sort((left, right) => left.first_event_seq - right.first_event_seq), [projectedTimeline.request_groups])
  const requestsByTask = useMemo(() => {
    const map = new Map<string, RequestGroupTimelineSummary[]>()
    requests.forEach((request) => map.set(request.task_id, [...(map.get(request.task_id) ?? []), request]))
    return map
  }, [requests])
  const sortedTasks = useMemo(() => [...projectedTimeline.tasks].sort((left, right) => left.first_event_seq - right.first_event_seq), [projectedTimeline.tasks])
  const taskById = useMemo(() => new Map(sortedTasks.map((task) => [task.task_id, task])), [sortedTasks])
  const subagentsByRequest = useMemo(() => {
    const map = new Map<string, SessionSummary[]>()
    subagentSessions.forEach((subagent) => {
      const requestId = subagent.subagent_spawn_request_group_id
      if (requestId) map.set(requestId, [...(map.get(requestId) ?? []), subagent])
    })
    return map
  }, [subagentSessions])
  const latestRequest = requests[requests.length - 1]
  const selectedRequest = requests.find((request) => request.request_group_id === selectedRequestId) ?? latestRequest
  useLayoutEffect(() => {
    const rail = requestRailRef.current
    const selectedButton = selectedRequestButtonRef.current
    if (!rail || !selectedButton) return
    const railBounds = rail.getBoundingClientRect()
    const selectedBounds = selectedButton.getBoundingClientRect()
    if (selectedBounds.top < railBounds.top) rail.scrollTop += selectedBounds.top - railBounds.top
    else if (selectedBounds.bottom > railBounds.bottom) rail.scrollTop += selectedBounds.bottom - railBounds.bottom
  }, [selectedRequest?.request_group_id, viewMode])
  useEffect(() => {
    if (!latestRequest || requests.some((request) => request.request_group_id === selectedRequestId)) return
    // Select once when an initially empty timeline first receives a request.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSelectedRequestId(latestRequest.request_group_id)
  }, [latestRequest, requests, selectedRequestId])
  const selectRequest = (requestId: string) => {
    setSelectedRequestId(requestId)
    setInspectedItem(null)
  }
  const requestNumbers = new Map(requests.map((request, index) => [request.request_group_id, index + 1]))
  const renderRequest = (request: RequestGroupTimelineSummary, showTask = false) => {
    const selected = request.request_group_id === selectedRequest?.request_group_id
    const task = taskById.get(request.task_id)
    return (
      <button ref={selected ? selectedRequestButtonRef : undefined} key={request.request_group_id} type="button" onClick={() => selectRequest(request.request_group_id)} aria-current={selected ? 'true' : undefined} className={`block w-full border-l-2 px-3 py-3 text-left transition-colors cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-[var(--color-accent)] ${selected ? 'border-[var(--color-accent)] bg-[var(--color-accent-soft)]' : 'border-transparent hover:bg-[var(--color-surface-highlight)]/40'}`}>
        <div className="flex items-center justify-between gap-2 text-[10px]">
          <span className="font-mono font-semibold uppercase tracking-wide text-[var(--color-accent)]">Request {String(requestNumbers.get(request.request_group_id)).padStart(2, '0')}</span>
          {request.tool_error_count > 0 ? <span className="text-[var(--color-danger)]">{request.tool_error_count} error{request.tool_error_count === 1 ? '' : 's'}</span> : <span className="text-[var(--color-text-muted)]">{request.completed ? 'Completed' : 'In progress'}</span>}
        </div>
        <MarkdownContent preview className="rc-md-navigation-preview mt-1" content={requestMessage(request.prompt) || 'Request details unavailable'} />
        {showTask && task && <p className="mt-1 truncate text-[10px] text-[var(--color-text-muted)]">{task.name}</p>}
        <div className="mt-1.5 flex flex-wrap gap-x-2 gap-y-0.5 text-[10px] text-[var(--color-text-muted)]"><span>{request.event_count.toLocaleString()} events</span><span>{formatCost(request.cost_cents)}</span>{(request.pii_redaction_count ?? 0) > 0 ? <span>PII masked · {(request.pii_redaction_count ?? 0).toLocaleString()}</span> : null}</div>
      </button>
    )
  }
  const connection = connectionState === 'live'
    ? { label: 'Live', color: 'bg-[var(--color-success)]' }
    : connectionState === 'reconnecting'
      ? { label: 'Reconnecting', color: 'bg-[var(--color-warning)]' }
      : connectionState === 'stale'
        ? { label: 'Offline', color: 'bg-[var(--color-danger)]' }
        : { label: 'Connecting', color: 'bg-[var(--color-text-muted)]' }

  return (
    <div ref={activityScrollRef} className="flex min-h-0 flex-1 flex-col">
      <div className="mb-4 flex shrink-0 flex-wrap items-center justify-between gap-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2.5">
        <div className="flex flex-wrap items-center gap-3">
          <p className="text-xs text-[var(--color-text-muted)]"><span className="font-semibold text-[var(--color-text-heading)]">{sortedTasks.length}</span> tasks · <span className="font-semibold text-[var(--color-text-heading)]">{requests.length}</span> requests · <span className="font-semibold text-[var(--color-text-heading)]">{projectedTimeline.total_event_count.toLocaleString()}</span> events{(projectedTimeline.pii_redaction_count ?? 0) > 0 ? <> · <span className="font-semibold text-[var(--color-text-heading)]">{(projectedTimeline.pii_redaction_count ?? 0).toLocaleString()}</span> PII masked</> : null}</p>
          <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-[var(--color-text-muted)]"><span className={`h-1.5 w-1.5 rounded-full ${connection.color}`} />{connection.label}</span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <ProcessKindFilter selectedKinds={selectedKinds} setSelectedKinds={setSelectedKinds} />
          <div className="flex rounded-lg bg-[var(--color-bg-subtle)] p-1">
          <button type="button" onClick={() => setViewMode('tasks')} className={`rounded-md px-3 py-1.5 text-xs font-medium cursor-pointer ${viewMode === 'tasks' ? 'bg-[var(--color-surface)] text-[var(--color-text-heading)] shadow-sm' : 'text-[var(--color-text-muted)]'}`}>By task</button>
          <button type="button" onClick={() => setViewMode('chronological')} className={`rounded-md px-3 py-1.5 text-xs font-medium cursor-pointer ${viewMode === 'chronological' ? 'bg-[var(--color-surface)] text-[var(--color-text-heading)] shadow-sm' : 'text-[var(--color-text-muted)]'}`}>Chronological</button>
          </div>
        </div>
      </div>

      {requests.length === 0 && (
        <Card className="p-8 text-center">
          <p className="text-sm font-medium text-[var(--color-text-heading)]">No recorded activity yet</p>
          <p className="mt-1 text-xs text-[var(--color-text-muted)]">New model and tool events will appear here automatically.</p>
        </Card>
      )}

      {selectedRequest && (
        <div className="grid min-h-0 flex-1 grid-rows-[7rem_minmax(0,1fr)] gap-4 md:grid-cols-[15rem_minmax(0,1fr)] md:grid-rows-1">
          <nav aria-label={viewMode === 'tasks' ? 'Requests by task' : 'Requests chronologically'} className="flex h-full min-h-0 min-w-0 flex-col border border-[var(--color-border)] bg-[var(--color-surface)]">
            <div className="flex shrink-0 items-center justify-between gap-2 border-b border-[var(--color-border)] px-3 py-3">
              <h3 className="text-xs font-semibold text-[var(--color-text-heading)]">Requests</h3>
              <button type="button" onClick={() => latestRequest && selectRequest(latestRequest.request_group_id)} disabled={selectedRequest.request_group_id === latestRequest?.request_group_id} className="cursor-pointer text-[10px] font-medium text-[var(--color-accent)] hover:underline disabled:cursor-default disabled:opacity-40">Jump to latest</button>
            </div>
            <div ref={requestRailRef} data-request-rail className="timeline-request-rail min-h-0 flex-1 overflow-y-auto overscroll-contain">
              {viewMode === 'tasks' ? <>
                {sortedTasks.map((task) => <section key={task.task_id} aria-label={task.name}>
                  <div className="border-b border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-3 py-3">
                    <h4 className="break-words text-xs font-semibold leading-5 text-[var(--color-text-heading)]">{task.name}</h4>
                    <p className="mt-0.5 text-[10px] text-[var(--color-text-muted)]">{taskTypeLabel(task.task_type)} · {task.completed ? 'Completed' : 'In progress'}</p>
                    <p className="mt-1 text-[10px] text-[var(--color-text-muted)]">{(requestsByTask.get(task.task_id) ?? []).length} requests · {formatTokens(task.input_tokens + task.output_tokens)} tokens · {formatCost(task.cost_cents)}</p>
                  </div>
                  {(requestsByTask.get(task.task_id) ?? []).map((request) => renderRequest(request))}
                </section>)}
                {requests.some((request) => !taskById.has(request.task_id)) && <section aria-label="Other activity"><h4 className="border-b border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-3 py-3 text-xs font-semibold text-[var(--color-text-heading)]">Other activity</h4>{requests.filter((request) => !taskById.has(request.task_id)).map((request) => renderRequest(request))}</section>}
              </> : requests.map((request) => renderRequest(request, true))}
            </div>
          </nav>
          <RequestActivityPanel key={selectedRequest.request_group_id} sessionId={sessionId} request={selectedRequest} task={taskById.get(selectedRequest.task_id)} requestNumber={requestNumbers.get(selectedRequest.request_group_id) ?? 1} selectedKinds={selectedKinds} liveEvents={liveEvents} subagents={subagentsByRequest.get(selectedRequest.request_group_id) ?? []} onInspect={inspectItem} selectedItemId={inspectedItem?.id} />
        </div>
      )}
      {inspectedItem && <EventInspector key={inspectedItem.id} item={inspectedItem} onClose={closeInspector} />}
    </div>
  )
}
