import { useState } from 'react'
import type { SessionDetail } from '../types'
import { completeSession, extractErrorMessage, getSessionDiagnostics } from '../api'
import { Button, Input } from './ui'

export function OutcomeDiagnostics({ session, onUpdate }: { session: SessionDetail; onUpdate: (session: SessionDetail) => void }) {
  const [evidence, setEvidence] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [diagnostics, setDiagnostics] = useState(session.cost_outcome?.diagnostics)
  const [loadingDiagnostics, setLoadingDiagnostics] = useState(false)
  const [diagnosticsError, setDiagnosticsError] = useState<string | null>(null)
  const loadDiagnostics = async () => {
    setLoadingDiagnostics(true)
    setDiagnosticsError(null)
    try {
      setDiagnostics(await getSessionDiagnostics(session.id))
    } catch (err) {
      setDiagnosticsError(extractErrorMessage(err))
    } finally {
      setLoadingDiagnostics(false)
    }
  }
  const percent = (bps?: number | null) => bps == null ? 'Unknown' : `${(bps / 100).toFixed(1)}%`
  const report = async (verdict: 'success' | 'failure' | 'unknown') => {
    setSaving(true)
    setError(null)
    try {
      onUpdate(await completeSession(session.id, 'user outcome report', { source: 'user', verdict, evidence: evidence.trim() }))
      setEvidence('')
    } catch (err) {
      setError(extractErrorMessage(err))
    } finally {
      setSaving(false)
    }
  }
  return <div className="mt-4 space-y-3 text-xs text-[var(--color-text-secondary)]">
    <div className="flex items-center gap-3">
      <Button variant="secondary" loading={loadingDiagnostics} disabled={loadingDiagnostics} onClick={() => void loadDiagnostics()}>
        {diagnostics ? 'Refresh cost diagnostics' : 'Load cost diagnostics'}
      </Button>
      <span>{diagnostics ? 'Snapshot from the last calculation.' : 'Call metrics are calculated on request.'}</span>
    </div>
    {diagnosticsError && <p role="alert">{diagnosticsError}</p>}
    {diagnostics && <div className="flex flex-wrap gap-4">
      <span title="Upstream provider prompt cache: sum of cached prompt tokens divided by sum of prompt tokens for calls with known usage. Providers that do not report prompt caching, such as local Ollama models, always read 0%.">Provider prompt cache: {percent(diagnostics?.cache.token_weighted_utilization_bps)}</span>
      <span title="ReinCore's own response cache: share of LLM calls answered from cache with no provider request. Independent of the provider.">Response cache hits: {percent(diagnostics?.reincore_cache?.response_hit_rate_bps)} ({diagnostics?.reincore_cache?.response_cache_hit_count ?? 0}/{diagnostics?.reincore_cache?.llm_call_count ?? 0})</span>
      <span title="ReinCore's tool memoization: share of tool results replayed from cache instead of re-executing the tool. Independent of the provider.">Tool memoization hits: {percent(diagnostics?.reincore_cache?.tool_hit_rate_bps)} ({diagnostics?.reincore_cache?.tool_cache_hit_count ?? 0}/{diagnostics?.reincore_cache?.tool_result_count ?? 0})</span>
      <span title="Provider calls where the response cache retrieved a candidate that did not clear the reuse threshold. The bracketed counts are how many a lower threshold would have reused instead - what the change would buy, not a claim that those reuses would have been correct.">Response cache near misses: {diagnostics?.reincore_cache?.near_miss_count ?? 0} (would reuse at 95%: {diagnostics?.reincore_cache?.would_reuse_at_9500_bps ?? 0}, 90%: {diagnostics?.reincore_cache?.would_reuse_at_9000_bps ?? 0}, 85%: {diagnostics?.reincore_cache?.would_reuse_at_8500_bps ?? 0})</span>
      <span title="Calls whose declared tool schema differed from the previous call in the session. Tools render before the system prompt, so each of these invalidated the entire cached prefix rather than only the conversation tail. On a gateway aggregating MCP servers this is the most likely silent cause of a collapsed cache-read rate.">Tool schema churn: {diagnostics?.reincore_cache?.tool_schema_changes ?? 0} of {diagnostics?.reincore_cache?.tool_schema_observations ?? 0} calls</span>
      <span title="Fraction of known call costs inside their predicted low–high interval, including both bounds.">Prediction interval coverage: {percent(diagnostics?.prediction.interval_coverage_bps)}</span>
      <span title="Actual estimated cost divided by predicted high cost; can exceed 100%. This is not accuracy.">Actual / predicted high: {percent(diagnostics?.prediction.mean_actual_to_predicted_high_ratio_bps)}</span>
      <span title="Mean absolute distance between estimated cost and the midpoint of its predicted interval.">Mean midpoint error: {diagnostics?.prediction.mean_absolute_midpoint_error_nano_usd == null ? 'Unknown' : `$${(diagnostics.prediction.mean_absolute_midpoint_error_nano_usd / 1e9).toFixed(6)}`}</span>
    </div>}
    {session.status === 'completed' && <div className="space-y-2">
      <label htmlFor="outcome-evidence">Record your assessment of this completed session</label>
      <Input id="outcome-evidence" value={evidence} onChange={e => setEvidence(e.target.value)} placeholder="Describe what you checked and the result" />
      <div className="flex flex-wrap gap-2">
        <Button variant="secondary" disabled={saving || !evidence.trim()} onClick={() => void report('success')}>Report success</Button>
        <Button variant="secondary" disabled={saving || !evidence.trim()} onClick={() => void report('failure')}>Report failure</Button>
        <Button variant="secondary" disabled={saving || !evidence.trim()} onClick={() => void report('unknown')}>Mark outcome unknown</Button>
      </div>
    </div>}
    {error && <p role="alert">{error}</p>}
  </div>
}
