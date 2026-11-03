// Providers: list, add/remove credentials, and configure model tiers + routing.
// Replaces `reincore provider list/add/remove`, `reincore routing`, and
// `reincore model` CLI commands with a GUI.
//
// Provider configuration opens in a right-side drawer panel (same pattern as
// the session event detail drawer) instead of expanding inside the card.

import { useEffect, useState, useCallback } from 'react'
import { createPortal } from 'react-dom'
import { invoke } from '@tauri-apps/api/core'
import { Cpu, Plus, Trash2, Key, Settings2, RefreshCw, Check, X } from 'lucide-react'
import { ProviderIcon } from '../components/ProviderIcon'
import {
  listProvidersStatus,
  addProviderCredential,
  removeProviderCredential,
  getProviderConfig,
  updateProviderConfig,
  getProviderCatalog,
  listOllamaModels,
  listCustomProviderModels,
  getHealth,
  extractErrorMessage,
} from '../api'
import { Card, Badge, Skeleton, EmptyState, Button, Input, Toggle, ListSelect } from '../components/ui'
import { ModelSelect } from '../components/ModelSelect'
import { catalogToOptions, ollamaToOptions, type ModelOption } from '../utils/model-options'
import type { ProviderStatus, ProviderConfigResponse, HealthStatus } from '../types'

export function ProvidersPage() {
  const [providers, setProviders] = useState<ProviderStatus[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedProviderId, setSelectedProviderId] = useState<string | null>(null)

  const reload = useCallback(() => {
    setLoading(true)
    setError(null)
    listProvidersStatus()
      .then(setProviders)
      .catch((err) => setError(extractErrorMessage(err)))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    reload()
  }, [reload])

  const selectedProvider = providers.find((p) => p.id === selectedProviderId) ?? null

  return (
    <div className="mx-auto w-full max-w-[1000px]">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="heading-display mb-1 text-3xl font-semibold text-[var(--color-text-heading)]">Providers</h1>
          <p className="text-sm text-[var(--color-text-muted)]">
            Configure AI providers, API keys, and model routing.
          </p>
        </div>
        <Button variant="secondary" onClick={reload} loading={loading}>
          <RefreshCw className="h-4 w-4" /> Refresh
        </Button>
      </div>

      {error && (
        <div className="mb-6 rounded-lg border border-[var(--color-danger)] bg-[rgba(244,63,94,0.08)] px-4 py-3 text-sm text-[var(--color-danger)]">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex flex-col gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-28 w-full rounded-xl" />
          ))}
        </div>
      ) : providers.length === 0 ? (
        <EmptyState
          icon={<Cpu className="h-10 w-10" />}
          title="No providers found"
          description="Make sure the daemon is running and try refreshing."
        />
      ) : (
        <div className="flex flex-col gap-4">
          {providers.map((p) => (
            <ProviderCard
              key={p.id}
              provider={p}
              selected={selectedProviderId === p.id}
              onConfigure={() => setSelectedProviderId(p.id)}
            />
          ))}
        </div>
      )}

      {selectedProvider && (
        <ProviderConfigPanel
          key={selectedProvider.id}
          provider={selectedProvider}
          onClose={() => setSelectedProviderId(null)}
          onChanged={reload}
        />
      )}
    </div>
  )
}

function ProviderCard({
  provider,
  selected,
  onConfigure,
}: {
  provider: ProviderStatus
  selected: boolean
  onConfigure: () => void
}) {
  return (
    <Card className={`p-5 transition-all ${selected ? 'border-[var(--color-primary)]' : ''}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
            <ProviderIcon id={provider.id} className="h-5 w-5 text-[var(--color-accent)]" />
          </div>
          <div className="flex flex-col">
            <span className="text-base font-semibold text-[var(--color-text-heading)]">{provider.label}</span>
            <span className="text-xs text-[var(--color-text-muted)]">{provider.source}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={provider.configured ? 'success' : 'warning'}>
            {provider.configured ? 'Configured' : 'Not configured'}
          </Badge>
          <button
            onClick={onConfigure}
            className={`flex h-8 w-8 items-center justify-center rounded-lg border transition-all cursor-pointer focus:outline-none focus:ring-1 focus:ring-[var(--color-primary)] ${
              selected
                ? 'border-[var(--color-primary)] bg-[var(--color-accent-soft)] text-[var(--color-accent)]'
                : 'border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-text-muted)] hover:border-[var(--color-border-strong)] hover:text-[var(--color-text-heading)]'
            }`}
            title="Configure"
            aria-label={`Configure ${provider.label}`}
          >
            <Settings2 className="h-4 w-4" />
          </button>
        </div>
      </div>
    </Card>
  )
}

/// Right-side drawer panel holding the credentials + model tier/routing
/// configuration for a single provider. Uses the same overlay + <aside>
/// pattern as the session event detail drawer.
function ProviderConfigPanel({
  provider,
  onClose,
  onChanged,
}: {
  provider: ProviderStatus
  onClose: () => void
  onChanged: () => void
}) {
  // DOM node pinned at the drawer's bottom edge. ProviderModelConfig
  // portals its Save action into it, making the button a direct flex child
  // of the drawer (below the scrollable form) so it always sits flush with
  // the modal's bottom, never inside the scrolling content.
  const [footerHost, setFooterHost] = useState<HTMLDivElement | null>(null)
  const [discoveryKey, setDiscoveryKey] = useState('')

  // Close on Escape, and lock the page scroll while the drawer is open so
  // scrolling over the backdrop cannot move the provider list behind it.
  // Same pattern as the session event inspector (SessionEventTimeline).
  // Locking body/html overflow only disables the page-level scrollbar -
  // the drawer's own scroll area keeps scrolling normally.
  useEffect(() => {
    const previousBodyOverflow = document.body.style.overflow
    const previousDocumentOverflow = document.documentElement.style.overflow
    document.body.style.overflow = 'hidden'
    document.documentElement.style.overflow = 'hidden'

    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => {
      window.removeEventListener('keydown', handler)
      document.body.style.overflow = previousBodyOverflow
      document.documentElement.style.overflow = previousDocumentOverflow
    }
  }, [onClose])

  return (
    <div className="fixed inset-0 z-[100]" role="dialog" aria-modal="true" aria-label={`${provider.label} configuration`}>
      <button
        type="button"
        aria-label="Close provider configuration"
        className="absolute inset-0 bg-black/45 backdrop-blur-[1px]"
        onClick={onClose}
      />
      <aside className="absolute inset-y-0 right-0 flex w-[min(560px,94vw)] flex-col border-l border-[var(--color-border)] bg-[var(--color-surface)] shadow-2xl">
        <div className="flex items-start gap-3 border-b border-[var(--color-border)] px-5 py-4">
          <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
            <ProviderIcon id={provider.id} className="h-5 w-5 text-[var(--color-accent)]" />
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant={provider.configured ? 'success' : 'warning'}>
                {provider.configured ? 'Configured' : 'Not configured'}
              </Badge>
              <span className="font-mono text-[11px] text-[var(--color-text-muted)]">{provider.source}</span>
            </div>
            <h3 className="mt-1 text-base font-semibold text-[var(--color-text-heading)]">{provider.label}</h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-[var(--color-text-muted)] hover:bg-[var(--color-surface-highlight)] hover:text-[var(--color-text-heading)] cursor-pointer"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="min-h-0 flex-1 overscroll-contain overflow-y-auto px-5 py-4">
          {/* Keying by provider id resets all panel state when switching providers. */}
          <ProviderCredentials key={`cred-${provider.id}`} provider={provider} onChanged={onChanged} onKeySaved={setDiscoveryKey} />
          <div className="mt-6 border-t border-[var(--color-border)] pt-4">
            <ProviderModelConfig key={`cfg-${provider.id}`} providerId={provider.id} footerHost={footerHost} discoveryKey={discoveryKey} onChanged={onChanged} />
          </div>
        </div>
        {/* Drawer footer. Empty for providers with nothing to save (Copilot);
            otherwise ProviderModelConfig portals the Save button here. */}
        <div ref={setFooterHost} className="shrink-0" />
      </aside>
    </div>
  )
}

function ProviderCredentials({
  provider,
  onChanged,
  onKeySaved,
}: {
  provider: ProviderStatus
  onChanged: () => void
  onKeySaved: (key: string) => void
}) {
  const [showKeyInput, setShowKeyInput] = useState(false)
  const [apiKey, setApiKey] = useState('')
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const [msgType, setMsgType] = useState<'success' | 'error'>('success')

  const handleAddKey = async () => {
    if (!apiKey.trim()) return
    setSaving(true)
    setMsg(null)
    try {
      const res = await addProviderCredential(provider.id, apiKey.trim())
      onKeySaved(apiKey.trim())
      setMsg(res.message)
      setMsgType('success')
      setApiKey('')
      setShowKeyInput(false)
      onChanged()
    } catch (err) {
      setMsg(extractErrorMessage(err))
      setMsgType('error')
    } finally {
      setSaving(false)
    }
  }

  const handleRemoveKey = async () => {
    setSaving(true)
    setMsg(null)
    try {
      const res = await removeProviderCredential(provider.id)
      onKeySaved('')
      setMsg(res.message)
      setMsgType('success')
      onChanged()
    } catch (err) {
      setMsg(extractErrorMessage(err))
      setMsgType('error')
    } finally {
      setSaving(false)
    }
  }

  const isOauthProvider = provider.id === 'copilot'
  const isOllama = provider.id === 'ollama'

  return (
    <div>
      <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-[var(--color-text-heading)]">
        <Key className="h-4 w-4" /> Credentials
      </h3>

      {isOauthProvider ? (
        <p className="text-xs text-[var(--color-text-muted)]">
          Copilot uses OAuth device flow. Run <code className="font-mono text-[var(--color-text)]">reincore provider add copilot</code> in the CLI to authenticate.
        </p>
      ) : isOllama ? (
        <p className="text-xs text-[var(--color-text-muted)]">
          Ollama runs locally without authentication. Optionally set an API key for Ollama Cloud.
        </p>
      ) : (
        <p className="mb-2 text-xs text-[var(--color-text-muted)]">
          Env var: <code className="font-mono text-[var(--color-text)]">{provider.env_var}</code>
        </p>
      )}

      {showKeyInput ? (
        <div className="mt-2 flex gap-2">
          <Input
            type="password"
            placeholder="Paste API key..."
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            className="flex-1"
          />
          <Button onClick={handleAddKey} loading={saving} disabled={!apiKey.trim()}>
            <Check className="h-4 w-4" /> Save
          </Button>
          <Button variant="ghost" onClick={() => { setShowKeyInput(false); setApiKey('') }}>
            Cancel
          </Button>
        </div>
      ) : (
        <div className="mt-2 flex gap-2">
          {!isOauthProvider && (
            <Button variant="secondary" onClick={() => setShowKeyInput(true)}>
              <Plus className="h-4 w-4" /> Add API key
            </Button>
          )}
          {provider.configured && !isOauthProvider && (
            <Button variant="danger" onClick={handleRemoveKey} loading={saving}>
              <Trash2 className="h-4 w-4" /> Remove key
            </Button>
          )}
        </div>
      )}

      {msg && (
        <div
          className={`mt-3 rounded-lg px-3 py-2 text-xs ${
            msgType === 'success'
              ? 'border border-[var(--color-success)] bg-[rgba(16,185,129,0.08)] text-[var(--color-success)]'
              : 'border border-[var(--color-danger)] bg-[rgba(244,63,94,0.08)] text-[var(--color-danger)]'
          }`}
        >
          {msg}
        </div>
      )}
    </div>
  )
}

function ProviderModelConfig({
  providerId,
  footerHost,
  discoveryKey,
  onChanged,
}: {
  providerId: string
  footerHost: HTMLDivElement | null
  discoveryKey: string
  onChanged: () => void
}) {
  const [config, setConfig] = useState<ProviderConfigResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [restarting, setRestarting] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const [msgType, setMsgType] = useState<'success' | 'error'>('success')
  const [catalogOptions, setCatalogOptions] = useState<ModelOption[]>([])
	const [customLoading, setCustomLoading] = useState(false)
	const [customDiscoveryMessage, setCustomDiscoveryMessage] = useState<string | null>(null)
  const [, setDaemon] = useState<HealthStatus | null>(null)

  const [standard, setStandard] = useState('')
	const [customName, setCustomName] = useState('')
  const [customBaseUrl, setCustomBaseUrl] = useState('')
  const [discoveryAttempt, setDiscoveryAttempt] = useState(0)
  const [lite, setLite] = useState('')
  const [pro, setPro] = useState('')
  const [ultra, setUltra] = useState('')

  // Routing policy overrides
  const [routingMode, setRoutingMode] = useState('')
  const [routingQuality, setRoutingQuality] = useState('')
  const [allowUpgrade, setAllowUpgrade] = useState(false)
  const [allowDegration, setAllowDegration] = useState(false)

  // Load provider config + model catalog + daemon health on mount / provider change.
  useEffect(() => {
    let cancelled = false
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true)
    setCatalogOptions([])

    Promise.all([getProviderConfig(providerId), fetchCatalog(providerId)])
      .then(([cfg, opts]) => {
        if (cancelled) return
        setConfig(cfg)
		setCustomName(cfg.name ?? '')
		setCustomBaseUrl(cfg.base_url ?? '')
        setCatalogOptions(opts)
        setStandard(cfg.standard ?? '')
        setLite(cfg.lite ?? '')
        setPro(cfg.pro ?? '')
        setUltra(cfg.ultra ?? '')
        setRoutingMode(cfg.routing.mode ?? '')
        setRoutingQuality(cfg.routing.minimum_quality ?? '')
        setAllowUpgrade(cfg.routing.allow_upgrade ?? false)
        setAllowDegration(cfg.routing.allow_degradation ?? false)
      })
      .catch(() => {})
      .finally(() => { if (!cancelled) setLoading(false) })

    // Fetch daemon health via HTTP so we know whether to restart after saving.
    getHealth()
      .then((health) => { if (!cancelled) setDaemon(health) })
      .catch(() => {})

    return () => { cancelled = true }
  }, [providerId])

  const handleSave = async () => {
    setSaving(true)
    setMsg(null)
    try {
      if (providerId === 'ollama' && !standard.trim()) {
        setMsg('Please select at least the standard Ollama model.')
        setMsgType('error')
        return
      }
	  if (providerId === 'custom' && (!customName.trim() || !customBaseUrl.trim() || !standard.trim())) {
		setMsg('Custom providers require a name, base URL, and standard model.')
		setMsgType('error')
		return
	  }

      const fields: Record<string, string> = {}
	  if (customName !== (config?.name ?? '')) fields.name = customName
	  if (customBaseUrl !== (config?.base_url ?? '')) fields.base_url = customBaseUrl
      if (standard !== (config?.standard ?? '')) fields.standard = standard
      if (lite !== (config?.lite ?? '')) fields.lite = lite
      if (pro !== (config?.pro ?? '')) fields.pro = pro
      if (ultra !== (config?.ultra ?? '')) fields.ultra = ultra

      // Routing policy overrides
      if (routingMode !== (config?.routing.mode ?? '')) fields.routing_mode = routingMode
      if (routingQuality !== (config?.routing.minimum_quality ?? '')) fields.routing_minimum_quality = routingQuality
      if (String(allowUpgrade) !== String(config?.routing.allow_upgrade ?? false)) fields.routing_allow_upgrade = String(allowUpgrade)
      if (String(allowDegration) !== String(config?.routing.allow_degradation ?? false)) fields.routing_allow_degradation = String(allowDegration)

      if (Object.keys(fields).length > 0) {
        const updated = await updateProviderConfig(providerId, fields)
        setConfig(updated)
        onChanged()
        setMsg('Configuration saved.')
        setMsgType('success')

        // Restart the daemon so the running process picks up the new config.
        // The daemon freezes provider configs at startup; without a restart,
        // model tier changes (e.g. switching the standard model) won't take
        // effect until the next full restart.
        //
        // Re-fetch health *now* rather than relying on the `daemon` state
        // captured on component mount - the daemon may have started/stopped
        // since then, and a stale null would silently skip the restart.
        let currentHealth: HealthStatus | null = null
        try {
          currentHealth = await getHealth()
          setDaemon(currentHealth)
        } catch {
          // Daemon is not running - nothing to restart.
        }

        if (currentHealth?.status === 'ok') {
          setRestarting(true)
          try {
            const isSingleProvider = currentHealth.mode === 'single_provider'
            const providerName = currentHealth.providers?.[0] ?? null
            const restarted = isSingleProvider && providerName
              ? await invoke<boolean>('start_daemon_with_provider', {
                  provider: providerName,
                  mode: currentHealth.api_mode || 'agentic',
                })
              : await invoke<boolean>('restart_daemon')
            if (restarted) {
              setMsg('Configuration saved and daemon restarted.')
            } else {
              setMsg('Configuration saved, but the daemon did not restart cleanly. Try restarting manually from Settings.')
              setMsgType('error')
            }
          } catch (restartErr) {
            const restartMsg = typeof restartErr === 'string'
              ? restartErr
              : (restartErr as { message?: string })?.message ?? String(restartErr)
            setMsg(`Configuration saved, but daemon restart failed: ${restartMsg}`)
            setMsgType('error')
          } finally {
            setRestarting(false)
          }
        } else {
          setMsg('Configuration saved. Start the daemon to apply changes.')
        }
      } else {
        setMsg('No changes to save.')
        setMsgType('success')
      }
    } catch (err) {
      setMsg(extractErrorMessage(err))
      setMsgType('error')
    } finally {
      setSaving(false)
    }
  }

  useEffect(() => {
    if (providerId !== 'custom' || loading) return
    let cancelled = false
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setCatalogOptions([])
    setCustomLoading(false)
    setCustomDiscoveryMessage(null)
    const baseUrl = customBaseUrl.trim().replace(/\/+$/, '')
    if (!baseUrl) return
    const timer = window.setTimeout(() => {
      setCustomLoading(true)
      listCustomProviderModels(discoveryKey || undefined, baseUrl)
        .then((models) => {
          if (cancelled) return
          setCatalogOptions(models.map((model) => ({ model_id: model.id, model_name: model.name || model.id })))
          setCustomDiscoveryMessage(models.length
            ? `${models.length} models available.`
            : 'No models were returned. Type each model ID exactly as the provider documents it.')
        })
        .catch(() => {
          if (cancelled) return
          setCatalogOptions([])
          setCustomDiscoveryMessage('Models could not be loaded. Check the API key and base URL, or type the exact provider model ID.')
        })
        .finally(() => { if (!cancelled) setCustomLoading(false) })
    }, 500)
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [providerId, loading, customBaseUrl, discoveryKey, discoveryAttempt])

  if (loading) return <Skeleton className="h-32 w-full rounded-lg" />
  const isCopilot = providerId === 'copilot'
  const isOllama = providerId === 'ollama'
  const isCustom = providerId === 'custom'

  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold text-[var(--color-text-heading)]">Model tiers</h3>
      <p className="mb-3 text-xs text-[var(--color-text-muted)]">
        {isCopilot
          ? 'Copilot models are determined by your GitHub Copilot subscription and selected model family. Use the CLI to configure the model family.'
          : isOllama
            ? 'Models are fetched live from your local Ollama server. You must select at least the Standard model before Ollama can be used.'
			: isCustom
				? 'Models are loaded from the provider’s /models endpoint when available. Otherwise, enter exact provider model IDs.'
				: 'Select models from the catalog, or type a custom model id. Empty fields use defaults or environment variables.'}
      </p>

	  {isCustom && (
		<div className="mb-4 flex flex-col gap-3 rounded-lg border border-[var(--color-border)] p-3">
			<Input
				label="Provider name"
				placeholder="OpenRouter"
				value={customName}
				onChange={(event) => setCustomName(event.target.value)}
			/>
			<Input
				label="Base URL"
				placeholder="https://openrouter.ai/api/v1"
				value={customBaseUrl}
				onChange={(event) => setCustomBaseUrl(event.target.value)}
			/>
			<div>
				<Button variant="secondary" onClick={() => setDiscoveryAttempt((attempt) => attempt + 1)} loading={customLoading}>
					<RefreshCw className="h-4 w-4" /> Load models
				</Button>
			</div>
			{customDiscoveryMessage && <p className="text-xs text-[var(--color-text-muted)]">{customDiscoveryMessage}</p>}
		</div>
	  )}

      {/* For Copilot, show model family info instead of tier dropdowns */}
      {isCopilot ? (
        <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] p-3 text-xs text-[var(--color-text-muted)]">
          Copilot model configuration is managed via the CLI:
          <br />
          <code className="font-mono text-[var(--color-text)]">reincore provider add copilot</code>
          <br />
          <br />
          This will guide you through the GitHub device flow and let you choose a model family (OpenAI or Anthropic).
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-3">
          <ModelSelect
            label="Standard *"
            value={standard}
            onChange={setStandard}
            options={catalogOptions}
            emptyLabel={isOllama ? 'Select a model...' : 'Auto (default)'}
          />
          <ModelSelect
            label="Lite"
            value={lite}
            onChange={setLite}
            options={catalogOptions}
            emptyLabel="Auto (derived)"
            placeholder="cheaper model id"
          />
          <ModelSelect
            label="Pro"
            value={pro}
            onChange={setPro}
            options={catalogOptions}
            emptyLabel="Auto (standard)"
            placeholder="stronger model id"
          />
          <ModelSelect
            label="Ultra"
            value={ultra}
            onChange={setUltra}
            options={catalogOptions}
            emptyLabel="Auto (standard)"
            placeholder="most capable model id"
          />
        </div>
      )}

      {/* Routing policy overrides */}
      {!isCopilot && (
        <>
          <h3 className="mt-6 mb-2 text-sm font-semibold text-[var(--color-text-heading)]">Routing policy</h3>
          <p className="mb-3 text-xs text-[var(--color-text-muted)]">
            Override the global routing policy for this provider. Empty fields inherit from the global policy.
          </p>
          <div className="grid grid-cols-2 gap-3">
            <ListSelect
              label="Mode"
              value={routingMode}
              onChange={setRoutingMode}
              options={[
                { value: '', title: 'Inherit from global' },
                { value: 'auto', title: 'Auto' },
                { value: 'custom', title: 'Custom' },
                { value: 'locked', title: 'Locked' },
              ]}
            />
            <ListSelect
              label="Minimum quality"
              value={routingQuality}
              onChange={setRoutingQuality}
              options={[
                { value: '', title: 'Inherit from global' },
                { value: 'economy', title: 'Economy' },
                { value: 'balanced', title: 'Balanced' },
                { value: 'high', title: 'High' },
              ]}
            />
          </div>
          <div className="mt-3 flex flex-col gap-3">
            <Toggle
              label="Allow upgrade"
              hint="Route to a stronger tier than standard when configured"
              checked={allowUpgrade}
              onChange={setAllowUpgrade}
            />
            <Toggle
              label="Allow degradation"
              hint="Let budget pressure downgrade the tier"
              checked={allowDegration}
              onChange={setAllowDegration}
            />
          </div>
        </>
      )}

      {config && (
        <div className="mt-4 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] p-3">
          <p className="mb-1 text-xs font-medium text-[var(--color-text-heading)]">Current routing policy</p>
          <div className="flex flex-wrap gap-2 text-xs text-[var(--color-text-muted)]">
            <span>Mode: <code className="font-mono text-[var(--color-text)]">{config.routing.mode}</code></span>
            <span>Quality: <code className="font-mono text-[var(--color-text)]">{config.routing.minimum_quality}</code></span>
            <span>Upgrade: <code className="font-mono text-[var(--color-text)]">{String(config.routing.allow_upgrade)}</code></span>
            <span>Degrade: <code className="font-mono text-[var(--color-text)]">{String(config.routing.allow_degradation)}</code></span>
          </div>
        </div>
      )}

      {/* The Save action renders through a portal into the drawer's footer
          node (owned by ProviderConfigPanel), which is a direct flex child
          of the drawer itself. That keeps the button pinned to the modal's
          bottom edge - always visible, never part of the scrolling form -
          while the save state stays local to this component. */}
      {!isCopilot && footerHost && createPortal(
        <div className="px-5 py-4">
          {msg && (
            <div
              className={`mb-2 rounded-lg px-3 py-2 text-xs ${
                msgType === 'success'
                  ? 'border border-[var(--color-success)] bg-[rgba(16,185,129,0.08)] text-[var(--color-success)]'
                  : 'border border-[var(--color-danger)] bg-[rgba(244,63,94,0.08)] text-[var(--color-danger)]'
              }`}
            >
              {msg}
            </div>
          )}
          <Button variant="secondary" onClick={handleSave} loading={saving || restarting}>
            <Check className="h-4 w-4" /> {restarting ? 'Restarting daemon...' : 'Save configuration'}
          </Button>
        </div>,
        footerHost,
      )}
    </div>
  )
}

/// Fetch the model catalog for a provider. For Copilot, return empty (handled
/// separately). For Ollama, fetch live from the local server. For all others,
/// fetch from the curated catalog endpoint.
async function fetchCatalog(providerId: string): Promise<ModelOption[]> {
  if (providerId === 'copilot') return []

  if (providerId === 'ollama') {
    try {
      const models = await listOllamaModels()
      return ollamaToOptions(models)
    } catch {
      return [] // Ollama server might not be running
    }
  }

  if (providerId === 'custom') return [] // Loaded by the cancellable discovery effect.

  try {
    const catalog = await getProviderCatalog(providerId)
    return catalogToOptions(catalog)
  } catch {
    return []
  }
}
