// Launch: start an integrated coding/agent app (e.g. Claude Code, Codex,
// Gemini CLI) routed through ReinCore. Mirrors `reincore launch <app>`
// from the CLI, but as a GUI: pick an app, pick a provider (defaults to
// the app's built-in default), pick a wire mode, and hit Launch. The
// backend negotiates with the running daemon - reusing a matching or
// multi-provider daemon, or asking (via the dialog below) before switching
// the single-provider daemon to a new provider.

import { useEffect, useState, useCallback } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { Rocket, RefreshCw, Terminal, CheckCircle2, XCircle, Search, FolderOpen, AlertTriangle } from 'lucide-react'
import { listProvidersStatus } from '../api'
import { Card, Badge, Button, Skeleton, ListSelect } from '../components/ui'
import { useLaunchAppsStore } from '../store/launchApps'
import type { ProviderStatus } from '../types'

/// Precheck state returned by `precheck_launch_app`.
type LaunchPrecheck =
  | { kind: 'start_single_provider'; provider: string }
  | { kind: 'reuse_multi_provider'; provider: string }
  | { kind: 'reuse_single_provider'; provider: string }
  | { kind: 'needs_switch_decision'; requested: string; current: string }

export function LaunchPage() {
  // Integrated-apps list is cached in a zustand store so navigating away
  // and back doesn't re-fetch (which does slow filesystem probes per app).
  const { apps, loading: appsLoading, error: appsError, fetchApps } = useLaunchAppsStore()

  const [providers, setProviders] = useState<ProviderStatus[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [selectedApp, setSelectedApp] = useState('')
  const [providerOverride, setProviderOverride] = useState('')
  const [mode, setMode] = useState('agentic')
  const [launchPath, setLaunchPath] = useState('')
  const [launching, setLaunching] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [resultType, setResultType] = useState<'success' | 'error'>('success')
  const [search, setSearch] = useState('')
  // The daemon-negotiation state for the currently selected app/provider,
  // refreshed on every relevant change so the Launch button can show what
  // will happen (start daemon / reuse daemon / needs switch).
  const [precheck, setPrecheck] = useState<LaunchPrecheck | null>(null)
  const [precheckError, setPrecheckError] = useState<string | null>(null)
  // Switch-conflict dialog state (shown when precheck.kind === 'needs_switch_decision').
  const [switchDialog, setSwitchDialog] = useState<LaunchPrecheck | null>(null)
  const [switching, setSwitching] = useState(false)

  const reload = useCallback(() => {
    setLoading(true)
    setError(null)

    // Load providers first - this is a fast HTTP call to the daemon and
    // unblocks the UI immediately (provider dropdown, config checks).
    listProvidersStatus()
      .then(setProviders)
      .catch(() => {})
      .finally(() => setLoading(false))

    // Load the integrated-apps list from the cache store. If the cache is
    // already populated this is a no-op; otherwise it fetches once.
    fetchApps()
  }, [fetchApps])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    reload()
  }, [reload])

  // Surface store errors in the page error banner.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (appsError) setError(appsError)
  }, [appsError])

  // Refresh the daemon-negotiation precheck whenever the inputs that affect
  // it change (selected app, provider override, wire mode).
  useEffect(() => {
    if (!selectedApp) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setPrecheck(null)
      setPrecheckError(null)
      return
    }
    let cancelled = false
    invoke<LaunchPrecheck>('precheck_launch_app', {
      app: selectedApp,
      provider: providerOverride || null,
      mode,
    })
      .then((p) => {
        if (!cancelled) {
          setPrecheck(p)
          setPrecheckError(null)
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setPrecheck(null)
          const msg = typeof e === 'string' ? e : (e as { message?: string })?.message ?? String(e)
          setPrecheckError(msg)
        }
      })
    return () => {
      cancelled = true
    }
  }, [selectedApp, providerOverride, mode])

  const doLaunch = async (switchDecision: 'use_requested' | 'use_current' | null) => {
    if (!selectedApp) return
    setLaunching(true)
    setResult(null)
    setError(null)
    try {
      const msg = await invoke<string>('launch_app', {
        app: selectedApp,
        provider: providerOverride || null,
        mode,
        path: launchPath.trim() || null,
        switchDecision,
      })
      setResult(msg)
      setResultType('success')
      // Refresh installed status after a launch attempt (force re-fetch).
      fetchApps(true)
    } catch (err: unknown) {
      const msg = typeof err === 'string' ? err : (err as { message?: string })?.message ?? String(err)
      // The backend reports an unresolved provider-switch conflict with a
      // CONFLICT: prefix; show the dialog instead of a plain error.
      if (typeof msg === 'string' && msg.startsWith('CONFLICT:')) {
        setSwitchDialog(precheck?.kind === 'needs_switch_decision' ? precheck : null)
        setResult(null)
      } else {
        setResult(msg)
        setResultType('error')
      }
    } finally {
      setLaunching(false)
      setSwitching(false)
    }
  }

  const handleLaunch = async () => {
    // If the precheck already knows a switch is needed, go straight to the
    // dialog instead of calling launch_app without a decision.
    if (precheck?.kind === 'needs_switch_decision') {
      setSwitchDialog(precheck)
      return
    }
    await doLaunch(null)
  }

  const handleSwitchChoice = async (choice: 'use_requested' | 'use_current' | 'cancel') => {
    if (choice === 'cancel') {
      setSwitchDialog(null)
      return
    }
    setSwitching(true)
    await doLaunch(choice)
    setSwitchDialog(null)
  }

  const filteredApps = apps.filter((a) => {
    if (!search.trim()) return true
    const q = search.toLowerCase()
    return a.id.toLowerCase().includes(q) || a.default_provider.toLowerCase().includes(q)
  })

  // Map a provider id (e.g. "openai") to its human-readable display name
  // (e.g. "OpenAI") using the providers list from the daemon. Falls back to
  // the id itself if the provider isn't in the list.
  const providerName = (id: string): string =>
    providers.find((p) => p.id === id)?.label ?? id.charAt(0).toUpperCase() + id.slice(1)

  // Determine whether the selected app's provider is configured.
  const selectedAppInfo = apps.find((a) => a.id === selectedApp)
  const effectiveProvider = providerOverride || selectedAppInfo?.default_provider || ''
  const providerInfo = providers.find((p) => p.id === effectiveProvider)
  const providerConfigured = providerInfo?.configured ?? !selectedAppInfo

  return (
    <div className="mx-auto w-full max-w-[1000px]">
      {/* Provider-switch conflict dialog: the daemon runs a different single
          provider than the requested one, or a multi-provider daemon that
          this app can't route through. Mirrors the CLI's 1/2/3 prompt. */}
      {switchDialog?.kind === 'needs_switch_decision' && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-lg rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-xl">
            <div className="mb-3 flex items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[rgba(245,158,11,0.12)]">
                <AlertTriangle className="h-5 w-5 text-[var(--color-warning)]" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-[var(--color-text-heading)]">
                  {switchDialog.current === 'multi_provider'
                    ? 'Restart ReinCore in single-provider mode?'
                    : 'Switch ReinCore\u2019s provider?'}
                </h2>
                <p className="mt-1 text-sm text-[var(--color-text-muted)]">
                  {switchDialog.current === 'multi_provider' ? (
                    <>
                      ReinCore is running in multi-provider mode, but{' '}
                      {selectedAppInfo?.display_name ?? 'this app'} can&apos;t route through a
                      multi-provider daemon. Launching it requires a single-provider daemon for{' '}
                      <strong>{providerName(switchDialog.requested)}</strong>.
                    </>
                  ) : (
                    <>
                      ReinCore is running in single-provider mode with{' '}
                      <strong>{providerName(switchDialog.current)}</strong>, but{' '}
                      {selectedAppInfo?.display_name ?? 'this app'} is configured for{' '}
                      <strong>{providerName(switchDialog.requested)}</strong>.
                    </>
                  )}
                </p>
              </div>
            </div>
            <p className="mb-5 rounded-lg bg-[var(--color-bg-subtle)] px-3 py-2 text-xs text-[var(--color-text-muted)]">
              {switchDialog.current === 'multi_provider'
                ? 'Switching stops the running daemon (in-flight requests are aborted) and restarts it in single-provider mode.'
                : `Switching stops the running daemon (any in-flight requests on ${providerName(switchDialog.current)} are aborted) and restarts it with ${providerName(switchDialog.requested)}.`}
            </p>
            <div className="flex flex-col gap-2">
              <Button
                variant="primary"
                loading={switching}
                disabled={switching}
                onClick={() => handleSwitchChoice('use_requested')}
                className="justify-center"
              >
                {switchDialog.current === 'multi_provider'
                  ? `Restart the daemon for ${providerName(switchDialog.requested)}`
                  : `Switch to ${providerName(switchDialog.requested)} (restart the daemon)`}
              </Button>
              {switchDialog.current !== 'multi_provider' && (
                <Button
                  variant="secondary"
                  loading={switching}
                  disabled={switching}
                  onClick={() => handleSwitchChoice('use_current')}
                  className="justify-center"
                >
                  Keep using {providerName(switchDialog.current)} for this launch
                </Button>
              )}
              <Button
                variant="secondary"
                disabled={switching}
                onClick={() => handleSwitchChoice('cancel')}
                className="justify-center"
              >
                Cancel
              </Button>
            </div>
          </div>
        </div>
      )}

      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="heading-display mb-1 text-3xl font-semibold text-[var(--color-text-heading)]">
            Launch
          </h1>
          <p className="text-sm text-[var(--color-text-muted)]">
            Start an integrated coding/agent app routed through a ReinCore provider.
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

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        {/* App catalog */}
        <div className="lg:col-span-3">
          <Card className="p-5">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
                <Terminal className="h-5 w-5 text-[var(--color-accent)]" />
              </div>
              <div>
                <span className="block text-base font-semibold text-[var(--color-text-heading)]">
                  Integrated programs
                </span>
                <span className="text-xs text-[var(--color-text-muted)]">
                  {appsLoading ? 'Loading...' : `${apps.length} supported · ${apps.filter((a) => a.installed).length} installed`}
                </span>
              </div>
            </div>

            {/* Search */}
            <div className="relative mb-4">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--color-text-muted)]" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search apps or providers..."
                className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] py-2 pl-9 pr-3 text-sm text-[var(--color-text-heading)] placeholder:text-[var(--color-text-muted)] outline-none transition-all focus:border-[var(--color-border-strong)] focus:ring-1 focus:ring-[var(--color-primary)]"
              />
            </div>

            {appsLoading ? (
              <div className="flex flex-col gap-2">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Skeleton key={i} className="h-14 w-full rounded-lg" />
                ))}
              </div>
            ) : filteredApps.length === 0 ? (
              <p className="py-8 text-center text-sm text-[var(--color-text-muted)]">
                No apps match "{search}".
              </p>
            ) : (
              <div className="flex flex-col gap-2 pr-1">
                {filteredApps.map((app) => (
                  <button
                    key={app.id}
                    onClick={() => {
                      setSelectedApp(app.id)
                      setProviderOverride('') // reset override when switching apps
                      setResult(null)
                    }}
                    className={`flex items-center justify-between rounded-lg border px-4 py-3 text-left transition-all cursor-pointer ${
                      selectedApp === app.id
                        ? 'border-[var(--color-primary)] bg-[var(--color-accent-soft)]'
                        : 'border-[var(--color-border)] hover:border-[var(--color-border-strong)] hover:bg-[var(--color-surface)]'
                    }`}
                  >
                    <div className="flex flex-col gap-0.5">
                      <span className="font-mono text-sm font-medium text-[var(--color-text-heading)]">
                        {app.display_name}
                      </span>
                      <span className="text-xs text-[var(--color-text-muted)]">
                        provider: {providerName(app.default_provider)}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {app.installed ? (
                        <Badge variant="success">
                          <CheckCircle2 className="mr-1 h-3 w-3" /> Installed
                        </Badge>
                      ) : (
                        <Badge variant="warning">
                          <XCircle className="mr-1 h-3 w-3" /> Not installed
                        </Badge>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </Card>
        </div>

        {/* Launch panel */}
        <div className="lg:col-span-2">
          <Card className="p-5">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
                <Rocket className="h-5 w-5 text-[var(--color-accent)]" />
              </div>
              <span className="text-base font-semibold text-[var(--color-text-heading)]">
                Launch configuration
              </span>
            </div>

            <div className="flex flex-col gap-4">
              {/* Selected app */}
              <div className="flex flex-col gap-1.5">
                <label className="text-sm font-medium text-[var(--color-text)]">App</label>
                <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-3.5 py-2.5 text-sm text-[var(--color-text-heading)]">
                  {selectedAppInfo?.display_name ?? (selectedApp || <span className="text-[var(--color-text-muted)]">Select an app →</span>)}
                </div>
              </div>

              {/* Provider override */}
              <ListSelect
                label="Provider (optional override)"
                value={providerOverride}
                onChange={(v) => {
                  setProviderOverride(v)
                  setResult(null)
                }}
                disabled={!selectedApp}
                placeholder={`Default (${selectedAppInfo ? providerName(selectedAppInfo.default_provider) : '-'})`}
                options={[
                  { value: '', title: `Default (${selectedAppInfo ? providerName(selectedAppInfo.default_provider) : '-'})` },
                  ...providers
                    .filter((p) => p.id !== 'copilot')
                    .map((p) => ({
                      value: p.id,
                      title: `${p.label} ${p.configured ? '' : '(not configured)'}`,
                      disabled: !p.configured,
                    })),
                ]}
              />

              {/* Wire mode */}
              <ListSelect
                label="API mode"
                value={mode}
                onChange={(v) => {
                  setMode(v)
                  setResult(null)
                }}
                options={[
                  { value: 'agentic', title: 'Agentic (Responses API)' },
                  { value: 'chat', title: 'Chat (Chat Completions)' },
                ]}
              />

              {/* Working directory */}
              <div className="flex flex-col gap-1.5">
                <label className="text-sm font-medium text-[var(--color-text)]">
                  Working directory <span className="text-[var(--color-text-muted)]">(optional)</span>
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={launchPath}
                    onChange={(e) => {
                      setLaunchPath(e.target.value)
                      setResult(null)
                    }}
                    placeholder="e.g. ~/projects/my-app"
                    className="flex-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-3.5 py-2.5 text-sm text-[var(--color-text-heading)] placeholder:text-[var(--color-text-muted)] outline-none transition-all focus:border-[var(--color-border-strong)] focus:ring-1 focus:ring-[var(--color-primary)]"
                  />
                  <button
                    type="button"
                    onClick={() => invoke<string | null>('pick_directory').then((p) => { if (p) { setLaunchPath(p); setResult(null) } }).catch(() => {})}
                    title="Browse for a directory"
                    className="flex shrink-0 items-center justify-center rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2.5 text-[var(--color-text-heading)] transition-all hover:border-[var(--color-border-strong)] hover:bg-[var(--color-surface-highlight)] cursor-pointer focus:outline-none focus:ring-1 focus:ring-[var(--color-primary)]"
                  >
                    <FolderOpen className="h-4 w-4" />
                  </button>
                </div>
                <p className="text-xs text-[var(--color-text-muted)]">
                  The app will launch from this directory. If it doesn't exist, it will be created. Leave empty to use the default.
                </p>
              </div>

              {/* Provider config warning */}
              {selectedApp && !providerConfigured && (
                <div className="rounded-lg border border-[var(--color-warning)] bg-[rgba(245,158,11,0.08)] px-3 py-2 text-xs text-[var(--color-warning)]">
                  {effectiveProvider === 'ollama' ? (
                    <>
                      Ollama needs a standard/default model before it can launch apps. Select one on
                      the <a href="#/providers" className="underline">Providers</a> page.
                    </>
                  ) : (
                    <>
                      Provider '{providerInfo?.label ?? effectiveProvider}' is not configured. Add an API
                      key on the <a href="#/providers" className="underline">Providers</a> page first.
                    </>
                  )}
                </div>
              )}

              {/* Not-installed hint */}
              {selectedAppInfo && !selectedAppInfo.installed && selectedAppInfo.install_hint && (
                <div className="rounded-lg border border-[var(--color-info)] bg-[rgba(56,189,248,0.08)] px-3 py-2 text-xs text-[var(--color-info)]">
                  <strong>{selectedAppInfo.display_name}</strong> doesn't appear to be installed.{' '}
                  {selectedAppInfo.install_hint}
                </div>
              )}

              {/* Daemon-negotiation hint: what will happen when Launch is
                  pressed, according to the currently running daemon. */}
              {selectedApp && precheck?.kind === 'reuse_multi_provider' && (
                <div className="rounded-lg border border-[var(--color-success)] bg-[rgba(16,185,129,0.08)] px-3 py-2 text-xs text-[var(--color-success)]">
                  Reusing the running multi-provider daemon. Requests will be routed to{' '}
                  <strong>{providerName(precheck.provider)}</strong> via the X-ReinCore-Provider
                  header - the daemon will not be restarted.
                </div>
              )}
              {selectedApp && precheck?.kind === 'reuse_single_provider' && (
                <div className="rounded-lg border border-[var(--color-success)] bg-[rgba(16,185,129,0.08)] px-3 py-2 text-xs text-[var(--color-success)]">
                  Reusing the running daemon for provider{' '}
                  <strong>{providerName(precheck.provider)}</strong> - it will not be restarted.
                </div>
              )}
              {selectedApp && precheck?.kind === 'needs_switch_decision' && (
                <div className="rounded-lg border border-[var(--color-warning)] bg-[rgba(245,158,11,0.08)] px-3 py-2 text-xs text-[var(--color-warning)]">
                  {precheck.current === 'multi_provider' ? (
                    <>
                      ReinCore is running in multi-provider mode, but this app can&apos;t route
                      through it. Launching will ask whether to restart the daemon in
                      single-provider mode for{' '}
                      <strong>{providerName(precheck.requested)}</strong>.
                    </>
                  ) : (
                    <>
                      ReinCore is running with provider{' '}
                      <strong>{providerName(precheck.current)}</strong>. Launching will ask
                      whether to switch to <strong>{providerName(precheck.requested)}</strong>{' '}
                      (daemon restart) or keep the current provider.
                    </>
                  )}
                </div>
              )}
              {selectedApp && precheckError && (
                <div className="rounded-lg border border-[var(--color-danger)] bg-[rgba(244,63,94,0.08)] px-3 py-2 text-xs text-[var(--color-danger)]">
                  {precheckError}
                </div>
              )}

              {/* Launch button */}
              <Button
                variant="primary"
                loading={launching}
                disabled={!selectedApp || !providerConfigured}
                onClick={handleLaunch}
                className="mt-2"
              >
                <Rocket className="h-4 w-4" /> Launch {selectedAppInfo?.display_name ?? 'app'}
              </Button>

              {/* Result */}
              {result && (
                <div
                  className={`whitespace-pre-wrap rounded-lg border px-3 py-2.5 text-xs ${
                    resultType === 'success'
                      ? 'border-[var(--color-success)] bg-[rgba(16,185,129,0.08)] text-[var(--color-success)]'
                      : 'border-[var(--color-danger)] bg-[rgba(244,63,94,0.08)] text-[var(--color-danger)]'
                  }`}
                >
                  {result}
                </div>
              )}
            </div>
          </Card>

          {/* CLI equivalents */}
          <Card className="mt-4 p-5">
            <span className="mb-2 block text-sm font-semibold text-[var(--color-text-heading)]">
              Prefer the terminal?
            </span>
            <p className="mb-3 text-xs text-[var(--color-text-muted)]">
              The same commands are available from the <code className="font-mono">reincore</code> CLI:
            </p>
            <div className="flex flex-col gap-1.5">
              <code className="block rounded-lg bg-[var(--color-bg-subtle)] px-3 py-2 font-mono text-xs text-[var(--color-text-heading)]">
                reincore launch claude
              </code>
              <code className="block rounded-lg bg-[var(--color-bg-subtle)] px-3 py-2 font-mono text-xs text-[var(--color-text-heading)]">
                reincore launch codex --mode agentic
              </code>
              <code className="block rounded-lg bg-[var(--color-bg-subtle)] px-3 py-2 font-mono text-xs text-[var(--color-text-heading)]">
                reincore launch gemini --provider gemini
              </code>
              <code className="block rounded-lg bg-[var(--color-bg-subtle)] px-3 py-2 font-mono text-xs text-[var(--color-text-heading)]">
                reincore launch aider --path ~/my-project
              </code>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}