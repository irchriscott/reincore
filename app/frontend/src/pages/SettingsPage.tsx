// Settings: local paths, compaction config, about. Replaces
// `reincore config`, `reincore migrate`, and informational CLI output.

import { useEffect, useState } from 'react'
import { invoke } from '@tauri-apps/api/core'
import {
  Database,
  FolderOpen,
  Info,
  Power,
  RefreshCw,
  RotateCcw,
  Cpu,
  Network,
  Minimize2,
  Waypoints,
  Wifi,
  Cable,
  ShieldCheck,
  Bell,
  Plus,
  Trash2,
} from 'lucide-react'
import {
  Card,
  Button,
  Badge,
  Toggle,
  ListSelect,
  Input,
} from '../components/ui'
import { listProvidersStatus } from '../api'
import { McpServers } from '../components/McpServers'
import type { ProviderStatus } from '../types'

interface DaemonStatus {
  running: boolean
  host: string
  port: number
  network_access: boolean
  mode: string | null
  provider: string | null
  api_mode: string | null
}

interface CompactionSettings {
  enabled: boolean
  mode: 'low' | 'mid' | 'max'
}

interface McpSettings {
  enabled: boolean
  config_path: string
  environment_override: boolean
}

interface TelemetrySettings {
  enabled: boolean
}

interface TelemetrySettingsWithHealth {
  enabled: boolean
  health: string
}

type SecretAction = 'allow' | 'redact' | 'block'

interface PrivacySettings {
  enabled: boolean
  provider_secret_action: SecretAction
  tool_secret_action: SecretAction
  persist_reasoning: boolean
  persist_provider_state: boolean
  retention_days: number
  max_persisted_string_bytes: number
  redaction_replacement: string
  additional_secret_values: string[]
  approved_providers: string[]
  environment_override: boolean
  config_path: string
}

// Mirrors reincore::notifications::LocalNotificationSettings, which is
// what `[notifications]` in ~/.reincore-io/config.toml deserializes into.
interface NotificationSettings {
  enabled: boolean
  run_outcomes: boolean
  agent_health: boolean
  budget: boolean
}

// The categories, in the order they are shown. Fewer than the cloud's: with
// no membership there is no access story to report locally.
const NOTIFICATION_CATEGORIES: {
  key: Exclude<keyof NotificationSettings, 'enabled'>
  label: string
  hint: string
}[] = [
  {
    key: 'run_outcomes',
    label: 'Run outcomes',
    hint: 'A run finishes, pauses at a checkpoint, resumes, or a batch completes. This is why you walk away from a long run.',
  },
  {
    key: 'budget',
    label: 'Budget',
    hint: 'Spend moves a session to a cheaper tier, which changes how it is routed.',
  },
  {
    key: 'agent_health',
    label: 'Agent health',
    hint: 'A run records a grounded finding, or the circuit breaker interrupts a loop. Diagnostic and fires more often, so it is off by default and better read in the app.',
  },
]

// Kept in step with LocalNotificationSettings::default() on the Rust side.
const DEFAULT_NOTIFICATION_SETTINGS: NotificationSettings = {
  enabled: true,
  run_outcomes: true,
  agent_health: false,
  budget: true,
}

const DEFAULT_PRIVACY_SETTINGS: PrivacySettings = {
  enabled: true,
  provider_secret_action: 'redact',
  tool_secret_action: 'block',
  persist_reasoning: false,
  persist_provider_state: false,
  retention_days: 30,
  max_persisted_string_bytes: 262144,
  redaction_replacement: '[REDACTED]',
  additional_secret_values: [],
  approved_providers: [],
  environment_override: false,
  config_path: '',
}

export function SettingsPage() {
  const [dbPath, setDbPath] = useState<string>('-')
  const [daemon, setDaemon] = useState<DaemonStatus | null>(null)
  const [pinging, setPinging] = useState(false)
  const [stopping, setStopping] = useState(false)
  const [restarting, setRestarting] = useState(false)
  const [compaction, setCompaction] = useState<CompactionSettings>({ enabled: false, mode: 'low' })
  const [compactionLoading, setCompactionLoading] = useState(true)
  const [compactionSaving, setCompactionSaving] = useState(false)
  const [compactionMessage, setCompactionMessage] = useState<string | null>(null)
  const [networkSaving, setNetworkSaving] = useState(false)
  const [networkError, setNetworkError] = useState<string | null>(null)

  const [privacy, setPrivacy] = useState<PrivacySettings>(DEFAULT_PRIVACY_SETTINGS)
  const [privacyLoading, setPrivacyLoading] = useState(true)
  const [privacySaving, setPrivacySaving] = useState(false)
  const [privacyMessage, setPrivacyMessage] = useState<string | null>(null)
  const [additionalSecrets, setAdditionalSecrets] = useState<string[]>([])
  const [approvedProviders, setApprovedProviders] = useState<string[]>([])

  const [notifications, setNotifications] = useState<NotificationSettings>(DEFAULT_NOTIFICATION_SETTINGS)
  const [notificationsLoading, setNotificationsLoading] = useState(true)
  const [notificationsSaving, setNotificationsSaving] = useState(false)
  const [notificationsMessage, setNotificationsMessage] = useState<string | null>(null)

  const [mcp, setMcp] = useState<McpSettings | null>(null)
  const [mcpSaving, setMcpSaving] = useState(false)
  const [mcpMessage, setMcpMessage] = useState<string | null>(null)

  // Telemetry settings
  const [telemetry, setTelemetry] = useState<TelemetrySettings>({ enabled: false })
  const [telemetryLoading, setTelemetryLoading] = useState(true)
  const [telemetrySaving, setTelemetrySaving] = useState(false)
  const [telemetryMessage, setTelemetryMessage] = useState<string | null>(null)
  const [telemetryHealth, setTelemetryHealth] = useState<string | null>(null)

  // Single-provider daemon start
  const [providers, setProviders] = useState<ProviderStatus[]>([])
  const [selectedProvider, setSelectedProvider] = useState('')
  const [providerMode, setProviderMode] = useState('agentic')
  const [startingProvider, setStartingProvider] = useState(false)
  const [switchingMulti, setSwitchingMulti] = useState(false)
  const [providerError, setProviderError] = useState<string | null>(null)
  const [providerSuccess, setProviderSuccess] = useState<string | null>(null)

  const refreshDaemon = async () => {
    setPinging(true)
    try {
      const status = await invoke<DaemonStatus>('daemon_status')
      setDaemon(status)
    } catch {
      setDaemon({ running: false, host: '127.0.0.1', port: 8989, network_access: false, mode: null, provider: null, api_mode: null })
    } finally {
      setPinging(false)
    }
  }

  useEffect(() => {
    invoke<McpSettings>('get_mcp_settings')
      .then(setMcp)
      .catch((error) => setMcpMessage(String(error)))
    invoke<string>('db_path').then(setDbPath).catch(() => {})
    invoke<CompactionSettings>('get_compression_settings')
      .then(setCompaction)
      .catch(() => setCompactionMessage('Could not load compaction settings.'))
      .finally(() => setCompactionLoading(false))
    invoke<PrivacySettings>('get_privacy_settings')
      .then((settings) => {
        setPrivacy(settings)
        setAdditionalSecrets(settings.additional_secret_values)
        setApprovedProviders(settings.approved_providers)
      })
      .catch(() => setPrivacyMessage('Could not load privacy settings.'))
      .finally(() => setPrivacyLoading(false))
    invoke<NotificationSettings>('get_notification_settings')
      .then(setNotifications)
      .catch(() => setNotificationsMessage('Could not load notification settings.'))
      .finally(() => setNotificationsLoading(false))
    invoke<TelemetrySettingsWithHealth>('get_telemetry_settings_with_health')
      .then((result) => {
        setTelemetry({ enabled: result.enabled })
        setTelemetryHealth(result.health)
      })
      .catch(() => setTelemetryMessage('Could not load telemetry settings.'))
      .finally(() => setTelemetryLoading(false))
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refreshDaemon()
    listProvidersStatus()
      .then((ps) => {
        setProviders(ps)
        // Provider/mode pre-selection is handled by the daemon-sync effect below.
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (!daemon) return
    const configured = providers.find((p) => p.id === daemon.provider)
    if (daemon.mode === 'single_provider' && daemon.provider && configured) {
      //eslint-disable-next-line react-hooks/set-state-in-effect
      setSelectedProvider(daemon.provider)
      if (daemon.api_mode) setProviderMode(daemon.api_mode)
    }
  }, [daemon, providers])

  const saveMcp = async (enabled: boolean) => {
    if (!mcp || mcpSaving) return
    setMcpSaving(true)
    try {
      setMcp(await invoke<McpSettings>('update_mcp_settings', { enabled }))
      setMcpMessage('Saved. Restart the daemon to apply this change.')
    } catch (error) {
      setMcpMessage(String(error))
    } finally {
      setMcpSaving(false)
    }
  }

  const handleStop = async () => {
    setStopping(true)
    try {
      await invoke<boolean>('stop_daemon')
    } finally {
      setStopping(false)
      await refreshDaemon()
    }
  }

  const handleRestart = async () => {
    setRestarting(true)
    try {
      // Preserve the current daemon mode on restart. If the daemon was
      // running in single-provider mode, re-start it with the same provider
      // and api_mode instead of falling back to multi-provider mode.
      if (daemon?.mode === 'single_provider' && daemon.provider) {
        await invoke<boolean>('start_daemon_with_provider', {
          provider: daemon.provider,
          mode: daemon.api_mode || 'agentic',
        })
      } else {
        await invoke<boolean>('restart_daemon')
      }
    } finally {
      setRestarting(false)
      await refreshDaemon()
    }
  }

  const handleStartWithProvider = async () => {
    if (!selectedProvider) return
    setStartingProvider(true)
    setProviderError(null)
    setProviderSuccess(null)
    try {
      const success = await invoke<boolean>('start_daemon_with_provider', {
        provider: selectedProvider,
        mode: providerMode,
      })
      if (success) {
        setProviderSuccess(`Daemon started with provider '${selectedProvider}' (${providerMode} mode).`)
      } else {
        setProviderError('Daemon started but /health is not responding yet. Check the logs.')
      }
      await refreshDaemon()
    } catch (err: unknown) {
      const msg = typeof err === 'string' ? err : (err as { message?: string })?.message ?? String(err)
      setProviderError(msg)
    } finally {
      setStartingProvider(false)
    }
  }

  const handleSwitchToMultiProvider = async () => {
    setSwitchingMulti(true)
    setProviderError(null)
    setProviderSuccess(null)
    try {
      await invoke<boolean>('restart_daemon')
      setProviderSuccess('Daemon restarted in multi-provider mode.')
      await refreshDaemon()
    } catch (err: unknown) {
      const msg = typeof err === 'string' ? err : (err as { message?: string })?.message ?? String(err)
      setProviderError(msg)
    } finally {
      setSwitchingMulti(false)
    }
  }

  const saveCompaction = async (next: CompactionSettings) => {
    setCompactionSaving(true)
    setCompactionMessage(null)
    try {
      const saved = await invoke<CompactionSettings>('update_compression_settings', {
        enabled: next.enabled,
        mode: next.mode,
      })
      setCompaction(saved)
      setCompactionMessage('Compaction settings saved.')
    } catch (err: unknown) {
      const msg = typeof err === 'string' ? err : (err as { message?: string })?.message ?? String(err)
      setCompactionMessage(msg)
    } finally {
      setCompactionSaving(false)
    }
  }

  const savePrivacy = async () => {
    const exactSecrets = Array.from(new Set(
      additionalSecrets.map((value) => value.trim()).filter(Boolean),
    ))
    if (exactSecrets.some((value) => value.length < 4)) {
      setPrivacyMessage('Each additional secret must contain at least four characters.')
      return
    }
    setPrivacySaving(true)
    setPrivacyMessage(null)
    try {
      const policy = {
        enabled: privacy.enabled,
        provider_secret_action: privacy.provider_secret_action,
        tool_secret_action: privacy.tool_secret_action,
        persist_reasoning: privacy.persist_reasoning,
        persist_provider_state: privacy.persist_provider_state,
        retention_days: privacy.retention_days,
        max_persisted_string_bytes: privacy.max_persisted_string_bytes,
        redaction_replacement: privacy.redaction_replacement,
        additional_secret_values: exactSecrets,
        approved_providers: approvedProviders,
      }
      const saved = await invoke<PrivacySettings>('update_privacy_settings', { policy })
      setPrivacy(saved)
      setAdditionalSecrets(saved.additional_secret_values)
      setApprovedProviders(saved.approved_providers)
      setPrivacyMessage(saved.environment_override
        ? 'Saved to config.toml. REINCORE_PRIVACY_POLICY still takes precedence; restart the daemon after changing or removing it.'
        : 'Saved to config.toml. Restart the daemon to apply these privacy settings.')
    } catch (error) {
      setPrivacyMessage(String(error))
    } finally {
      setPrivacySaving(false)
    }
  }

  // Saved on change rather than behind a Save button, because the poll loop
  // lives in this process and re-reads the settings every few seconds: the
  // switch really does take effect the moment it moves. The optimistic set is
  // put back if the write fails, so the switch never shows a state that is
  // not on disk.
  const saveNotifications = async (next: NotificationSettings) => {
    const previous = notifications
    setNotifications(next)
    setNotificationsSaving(true)
    setNotificationsMessage(null)
    try {
      setNotifications(await invoke<NotificationSettings>('update_notification_settings', {
        settings: next,
      }))
    } catch (error) {
      setNotifications(previous)
      setNotificationsMessage(String(error))
    } finally {
      setNotificationsSaving(false)
    }
  }

  const resetPrivacy = () => {
    setPrivacy({
      ...DEFAULT_PRIVACY_SETTINGS,
      environment_override: privacy.environment_override,
      config_path: privacy.config_path,
    })
    setAdditionalSecrets([])
    setApprovedProviders([])
    setPrivacyMessage('Defaults restored in the form. Save to update config.toml.')
  }

  const handleNetworkAccess = async (enabled: boolean) => {
    setNetworkSaving(true)
    setNetworkError(null)
    try {
      await invoke<DaemonStatus>('update_network_access', { enabled })
      const restarted = daemon?.mode === 'single_provider' && daemon.provider
        ? await invoke<boolean>('start_daemon_with_provider', {
            provider: daemon.provider,
            mode: daemon.api_mode || 'agentic',
          })
        : await invoke<boolean>('restart_daemon')
      if (!restarted) {
        setNetworkError('The setting was saved, but the daemon did not restart successfully.')
      }
      await refreshDaemon()
    } catch (err: unknown) {
      const msg = typeof err === 'string' ? err : (err as { message?: string })?.message ?? String(err)
      setNetworkError(msg)
      await refreshDaemon()
    } finally {
      setNetworkSaving(false)
    }
  }

  const saveTelemetry = async (next: TelemetrySettings) => {
    setTelemetrySaving(true)
    setTelemetryMessage(null)
    setTelemetryHealth(null)
    try {
      const saved = await invoke<TelemetrySettingsWithHealth>('update_telemetry_settings_with_health', {
        enabled: next.enabled,
      })
      setTelemetry({ enabled: saved.enabled })
      setTelemetryHealth(saved.health)
      setTelemetryMessage(saved.enabled
        ? 'Telemetry enabled. Traces will export to the bundled Jaeger container after the daemon restarts.'
        : 'Telemetry disabled. Restart the daemon to stop trace export.')
    } catch (err: unknown) {
      const msg = typeof err === 'string' ? err : (err as { message?: string })?.message ?? String(err)
      setTelemetryMessage(msg)
    } finally {
      setTelemetrySaving(false)
    }
  }

  const handleOpenTelemetryUi = async () => {
    try {
      await invoke('open_telemetry_ui')
    } catch (err: unknown) {
      const msg = typeof err === 'string' ? err : (err as { message?: string })?.message ?? String(err)
      setTelemetryMessage(msg)
    }
  }

  const daemonAddress = daemon ? `${daemon.host}:${daemon.port}` : '127.0.0.1:8989'

  return (
    <div className="mx-auto w-full max-w-[1000px]">
      <h1 className="heading-display mb-6 text-3xl font-semibold text-[var(--color-text-heading)]">Settings</h1>

      {/* Daemon status */}
      <Card className="mb-6 p-6">
        <div className="mb-3 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
            <RefreshCw className="h-5 w-5 text-[var(--color-accent)]" />
          </div>
          <span className="text-base font-semibold text-[var(--color-text-heading)]">Daemon</span>
        </div>
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2">
            <Badge variant={daemon === null ? 'default' : daemon.running ? 'success' : 'danger'}>
              {daemon === null ? 'Checking...' : daemon.running ? 'Running' : 'Stopped'}
            </Badge>
            <span className="text-xs text-[var(--color-text-muted)]">{daemonAddress}</span>
            {daemon?.running && daemon.mode === 'single_provider' && daemon.provider && (
              <Badge variant="primary">Provider: {daemon.provider}</Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button variant="secondary" loading={pinging} onClick={refreshDaemon}>
              Re-check
            </Button>
            <Button
              variant="secondary"
              loading={stopping}
              disabled={!daemon?.running}
              onClick={handleStop}
              title="Stop the background daemon"
            >
              <Power className="mr-1 h-4 w-4" /> Stop
            </Button>
            <Button
              variant="primary"
              loading={restarting}
              onClick={handleRestart}
              title={daemon?.running ? 'Restart the background daemon' : 'Start the background daemon'}
            >
              <RotateCcw className="mr-1 h-4 w-4" /> {daemon?.running ? 'Restart' : 'Start'}
            </Button>
          </div>
        </div>
        <p className="mt-3 text-xs text-[var(--color-text-muted)]">
          The daemon runs as a separate process and keeps serving even when the desktop app is closed.
        </p>

      </Card>

      {/* Single-provider daemon mode */}
      <Card className="mb-6 p-6">
          <div className="mb-3 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
              <Cpu className="h-5 w-5 text-[var(--color-accent)]" />
            </div>
            <span className="text-base font-semibold text-[var(--color-text-heading)]">Single-provider mode</span>
          </div>
          <p className="mb-4 text-xs text-[var(--color-text-muted)]">
            Restart the daemon restricted to one configured provider and API mode.
          </p>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <ListSelect
              label="Provider"
              value={selectedProvider}
              onChange={setSelectedProvider}
              placeholder="Select a provider..."
              options={[
                { value: '', title: 'Select a provider...' },
                ...providers.filter((p) => p.id !== 'copilot').map((p) => ({
                  value: p.id,
                  title: `${p.label} ${p.configured ? '' : '(not configured)'}`,
                  disabled: !p.configured,
                })),
              ]}
            />
            <ListSelect
              label="API mode"
              value={providerMode}
              onChange={setProviderMode}
              options={[
                { value: 'agentic', title: 'Agentic (Responses API)' },
                { value: 'chat', title: 'Chat (Chat Completions)' },
              ]}
            />
          </div>

          {selectedProvider && (() => {
            const p = providers.find((provider) => provider.id === selectedProvider)
            if (!p || p.configured) return null
            return (
              <div className="mt-3 rounded-lg border border-[var(--color-warning)] bg-[rgba(245,158,11,0.08)] px-3 py-2 text-xs text-[var(--color-warning)]">
                {p.id === 'ollama' ? (
                  <>
                    Ollama needs a standard/default model. Select one on the{' '}
                    <a href="#/providers" className="underline">Providers</a> page.
                  </>
                ) : (
                  <>
                    Provider '{p.label}' is not configured. Add an API key on the{' '}
                    <a href="#/providers" className="underline">Providers</a> page first.
                  </>
                )}
              </div>
            )
          })()}

          {providerError && (
            <div className="mt-3 rounded-lg border border-[var(--color-danger)] bg-[rgba(244,63,94,0.08)] px-3 py-2 text-xs text-[var(--color-danger)]">
              {providerError}
            </div>
          )}
          {providerSuccess && (
            <div className="mt-3 rounded-lg border border-[var(--color-success)] bg-[rgba(16,185,129,0.08)] px-3 py-2 text-xs text-[var(--color-success)]">
              {providerSuccess}
            </div>
          )}

          <div className="mt-4 flex flex-wrap items-center gap-3">
            <Button
              variant="primary"
              loading={startingProvider}
              disabled={!selectedProvider}
              onClick={handleStartWithProvider}
            >
              <RotateCcw className="mr-1 h-4 w-4" /> Start / Restart with provider
            </Button>
            {daemon?.running && daemon.mode === 'single_provider' && (
              <Button
                variant="secondary"
                loading={switchingMulti}
                onClick={handleSwitchToMultiProvider}
                title="Restart the daemon in multi-provider mode"
              >
                <Network className="mr-1 h-4 w-4" /> Switch to multi-provider
              </Button>
            )}
          </div>
      </Card>

      {/* Compaction */}
      <Card className="mb-6 p-6">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
            <Minimize2 className="h-5 w-5 text-[var(--color-accent)]" />
          </div>
          <div>
            <div className="text-base font-semibold text-[var(--color-text-heading)]">Context compaction</div>
            <div className="text-xs text-[var(--color-text-muted)]">Reduce long session context before provider requests.</div>
          </div>
        </div>
        <div className={compactionLoading ? 'pointer-events-none opacity-60' : ''}>
          <Toggle
            label="Enable compaction"
            hint="Applies to new requests without restarting the daemon."
            checked={compaction.enabled}
            onChange={(enabled) => saveCompaction({ ...compaction, enabled })}
          />
          <div className="mt-4">
            <ListSelect
              label="Compaction mode"
              value={compaction.mode}
              onChange={(v) => saveCompaction({ ...compaction, mode: v as CompactionSettings['mode'] })}
              disabled={!compaction.enabled || compactionSaving}
              options={[
                { value: 'low', title: 'Low', subtitle: 'Keep about 75% of recent context' },
                { value: 'mid', title: 'Mid', subtitle: 'Keep about 50% and summarize older turns' },
                { value: 'max', title: 'Max', subtitle: 'Retain only essential recent context' },
              ]}
            />
          </div>
        </div>
        {compactionMessage && (
          <p className="mt-3 text-xs text-[var(--color-text-muted)]">{compactionMessage}</p>
        )}
      </Card>

      {/* Network access */}
      <Card className="mb-6 p-6">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
            <Wifi className="h-5 w-5 text-[var(--color-accent)]" />
          </div>
          <div>
            <div className="text-base font-semibold text-[var(--color-text-heading)]">Local network access</div>
            <div className="text-xs text-[var(--color-text-muted)]">Allow other devices on this network to reach the ReinCore daemon.</div>
          </div>
        </div>
        <div className={networkSaving ? 'pointer-events-none opacity-60' : ''}>
          <Toggle
            label="Make the daemon available on the network"
            hint={daemon?.network_access ? `Listening on all interfaces at port ${daemon.port}.` : 'Restricted to this computer (127.0.0.1).'}
            checked={daemon?.network_access ?? false}
            onChange={handleNetworkAccess}
          />
        </div>
        {daemon?.network_access && (
          <p className="mt-3 rounded-lg border border-[var(--color-warning)] bg-[rgba(245,158,11,0.08)] px-3 py-2 text-xs text-[var(--color-warning)]">
            ReinCore's local API does not require authentication. Only enable this on a trusted network and allow port {daemon.port} through your firewall if needed.
          </p>
        )}
        {networkError && (
          <p className="mt-3 text-xs text-[var(--color-danger)]">{networkError}</p>
        )}
      </Card>

      <Card className="mb-6 p-6">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
            <Cable aria-hidden="true" className="h-5 w-5 text-[var(--color-accent)]" />
          </div>
          <div className="text-base font-semibold text-[var(--color-text-heading)]">MCP tools</div>
        </div>
        <fieldset disabled={!mcp || mcpSaving} className={!mcp || mcpSaving ? 'opacity-60' : ''}>
          <Toggle
            label="Enable MCP on daemon startup"
            hint="Connect MCP clients and configured upstream tools. Restart the daemon after changing this setting."
            checked={mcp?.enabled ?? false}
            onChange={saveMcp}
          />
        </fieldset>
        {mcp && (
          <div className="mt-4 text-xs text-[var(--color-text-muted)]">
            <p>Manage servers below or edit this file, then restart the daemon.</p>
            <p className="mt-2 break-all font-mono">{mcp.config_path}</p>
            {mcp.environment_override && (
              <p className="mt-2">REINCORE_MCP_CONFIG overrides the MCP path in config.toml.</p>
            )}
          </div>
        )}
        {mcpMessage && <p className="mt-3 text-xs text-[var(--color-text-muted)]">{mcpMessage}</p>}
        <McpServers />
      </Card>

      {/* Telemetry */}
      <Card className="mb-6 p-6">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
            <Waypoints className="h-5 w-5 text-[var(--color-accent)]" />
          </div>
          <div>
            <div className="text-base font-semibold text-[var(--color-text-heading)]">Telemetry traces</div>
            <div className="text-xs text-[var(--color-text-muted)]">Export OpenTelemetry traces to the bundled Jaeger container.</div>
          </div>
        </div>
        <div className={telemetryLoading || telemetrySaving ? 'pointer-events-none opacity-60' : ''}>
          <Toggle
            label="Enable OpenTelemetry traces"
            hint={telemetry.enabled ? 'Traces export to Jaeger on localhost:4317.' : 'No traces are exported.'}
            checked={telemetry.enabled}
            onChange={(enabled) => saveTelemetry({ ...telemetry, enabled })}
          />
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <Button
            variant="secondary"
            onClick={handleOpenTelemetryUi}
            disabled={!telemetry.enabled}
          >
            Open Jaeger UI
          </Button>
        </div>
        {telemetryMessage && (
          <p className="mt-3 text-xs text-[var(--color-text-muted)]">{telemetryMessage}</p>
        )}
        {telemetryHealth && !telemetryHealth.toLowerCase().startsWith('opentelemetry is disabled') && (
          <p className={`mt-3 text-xs ${telemetryHealth.toLowerCase().includes('reachable') || telemetryHealth.toLowerCase().includes('enabled')
            ? 'text-[var(--color-success)]'
            : 'text-[var(--color-danger)]'}`}>
            {telemetryHealth}
          </p>
        )}
      </Card>

      {/* Privacy boundaries */}
      <Card className="mb-6 p-6">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
            <ShieldCheck className="h-5 w-5 text-[var(--color-accent)]" />
          </div>
          <div>
            <div className="text-base font-semibold text-[var(--color-text-heading)]">Privacy boundaries</div>
            <div className="text-xs text-[var(--color-text-muted)]">Control sensitive data leaving ReinCore or entering durable storage.</div>
          </div>
        </div>

        <fieldset disabled={privacyLoading || privacySaving} className={privacyLoading || privacySaving ? 'opacity-60' : ''}>
          <Toggle
            label="Enforce privacy policy"
            hint={privacy.enabled ? 'Boundary checks and storage restrictions are active.' : 'Privacy checks are disabled.'}
            checked={privacy.enabled}
            onChange={(enabled) => setPrivacy({ ...privacy, enabled })}
          />

          <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <ListSelect
                label="Provider requests"
                value={privacy.provider_secret_action}
                onChange={(value) => setPrivacy({ ...privacy, provider_secret_action: value as SecretAction })}
                options={[
                  { value: 'redact', title: 'Redact detected secrets' },
                  { value: 'block', title: 'Block the entire request' },
                  { value: 'allow', title: 'Allow unchanged' },
                ]}
              />
              <p className="mt-2 text-xs text-[var(--color-text-muted)]">
                Choose how ReinCore handles detected secrets before sending a request to a provider.
              </p>
            </div>
            <div>
              <ListSelect
                label="External tool calls"
                value={privacy.tool_secret_action}
                onChange={(value) => setPrivacy({ ...privacy, tool_secret_action: value as SecretAction })}
                options={[
                  { value: 'block', title: 'Block calls with secrets' },
                  { value: 'redact', title: 'Redact detected secrets' },
                  { value: 'allow', title: 'Allow unchanged' },
                ]}
              />
              <p className="mt-2 text-xs text-[var(--color-text-muted)]">
                Redact replaces secrets with placeholders and continues the tool call. Block refuses calls containing secrets.
              </p>
            </div>
          </div>

          <div className="mt-5 flex flex-col gap-4">
            <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] p-3.5">
              <Toggle
                label="Persist provider reasoning"
                hint="Off by default so raw reasoning remains ephemeral. Enable only when durable reasoning is required."
                checked={privacy.persist_reasoning}
                onChange={(persist_reasoning) => setPrivacy({ ...privacy, persist_reasoning })}
              />
            </div>
            <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] p-3.5">
              <Toggle
                label="Persist provider replay state"
                hint="State is omitted whole when off; signed provider state is never redacted in place."
                checked={privacy.persist_provider_state}
                onChange={(persist_provider_state) => setPrivacy({ ...privacy, persist_provider_state })}
              />
            </div>
          </div>

          <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <Input
                label="Retention (days)"
                type="number"
                min={1}
                max={3650}
                value={privacy.retention_days}
                onChange={(event) => setPrivacy({ ...privacy, retention_days: Number(event.currentTarget.value) })}
              />
              <p className="mt-2 text-xs text-[var(--color-text-muted)]">
                Default maximum age for persisted events when no explicit retention cutoff is provided.
              </p>
            </div>
            <div>
              <Input
                label="Maximum persisted string (bytes)"
                type="number"
                min={1024}
                max={4194304}
                step={1024}
                value={privacy.max_persisted_string_bytes}
                onChange={(event) => setPrivacy({ ...privacy, max_persisted_string_bytes: Number(event.currentTarget.value) })}
              />
              <p className="mt-2 text-xs text-[var(--color-text-muted)]">
                Limits the size of each string written to durable storage, from 1 KiB to 4 MiB.
              </p>
            </div>
            <div className="sm:col-span-2">
              <Input
                label="Redaction replacement"
                value={privacy.redaction_replacement}
                onChange={(event) => setPrivacy({ ...privacy, redaction_replacement: event.currentTarget.value })}
              />
              <p className="mt-2 text-xs text-[var(--color-text-muted)]">
                Text substituted for each detected secret in provider requests, logs, and persisted content.
              </p>
            </div>
          </div>

          <div className="mt-5">
            <ListSelect
              multiple
              label="Approved providers"
              placeholder="All supported providers"
              value={approvedProviders}
              onChange={setApprovedProviders}
              options={providers.map((provider) => ({
                value: provider.id,
                title: provider.label,
                subtitle: `${provider.id}${provider.configured ? ' · configured' : ''}`,
              }))}
            />
            <p className="mt-2 text-xs text-[var(--color-text-muted)]">
              Limit outbound requests to the selected providers. Leave this empty to approve every supported provider.
            </p>
          </div>

          <div className="mt-5">
            <div>
              <div className="text-sm font-medium text-[var(--color-text)]">Additional exact secret values</div>
              <p className="mt-1 text-xs text-[var(--color-text-muted)]">
                Add exact values of at least four characters to augment built-in secret detection. Values are stored in plaintext in local config.toml, so prefer environment configuration for real credentials.
              </p>
            </div>
            {additionalSecrets.length > 0 && (
              <div className="mt-3 flex flex-col gap-3">
                {additionalSecrets.map((secret, index) => (
                  <div key={`privacy-secret-${index}`} className="flex flex-col gap-2 sm:flex-row sm:items-end">
                    <div className="min-w-0 flex-1">
                      <Input
                        className="w-full font-mono"
                        label={`Secret ${index + 1}`}
                        type="password"
                        minLength={4}
                        autoComplete="new-password"
                        placeholder="Enter an exact secret value"
                        value={secret}
                        onChange={(event) => {
                          const nextValue = event.currentTarget.value
                          setAdditionalSecrets((values) => values.map((value, valueIndex) => (
                            valueIndex === index ? nextValue : value
                          )))
                        }}
                      />
                    </div>
                    <Button
                      type="button"
                      variant="secondary"
                      className="w-full sm:w-auto"
                      aria-label={`Remove secret ${index + 1}`}
                      title={`Remove secret ${index + 1}`}
                      onClick={() => setAdditionalSecrets((values) => values.filter((_, valueIndex) => valueIndex !== index))}
                    >
                      <Trash2 className="h-4 w-4" /> Remove
                    </Button>
                  </div>
                ))}
              </div>
            )}
            <Button
              type="button"
              variant="secondary"
              className="mt-3"
              onClick={() => setAdditionalSecrets((values) => [...values, ''])}
            >
              <Plus className="h-4 w-4" /> Add secret
            </Button>
          </div>
        </fieldset>

        {(privacy.provider_secret_action === 'allow' || privacy.tool_secret_action === 'allow' || privacy.persist_reasoning || privacy.persist_provider_state || !privacy.enabled) && (
          <p className="mt-4 rounded-lg border border-[var(--color-warning)] bg-[rgba(245,158,11,0.08)] px-3 py-2 text-xs text-[var(--color-warning)]">
            One or more selections reduce the default privacy protections. Review them before saving.
          </p>
        )}
        {privacy.environment_override && (
          <p className="mt-4 rounded-lg border border-[var(--color-warning)] bg-[rgba(245,158,11,0.08)] px-3 py-2 text-xs text-[var(--color-warning)]">
            REINCORE_PRIVACY_POLICY currently overrides the settings saved here.
          </p>
        )}

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <Button variant="primary" loading={privacySaving} disabled={privacyLoading} onClick={savePrivacy}>
            Save privacy settings
          </Button>
          <Button variant="secondary" disabled={privacyLoading || privacySaving} onClick={resetPrivacy}>
            Restore defaults
          </Button>
        </div>
        {privacyMessage && (
          <p className="mt-3 text-xs text-[var(--color-text-muted)]">{privacyMessage}</p>
        )}
      </Card>

      {/* Notifications */}
      <Card className="mb-6 p-6">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
            <Bell className="h-5 w-5 text-[var(--color-accent)]" />
          </div>
          <div>
            <div className="text-base font-semibold text-[var(--color-text-heading)]">Notifications</div>
            <div className="text-xs text-[var(--color-text-muted)]">Choose which local events are worth an OS toast. Nothing is sent anywhere.</div>
          </div>
        </div>

        <fieldset disabled={notificationsLoading || notificationsSaving} className={notificationsLoading || notificationsSaving ? 'opacity-60' : ''}>
          <Toggle
            label="Desktop notifications"
            hint={notifications.enabled
              ? 'The app watches the local event log and shows a toast for the categories below.'
              : 'No toasts are shown. Everything still appears in the app.'}
            checked={notifications.enabled}
            onChange={(enabled) => saveNotifications({ ...notifications, enabled })}
          />

          <div className={`mt-5 flex flex-col gap-4 ${notifications.enabled ? '' : 'pointer-events-none opacity-50'}`}>
            {NOTIFICATION_CATEGORIES.map((category) => (
              <div key={category.key} className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] p-3.5">
                <Toggle
                  label={category.label}
                  hint={category.hint}
                  checked={notifications[category.key]}
                  onChange={(value) => saveNotifications({ ...notifications, [category.key]: value })}
                />
              </div>
            ))}
          </div>
        </fieldset>

        <p className="mt-5 text-xs leading-relaxed text-[var(--color-text-muted)]">
          A toast says what happened and nothing more. Titles and bodies are fixed strings chosen by
          the event type, so no prompt or response text ever reaches the notification centre &mdash; where
          it would outlive the run and sit outside the retention rules above.
        </p>
        <p className="mt-2 text-xs leading-relaxed text-[var(--color-text-muted)]">
          Saved to <code className="font-mono">[notifications]</code> in config.toml and applied straight away.
          Unlike the privacy policy, this needs no daemon restart: the watcher runs in the app.
        </p>
        {notificationsMessage && (
          <p className="mt-3 text-xs text-[var(--color-text-muted)]">{notificationsMessage}</p>
        )}
      </Card>

      {/* Paths */}
      <Card className="mb-6 p-6">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
            <Database className="h-5 w-5 text-[var(--color-accent)]" />
          </div>
          <span className="text-base font-semibold text-[var(--color-text-heading)]">Data paths</span>
        </div>
        <div className="flex flex-col gap-3">
          <div>
            <div className="mb-1 flex items-center gap-2 text-sm font-medium text-[var(--color-text)]">
              <FolderOpen className="h-4 w-4" /> Configuration file
            </div>
            <code className="block rounded-lg bg-[var(--color-bg-subtle)] px-3 py-2 font-mono text-xs text-[var(--color-text-heading)] break-all">
              {privacy.config_path || '-'}
            </code>
          </div>
          <div>
            <div className="mb-1 flex items-center gap-2 text-sm font-medium text-[var(--color-text)]">
              <Database className="h-4 w-4" /> SQLite database
            </div>
            <code className="block rounded-lg bg-[var(--color-bg-subtle)] px-3 py-2 font-mono text-xs text-[var(--color-text-heading)] break-all">
              {dbPath}
            </code>
          </div>
        </div>
      </Card>

      {/* About */}
      <Card className="p-6">
        <div className="mb-3 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-accent-soft)]">
            <Info className="h-5 w-5 text-[var(--color-accent)]" />
          </div>
          <span className="text-base font-semibold text-[var(--color-text-heading)]">About</span>
        </div>
        <p className="text-sm text-[var(--color-text)]">
          ReinCore is a session-aware gateway for AI agents. The desktop app manages the engine
          and provides a visual interface over its HTTP API. The CLI (<code className="font-mono">reincore</code>)
          remains available and talks to the same engine.
        </p>
        <p className="mt-2 text-xs text-[var(--color-text-muted)]">Version 0.1.0 · Desktop edition</p>
      </Card>
    </div>
  )
}
