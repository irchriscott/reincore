// McpAuthSection: the Authentication part of the local MCP server form. The
// type decides which settings and secret are asked for. Secrets are stored in
// mcp.json or read from an environment variable; OAuth sign-in opens the
// system browser and the finished sign-in is saved beside mcp.json.

import { useEffect, useRef, useState, type ChangeEvent } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { LogIn, Plus, Search, Trash2, Upload } from 'lucide-react'
import { Badge, Button, Input, ListSelect, Radio } from './ui'
import {
  AUTH_OPTIONS,
  GOOGLE_CLIENT_CREDENTIALS,
  GOOGLE_CLOUD_SCOPE,
  MAX_HEADERS,
  awsSigningFromUrl,
  clientCredentialsUnsupported,
  secretLabel,
  splitScopes,
  usesGoogleClientCredentials,
  type AuthDraft,
  type AuthorizationMetadata,
  type FinishedSignIn,
  type LocalAuth,
} from '../utils/mcp-auth'

const area = 'w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] p-3 font-mono text-xs text-[var(--color-text)]'
const compact = 'min-w-0 flex-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-3 py-2 font-mono text-xs text-[var(--color-text-heading)] outline-none focus:border-[var(--color-border-strong)] focus:ring-1 focus:ring-[var(--color-primary)]'

function host(value: string | undefined): string {
  if (!value) return ''
  try { return new URL(value).host } catch { return value }
}

export function McpAuthSection({ draft, onChange, url, timeoutMs, serverName, savedOauth, authorization, onAuthorization, onError, busy }: {
  draft: AuthDraft
  onChange: (update: (draft: AuthDraft) => AuthDraft) => void
  url: string
  timeoutMs: number
  // The saved server being edited, whose stored sign-in and client secret
  // may be reused.
  serverName: string | null
  // Settings of the sign-in stored for that server, when it still applies.
  savedOauth: LocalAuth | null
  authorization: AuthorizationMetadata | null
  onAuthorization: (metadata: AuthorizationMetadata | null) => void
  onError: (message: string) => void
  busy: boolean
}) {
  const [signingIn, setSigningIn] = useState(false)
  const [detecting, setDetecting] = useState(false)
  const keyFile = useRef<HTMLInputElement>(null)
  const signingInRef = useRef(false)
  const endpoint = url.trim()
  const disabled = busy || signingIn

  // A sign-in still waiting for the browser is abandoned with the form.
  useEffect(() => () => { if (signingInRef.current) void invoke('cancel_mcp_sign_in') }, [])

  const set = <K extends keyof AuthDraft>(key: K, value: AuthDraft[K]) => onChange(d => ({ ...d, [key]: value }))

  const changeType = (type: string) => {
    onError('')
    onChange(d => {
      const next: AuthDraft = { ...d, type: type as AuthDraft['type'] }
      if (type === 'aws_sigv4' && (!d.region || !d.service)) {
        const signing = awsSigningFromUrl(endpoint)
        if (signing) { next.region = d.region || signing.region; next.service = d.service || signing.service }
      }
      if (type === 'google_service_account' && !d.scopes.trim()) next.scopes = GOOGLE_CLOUD_SCOPE
      // Both OAuth types take a client secret, so switching to sign-in keeps it.
      if (type === 'oauth' && d.type === 'client_credentials' && !d.clientSecret && d.secretSource === 'inline')
        next.clientSecret = d.secret
      if (type === 'oauth' && !d.scopes.trim() && authorization)
        next.scopes = (authorization.challenge_scopes.length ? authorization.challenge_scopes : authorization.scopes_supported).join(' ')
      return next
    })
  }

  async function detect() {
    if (!endpoint) { onError('Enter the endpoint URL first.'); return }
    setDetecting(true)
    onError('')
    try {
      const metadata = await invoke<AuthorizationMetadata>('mcp_oauth_metadata', { url: endpoint, timeoutMs })
      onAuthorization(metadata)
      onChange(d => ({
        ...d,
        tokenEndpoint: d.tokenEndpoint || metadata.token_endpoint || '',
        scopes: d.scopes || (d.type === 'oauth'
          ? (metadata.challenge_scopes.length ? metadata.challenge_scopes : metadata.scopes_supported).join(' ')
          : ''),
      }))
      if (!metadata.token_endpoint)
        onError('This server does not publish OAuth details. Use a token, API key or another type from its documentation.')
    } catch (error) {
      onError(`Could not read the server's sign-in details: ${String(error)}`)
    } finally {
      setDetecting(false)
    }
  }

  async function signIn() {
    if (!endpoint) { onError('Enter the endpoint URL before signing in.'); return }
    setSigningIn(true)
    signingInRef.current = true
    onError('')
    try {
      const finished = await invoke<FinishedSignIn>('sign_in_mcp_server', {
        url: endpoint,
        clientId: draft.clientId.trim() || null,
        clientSecret: draft.clientId.trim() && draft.clientSecret ? draft.clientSecret : null,
        scopes: draft.scopes.trim() ? splitScopes(draft.scopes) : null,
        timeoutMs,
        serverName,
      })
      onChange(d => ({
        ...d,
        signIn: { ...finished, url: endpoint },
        clientId: finished.auth.type === 'oauth' ? finished.auth.client_id : d.clientId,
        scopes: finished.auth.type === 'oauth' ? (finished.auth.scopes ?? []).join(' ') : d.scopes,
      }))
    } catch (error) {
      const message = String(error)
      if (message !== 'Sign-in cancelled.') onError(message)
    } finally {
      setSigningIn(false)
      signingInRef.current = false
    }
  }

  async function readKeyFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    if (file.size > 16384) { onError('That file is too large to be a service account key.'); return }
    const text = (await file.text()).trim()
    onChange(d => ({ ...d, secretSource: 'inline', secret: text }))
  }

  const secretFields = (placeholder: string) => (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-4">
        <Radio name="mcp-secret-source" label="Store in mcp.json" size="sm" checked={draft.secretSource === 'inline'}
          disabled={disabled} onChange={() => set('secretSource', 'inline')} />
        <Radio name="mcp-secret-source" label="Environment variable" size="sm" checked={draft.secretSource === 'env'}
          disabled={disabled} onChange={() => set('secretSource', 'env')} />
      </div>
      {draft.secretSource === 'env' ? (
        <Input label={`${secretLabel(draft.type)} environment variable`} placeholder="MCP_API_KEY" autoComplete="off"
          value={draft.secretEnv} disabled={disabled} onChange={event => set('secretEnv', event.target.value)} />
      ) : draft.type === 'google_service_account' ? (
        <div>
          <div className="mb-1.5 flex items-center justify-between gap-2">
            <label htmlFor="mcp-service-account-key" className="text-sm font-medium text-[var(--color-text)]">Service account key (JSON)</label>
            <Button type="button" variant="ghost" className="px-2 py-1 text-xs" disabled={disabled} onClick={() => keyFile.current?.click()}>
              <Upload className="h-3.5 w-3.5" /> Upload key file
            </Button>
            <input ref={keyFile} type="file" accept="application/json,.json" className="hidden" onChange={readKeyFile} />
          </div>
          <textarea id="mcp-service-account-key" className={area} rows={6} spellCheck={false} autoComplete="off"
            placeholder='{"type": "service_account", "client_email": "...", "private_key": "..."}'
            value={draft.secret} disabled={disabled} onChange={event => set('secret', event.target.value)} />
        </div>
      ) : (
        <Input label={secretLabel(draft.type)} type="password" autoComplete="off" placeholder={placeholder}
          value={draft.secret} disabled={disabled} onChange={event => set('secret', event.target.value)} />
      )}
      <p className="text-xs text-[var(--color-text-muted)]">
        {draft.secretSource === 'env'
          ? 'Read when the daemon starts; set it in the environment the daemon runs in.'
          : 'Stored in plain text in your local mcp.json file.'}
      </p>
    </div>
  )

  const scopesInput = (placeholder = 'e.g. files.read files.write') => (
    <Input label="Scopes (space-separated)" placeholder={placeholder} value={draft.scopes} disabled={disabled}
      onChange={event => onChange(d => ({ ...d, scopes: event.target.value, signIn: d.type === 'oauth' ? null : d.signIn }))} />
  )

  return (
    <div className="space-y-3 rounded-lg border border-[var(--color-border)] p-3">
      <ListSelect label="Authentication" value={draft.type} disabled={disabled} options={AUTH_OPTIONS} onChange={changeType} />

      {draft.type === 'none' && (
        <p className="text-xs text-[var(--color-text-muted)]">No credentials are sent. If loading tools reports that the server needs authentication, choose the type it asks for.</p>
      )}

      {draft.type === 'bearer' && secretFields('Upstream MCP bearer token')}

      {draft.type === 'header' && <>
        <Input label="Header name" placeholder="x-api-key" value={draft.headerName} disabled={disabled}
          onChange={event => set('headerName', event.target.value)} />
        {secretFields('API key')}
      </>}

      {draft.type === 'basic' && <>
        <Input label="Username" placeholder="you@example.com" autoComplete="off" value={draft.username} disabled={disabled}
          onChange={event => set('username', event.target.value)} />
        {secretFields('API token')}
      </>}

      {draft.type === 'oauth' && <>
        <p className="rounded-lg bg-[var(--color-bg-subtle)] p-3 text-xs text-[var(--color-text-muted)]">
          {authorization?.token_endpoint
            ? <>Signs in with <strong className="text-[var(--color-text-heading)]">{host(authorization.issuer) || 'the provider'}</strong> in your browser. {authorization.dynamic_registration
              ? 'ReinCore registers itself, so a client ID is optional.'
              : 'This provider needs your own OAuth client: create a Desktop app client and enter its client ID and secret.'}</>
            : 'ReinCore opens your browser to sign in and receives the result on 127.0.0.1. Use Detect to check whether the provider needs your own OAuth client (a Desktop app client).'}
        </p>
        <Input label={authorization?.token_endpoint && !authorization.dynamic_registration ? 'Client ID' : 'Client ID (optional)'}
          autoComplete="off" placeholder="From your OAuth client" value={draft.clientId} disabled={disabled}
          onChange={event => onChange(d => ({ ...d, clientId: event.target.value, signIn: null }))} />
        <Input label={savedOauth && draft.clientId.trim() === (savedOauth.type === 'oauth' ? savedOauth.client_id : '') ? 'Client secret (leave blank to keep current)' : 'Client secret (optional)'}
          type="password" autoComplete="off" placeholder={draft.clientId.trim() ? 'From your OAuth client' : 'Needs a client ID'}
          value={draft.clientSecret} disabled={disabled || !draft.clientId.trim()}
          onChange={event => onChange(d => ({ ...d, clientSecret: event.target.value, signIn: null }))} />
        {scopesInput()}
        {authorization && authorization.scopes_supported.length > 0 && (
          <p className="break-words text-xs text-[var(--color-text-muted)]">
            Supported: {authorization.scopes_supported.slice(0, 12).join(' ')}{authorization.scopes_supported.length > 12 ? ' ...' : ''}
          </p>
        )}
        <div className="flex flex-wrap items-center gap-2">
          {signingIn ? <>
            <Button type="button" loading disabled>Waiting for the browser...</Button>
            <Button type="button" variant="secondary" onClick={() => void invoke('cancel_mcp_sign_in')}>Cancel</Button>
          </> : (
            <Button type="button" variant={draft.signIn || savedOauth ? 'secondary' : 'primary'} disabled={busy || !endpoint} onClick={signIn}>
              <LogIn className="h-4 w-4" /> {draft.signIn || savedOauth ? 'Sign in again' : 'Sign in'}
            </Button>
          )}
          <Button type="button" variant="ghost" loading={detecting} disabled={disabled || !endpoint} onClick={detect}>
            {!detecting && <Search className="h-4 w-4" />} Detect
          </Button>
          {draft.signIn
            ? <Badge variant="success">Signed in · save within an hour</Badge>
            : savedOauth?.type === 'oauth' ? <Badge variant="info">Signed in via {host(savedOauth.issuer ?? savedOauth.token_endpoint)}</Badge> : null}
        </div>
        <p className="text-xs text-[var(--color-text-muted)]">Sign-in tokens are kept in mcp.credentials.json beside mcp.json, not in the server JSON, and refresh automatically.</p>
      </>}

      {draft.type === 'client_credentials' && <>
        {usesGoogleClientCredentials(draft)
          ? <p role="alert" className="rounded-lg bg-[rgba(245,158,11,0.12)] p-3 text-xs text-[var(--color-warning)]">{GOOGLE_CLIENT_CREDENTIALS}</p>
          : clientCredentialsUnsupported(authorization?.grant_types_supported) && (
            <p role="alert" className="rounded-lg bg-[rgba(245,158,11,0.12)] p-3 text-xs text-[var(--color-warning)]">
              This server's authorization server does not list client credentials among the ways it issues tokens.
              {authorization?.authorization_endpoint ? ' Choose OAuth sign-in instead.' : ''}
            </p>
          )}
        <div className="flex items-end gap-2">
          <div className="min-w-0 flex-1">
            <Input label="Token endpoint" type="url" placeholder="https://auth.example.com/oauth/token" value={draft.tokenEndpoint}
              disabled={disabled} onChange={event => set('tokenEndpoint', event.target.value)} />
          </div>
          <Button type="button" variant="secondary" loading={detecting} disabled={disabled || !endpoint} onClick={detect}>
            {!detecting && <Search className="h-4 w-4" />} Detect
          </Button>
        </div>
        <Input label="Client ID" autoComplete="off" value={draft.clientId} disabled={disabled} onChange={event => set('clientId', event.target.value)} />
        {secretFields('From your OAuth client')}
        {scopesInput()}
        <Input label="Audience (optional)" value={draft.audience} disabled={disabled} onChange={event => set('audience', event.target.value)} />
        <ListSelect label="Client authentication" value={draft.clientAuth} disabled={disabled}
          onChange={value => set('clientAuth', value as AuthDraft['clientAuth'])}
          options={[
            { value: 'client_secret_basic', title: 'HTTP Basic', subtitle: 'Client ID and secret in the Authorization header (most common)' },
            { value: 'client_secret_post', title: 'Form fields', subtitle: 'Client ID and secret in the request body' },
          ]} />
      </>}

      {draft.type === 'google_service_account' && <>
        {secretFields('')}
        {scopesInput(GOOGLE_CLOUD_SCOPE)}
        <Input label="Act as user (optional)" placeholder="user@your-domain.com" value={draft.subject} disabled={disabled}
          onChange={event => set('subject', event.target.value)} />
      </>}

      {draft.type === 'aws_sigv4' && <>
        <Input label="Access key ID" autoComplete="off" placeholder="AKIA..." value={draft.accessKeyId} disabled={disabled}
          onChange={event => set('accessKeyId', event.target.value)} />
        {secretFields('From your AWS credentials')}
        <div className="grid grid-cols-2 gap-2">
          <Input label="Region" placeholder="us-east-1" value={draft.region} disabled={disabled} onChange={event => set('region', event.target.value)} />
          <Input label="Service" placeholder="aws-mcp" value={draft.service} disabled={disabled} onChange={event => set('service', event.target.value)} />
        </div>
        <Input label="Session token environment variable (optional)" placeholder="AWS_SESSION_TOKEN" value={draft.sessionTokenEnv}
          disabled={disabled} onChange={event => set('sessionTokenEnv', event.target.value)} />
      </>}

      <div className="space-y-2 border-t border-[var(--color-border)] pt-3">
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm font-medium text-[var(--color-text)]">Extra headers</span>
          <Button type="button" variant="ghost" className="px-2 py-1 text-xs" disabled={disabled || draft.headers.length >= MAX_HEADERS}
            onClick={() => set('headers', [...draft.headers, { name: '', value: '' }])}>
            <Plus className="h-3.5 w-3.5" /> Add header
          </Button>
        </div>
        {draft.headers.length === 0
          ? <p className="text-xs text-[var(--color-text-muted)]">None. Some servers need one, such as x-goog-user-project for Google Cloud quota.</p>
          : draft.headers.map((header, index) => (
            <div key={index} className="flex items-center gap-2">
              <input aria-label={`Header ${index + 1} name`} placeholder="x-goog-user-project" className={compact} value={header.name} disabled={disabled}
                onChange={event => set('headers', draft.headers.map((h, i) => i === index ? { ...h, name: event.target.value } : h))} />
              <input aria-label={`Header ${index + 1} value`} placeholder="value" className={compact} value={header.value} disabled={disabled}
                onChange={event => set('headers', draft.headers.map((h, i) => i === index ? { ...h, value: event.target.value } : h))} />
              <Button type="button" variant="ghost" className="px-2" aria-label={`Remove header ${index + 1}`} title="Remove header" disabled={disabled}
                onClick={() => set('headers', draft.headers.filter((_, i) => i !== index))}>
                <Trash2 className="h-4 w-4 text-[var(--color-danger)]" />
              </Button>
            </div>
          ))}
      </div>
    </div>
  )
}
