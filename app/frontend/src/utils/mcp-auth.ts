// MCP server authentication in the desktop app: the form draft behind the
// Authentication section, how it becomes a local server's `auth`, secret
// source and `headers`, and how a saved server is described.
//
// Local secrets are stored inline in mcp.json or read from an environment
// variable. OAuth sign-in tokens are never part of the server JSON: the app
// keeps them in mcp.credentials.json beside it.

import type { Server } from './mcp-server-form'

export type ClientAuth = 'none' | 'client_secret_basic' | 'client_secret_post'

export type LocalAuth =
  | { type: 'none' }
  | { type: 'bearer' }
  | { type: 'header'; name: string }
  | { type: 'basic'; username: string }
  | { type: 'oauth'; client_id: string; token_endpoint: string; scopes?: string[]; resource?: string; client_auth?: ClientAuth; issuer?: string }
  | { type: 'client_credentials'; token_endpoint: string; client_id: string; scopes?: string[]; audience?: string; resource?: string; client_auth?: ClientAuth }
  | { type: 'google_service_account'; scopes: string[]; subject?: string }
  | { type: 'aws_sigv4'; access_key_id: string; region: string; service: string }

export type AuthType = LocalAuth['type']

// What a server publishes about signing in (mcp_oauth_metadata).
export type AuthorizationMetadata = {
  requires_auth: boolean
  resource: string
  issuer?: string
  authorization_endpoint?: string
  token_endpoint?: string
  dynamic_registration: boolean
  scopes_supported: string[]
  challenge_scopes: string[]
  grant_types_supported: string[]
}

// A sign-in finished in the browser and waiting to be saved with a server.
export type FinishedSignIn = { id: string; auth: LocalAuth; expires_in_seconds: number }

export type HeaderDraft = { name: string; value: string }

export type AuthDraft = {
  type: AuthType
  // Every type's settings are kept so switching back does not lose them.
  headerName: string
  username: string
  clientId: string
  // OAuth only: sent when signing in, never stored in mcp.json.
  clientSecret: string
  tokenEndpoint: string
  scopes: string
  audience: string
  clientAuth: Exclude<ClientAuth, 'none'>
  subject: string
  accessKeyId: string
  region: string
  service: string
  secretSource: 'inline' | 'env'
  secret: string
  secretEnv: string
  sessionTokenEnv: string
  headers: HeaderDraft[]
  signIn: (FinishedSignIn & { url: string }) | null
}

export const AUTH_OPTIONS: { value: AuthType; title: string; subtitle: string }[] = [
  { value: 'none', title: 'None', subtitle: 'Public server; no credentials are sent' },
  { value: 'bearer', title: 'Bearer token', subtitle: 'Authorization: Bearer <token>' },
  { value: 'header', title: 'API key header', subtitle: 'Any header, e.g. x-api-key or X-Goog-Api-Key' },
  { value: 'basic', title: 'Basic auth', subtitle: 'Username with a password or API token' },
  { value: 'oauth', title: 'OAuth sign-in', subtitle: 'Sign in with the provider in your browser' },
  { value: 'client_credentials', title: 'OAuth client credentials', subtitle: 'Machine-to-machine client ID and secret' },
  { value: 'google_service_account', title: 'Google service account', subtitle: 'Service account JSON key' },
  { value: 'aws_sigv4', title: 'AWS Signature V4', subtitle: 'AWS access key; every request is signed' },
]

export const GOOGLE_CLOUD_SCOPE = 'https://www.googleapis.com/auth/cloud-platform'
export const MAX_HEADERS = 16

const HEADER_NAME = /^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,64}$/
const ENV_NAME = /^[^=\0\s][^=\0]*$/
const RESERVED_HEADERS = new Set([
  'accept', 'connection', 'content-length', 'content-type', 'expect', 'host', 'keep-alive',
  'last-event-id', 'mcp-protocol-version', 'mcp-session-id', 'proxy-authorization',
  'proxy-connection', 'te', 'trailer', 'transfer-encoding', 'upgrade', 'x-amz-date',
  'x-amz-security-token',
])

// The authentication a server uses. Servers written before typed
// authentication have no `auth` and use their bearer token when one is set.
export function serverAuth(server: Server): LocalAuth {
  if (server.auth) return server.auth as LocalAuth
  return server.bearer_token || server.bearer_token_env ? { type: 'bearer' } : { type: 'none' }
}

export function defaultAuthDraft(): AuthDraft {
  return {
    type: 'none', headerName: '', username: '', clientId: '', clientSecret: '', tokenEndpoint: '',
    scopes: '', audience: '', clientAuth: 'client_secret_basic', subject: '', accessKeyId: '',
    region: '', service: '', secretSource: 'inline', secret: '', secretEnv: '', sessionTokenEnv: '',
    headers: [], signIn: null,
  }
}

export function authDraftFromServer(server: Server): AuthDraft {
  const draft = defaultAuthDraft()
  const auth = serverAuth(server)
  draft.type = auth.type
  draft.secret = server.secret ?? server.bearer_token ?? ''
  draft.secretEnv = server.secret_env ?? server.bearer_token_env ?? ''
  draft.secretSource = draft.secretEnv && !draft.secret ? 'env' : 'inline'
  draft.sessionTokenEnv = server.session_token_env ?? ''
  draft.headers = Object.entries(server.headers ?? {}).map(([name, value]) => ({ name, value }))
  switch (auth.type) {
    case 'header':
      draft.headerName = auth.name
      break
    case 'basic':
      draft.username = auth.username
      break
    case 'oauth':
      draft.clientId = auth.client_id
      draft.scopes = (auth.scopes ?? []).join(' ')
      break
    case 'client_credentials':
      draft.clientId = auth.client_id
      draft.tokenEndpoint = auth.token_endpoint
      draft.scopes = (auth.scopes ?? []).join(' ')
      draft.audience = auth.audience ?? ''
      draft.clientAuth = auth.client_auth === 'client_secret_post' ? 'client_secret_post' : 'client_secret_basic'
      break
    case 'google_service_account':
      draft.scopes = auth.scopes.join(' ')
      draft.subject = auth.subject ?? ''
      break
    case 'aws_sigv4':
      draft.accessKeyId = auth.access_key_id
      draft.region = auth.region
      draft.service = auth.service
      break
  }
  return draft
}

function parseUrl(value: string): URL | null {
  try { return new URL(value) } catch { return null }
}

// Google has no client credentials grant: its token endpoint refuses every
// such request, whatever the client.
const GOOGLE_TOKEN_HOSTS = new Set(['oauth2.googleapis.com', 'accounts.google.com', 'www.googleapis.com'])

export const GOOGLE_CLIENT_CREDENTIALS =
  'Google does not support OAuth client credentials. For Gmail, Drive, Calendar and other user data, choose OAuth sign-in and enter this client ID and secret there; for Google Cloud services, choose Google service account.'

export function usesGoogleClientCredentials(draft: AuthDraft): boolean {
  if (draft.type !== 'client_credentials') return false
  if (draft.clientId.trim().endsWith('.apps.googleusercontent.com')) return true
  try {
    return GOOGLE_TOKEN_HOSTS.has(new URL(draft.tokenEndpoint.trim()).hostname)
  } catch {
    return false
  }
}

// Whether published authorization server metadata rules out client credentials.
export function clientCredentialsUnsupported(grantTypes: string[] | undefined): boolean {
  return !!grantTypes?.length && !grantTypes.includes('client_credentials')
}

export function splitScopes(scopes: string): string[] {
  return [...new Set(scopes.split(/\s+/).filter(Boolean))]
}

function buildAuth(draft: AuthDraft, savedOauth: LocalAuth | null): LocalAuth | null {
  const optional = (value: string) => (value.trim() ? value.trim() : undefined)
  switch (draft.type) {
    case 'none':
    case 'bearer':
      return { type: draft.type }
    case 'header':
      return { type: 'header', name: draft.headerName.trim() }
    case 'basic':
      return { type: 'basic', username: draft.username.trim() }
    case 'oauth':
      return draft.signIn?.auth ?? savedOauth
    case 'client_credentials':
      return {
        type: 'client_credentials', token_endpoint: draft.tokenEndpoint.trim(), client_id: draft.clientId.trim(),
        scopes: splitScopes(draft.scopes), audience: optional(draft.audience), client_auth: draft.clientAuth,
      }
    case 'google_service_account':
      return { type: 'google_service_account', scopes: splitScopes(draft.scopes), subject: optional(draft.subject) }
    case 'aws_sigv4':
      return { type: 'aws_sigv4', access_key_id: draft.accessKeyId.trim(), region: draft.region.trim(), service: draft.service.trim() }
  }
}

// The server fields the draft sets. Legacy bearer fields are replaced by
// `auth` with `secret` or `secret_env`. `savedOauth` is the settings of the
// sign-in stored for the server being edited, when it still applies.
export type AuthServerFields = Pick<
  Server,
  'auth' | 'bearer_token' | 'bearer_token_env' | 'secret' | 'secret_env' | 'session_token_env' | 'headers'
>

export function authFields(draft: AuthDraft, savedOauth: LocalAuth | null): AuthServerFields {
  const fields: AuthServerFields = {
    auth: buildAuth(draft, savedOauth) ?? undefined,
    bearer_token: null,
    bearer_token_env: null,
    secret: undefined,
    secret_env: undefined,
    session_token_env: undefined,
    headers: undefined,
  }
  if (draft.type !== 'none' && draft.type !== 'oauth') {
    if (draft.secretSource === 'env') fields.secret_env = draft.secretEnv.trim() || undefined
    else fields.secret = draft.secret || undefined
  }
  if (draft.type === 'aws_sigv4' && draft.sessionTokenEnv.trim()) fields.session_token_env = draft.sessionTokenEnv.trim()
  const headers = Object.fromEntries(
    draft.headers.filter(header => header.name.trim() || header.value.trim())
      .map(header => [header.name.trim(), header.value.trim()]),
  )
  if (Object.keys(headers).length) fields.headers = headers
  return fields
}

export function secretLabel(type: AuthType): string {
  return {
    none: 'Credential', bearer: 'Bearer token', header: 'Header value', basic: 'Password or API token',
    oauth: 'Credential', client_credentials: 'Client secret', google_service_account: 'Service account key',
    aws_sigv4: 'Secret access key',
  }[type]
}

// A validation message for the draft, or null when it can be used.
export function authDraftError(draft: AuthDraft, url: string, savedOauth: LocalAuth | null): string | null {
  const scopes = splitScopes(draft.scopes)
  if (scopes.length > 64 || scopes.some(scope => scope.length > 256 || /["\\]/.test(scope) || /[^\x21-\x7e]/.test(scope)))
    return 'Scopes are separated by spaces and cannot contain quotes or backslashes (up to 64).'
  switch (draft.type) {
    case 'bearer':
      if (draft.secretSource === 'inline' && draft.secret && !/^[\x21-\x7e]{1,16384}$/.test(draft.secret))
        return 'The bearer token must be printable ASCII without spaces.'
      break
    case 'header': {
      const name = draft.headerName.trim()
      if (!HEADER_NAME.test(name)) return 'Enter a valid header name, such as x-api-key.'
      if (RESERVED_HEADERS.has(name.toLowerCase())) return `The ${name} header is managed by ReinCore and cannot be used.`
      if (draft.secretSource === 'inline' && draft.secret && (draft.secret.trim() !== draft.secret || /[^\t\x20-\x7e]/.test(draft.secret)))
        return 'The header value must be printable ASCII without leading or trailing spaces.'
      break
    }
    case 'basic':
      if (!draft.username.trim() || draft.username.includes(':')) return 'Enter a username; it cannot contain a colon.'
      break
    case 'oauth':
      if (draft.signIn && draft.signIn.url !== url.trim()) return 'The sign-in was for a different endpoint URL. Sign in again.'
      if (!draft.signIn && !savedOauth) return 'Sign in with the provider first.'
      break
    case 'client_credentials': {
      const endpoint = parseUrl(draft.tokenEndpoint.trim())
      const loopback = endpoint?.protocol === 'http:' && ['127.0.0.1', '[::1]'].includes(endpoint.hostname)
      if (!endpoint || !(endpoint.protocol === 'https:' || loopback)) return 'Enter the HTTPS token endpoint of the authorization server.'
      if (!/^[\x21-\x7e]+$/.test(draft.clientId.trim())) return 'Enter the OAuth client ID.'
      if (usesGoogleClientCredentials(draft)) return GOOGLE_CLIENT_CREDENTIALS
      break
    }
    case 'google_service_account':
      if (scopes.length === 0) return 'Enter at least one scope for the service account.'
      if (draft.secretSource === 'inline' && draft.secret) {
        try {
          const key = JSON.parse(draft.secret) as { type?: string; client_email?: string; private_key?: string }
          if (key.type !== 'service_account' || !key.client_email || !key.private_key) throw new Error()
        } catch {
          return 'The service account key must be the JSON key file downloaded from Google Cloud.'
        }
      }
      break
    case 'aws_sigv4':
      if (!/^[A-Z0-9]{16,128}$/.test(draft.accessKeyId.trim())) return 'Enter a valid AWS access key ID.'
      if (!/^[a-z0-9-]{1,32}$/.test(draft.region.trim())) return 'Enter the AWS region, such as us-east-1.'
      if (!/^[a-z0-9-]{1,64}$/.test(draft.service.trim())) return 'Enter the AWS service name used for signing, such as aws-mcp.'
      if (draft.sessionTokenEnv && !ENV_NAME.test(draft.sessionTokenEnv.trim())) return 'Enter a valid session token environment variable name.'
      break
  }
  if (draft.type !== 'none' && draft.type !== 'oauth') {
    const label = secretLabel(draft.type).toLowerCase()
    if (draft.secretSource === 'env') {
      if (!ENV_NAME.test(draft.secretEnv.trim())) return `Enter the environment variable that holds the ${label}.`
    } else if (!draft.secret) {
      return `Enter the ${label}.`
    }
  }
  if (draft.headers.length > MAX_HEADERS) return `Add at most ${MAX_HEADERS} extra headers.`
  const names = new Set<string>()
  for (const header of draft.headers) {
    const name = header.name.trim()
    if (!name && !header.value.trim()) continue
    if (!HEADER_NAME.test(name)) return `"${name}" is not a valid header name.`
    const lower = name.toLowerCase()
    if (lower === 'authorization') return 'Send credentials with an authentication type, not as an extra header.'
    if (RESERVED_HEADERS.has(lower)) return `The ${name} header is managed by ReinCore and cannot be used.`
    if (names.has(lower)) return `The ${name} header is listed twice.`
    names.add(lower)
    if (!header.value.trim() || header.value.trim().length > 4096 || /[^\t\x20-\x7e]/.test(header.value))
      return `Enter a printable value for the ${name} header.`
  }
  return null
}

// What to do about an upstream refusal, from the codes the engine keeps: the
// HTTP status, the Bearer challenge's error and a Google API error reason.
export function authRejectionMessage(
  rejection: { status: number; error?: string; reason?: string } | null | undefined,
  type: string,
): string {
  const oauth = type === 'oauth'
  if (rejection?.reason === 'SERVICE_DISABLED') {
    return "The server's API is not enabled in the Google Cloud project that owns the OAuth client. For Gmail, enable both the Gmail API and the Gmail MCP API in that project, wait a few minutes, then load tools again."
  }
  if (rejection?.reason === 'ACCESS_TOKEN_SCOPE_INSUFFICIENT' || rejection?.error === 'insufficient_scope') {
    return oauth
      ? 'The sign-in did not grant every permission this server needs. Sign in again and allow all requested permissions (Google lets you untick some), then load tools.'
      : 'These credentials do not have the permissions (scopes) this server needs.'
  }
  const base = oauth
    ? 'The server did not accept the sign-in. Sign in again, then load tools.'
    : 'The server rejected these credentials. Check them and try again.'
  if (!rejection) return base
  const hint =
    oauth && rejection.error === 'invalid_token'
      ? " If it keeps happening, check that the server's API is enabled in the OAuth client's project and that your account may use the server (some are in preview)."
      : ''
  const codes = [rejection.error, rejection.reason].filter(Boolean).join(', ')
  return `${base} (HTTP ${rejection.status}${codes ? `: ${codes}` : ''})${hint}`
}

// The refusal codes at the end of the engine's "rejected the credentials" error.
export function parseAuthRejection(message: string): { status: number; error?: string; reason?: string } | null {
  const match = /status=(\d+)(?: error=([\w.-]+))?(?: reason=([\w.-]+))?/.exec(message)
  return match ? { status: Number(match[1]), error: match[2], reason: match[3] } : null
}

// Region and signing name from an AWS endpoint such as
// https://aws-mcp.us-east-1.api.aws/mcp.
export function awsSigningFromUrl(url: string): { service: string; region: string } | null {
  try {
    const match = new URL(url).hostname.match(/^([a-z0-9-]+)\.([a-z]{2}(?:-[a-z]+)+-\d)\.(?:api\.aws|amazonaws\.com)$/)
    return match ? { service: match[1], region: match[2] } : null
  } catch {
    return null
  }
}

function host(value: string | undefined): string {
  if (!value) return ''
  try { return new URL(value).host } catch { return value }
}

// A one-line description of how a saved server authenticates.
export function describeServerAuth(server: Server, signedIn: boolean): string {
  const auth = serverAuth(server)
  const source = server.secret_env || server.bearer_token_env
    ? ` · from $${server.secret_env ?? server.bearer_token_env}`
    : server.secret || server.bearer_token ? ' · stored in mcp.json' : ' · no credential set'
  switch (auth.type) {
    case 'none': return 'None (public server)'
    case 'bearer': return `Bearer token${source}`
    case 'header': return `API key header ${auth.name}${source}`
    case 'basic': return `Basic auth as ${auth.username}${source}`
    case 'oauth': return `OAuth sign-in via ${host(auth.issuer ?? auth.token_endpoint)}${signedIn ? '' : ' · sign in again'}`
    case 'client_credentials': return `OAuth client credentials (${auth.client_id})${source}`
    case 'google_service_account': return `Google service account${auth.subject ? ` acting as ${auth.subject}` : ''}${source}`
    case 'aws_sigv4': return `AWS SigV4 · ${auth.service} in ${auth.region}${source}`
  }
}
