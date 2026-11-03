import type { LocalAuth } from './mcp-auth'

// The engine's server-name rule: tools reach models as <server>__<tool>.
const SERVER_NAME = /^[A-Za-z0-9_]{1,32}$/

export const SERVER_NAME_RULE =
  'Use letters, numbers and single underscores, up to 32 characters, such as google_gmail. Tools are exposed to models as <server>__<tool>, so spaces, hyphens and double underscores are not allowed.'

export function serverNameError(name: string): string | null {
  if (!name.trim()) return 'Enter a server name.'
  return SERVER_NAME.test(name.trim()) && !name.includes('__')
    ? null
    : `That server name cannot be used. ${SERVER_NAME_RULE}`
}

export type Policy = { read_only: boolean; cache?: unknown }

export type Server = {
  name: string
  url: string
  bearer_token?: string | null
  bearer_token_env?: string | null
  auth?: LocalAuth
  secret?: string
  secret_env?: string
  session_token_env?: string
  headers?: Record<string, string>
  timeout_ms: number
  allow_loopback_http?: boolean
  access: { mode: 'local'; project_path?: string; project_id?: string; project_ids?: string[] }
  tools: Record<string, Policy>
}

// signed_in names the servers with a usable stored OAuth sign-in.
export type Config = { servers: Server[]; signed_in?: string[] }

export type Draft = Omit<Server, 'tools'> & {
  tools: Array<Policy & { name: string }>
}

export const empty = (): Draft => ({
  name: '',
  url: '',
  timeout_ms: 30000,
  allow_loopback_http: false,
  access: { mode: 'local', project_ids: [] },
  tools: []
})

export function draft(server: Server): Draft {
  if ( !server ||
    typeof server.name !== 'string' ||
    typeof server.url !== 'string' ||
    typeof server.timeout_ms !== 'number' ||
    server.access?.mode !== 'local' ||
    !server.tools ||
    Array.isArray(server.tools) ||
    typeof server.tools !== 'object'
  )
    throw new Error(
      'Enter one local server object with name, url, timeout_ms, access and tools.'
    )

  const known = (value: object, keys: string[]) =>
    Object.keys(value).every((key) => keys.includes(key))

  if (
    !known(server, [
      'name',
      'url',
      'bearer_token',
      'bearer_token_env',
      'auth',
      'secret',
      'secret_env',
      'session_token_env',
      'headers',
      'timeout_ms',
      'allow_loopback_http',
      'access',
      'tools'
    ]) || !known(server.access, ['mode', 'project_path', 'project_id', 'project_ids']) ||
    (server.auth !== undefined &&
      (typeof server.auth !== 'object' || server.auth === null || typeof server.auth.type !== 'string')) ||
    (server.headers !== undefined &&
      (typeof server.headers !== 'object' || server.headers === null || Array.isArray(server.headers))) ||
    !Number.isSafeInteger(server.timeout_ms) ||
    (server.allow_loopback_http !== undefined &&
      typeof server.allow_loopback_http !== 'boolean') ||
    (server.bearer_token != null && typeof server.bearer_token !== 'string') ||
    (server.bearer_token_env != null &&
      typeof server.bearer_token_env !== 'string')
  ) {
    throw new Error(
      'Invalid or unknown server fields. Correct the JSON before switching to Form.'
    )
  }

  if (server.bearer_token != null && server.bearer_token_env != null) {
    throw new Error('Set bearer_token or bearer_token_env, not both.')
  }

  const access = server.access
  const scopes = [access.project_id, access.project_ids, access.project_path].filter(v => v !== undefined)
  if (scopes.length !== 1 ||
    (access.project_id !== undefined && typeof access.project_id !== 'string') ||
    (access.project_path !== undefined && typeof access.project_path !== 'string') ||
    (access.project_ids !== undefined && (!Array.isArray(access.project_ids) || access.project_ids.some(id => typeof id !== 'string')))) {
    throw new Error('Choose project IDs or a local directory, not both.')
  }

  const tools = Object.entries(server.tools).map(([name, policy]) => {
    if (!policy ||
      typeof policy !== 'object' ||
      Array.isArray(policy) ||
      typeof policy.read_only !== 'boolean' ||
      !known(policy, ['read_only', 'cache'])
    )
      throw new Error('Each tool needs a read_only boolean.')

    return { ...policy, name }
  })

  return { ...server, tools, access: access.project_id !== undefined
    ? { mode: 'local', project_ids: [access.project_id] } : access }
}

export function object(value: Draft, allowIncomplete = false): Server {
  const tools = value.tools
  if ((!allowIncomplete && tools.length === 0) || tools.length > 256) {
    throw new Error('Select between 1 and 256 tools.')
  }
  const names = tools.map((tool) => tool.name)

  if (
    (!allowIncomplete && names.some((name) => !name.trim())) ||
    new Set(names).size !== names.length
  )
    throw new Error('Tool names must be non-empty and unique.')

  return {
    ...value,
    bearer_token: value.bearer_token || null,
    bearer_token_env: value.bearer_token_env || null,
    tools: Object.fromEntries(
      tools.map(({ name, ...policy }) => [name, policy])
    )
  }
}
