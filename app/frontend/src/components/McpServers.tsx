import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { FolderOpen, RefreshCw, X } from 'lucide-react'
import { listLocalProjects } from '../api'
import type { LocalProject } from '../types'
import { invoke } from '@tauri-apps/api/core'
import { Button, Checkbox, Input, ListSelect } from './ui'
import { McpServerDetails } from './McpServerDetails'
import { McpDeleteDialog } from './McpDeleteDialog'
import { McpAuthSection } from './McpAuthSection'
import {
  authDraftError,
  authDraftFromServer,
  authRejectionMessage,
  parseAuthRejection,
  authFields,
  defaultAuthDraft,
  serverAuth,
  type AuthDraft,
  type AuthorizationMetadata
} from '../utils/mcp-auth'

import {
  draft,
  empty,
  object,
  serverNameError,
  SERVER_NAME_RULE,
  type Config,
  type Draft,
  type Server
} from '../utils/mcp-server-form'

type DiscoveredTool = { name: string; description: string; input_schema: unknown }

const area = 'w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] p-3 font-mono text-xs text-[var(--color-text)]'

export function McpServers() {
  const [servers, setServers] = useState<Server[]>([])
  const [loaded, setLoaded] = useState(false)
  const [busy, setBusy] = useState(true)
  const operation = useRef(true)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [originalName, setOriginalName] = useState<string | null>(null)
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)
  const [deleteError, setDeleteError] = useState('')
  const closeDelete = useCallback(() => {
    if (!operation.current) { setPendingDelete(null); setDeleteError('') }
  }, [])
  const [mode, setMode] = useState<'form' | 'json'>('form')
  const [value, setValue] = useState<Draft>(empty)
  const [auth, setAuth] = useState<AuthDraft>(defaultAuthDraft)
  const [authorization, setAuthorization] = useState<AuthorizationMetadata | null>(null)
  const [authNotice, setAuthNotice] = useState('')
  // Servers with a usable stored OAuth sign-in.
  const [signedIn, setSignedIn] = useState<string[]>([])
  const [raw, setRaw] = useState('')
  const [editorOpen, setEditorOpen] = useState(false)
  const [projects, setProjects] = useState<LocalProject[]>([])
  const [projectsLoading, setProjectsLoading] = useState(false)
  const [projectsError, setProjectsError] = useState('')
  const [inventory, setInventory] = useState<{ key: string; tools: DiscoveredTool[] } | null>(null)
  const [toolSearch, setToolSearch] = useState('')
  const [discovering, setDiscovering] = useState(false)
  // The stored sign-in of the server being edited applies only while its
  // endpoint is unchanged.
  const savedServer = originalName === null ? null : servers.find(server => server.name === originalName) ?? null
  const savedAuth = savedServer ? serverAuth(savedServer) : null
  const savedOauth = savedServer && savedAuth?.type === 'oauth' && signedIn.includes(savedServer.name) &&
    savedServer.url === value.url.trim() ? savedAuth : null
  const fields = authFields(auth, savedOauth)
  const connection = {
    url: value.url,
    bearer_token: null,
    bearer_token_env: null,
    auth: fields.auth ?? null,
    secret: fields.secret ?? null,
    secret_env: fields.secret_env ?? null,
    session_token_env: fields.session_token_env ?? null,
    headers: fields.headers ?? {},
    allow_loopback_http: !!value.allow_loopback_http,
    timeout_ms: value.timeout_ms
  }
  const connectionKey = JSON.stringify([connection, auth.signIn?.id ?? null])
  const discoveredTools = inventory?.key === connectionKey ? inventory.tools : null
  const availableTools = [
    ...(discoveredTools ?? []),
    ...value.tools.filter(tool => !discoveredTools?.some(item => item.name === tool.name))
      .map(tool => ({ name: tool.name, description: discoveredTools ? 'Previously selected; not returned by this server. Remove it or refresh before saving.' : 'Previously configured tool.', input_schema: null }))
  ]
  const filteredTools = availableTools.filter(tool =>
    `${tool.name} ${tool.description}`.toLowerCase().includes(toolSearch.toLowerCase()))
  const missingTools = discoveredTools !== null && value.tools.some(tool => !discoveredTools.some(item => item.name === tool.name))
  const panel = useRef<HTMLElement>(null)
  const closeEditor = useCallback(() => {
    if (!operation.current) setEditorOpen(false)
  }, [])

  useEffect(() => {
    let active = true
    listLocalProjects().then(items => {
      if (active) { setProjects(items); setProjectsError('') }
    }).catch(error => { if (active) setProjectsError(String(error)) })
      .finally(() => { if (active) setProjectsLoading(false) })
    return () => { active = false }
  }, [editorOpen])

  useEffect(() => {
    if (!editorOpen) return
    const opener = document.activeElement as HTMLElement | null
    const bodyOverflow = document.body.style.overflow
    const htmlOverflow = document.documentElement.style.overflow
    document.body.style.overflow = 'hidden'
    document.documentElement.style.overflow = 'hidden'
    panel.current?.focus()
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); closeEditor() }
      if (event.key !== 'Tab') return
      const elements = Array.from(panel.current?.querySelectorAll<HTMLElement>(
        'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex="0"]'
      ) ?? []).filter(element => element.getClientRects().length > 0)
      const first = elements[0]
      const last = elements.at(-1)
      if (!first) { event.preventDefault(); panel.current?.focus(); return }
      if (event.shiftKey && (document.activeElement === first || document.activeElement === panel.current)) {
        event.preventDefault(); last?.focus()
      } else if (!event.shiftKey && (document.activeElement === last || document.activeElement === panel.current)) {
        event.preventDefault(); first.focus()
      }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.body.style.overflow = bodyOverflow
      document.documentElement.style.overflow = htmlOverflow
      opener?.focus()
    }
  }, [editorOpen, closeEditor])

  async function discoverTools() {
    if (operation.current) return
    const authError = authDraftError(auth, value.url, savedOauth)
    if (authError) { setError(authError); return }
    operation.current = true
    setBusy(true)
    setDiscovering(true)
    setError('')
    setAuthNotice('')
    try {
      const tools = await invoke<DiscoveredTool[]>('discover_mcp_tools', {
        connection,
        signInId: auth.signIn?.id ?? null,
        serverName: auth.type === 'oauth' && !auth.signIn && savedOauth ? originalName : null
      })
      setInventory({ key: connectionKey, tools })
    } catch (error) {
      const message = String(error)
      if (/rejected the credentials|sign in/i.test(message)) {
        await explainRefusal(message)
      } else {
        setError(`Could not load tools: ${message}`)
      }
    } finally {
      operation.current = false
      setBusy(false)
      setDiscovering(false)
    }
  }

  // A refused probe says what the server needs: with no authentication set,
  // look for OAuth sign-in and switch to it when the server offers it.
  async function explainRefusal(message: string) {
    if (auth.type !== 'none') {
      setError(authRejectionMessage(parseAuthRejection(message), auth.type))
      return
    }
    try {
      const metadata = await invoke<AuthorizationMetadata>('mcp_oauth_metadata', { url: value.url.trim(), timeoutMs: value.timeout_ms })
      setAuthorization(metadata)
      if (metadata.authorization_endpoint && metadata.token_endpoint) {
        const scopes = metadata.challenge_scopes.length ? metadata.challenge_scopes : metadata.scopes_supported
        setAuth(current => ({ ...current, type: 'oauth', scopes: current.scopes || scopes.join(' ') }))
        setAuthNotice('This server requires sign-in. Authentication is now set to OAuth sign-in: sign in, then load tools again.')
        return
      }
    } catch {
      // Fall through to the general explanation.
    }
    setAuthNotice('This server requires authentication. Choose the type its documentation describes, then load tools again.')
  }

  function selectTool(name: string, selected: boolean) {
    setValue(current => ({
      ...current,
      tools: selected
        ? current.tools.some(tool => tool.name === name) || current.tools.length >= 256
          ? current.tools : [...current.tools, { name, read_only: false }]
        : current.tools.filter(tool => tool.name !== name)
    }))
  }

  async function pickDirectory() {
    if (operation.current) return
    operation.current = true
    setBusy(true)
    setError('')
    try {
      const path = await invoke<string | null>('pick_directory')
      if (path) {
        setValue(current => ({
          ...current,
          access: { mode: 'local', project_path: path }
        }))
      }
    } catch (error) {
      setError(`Could not open folder picker: ${String(error)}`)
    } finally {
      operation.current = false
      setBusy(false)
    }
  }

  async function reloadProjects() {
    setProjectsLoading(true)
    setProjectsError('')
    try { setProjects(await listLocalProjects()) }
    catch (error) { setProjectsError(String(error)) }
    finally { setProjectsLoading(false) }
  }

  function openEditor() {
    setInventory(null)
    setAuthorization(null)
    setAuthNotice('')
    setToolSearch('')
    setProjectsLoading(true)
    setProjectsError('')
    setEditorOpen(true)
  }

  const selectedProjects = value.access.project_ids ?? []
  const projectOptions = [
    ...projects,
    ...selectedProjects.filter(id => !projects.some(project => project.id === id))
      .map(id => ({ id, name: `Unavailable project (${id})`, directories: [] }))
  ]


  useEffect(() => {
    let active = true
    invoke<Config>('list_mcp_servers')
      .then((config) => {
        if (!active) return
        setServers(config.servers)
        setSignedIn(config.signed_in ?? [])
        setLoaded(true)
      })
      .catch((error) => {
        if (active) setError(String(error))
      })
      .finally(() => {
        if (active) {
          operation.current = false
          setBusy(false)
        }
      })
    return () => {
      active = false
    }
  }, [])

  async function reload() {
    if (operation.current) return
    operation.current = true
    setBusy(true)
    setError('')
    try {
      const config = await invoke<Config>('list_mcp_servers')
      setServers(config.servers)
      setSignedIn(config.signed_in ?? [])
      setLoaded(true)
    } catch (error) {
      setError(String(error))
    } finally {
      operation.current = false
      setBusy(false)
    }
  }

  function reset() {
    setPendingDelete(null)
    setOriginalName(null)
    setValue(empty())
    setAuth(defaultAuthDraft())
    setRaw('')
    setMode('form')
  }

  function edit(server: Server) {
    try {
      setValue(draft(server))
      setAuth(authDraftFromServer(server))
      setPendingDelete(null)
      setRaw(JSON.stringify(server, null, 2))
      setOriginalName(server.name)
      setError('')
      setMode('form')
      openEditor()
    } catch (error) {
      setError(String(error))
    }
  }

  function switchMode(next: 'form' | 'json') {
    if (next === mode) return
    try {
      if (next === 'json') setRaw(JSON.stringify(object({ ...value, ...authFields(auth, savedOauth) }, true), null, 2))
      else {
        const parsed = JSON.parse(raw)
        setValue(draft(parsed))
        // A sign-in finished in the form still applies when the JSON keeps its settings.
        setAuth(current => ({ ...authDraftFromServer(parsed), signIn: current.signIn }))
      }
      setMode(next)
      setError('')
    } catch (error) {
      setError(String(error))
    }
  }

  async function save() {
    if (operation.current || !loaded) return
    operation.current = true
    setBusy(true)
    setError('')
    setMessage('')

    try {
      if (mode === 'form') {
        const nameError = serverNameError(value.name)
        if (nameError) throw new Error(nameError)
        const authError = authDraftError(auth, value.url, savedOauth)
        if (authError) throw new Error(authError)
      }
      const server: Server = mode === 'json' ? JSON.parse(raw) : object({ ...value, ...authFields(auth, savedOauth) })
      // A finished sign-in is saved when the server still uses its settings.
      const signInId = auth.signIn && JSON.stringify(server.auth) === JSON.stringify(auth.signIn.auth) ? auth.signIn.id : null
      if (mode === 'form' && server.access.project_ids) {
        server.access.project_ids = server.access.project_ids.map(id => id.trim()).filter(Boolean)
      }
      const config = await invoke<Config>('save_mcp_server', {
        server,
        originalName,
        signInId
      })
      setServers(config.servers)
      setSignedIn(config.signed_in ?? [])
      reset()
      setEditorOpen(false)
      setMessage('Saved. Restart the daemon to apply changes.')
    } catch (error) {
      setError(String(error))
    } finally {
      operation.current = false
      setBusy(false)
    }
  }

  async function remove(name: string) {
    if (operation.current || !loaded) return
    operation.current = true
    setBusy(true)
    setDeleteError('')
    setMessage('')

    try {
      const config = await invoke<Config>('delete_mcp_server', { name })
      setServers(config.servers)
      setSignedIn(config.signed_in ?? [])
      setPendingDelete(null)
      if (originalName === name) reset()
      setMessage(
        'Deleted from configuration. Restart the daemon to stop using this server.'
      )
    } catch (error) {
      setDeleteError(String(error))
    } finally {
      operation.current = false
      setBusy(false)
    }
  }

  return (
    <div className="mt-5 border-t border-[var(--color-border)] pt-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold">Configured servers</h3>
        <div className="flex gap-2">
          <Button id="mcp-add-server" disabled={busy || !loaded} onClick={() => { reset(); setError(''); setMessage(''); openEditor() }}>Add MCP server</Button>
          <Button
            type="button"
            variant="secondary"
            disabled={busy}
            onClick={reload}
          >
            Reload servers
          </Button>
        </div>
      </div>
      {!loaded && !error && <p className="text-xs">Loading servers...</p>}
      {loaded && !servers.length && (
        <p className="mb-3 text-xs text-[var(--color-text-muted)]">
          No servers configured.
        </p>
      )}
      <ul className="space-y-3">
        {servers.map(server => <li key={server.name}>
          <McpServerDetails server={server} projects={projects} busy={busy} signedIn={signedIn.includes(server.name)}
            onEdit={() => edit(server)}
            onDelete={() => { setDeleteError(''); setPendingDelete(server.name) }} />
        </li>)}
      </ul>
      {pendingDelete !== null && <McpDeleteDialog name={pendingDelete} busy={busy} error={deleteError}
        onClose={closeDelete} onConfirm={() => remove(pendingDelete)} />}
      {editorOpen && createPortal(
        <div className="fixed inset-0 z-[100]" role="dialog" aria-modal="true" aria-labelledby="mcp-editor-title">
          <button type="button" aria-label="Close MCP configuration" disabled={busy}
            className="absolute inset-0 bg-black/45 backdrop-blur-[1px]" onClick={closeEditor} />
          <aside ref={panel} tabIndex={-1}
            className="absolute inset-y-0 right-0 flex w-[min(560px,94vw)] flex-col border-l border-[var(--color-border)] bg-[var(--color-surface)] shadow-2xl outline-none">
            <div className="flex items-center justify-between border-b border-[var(--color-border)] px-5 py-4">
              <h3 id="mcp-editor-title" className="text-base font-semibold text-[var(--color-text-heading)]">{originalName ? `Edit ${originalName}` : 'Add MCP server'}</h3>
              <Button type="button" variant="ghost" aria-label="Close" disabled={busy} onClick={closeEditor}><X className="h-4 w-4" /></Button>
            </div>
            <form
              className="flex min-h-0 flex-1 flex-col"
              onSubmit={(event) => {
                event.preventDefault()
                void save()
              }}
            >
              <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-5 py-4">
                <fieldset disabled={busy || !loaded}>
                  <div className="mb-3 flex gap-2">
                    <Button
                      type="button"
                      variant={mode === 'form' ? 'primary' : 'secondary'}
                      onClick={() => switchMode('form')}
                    >
                      Form
                    </Button>
                    <Button
                      type="button"
                      variant={mode === 'json' ? 'primary' : 'secondary'}
                      onClick={() => switchMode('json')}
                    >
                      Raw JSON
                    </Button>
                  </div>
                  {mode === 'json' ? (
                    <label className="block text-sm">
                      Server JSON
                      <textarea
                        className={area}
                        rows={16}
                        value={raw}
                        onChange={(event) => setRaw(event.target.value)}
                        spellCheck={false}
                        required
                      />
                    </label>
                  ) : (
                    <div className="space-y-3">
                      <Input
                        label="Server name"
                        title={SERVER_NAME_RULE}
                        placeholder="e.g. google_gmail"
                        value={value.name}
                        required
                        disabled={originalName !== null}
                        onChange={(event) =>
                          setValue({ ...value, name: event.target.value })
                        }
                      />
                      <Input
                        label="MCP endpoint URL"
                        type="url"
                        placeholder="https://mcp.example.com/mcp"
                        value={value.url}
                        required
                        onChange={(event) =>
                          setValue({ ...value, url: event.target.value })
                        }
                      />
                      <ListSelect label="Server scope" disabled={busy}
                        value={value.access.project_ids !== undefined ? 'project' : 'path'}
                        options={[{ value: 'project', title: 'Projects (all assigned paths)' }, { value: 'path', title: 'Local directory' }]}
                        onChange={scope => setValue({
                          ...value, access: scope === 'project'
                            ? { mode: 'local', project_ids: [] } : { mode: 'local', project_path: '' }
                        })} />
                      {value.access.project_ids !== undefined ? (
                        <div className="space-y-2">
                          <ListSelect multiple label="Select projects" disabled={busy} placeholder="Choose projects"
                            value={selectedProjects}
                            options={projectOptions.map(project => ({
                              value: project.id, title: project.name,
                              subtitle: project.directories.join(', '),
                              disabled: !selectedProjects.includes(project.id) && selectedProjects.length >= 256
                            }))}
                            onChange={projectIds => setValue({ ...value, access: { mode: 'local', project_ids: projectIds } })} />
                          <p className="text-xs text-[var(--color-text-muted)]">{selectedProjects.length} selected · Up to 256 projects</p>
                          {projectsLoading && <p role="status" className="text-xs">Loading projects...</p>}
                          {projectsError && <div role="alert" className="text-xs text-[var(--color-danger)]">
                            <p>Could not load projects. Existing selections are preserved.</p>
                            <Button type="button" variant="secondary" disabled={projectsLoading} onClick={reloadProjects}>Retry</Button>
                          </div>}
                        </div>
                      ) : (
                        <div>
                          <label htmlFor="mcp-project-directory" className="mb-1.5 block text-sm font-medium text-[var(--color-text)]">
                            Local project directory
                          </label>
                          <div className="flex items-stretch gap-2">
                            <div className="min-w-0 flex-1">
                              <Input
                                id="mcp-project-directory"
                                className="w-full"
                                placeholder="Absolute path to an existing directory"
                                value={value.access.project_path ?? ''}
                                required
                                onChange={(event) => setValue({
                                  ...value,
                                  access: { mode: 'local', project_path: event.target.value }
                                })}
                              />
                            </div>
                            <button
                              type="button"
                              disabled={busy}
                              onClick={pickDirectory}
                              title="Browse for a directory"
                              aria-label="Browse for a directory"
                              className="flex shrink-0 items-center justify-center rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2.5 text-[var(--color-text-heading)] transition-all hover:border-[var(--color-border-strong)] hover:bg-[var(--color-surface-highlight)] cursor-pointer focus:outline-none focus:ring-1 focus:ring-[var(--color-primary)] disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                              <FolderOpen className="h-4 w-4" />
                            </button>
                          </div>
                        </div>
                      )}
                      <p className="text-xs text-[var(--color-text-muted)]">
                        Project scope uses one connection for every session with one of those project IDs,
                        across all its paths. Directory scope matches only that directory.
                      </p>
                      <McpAuthSection draft={auth} busy={busy} url={value.url}
                        timeoutMs={value.timeout_ms >= 100 && value.timeout_ms <= 120000 ? value.timeout_ms : 30000}
                        serverName={originalName} savedOauth={savedOauth}
                        authorization={authorization} onAuthorization={setAuthorization} onError={setError}
                        onChange={update => { setAuthNotice(''); setAuth(update) }} />
                      {authNotice && <p role="status" className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-3 py-2 text-xs">{authNotice}</p>}
                      <Input
                        label="Timeout (milliseconds)"
                        type="number"
                        min={100}
                        max={120000}
                        value={value.timeout_ms}
                        required
                        onChange={(event) =>
                          setValue({ ...value, timeout_ms: Number(event.target.value) })
                        }
                      />
                      <Checkbox id="mcp-loopback" label="Allow HTTP on loopback only (127.0.0.1, ::1, localhost)"
                        checked={!!value.allow_loopback_http} disabled={busy}
                        onChange={checked => setValue({ ...value, allow_loopback_http: checked })} />
                      <p className="text-xs text-[var(--color-text-muted)]">
                        Allow only tools you intend to expose. Mark read-only only if
                        the tool cannot mutate data. Advanced cache settings are
                        available in Raw JSON.
                      </p>
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-sm font-medium">Upstream tools</span>
                        <Button type="button" variant="secondary" disabled={busy || !value.url.trim()} loading={discovering} onClick={discoverTools}>
                          {!discovering && <RefreshCw className="h-4 w-4" />}
                          {discoveredTools === null ? 'Connect & load tools' : 'Refresh tools'}
                        </Button>
                      </div>
                      {availableTools.length > 0 && <>
                        <Input aria-label="Search upstream tools" placeholder="Search tools by name or description" value={toolSearch} onChange={event => setToolSearch(event.target.value)} />
                        <div className="flex items-center justify-between gap-2 text-xs">
                          <span>{value.tools.length} selected · Up to 256 tools</span>
                          <div className="flex gap-2">
                            <Button type="button" variant="ghost" disabled={busy || discoveredTools === null || new Set([...value.tools.map(tool => tool.name), ...filteredTools.map(tool => tool.name)]).size > 256}
                              onClick={() => setValue(current => ({ ...current, tools: [...current.tools, ...filteredTools.filter(tool => !current.tools.some(item => item.name === tool.name)).map(tool => ({ name: tool.name, read_only: false }))] }))}>
                              Select all shown
                            </Button>
                            <Button type="button" variant="ghost" disabled={busy} onClick={() => setValue({ ...value, tools: value.tools.filter(tool => !filteredTools.some(item => item.name === tool.name)) })}>Clear shown</Button>
                          </div>
                        </div>
                        <div className="max-h-80 overflow-y-auto rounded-lg border border-[var(--color-border)] divide-y divide-[var(--color-border)]">
                          {filteredTools.map((tool, index) => {
                            const selected = value.tools.find(item => item.name === tool.name)
                            return <div key={tool.name} className="space-y-2 p-3">
                              <Checkbox id={`mcp-discovered-${index}`} checked={!!selected}
                                disabled={busy || (!selected && value.tools.length >= 256)}
                                onChange={checked => selectTool(tool.name, checked)}
                                label={<span className="break-all font-medium">{tool.name}</span>} />
                              {tool.description && <p className="whitespace-pre-wrap break-words text-xs text-[var(--color-text-muted)]">{tool.description}</p>}
                              {selected && <Checkbox id={`mcp-tool-read-only-${index}`} label="Read-only" checked={selected.read_only} disabled={busy}
                                onChange={checked => setValue(current => ({ ...current, tools: current.tools.map(item => item.name === tool.name ? { ...item, read_only: checked, ...(checked ? {} : { cache: null }) } : item) }))} />}
                            </div>
                          })}
                          {filteredTools.length === 0 && <p className="p-3 text-xs text-[var(--color-text-muted)]">No matching tools.</p>}
                        </div>
                      </>}
                      {availableTools.length === 0 && <p className="text-xs text-[var(--color-text-muted)]">
                        {discoveredTools === null ? 'Connect to discover tools, then select which ones to enable.' : 'This server returned no tools.'}
                      </p>}
                      {missingTools && <p role="alert" className="text-xs text-[var(--color-danger)]">Some selected tools were not returned. Deselect them or refresh before saving.</p>}

                    </div>
                  )}
                </fieldset>
                {error && <p role="alert" className="mt-3 text-xs text-[var(--color-danger)]">{error}</p>}
              </div>
              <div className="flex shrink-0 justify-end gap-2 border-t border-[var(--color-border)] px-5 py-4">
                <Button type="button" variant="secondary" disabled={busy} onClick={closeEditor}>Cancel</Button>
                <Button type="submit" loading={busy} disabled={busy || (mode === 'form' && (value.tools.length === 0 || missingTools || (value.access.project_ids !== undefined && selectedProjects.length === 0)))}>
                  {originalName ? 'Save changes' : 'Add server'}
                </Button>
              </div>
            </form>
          </aside>
        </div>, document.body
      )}
      {!editorOpen && error && (
        <p role="alert" className="mt-3 text-xs text-[var(--color-danger)]">
          {error}
        </p>
      )}
      {message && (
        <p
          role="status"
          className="mt-3 text-xs text-[var(--color-text-muted)]"
        >
          {message}
        </p>
      )}
    </div>
  )
}
