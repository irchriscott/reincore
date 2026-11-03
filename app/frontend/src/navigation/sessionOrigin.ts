// Session-origin navigation: when a user opens a session detail page from
// a specific context (project detail, dashboard, sessions list), we record
// that context in router location.state so the top-bar "back" link can
// return them to where they came from instead of always going to /sessions.

/** Discriminated union describing where a session detail visit originated. */
export type SessionOrigin =
  | { kind: 'sessions' }
  | { kind: 'dashboard' }
  | { kind: 'project'; projectId: string }

/** Shape of the location.state object we attach when navigating. */
export interface SessionOriginState {
  from?: SessionOrigin
}

/** Type guard for the origin object stored in router state. */
export function isSessionOrigin(value: unknown): value is SessionOrigin {
  if (typeof value !== 'object' || value === null) return false
  const kind = (value as Record<string, unknown>).kind
  return kind === 'sessions' || kind === 'dashboard' || kind === 'project'
}

/** Build the `state` argument for navigate()/Link when opening a session
 *  detail page from a given origin. */
export function sessionOriginState(origin: SessionOrigin): SessionOriginState {
  return { from: origin }
}

/** Resolve the router location.state (which may be anything, including
 *  `null` for direct deep links or browser refresh) into an origin. */
export function resolveSessionOrigin(state: unknown): SessionOrigin | null {
  if (typeof state !== 'object' || state === null) return null
  const from = (state as SessionOriginState).from
  return isSessionOrigin(from) ? from : null
}

/** Route to navigate back to for a given origin, plus a human label. */
export function originBackLink(origin: SessionOrigin | null): { to: string; label: string } {
  switch (origin?.kind) {
    case 'project':
      return { to: `/projects/${origin.projectId}`, label: 'Back to project' }
    case 'dashboard':
      return { to: '/', label: 'Back to dashboard' }
    case 'sessions':
      return { to: '/sessions', label: 'Back to sessions' }
    default:
      // No recorded origin (deep link, refresh, or in-app nav from an
      // unknown source) - fall back to the sessions list.
      return { to: '/sessions', label: 'Back to sessions' }
  }
}