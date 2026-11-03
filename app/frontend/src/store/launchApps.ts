// Cache store for the integrated-apps list returned by the Tauri
// `list_integrated_apps` command.  The call does filesystem / PATH probes
// for every supported app and can take a second or two, so we cache the
// result in a zustand store that survives across page navigations instead
// of re-fetching every time the Launch page is mounted.

import { create } from 'zustand'
import { invoke } from '@tauri-apps/api/core'

export interface IntegratedApp {
  id: string
  display_name: string
  default_provider: string
  installed: boolean
  install_hint: string | null
}

/// Collapse alias app ids (e.g. "codex-app", "gemini-cli", "open-code") to
/// a single canonical id so the catalog doesn't show duplicates.
function canonicalAppId(id: string): string {
  const lower = id.toLowerCase()
  if (lower === 'codex-app') return 'codex'
  if (lower === 'gemini-cli') return 'gemini'
  if (lower === 'open-code' || lower === 'open code') return 'opencode'
  if (lower === 'open-claw' || lower === 'open claw') return 'openclaw'
  return lower
}

interface LaunchAppsState {
  /// The cached, de-duplicated, sorted app list.  Empty until the first
  /// successful fetch completes.
  apps: IntegratedApp[]
  /// True while a fetch is in progress (but no cached data is available yet).
  loading: boolean
  /// Error message from the last fetch attempt, or null.
  error: string | null
  /// Whether at least one successful fetch has completed.
  loaded: boolean

  /// Fetch the app list from the backend.  If `force` is false and the cache
  /// is already populated, this is a no-op - the cached data is returned
  /// immediately.  Pass `force: true` to re-fetch (e.g. the Refresh button).
  fetchApps: (force?: boolean) => Promise<void>
}

export const useLaunchAppsStore = create<LaunchAppsState>((set, get) => ({
  apps: [],
  loading: false,
  error: null,
  loaded: false,

  fetchApps: async (force = false) => {
    const { loaded, loading } = get()
    // Skip if already loaded (or currently loading) and a refresh isn't forced.
    if ((loaded || loading) && !force) return

    set({ loading: true, error: null })
    try {
      const raw = await invoke<IntegratedApp[]>('list_integrated_apps')
      const seen = new Set<string>()
      const unique = raw
        .filter((a) => {
          const canonical = canonicalAppId(a.id)
          if (seen.has(canonical)) return false
          seen.add(canonical)
          return true
        })
        .sort((a, b) => a.display_name.localeCompare(b.display_name))

      set({ apps: unique, loading: false, loaded: true, error: null })
    } catch (err: unknown) {
      const msg =
        typeof err === 'string'
          ? err
          : (err as { message?: string })?.message ?? String(err)
      set({ loading: false, error: msg })
    }
  },
}))