// Daemon health hook - polls /health on mount and every 10s so the sidebar
// status dot reflects whether the embedded daemon is actually serving.

import { useEffect, useState } from 'react'
import { getHealth } from '../api'

export type DaemonState = 'running' | 'stopped' | 'checking'

export function useDaemonHealth() {
  const [state, setState] = useState<DaemonState>('checking')
  const [mode, setMode] = useState<string>('')
  const [provider, setProvider] = useState<string>('')

  useEffect(() => {
    let cancelled = false

    async function check() {
      try {
        const health = await getHealth()
        if (cancelled) return
        setState('running')
        setMode(health.mode || '')
        setProvider(health.mode === 'single_provider' ? health.providers?.[0] || '' : '')
      } catch {
        if (!cancelled) {
          setState('stopped')
          setMode('')
          setProvider('')
        }
      }
    }

    check()
    const interval = setInterval(check, 10_000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  return { state, mode, provider }
}
