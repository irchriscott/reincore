import { useEffect, useRef, useState } from 'react'
import { DAEMON_BASE } from '../api'
import type { SessionEvent } from '../types'

export type SessionStreamState = 'connecting' | 'live' | 'reconnecting' | 'stale'

interface SessionEventEnvelope {
  schema_version: number
  event: SessionEvent
}

function isSessionEvent(value: unknown): value is SessionEvent {
  if (!value || typeof value !== 'object') return false
  const event = value as Partial<SessionEvent>
  return typeof event.session_id === 'string'
    && typeof event.seq === 'number'
    && typeof event.event_type === 'string'
    && typeof event.created_at === 'string'
    && Boolean(event.payload && typeof event.payload === 'object')
}

/**
 * Subscribe to the daemon's resumable session-event SSE feed.
 * EventSource automatically reconnects with Last-Event-ID; the server repairs
 * any missed sequence range from the durable event log.
 */
export function useSessionEventStream(
  sessionId: string | undefined,
  initialAfterSeq: number | undefined,
  enabled: boolean,
) {
  const [events, setEvents] = useState<SessionEvent[]>([])
  const [cursorState, setCursorState] = useState<{ sessionId?: string; seq: number }>({ seq: 0 })
  const [connection, setConnection] = useState<{ sessionId?: string; state: SessionStreamState }>({ state: 'connecting' })
  const activeSessionRef = useRef(sessionId)
  const initialCursorRef = useRef(initialAfterSeq ?? 0)
  const lastSeqRef = useRef(initialAfterSeq ?? 0)

  useEffect(() => {
    if (activeSessionRef.current !== sessionId) {
      activeSessionRef.current = sessionId
      initialCursorRef.current = initialAfterSeq ?? 0
      lastSeqRef.current = initialAfterSeq ?? 0
      return
    }
    initialCursorRef.current = Math.max(initialCursorRef.current, initialAfterSeq ?? 0)
  }, [initialAfterSeq, sessionId])

  useEffect(() => {
    if (!sessionId || !enabled) return

    const cursor = Math.max(initialCursorRef.current, lastSeqRef.current)
    const url = `${DAEMON_BASE}/v1/sessions/${encodeURIComponent(sessionId)}/events/stream?after_seq=${cursor}`
    const source = new EventSource(url)

    const onSessionEvent = (message: MessageEvent<string>) => {
      try {
        const envelope = JSON.parse(message.data) as Partial<SessionEventEnvelope>
        if (envelope.schema_version !== 1 || !isSessionEvent(envelope.event)) return
        const event = envelope.event
        if (event.session_id !== sessionId || event.seq <= 0) return

        lastSeqRef.current = Math.max(lastSeqRef.current, event.seq)
        setCursorState({ sessionId, seq: lastSeqRef.current })
        setEvents((current) => {
          if (current.some((known) => known.seq === event.seq)) return current
          return [...current, event]
            .sort((left, right) => left.seq - right.seq)
            .slice(-200)
        })
        setConnection({ sessionId, state: 'live' })
      } catch {
        // Ignore malformed frames; the sequence cursor is not advanced, so a
        // reconnect can replay the canonical event from storage.
      }
    }

    const onStreamError = () => {
      setConnection({ sessionId, state: 'stale' })
      source.close()
    }

    source.onopen = () => setConnection({ sessionId, state: 'live' })
    source.onerror = () => setConnection({ sessionId, state: 'reconnecting' })
    source.addEventListener('session_event', onSessionEvent as EventListener)
    source.addEventListener('stream_error', onStreamError)

    return () => {
      source.removeEventListener('session_event', onSessionEvent as EventListener)
      source.removeEventListener('stream_error', onStreamError)
      source.close()
    }
  }, [enabled, sessionId])

  const sessionEvents = events.filter((event) => event.session_id === sessionId)
  const lastSeq = cursorState.sessionId === sessionId ? cursorState.seq : initialAfterSeq ?? 0
  const connectionState = connection.sessionId === sessionId ? connection.state : 'connecting'
  return { events: sessionEvents, lastSeq, connectionState }
}
