import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { Trash2, X } from 'lucide-react'
import { Button } from './ui'

export function McpDeleteDialog({ name, busy, error, onClose, onConfirm }: {
  name: string
  busy: boolean
  error: string
  onClose: () => void
  onConfirm: () => void
}) {
  const dialog = useRef<HTMLDivElement>(null)
  const cancel = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    cancel.current?.focus()
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); onClose(); return }
      if (event.key !== 'Tab') return
      const buttons = Array.from(dialog.current?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)') ?? [])
      const first = buttons[0]
      const last = buttons.at(-1)
      if (!first) { event.preventDefault(); dialog.current?.focus(); return }
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
    }
    const onFocus = (event: FocusEvent) => {
      if (dialog.current && !dialog.current.contains(event.target as Node)) {
        const target = dialog.current.querySelector<HTMLElement>('button:not(:disabled)')
          ; (target ?? dialog.current).focus()
      }
    }
    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('focusin', onFocus)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('focusin', onFocus)
      document.body.style.overflow = previousOverflow
      if (opener?.isConnected) opener.focus()
      else document.getElementById('mcp-add-server')?.focus()
    }
  }, [onClose])

  return createPortal(
    <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/45 p-4 backdrop-blur-[1px]"
      onClick={event => { if (event.target === event.currentTarget) onClose() }}>
      <div ref={dialog} tabIndex={-1} role="alertdialog" aria-modal="true" aria-labelledby="mcp-delete-title" aria-describedby="mcp-delete-description" aria-busy={busy}
        className="max-h-[90vh] w-full max-w-md overflow-y-auto rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] shadow-2xl outline-none">
        <div className="flex items-center justify-between gap-3 border-b border-[var(--color-border)] px-5 py-4">
          <div className="flex items-center gap-3">
            <Trash2 aria-hidden="true" className="h-5 w-5 text-[var(--color-danger)]" />
            <h3 id="mcp-delete-title" className="text-base font-semibold text-[var(--color-text-heading)]">Delete MCP server?</h3>
          </div>
          <Button type="button" variant="ghost" aria-label="Close delete confirmation" disabled={busy} onClick={onClose}><X className="h-4 w-4" /></Button>
        </div>
        <div className="space-y-3 px-5 py-4 text-sm">
          <p id="mcp-delete-description">Remove <strong className="break-all text-[var(--color-text-heading)]">{name}</strong> and its tool configuration from this device?</p>
          <p className="text-xs text-[var(--color-text-muted)]">Restart the daemon after deletion to stop using this connection. The upstream MCP server and its data will remain unchanged.</p>
          {error && <p role="alert" className="break-words text-xs text-[var(--color-danger)]">{error}</p>}
        </div>
        <div className="flex justify-end gap-2 border-t border-[var(--color-border)] px-5 py-4">
          <button ref={cancel} type="button" disabled={busy} onClick={onClose}
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-2.5 text-sm font-semibold hover:bg-[var(--color-surface-highlight)] focus:outline-none focus:ring-1 focus:ring-[var(--color-primary)] disabled:opacity-50">Cancel</button>
          <Button type="button" variant="danger" disabled={busy} loading={busy} onClick={onConfirm}>Delete server</Button>
        </div>
      </div>
    </div>, document.body
  )
}
