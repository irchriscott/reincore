export function TaggedFiles({ files, compact = false, className = '' }: { files: string[]; compact?: boolean; className?: string }) {
  if (files.length === 0) return null

  const visibleFiles = compact ? files.slice(0, 2) : files
  return (
    <section aria-label="Tagged files" className={`min-w-0 shrink-0 border border-[var(--color-border)] bg-[var(--color-bg-subtle)]/30 px-3 py-2 ${className}`}>
      <h4 className="text-xs font-semibold text-[var(--color-text-secondary)]">Tagged files</h4>
      <ul className="mt-1 list-disc space-y-1 pl-4 text-xs leading-4 text-[var(--color-text-secondary)]">
        {visibleFiles.map((file, index) => <li key={`${index}-${file}`} title={file}><span className={compact ? 'block truncate' : '[overflow-wrap:anywhere]'}>{file}</span></li>)}
      </ul>
      {visibleFiles.length < files.length && <p className="mt-1 text-[11px] text-[var(--color-text-muted)]">+{files.length - visibleFiles.length} more in the full request view</p>}
    </section>
  )
}
