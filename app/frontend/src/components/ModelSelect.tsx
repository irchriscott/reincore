// Model tier selector - dropdown when a catalog is available, free-text when
// not. Mirrors the ProviderModelField design from reincore-frontend.

import type { ModelOption } from '../utils/model-options'
import { ListSelect } from './ui'
import type { ListSelectOption } from './ui'

interface ModelSelectProps {
  label: string
  value: string
  onChange: (value: string) => void
  options: ModelOption[]
  emptyLabel?: string
  placeholder?: string
}

export function ModelSelect({
  label,
  value,
  onChange,
  options,
  emptyLabel = 'Auto (default)',
  placeholder = 'Type a model id...',
}: ModelSelectProps) {
  const hasCatalog = options.length > 0

  return (
    <div className="flex flex-col gap-1.5">
      {hasCatalog ? (
        <ListSelect
          label={label}
          value={value}
          onChange={onChange}
          placeholder={emptyLabel}
          options={[
            { value: '', title: emptyLabel },
            // If the current value isn't in the catalog, inject it so it's not lost.
            ...(value && !options.some((o) => o.model_id === value)
              ? [{ value, title: `${value} (current)` } as ListSelectOption]
              : []),
            ...options.map((o) => ({ value: o.model_id, title: o.model_name })),
          ]}
        />
      ) : (
        <>
          <label className="text-sm font-medium text-[var(--color-text)]">{label}</label>
          <input
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder={placeholder}
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-3.5 py-2.5 text-sm text-[var(--color-text-heading)] placeholder:text-[var(--color-text-muted)] outline-none transition-all focus:border-[var(--color-border-strong)] focus:ring-1 focus:ring-[var(--color-primary)]"
          />
        </>
      )}
    </div>
  )
}