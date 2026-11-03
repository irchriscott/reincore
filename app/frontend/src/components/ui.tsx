// Reusable UI primitives for the ReinCore desktop app.
// Mirrors the design system in reincore-frontend/src/components/ui.tsx.

import { type ReactNode, type ButtonHTMLAttributes, type InputHTMLAttributes, useState, useRef, useEffect, useId } from 'react'
import { Loader2, Sun, Moon, Check, ChevronDown } from 'lucide-react'

// Button

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost'
  loading?: boolean
}

export function Button({ variant = 'primary', loading, children, className = '', disabled, ...props }: ButtonProps) {
  const base = 'inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer focus:outline-none focus:ring-1 focus:ring-[var(--color-primary)]'
  const variants = {
    primary: 'bg-[var(--color-primary)] text-white shadow-[0_0_20px_-6px_var(--color-primary-glow)] hover:bg-[var(--color-primary-dark)] hover:shadow-[0_0_28px_-6px_var(--color-primary-glow)]',
    secondary: 'bg-[var(--color-surface)] text-[var(--color-text-heading)] border border-[var(--color-border)] hover:bg-[var(--color-surface-highlight)] hover:border-[var(--color-border-strong)]',
    danger: 'bg-[var(--color-danger)] text-white hover:opacity-90',
    ghost: 'text-[var(--color-text)] hover:bg-[var(--color-surface)] hover:text-[var(--color-text-heading)]',
  }
  return (
    <button className={`${base} ${variants[variant]} ${className}`} disabled={disabled || loading} {...props}>
      {loading && <Loader2 className="h-4 w-4 animate-spin" />}
      {children}
    </button>
  )
}

// Input

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
}

export function Input({ label, error, className = '', id, ...props }: InputProps) {
  const inputId = id || label?.toLowerCase().replace(/\s+/g, '_')
  return (
    <div className="flex flex-col gap-1.5">
      {label && (
        <label htmlFor={inputId} className="text-sm font-medium text-[var(--color-text)]">
          {label}
        </label>
      )}
      <input
        id={inputId}
        className={`rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-3.5 py-2.5 text-sm text-[var(--color-text-heading)] placeholder:text-[var(--color-text-muted)] outline-none transition-all focus:border-[var(--color-border-strong)] focus:ring-1 focus:ring-[var(--color-primary)] ${error ? 'border-[var(--color-danger)]' : ''} ${className}`}
        {...props}
      />
      {error && <span className="text-xs text-[var(--color-danger)]">{error}</span>}
    </div>
  )
}

// Card

export function Card({ children, className = '', onClick }: { children: ReactNode; className?: string; onClick?: (e: React.MouseEvent) => void }) {
  return (
    <div className={`rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] ${className}`} onClick={onClick}>
      {children}
    </div>
  )
}

// Badge

export function Badge({ children, variant = 'default', className = '', ...props }: React.HTMLAttributes<HTMLSpanElement> & { children: ReactNode; variant?: 'default' | 'primary' | 'warning' | 'danger' | 'info' | 'success' }) {
  const variants = {
    default: 'bg-[var(--color-surface-highlight)] text-[var(--color-text)]',
    primary: 'bg-[var(--color-accent-soft)] text-[var(--color-accent)]',
    warning: 'bg-[rgba(245,158,11,0.12)] text-[var(--color-warning)]',
    danger: 'bg-[rgba(244,63,94,0.12)] text-[var(--color-danger)]',
    info: 'bg-[rgba(56,189,248,0.12)] text-[var(--color-info)]',
    success: 'bg-[rgba(16,185,129,0.12)] text-[var(--color-success)]',
  }
  return (
    <span {...props} className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${variants[variant]} ${className}`}>
      {children}
    </span>
  )
}

// Skeleton

export function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-[var(--color-surface-highlight)] ${className}`} />
}

// EmptyState

export function EmptyState({ icon, title, description, action }: { icon?: ReactNode; title: string; description?: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-[var(--color-border)] py-16 text-center">
      {icon && <div className="mb-4 text-[var(--color-text-muted)]">{icon}</div>}
      <h3 className="mb-1 text-base font-semibold text-[var(--color-text-heading)]">{title}</h3>
      {description && <p className="mb-4 max-w-sm text-sm text-[var(--color-text-muted)]">{description}</p>}
      {action}
    </div>
  )
}

// Logo

// Core R mark, drawn on a 36-unit grid so its edges land on whole pixels in the 36px tile.
export function LogoMark({ className = '' }: { className?: string }) {
  return (
    <svg viewBox="0 0 36 36" fill="currentColor" aria-hidden="true" className={className}>
      <path d="M9 7h5v22H9zM16 7h11v11H16zM16 20h5l6 9h-5z" />
    </svg>
  )
}

export function Logo({ showText = true }: { showText?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <div className="relative flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--color-primary)] text-white shadow-[0_0_20px_-6px_var(--color-primary-glow)]">
        <LogoMark className="h-full w-full" />
      </div>
      {showText && (
        <span className="font-ui text-lg font-semibold tracking-tight text-[var(--color-text-heading)]">
          ReinCore
        </span>
      )}
    </div>
  )
}

// ThemeToggle

export function ThemeToggle({ theme, onToggle }: { theme: string; onToggle: () => void }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
      className="flex h-10 w-10 items-center justify-center rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-text-heading)] transition-all hover:border-[var(--color-border-strong)] hover:bg-[var(--color-surface-highlight)] cursor-pointer focus:outline-none focus:ring-1 focus:ring-[var(--color-primary)]"
    >
      {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  )
}

// StatusDot

export function StatusDot({ status }: { status: 'running' | 'stopped' | 'unknown' }) {
  const color = status === 'running' ? 'var(--color-success)' : status === 'stopped' ? 'var(--color-danger)' : 'var(--color-text-muted)'
  return (
    <span className="inline-flex items-center gap-2 text-xs font-medium text-[var(--color-text)]">
      <span className="relative flex h-2 w-2">
        {status === 'running' && (
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75" style={{ backgroundColor: color }} />
        )}
        <span className="relative inline-flex h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
      </span>
      {status === 'running' ? 'Running' : status === 'stopped' ? 'Stopped' : 'Unknown'}
    </span>
  )
}

// Checkbox

interface CheckboxProps {
  id?: string
  checked?: boolean
  indeterminate?: boolean
  onChange?: (checked: boolean) => void
  disabled?: boolean
  label?: ReactNode
  className?: string
  size?: 'sm' | 'md'
}

export function Checkbox({
  id, checked, indeterminate, onChange, disabled, label, className = '', size = 'md',
}: CheckboxProps) {
  const checkboxId = id || (typeof label === 'string' ? `chk-${label.toLowerCase().replace(/\s+/g, '_')}` : undefined)
  const sizeClasses = {
    sm: 'h-4 w-4 rounded',
    md: 'h-5 w-5 rounded-md',
  }
  const iconSize = size === 'sm' ? 'h-3 w-3' : 'h-3.5 w-3.5'
  const isChecked = checked || indeterminate

  return (
    <label htmlFor={checkboxId}
      className={`inline-flex cursor-pointer items-center gap-2.5 ${disabled ? 'cursor-not-allowed opacity-60' : ''} ${className}`}>
      <span className={`relative flex shrink-0 items-center justify-center border transition-all ${sizeClasses[size]} ${isChecked ? 'border-[var(--color-primary)] bg-[var(--color-primary)] text-white' : 'border-[var(--color-border)] bg-[var(--color-bg-subtle)] text-transparent hover:border-[var(--color-border-strong)]'} ${disabled ? '' : 'focus-within:ring-1 focus-within:ring-[var(--color-primary)]'}`}>
        {indeterminate ? (
          <span className="block h-0.5 w-2 rounded-full bg-current" />
        ) : (
          <Check className={`${iconSize} transition-transform ${isChecked ? 'scale-100' : 'scale-0'}`} strokeWidth={3} />
        )}
        <input id={checkboxId} type="checkbox" checked={!!checked}
          onChange={(e) => onChange?.(e.target.checked)} disabled={disabled} className="sr-only" />
      </span>
      {label && <span className="text-sm text-[var(--color-text-heading)]">{label}</span>}
    </label>
  )
}

// Radio

interface RadioProps {
  id?: string
  name?: string
  checked?: boolean
  onChange?: () => void
  disabled?: boolean
  label?: ReactNode
  className?: string
  size?: 'sm' | 'md'
}

export function Radio({ id, name, checked, onChange, disabled, label, className = '', size = 'md' }: RadioProps) {
  const radioId = id || (typeof label === 'string' ? `rad-${label.toLowerCase().replace(/\s+/g, '_').slice(0, 24)}` : undefined)
  const sizeClasses = { sm: 'h-3.5 w-3.5', md: 'h-4 w-4' }
  const dotSize = size === 'sm' ? 'h-1.5 w-1.5' : 'h-2 w-2'

  return (
    <label htmlFor={radioId}
      className={`inline-flex cursor-pointer items-center gap-2.5 ${disabled ? 'cursor-not-allowed opacity-60' : ''} ${className}`}>
      <span className={`relative flex shrink-0 items-center justify-center rounded-full border transition-all ${sizeClasses[size]} ${checked ? 'border-[var(--color-primary)] bg-[var(--color-bg-subtle)]' : 'border-[var(--color-border)] bg-[var(--color-bg-subtle)] hover:border-[var(--color-border-strong)]'} ${disabled ? '' : 'focus-within:ring-1 focus-within:ring-[var(--color-primary)]'}`}>
        <span className={`rounded-full bg-[var(--color-primary)] transition-transform ${dotSize} ${checked ? 'scale-100' : 'scale-0'}`} />
        <input id={radioId} type="radio" name={name} checked={!!checked}
          onChange={onChange} disabled={disabled} className="sr-only" />
      </span>
      {label && <span className="text-sm text-[var(--color-text-heading)]">{label}</span>}
    </label>
  )
}

// Toggle

export function Toggle({
  label, hint, checked, onChange,
}: {
  label: string
  hint?: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <button type="button" onClick={() => onChange(!checked)}
        className="flex flex-1 items-center gap-1.5 text-left">
        <span>
          <span className="text-sm font-medium text-[var(--color-text)]">{label}</span>
          {hint && <span className="block text-xs text-[var(--color-text-muted)]">{hint}</span>}
        </span>
      </button>
      <button type="button" onClick={() => onChange(!checked)}
        aria-pressed={checked} aria-label={label}
        className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors ${
          checked ? 'bg-[var(--color-primary)]' : 'bg-[var(--color-surface-highlight)]'
        }`}>
        <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
          checked ? 'translate-x-4' : 'translate-x-0.5'
        }`} />
      </button>
    </div>
  )
}

// ListSelect
// A custom dropdown that supports a title and optional subtitle per option.
// Used throughout the app for all dropdown selections.

export interface ListSelectOption {
  value: string
  title: string
  subtitle?: string
  disabled?: boolean
}

type ListSelectProps = {
  label?: string
  options: ListSelectOption[]
  placeholder?: string
  className?: string
  disabled?: boolean
} & ({
  multiple?: false
  value: string
  onChange: (value: string) => void
} | {
  multiple: true
  value: string[]
  onChange: (value: string[]) => void
})

export function ListSelect(props: ListSelectProps) {
  const { label, options, placeholder, className = '', disabled = false } = props
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const ref = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const id = useId()

  useEffect(() => {
    const handler = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const values = props.multiple ? props.value : [props.value]
  const selected = options.filter(option => values.includes(option.value))
  const displayText = props.multiple
    ? (values.length > 2 ? `${values.length} selected` : selected.map(option => option.title).join(', '))
    : selected[0]?.title
  const visible = props.multiple
    ? options.filter(option => `${option.title} ${option.subtitle ?? ''} ${option.value}`.toLowerCase().includes(search.toLowerCase()))
    : options

  function choose(option: ListSelectOption, checked: boolean) {
    if (option.disabled || disabled) return
    if (props.multiple) {
      props.onChange(checked ? Array.from(new Set([...props.value, option.value])) : props.value.filter(value => value !== option.value))
    } else {
      props.onChange(option.value)
      setOpen(false)
      trigger.current?.focus()
    }
  }

  return (
    <div className={`relative flex flex-col gap-1.5 ${className}`} ref={ref}
      onKeyDown={event => {
        if (event.key === 'Tab' && open) {
          // Wait for Tab's default focus movement. A pointer click on a
          // checkbox label may focus the surrounding drawer first, so blur
          // alone must never unmount the options before the click completes.
          requestAnimationFrame(() => {
            if (ref.current && !ref.current.contains(document.activeElement)) setOpen(false)
          })
        }
        if (event.key === 'Escape' && open) {
          event.preventDefault()
          event.stopPropagation()
          setOpen(false)
          trigger.current?.focus()
        }
      }}>
      {label && <label htmlFor={id} className="text-sm font-medium text-[var(--color-text)]">{label}</label>}
      <button id={id} ref={trigger} type="button" disabled={disabled}
        aria-expanded={open && !disabled} aria-controls={`${id}-options`}
        onClick={() => { setSearch(''); setOpen(value => !value) }}
        className="flex w-full items-center justify-between rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-3.5 py-2.5 text-left text-sm text-[var(--color-text-heading)] outline-none transition-all hover:border-[var(--color-border-strong)] focus:border-[var(--color-border-strong)] focus:ring-1 focus:ring-[var(--color-primary)] cursor-pointer disabled:cursor-not-allowed disabled:opacity-50">
        <span className={`truncate ${displayText ? '' : 'text-[var(--color-text-muted)]'}`}>{displayText || placeholder || 'Select...'}</span>
        <ChevronDown className={`h-4 w-4 shrink-0 text-[var(--color-text-muted)] transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && !disabled && (
        <div id={`${id}-options`} role="group" aria-label={label || 'Options'}
          className="absolute top-full left-0 right-0 z-50 mt-1 overflow-hidden rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] shadow-lg">
          {props.multiple && <div className="border-b border-[var(--color-border)] p-2">
            <Input aria-label={`Search ${label || 'options'}`} placeholder="Search options"
              value={search} onChange={event => setSearch(event.target.value)} />
          </div>}
          <div className="max-h-56 overflow-y-auto">
            {visible.map(option => props.multiple ? (
              <div key={option.value} className="px-3.5 py-2.5 hover:bg-[var(--color-surface-highlight)]">
                <Checkbox id={`${id}-${option.value}`} checked={values.includes(option.value)} disabled={option.disabled}
                  className="w-full" onChange={checked => choose(option, checked)}
                  label={<span className="block min-w-0 break-words">
                    <span className="block text-sm font-medium">{option.title}</span>
                    {option.subtitle && <span className="block break-all text-xs text-[var(--color-text-muted)]">{option.subtitle}</span>}
                  </span>} />
              </div>
            ) : (
              <button key={option.value} type="button" disabled={option.disabled} onClick={() => choose(option, true)}
                className={`flex w-full flex-col items-start gap-0.5 px-3.5 py-2.5 text-left transition-colors hover:bg-[var(--color-surface-highlight)] cursor-pointer disabled:cursor-not-allowed disabled:opacity-40 ${values.includes(option.value) ? 'bg-[var(--color-surface-highlight)]' : ''}`}>
                <div className="flex w-full items-center justify-between gap-2">
                  <span className="truncate text-sm font-medium text-[var(--color-text-heading)]">{option.title}</span>
                  {values.includes(option.value) && <Check className="h-3.5 w-3.5 shrink-0 text-[var(--color-primary)]" />}
                </div>
                {option.subtitle && <span className="truncate text-xs text-[var(--color-text-muted)]">{option.subtitle}</span>}
              </button>
            ))}
            {visible.length === 0 && <p className="px-3.5 py-3 text-xs text-[var(--color-text-muted)]">{search ? 'No matching options.' : 'No options available.'}</p>}
          </div>
        </div>
      )}
    </div>
  )
}

// DatePicker
// A date/month picker with a "General (all ...)" option and a native
// date/month input for specific dates.

interface DatePickerProps {
  label?: string
  mode: 'day' | 'month'
  value: string // '' = general/default, or 'YYYY-MM-DD' / 'YYYY-MM'
  onChange: (value: string) => void
  className?: string
}

export function DatePicker({ label, mode, value, onChange, className = '' }: DatePickerProps) {
  const isGeneral = value === ''
  const isSpecific = value !== ''

  return (
    <div className={`flex flex-col gap-3 ${className}`}>
      {label && (
        <label className="text-sm font-medium text-[var(--color-text)]">{label}</label>
      )}
      {/* General vs Specific toggle */}
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => onChange('')}
          className={`rounded-lg border px-3 py-2 text-sm font-medium transition-colors cursor-pointer ${
            isGeneral
              ? 'border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-text-heading)]'
              : 'border-transparent text-[var(--color-text)] hover:bg-[var(--color-surface)]'
          }`}
        >
          General (all {mode === 'day' ? 'days' : 'months'})
        </button>
        <button
          type="button"
          onClick={() => onChange(mode === 'day' ? todayISO() : currentMonthISO())}
          className={`rounded-lg border px-3 py-2 text-sm font-medium transition-colors cursor-pointer ${
            isSpecific
              ? 'border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-text-heading)]'
              : 'border-transparent text-[var(--color-text)] hover:bg-[var(--color-surface)]'
          }`}
        >
          Specific {mode === 'day' ? 'date' : 'month'}
        </button>
      </div>
      {/* Native date / month input */}
      {isSpecific && (
        <input
          type={mode === 'day' ? 'date' : 'month'}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-3.5 py-2.5 text-sm text-[var(--color-text-heading)] outline-none transition-all focus:border-[var(--color-border-strong)] focus:ring-1 focus:ring-[var(--color-primary)]"
        />
      )}
    </div>
  )
}

function todayISO(): string {
  const d = new Date()
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}

function currentMonthISO(): string {
  const d = new Date()
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  return `${y}-${m}`
}
