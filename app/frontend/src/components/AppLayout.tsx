// App layout: sidebar navigation + topbar for the desktop app.
// The workspace switcher is replaced by a daemon status dot, and the
// account section is replaced by theme + about.

import { type ReactNode, useEffect, useState } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { LayoutDashboard, Activity, Wallet, Cpu, Rocket, Settings, Menu, X, FolderKanban, ArrowLeft } from 'lucide-react'
import { Logo, ThemeToggle, StatusDot } from './ui'
import { useTheme } from '../hooks/useTheme'
import { useDaemonHealth, type DaemonState } from '../hooks/useDaemonHealth'
import { listProvidersStatus } from '../api'
import { originBackLink, resolveSessionOrigin } from '../navigation/sessionOrigin'
import type { ProviderStatus } from '../types'

// Map the daemon health hook's three-state value onto the StatusDot's
// three-state prop ('checking' -> 'unknown').
function dotStatus(state: DaemonState): 'running' | 'stopped' | 'unknown' {
  return state === 'running' ? 'running' : state === 'stopped' ? 'stopped' : 'unknown'
}

export function AppLayout({ children }: { children: ReactNode }) {
  const { theme, toggle } = useTheme()
  const { state, mode, provider } = useDaemonHealth()
  const [providers, setProviders] = useState<ProviderStatus[]>([])
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const location = useLocation()

  // Session and project detail pages get a persistent back link in the
  // sticky top bar so navigation stays reachable while scrolling.
  // For session details, the back target comes from the origin recorded in
  // router state when the user navigated in (project detail, dashboard, or
  // sessions list) so the user returns to where they actually came from.
  const isSessionDetail = /^\/sessions\/[^/]+$/.test(location.pathname)
  const isProjectDetail = /^\/projects\/[^/]+$/.test(location.pathname)
  const sessionBackLink = originBackLink(resolveSessionOrigin(location.state))
  const backLink = isSessionDetail
    ? sessionBackLink
    : isProjectDetail
      ? { to: '/projects', label: 'Back to projects' }
      : null

  // Scroll to top whenever the route changes. The document (html/body) is
  // the scroll container - #root and <main> are not height-constrained, so
  // resetting <main> would be a no-op and the previous page's scroll offset
  // would carry over. `behavior: 'instant'` overrides the global
  // `html { scroll-behavior: smooth }` so the jump is immediate instead of
  // animating on every navigation.
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' })
  }, [location.pathname])

  useEffect(() => {
    listProvidersStatus().then(setProviders).catch(() => {})
  }, [])

  const providerName = (rawId: string) => {
    const p = providers.find((x) => x.id === rawId)
    return p?.label || rawId
  }

  const navItems = [
    { label: 'Dashboard', to: '/', icon: LayoutDashboard, end: true },
    { label: 'Projects', to: '/projects', icon: FolderKanban },
    { label: 'Sessions', to: '/sessions', icon: Activity },
    { label: 'Budget', to: '/budget', icon: Wallet },
    { label: 'Providers', to: '/providers', icon: Cpu },
    { label: 'Launch', to: '/launch', icon: Rocket },
    { label: 'Settings', to: '/settings', icon: Settings },
  ]

  const renderNavLink = (item: { label: string; to: string; icon: typeof LayoutDashboard; end?: boolean }) => (
    <NavLink
      key={item.to}
      to={item.to}
      end={item.end ?? item.to === '/'}
      onClick={() => setSidebarOpen(false)}
      className={({ isActive }) =>
        `flex items-center gap-3 rounded-lg border px-3 py-2.5 text-sm font-medium transition-colors ${
          isActive
            ? 'border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-text-heading)]'
            : 'border-transparent text-[var(--color-text)] hover:bg-[var(--color-surface)] hover:text-[var(--color-text-heading)]'
        }`
      }
    >
      <item.icon className="h-4 w-4" />
      {item.label}
    </NavLink>
  )

  return (
    <div className="flex min-h-screen bg-[var(--color-bg)]">
      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r border-[var(--color-border)] bg-[var(--color-bg-elevated)] transition-transform duration-300 lg:translate-x-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Logo header */}
        <div className="flex h-16 shrink-0 items-center justify-between border-b border-[var(--color-border)] px-6">
          <Logo />
          <button
            className="text-[var(--color-text-muted)] lg:hidden"
            onClick={() => setSidebarOpen(false)}
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Daemon status */}
        <div className="shrink-0 border-b border-[var(--color-border)] p-3">
          <div className="flex items-center justify-between rounded-lg bg-[var(--color-surface)] px-3 py-2.5">
            <div className="flex flex-col">
              <span className="text-xs font-medium text-[var(--color-text-muted)]">Engine</span>
              <span className="text-xs text-[var(--color-text)]">
                {mode === 'single_provider' && provider ? `${providerName(provider)} provider` : mode === 'multi_provider' ? 'Multi-provider' : state === 'running' ? 'Engine' : 'Checking...'}
              </span>
            </div>
            <StatusDot status={dotStatus(state)} />
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex flex-1 flex-col gap-1 p-4">
          {navItems.map(renderNavLink)}
        </nav>

        {/* Bottom: theme toggle */}
        <div className="shrink-0 border-t border-[var(--color-border)] p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-[var(--color-text-muted)]">Theme</span>
            <ThemeToggle theme={theme} onToggle={toggle} />
          </div>
        </div>
      </aside>

      {/* Overlay for mobile sidebar */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main content - `min-w-0` is critical: without it the flex item's
          `min-width: auto` prevents shrinking below content intrinsic width,
          causing horizontal overflow when the sidebar is position:fixed. */}
      <div className="flex min-w-0 flex-1 flex-col lg:ml-64">
        {/* Topbar (mobile menu toggle only) */}
        <header className="sticky top-0 z-30 flex h-16 items-center border-b border-[var(--color-border)] bg-[var(--color-bg)]/80 px-6 backdrop-blur-xl">
          <button
            className="text-[var(--color-text-heading)] lg:hidden"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu className="h-5 w-5" />
          </button>
          {backLink && (
            <Link
              to={backLink.to}
              className="ml-4 flex items-center gap-2 text-sm text-[var(--color-text-muted)] transition-colors hover:text-[var(--color-text-heading)] lg:ml-0"
            >
              <ArrowLeft className="h-4 w-4" /> {backLink.label}
            </Link>
          )}
          <div className="ml-auto flex items-center gap-3">
            <span className="hidden text-xs text-[var(--color-text-muted)] sm:inline">
              {mode === 'single_provider' && provider ? `Single provider: ${providerName(provider)}` : '127.0.0.1:8989'}
            </span>
            <StatusDot status={dotStatus(state)} />
          </div>
        </header>

        {/* Page content */}
        <main className={`min-w-0 flex-1 p-6 lg:p-8 ${isSessionDetail ? 'overflow-x-clip pb-0 lg:pb-0' : 'overflow-x-hidden'}`}>
          {children}
        </main>
      </div>
    </div>
  )
}
