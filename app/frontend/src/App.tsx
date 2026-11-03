// App: top-level router. All routes are "protected" in the sense that they
// assume the embedded daemon is running - there is no auth layer in the
// desktop app with no auth layer (unlike the removed cloud
// backend). The sidebar shows the daemon status instead.

import { useEffect } from 'react'
import { HashRouter, Routes, Route } from 'react-router-dom'
import { AppLayout } from './components/AppLayout'
import { DashboardPage } from './pages/DashboardPage'
import { SessionsPage, SessionDetailPage } from './pages/SessionsPage'
import { BudgetPage } from './pages/BudgetPage'
import { ProjectDetailPage } from './pages/ProjectDetailPage'
import { ProjectsPage } from './pages/ProjectsPage'
import { ProvidersPage } from './pages/ProvidersPage'
import { LaunchPage } from './pages/LaunchPage'
import { SettingsPage } from './pages/SettingsPage'
import { useLaunchAppsStore } from './store/launchApps'

export default function App() {
  // Kick off the integrated-apps list probe (filesystem / PATH checks) in
  // the background as soon as the app starts. The store caches the result,
  // so the Launch page renders instantly when the user navigates to it
  // instead of waiting on slow per-app filesystem probes.
  const { fetchApps } = useLaunchAppsStore()
  useEffect(() => {
    fetchApps()
  }, [fetchApps])

  return (
    <HashRouter>
      <AppLayout>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/projects" element={<ProjectsPage />} />
          <Route path="/projects/:id" element={<ProjectDetailPage />} />
          <Route path="/sessions" element={<SessionsPage />} />
          <Route path="/sessions/:id" element={<SessionDetailPage />} />
          <Route path="/budget" element={<BudgetPage />} />
          <Route path="/providers" element={<ProvidersPage />} />
          <Route path="/launch" element={<LaunchPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </AppLayout>
    </HashRouter>
  )
}