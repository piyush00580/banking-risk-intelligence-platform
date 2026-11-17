import { useState } from 'react'
import AppShell from './components/AppShell'
import Overview from './pages/Overview'
import Placeholder from './pages/Placeholder'

const titles: Record<string, string> = { credit: 'Credit Intelligence', fraud: 'Fraud Intelligence', models: 'Model Intelligence', portfolio: 'Portfolio Analytics' }
export default function App() {
  const [page, setPage] = useState('overview')
  return <AppShell page={page} onNavigate={setPage}>{page === 'overview' ? <Overview/> : <Placeholder title={titles[page] ?? 'Overview'}/>}</AppShell>
}
