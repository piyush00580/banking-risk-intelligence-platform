import type { ReactNode } from 'react'

const links = [
  ['Overview', 'overview', '◫'], ['Credit Intelligence', 'credit', '◈'], ['Fraud Intelligence', 'fraud', '⌁'],
  ['Model Intelligence', 'models', '⌘'], ['Portfolio Analytics', 'portfolio', '▥'],
] as const
export default function AppShell({ page, onNavigate, children }: { page: string; onNavigate: (page: string) => void; children: ReactNode }) {
  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#overview" onClick={() => onNavigate('overview')}><span className="brand-mark">B</span><span><b>BRI</b><small>RISK INTELLIGENCE</small></span></a>
      <div className="workspace"><span className="workspace-dot"/> BANKING GROUP <span className="chevron">⌄</span></div>
      <p className="nav-label">WORKSPACE</p>
      <nav aria-label="Main navigation">{links.map(([label, id, icon]) => <button key={id} className={`nav-link ${page === id ? 'active' : ''}`} onClick={() => onNavigate(id)} aria-current={page === id ? 'page' : undefined}><span className="nav-icon">{icon}</span>{label}{id === 'overview' && <span className="nav-active-dot"/>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="support-card"><span className="support-mark">?</span><b>Need a hand?</b><small>Explore the platform guide</small><a href="https://github.com/piyush00580/banking-risk-intelligence-platform" target="_blank" rel="noreferrer">View documentation ↗</a></div><div className="user"><div className="avatar">PA</div><span><b>Platform Analyst</b><small>Risk &amp; Analytics</small></span><span className="more">···</span></div></div>
    </aside>
    <main className="main-area"><header className="topbar"><div className="breadcrumb">Risk Intelligence <span>/</span> <b>{links.find(x => x[1] === page)?.[0] ?? 'Overview'}</b></div><div className="top-actions"><span className="environment"><i/> DEVELOPMENT</span><div className="avatar small-avatar" aria-label="Platform Analyst">PA</div></div></header>{children}<footer className="footer"><span>Banking Risk Intelligence <span className="footer-sep">·</span> Internal analytics</span><span>Benchmark data <span className="footer-sep">·</span> No customer records displayed</span></footer></main>
  </div>
}
