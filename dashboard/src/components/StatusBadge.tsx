import type { ReactNode } from 'react'
export default function StatusBadge({ tone, children }: { tone: 'good' | 'muted' | 'warn' | 'bad'; children: ReactNode }) { return <span className={`status-badge ${tone}`}><i/>{children}</span> }
