export default function MetricCard({ label, value, detail, icon, accent = '' }: { label: string; value: string; detail: string; icon: string; accent?: string }) {
  return <article className="metric-card"><div className="metric-top"><span>{label}</span><span className={`metric-icon ${accent}`}>{icon}</span></div><strong>{value}</strong><p>{detail}</p><div className="metric-rule"><i/></div></article>
}
