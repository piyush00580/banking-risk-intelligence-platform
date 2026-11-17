export default function SnapshotState({ error }: { error?: string }) {
  if (error) return <div className="empty-state error-state" role="alert"><span>!</span><b>Benchmark snapshot unavailable</b><p>{error}. Regenerate Phase 9.1 exports and run the snapshot builder.</p></div>
  return <div className="empty-state loading-state" aria-live="polite"><span>◫</span><b>Loading benchmark snapshot</b><div className="skeleton-lines"><i/><i/><i/></div></div>
}
