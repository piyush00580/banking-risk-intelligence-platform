import type { Health, Snapshot } from '../types'

const base = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')
async function json<T>(url: string): Promise<T> {
  let response: Response
  try {
    response = await fetch(url, { signal: AbortSignal.timeout(8000), headers: { Accept: 'application/json' } })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'TimeoutError') throw new Error('Request timed out')
    throw new Error('Unable to connect to the service')
  }
  if (!response.ok) throw new Error(`Request failed (${response.status})`)
  try { return await response.json() as T } catch { throw new Error('Service returned an invalid response') }
}
export async function getHealth(): Promise<Health> {
  const envelope = await json<{ success: boolean; data: Health; error?: { message: string } }>(`${base}/api/health`)
  if (!envelope.success) throw new Error(envelope.error?.message ?? 'Health check failed')
  return envelope.data
}
export const getSnapshot = () => json<Snapshot>('/data/overview.json')
