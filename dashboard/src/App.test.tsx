import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from './App'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })
const health = { success: true, data: { status: 'available', service_name: 'Banking Risk Intelligence API', api_version: '1.0.0', model_readiness: { ready: true, credit_risk: { artifact_available: true, metadata_available: true, loaded: false }, transaction_fraud: { artifact_available: true, loaded: false } } } }
const snapshot = { source: 'Phase 9.1 local benchmark exports', credit: { scope: 'held_out_test', applications: 6483, observed_defaults: 560, observed_default_rate: 560 / 6483, mean_predicted_default_probability: 0.12, by_grade: [], in_sample_rows_excluded: 25928 }, fraud: { scope: 'held_out_test', transactions: 56746, fraud: 95, legitimate: 56651, true_positives: 71, false_positives: 2, true_negatives: 56649, false_negatives: 24, precision: 0.9726, recall: 0.7474, f1: 0.8452, pr_auc: 0.8297, roc_auc: 0.9725, threshold: 0.35 } }
describe('dashboard foundation', () => {
  it('renders verified overview data and API status', async () => {
    vi.stubGlobal('fetch', vi.fn((url: string) => Promise.resolve({ ok: true, json: async () => url.includes('/api/health') ? health : snapshot })) as unknown as typeof fetch)
    render(<App/>); expect(screen.getByRole('heading', { name: 'Overview ✳' })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText('6,483')).toBeInTheDocument())
    expect(screen.getByText('71')).toBeInTheDocument(); expect(screen.getByText('PHASE 9.1 EXPORTS')).toBeInTheDocument()
  })
  it('labels unfinished sections clearly', () => { render(<App/>); fireEvent.click(screen.getByRole('button', { name: /Credit Intelligence/ })); expect(screen.getByText('This workspace is in development')).toBeInTheDocument() })
  it('keeps snapshot data available when the API is offline', async () => {
    vi.stubGlobal('fetch', vi.fn((url: string) => url.includes('/api/health') ? Promise.reject(new Error('offline')) : Promise.resolve({ ok: true, json: async () => snapshot })) as unknown as typeof fetch)
    render(<App/>); await waitFor(() => expect(screen.getByText('Unavailable')).toBeInTheDocument())
    expect(screen.getByText('6,483')).toBeInTheDocument()
  })
})
