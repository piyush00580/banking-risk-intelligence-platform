export type Health = {
  status: string; service_name: string; api_version: string
  model_readiness: {
    ready: boolean
    credit_risk: { artifact_available: boolean; metadata_available: boolean; loaded: boolean }
    transaction_fraud: { artifact_available: boolean; loaded: boolean }
  }
}
export type Snapshot = {
  source: string
  credit: { scope: string; applications: number; observed_defaults: number; observed_default_rate: number; mean_predicted_default_probability: number; by_grade: { grade: string; applications: number; observed_defaults: number; mean_predicted_default_probability: number }[]; in_sample_rows_excluded: number }
  fraud: { scope: string; transactions: number; fraud: number; legitimate: number; true_positives: number; false_positives: number; true_negatives: number; false_negatives: number; precision: number; recall: number; f1: number; pr_auc: number; roc_auc: number; threshold: number }
}
