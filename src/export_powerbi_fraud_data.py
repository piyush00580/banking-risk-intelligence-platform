"""Export the established held-out fraud test split for Power BI."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_feature_engineering import clean_raw_fraud_data, split_fraud_data_stratified
from fraud_inference import DEFAULT_ARTIFACT_PATH, load_fraud_model, predict_fraud

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "powerbi"
SCORED_CSV = OUTPUT_DIR / "fraud_test_scored.csv"
KPI_CSV = OUTPUT_DIR / "fraud_kpi_summary.csv"
METADATA_JSON = OUTPUT_DIR / "fraud_dataset_metadata.json"
EXPECTED_THRESHOLD = 0.35


def compute_fraud_test_metrics(actual: pd.Series, scores: pd.Series, threshold: float) -> dict[str, Any]:
    """Calculate test-only confusion counts and standard ranking/classification metrics."""
    y_true = np.asarray(actual, dtype=np.int64)
    probabilities = np.asarray(scores, dtype=np.float64)
    if len(y_true) != len(probabilities) or len(y_true) == 0:
        raise ValueError("Actual labels and fraud scores must have the same non-zero length.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Fraud labels must be binary 0/1 values.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Fraud probabilities must be finite and in [0, 1].")
    if len(np.unique(y_true)) != 2:
        raise ValueError("Held-out test metrics require both legitimate and fraud labels.")
    prediction = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "test_total_transactions": int(len(y_true)),
        "test_actual_fraud_transactions": int(y_true.sum()),
        "test_actual_legitimate_transactions": int(len(y_true) - y_true.sum()),
        "test_fraud_prevalence": float(y_true.mean()),
        "test_true_positives": int(tp),
        "test_false_positives": int(fp),
        "test_true_negatives": int(tn),
        "test_false_negatives": int(fn),
        "test_precision": float(precision_score(y_true, prediction, zero_division=0)),
        "test_recall": float(recall_score(y_true, prediction, zero_division=0)),
        "test_f1_score": float(f1_score(y_true, prediction, zero_division=0)),
        "test_pr_auc_average_precision": float(average_precision_score(y_true, probabilities)),
        "test_roc_auc": float(roc_auc_score(y_true, probabilities)),
        "decision_threshold": float(threshold),
    }


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.10g", lineterminator="\n")
    temporary.replace(destination)


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def export_fraud_data(
    data_path: str | Path = DATA_PATH,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    output_dir: str | Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Recreate the Phase 6 test split, score it with the saved bundle, and export data/KPIs."""
    source = Path(data_path)
    artifact = Path(artifact_path)
    destination = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(f"Fraud benchmark dataset not found: {source}")
    if not artifact.is_file():
        raise FileNotFoundError(f"Packaged fraud model artifact not found: {artifact}")

    raw = pd.read_csv(source)
    cleaned = clean_raw_fraud_data(raw)
    X_train, X_val, X_test, y_train, y_val, y_test = split_fraud_data_stratified(
        cleaned, target_col="Class", train_ratio=0.64, val_ratio=0.16,
        test_ratio=0.20, random_state=42,
    )
    # Sort only for stable export order; membership and split indices stay unchanged.
    X_test = X_test.sort_index()
    y_test = y_test.loc[X_test.index]

    bundle = load_fraud_model(artifact)
    threshold = float(bundle["threshold"])
    if threshold != EXPECTED_THRESHOLD:
        raise ValueError(f"Expected the established fraud threshold {EXPECTED_THRESHOLD}; artifact has {threshold}.")
    prediction = predict_fraud(X_test, artifact=bundle)
    if not prediction.index.equals(X_test.index):
        raise RuntimeError("Fraud inference row alignment changed during scoring.")
    scores = prediction["fraud_probability"].astype(float)
    if scores.isna().any() or not np.isfinite(scores.to_numpy()).all():
        raise RuntimeError("Fraud inference produced missing or infinite scores.")
    expected_class = (scores.to_numpy() >= threshold).astype(np.int64)
    if not np.array_equal(expected_class, prediction["predicted_class"].to_numpy(dtype=np.int64)):
        raise RuntimeError("Fraud predictions do not match the packaged 0.35 threshold.")
    if y_test.isna().any() or not y_test.isin([0, 1]).all():
        raise RuntimeError("Held-out fraud labels contain missing or invalid values.")

    row_ids = [f"fraud_benchmark_test_row_{int(index):06d}" for index in X_test.index]
    scored = X_test.copy()
    scored.insert(0, "dataset_row_id", row_ids)
    scored["Class"] = y_test.astype(np.int64)
    scored["fraud_probability"] = scores.to_numpy()
    scored["predicted_class"] = prediction["predicted_class"].astype(np.int64).to_numpy()
    scored["risk_category"] = prediction["risk_category"].astype(str).to_numpy()
    scored["decision_threshold"] = threshold
    scored["evaluation_partition"] = "held_out_test"
    raw_columns = list(X_test.columns)
    export_columns = [
        "dataset_row_id", *raw_columns, "Class", "fraud_probability", "predicted_class",
        "risk_category", "decision_threshold", "evaluation_partition",
    ]
    scored = scored.loc[:, export_columns]
    metrics = compute_fraud_test_metrics(y_test, scores, threshold)
    kpi = pd.DataFrame([metrics])

    destination.mkdir(parents=True, exist_ok=True)
    scored_path = destination / SCORED_CSV.name
    kpi_path = destination / KPI_CSV.name
    metadata_path = destination / METADATA_JSON.name
    _atomic_csv(scored, scored_path)
    _atomic_csv(kpi, kpi_path)
    model_metadata = bundle.get("metadata", {})
    metadata = {
        "dataset_name": "ULB/Worldline European cardholder fraud benchmark held-out test partition",
        "source_file": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else source.name,
        "source_provenance": "See Phase 6.1 documentation and src/download_fraud_data.py; no new data was downloaded.",
        "raw_row_count": int(len(raw)),
        "cleaned_row_count": int(len(cleaned)),
        "duplicate_rows_removed": int(len(raw) - len(cleaned)),
        "partitioning": {
            "method": "clean_raw_fraud_data followed by split_fraud_data_stratified (two-stage stratified split)",
            "train_ratio": 0.64,
            "validation_ratio": 0.16,
            "test_ratio": 0.20,
            "random_state": 42,
            "train_rows": int(len(X_train)),
            "train_fraud_rows": int(y_train.sum()),
            "validation_rows": int(len(X_val)),
            "validation_fraud_rows": int(y_val.sum()),
            "test_rows": int(len(X_test)),
            "test_fraud_rows": int(y_test.sum()),
            "test_partition_used_for_fit_or_threshold_selection": False,
        },
        "model": {
            "name": model_metadata.get("model_name", "Fraud Detection XGBoost Champion"),
            "version": model_metadata.get("model_version"),
            "artifact": str(artifact.relative_to(ROOT)) if artifact.is_relative_to(ROOT) else artifact.name,
            "decision_threshold": threshold,
            "risk_categories": {
                "low": "probability < 0.10",
                "moderate": "0.10 <= probability < 0.35",
                "high": "0.35 <= probability < 0.75",
                "very_high": "probability >= 0.75",
            },
        },
        "export": {
            "scored_csv": scored_path.name,
            "kpi_summary_csv": kpi_path.name,
            "row_identifier": "dataset_row_id is a generated ID from the cleaned benchmark CSV row index; it is not a real transaction or customer ID.",
            "scored_row_count": int(len(scored)),
            "columns": export_columns,
            "kpi_metrics": metrics,
        },
        "limitations": [
            "This export is the benchmark's held-out test partition, not live transaction monitoring.",
            "Time is elapsed seconds within the source recording and is not a calendar timestamp or time of day.",
            "V1-V28 are anonymized PCA components and have no documented direct business meaning.",
            "Fraud probabilities and risk categories are model outputs, not calibrated financial-loss probabilities.",
            "The benchmark fraud data is independent of the credit-risk dataset and has no shared customer key.",
        ],
    }
    _atomic_json(metadata, metadata_path)
    print(f"Fraud test CSV: {scored_path} ({len(scored):,} rows)")
    print(f"Fraud KPI CSV: {kpi_path}")
    print(f"Fraud metadata: {metadata_path}")
    print(json.dumps(metrics, indent=2))
    return metadata


if __name__ == "__main__":
    export_fraud_data()
