"""Score the cleaned credit benchmark with the saved model for Power BI."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from risk_intelligence_engine import DEFAULT_CREDIT_ARTIFACT, DEFAULT_CREDIT_METADATA, load_credit_model

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "credit_risk_dataset.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "powerbi"
SCORED_CSV = OUTPUT_DIR / "credit_scored.csv"
METADATA_JSON = OUTPUT_DIR / "credit_dataset_metadata.json"
EXPECTED_THRESHOLD = 0.45


def prepare_credit_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the existing credit notebook's duplicate and anomaly handling exactly."""
    before = len(raw)
    clean = raw.drop_duplicates(keep="first").copy()
    duplicate_rows_removed = before - len(clean)
    invalid_age_rows = int((clean["person_age"] > 100).sum())
    clean = clean.loc[clean["person_age"] <= 100].copy()
    invalid_employment_rows = int((clean["person_emp_length"] > clean["person_age"]).sum())
    clean.loc[clean["person_emp_length"] > clean["person_age"], "person_emp_length"] = np.nan
    return clean, {
        "raw_rows": int(before),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "age_over_100_rows_removed": invalid_age_rows,
        "employment_length_rows_masked": invalid_employment_rows,
        "cleaned_rows": int(len(clean)),
    }


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.10g", lineterminator="\n")
    temporary.replace(destination)


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def export_credit_data(
    data_path: str | Path = DATA_PATH,
    artifact_path: str | Path = DEFAULT_CREDIT_ARTIFACT,
    metadata_path: str | Path = DEFAULT_CREDIT_METADATA,
    output_dir: str | Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Score all reproducibly cleaned credit rows; retain split membership and observed labels."""
    source = Path(data_path)
    artifact = Path(artifact_path)
    model_metadata_path = Path(metadata_path)
    destination = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(f"Credit benchmark dataset not found: {source}")
    if not artifact.is_file() or not model_metadata_path.is_file():
        raise FileNotFoundError("Credit model artifact and metadata are required for export.")

    raw = pd.read_csv(source)
    clean, cleaning = prepare_credit_dataset(raw)
    if "loan_status" not in clean.columns:
        raise ValueError("Credit dataset must contain the actual loan_status target column.")
    X = clean.drop(columns=["loan_status"])
    y = clean["loan_status"].astype(np.int64)
    X_train, X_test, _y_train, _y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    model, model_metadata = load_credit_model(artifact, model_metadata_path)
    threshold = float(model_metadata["decision_thresholds"]["selected_optimal_threshold"])
    if threshold != EXPECTED_THRESHOLD:
        raise ValueError(f"Expected the established credit threshold {EXPECTED_THRESHOLD}; metadata has {threshold}.")
    model_features = list(model.feature_names_in_)
    missing = [column for column in model_features if column not in X.columns]
    extra = [column for column in X.columns if column not in model_features]
    if missing or extra:
        raise ValueError(f"Credit feature schema mismatch; missing={missing}, unexpected={extra}.")
    X = X.loc[:, model_features]
    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)[:, 1]
    if len(probabilities) != len(clean) or not np.isfinite(probabilities).all():
        raise RuntimeError("Credit model returned missing or non-finite probabilities.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise RuntimeError("Credit model returned probability scores outside [0, 1].")
    predicted = (probabilities >= threshold).astype(np.int64)
    tiers = np.select(
        [probabilities < 0.15, probabilities < threshold],
        ["Tier 1 (Prime)", "Tier 2 (Moderate)"],
        default="Tier 3 (High Default Risk)",
    )
    train_indices, test_indices = set(X_train.index), set(X_test.index)
    partition = np.where(clean.index.isin(train_indices), "train_in_sample", "held_out_test")
    if len(train_indices & test_indices) or len(train_indices | test_indices) != len(clean):
        raise RuntimeError("Credit train/test partition membership is not a complete disjoint split.")

    scored = X.copy()
    scored.insert(0, "credit_dataset_row_id", [f"credit_benchmark_row_{int(i):06d}" for i in clean.index])
    scored["loan_status"] = y.to_numpy()
    scored["predicted_default_probability"] = probabilities
    scored["predicted_class"] = predicted
    scored["risk_tier"] = tiers
    scored["model_partition"] = partition
    scored_columns = [
        "credit_dataset_row_id", *model_features, "loan_status", "predicted_default_probability",
        "predicted_class", "risk_tier", "model_partition",
    ]
    scored = scored.loc[:, scored_columns]

    destination.mkdir(parents=True, exist_ok=True)
    scored_path = destination / SCORED_CSV.name
    metadata_path_out = destination / METADATA_JSON.name
    _atomic_csv(scored, scored_path)
    metadata = {
        "dataset_name": "Cleaned credit-risk benchmark dataset scored by the saved champion",
        "source_file": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else source.name,
        "model": {
            "name": model_metadata["model_name"],
            "version": model_metadata.get("model_version"),
            "artifact": str(artifact.relative_to(ROOT)) if artifact.is_relative_to(ROOT) else artifact.name,
            "decision_threshold": threshold,
            "risk_tiers": {
                "Tier 1 (Prime)": "probability < 0.15",
                "Tier 2 (Moderate)": "0.15 <= probability < 0.45",
                "Tier 3 (High Default Risk)": "probability >= 0.45",
            },
        },
        "cleaning": cleaning,
        "partitioning": {
            "method": "Stratified train_test_split matching notebooks/credit_risk_modeling.ipynb",
            "train_ratio": 0.80,
            "test_ratio": 0.20,
            "random_state": 42,
            "train_rows": int(len(X_train)),
            "held_out_test_rows": int(len(X_test)),
        },
        "export": {
            "file": scored_path.name,
            "row_count": int(len(scored)),
            "columns": scored_columns,
            "row_identifier": "credit_dataset_row_id is a generated label for original source CSV row position; it is not a customer_id.",
            "loan_status": "Observed target retained separately; model outputs are predictions, not observed outcomes.",
            "prediction_scope": "Training rows are in-sample predictions; rows marked held_out_test are the unchanged original model test split. No threshold was retuned.",
        },
        "limitations": [
            "The full export includes in-sample predictions on training rows; use model_partition to filter held_out_test for the original held-out assessment.",
            "This export's row identifier does not map to PostgreSQL customer_id or loans.loan_id.",
            "The credit benchmark is independent of the fraud benchmark; the datasets must not be joined.",
        ],
    }
    _atomic_json(metadata, metadata_path_out)
    print(f"Credit scored CSV: {scored_path} ({len(scored):,} rows)")
    print(f"Credit metadata: {metadata_path_out}")
    print(f"Training rows (in-sample): {len(X_train):,}; held-out rows: {len(X_test):,}")
    return metadata


if __name__ == "__main__":
    export_credit_data()
