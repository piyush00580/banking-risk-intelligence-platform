"""Train and serialize the reproducible Phase 6.5 XGBoost fraud champion."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_advanced_models import build_xgboost_fraud_model, find_best_advanced_threshold
from fraud_feature_engineering import FraudFeatureEngineer, clean_raw_fraud_data, split_fraud_data_stratified

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
ARTIFACT_PATH = ROOT / "models" / "fraud_detection_xgb.joblib"
METADATA_PATH = ROOT / "models" / "fraud_detection_metadata.json"
THRESHOLD = 0.35
CONFIG = {
    "n_estimators": 150, "max_depth": 7, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "scale_pos_weight": 1.0, "random_state": 42, "reg_lambda": 1.0,
}


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    raw_df = pd.read_csv(DATA_PATH)
    cleaned_df = clean_raw_fraud_data(raw_df)
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_fraud_data_stratified(
        cleaned_df, target_col="Class", train_ratio=0.64, val_ratio=0.16,
        test_ratio=0.20, random_state=42,
    )
    feature_engineer = FraudFeatureEngineer(time_scaler="robust", preserve_original=True)
    feature_engineer.fit(X_train_raw)
    X_train = feature_engineer.transform(X_train_raw)
    X_val = feature_engineer.transform(X_val_raw)
    model = build_xgboost_fraud_model(**CONFIG, n_jobs=4)
    model.fit(X_train, y_train)

    # Confirm the established validation operating point without using test data.
    validation_threshold, validation_f1, _ = find_best_advanced_threshold(model, X_val, y_val)
    if validation_threshold != THRESHOLD:
        raise RuntimeError(
            f"Expected validation-selected threshold 0.35; current reproduction selected {validation_threshold:.2f}."
        )
    pre_save = model.predict_proba(X_val.iloc[:32])[:, 1]
    bundle = {
        "model": model,
        "feature_engineer": feature_engineer,
        "threshold": THRESHOLD,
        "raw_feature_names": list(X_train_raw.columns),
        "feature_names": list(X_train.columns),
        "metadata": {
            "model_name": "Fraud Detection XGBoost Champion",
            "model_version": "1.0.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_config": CONFIG,
            "threshold": THRESHOLD,
            "validation_f1_at_threshold": float(validation_f1),
        },
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT_PATH, compress=3)
    reloaded = joblib.load(ARTIFACT_PATH)
    post_save = reloaded["model"].predict_proba(X_val.iloc[:32])[:, 1]
    max_abs_diff = float(np.max(np.abs(pre_save - post_save)))
    if not np.allclose(pre_save, post_save, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"Serialization changed predictions (max absolute difference {max_abs_diff}).")
    if reloaded["feature_names"] != list(X_train.columns):
        raise RuntimeError("Feature names/order did not survive serialization.")
    if reloaded["threshold"] != THRESHOLD:
        raise RuntimeError("Decision threshold did not survive serialization.")

    metadata = {
        "model_name": "Fraud Detection XGBoost Champion",
        "model_version": "1.0.0",
        "created_at_utc": bundle["metadata"]["created_at_utc"],
        "algorithm": "XGBoost binary classifier",
        "training_configuration": CONFIG,
        "decision_threshold": THRESHOLD,
        "risk_categories": {
            "low": "probability < 0.10",
            "moderate": "0.10 <= probability < 0.35",
            "high": "0.35 <= probability < 0.75",
            "very_high": "probability >= 0.75",
            "note": "Score bands are not calibrated financial-loss probabilities.",
        },
        "feature_engineering": {
            "class": "FraudFeatureEngineer(time_scaler='robust', preserve_original=True)",
            "learned_parameters_fit_on": "training partition only",
            "features_added": ["Amount_log", "is_zero_amount", "Time_scaled", "V14_V4_interaction", "V12_V11_interaction"],
            "raw_feature_names_in_order": bundle["raw_feature_names"],
            "model_feature_names_in_order": bundle["feature_names"],
        },
        "dataset": {
            "source_path": "data/raw/creditcard.csv",
            "provenance": "ULB/Worldline European cardholder fraud benchmark; see Phase 6.1 documentation and data acquisition script.",
            "raw_rows": int(len(raw_df)),
            "cleaned_rows": int(len(cleaned_df)),
            "duplicate_rows_removed": int(len(raw_df) - len(cleaned_df)),
            "partitioning": "stratified 64/16/20 via split_fraud_data_stratified, random_state=42",
            "train_rows": int(len(X_train_raw)), "train_fraud_rows": int(y_train.sum()),
            "validation_rows": int(len(X_val_raw)), "validation_fraud_rows": int(y_val.sum()),
            "test_rows": int(len(X_test_raw)), "test_fraud_rows": int(y_test.sum()),
            "test_partition_used_for_fitting_or_threshold_selection": False,
        },
        "validation": {
            "selected_threshold_reproduced": float(validation_threshold),
            "f1_at_threshold": float(validation_f1),
            "serialization_check_rows": 32,
            "serialization_max_absolute_prediction_difference": max_abs_diff,
            "serialization_allclose_atol": 1e-12,
        },
        "environment": {
            "python": platform.python_version(), "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": [
            "Training and validation are from a single highly imbalanced benchmark dataset.",
            "Output probabilities are model scores and have not been calibrated as financial-loss probabilities.",
            "V1-V28 are anonymized PCA features without directly interpretable business meanings.",
            "No transaction-level temporal drift or production latency evaluation is included.",
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved artifact: {ARTIFACT_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Partitions: train={len(X_train_raw):,} ({int(y_train.sum())} fraud), validation={len(X_val_raw):,} ({int(y_val.sum())} fraud), test={len(X_test_raw):,} ({int(y_test.sum())} fraud; not used)")
    print(f"Validation threshold={validation_threshold:.2f}, F1={validation_f1:.4f}")
    print(f"Serialization max |probability difference| on {len(pre_save)} validation rows: {max_abs_diff:.3g}")
    print(f"Raw feature count={len(bundle['raw_feature_names'])}; engineered feature count={len(bundle['feature_names'])}")


if __name__ == "__main__":
    main()
