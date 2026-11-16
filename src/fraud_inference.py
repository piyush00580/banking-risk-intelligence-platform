"""Reusable raw-transaction inference for the packaged fraud champion."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "models" / "fraud_detection_xgb.joblib"
OUTPUT_COLUMNS = ["fraud_probability", "predicted_class", "risk_category"]


class FraudInputError(ValueError):
    """Raised when raw transactions do not match the expected fraud input schema."""


def load_fraud_model(artifact_path: str | Path | None = None) -> dict[str, Any]:
    """Load a fraud model bundle saved by ``src/package_fraud_model.py``."""
    path = Path(artifact_path) if artifact_path is not None else DEFAULT_ARTIFACT_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Fraud model artifact not found at {path}. Run `python src/package_fraud_model.py` first."
        )
    bundle = joblib.load(path)
    required = {"model", "feature_engineer", "threshold", "raw_feature_names", "feature_names", "metadata"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(f"Invalid fraud model artifact: expected bundle fields {sorted(required)}")
    if not np.isclose(float(bundle["threshold"]), 0.35, rtol=0.0, atol=0.0):
        raise ValueError("Fraud model artifact threshold must be exactly 0.35")
    return bundle


def _validate_raw_transactions(transactions: pd.DataFrame, raw_feature_names: list[str]) -> pd.DataFrame:
    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("transactions must be a pandas DataFrame of raw transaction features.")
    if transactions.empty:
        raise FraudInputError("transactions must contain at least one row.")
    if transactions.columns.has_duplicates:
        duplicates = transactions.columns[transactions.columns.duplicated()].tolist()
        raise FraudInputError(f"Raw feature columns must be unique; duplicates: {duplicates}")
    if "Class" in transactions.columns:
        raise FraudInputError("Input must not contain the target column 'Class'.")

    missing = [name for name in raw_feature_names if name not in transactions.columns]
    extra = [name for name in transactions.columns if name not in raw_feature_names]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing columns: {missing}")
        if extra:
            details.append(f"unexpected columns: {extra}")
        raise FraudInputError("Raw feature schema mismatch (" + "; ".join(details) + ").")

    frame = transactions.loc[:, raw_feature_names].copy()
    non_numeric = [name for name in raw_feature_names if not pd.api.types.is_numeric_dtype(frame[name])]
    if non_numeric:
        raise FraudInputError(f"All raw features must be numeric; non-numeric columns: {non_numeric}")
    values = frame.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        bad = frame.columns[~np.isfinite(values).all(axis=0)].tolist()
        raise FraudInputError(f"Raw features contain null or infinite values in columns: {bad}")
    if (frame["Amount"] < 0).any():
        raise FraudInputError("Amount must be non-negative; provide the original transaction amount.")
    if (frame["Time"] < 0).any():
        raise FraudInputError("Time must be non-negative elapsed time.")
    return frame


def predict_fraud(transactions: pd.DataFrame, artifact: dict[str, Any] | str | Path | None = None) -> pd.DataFrame:
    """Score raw transaction rows and return probability, class and ordinal risk band.

    Risk bands are score groupings, not calibrated financial-loss probabilities:
    low < 0.10; moderate [0.10, 0.35); high [0.35, 0.75); very_high >= 0.75.
    Class 1 is assigned at probability >= the validation-selected 0.35 threshold.
    """
    bundle = load_fraud_model(artifact) if artifact is None or isinstance(artifact, (str, Path)) else artifact
    raw = _validate_raw_transactions(transactions, bundle["raw_feature_names"])
    engineered = bundle["feature_engineer"].transform(raw)
    expected = list(bundle["feature_names"])
    if list(engineered.columns) != expected:
        raise FraudInputError("Transformed feature names or order do not match the packaged model schema.")
    probabilities = np.asarray(bundle["model"].predict_proba(engineered), dtype=np.float64)[:, 1]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise RuntimeError("Model returned invalid fraud probabilities outside [0, 1].")
    threshold = float(bundle["threshold"])
    predicted = (probabilities >= threshold).astype(np.int64)
    categories = np.select(
        [probabilities < 0.10, probabilities < threshold, probabilities < 0.75],
        ["low", "moderate", "high"], default="very_high"
    )
    return pd.DataFrame({
        "fraud_probability": probabilities,
        "predicted_class": predicted,
        "risk_category": categories,
    }, index=transactions.index)
