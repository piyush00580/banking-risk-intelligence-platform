"""Independent credit-default and transaction-fraud scoring pathways."""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_DIR = str(PROJECT_ROOT / "src")
# The fraud artifact stores its transformer under this top-level module name.
# Make the project source directory importable before loading that trusted artifact.
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

try:  # Support both `import src.risk_intelligence_engine` and src/ on sys.path.
    from .fraud_inference import load_fraud_model, predict_fraud
except ImportError:  # Direct-module imports used by the existing tests and scripts.
    from fraud_inference import load_fraud_model, predict_fraud

DEFAULT_CREDIT_ARTIFACT = PROJECT_ROOT / "models" / "credit_risk_xgb.joblib"
DEFAULT_CREDIT_METADATA = PROJECT_ROOT / "models" / "credit_risk_metadata.json"
DEFAULT_CREDIT_THRESHOLD = 0.45
CREDIT_NUMERIC_FEATURES = (
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
)
CREDIT_CATEGORICAL_FEATURES = (
    "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file",
)
RESPONSE_COLUMNS = [
    "risk_type", "model_name", "model_version", "risk_score", "predicted_class",
    "risk_category", "decision_threshold", "score_interpretation", "metadata",
]


class RiskInputError(ValueError):
    """Raised when inputs do not match the selected model's raw feature contract."""


@lru_cache(maxsize=4)
def _load_fraud_bundle_cached(artifact_path: str):
    """Keep each trusted fraud artifact loaded once per process and resolved path."""
    return load_fraud_model(artifact_path)


def clear_model_caches() -> None:
    """Clear process-local loaded model caches, for controlled artifact refreshes/tests."""
    load_credit_model.cache_clear()
    _load_fraud_bundle_cached.cache_clear()


def model_cache_status() -> dict[str, bool]:
    """Return whether each model family has a cached instance in this process."""
    return {
        "credit_risk": load_credit_model.cache_info().currsize > 0,
        "transaction_fraud": _load_fraud_bundle_cached.cache_info().currsize > 0,
    }


@lru_cache(maxsize=4)
def load_credit_model(
    artifact_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load the existing serialized credit pipeline and its required metadata."""
    artifact = Path(artifact_path) if artifact_path is not None else DEFAULT_CREDIT_ARTIFACT
    metadata_file = Path(metadata_path) if metadata_path is not None else DEFAULT_CREDIT_METADATA
    if not artifact.is_file():
        raise FileNotFoundError(f"Credit-risk artifact not found at {artifact}.")
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Credit-risk metadata not found at {metadata_file}.")
    model = joblib.load(artifact)
    if not hasattr(model, "predict_proba") or not hasattr(model, "feature_names_in_"):
        raise ValueError("Credit-risk artifact must be a fitted pipeline with predict_proba and feature_names_in_.")
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        schema = metadata["feature_schema"]
        thresholds = metadata["decision_thresholds"]
        threshold = float(thresholds["selected_optimal_threshold"])
        model_name = metadata["model_name"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Credit-risk metadata is malformed or incomplete: {metadata_file}") from exc
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Credit-risk metadata must define a decision threshold in [0, 1].")
    expected = list(model.feature_names_in_)
    metadata_features = list(schema.get("numerical_features", [])) + list(schema.get("categorical_features", []))
    if len(expected) != len(set(expected)) or set(expected) != set(metadata_features):
        raise ValueError("Credit-risk artifact feature schema does not match its metadata.")
    return model, metadata


def _credit_frame(applicant_data: pd.DataFrame | Mapping[str, Any], feature_names: list[str]) -> pd.DataFrame:
    if isinstance(applicant_data, pd.DataFrame):
        frame = applicant_data.copy()
    elif isinstance(applicant_data, Mapping):
        frame = pd.DataFrame([dict(applicant_data)])
    else:
        raise TypeError("applicant_data must be a pandas DataFrame or a mapping for one applicant.")
    if frame.empty:
        raise RiskInputError("applicant_data must contain at least one applicant.")
    if frame.columns.has_duplicates:
        duplicate_names = frame.columns[frame.columns.duplicated()].tolist()
        raise RiskInputError(f"Applicant columns must be unique; duplicates: {duplicate_names}")
    missing = [name for name in feature_names if name not in frame.columns]
    extra = [name for name in frame.columns if name not in feature_names]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing columns: {missing}")
        if extra:
            details.append(f"unexpected columns: {extra}")
        raise RiskInputError("Credit applicant schema mismatch (" + "; ".join(details) + ").")

    frame = frame.loc[:, feature_names]
    for name in CREDIT_NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(frame[name]):
            raise RiskInputError(f"Credit numeric feature '{name}' must contain numeric values.")
        numeric = frame[name].to_numpy(dtype=np.float64, na_value=np.nan)
        if np.isinf(numeric).any():
            raise RiskInputError(f"Credit numeric feature '{name}' contains an infinite value.")
    for name in CREDIT_CATEGORICAL_FEATURES:
        invalid = [value for value in frame[name].dropna().unique() if not isinstance(value, str)]
        if invalid:
            raise RiskInputError(f"Credit categorical feature '{name}' must contain text values or missing values.")
    # Missing values are allowed because the serialized training pipeline includes its fitted imputers.
    return frame


def _response_frame(
    index: pd.Index,
    *, risk_type: str,
    model_name: str,
    model_version: str | None,
    scores: np.ndarray,
    predictions: np.ndarray,
    categories: list[str],
    threshold: float,
    interpretation: str,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    count = len(scores)
    return pd.DataFrame({
        "risk_type": [risk_type] * count,
        "model_name": [model_name] * count,
        "model_version": [model_version] * count,
        "risk_score": scores.astype(float),
        "predicted_class": predictions.astype(int),
        "risk_category": categories,
        "decision_threshold": [float(threshold)] * count,
        "score_interpretation": [interpretation] * count,
        "metadata": [metadata.copy() for _ in range(count)],
    }, index=index, columns=RESPONSE_COLUMNS)


def score_credit_risk(
    applicant_data: pd.DataFrame | Mapping[str, Any],
    artifact_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> pd.DataFrame:
    """Score one or more applicants for loan default using the existing credit pipeline.

    The returned ``risk_score`` is the model estimate of loan default. Risk tiers
    retain the credit notebook interpretation: Tier 1 below 0.15, Tier 2 from
    0.15 to below the selected 0.45 threshold, and Tier 3 at or above 0.45.
    """
    model, metadata = load_credit_model(artifact_path, metadata_path)
    frame = _credit_frame(applicant_data, list(model.feature_names_in_))
    probabilities = np.asarray(model.predict_proba(frame), dtype=np.float64)[:, 1]
    if not np.isfinite(probabilities).all() or ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise RuntimeError("Credit model returned invalid probabilities outside [0, 1].")
    threshold = float(metadata["decision_thresholds"]["selected_optimal_threshold"])
    predictions = (probabilities >= threshold).astype(int)
    categories = [
        "Tier 1 (Prime)" if score < 0.15 else
        "Tier 2 (Moderate)" if score < threshold else
        "Tier 3 (High Default Risk)"
        for score in probabilities
    ]
    response_metadata = {
        "decision_threshold_selection": metadata["decision_thresholds"].get("selection_criterion"),
        "default_threshold": metadata["decision_thresholds"].get("default_threshold"),
        "artifact": str(Path(artifact_path) if artifact_path is not None else DEFAULT_CREDIT_ARTIFACT.relative_to(PROJECT_ROOT)),
    }
    return _response_frame(
        frame.index, risk_type="credit_default", model_name=str(metadata["model_name"]),
        model_version=metadata.get("model_version"), scores=probabilities, predictions=predictions,
        categories=categories, threshold=threshold,
        interpretation="Estimated probability of loan default from the credit-risk model.",
        metadata=response_metadata,
    )


def score_transaction_fraud(
    transaction_data: pd.DataFrame,
    artifact_path: str | Path | None = None,
) -> pd.DataFrame:
    """Score raw transaction rows using the existing fraud inference module.

    ``risk_score`` is the model estimate of transaction fraud, not loan default.
    Fraud preprocessing, threshold and risk bands remain owned by fraud_inference.
    """
    fraud_artifact = Path(artifact_path).resolve() if artifact_path is not None else (PROJECT_ROOT / "models" / "fraud_detection_xgb.joblib")
    bundle = _load_fraud_bundle_cached(str(fraud_artifact))
    predictions = predict_fraud(transaction_data, artifact=bundle)
    scores = predictions["fraud_probability"].to_numpy(dtype=np.float64)
    classes = predictions["predicted_class"].to_numpy(dtype=int)
    threshold = float(bundle["threshold"])
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise RuntimeError("Fraud inference returned invalid probabilities outside [0, 1].")
    if not np.array_equal(classes, (scores >= threshold).astype(int)):
        raise RuntimeError("Fraud inference class does not match the saved decision threshold.")
    model_metadata = bundle.get("metadata", {})
    response_metadata = {
        "risk_bands": {
            "low": "probability < 0.10",
            "moderate": "0.10 <= probability < 0.35",
            "high": "0.35 <= probability < 0.75",
            "very_high": "probability >= 0.75",
        },
        "artifact": str(Path(artifact_path) if artifact_path is not None else Path("models/fraud_detection_xgb.joblib")),
    }
    return _response_frame(
        predictions.index, risk_type="transaction_fraud",
        model_name=str(model_metadata.get("model_name", "Fraud Detection XGBoost Champion")),
        model_version=model_metadata.get("model_version"), scores=scores, predictions=classes,
        categories=predictions["risk_category"].astype(str).tolist(), threshold=threshold,
        interpretation="Estimated probability that the transaction is fraudulent.",
        metadata=response_metadata,
    )
