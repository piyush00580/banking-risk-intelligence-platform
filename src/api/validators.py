"""Request validation for the single-applicant and single-transaction endpoints."""
from __future__ import annotations

import math
from typing import Any

try:
    from ..risk_intelligence_engine import CREDIT_CATEGORICAL_FEATURES, CREDIT_NUMERIC_FEATURES
except ImportError:
    from risk_intelligence_engine import CREDIT_CATEGORICAL_FEATURES, CREDIT_NUMERIC_FEATURES

CREDIT_FIELDS = (
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_default_on_file", "cb_person_cred_hist_length",
)
FRAUD_FIELDS = ("Time", *(f"V{i}" for i in range(1, 29)), "Amount")


class RequestValidationError(ValueError):
    """Invalid endpoint payload with a safe client-facing explanation."""


def _check_object_fields(payload: dict[str, Any], fields: tuple[str, ...], name: str) -> None:
    if not isinstance(payload, dict):
        raise RequestValidationError("The JSON request body must be an object.")
    missing = [field for field in fields if field not in payload]
    extra = [field for field in payload if field not in fields]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing fields: {missing}")
        if extra:
            parts.append(f"unexpected fields: {extra}")
        raise RequestValidationError(f"Invalid {name} request (" + "; ".join(parts) + ").")


def validate_credit_payload(payload: Any) -> dict[str, Any]:
    """Validate a credit applicant while preserving nulls handled by fitted imputers."""
    _check_object_fields(payload, CREDIT_FIELDS, "credit-risk")
    validated = dict(payload)
    for field in CREDIT_NUMERIC_FEATURES:
        value = validated[field]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RequestValidationError(f"Field '{field}' must be a finite number or null.")
        try:
            finite = math.isfinite(float(value))
        except (OverflowError, ValueError):
            finite = False
        if not finite:
            raise RequestValidationError(f"Field '{field}' must be finite.")
    for field in CREDIT_CATEGORICAL_FEATURES:
        value = validated[field]
        if value is not None and not isinstance(value, str):
            raise RequestValidationError(f"Field '{field}' must be text or null.")
    return validated


def validate_fraud_payload(payload: Any) -> dict[str, float]:
    """Validate one complete raw fraud transaction; Class is never accepted."""
    if isinstance(payload, dict) and "Class" in payload:
        raise RequestValidationError("Field 'Class' is the target and must not be included.")
    _check_object_fields(payload, FRAUD_FIELDS, "fraud-risk")
    validated: dict[str, float] = {}
    for field in FRAUD_FIELDS:
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RequestValidationError(f"Field '{field}' must be a finite number.")
        try:
            number = float(value)
        except (OverflowError, ValueError):
            raise RequestValidationError(f"Field '{field}' must be finite.") from None
        if not math.isfinite(number):
            raise RequestValidationError(f"Field '{field}' must be finite.")
        validated[field] = number
    if validated["Amount"] < 0:
        raise RequestValidationError("Field 'Amount' must be non-negative.")
    if validated["Time"] < 0:
        raise RequestValidationError("Field 'Time' must be non-negative.")
    return validated
