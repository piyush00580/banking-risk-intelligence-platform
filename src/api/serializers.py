"""JSON-safe serialization for risk engine DataFrame responses."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from flask import jsonify

try:
    from ..risk_intelligence_engine import RESPONSE_COLUMNS
except ImportError:
    from risk_intelligence_engine import RESPONSE_COLUMNS


def json_safe(value: Any) -> Any:
    """Recursively convert pandas/NumPy values to strict JSON values."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def success_response(result: pd.DataFrame):
    """Wrap a one-row engine result in the public success/data envelope."""
    if not isinstance(result, pd.DataFrame) or len(result) != 1:
        raise ValueError("Risk engine must return exactly one result row for these endpoints.")
    missing = [column for column in RESPONSE_COLUMNS if column not in result.columns]
    if missing:
        raise ValueError(f"Risk engine response is missing fields: {missing}")
    data = {column: json_safe(result.iloc[0][column]) for column in RESPONSE_COLUMNS}
    return jsonify({"success": True, "data": data})


def error_response(status: int, code: str, message: str):
    """Create the standard non-leaking API error response."""
    return jsonify({"success": False, "error": {"code": code, "message": message}}), status
