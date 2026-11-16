"""API route handlers; model logic remains in the Phase 7 engine."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Blueprint, current_app, request
from werkzeug.exceptions import BadRequest, HTTPException, UnsupportedMediaType

try:
    from ..fraud_inference import FraudInputError
    from ..risk_intelligence_engine import RiskInputError, model_cache_status, score_credit_risk, score_transaction_fraud
except ImportError:
    from fraud_inference import FraudInputError
    from risk_intelligence_engine import RiskInputError, model_cache_status, score_credit_risk, score_transaction_fraud
from .serializers import error_response, success_response
from .validators import RequestValidationError, validate_credit_payload, validate_fraud_payload

api = Blueprint("api", __name__, url_prefix="/api")


def _request_object():
    if not request.is_json:
        return None, error_response(400, "INVALID_JSON", "Content-Type must be application/json.")
    try:
        payload = request.get_json(silent=False)
    except (BadRequest, UnsupportedMediaType):
        return None, error_response(400, "INVALID_JSON", "Request body must contain valid JSON.")
    if not isinstance(payload, dict):
        return None, error_response(400, "MALFORMED_REQUEST", "JSON request body must be an object.")
    return payload, None


@api.get("/health")
def health():
    """Report API availability separately from local artifact readiness."""
    credit_artifact = Path(current_app.config["CREDIT_ARTIFACT_PATH"])
    credit_metadata = Path(current_app.config["CREDIT_METADATA_PATH"])
    fraud_artifact = Path(current_app.config["FRAUD_ARTIFACT_PATH"])
    credit_available = credit_artifact.is_file() and credit_metadata.is_file()
    fraud_available = fraud_artifact.is_file()
    loaded = model_cache_status()
    return {
        "success": True,
        "data": {
            "status": "available",
            "service_name": current_app.config["SERVICE_NAME"],
            "api_version": current_app.config["API_VERSION"],
            "model_readiness": {
                "ready": bool(credit_available and fraud_available),
                "credit_risk": {
                    "artifact_available": credit_artifact.is_file(),
                    "metadata_available": credit_metadata.is_file(),
                    "loaded": loaded["credit_risk"],
                },
                "transaction_fraud": {
                    "artifact_available": fraud_available,
                    "loaded": loaded["transaction_fraud"],
                },
            },
        },
    }


@api.post("/credit-risk/score")
def credit_risk_score():
    payload, error = _request_object()
    if error is not None:
        return error
    try:
        applicant = validate_credit_payload(payload)
        result = score_credit_risk(
            pd.DataFrame([applicant]),
            artifact_path=current_app.config["CREDIT_ARTIFACT_PATH"],
            metadata_path=current_app.config["CREDIT_METADATA_PATH"],
        )
        return success_response(result)
    except RequestValidationError as exc:
        return error_response(422, "VALIDATION_ERROR", str(exc))
    except (RiskInputError, FraudInputError) as exc:
        return error_response(422, "VALIDATION_ERROR", str(exc))
    except FileNotFoundError:
        return error_response(503, "MODEL_NOT_READY", "A required risk model is unavailable.")
    except (ValueError, TypeError) as exc:
        # Artifact/metadata failures are readiness errors; request fields were already validated.
        current_app.logger.warning("Credit scoring configuration or inference validation failed: %s", type(exc).__name__)
        return error_response(503, "MODEL_NOT_READY", "The credit-risk model is not ready.")
    except Exception:
        current_app.logger.exception("Unexpected credit-risk inference error")
        return error_response(500, "INTERNAL_ERROR", "An unexpected server error occurred.")


@api.post("/fraud-risk/score")
def fraud_risk_score():
    payload, error = _request_object()
    if error is not None:
        return error
    try:
        transaction = validate_fraud_payload(payload)
        result = score_transaction_fraud(
            pd.DataFrame([transaction]), artifact_path=current_app.config["FRAUD_ARTIFACT_PATH"]
        )
        return success_response(result)
    except RequestValidationError as exc:
        return error_response(422, "VALIDATION_ERROR", str(exc))
    except (RiskInputError, FraudInputError) as exc:
        return error_response(422, "VALIDATION_ERROR", str(exc))
    except FileNotFoundError:
        return error_response(503, "MODEL_NOT_READY", "A required risk model is unavailable.")
    except (ValueError, TypeError) as exc:
        current_app.logger.warning("Fraud scoring configuration or inference validation failed: %s", type(exc).__name__)
        return error_response(503, "MODEL_NOT_READY", "The fraud-risk model is not ready.")
    except Exception:
        current_app.logger.exception("Unexpected fraud-risk inference error")
        return error_response(500, "INTERNAL_ERROR", "An unexpected server error occurred.")


def register_route_error_handlers(app):
    """Install consistent JSON responses for unsupported routes and methods."""
    @app.errorhandler(404)
    def not_found(_error):
        return error_response(404, "NOT_FOUND", "The requested API route was not found.")

    @app.errorhandler(405)
    def method_not_allowed(_error):
        return error_response(405, "METHOD_NOT_ALLOWED", "The HTTP method is not supported for this route.")

    @app.errorhandler(HTTPException)
    def http_error(error):
        if error.code == 400:
            return error_response(400, "INVALID_JSON", "Request body must contain valid JSON.")
        return error_response(int(error.code or 500), "HTTP_ERROR", "The request could not be processed.")
