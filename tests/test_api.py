"""Flask API contract tests using mocked Phase 7 inference functions."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.api.app import create_app
from src.api.validators import CREDIT_FIELDS, FRAUD_FIELDS
from src.risk_intelligence_engine import RESPONSE_COLUMNS, clear_model_caches


CREDIT_PAYLOAD = {
    "person_age": 38, "person_income": 110000.0, "person_home_ownership": "OWN",
    "person_emp_length": 12.0, "loan_intent": "HOMEIMPROVEMENT", "loan_grade": "A",
    "loan_amnt": 5000.0, "loan_int_rate": 7.25, "loan_percent_income": 0.05,
    "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 10,
}
FRAUD_PAYLOAD = {name: 0.0 for name in FRAUD_FIELDS}
FRAUD_PAYLOAD["Time"] = 1000.0
FRAUD_PAYLOAD["Amount"] = 12.5


def engine_result(risk_type: str, score: float, category: str, threshold: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "risk_type": risk_type,
        "model_name": "Example Model",
        "model_version": "1.0.0",
        "risk_score": np.float32(score),
        "predicted_class": np.int64(score >= threshold),
        "risk_category": category,
        "decision_threshold": np.float64(threshold),
        "score_interpretation": f"Example {risk_type} score.",
        "metadata": {"checks": [np.int64(1)], "threshold_copy": np.float32(threshold)},
    }], columns=RESPONSE_COLUMNS)


class RiskApiTests(unittest.TestCase):
    def setUp(self):
        clear_model_caches()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.credit_artifact = self.root / "credit.joblib"
        self.credit_metadata = self.root / "credit.json"
        self.fraud_artifact = self.root / "fraud.joblib"
        self.app = create_app({
            "TESTING": True,
            "CREDIT_ARTIFACT_PATH": str(self.credit_artifact),
            "CREDIT_METADATA_PATH": str(self.credit_metadata),
            "FRAUD_ARTIFACT_PATH": str(self.fraud_artifact),
        })
        self.client = self.app.test_client()

    def post_json(self, route: str, payload):
        return self.client.post(route, json=payload)

    def test_health_separates_service_availability_from_model_readiness(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()["data"]
        self.assertEqual(data["status"], "available")
        self.assertEqual(data["service_name"], "Banking Risk Intelligence API")
        self.assertFalse(data["model_readiness"]["ready"])
        self.assertFalse(data["model_readiness"]["credit_risk"]["loaded"])
        for artifact in (self.credit_artifact, self.credit_metadata, self.fraud_artifact):
            artifact.touch()
        ready = self.client.get("/api/health").get_json()["data"]["model_readiness"]
        self.assertTrue(ready["ready"])
        self.assertFalse(ready["transaction_fraud"]["loaded"])

    def test_credit_success_uses_engine_and_serializes_all_fields(self):
        expected = engine_result("credit_default", 0.63, "Tier 3 (High Default Risk)", 0.45)
        with patch("src.api.routes.score_credit_risk", return_value=expected) as scorer:
            response = self.post_json("/api/credit-risk/score", CREDIT_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["success"])
        self.assertEqual(list(body["data"]), RESPONSE_COLUMNS)
        self.assertEqual(body["data"]["risk_type"], "credit_default")
        self.assertEqual(body["data"]["risk_score"], float(np.float32(0.63)))
        self.assertEqual(body["data"]["predicted_class"], 1)
        self.assertEqual(body["data"]["decision_threshold"], 0.45)
        self.assertEqual(body["data"]["metadata"]["checks"], [1])
        scorer.assert_called_once()
        args, kwargs = scorer.call_args
        self.assertEqual(list(args[0].columns), list(CREDIT_FIELDS))
        self.assertEqual(kwargs["artifact_path"], str(self.credit_artifact))

    def test_fraud_success_preserves_engine_result(self):
        expected = engine_result("transaction_fraud", 0.8, "very_high", 0.35)
        with patch("src.api.routes.score_transaction_fraud", return_value=expected) as scorer:
            response = self.post_json("/api/fraud-risk/score", FRAUD_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()["data"]
        self.assertEqual(list(data), RESPONSE_COLUMNS)
        self.assertEqual(data["risk_type"], "transaction_fraud")
        self.assertEqual(data["risk_score"], float(np.float32(0.8)))
        self.assertEqual(data["risk_category"], "very_high")
        self.assertEqual(data["decision_threshold"], 0.35)
        args, kwargs = scorer.call_args
        self.assertEqual(list(args[0].columns), list(FRAUD_FIELDS))
        self.assertEqual(kwargs["artifact_path"], str(self.fraud_artifact))

    def test_missing_and_extra_fields_return_422(self):
        missing = dict(CREDIT_PAYLOAD)
        missing.pop("loan_amnt")
        response = self.post_json("/api/credit-risk/score", missing)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["error"]["code"], "VALIDATION_ERROR")
        extra = dict(CREDIT_PAYLOAD, loan_status=0)
        self.assertEqual(self.post_json("/api/credit-risk/score", extra).status_code, 422)
        fraud_extra = dict(FRAUD_PAYLOAD, extra_feature=4)
        self.assertEqual(self.post_json("/api/fraud-risk/score", fraud_extra).status_code, 422)

    def test_invalid_json_and_non_object_json_return_400(self):
        response = self.client.post("/api/credit-risk/score", data="{broken", content_type="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["code"], "INVALID_JSON")
        self.assertEqual(self.post_json("/api/credit-risk/score", [CREDIT_PAYLOAD]).status_code, 400)
        response = self.client.post("/api/credit-risk/score", data="{}", content_type="text/plain")
        self.assertEqual(response.status_code, 400)

    def test_invalid_types_non_finite_values_and_fraud_target_return_422(self):
        invalid_credit = dict(CREDIT_PAYLOAD, person_age="38")
        self.assertEqual(self.post_json("/api/credit-risk/score", invalid_credit).status_code, 422)
        invalid_fraud = dict(FRAUD_PAYLOAD, V3="not numeric")
        self.assertEqual(self.post_json("/api/fraud-risk/score", invalid_fraud).status_code, 422)
        non_finite = dict(FRAUD_PAYLOAD, V4=float("nan"))
        self.assertEqual(self.post_json("/api/fraud-risk/score", non_finite).status_code, 422)
        target = dict(FRAUD_PAYLOAD, Class=0)
        response = self.post_json("/api/fraud-risk/score", target)
        self.assertEqual(response.status_code, 422)
        self.assertIn("target", response.get_json()["error"]["message"])

    def test_unknown_route_and_method_have_json_errors(self):
        response = self.client.get("/api/no-such-route")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["error"]["code"], "NOT_FOUND")
        response = self.client.get("/api/credit-risk/score")
        self.assertEqual(response.status_code, 405)
        self.assertEqual(response.get_json()["error"]["code"], "METHOD_NOT_ALLOWED")

    def test_missing_artifact_maps_to_safe_503(self):
        with patch("src.api.routes.score_credit_risk", side_effect=FileNotFoundError("C:/secret/model.joblib")):
            response = self.post_json("/api/credit-risk/score", CREDIT_PAYLOAD)
        self.assertEqual(response.status_code, 503)
        body = response.get_json()
        self.assertEqual(body["error"]["code"], "MODEL_NOT_READY")
        self.assertNotIn("secret", response.get_data(as_text=True))
        self.assertNotIn("C:/", response.get_data(as_text=True))

    def test_unexpected_inference_error_is_generic_500(self):
        with patch("src.api.routes.score_transaction_fraud", side_effect=RuntimeError("private stack detail")):
            response = self.post_json("/api/fraud-risk/score", FRAUD_PAYLOAD)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"]["code"], "INTERNAL_ERROR")
        self.assertNotIn("private stack detail", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()

