"""Unit tests for independent integrated credit and fraud scoring paths."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import risk_intelligence_engine as engine


CREDIT_FEATURES = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
]
CREDIT_SAMPLE = {
    "person_age": 38, "person_income": 110000.0, "person_home_ownership": "OWN",
    "person_emp_length": 12.0, "loan_intent": "HOMEIMPROVEMENT", "loan_grade": "A",
    "loan_amnt": 5000.0, "loan_int_rate": 7.25, "loan_percent_income": 0.05,
    "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 10,
}


class FakeCreditPipeline:
    feature_names_in_ = np.array(CREDIT_FEATURES, dtype=object)

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)
        self.seen = None

    def predict_proba(self, frame):
        self.seen = frame.copy()
        selected = self.scores[:len(frame)]
        return np.column_stack([1.0 - selected, selected])


def credit_metadata():
    return {
        "model_name": "Credit Risk XGBoost Classifier",
        "model_version": "1.0.0",
        "feature_schema": {
            "numerical_features": [
                "person_age", "person_income", "person_emp_length", "loan_amnt",
                "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
            ],
            "categorical_features": [
                "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file",
            ],
        },
        "decision_thresholds": {
            "default_threshold": 0.5,
            "selected_optimal_threshold": 0.45,
            "selection_criterion": "OOF F1 selection on the training partition",
        },
    }


class RiskIntelligenceEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifact = self.root / "credit.joblib"
        self.artifact.write_bytes(b"test stub")
        self.metadata_path = self.root / "credit.json"
        self.metadata_path.write_text(json.dumps(credit_metadata()), encoding="utf-8")
        self.credit_model = FakeCreditPipeline([0.10, 0.30, 0.45, 0.80])
        self.credit_loader = patch.object(engine.joblib, "load", return_value=self.credit_model)
        self.credit_loader.start()
        self.addCleanup(self.credit_loader.stop)

    def score_credit(self, data):
        return engine.score_credit_risk(data, self.artifact, self.metadata_path)

    def test_credit_scoring_schema_tiers_and_threshold(self):
        applicants = pd.DataFrame([CREDIT_SAMPLE] * 4, index=[4, 5, 6, 7])
        result = self.score_credit(applicants)
        self.assertEqual(list(result.columns), engine.RESPONSE_COLUMNS)
        self.assertEqual(result.index.tolist(), [4, 5, 6, 7])
        self.assertEqual(result.risk_type.unique().tolist(), ["credit_default"])
        self.assertEqual(result.risk_score.tolist(), [0.1, 0.3, 0.45, 0.8])
        self.assertEqual(result.predicted_class.tolist(), [0, 0, 1, 1])
        self.assertEqual(result.risk_category.tolist(), [
            "Tier 1 (Prime)", "Tier 2 (Moderate)", "Tier 3 (High Default Risk)", "Tier 3 (High Default Risk)"
        ])
        self.assertTrue(result.risk_score.between(0, 1).all())
        self.assertEqual(result.decision_threshold.unique().tolist(), [0.45])
        self.assertIn("loan default", result.score_interpretation.iloc[0])
        self.assertEqual(list(self.credit_model.seen.columns), CREDIT_FEATURES)

    def test_credit_accepts_one_mapping_and_training_pipeline_imputation(self):
        sample = dict(CREDIT_SAMPLE)
        sample["person_emp_length"] = np.nan
        result = self.score_credit(sample)
        self.assertEqual(len(result), 1)
        self.assertTrue(pd.isna(self.credit_model.seen.iloc[0]["person_emp_length"]))

    def test_credit_rejects_missing_extra_and_invalid_values(self):
        missing = dict(CREDIT_SAMPLE)
        missing.pop("loan_amnt")
        with self.assertRaisesRegex(engine.RiskInputError, "missing columns"):
            self.score_credit(missing)
        extra = dict(CREDIT_SAMPLE, loan_status=0)
        with self.assertRaisesRegex(engine.RiskInputError, "unexpected columns"):
            self.score_credit(extra)
        bad_numeric = dict(CREDIT_SAMPLE, person_income=np.inf)
        with self.assertRaisesRegex(engine.RiskInputError, "infinite"):
            self.score_credit(bad_numeric)
        bad_category = dict(CREDIT_SAMPLE, loan_grade=3)
        with self.assertRaisesRegex(engine.RiskInputError, "text values"):
            self.score_credit(bad_category)

    def test_credit_artifact_and_metadata_errors(self):
        with self.assertRaises(FileNotFoundError):
            engine.load_credit_model(self.root / "missing.joblib", self.metadata_path)
        with self.assertRaises(FileNotFoundError):
            engine.load_credit_model(self.artifact, self.root / "missing.json")

    def test_fraud_path_preserves_threshold_bands_and_schema_without_credit_load(self):
        # Use the existing 30-column fraud schema while keeping model work mocked.
        raw_names = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
        transactions = pd.DataFrame([{name: 0.0 for name in raw_names}], index=[12])
        transactions.loc[12, "Time"] = 1000.0
        transactions.loc[12, "Amount"] = 12.5
        bundle = {
            "threshold": 0.35,
            "metadata": {"model_name": "Fraud Detection XGBoost Champion", "model_version": "1.0.0"},
        }
        predicted = pd.DataFrame({
            "fraud_probability": [0.80], "predicted_class": [1], "risk_category": ["very_high"]
        }, index=transactions.index)
        with patch.object(engine, "load_fraud_model", return_value=bundle) as loader, \
             patch.object(engine, "predict_fraud", return_value=predicted) as predictor, \
             patch.object(engine, "load_credit_model", side_effect=AssertionError("credit model loaded")):
            result = engine.score_transaction_fraud(transactions)
        loader.assert_called_once_with(None)
        predictor.assert_called_once_with(transactions, artifact=bundle)
        self.assertEqual(list(result.columns), engine.RESPONSE_COLUMNS)
        self.assertEqual(result.index.tolist(), [12])
        self.assertEqual(result.risk_type.iloc[0], "transaction_fraud")
        self.assertEqual(result.risk_score.iloc[0], 0.80)
        self.assertEqual(result.predicted_class.iloc[0], 1)
        self.assertEqual(result.risk_category.iloc[0], "very_high")
        self.assertEqual(result.decision_threshold.iloc[0], 0.35)
        self.assertIn("transaction is fraudulent", result.score_interpretation.iloc[0])
        self.assertEqual(result.metadata.iloc[0]["risk_bands"]["moderate"], "0.10 <= probability < 0.35")

    def test_fraud_artifact_not_found_and_lazy_model_loading(self):
        missing_fraud_artifact = self.root / "missing-fraud.joblib"
        with patch.object(engine, "load_credit_model", side_effect=AssertionError("credit model loaded")):
            with self.assertRaisesRegex(FileNotFoundError, "Fraud model artifact not found"):
                engine.score_transaction_fraud(pd.DataFrame(), missing_fraud_artifact)

    def test_fraud_load_errors_propagate_without_loading_credit_model(self):
        with patch.object(engine, "load_fraud_model", side_effect=FileNotFoundError("fraud bundle missing")), \
             patch.object(engine, "load_credit_model", side_effect=AssertionError("credit model loaded")):
            with self.assertRaisesRegex(FileNotFoundError, "fraud bundle missing"):
                engine.score_transaction_fraud(pd.DataFrame())

    def test_pathways_keep_model_specific_scores_separate(self):
        credit = self.score_credit(CREDIT_SAMPLE)
        transactions = pd.DataFrame([{name: 0.0 for name in ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]}])
        fraud_bundle = {"threshold": 0.35, "metadata": {"model_name": "Fraud", "model_version": "1"}}
        fraud_prediction = pd.DataFrame({
            "fraud_probability": [0.70], "predicted_class": [1], "risk_category": ["high"]
        })
        with patch.object(engine, "load_fraud_model", return_value=fraud_bundle), \
             patch.object(engine, "predict_fraud", return_value=fraud_prediction):
            fraud = engine.score_transaction_fraud(transactions)
        self.assertEqual(credit.risk_type.iloc[0], "credit_default")
        self.assertEqual(fraud.risk_type.iloc[0], "transaction_fraud")
        self.assertEqual(credit.risk_score.iloc[0], 0.10)
        self.assertEqual(fraud.risk_score.iloc[0], 0.70)
        self.assertNotIn("combined_risk_score", credit.columns)
        self.assertNotIn("combined_risk_score", fraud.columns)


if __name__ == "__main__":
    unittest.main()
