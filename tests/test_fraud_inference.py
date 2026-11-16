"""Offline checks for the packaged fraud inference interface."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_inference import FraudInputError, load_fraud_model, predict_fraud
from fraud_feature_engineering import clean_raw_fraud_data, split_fraud_data_stratified


class FraudInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_fraud_model()
        raw_path = ROOT / "data" / "raw" / "creditcard.csv"
        if not raw_path.is_file():
            raise unittest.SkipTest("Local creditcard.csv is required; tests do not download data.")
        raw = clean_raw_fraud_data(pd.read_csv(raw_path))
        _x_train, _x_val, x_test, _y_train, _y_val, _y_test = split_fraud_data_stratified(
            raw, target_col="Class", train_ratio=0.64, val_ratio=0.16, test_ratio=0.20, random_state=42
        )
        cls.one = x_test.iloc[[0]].copy()
        cls.many = x_test.iloc[:5].copy()

    def test_artifact_loads_and_schema_is_preserved(self):
        self.assertEqual(self.bundle["threshold"], 0.35)
        self.assertEqual(self.bundle["feature_names"], self.bundle["feature_engineer"].get_feature_names_out())
        reloaded = load_fraud_model()
        np.testing.assert_array_equal(self.bundle["feature_names"], reloaded["feature_names"])

    def test_single_row_inference_and_output_contract(self):
        result = predict_fraud(self.one, self.bundle)
        self.assertEqual(result.shape, (1, 3))
        self.assertEqual(list(result.columns), ["fraud_probability", "predicted_class", "risk_category"])
        self.assertTrue(np.issubdtype(result.fraud_probability.dtype, np.floating))
        self.assertTrue(np.issubdtype(result.predicted_class.dtype, np.integer))
        self.assertTrue(result.risk_category.isin(["low", "moderate", "high", "very_high"]).all())

    def test_multi_row_inference_probabilities_and_threshold(self):
        result = predict_fraud(self.many, self.bundle)
        self.assertEqual(len(result), len(self.many))
        self.assertTrue(result.fraud_probability.between(0.0, 1.0).all())
        expected = (result.fraud_probability >= 0.35).astype(int)
        np.testing.assert_array_equal(result.predicted_class.to_numpy(), expected.to_numpy())

    def test_missing_column_rejected(self):
        with self.assertRaisesRegex(FraudInputError, "missing columns"):
            predict_fraud(self.one.drop(columns=["V1"]), self.bundle)

    def test_unexpected_column_rejected(self):
        malformed = self.one.assign(unexpected=1.0)
        with self.assertRaisesRegex(FraudInputError, "unexpected columns"):
            predict_fraud(malformed, self.bundle)

    def test_null_infinite_and_non_numeric_values_rejected(self):
        for invalid in (np.nan, np.inf, "not-a-number"):
            malformed = self.one.copy()
            if isinstance(invalid, str):
                malformed = malformed.astype(object)
            malformed.iloc[0, 0] = invalid
            with self.subTest(invalid=invalid):
                with self.assertRaises(FraudInputError):
                    predict_fraud(malformed, self.bundle)

    def test_invalid_amount_and_time_rejected(self):
        for column in ("Amount", "Time"):
            malformed = self.one.copy()
            malformed.loc[malformed.index[0], column] = -1.0
            with self.subTest(column=column):
                with self.assertRaisesRegex(FraudInputError, "non-negative"):
                    predict_fraud(malformed, self.bundle)

    def test_reload_prediction_consistency(self):
        first = predict_fraud(self.many, self.bundle)
        reloaded = load_fraud_model()
        second = predict_fraud(self.many, reloaded)
        np.testing.assert_allclose(first.fraud_probability, second.fraud_probability, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(first.predicted_class, second.predicted_class)
        np.testing.assert_array_equal(first.risk_category, second.risk_category)


if __name__ == "__main__":
    unittest.main()
