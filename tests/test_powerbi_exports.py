"""Focused unit checks for Power BI export calculations and SQL safety."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from export_powerbi_credit_data import prepare_credit_dataset
from export_powerbi_fraud_data import compute_fraud_test_metrics


class PowerBIExportTests(unittest.TestCase):
    def test_fraud_kpi_metrics_and_confusion_sum(self):
        actual = pd.Series([0, 0, 1, 1])
        probabilities = pd.Series([0.10, 0.40, 0.35, 0.90])
        metrics = compute_fraud_test_metrics(actual, probabilities, threshold=0.35)
        self.assertEqual(metrics["test_total_transactions"], 4)
        self.assertEqual(metrics["test_actual_fraud_transactions"], 2)
        self.assertEqual(metrics["test_actual_legitimate_transactions"], 2)
        self.assertEqual(
            metrics["test_true_positives"] + metrics["test_false_positives"]
            + metrics["test_true_negatives"] + metrics["test_false_negatives"],
            4,
        )
        self.assertEqual((metrics["test_true_positives"], metrics["test_false_positives"]), (2, 1))
        self.assertEqual((metrics["test_true_negatives"], metrics["test_false_negatives"]), (1, 0))
        self.assertAlmostEqual(metrics["test_precision"], 2 / 3)
        self.assertAlmostEqual(metrics["test_recall"], 1.0)
        self.assertAlmostEqual(metrics["test_f1_score"], 0.8)
        self.assertAlmostEqual(metrics["test_pr_auc_average_precision"], 5 / 6)
        self.assertAlmostEqual(metrics["test_roc_auc"], 0.75)
        self.assertEqual(metrics["decision_threshold"], 0.35)

    def test_credit_cleaning_matches_notebook_rules(self):
        raw = pd.DataFrame([
            {"person_age": 30, "person_emp_length": 5.0, "loan_status": 0},
            {"person_age": 40, "person_emp_length": 45.0, "loan_status": 1},
            {"person_age": 101, "person_emp_length": 3.0, "loan_status": 0},
            {"person_age": 30, "person_emp_length": 5.0, "loan_status": 0},
        ])
        clean, stats = prepare_credit_dataset(raw)
        self.assertEqual(stats["duplicate_rows_removed"], 1)
        self.assertEqual(stats["age_over_100_rows_removed"], 1)
        self.assertEqual(stats["employment_length_rows_masked"], 1)
        self.assertEqual(len(clean), 2)
        self.assertTrue(pd.isna(clean.loc[1, "person_emp_length"]))
        self.assertEqual(clean["loan_status"].tolist(), [0, 1])

    def test_sql_definitions_are_idempotent_and_non_destructive(self):
        sql = (ROOT / "sql" / "powerbi_views.sql").read_text(encoding="utf-8-sig").upper()
        expected_views = [
            "VW_CREDIT_PORTFOLIO_SUMMARY", "VW_CREDIT_RISK_BY_GRADE",
            "VW_CREDIT_RISK_BY_INTENT", "VW_CREDIT_RISK_BY_CUSTOMER_SEGMENT",
            "VW_CREDIT_LOAN_DETAILS",
        ]
        for name in expected_views:
            self.assertIn(f"CREATE OR REPLACE VIEW {name}", sql)
        self.assertIn("NULLIF(", sql)
        for destructive in ("DROP TABLE", "TRUNCATE", "DELETE FROM", "INSERT INTO"):
            self.assertNotIn(destructive, sql)
        self.assertNotIn("FRAUD_ALERTS", sql)
        self.assertNotIn("TRANSACTIONS", sql)


if __name__ == "__main__":
    unittest.main()
