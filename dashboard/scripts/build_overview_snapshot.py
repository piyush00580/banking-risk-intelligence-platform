"""Build a compact, privacy-safe dashboard snapshot from Phase 9.1 exports."""
from __future__ import annotations

import json
from pathlib import Path

import csv
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ROOT / "data" / "processed" / "powerbi"
OUTPUT = ROOT / "dashboard" / "public" / "data" / "overview.json"


def main() -> None:
    with (EXPORTS / "credit_scored.csv").open(newline="", encoding="utf-8") as handle:
        credit = list(csv.DictReader(handle))
    with (EXPORTS / "fraud_kpi_summary.csv").open(newline="", encoding="utf-8") as handle:
        fraud = next(csv.DictReader(handle))
    heldout = [row for row in credit if row["model_partition"] == "held_out_test"]
    grouped: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0.0])
    for row in heldout:
        group = grouped[row["loan_grade"]]
        group[0] += 1
        group[1] += int(row["loan_status"])
        group[2] += float(row["predicted_default_probability"])
    snapshot = {
        "source": "Phase 9.1 local benchmark exports",
        "credit": {
            "scope": "held_out_test",
            "applications": len(heldout),
            "observed_defaults": sum(int(row["loan_status"]) for row in heldout),
            "observed_default_rate": sum(int(row["loan_status"]) for row in heldout) / len(heldout),
            "mean_predicted_default_probability": sum(float(row["predicted_default_probability"]) for row in heldout) / len(heldout),
            "by_grade": [
                {"grade": grade, "applications": int(values[0]),
                 "observed_defaults": int(values[1]),
                 "mean_predicted_default_probability": values[2] / values[0]}
                for grade, values in sorted(grouped.items())
            ],
            "in_sample_rows_excluded": sum(row["model_partition"] == "train_in_sample" for row in credit),
        },
        "fraud": {
            "scope": "held_out_test", "transactions": int(fraud["test_total_transactions"]),
            "fraud": int(fraud["test_actual_fraud_transactions"]),
            "legitimate": int(fraud["test_actual_legitimate_transactions"]),
            "true_positives": int(fraud["test_true_positives"]), "false_positives": int(fraud["test_false_positives"]),
            "true_negatives": int(fraud["test_true_negatives"]), "false_negatives": int(fraud["test_false_negatives"]),
            "precision": float(fraud["test_precision"]), "recall": float(fraud["test_recall"]),
            "f1": float(fraud["test_f1_score"]), "pr_auc": float(fraud["test_pr_auc_average_precision"]),
            "roc_auc": float(fraud["test_roc_auc"]), "threshold": float(fraud["decision_threshold"]),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
