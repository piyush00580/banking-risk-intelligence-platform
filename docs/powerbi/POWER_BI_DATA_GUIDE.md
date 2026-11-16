# Power BI Data Guide

Phase 9.1 prepares data sources for a later Power BI implementation. No `.pbix` file or finished dashboard is included in this milestone.

## Local prerequisites

- PostgreSQL connection details: host `localhost`, port `5432`, database `banking_risk_intelligence`, and a database role authorized to read the credit tables and create views. The repository's credit loader uses the `postgres` role. Supply the password through your local PostgreSQL/Power BI credential prompt; do not put passwords in this guide, SQL files, or source control.
- Local model artifacts:
  - `models/credit_risk_xgb.joblib`
  - `models/credit_risk_metadata.json`
  - `models/fraud_detection_xgb.joblib`
- Source datasets at `data/raw/credit_risk_dataset.csv` and `data/raw/creditcard.csv` for generating the exports.
- Python dependencies from `requirements.txt` in the project environment.

The joblib model files are ignored by Git. Export scripts fail clearly when a required local artifact or source dataset is missing.

## Create the PostgreSQL views

From the repository root, execute:

```powershell
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -X -v ON_ERROR_STOP=1 `
  -U postgres -d banking_risk_intelligence -f sql/powerbi_views.sql
```

Provide the database password through your local PostgreSQL prompt or approved credential setup. The script only uses `CREATE OR REPLACE VIEW`; it does not alter tables or reload data. It creates:

- `vw_credit_portfolio_summary`
- `vw_credit_risk_by_grade`
- `vw_credit_risk_by_intent`
- `vw_credit_risk_by_customer_segment`
- `vw_credit_loan_details`

Default rates are fractions from 0 to 1 and use only loans with a known `loan_status` in the denominator. Total-loan counts include rows whose status is null. The customer age segments are Under 25, 25-34, 35-44, 45-54, 55+, and Unknown. Annual income segments are below 25k, 25k to below 50k, 50k to below 100k, 100k+, and Unknown, in the source dataset's numeric units. `loan_to_income_ratio` is the original `loans.loan_percent_income` value; the view does not recalculate it.

The environment used for this milestone had the PostgreSQL 18 client but could not authenticate to the local database because no `PGPASSWORD` was configured. The views have not been applied or live-queried in that database. Run the command above locally and verify each view before connecting Power BI.

## Generate the model-scored exports

Run from the repository root without retraining either model:

```powershell
python src/export_powerbi_fraud_data.py
python src/export_powerbi_credit_data.py
```

Files are written under the Git-ignored `data/processed/powerbi/` directory.

| File | Grain and contents |
|---|---|
| `fraud_test_scored.csv` | One row per reproducible held-out fraud benchmark test transaction: generated `dataset_row_id`, the 30 raw features, actual `Class`, fraud score, predicted class, risk category, threshold, and evaluation partition. |
| `fraud_kpi_summary.csv` | One-row summary of held-out fraud test size, prevalence, TP/FP/TN/FN, precision, recall, F1, average precision (PR-AUC), ROC-AUC, and threshold. |
| `fraud_dataset_metadata.json` | Dataset provenance, cleaning and split counts, model/version, threshold, exported columns, KPI values, row identifier definition, and limitations. |
| `credit_scored.csv` | One row per cleaned credit benchmark record: generated source-row key, the model's 11 raw features, observed `loan_status`, predicted default probability/class, risk tier, and `model_partition`. |
| `credit_dataset_metadata.json` | Credit cleaning and partition counts, model/version, threshold, export meaning, and the in-sample/held-out limitation. |

The fraud ID is derived from the cleaned benchmark CSV row index and is not a real transaction/customer identifier. The credit ID is derived from the original credit CSV row position and is not a PostgreSQL `customer_id` or `loan_id`.

The fraud export exactly reuses `clean_raw_fraud_data()` and `split_fraud_data_stratified()` with the established 64/16/20 split and `random_state=42`. The model artifact's fitted feature engineer and XGBoost are reused, with threshold 0.35. The test split is not used for fitting or threshold selection. In this export, the fraud test partition contains 56,746 rows (95 fraud, 56,651 legitimate); confusion counts are 71 TP, 2 FP, 56,649 TN, and 24 FN. AP/PR-AUC is 0.8297, ROC-AUC 0.9725, precision 0.9726, recall 0.7474, and F1 0.8452.

The credit export reproduces the credit notebook cleaning and stratified 80/20 partition (`random_state=42`) and uses the serialized pipeline at threshold 0.45. Predictions on the 25,928 training rows are in-sample; the 6,483 rows marked `held_out_test` remain the original held-out partition. The export does not retune the threshold or replace the held-out evaluation. `loan_status` is the observed outcome; probability and tier columns are model estimates.

## Connect Power BI Desktop

1. In Power BI Desktop, choose **Get data** and select **PostgreSQL database**.
2. Enter server `localhost:5432` and database `banking_risk_intelligence`.
3. Select the appropriate database authentication method and enter credentials in Power BI's credential dialog. Do not embed a password in a PBIX query or this repository.
4. In Navigator, select the five `vw_credit_*` views. Prefer the loan detail view for drill-through and the aggregate views for their corresponding summaries.
5. To import scored data, choose **Get data > Text/CSV** and browse to `data/processed/powerbi/fraud_test_scored.csv` or `credit_scored.csv`. Load `fraud_kpi_summary.csv` as a separate single-row fraud benchmark KPI source.

If connecting to PostgreSQL requires a local Power BI connector/driver, install it through your organization's approved software process. This repository does not configure client credentials or drivers.

## Suggested model and report pages

- **Credit portfolio:** use the portfolio summary for total loans, defaults, default rate, and amount cards; add grade and intent breakdown visuals from their dedicated views.
- **Credit segments and detail:** use customer-segment default rate by age/income/home ownership, then drill through to `vw_credit_loan_details`. Use `customer_id` and `loan_id` only within the PostgreSQL credit source.
- **Fraud benchmark evaluation:** use the one-row KPI summary for test metrics, and `fraud_test_scored.csv` for actual-versus-predicted counts, score distribution, threshold, and risk-category breakdown. Label this as held-out benchmark evaluation, not live monitoring.
- **Risk intelligence overview:** place credit portfolio KPIs and fraud benchmark KPIs in separate, clearly labeled sections. Do not create a relationship between these datasets or calculate a combined risk score.

No relationship should connect the credit benchmark rows, PostgreSQL loan/customer records, and fraud benchmark transactions through the generated row IDs. The row identifiers are for deterministic export identity only, and the datasets do not provide a common customer key.

## Data limitations and interpretation

- The credit dataset and the European card fraud benchmark are independent datasets; they do not represent the same customers or portfolio.
- The existing PostgreSQL `transactions` and `fraud_alerts` tables are defined by the schema, but `src/load_credit_data.py` only loads `customers` and `loans`. The credit loader does not populate transactions or fraud alerts.
- Fraud benchmark `Time` is elapsed seconds in the source recording, not a real calendar timestamp or time of day.
- Fraud features `V1`-`V28` are anonymized PCA components without documented direct business meanings.
- Fraud probabilities and risk categories are model scores, not calibrated financial-loss probabilities.
- Credit default and transaction fraud probabilities estimate different outcomes and must not be combined into one score.
- Model performance metrics are benchmark results on a held-out test partition, not live production monitoring or evidence of current operating performance.
- Credit training-partition predictions are in-sample. Filter `model_partition = held_out_test` when displaying original credit test-partition predictions.
- Credit row-level export contains sensitive financial attributes from the benchmark dataset. Apply appropriate access controls when sharing the generated files.
