# Banking Risk Intelligence Platform

An integrated financial risk analytics platform combining credit risk prediction and transaction fraud detection.

## Objectives

- Predict loan default risk
- Detect potentially fraudulent transactions
- Generate risk scores and categories
- Perform SQL-based financial analytics
- Provide REST APIs for model predictions
- Build an interactive Power BI risk dashboard

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- PostgreSQL
- Flask
- Power BI

## Project Modules

1. Credit Risk Prediction
2. Fraud Detection
3. Risk Intelligence Engine
4. SQL Analytics
5. REST API
6. Power BI Dashboard

## Status

Phase 9.2 - React frontend foundation implemented; detailed analytics and live database integration pending

## Implementation Progress

- **Phase 1: Project Foundation** — Repository structure, dependencies, virtual environment, and git tracking initialized.
- **Phase 2: Dataset Inspection & Validation** — Comprehensive data quality audit on 32,581 records (`notebooks/credit_risk_data_inspection.ipynb`), identifying missing values, duplicates, and domain constraints.
- **Phase 3: Credit Risk Exploratory Data Analysis** — Statistical analysis and visualizations (`notebooks/credit_risk_eda.ipynb`) benchmarking default patterns across loan grades, loan intent, and borrower characteristics.
- **Phase 4: Database Design & SQL Analytics**:
  - Relational schema defined in `sql/schema.sql` (`customers`, `loans`, `transactions`, `fraud_alerts`).
  - Credit data cleaning and ingestion via `src/load_credit_data.py`, populating 32,576 records each in `customers` and `loans`.
  - Five core credit risk analytical queries implemented in `sql/credit_queries.sql` and verified against PostgreSQL (`banking_risk_intelligence`):
    - Overall default rate: 21.82% (7,108 defaults / 32,576 loans).
    - Default rate by loan grade (Grade A at 9.96% to Grade G at 98.44%).
    - Default rate by loan purpose (Debt Consolidation highest at 28.59%).
    - Loan amounts and interest rates by outcome.
    - Customer-level loan risk summary join.
- **Phase 5: Credit Risk Modeling**:
  - **Milestone 5.1 (Multi-Architecture Benchmark)**:
    - Data hygiene & integrity: Audited and removed 165 duplicates; excluded 5 impossible ages (>100); masked 2 employment length anomalies (>age) for imputation.
    - Leakage-safe pipelines: `ColumnTransformer` with median imputation and standard scaling for numerical attributes; most-frequent imputation and one-hot encoding for categorical features (fitted strictly on training set).
    - Stratified 80/20 train/test split ($n=25,928$ train, $n=6,483$ test) preserving 21.87% portfolio default prevalence.
    - Multi-architecture benchmark: Logistic Regression (Standard & Balanced), Random Forest, and XGBoost.
    - Champion model: **XGBoost Classifier** delivering **ROC-AUC: 0.9409**, **PR-AUC: 0.8955**, **F1-Score: 0.8278**, **Precision: 96.97%**, and **Recall: 72.21%** on held-out test data.
  - **Milestone 5.2 (Optimization, Explainability & Serialization)**:
    - **Threshold Tuning**: Executed a 17-point threshold sweep ($T \in [0.10, 0.90]$) using strictly 5-fold Stratified Cross-Validation out-of-fold probabilities on the training partition without test set contamination.
    - **Business Operating Threshold**: Selected **$T = 0.45$** based on validation F1-score maximization (F1 = 0.8277), reducing missed test defaults (False Negatives) from 394 to 387 while preserving 95.55% precision.
    - **SHAP Explainability**: Implemented `shap.TreeExplainer` providing global feature impact beeswarm distributions and individual applicant adverse action waterfall explanations (ECOA/FCRA compliance).
    - **Temporal Leakage Audit**: Evaluated `loan_grade` and `loan_int_rate` origination timing, distinguishing pre-underwriting screening from post-pricing underwriting verification gates.
    - **Production Serialization**: Packaged end-to-end champion pipeline to `models/credit_risk_xgb.joblib` and runtime configuration to `models/credit_risk_metadata.json`, with verified inference reproduction.
- **Phase 6: Fraud Detection**:
  - **Milestone 6.1 (Dataset Acquisition & Integrity Check)**: Acquired the ULB/Worldline benchmark European cardholder fraud dataset (`data/raw/creditcard.csv`) with 284,807 transactions across 31 features, documenting ODbL v1.0 provenance and verification script `src/download_fraud_data.py`.
  - **Milestone 6.2 (Exploratory Data Analysis)**:
    - Completed exhaustive statistical analysis in `notebooks/fraud_detection_eda.ipynb` (33 cells, fully executed).
    - Data hygiene: Audited and removed 1,081 exact duplicates (1,062 legitimate, 19 fraudulent); verified 0 missing cells.
    - Cleaned distribution: 283,253 legitimate vs. 473 fraudulent transactions (0.1667% prevalence, ~599:1 imbalance).
    - Amount dynamics: Legitimate median €22.00 vs. fraud median €9.82; fraud mean (€123.87) exceeds legitimate mean (€88.41) due to right-skewed heavy fraud tail up to €2,125.87.
    - Zero-amount card testing: €0.00 authorizations account for 5.29% of fraud vs. 0.63% of legitimate transactions (8.4x relative elevation).
    - Temporal risk partitioning: Discovered elevated nocturnal fraud spikes in Window 0 (0.4473%) and Window 4 (0.5196%) across 8 sequential ~6-hour intervals.
    - Hypothesis testing: Mann-Whitney U tests with Benjamini-Hochberg FDR correction ($\alpha=0.05$) identified $V_{14}, V_4, V_{12}, V_{11}, V_{10}, V_3, V_2, V_{16}, V_9, V_7$ as most discriminative signals ($p < 10^{-135}$).
  - **Milestone 6.3 (Feature Engineering & Partitioning Pipeline)**:
    - Implemented modular, scikit-learn compatible transformer `FraudFeatureEngineer` in `src/fraud_feature_engineering.py` and analytical verification notebook `notebooks/fraud_feature_engineering.ipynb` (22 cells, fully executed).
    - Engineered 5 non-redundant domain features: `Amount_log` ($\log(1+\text{Amount})$ for skew dampening), `is_zero_amount` (binary indicator for micro-authorizations), `Time_scaled` (leakage-safe robust scaling of elapsed seconds without unanchored diurnal assumptions), and cross-feature interactions `V14_V4_interaction` ($r = -0.4285$ with Class) and `V12_V11_interaction` ($r = -0.2798$ with Class).
    - Rigorously compared Stratified Random (64/16/20) and Chronological (64/16/20) data splitting strategies.
    - Recommended 3-way Stratified Partitioning for Phase 6.4 modeling (Train: 181,584 txs / 303 frauds; Val: 45,396 txs / 76 frauds; Test: 56,746 txs / 94 frauds) to ensure minority class statistical power during threshold tuning over an unanchored 48-hour recording window.
    - Validated zero data leakage: strict disjoint indices, scaler parameters fit solely on training data, target isolation, and zero null or infinite values.
  - **Milestone 6.4 (Baseline Fraud Detection Modelling)**:
    - Developed reusable baseline modeling module `src/fraud_baseline_models.py` and comprehensive benchmark notebook `notebooks/fraud_baseline_modeling.ipynb` (20 cells, fully executed).
    - Benchmarked three distinct baseline architectures across identical leakage-safe stratified partitions (181,584 train, 45,396 validation, 56,746 test):
      - **Standard Logistic Regression**: PR-AUC 0.8037, ROC-AUC 0.9818, Tuned $T=0.20 \to$ F1: 0.7598 (Precision: 80.95%, Recall: 71.58%, 27 FN, 16 FP).
      - **Balanced Logistic Regression**: PR-AUC 0.7686, ROC-AUC 0.9788, Tuned $T=0.90 \to$ F1: 0.3885 (Precision: 24.58%, Recall: 92.63%, 7 FN, 271 FP).
      - **Random Forest Baseline (Champion)**: **PR-AUC 0.8529**, ROC-AUC 0.9790, Tuned $T=0.60 \to$ **F1: 0.8046** (Precision: 88.61%, Recall: 73.68%, 25 FN, **9 FP**).
    - Validated that decision threshold optimization strictly on the validation partition substantially improves operational trade-offs without test-set contamination.
  - **Milestone 6.5 (Advanced Fraud Detection Model Benchmarking)**:
      

    **Status: Completed**

    Implemented and benchmarked advanced gradient-boosting architectures using XGBoost and LightGBM, alongside the existing Logistic Regression and Random Forest baselines.

    **Models evaluated:**
    - Standard Logistic Regression
    - Balanced Logistic Regression
    - Random Forest
    - XGBoost
    - LightGBM

    **Final held-out test results:**

    | Metric | Random Forest | XGBoost |
    |---|---:|---:|
    | PR-AUC | 0.7943 | **0.8297** |
    | ROC-AUC | 0.9492 | **0.9725** |
    | Precision | 94.37% | **97.26%** |
    | Recall | 70.53% | **74.74%** |
    | F1-score | 0.8072 | **0.8452** |
    | Decision threshold | 0.60 | 0.35 |

    **Key findings:**
    - XGBoost achieved the strongest reported test performance among the highlighted tree-based models.
    - Validation-based threshold tuning was performed without using the test set for threshold selection.
    - XGBoost achieved 71 true positives, 24 missed fraud cases and 2 false positives on the held-out test set.
    - Feature importance analysis highlighted V14, V10 and V12_V11_interaction among the leading XGBoost features.
    - XGBoost is selected as the provisional champion for the next integration phase.

    **Technical deliverables:**
    - `src/fraud_advanced_models.py`
    - `notebooks/fraud_advanced_modeling.ipynb`
    - Updated `requirements.txt`
    - Updated project documentation

  - **Milestone 6.6 (Fraud Model Explainability and Risk Interpretation)**:
    - Added and executed `notebooks/fraud_model_explainability.ipynb` using `shap.TreeExplainer` for the Phase 6.5 XGBoost provisional champion. Data preparation reproduced the same raw dataset, 1,081 exact duplicate removals, stratified 64/16/20 split (181,584 train / 45,396 validation / 56,746 test), and training-only robust time scaling and feature engineering. The train partition contains 302 frauds, validation 76, and test 95.
    - Refit the unchanged champion configuration (`n_estimators=150`, `max_depth=7`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=1.0`, `random_state=42`). Validation-selected threshold was 0.35 (validation F1 0.9000); held-out test confusion counts reproduced TP=71, FP=2, FN=24, TN=56,649.
    - Used a fixed-seed sample of up to 1,000 held-out test transactions for global beeswarm and mean-absolute-SHAP bar plots. Top SHAP features were V4, V14, V10, V12, and `V12_V11_interaction`. Compared SHAP ranking with the XGBoost `feature_importances_` ranking: V14 and V10 ranked first and second by estimator importance, while V4 ranked fifth; both methods placed V12 fourth and `V12_V11_interaction` among the leading features.
    - Generated local waterfall explanations from actual held-out test cases: TP index 41,828 (fraud score 0.995800), FP index 119,616 (0.972888), FN index 68,371 (0.000015), and TN index 109,664 (0.000003), at threshold 0.35. Positive SHAP values push a case toward fraud; negative values push toward legitimate. Anonymized PCA features V1-V28 are not assigned unsupported business meanings; engineered feature interpretations are limited to their formulas.
    - Limitations: SHAP global findings describe the sampled test transactions and this fitted model, not causal effects or calibrated loss estimates. Correlated features and engineered interactions can share attribution; a single local example per confusion category is illustrative. SHAP values are model-margin (log-odds) contributions, not probability-point changes.

  - **Milestone 6.7 (Fraud Model Packaging, Reproducible Inference & Readiness)**:
    - Packaged the Phase 6.5 XGBoost champion and fitted `FraudFeatureEngineer(time_scaler="robust")` together in `models/fraud_detection_xgb.joblib` (ignored by Git under the existing `models/*.joblib` rule). Reproducible training entry point: `python src/package_fraud_model.py`. Metadata is saved separately to `models/fraud_detection_metadata.json`.
    - Training repeats exact-deduplication and the same stratified 64/16/20 split with `random_state=42`. The engineer and model fit only on the training partition; validation reproduces the established threshold 0.35 (F1 0.9000). Test data is not used to fit or tune. Phase 6.5 saved notebook outputs were checked and match its documented XGBoost test metrics (PR-AUC 0.8297, ROC-AUC 0.9725, F1 0.8452; TP=71, FP=2, FN=24, TN=56,649).
    - Reusable API in `src/fraud_inference.py`: `load_fraud_model()` and `predict_fraud(raw_transactions)`. Input is a pandas DataFrame with the 30 numeric raw fields (`Time`, `Amount`, `V1` through `V28`), without `Class`; missing/extra columns, duplicate names, null/infinite/non-numeric values, negative amount/time are rejected. The result contains `fraud_probability`, `predicted_class` (threshold 0.35), and `risk_category`.

      ```python
      import pandas as pd
      from src.fraud_inference import predict_fraud

      raw_batch = pd.read_csv("transactions_to_score.csv")  # 30 raw feature columns; no Class
      predictions = predict_fraud(raw_batch)
      print(predictions.head())
      ```

    - Risk bands are score groupings: low `<0.10`, moderate `[0.10, 0.35)`, high `[0.35, 0.75)`, very high `>=0.75`. These are not calibrated financial-loss probabilities. Run `python -m unittest discover -s tests -p "test_fraud_inference.py" -v` after packaging to verify artifact loading, single/batch predictions, schema/value validation, threshold consistency and reload consistency.
    - Reproducibility check on 32 validation rows found maximum absolute prediction difference 0 after joblib save/reload (tolerance `1e-12`); all 30 raw and 35 engineered feature names/order are saved and verified. Test partition is 56,746 transactions, including 95 frauds; it is held out.
    - Readiness limitations: scores have not been calibrated as financial-loss probabilities; the anonymized PCA features lack direct business interpretation; the dataset is highly imbalanced and no temporal drift, production latency, or live integration assessment is included. This milestone packages inference only; later platform/API phases remain separate.

- **Phase 7: Integrated Risk Intelligence Engine**:
  - Added `src/risk_intelligence_engine.py` with two independent, lazy-loaded scoring pathways: `score_credit_risk(applicant_data)` and `score_transaction_fraud(transaction_data)`. The engine wraps the saved credit pipeline and delegates transaction preprocessing and scoring to `src/fraud_inference.py`.
  - Credit inputs use the model's 11 raw fields: numeric `person_age`, `person_income`, `person_emp_length`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`; and categorical `person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file`. The serialized preprocessing pipeline handles missing values with its fitted imputers. Fraud inputs retain the 30 numeric fields (`Time`, `V1`-`V28`, `Amount`), excluding `Class`.
  - Both pathways return the same columns: `risk_type`, `model_name`, `model_version`, `risk_score`, `predicted_class`, `risk_category`, `decision_threshold`, `score_interpretation`, and `metadata`. Credit score means estimated loan-default probability; fraud score means estimated transaction-fraud probability. The pathways do not create or imply an overall combined probability or tier.
  - Example usage from the repository root:

    ```python
    import pandas as pd
    from src.risk_intelligence_engine import score_credit_risk, score_transaction_fraud

    applicant = pd.DataFrame([{
        "person_age": 38, "person_income": 110000.0, "person_home_ownership": "OWN",
        "person_emp_length": 12.0, "loan_intent": "HOMEIMPROVEMENT", "loan_grade": "A",
        "loan_amnt": 5000.0, "loan_int_rate": 7.25, "loan_percent_income": 0.05,
        "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 10,
    }])
    credit_result = score_credit_risk(applicant)  # credit risk score, threshold 0.45

    raw_transactions = pd.read_csv("transactions_to_score.csv")  # 30 raw features; no Class
    fraud_result = score_transaction_fraud(raw_transactions)  # fraud score, threshold 0.35
    ```

  - Credit risk interpretation preserves the credit notebook's tiers: `Tier 1 (Prime)` below 0.15, `Tier 2 (Moderate)` from 0.15 to below 0.45, and `Tier 3 (High Default Risk)` at or above 0.45. Fraud categories and threshold remain as defined in Phase 6.7. Both saved model artifacts and metadata files must exist locally; joblib artifacts remain ignored by Git.
  - Tests use mocks for the new engine unit cases and include artifact-path/error validation. Run all project tests with `python -m unittest discover -s tests -v`; the credit scoring pipeline is never retrained during inference. No combined score is calculated because the benchmark datasets do not share a verified customer identity, and the two model scores are not calibrated to a common outcome.

- **Phase 8: Flask REST API**:
  - Added a modular Flask app factory, routes, request validators and JSON serializers under `src/api/`. The API calls the Phase 7 engine; it contains no model training or duplicated feature engineering. Flask was already pinned in `requirements.txt`, so no dependency change was needed.
  - `GET /api/health` reports service availability separately from model readiness. It checks for the local credit artifact and metadata plus the fraud artifact without loading models. Per-model `loaded` fields reflect process-local model caches.
  - `POST /api/credit-risk/score` accepts one JSON object with the exact 11 Phase 7 applicant fields. The selected threshold remains 0.45 and the existing credit tiers are preserved. `POST /api/fraud-risk/score` accepts one JSON object with `Time`, `V1` through `V28`, and `Amount`; it rejects `Class`, keeps threshold 0.35 and delegates preprocessing to the fraud engine.
  - Successful requests use `{ "success": true, "data": { ... } }` with the Phase 7 response fields: `risk_type`, `model_name`, `model_version`, `risk_score`, `predicted_class`, `risk_category`, `decision_threshold`, `score_interpretation`, and `metadata`. Scores remain separate estimates of loan default and transaction fraud; no combined score is produced.
  - Error responses use `{ "success": false, "error": { "code": "...", "message": "..." } }`: 400 invalid JSON, 422 validation failure, 404 unknown route, 405 method not allowed, 503 model unavailable, and 500 unexpected internal error. Responses do not expose local paths or stack traces.
  - Model instances are loaded lazily and cached in-process after the first score request; health checks do not trigger loading. Restart the service after replacing artifacts, or call `clear_model_caches()` from the engine when controlled reload is needed. Required local artifacts are `models/credit_risk_xgb.joblib`, `models/credit_risk_metadata.json`, and `models/fraud_detection_xgb.joblib`; the joblib files are ignored by Git.

  **Start locally from the repository root:**

  ```powershell
  python -m src.api.app
  ```

  The development server listens on `http://127.0.0.1:5000` with debug mode disabled.

  **Example credit-risk JSON body:**

  ```json
  {
    "person_age": 38,
    "person_income": 110000.0,
    "person_home_ownership": "OWN",
    "person_emp_length": 12.0,
    "loan_intent": "HOMEIMPROVEMENT",
    "loan_grade": "A",
    "loan_amnt": 5000.0,
    "loan_int_rate": 7.25,
    "loan_percent_income": 0.05,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 10
  }
  ```

  **Example fraud-risk JSON body:**

  ```json
  {
    "Time": 1000.0,
    "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0,
    "V5": 0.0, "V6": 0.0, "V7": 0.0, "V8": 0.0,
    "V9": 0.0, "V10": 0.0, "V11": 0.0, "V12": 0.0,
    "V13": 0.0, "V14": 0.0, "V15": 0.0, "V16": 0.0,
    "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
    "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0,
    "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0,
    "Amount": 12.5
  }
  ```

  **Credit-risk request (PowerShell):**

  ```powershell
  $creditApplicant = @{
      person_age = 38; person_income = 110000.0; person_home_ownership = "OWN"
      person_emp_length = 12.0; loan_intent = "HOMEIMPROVEMENT"; loan_grade = "A"
      loan_amnt = 5000.0; loan_int_rate = 7.25; loan_percent_income = 0.05
      cb_person_default_on_file = "N"; cb_person_cred_hist_length = 10
  } | ConvertTo-Json
  Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/credit-risk/score" `
      -Method Post -ContentType "application/json" -Body $creditApplicant
  ```

  **Fraud-risk request (PowerShell):**

  ```powershell
  $transaction = @{ Time = 1000.0; Amount = 12.5 }
  1..28 | ForEach-Object { $transaction["V$_"] = 0.0 }
  $fraudTransaction = $transaction | ConvertTo-Json
  Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/fraud-risk/score" `
      -Method Post -ContentType "application/json" -Body $fraudTransaction
  ```

  Example success envelope:

  ```json
  {
    "success": true,
    "data": {
      "risk_type": "credit_default",
      "model_name": "Credit Risk XGBoost Classifier",
      "model_version": "1.0.0",
      "risk_score": 0.000924,
      "predicted_class": 0,
      "risk_category": "Tier 1 (Prime)",
      "decision_threshold": 0.45,
      "score_interpretation": "Estimated probability of loan default from the credit-risk model.",
      "metadata": {
        "decision_threshold_selection": "Empirical F1-score maximization via 5-fold Stratified Cross-Validation on training partition without test contamination",
        "default_threshold": 0.5,
        "artifact": "models/credit_risk_xgb.joblib"
      }
    }
  }
  ```

  Run API tests with `python -m unittest discover -s tests -p "test_api.py" -v`. Run all API, Phase 6 and Phase 7 regression tests with `python -m unittest discover -s tests -v`. Ordinary API tests mock engine scoring; live checks require the local model artifacts.

- **Phase 9: Power BI Dashboard**:
  - **Phase 9.1 - Data Preparation and Integration**: Added non-destructive Power BI SQL views in `sql/powerbi_views.sql`, reproducible credit and held-out fraud export scripts in `src/export_powerbi_credit_data.py` and `src/export_powerbi_fraud_data.py`, focused tests, and `docs/powerbi/POWER_BI_DATA_GUIDE.md`.
  - Generated local files under the Git-ignored `data/processed/powerbi/`: `fraud_test_scored.csv` (56,746 held-out test rows), `fraud_kpi_summary.csv`, `fraud_dataset_metadata.json`, `credit_scored.csv` (32,411 cleaned records), and `credit_dataset_metadata.json`. Re-running both export scripts produced byte-identical files.
  - Fraud export reproduces the established duplicate cleanup and stratified 64/16/20 split (`random_state=42`) and uses the saved model at threshold 0.35. Held-out test KPIs: 95 frauds, 56,651 legitimate, TP=71, FP=2, TN=56,649, FN=24, precision=0.9726, recall=0.7474, F1=0.8452, AP/PR-AUC=0.8297, ROC-AUC=0.9725.
  - Credit export reproduces credit notebook cleaning and stratified 80/20 split (`random_state=42`), using the saved pipeline and threshold 0.45. It labels 25,928 training rows as in-sample and 6,483 original held-out rows separately; it does not tune or retrain the model.
  - PostgreSQL views use only the existing `customers` and `loans` tables. SQL static safety/definition checks passed, but this environment could not authenticate to the local database (no password configured), so the views have not been applied or queried yet. Phase 9.1 data preparation is implemented; live database validation remains pending. No Power BI dashboard or `.pbix` file is claimed as complete.
  - The credit benchmark and fraud benchmark remain independent; do not join them or combine their scores. See `docs/powerbi/POWER_BI_DATA_GUIDE.md` for connection steps, exports, report suggestions, and caveats.
  - **Phase 9.2 - React Frontend Foundation and Design System**: Added a standalone React 18, TypeScript, and Vite application under `dashboard/`. The responsive shell provides Overview, Credit Intelligence, Fraud Intelligence, Model Intelligence, and Portfolio Analytics navigation; the latter four detailed workspaces are explicitly marked in development. The Overview reads live service health from the Phase 8 `GET /api/health` endpoint and separately reads an aggregate JSON snapshot generated from Phase 9.1 exports by `dashboard/scripts/build_overview_snapshot.py`. It displays only the original held-out credit partition (6,483 applications; 1,418 observed defaults; 21.87% observed default rate; 21.94% mean estimated default probability) and the independent fraud held-out metrics at threshold 0.35 (56,746 transactions; 95 frauds; TP=71, FP=2, TN=56,649, FN=24; precision=97.26%, recall=74.74%, F1=84.52%, PR-AUC=82.97%, ROC-AUC=97.25%). This is benchmark data, not live portfolio activity; no underlying customer rows are displayed. The dashboard does not call scoring endpoints or combine the credit and fraud benchmarks.
  - Start the Flask API from the repository root with `python -m src.api.app`, then run the frontend with `cd dashboard; npm install; npm run dev`. The Vite development proxy forwards `/api` requests to `http://127.0.0.1:5000`; set `VITE_API_BASE_URL` when using a separately hosted API. Refresh the aggregate snapshot after regenerating Phase 9.1 exports with `python scripts/build_overview_snapshot.py` from `dashboard/`. Frontend verification: `npm run build` passes; `npm test` passes all three overview, API-offline fallback, and navigation checks. Detailed analytical dashboard workspaces and live database integration remain future work.

## Database Execution

To execute the credit risk analytical queries against PostgreSQL:

```powershell
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -d banking_risk_intelligence -f sql/credit_queries.sql
```

