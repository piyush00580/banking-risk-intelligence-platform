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

Phase 6 — Fraud Detection (Milestone 6.4: Baseline Fraud Detection Modelling Completed)

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

## Database Execution

To execute the credit risk analytical queries against PostgreSQL:

```powershell
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -d banking_risk_intelligence -f sql/credit_queries.sql
```
