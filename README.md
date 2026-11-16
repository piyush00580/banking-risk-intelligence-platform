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

Phase 6 - Fraud Detection (Milestone 6.7: Model Packaging and Reproducible Inference Completed)

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

## Database Execution

To execute the credit risk analytical queries against PostgreSQL:

```powershell
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -d banking_risk_intelligence -f sql/credit_queries.sql
```

