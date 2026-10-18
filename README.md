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

Phase 4 — Database Design & SQL Analytics (Credit Risk Milestone Completed)

## Implementation Progress

- **Phase 1: Project Foundation** — Repository structure, dependencies, virtual environment, and git tracking initialized.
- **Phase 2: Dataset Inspection & Validation** — Comprehensive data quality audit on 32,581 records (`notebooks/credit_risk_data_inspection.ipynb`), identifying missing values, duplicates, and domain constraints.
- **Phase 3: Credit Risk Exploratory Data Analysis** — Statistical analysis and visualizations (`notebooks/credit_risk_eda.ipynb`) benchmarking default patterns across loan grades, loan intent, and borrower characteristics.
- **Phase 4: Database Design & SQL Analytics (In Progress)**:
  - Relational schema defined in `sql/schema.sql` (`customers`, `loans`, `transactions`, `fraud_alerts`).
  - Credit data cleaning and ingestion via `src/load_credit_data.py`, populating 32,576 records each in `customers` and `loans`.
  - Five core credit risk analytical queries implemented in `sql/credit_queries.sql` and verified against PostgreSQL (`banking_risk_intelligence`):
    - Overall default rate: 21.82% (7,108 defaults / 32,576 loans).
    - Default rate by loan grade (Grade A at 9.96% to Grade G at 98.44%).
    - Default rate by loan purpose (Debt Consolidation highest at 28.59%).
    - Loan amounts and interest rates by outcome.
    - Customer-level loan risk summary join.

## Database Execution

To execute the credit risk analytical queries against PostgreSQL:

```powershell
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -d banking_risk_intelligence -f sql/credit_queries.sql
```