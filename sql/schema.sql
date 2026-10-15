-- Banking Risk Intelligence Platform
-- Phase 4: Database schema

CREATE TABLE IF NOT EXISTS customers (
    customer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    person_age INTEGER CHECK (person_age >= 18),
    person_income NUMERIC(14, 2) CHECK (person_income > 0),
    person_home_ownership VARCHAR(30),
    person_emp_length NUMERIC(6, 2),
    cb_person_default_on_file CHAR(1)
        CHECK (cb_person_default_on_file IN ('Y', 'N')),
    cb_person_cred_hist_length INTEGER
        CHECK (cb_person_cred_hist_length >= 0)
);

CREATE TABLE IF NOT EXISTS loans (
    loan_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id BIGINT NOT NULL REFERENCES customers(customer_id),
    loan_intent VARCHAR(40),
    loan_grade CHAR(1),
    loan_amnt NUMERIC(12, 2) CHECK (loan_amnt > 0),
    loan_int_rate NUMERIC(6, 3),
    loan_percent_income NUMERIC(8, 5),
    loan_status SMALLINT CHECK (loan_status IN (0, 1))
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    external_transaction_id VARCHAR(100) UNIQUE,
    customer_id BIGINT REFERENCES customers(customer_id),
    transaction_timestamp TIMESTAMP,
    transaction_amount NUMERIC(14, 2) CHECK (transaction_amount >= 0),
    transaction_type VARCHAR(40),
    merchant_category VARCHAR(80),
    transaction_status VARCHAR(30)
);

CREATE TABLE IF NOT EXISTS fraud_alerts (
    alert_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    transaction_id BIGINT NOT NULL REFERENCES transactions(transaction_id),
    fraud_score NUMERIC(6, 5)
        CHECK (fraud_score >= 0 AND fraud_score <= 1),
    predicted_fraud SMALLINT
        CHECK (predicted_fraud IN (0, 1)),
    investigation_status VARCHAR(30) DEFAULT 'PENDING',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common analytical joins and filters
CREATE INDEX IF NOT EXISTS idx_loans_customer_id
    ON loans(customer_id);

CREATE INDEX IF NOT EXISTS idx_loans_status
    ON loans(loan_status);

CREATE INDEX IF NOT EXISTS idx_transactions_customer_id
    ON transactions(customer_id);

CREATE INDEX IF NOT EXISTS idx_transactions_timestamp
    ON transactions(transaction_timestamp);

CREATE INDEX IF NOT EXISTS idx_fraud_alerts_transaction_id
    ON fraud_alerts(transaction_id);