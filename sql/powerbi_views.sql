-- Phase 9.1: Non-destructive Power BI analytical views for credit data.
-- Default rates are fractions on labeled loans (0.0-1.0); NULL statuses are excluded
-- from the rate denominator but remain visible in total-loan counts.

CREATE OR REPLACE VIEW vw_credit_portfolio_summary AS
SELECT
    COUNT(*) AS total_loans,
    COUNT(*) FILTER (WHERE l.loan_status = 1) AS total_defaults,
    COUNT(*) FILTER (WHERE l.loan_status = 0) AS non_default_loans,
    COUNT(*) FILTER (WHERE l.loan_status = 1)::NUMERIC
        / NULLIF(COUNT(*) FILTER (WHERE l.loan_status IN (0, 1)), 0) AS default_rate,
    SUM(l.loan_amnt) AS total_loan_amount,
    AVG(l.loan_amnt) AS average_loan_amount,
    AVG(l.loan_int_rate) AS average_interest_rate
FROM loans AS l;

CREATE OR REPLACE VIEW vw_credit_risk_by_grade AS
SELECT
    l.loan_grade,
    COUNT(*) AS total_loans,
    COUNT(*) FILTER (WHERE l.loan_status = 1) AS defaults,
    COUNT(*) FILTER (WHERE l.loan_status = 1)::NUMERIC
        / NULLIF(COUNT(*) FILTER (WHERE l.loan_status IN (0, 1)), 0) AS default_rate,
    AVG(l.loan_amnt) AS average_loan_amount,
    AVG(l.loan_int_rate) AS average_interest_rate
FROM loans AS l
GROUP BY l.loan_grade;

CREATE OR REPLACE VIEW vw_credit_risk_by_intent AS
SELECT
    l.loan_intent,
    COUNT(*) AS total_loans,
    COUNT(*) FILTER (WHERE l.loan_status = 1) AS defaults,
    COUNT(*) FILTER (WHERE l.loan_status = 1)::NUMERIC
        / NULLIF(COUNT(*) FILTER (WHERE l.loan_status IN (0, 1)), 0) AS default_rate,
    AVG(l.loan_amnt) AS average_loan_amount
FROM loans AS l
GROUP BY l.loan_intent;

-- Age bands: Under 25, 25-34, 35-44, 45-54, 55+, Unknown.
-- Annual income bands use source numeric units: <25k, 25k-<50k, 50k-<100k, 100k+, Unknown.
CREATE OR REPLACE VIEW vw_credit_risk_by_customer_segment AS
SELECT
    c.person_home_ownership AS home_ownership,
    CASE
        WHEN c.person_age IS NULL THEN 'Unknown'
        WHEN c.person_age < 25 THEN 'Under 25'
        WHEN c.person_age < 35 THEN '25-34'
        WHEN c.person_age < 45 THEN '35-44'
        WHEN c.person_age < 55 THEN '45-54'
        ELSE '55+'
    END AS age_segment,
    CASE
        WHEN c.person_income IS NULL THEN 'Unknown'
        WHEN c.person_income < 25000 THEN 'Under 25k'
        WHEN c.person_income < 50000 THEN '25k-<50k'
        WHEN c.person_income < 100000 THEN '50k-<100k'
        ELSE '100k+'
    END AS income_segment,
    COUNT(*) AS total_loans,
    COUNT(*) FILTER (WHERE l.loan_status = 1) AS defaults,
    COUNT(*) FILTER (WHERE l.loan_status = 1)::NUMERIC
        / NULLIF(COUNT(*) FILTER (WHERE l.loan_status IN (0, 1)), 0) AS default_rate,
    AVG(l.loan_amnt) AS average_loan_amount
FROM customers AS c
JOIN loans AS l ON l.customer_id = c.customer_id
GROUP BY
    c.person_home_ownership,
    CASE
        WHEN c.person_age IS NULL THEN 'Unknown'
        WHEN c.person_age < 25 THEN 'Under 25'
        WHEN c.person_age < 35 THEN '25-34'
        WHEN c.person_age < 45 THEN '35-44'
        WHEN c.person_age < 55 THEN '45-54'
        ELSE '55+'
    END,
    CASE
        WHEN c.person_income IS NULL THEN 'Unknown'
        WHEN c.person_income < 25000 THEN 'Under 25k'
        WHEN c.person_income < 50000 THEN '25k-<50k'
        WHEN c.person_income < 100000 THEN '50k-<100k'
        ELSE '100k+'
    END;

CREATE OR REPLACE VIEW vw_credit_loan_details AS
SELECT
    c.customer_id,
    l.loan_id,
    c.person_age,
    c.person_income,
    c.person_home_ownership AS home_ownership,
    c.person_emp_length AS employment_length,
    c.cb_person_cred_hist_length AS credit_history_length,
    l.loan_intent,
    l.loan_grade,
    l.loan_amnt AS loan_amount,
    l.loan_int_rate AS interest_rate,
    -- Preserve the source model feature; do not recompute or overwrite it.
    l.loan_percent_income AS loan_to_income_ratio,
    l.loan_status,
    CASE
        WHEN l.loan_status = 1 THEN 'Default'
        WHEN l.loan_status = 0 THEN 'Non-default'
        ELSE 'Unknown'
    END AS outcome_description
FROM customers AS c
JOIN loans AS l ON l.customer_id = c.customer_id;
