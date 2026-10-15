-- 1. Overall default rate
SELECT
    COUNT(*) AS total_loans,
    SUM(CASE WHEN loan_status = 1 THEN 1 ELSE 0 END) AS defaults,
    ROUND(
        100.0 * AVG(loan_status::NUMERIC), 2
    ) AS default_rate_pct
FROM loans;

-- 2. Default rate by loan grade
SELECT
    loan_grade,
    COUNT(*) AS total_loans,
    SUM(loan_status) AS defaults,
    ROUND(100.0 * AVG(loan_status::NUMERIC), 2) AS default_rate_pct
FROM loans
WHERE loan_status IS NOT NULL
GROUP BY loan_grade
ORDER BY default_rate_pct DESC;

-- 3. Default rate by loan purpose
SELECT
    loan_intent,
    COUNT(*) AS total_loans,
    ROUND(100.0 * AVG(loan_status::NUMERIC), 2) AS default_rate_pct
FROM loans
WHERE loan_status IS NOT NULL
GROUP BY loan_intent
ORDER BY default_rate_pct DESC;

-- 4. Loan amount and interest rate by outcome
SELECT
    loan_status,
    COUNT(*) AS total_loans,
    ROUND(AVG(loan_amnt), 2) AS avg_loan_amount,
    ROUND(AVG(loan_int_rate), 2) AS avg_interest_rate
FROM loans
GROUP BY loan_status
ORDER BY loan_status;

-- 5. Customer-level loan risk summary
SELECT
    c.customer_id,
    c.person_age,
    c.person_home_ownership,
    c.person_income,
    l.loan_intent,
    l.loan_grade,
    l.loan_amnt,
    l.loan_status,
    CASE
        WHEN l.loan_status = 1 THEN 'Default'
        WHEN l.loan_status = 0 THEN 'Non-default'
        ELSE 'Unknown'
    END AS risk_outcome
FROM customers c
JOIN loans l
    ON c.customer_id = l.customer_id
ORDER BY l.loan_amnt DESC
LIMIT 20;