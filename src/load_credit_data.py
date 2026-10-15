import os
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "credit_risk_dataset.csv"

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "banking_risk_intelligence",
    "user": "postgres",
    "password": os.environ.get("PGPASSWORD"),
}

if not DB_CONFIG["password"]:
    raise RuntimeError(
        "Set the PGPASSWORD environment variable before running this script."
    )

df = pd.read_csv(CSV_PATH)

# Exclude impossible ages from the import without changing the raw CSV.
df = df.loc[df["person_age"] <= 100].copy()

# Treat employment lengths that exceed age as missing.
invalid_employment = df["person_emp_length"] > df["person_age"]
df.loc[invalid_employment, "person_emp_length"] = None

def clean_value(value):
    return None if pd.isna(value) else value

conn = psycopg2.connect(**DB_CONFIG)

try:
    with conn:
        with conn.cursor() as cur:
            for row in df.itertuples(index=False, name=None):
                (
                    age, income, home, emp_length, intent, grade,
                    loan_amnt, interest_rate, loan_status, ratio,
                    previous_default, credit_history
                ) = row

                cur.execute(
                    """
                    INSERT INTO customers (
                        person_age, person_income, person_home_ownership,
                        person_emp_length, cb_person_default_on_file,
                        cb_person_cred_hist_length
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING customer_id
                    """,
                    (
                        clean_value(age), clean_value(income),
                        clean_value(home), clean_value(emp_length),
                        clean_value(previous_default),
                        clean_value(credit_history)
                    ),
                )
                customer_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO loans (
                        customer_id, loan_intent, loan_grade,
                        loan_amnt, loan_int_rate,
                        loan_percent_income, loan_status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        customer_id, clean_value(intent), clean_value(grade),
                        clean_value(loan_amnt), clean_value(interest_rate),
                        clean_value(ratio), clean_value(loan_status)
                    ),
                )

    print(f"Imported {len(df):,} customer records.")
    print(f"Imported {len(df):,} loan records.")

finally:
    conn.close()