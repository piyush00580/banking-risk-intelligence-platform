"""
Dataset Acquisition Script: Credit Card Fraud Detection Dataset
===============================================================

Origin and Source:
------------------
The dataset is the canonical Credit Card Fraud Detection benchmark originally published
by Worldline and the Machine Learning Group (MLG) of Universite Libre de Bruxelles (ULB):
    Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Gianluca Bontempi.
    "Calibrating Probability with Undersampling for Unbalanced Classification."
    IEEE Symposium Series on Computational Intelligence (SSCI), 2015.

Public Mirror:
--------------
Hosted via Hugging Face Datasets repository under Open Database License (ODbL) v1.0.
URL: https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/resolve/main/creditcard.csv

Dataset Schema:
---------------
- Total records: 284,807 transactions (European cardholders in September 2013).
- Features:
  - Time: Number of seconds elapsed between each transaction and the first transaction.
  - V1-V28: Principal components obtained with PCA (due to confidentiality).
  - Amount: Transaction amount in Euros.
  - Class: 1 for fraudulent transactions, 0 for legitimate transactions.
- Imbalance: 492 fraud cases (0.172% fraud prevalence).
"""

import sys
import time
import urllib.request
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DESTINATION_FILE = DATA_DIR / "creditcard.csv"

DATA_URL = (
    "https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/resolve/main/creditcard.csv"
)

def download_dataset():
    if DESTINATION_FILE.exists() and DESTINATION_FILE.stat().st_size > 140_000_000:
        print(f"Dataset already exists at {DESTINATION_FILE} ({DESTINATION_FILE.stat().st_size:,} bytes).")
        return

    print(f"Downloading Credit Card Fraud Detection dataset from:")
    print(f"  {DATA_URL}")
    print(f"Saving to: {DESTINATION_FILE}")

    start_time = time.time()
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    with urllib.request.urlopen(req) as response, open(DESTINATION_FILE, "wb") as out_file:
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                sys.stdout.write(f"\rDownloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)")
                sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\nDownload completed in {elapsed:.1f} seconds. File size: {DESTINATION_FILE.stat().st_size:,} bytes.")

def validate_dataset():
    print("\nValidating downloaded dataset structure...")
    df = pd.read_csv(DESTINATION_FILE)
    print(f"Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    assert df.shape == (284807, 31), f"Unexpected shape: {df.shape}"
    assert "Class" in df.columns, "Target column 'Class' missing!"
    
    fraud_count = int(df["Class"].sum())
    non_fraud_count = len(df) - fraud_count
    fraud_rate = (fraud_count / len(df)) * 100
    
    print(f"Legitimate Transactions (Class=0): {non_fraud_count:,} ({100 - fraud_rate:.3f}%)")
    print(f"Fraudulent Transactions (Class=1): {fraud_count:,} ({fraud_rate:.3f}%)")
    print("Dataset validation successful!")

if __name__ == "__main__":
    download_dataset()
    validate_dataset()

