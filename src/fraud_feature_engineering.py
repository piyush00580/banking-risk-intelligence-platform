"""
fraud_feature_engineering.py
----------------------------
Reusable feature engineering and data splitting pipeline for transaction-level
credit card fraud detection within the Banking Risk Intelligence Platform.

Features Engineered:
1. Amount_log: np.log1p(Amount) - Handles extreme right-skew in transaction amounts
2. is_zero_amount: (Amount == 0).astype(int) - Captures card-testing / authorization pings
3. Time_scaled: Scaled elapsed time (e.g. RobustScaler or StandardScaler) fitted strictly on training data
4. V14_V4_interaction: V14 * V4 - Interaction between top two separating features
5. V12_V11_interaction: V12 * V11 - Interaction between complementary separating features

Leakage-safe design:
- All learned transformations (scalers) fit solely on the training partition.
- Original raw features are preserved.
- Target 'Class' is strictly separated from feature matrices.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for engineering transaction-level fraud features.
    
    Transforms:
    - Amount_log: log1p(Amount)
    - is_zero_amount: binary indicator for Amount == 0.0
    - V14_V4_interaction: V14 * V4
    - V12_V11_interaction: V12 * V11
    - Time_scaled: Learned scaling of elapsed time (StandardScaler or RobustScaler)
    
    Parameters:
    -----------
    time_scaler : str, default='robust'
        Type of scaler for elapsed time: 'robust', 'standard', or None.
    preserve_original : bool, default=True
        Whether to retain the original Time and Amount columns in the output DataFrame.
    """

    def __init__(self, time_scaler: str = "robust", preserve_original: bool = True):
        self.time_scaler = time_scaler
        self.preserve_original = preserve_original
        self.scaler_: Optional[Union[RobustScaler, StandardScaler]] = None
        self.feature_names_out_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FraudFeatureEngineer":
        """
        Fit learned parameters (e.g., time scaler) strictly on the provided training partition.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix containing at least 'Time', 'Amount', 'V4', 'V11', 'V12', 'V14'.
        y : Optional[pd.Series]
            Ignored (present for scikit-learn Pipeline API compatibility).
            
        Returns:
        --------
        self : FraudFeatureEngineer
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame with named columns.")

        if self.time_scaler == "robust":
            self.scaler_ = RobustScaler()
            self.scaler_.fit(X[["Time"]])
        elif self.time_scaler == "standard":
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X[["Time"]])
        elif self.time_scaler is None:
            self.scaler_ = None
        else:
            raise ValueError(f"Unknown time_scaler '{self.time_scaler}'. Choose 'robust', 'standard', or None.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix to transform.
            
        Returns:
        --------
        pd.DataFrame
            Feature matrix with engineered features added.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        # Ensure target 'Class' is not modified or leaked
        has_class = "Class" in X.columns
        cols_to_process = [c for c in X.columns if c != "Class"]
        X_out = X[cols_to_process].copy()

        # 1. Transaction Amount Features
        if "Amount" in X_out.columns:
            X_out["Amount_log"] = np.log1p(X_out["Amount"].clip(lower=0))
            X_out["is_zero_amount"] = (X_out["Amount"] == 0.0).astype(int)

        # 2. Transaction Time Features (Leakage-safe learned scaling)
        if "Time" in X_out.columns:
            if self.scaler_ is not None:
                scaled_time = self.scaler_.transform(X_out[["Time"]])
                X_out["Time_scaled"] = scaled_time.ravel()
            else:
                X_out["Time_scaled"] = X_out["Time"].astype(float)

        # 3. PCA Feature Interaction Terms
        if "V14" in X_out.columns and "V4" in X_out.columns:
            X_out["V14_V4_interaction"] = X_out["V14"] * X_out["V4"]

        if "V12" in X_out.columns and "V11" in X_out.columns:
            X_out["V12_V11_interaction"] = X_out["V12"] * X_out["V11"]

        # Drop originals only if explicitly requested
        if not self.preserve_original:
            drop_cols = [c for c in ["Time", "Amount"] if c in X_out.columns]
            X_out.drop(columns=drop_cols, inplace=True)

        self.feature_names_out_ = list(X_out.columns)
        return X_out

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Return names of output features."""
        if self.feature_names_out_ is not None:
            return self.feature_names_out_
        raise RuntimeError("FraudFeatureEngineer must be fitted and transformed first.")


def clean_raw_fraud_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows across all dimensions in a deterministic, reproducible manner.
    
    Parameters:
    -----------
    df_raw : pd.DataFrame
        Raw credit card transaction DataFrame (284,807 rows, 31 cols).
        
    Returns:
    --------
    pd.DataFrame
        Deduplicated DataFrame (283,726 rows, 31 cols), indexed monotonically.
    """
    df_clean = df_raw.drop_duplicates(keep="first").reset_index(drop=True)
    return df_clean


def split_fraud_data_stratified(
    df: pd.DataFrame,
    target_col: str = "Class",
    train_ratio: float = 0.64,
    val_ratio: float = 0.16,
    test_ratio: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Execute a 3-way stratified split preserving minority fraud prevalence across partitions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned transaction dataset.
    target_col : str
        Target column name (default 'Class').
    train_ratio : float (default 0.64)
    val_ratio : float (default 0.16)
    test_ratio : float (default 0.20)
    random_state : int (default 42)
    
    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Step 1: Split off test set
    test_size = test_ratio
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Step 2: Split remaining train_val into train and val
    val_rel_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_rel_size, stratify=y_train_val, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_fraud_data_chronological(
    df: pd.DataFrame,
    time_col: str = "Time",
    target_col: str = "Class",
    train_ratio: float = 0.64,
    val_ratio: float = 0.16,
    test_ratio: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Execute a 3-way chronological split based on the elapsed Time column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned transaction dataset.
    time_col : str
        Time feature column name (default 'Time').
    target_col : str
        Target column name (default 'Class').
    train_ratio, val_ratio, test_ratio : float
    
    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    df_train = df_sorted.iloc[:n_train]
    df_val = df_sorted.iloc[n_train:n_train + n_val]
    df_test = df_sorted.iloc[n_train + n_val:]

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_val = df_val.drop(columns=[target_col])
    y_val = df_val[target_col]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    return X_train, X_val, X_test, y_train, y_val, y_test
