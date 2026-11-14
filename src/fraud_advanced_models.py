"""
fraud_advanced_models.py
------------------------
Reusable advanced gradient-boosting model architectures and benchmark utilities
for transaction-level fraud detection within the Banking Risk Intelligence Platform.

Models implemented:
1. XGBoost Classifier (Standard, Weighted, and Regularized configurations)
2. LightGBM Classifier (Standard, Regularized/MinChild, and Weighted configurations)

Supports controlled hyperparameter grids, validation threshold tuning,
and unified evaluation against Phase 6.4 baseline models.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def build_xgboost_fraud_model(
    n_estimators: int = 150,
    max_depth: int = 5,
    learning_rate: float = 0.08,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
    n_jobs: int = 4
) -> XGBClassifier:
    """
    Construct an XGBoost Classifier configured for imbalanced fraud detection.
    """
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric='logloss',
        n_jobs=n_jobs
    )


def build_lightgbm_fraud_model(
    n_estimators: int = 150,
    num_leaves: int = 31,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    min_child_samples: int = 50,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
    n_jobs: int = 4
) -> LGBMClassifier:
    """
    Construct a LightGBM Classifier configured with regularization and min_child_samples
    to prevent leaf-wise overfitting on extreme class imbalance.
    """
    return LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        verbose=-1,
        n_jobs=n_jobs
    )


def evaluate_advanced_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Compute comprehensive fraud detection metrics for an advanced model at a specified threshold.
    """
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    
    pr_auc = float(average_precision_score(y, probs))
    roc_auc = float(roc_auc_score(y, probs))
    prec = float(precision_score(y, preds, zero_division=0))
    rec = float(recall_score(y, preds, zero_division=0))
    f1 = float(f1_score(y, preds, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    return {
        'threshold': threshold,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'specificity': spec,
        'fpr': fpr,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'confusion_matrix': cm,
        'probabilities': probs,
        'predictions': preds
    }


def find_best_advanced_threshold(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    thresholds: Optional[np.ndarray] = None
) -> Tuple[float, float, pd.DataFrame]:
    """
    Perform a grid sweep over decision thresholds on the validation partition to identify
    the operating threshold that maximizes F1-score without test-set leakage.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)
        
    probs = model.predict_proba(X_val)[:, 1]
    
    records = []
    best_t = 0.50
    best_f1 = -1.0
    
    for t in thresholds:
        t_r = round(float(t), 2)
        preds = (probs >= t_r).astype(int)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = cm.ravel()
        
        records.append({
            'threshold': t_r,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t_r
            
    df_sweep = pd.DataFrame(records)
    return best_t, best_f1, df_sweep
