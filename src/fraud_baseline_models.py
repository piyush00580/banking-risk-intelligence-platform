"""
fraud_baseline_models.py
------------------------
Reusable baseline modeling and evaluation pipelines for transaction-level fraud detection
within the Banking Risk Intelligence Platform.

Baseline Models:
1. Standard Logistic Regression: L2-regularized linear model with StandardScaler.
2. Balanced Logistic Regression: Class-weighted linear model (class_weight='balanced') with StandardScaler.
3. Random Forest Classifier: Ensemble tree architecture (class_weight='balanced_subsample').

Evaluation functions support threshold-aware scoring on held-out sets, returning
comprehensive metrics (PR-AUC, ROC-AUC, Precision, Recall, F1, Confusion Matrix).
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)


def build_standard_logistic_regression(
    max_iter: int = 1000,
    random_state: int = 42,
    C: float = 1.0
) -> Pipeline:
    """
    Construct Model A: Standard Logistic Regression pipeline with StandardScaler.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs'
        ))
    ])


def build_balanced_logistic_regression(
    max_iter: int = 1000,
    random_state: int = 42,
    C: float = 1.0
) -> Pipeline:
    """
    Construct Model B: Balanced Logistic Regression pipeline with StandardScaler and class_weight='balanced'.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            C=C,
            class_weight='balanced',
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs'
        ))
    ])


def build_random_forest_baseline(
    n_estimators: int = 100,
    max_depth: Optional[int] = 12,
    min_samples_split: int = 10,
    min_samples_leaf: int = 2,
    class_weight: str = 'balanced_subsample',
    random_state: int = 42,
    n_jobs: int = -1
) -> RandomForestClassifier:
    """
    Construct Model C: Baseline Random Forest Classifier.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )


def evaluate_fraud_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Compute comprehensive fraud detection metrics for a given model and decision threshold.
    
    Parameters:
    -----------
    model : fitted classifier or pipeline
    X : pd.DataFrame
    y : pd.Series
    threshold : float (default=0.50)
    
    Returns:
    --------
    dict containing PR-AUC, ROC-AUC, Precision, Recall, F1, Confusion Matrix, and Counts.
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
    
    return {
        'threshold': threshold,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'confusion_matrix': cm,
        'probabilities': probs,
        'predictions': preds
    }


def find_best_f1_threshold(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    thresholds: Optional[np.ndarray] = None
) -> Tuple[float, float, pd.DataFrame]:
    """
    Perform a grid sweep over decision thresholds on the validation partition to select
    the threshold that maximizes F1-score without test-set contamination.
    
    Parameters:
    -----------
    model : fitted classifier or pipeline
    X_val : validation features
    y_val : validation labels
    thresholds : array of thresholds to test (default np.arange(0.05, 0.95, 0.05))
    
    Returns:
    --------
    best_threshold, best_f1, sweep_dataframe
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)
        
    probs = model.predict_proba(X_val)[:, 1]
    
    records = []
    best_t = 0.50
    best_f1 = -1.0
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = cm.ravel()
        
        records.append({
            'threshold': round(float(t), 2),
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
            best_t = round(float(t), 2)
            
    df_sweep = pd.DataFrame(records)
    return best_t, best_f1, df_sweep
