"""
=============================================================================
  MODEL EVALUATOR — Bankruptcy Prediction System
=============================================================================
  Comprehensive model evaluation with all requested metrics:
  Accuracy, Precision, Recall, F1 Score, ROC-AUC.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the model on TRAINING data with all requested metrics.
    Also computes test set metrics for comparison.

    Parameters
    ----------
    model : XGBClassifier
    X_train, y_train : Training data
    X_test, y_test : Test data

    Returns
    -------
    results : dict
        All computed metrics and predictions.
    """
    print("=" * 70)
    print("  📊  MODEL EVALUATION RESULTS")
    print("=" * 70)

    # ── Training Set Predictions ──
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    # ── Test Set Predictions ──
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # ══════════════════════════════════════════════════════════════════════
    #  TRAINING SET METRICS (as requested)
    # ══════════════════════════════════════════════════════════════════════
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    train_roc_auc = roc_auc_score(y_train, y_train_proba)

    print()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │           TRAINING SET PERFORMANCE METRICS              │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print(f"  │   Accuracy:   {train_accuracy:.4f}  "
          f"({train_accuracy*100:.2f}%)                       │")
    print(f"  │   Precision:  {train_precision:.4f}  "
          f"({train_precision*100:.2f}%)                       │")
    print(f"  │   Recall:     {train_recall:.4f}  "
          f"({train_recall*100:.2f}%)                       │")
    print(f"  │   F1 Score:   {train_f1:.4f}  "
          f"({train_f1*100:.2f}%)                       │")
    print(f"  │   ROC-AUC:    {train_roc_auc:.4f}  "
          f"({train_roc_auc*100:.2f}%)                       │")
    print("  └─────────────────────────────────────────────────────────┘")

    # ══════════════════════════════════════════════════════════════════════
    #  TEST SET METRICS (for comparison / generalization check)
    # ══════════════════════════════════════════════════════════════════════
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    print()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │            TEST SET PERFORMANCE METRICS                 │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print(f"  │   Accuracy:   {test_accuracy:.4f}  "
          f"({test_accuracy*100:.2f}%)                       │")
    print(f"  │   Precision:  {test_precision:.4f}  "
          f"({test_precision*100:.2f}%)                       │")
    print(f"  │   Recall:     {test_recall:.4f}  "
          f"({test_recall*100:.2f}%)                       │")
    print(f"  │   F1 Score:   {test_f1:.4f}  "
          f"({test_f1*100:.2f}%)                       │")
    print(f"  │   ROC-AUC:    {test_roc_auc:.4f}  "
          f"({test_roc_auc*100:.2f}%)                       │")
    print("  └─────────────────────────────────────────────────────────┘")

    # ── Confusion Matrix ──
    print()
    print("  📋  Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"                     Predicted")
    print(f"                   Not Bankrupt  Bankrupt")
    print(f"  Actual Not Bankrupt   {tn:>5}        {fp:>5}")
    print(f"  Actual Bankrupt       {fn:>5}        {tp:>5}")
    print()
    print(f"  True Negatives:  {tn:>5}  |  False Positives: {fp:>5}")
    print(f"  False Negatives: {fn:>5}  |  True Positives:  {tp:>5}")

    # ── Classification Report ──
    print()
    print("  📋  Detailed Classification Report (Test Set):")
    print("  " + "-" * 60)
    report = classification_report(y_test, y_test_pred,
                                   target_names=["Not Bankrupt", "Bankrupt"])
    for line in report.split('\n'):
        print(f"  {line}")
    print()

    # ── ROC Curve Data ──
    train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
    test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)

    # ── Precision-Recall Curve Data ──
    train_pr_precision, train_pr_recall, _ = precision_recall_curve(
        y_train, y_train_proba)
    test_pr_precision, test_pr_recall, _ = precision_recall_curve(
        y_test, y_test_proba)
    train_avg_precision = average_precision_score(y_train, y_train_proba)
    test_avg_precision = average_precision_score(y_test, y_test_proba)

    results = {
        # Training metrics
        "train_accuracy": train_accuracy,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1": train_f1,
        "train_roc_auc": train_roc_auc,
        "train_predictions": y_train_pred,
        "train_probabilities": y_train_proba,
        # Test metrics
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "test_predictions": y_test_pred,
        "test_probabilities": y_test_proba,
        # Confusion matrix
        "confusion_matrix": cm,
        # ROC curve data
        "train_fpr": train_fpr,
        "train_tpr": train_tpr,
        "test_fpr": test_fpr,
        "test_tpr": test_tpr,
        # Precision-Recall curve data
        "train_pr_precision": train_pr_precision,
        "train_pr_recall": train_pr_recall,
        "test_pr_precision": test_pr_precision,
        "test_pr_recall": test_pr_recall,
        "train_avg_precision": train_avg_precision,
        "test_avg_precision": test_avg_precision,
    }

    return results
