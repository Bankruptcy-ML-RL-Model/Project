"""
=============================================================================
  MODEL TRAINER — Bankruptcy Prediction System
=============================================================================
  Trains the XGBoost Classifier with class imbalance handling,
  and provides feature importance extraction.
=============================================================================
"""

import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from config import XGBOOST_PARAMS, RANDOM_STATE


def compute_scale_pos_weight(y_train):
    """
    Compute scale_pos_weight to handle class imbalance.

    scale_pos_weight = count(negative) / count(positive)
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    weight = neg / pos if pos > 0 else 1.0
    return weight


def train_xgboost(X_train, y_train, feature_names=None):
    """
    Train XGBoost Classifier with optimized hyperparameters.

    Handles class imbalance via scale_pos_weight.

    Parameters
    ----------
    X_train : array-like
        Scaled training features.
    y_train : array-like
        Training labels.
    feature_names : list, optional
        Feature names for importance analysis.

    Returns
    -------
    model : XGBClassifier
        Trained model.
    training_time : float
        Time taken to train (seconds).
    """
    print("=" * 70)
    print("  🤖  TRAINING XGBoost CLASSIFIER")
    print("=" * 70)

    # Compute class weight for imbalanced data
    scale_weight = compute_scale_pos_weight(y_train)
    print(f"  ⚖️   Class imbalance weight (scale_pos_weight): {scale_weight:.2f}")
    print()

    # Build model with parameters
    params = XGBOOST_PARAMS.copy()
    params["scale_pos_weight"] = scale_weight

    model = XGBClassifier(**params)

    # Print hyperparameters
    print(f"  📋  Hyperparameters:")
    print(f"       n_estimators:      {params['n_estimators']}")
    print(f"       max_depth:         {params['max_depth']}")
    print(f"       learning_rate:     {params['learning_rate']}")
    print(f"       subsample:         {params['subsample']}")
    print(f"       colsample_bytree:  {params['colsample_bytree']}")
    print(f"       min_child_weight:  {params['min_child_weight']}")
    print(f"       gamma:             {params['gamma']}")
    print(f"       reg_alpha:         {params['reg_alpha']}")
    print(f"       reg_lambda:        {params['reg_lambda']}")
    print()

    # Train
    print(f"  ⏳  Training in progress...")
    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"  ✅  Training completed in {training_time:.2f} seconds")
    print()

    return model, training_time


def get_feature_importance(model, feature_names):
    """
    Extract and rank feature importances from the trained model.

    Returns
    -------
    importance_df : pd.DataFrame
        DataFrame sorted by importance (descending).
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    importance_df['Rank'] = range(1, len(importance_df) + 1)
    importance_df['Cumulative'] = importance_df['Importance'].cumsum()

    print("=" * 70)
    print("  📊  TOP 15 MOST IMPORTANT FEATURES")
    print("=" * 70)
    for _, row in importance_df.head(15).iterrows():
        bar = "█" * int(row['Importance'] * 100)
        print(f"  {row['Rank']:>2}. {row['Feature'][:42]:<42} "
              f"{row['Importance']:.4f}  {bar}")
    print()

    return importance_df
