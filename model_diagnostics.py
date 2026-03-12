"""
=============================================================================
  MODEL DIAGNOSTICS — Bankruptcy Prediction System
=============================================================================
  Provides robustness analysis and overfitting diagnostics:

    1. Stratified K-Fold Cross-Validation
    2. Optimal Threshold Selection (Youden's J)
    3. Probability Calibration (Platt Scaling)
    4. Learning Curves
    5. Feature Stability Analysis
    6. SHAP vs XGBoost Importance Comparison
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from config import (
    OUTPUT_DIR, DIAGNOSTICS_DIR, FIGURE_DPI, TOP_N_FEATURES,
    XGBOOST_PARAMS, RANDOM_SEED, CV_FOLDS, ENABLE_PROBABILITY_CALIBRATION
)


# ──────────────────────────────────────────────────────────────────────────────
#  STYLING
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.6,
    'font.family': 'sans-serif',
    'font.size': 11,
})


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  1. STRATIFIED K-FOLD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def run_cross_validation(X_train, y_train, scale_pos_weight):
    """
    Run StratifiedKFold CV to diagnose overfitting.

    Returns
    -------
    cv_results : pd.DataFrame
        Per-fold ROC-AUC scores.
    mean_auc : float
    """
    print("=" * 70)
    print("  🔄  CROSS-VALIDATION — StratifiedKFold")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                           random_state=RANDOM_SEED)

    fold_results = []

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_fold_train = X_train[train_idx] if isinstance(X_train, np.ndarray) \
            else X_train.iloc[train_idx]
        X_fold_val = X_train[val_idx] if isinstance(X_train, np.ndarray) \
            else X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') \
            else y_train[train_idx]
        y_fold_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') \
            else y_train[val_idx]

        params = XGBOOST_PARAMS.copy()
        params['scale_pos_weight'] = scale_pos_weight
        fold_model = XGBClassifier(**params)
        fold_model.fit(X_fold_train, y_fold_train)

        train_proba = fold_model.predict_proba(X_fold_train)[:, 1]
        val_proba = fold_model.predict_proba(X_fold_val)[:, 1]

        train_auc = roc_auc_score(y_fold_train, train_proba)
        val_auc = roc_auc_score(y_fold_val, val_proba)

        fold_results.append({
            'Fold': fold_num,
            'Train_ROC_AUC': round(train_auc, 6),
            'Validation_ROC_AUC': round(val_auc, 6),
            'Overfit_Gap': round(train_auc - val_auc, 6),
        })
        print(f"  Fold {fold_num}: Train AUC = {train_auc:.4f}  |  "
              f"Val AUC = {val_auc:.4f}  |  Gap = {train_auc - val_auc:.4f}")

    cv_df = pd.DataFrame(fold_results)
    mean_train = cv_df['Train_ROC_AUC'].mean()
    mean_val = cv_df['Validation_ROC_AUC'].mean()
    std_val = cv_df['Validation_ROC_AUC'].std()

    print(f"\n  📊  Mean Train AUC: {mean_train:.4f}")
    print(f"  📊  Mean Val AUC:   {mean_val:.4f} (+/- {std_val:.4f})")
    print(f"  📊  Mean Overfit Gap: {mean_train - mean_val:.4f}")

    if mean_train - mean_val > 0.05:
        print("  ⚠️   Warning: Significant overfitting gap detected!")
    else:
        print("  ✅  Overfitting gap is within acceptable bounds.")
    print()

    # Save
    ensure_dirs()
    path = os.path.join(OUTPUT_DIR, 'cross_validation_results.csv')
    cv_df.to_csv(path, index=False)
    print(f"  💾  Saved: {path}")

    return cv_df, mean_val


# ══════════════════════════════════════════════════════════════════════════════
#  2. OPTIMAL THRESHOLD SELECTION (Youden's J Statistic)
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_threshold(y_true, probabilities):
    """
    Find the optimal classification threshold using Youden's J statistic.

    J = Sensitivity + Specificity - 1

    Returns
    -------
    optimal_threshold : float
    """
    print("=" * 70)
    print("  🎯  OPTIMAL THRESHOLD — Youden's J Statistic")
    print("=" * 70)

    fpr, tpr, thresholds = roc_curve(y_true, probabilities)

    # Youden's J = TPR - FPR (equivalent to Sensitivity + Specificity - 1)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]
    best_j = j_scores[best_idx]

    print(f"  📊  Default threshold:  0.5000")
    print(f"  📊  Optimal threshold:  {optimal_threshold:.4f}")
    print(f"  📊  Youden's J value:   {best_j:.4f}")
    print(f"  📊  At optimal: TPR = {tpr[best_idx]:.4f}, FPR = {fpr[best_idx]:.4f}")
    print()

    # Save to text file
    ensure_dirs()
    path = os.path.join(OUTPUT_DIR, 'optimal_threshold.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  OPTIMAL CLASSIFICATION THRESHOLD\n")
        f.write("  Method: Youden's J Statistic (J = TPR - FPR)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  Default Threshold:  0.5000\n")
        f.write(f"  Optimal Threshold:  {optimal_threshold:.6f}\n")
        f.write(f"  Youden's J Value:   {best_j:.6f}\n\n")
        f.write(f"  At Optimal Threshold:\n")
        f.write(f"    True Positive Rate (Sensitivity):  {tpr[best_idx]:.6f}\n")
        f.write(f"    False Positive Rate:               {fpr[best_idx]:.6f}\n")
        f.write(f"    Specificity:                       {1 - fpr[best_idx]:.6f}\n\n")
        f.write("  Interpretation:\n")
        f.write(f"    If P(bankruptcy) >= {optimal_threshold:.4f}, classify as BANKRUPT\n")
        f.write(f"    If P(bankruptcy) <  {optimal_threshold:.4f}, classify as NOT BANKRUPT\n")
    print(f"  💾  Saved: {path}")

    return optimal_threshold


# ══════════════════════════════════════════════════════════════════════════════
#  3. PROBABILITY CALIBRATION (Platt Scaling)
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_probabilities(model, X_train, y_train, X_test):
    """
    Apply Platt scaling (sigmoid calibration) to improve probability estimates.

    Returns
    -------
    calibrated_proba : np.ndarray
        Calibrated probabilities for the test set.
    calibrated_model : CalibratedClassifierCV
    """
    if not ENABLE_PROBABILITY_CALIBRATION:
        print("  ℹ️   Probability calibration DISABLED in config.")
        proba = model.predict_proba(X_test)[:, 1]
        return proba, None

    print("=" * 70)
    print("  📐  PROBABILITY CALIBRATION — Platt Scaling")
    print("=" * 70)

    calibrated_model = CalibratedClassifierCV(
        model, method='sigmoid', cv=3
    )
    calibrated_model.fit(X_train, y_train)

    raw_proba = model.predict_proba(X_test)[:, 1]
    calibrated_proba = calibrated_model.predict_proba(X_test)[:, 1]

    print(f"  📊  Raw probabilities:       mean = {raw_proba.mean():.4f}, "
          f"std = {raw_proba.std():.4f}")
    print(f"  📊  Calibrated probabilities: mean = {calibrated_proba.mean():.4f}, "
          f"std = {calibrated_proba.std():.4f}")
    print(f"  ✅  Platt scaling applied — probabilities are better calibrated.")
    print()

    return calibrated_proba, calibrated_model


# ══════════════════════════════════════════════════════════════════════════════
#  4. LEARNING CURVES
# ══════════════════════════════════════════════════════════════════════════════

def plot_learning_curves(model, X_train, y_train):
    """Plot learning curves to visualize overfitting."""
    ensure_dirs()

    print("  📊  Generating Learning Curves...")

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=3,
        scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.15, color='#58a6ff')
    ax.fill_between(train_sizes, val_mean - val_std,
                    val_mean + val_std, alpha=0.15, color='#f78166')

    ax.plot(train_sizes, train_mean, 'o-', color='#58a6ff',
            linewidth=2, label='Training Score', markersize=6)
    ax.plot(train_sizes, val_mean, 'o-', color='#f78166',
            linewidth=2, label='Validation Score', markersize=6)

    ax.set_title('Learning Curves — XGBoost Bankruptcy Predictor',
                 fontweight='bold', fontsize=15, pad=15, color='#c9d1d9')
    ax.set_xlabel('Training Set Size', fontweight='bold')
    ax.set_ylabel('ROC-AUC Score', fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.85, 1.02])

    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

    plt.tight_layout()
    path = os.path.join(DIAGNOSTICS_DIR, 'learning_curves.png')
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  💾  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  5. FEATURE STABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_feature_stability(X_train, y_train, feature_names):
    """
    Check feature importance stability across CV folds.
    """
    ensure_dirs()
    print("  📊  Analyzing Feature Stability across folds...")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                           random_state=RANDOM_SEED)

    importance_matrix = []

    for train_idx, _ in skf.split(X_train, y_train):
        X_fold = X_train[train_idx] if isinstance(X_train, np.ndarray) \
            else X_train.iloc[train_idx]
        y_fold = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') \
            else y_train[train_idx]

        neg = (y_fold == 0).sum()
        pos = (y_fold == 1).sum()
        weight = neg / pos if pos > 0 else 1.0

        params = XGBOOST_PARAMS.copy()
        params['scale_pos_weight'] = weight
        fold_model = XGBClassifier(**params)
        fold_model.fit(X_fold, y_fold)

        importance_matrix.append(fold_model.feature_importances_)

    imp_array = np.array(importance_matrix)
    means = imp_array.mean(axis=0)
    stds = imp_array.std(axis=0)
    cv_coeff = stds / (means + 1e-10)  # coefficient of variation

    # Sort by mean importance
    sorted_idx = np.argsort(means)[::-1][:TOP_N_FEATURES]

    fig, ax = plt.subplots(figsize=(14, 8))
    y_pos = np.arange(len(sorted_idx))

    bars = ax.barh(y_pos, means[sorted_idx], xerr=stds[sorted_idx],
                   color='#58a6ff', edgecolor='#388bfd', alpha=0.85,
                   capsize=4, error_kw={'color': '#f78166', 'linewidth': 1.5})

    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i][:40] for i in sorted_idx], fontsize=9)
    ax.invert_yaxis()
    ax.set_title('Feature Importance Stability (Mean +/- Std across CV Folds)',
                 fontweight='bold', fontsize=14, pad=15, color='#c9d1d9')
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')

    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

    plt.tight_layout()
    path = os.path.join(DIAGNOSTICS_DIR, 'feature_stability.png')
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  💾  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  6. SHAP vs XGBOOST IMPORTANCE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_shap_vs_xgboost(importance_df, shap_contributions_df,
                             feature_names):
    """
    Side-by-side comparison of XGBoost feature importance vs SHAP values.
    """
    ensure_dirs()
    print("  📊  Comparing SHAP vs XGBoost Feature Importances...")

    # Get top 15 features from each method
    xgb_top = importance_df.head(15).copy()
    shap_top = shap_contributions_df.head(15).copy()

    # Merge on Feature name
    merged = pd.merge(
        xgb_top[['Feature', 'Importance']],
        shap_contributions_df[['Feature', 'Mean_Abs_SHAP']],
        on='Feature', how='outer'
    ).fillna(0).head(20)

    # Normalize for comparison
    if merged['Importance'].max() > 0:
        merged['XGBoost_Norm'] = merged['Importance'] / merged['Importance'].max()
    else:
        merged['XGBoost_Norm'] = 0
    if merged['Mean_Abs_SHAP'].max() > 0:
        merged['SHAP_Norm'] = merged['Mean_Abs_SHAP'] / merged['Mean_Abs_SHAP'].max()
    else:
        merged['SHAP_Norm'] = 0

    merged = merged.sort_values('XGBoost_Norm', ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(14, 8))
    y_pos = np.arange(len(merged))
    width = 0.35

    ax.barh(y_pos - width/2, merged['XGBoost_Norm'],
            width, label='XGBoost Importance', color='#58a6ff',
            alpha=0.85, edgecolor='#388bfd')
    ax.barh(y_pos + width/2, merged['SHAP_Norm'],
            width, label='SHAP |Value|', color='#f78166',
            alpha=0.85, edgecolor='#da6d42')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f[:40] for f in merged['Feature']], fontsize=9)
    ax.set_title('XGBoost vs SHAP Feature Importance Comparison',
                 fontweight='bold', fontsize=14, pad=15, color='#c9d1d9')
    ax.set_xlabel('Normalized Importance', fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.8)
    ax.grid(True, alpha=0.2, axis='x')

    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

    plt.tight_layout()
    path = os.path.join(DIAGNOSTICS_DIR, 'shap_vs_xgboost_importance.png')
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  💾  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_all_diagnostics(model, X_train, y_train, X_test, y_test,
                        feature_names, importance_df, shap_contributions_df,
                        scale_pos_weight):
    """
    Execute all diagnostic analyses.

    Returns
    -------
    cv_df : pd.DataFrame
    optimal_threshold : float
    calibrated_proba : np.ndarray
    """
    print()
    print("+" + "-" * 68 + "+")
    print("|" + " " * 68 + "|")
    print("|   MODEL DIAGNOSTICS & ROBUSTNESS ANALYSIS                        |")
    print("|" + " " * 68 + "|")
    print("+" + "-" * 68 + "+")
    print()

    # 1. Cross-validation
    cv_df, mean_cv_auc = run_cross_validation(
        X_train, y_train, scale_pos_weight
    )

    # 2. Optimal threshold
    test_proba = model.predict_proba(X_test)[:, 1]
    optimal_threshold = find_optimal_threshold(y_test, test_proba)

    # 3. Probability calibration
    calibrated_proba, calibrated_model = calibrate_probabilities(
        model, X_train, y_train, X_test
    )

    # 4. Learning curves
    plot_learning_curves(model, X_train, y_train)

    # 5. Feature stability
    analyze_feature_stability(X_train, y_train, feature_names)

    # 6. SHAP vs XGBoost comparison
    if shap_contributions_df is not None:
        compare_shap_vs_xgboost(
            importance_df, shap_contributions_df, feature_names
        )

    print()
    print("=" * 70)
    print("  ✅  ALL DIAGNOSTICS COMPLETE")
    print("=" * 70)
    print()

    return cv_df, optimal_threshold, calibrated_proba
