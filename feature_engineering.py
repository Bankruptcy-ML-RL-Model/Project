"""
=============================================================================
  FEATURE ENGINEERING — Bankruptcy Prediction System
=============================================================================
  Performs exploratory data analysis, feature correlation analysis,
  and optional preprocessing (scaling) for the model pipeline.

  NOTE: XGBoost is tree-based and does NOT require feature scaling.
        Scaling is controlled via USE_FEATURE_SCALING in config.py.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import TARGET_COLUMN, USE_FEATURE_SCALING


def analyze_features(df):
    """
    Print a comprehensive feature analysis report.

    Parameters
    ----------
    df : pd.DataFrame
        The complete dataset.

    Returns
    -------
    feature_stats : pd.DataFrame
        Statistical summary of all features.
    """
    print("=" * 70)
    print("  🔍  FEATURE ANALYSIS REPORT")
    print("=" * 70)

    features = df.drop(columns=[TARGET_COLUMN])

    # Basic statistics
    print(f"\n  📊  Total Features: {features.shape[1]}")
    print(f"  📊  Total Samples:  {features.shape[0]:,}")
    print(f"  📊  Data Types:     {dict(features.dtypes.value_counts())}")

    # Feature categories by domain knowledge
    profitability = [c for c in features.columns if any(
        kw in c.lower() for kw in ['roa', 'profit', 'margin', 'eps', 'income']
    )]
    liquidity = [c for c in features.columns if any(
        kw in c.lower() for kw in ['current ratio', 'quick ratio', 'cash']
    )]
    leverage = [c for c in features.columns if any(
        kw in c.lower() for kw in ['debt', 'equity', 'liability', 'leverage',
                                    'borrowing', 'worth']
    )]
    growth = [c for c in features.columns if 'growth' in c.lower()]
    other = [c for c in features.columns if c not in
             profitability + liquidity + leverage + growth]

    print(f"\n  📂  Feature Categories:")
    print(f"       Profitability:  {len(profitability)} features")
    print(f"       Liquidity:      {len(liquidity)} features")
    print(f"       Leverage:       {len(leverage)} features")
    print(f"       Growth:         {len(growth)} features")
    print(f"       Other:          {len(other)} features")

    # Descriptive statistics
    stats = features.describe().T
    stats['skewness'] = features.skew()
    stats['kurtosis'] = features.kurtosis()

    print(f"\n  📈  Features with high skewness (|skew| > 2): "
          f"{(stats['skewness'].abs() > 2).sum()}")
    print(f"  📈  Features with high kurtosis (|kurt| > 7): "
          f"{(stats['kurtosis'].abs() > 7).sum()}")

    # Correlation with target
    corr_with_target = df.corr(numeric_only=True)[TARGET_COLUMN].drop(
        TARGET_COLUMN).sort_values(ascending=False)

    print(f"\n  🎯  Top 10 Features Correlated with Bankruptcy:")
    print(f"  {'─' * 60}")
    for i, (feat, corr) in enumerate(corr_with_target.head(10).items()):
        bar = "█" * int(abs(corr) * 40)
        print(f"  {i+1:>2}. {feat[:45]:<45} {corr:>+.4f}  {bar}")

    print(f"\n  🎯  Top 10 Negatively Correlated Features:")
    print(f"  {'─' * 60}")
    for i, (feat, corr) in enumerate(corr_with_target.tail(10).items()):
        bar = "█" * int(abs(corr) * 40)
        print(f"  {i+1:>2}. {feat[:45]:<45} {corr:>+.4f}  {bar}")

    print()
    return stats, corr_with_target


def get_correlation_matrix(df):
    """
    Compute the full feature correlation matrix.

    Returns
    -------
    corr_matrix : pd.DataFrame
    """
    return df.corr(numeric_only=True)


def preprocess_features(X_train, X_test, use_scaling=None):
    """
    Optionally scale features using StandardScaler.

    XGBoost does NOT require feature scaling (tree-based model).
    Scaling is controlled by the USE_FEATURE_SCALING config flag.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
    X_test  : pd.DataFrame or np.ndarray
    use_scaling : bool or None
        Override config flag. If None, uses USE_FEATURE_SCALING from config.

    Returns
    -------
    X_train_out : np.ndarray
    X_test_out  : np.ndarray
    scaler : StandardScaler or None
    """
    if use_scaling is None:
        use_scaling = USE_FEATURE_SCALING

    print("=" * 70)
    if use_scaling:
        print("  ⚙️   PREPROCESSING — Feature Scaling (StandardScaler)")
        print("=" * 70)

        scaler = StandardScaler()
        X_train_out = scaler.fit_transform(X_train)
        X_test_out = scaler.transform(X_test)

        print(f"  ✅  Training features scaled: mean ≈ {X_train_out.mean():.4f}, "
              f"std ≈ {X_train_out.std():.4f}")
        print(f"  ✅  Test features scaled:     mean ≈ {X_test_out.mean():.4f}, "
              f"std ≈ {X_test_out.std():.4f}")
    else:
        print("  ⚙️   PREPROCESSING — Scaling SKIPPED (XGBoost is tree-based)")
        print("=" * 70)
        print("  ℹ️   XGBoost does not require feature normalization.")
        print("  ℹ️   Set USE_FEATURE_SCALING = True in config.py to enable.")

        X_train_out = X_train.values if hasattr(X_train, 'values') else X_train
        X_test_out = X_test.values if hasattr(X_test, 'values') else X_test
        scaler = None

    print()
    return X_train_out, X_test_out, scaler
