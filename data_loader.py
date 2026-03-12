"""
=============================================================================
  DATA LOADER — Bankruptcy Prediction System
=============================================================================
  Handles data ingestion, validation, and train/test splitting.
=============================================================================
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE


def load_dataset(filepath=DATA_PATH):
    """
    Load the corporate bankruptcy dataset from CSV.

    Returns
    -------
    df : pd.DataFrame
        The complete dataset.
    """
    print("=" * 70)
    print("  📂  LOADING DATASET")
    print("=" * 70)

    df = pd.read_csv(filepath)

    print(f"  ✅  Dataset loaded successfully")
    print(f"  📊  Shape: {df.shape[0]:,} samples × {df.shape[1]} features")
    print(f"  🎯  Target column: '{TARGET_COLUMN}'")
    print(f"  🔢  Missing values: {df.isnull().sum().sum()}")
    print()

    # Target distribution
    target_counts = df[TARGET_COLUMN].value_counts()
    total = len(df)
    print(f"  📈  Target Distribution:")
    print(f"       Not Bankrupt (0): {target_counts.get(0, 0):>6,}  "
          f"({target_counts.get(0, 0)/total*100:.1f}%)")
    print(f"       Bankrupt     (1): {target_counts.get(1, 0):>6,}  "
          f"({target_counts.get(1, 0)/total*100:.1f}%)")
    print(f"       Imbalance Ratio:  1:{target_counts.get(0, 0) // max(target_counts.get(1, 0), 1)}")
    print()

    return df


def split_dataset(df, target_col=TARGET_COLUMN, test_size=TEST_SIZE,
                  random_state=RANDOM_STATE):
    """
    Split dataset into training (80%) and testing (20%) sets.

    Uses stratified splitting to preserve class distribution.

    Returns
    -------
    X_train, X_test, y_train, y_test : array-like
    """
    print("=" * 70)
    print("  ✂️   SPLITTING DATASET (80% Train / 20% Test)")
    print("=" * 70)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Preserve class distribution
    )

    print(f"  📦  Training set: {X_train.shape[0]:,} samples")
    print(f"  📦  Testing set:  {X_test.shape[0]:,} samples")
    print(f"  🎯  Train target — 0: {(y_train == 0).sum():,}  |  1: {(y_train == 1).sum():,}")
    print(f"  🎯  Test target  — 0: {(y_test == 0).sum():,}  |  1: {(y_test == 1).sum():,}")
    print()

    return X_train, X_test, y_train, y_test
