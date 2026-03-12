"""
=============================================================================
  RISK CLASSIFIER — Bankruptcy Prediction System
=============================================================================
  Converts raw bankruptcy probabilities into human-readable risk categories.

  Probability × 100 → Risk Score:
      0–20   → 🟢 Safe
     20–40   → 🔵 Low Risk
     40–60   → 🟡 Moderate Risk
     60–80   → 🟠 High Risk
     80–100  → 🔴 Critical Risk
=============================================================================
"""

import numpy as np
import pandas as pd
from config import RISK_CATEGORIES


def classify_risk(probability):
    """
    Convert a single bankruptcy probability to a risk category.

    Parameters
    ----------
    probability : float
        Raw probability (0.0 to 1.0) from the model.

    Returns
    -------
    risk_score : float
        Probability multiplied by 100.
    risk_category : str
        Human-readable risk label.
    """
    risk_score = probability * 100

    for lower, upper, label in RISK_CATEGORIES:
        if lower <= risk_score < upper:
            return risk_score, label

    # Edge case: exactly 100
    return risk_score, RISK_CATEGORIES[-1][2]


def classify_all_risks(probabilities, original_indices=None):
    """
    Classify all samples into risk categories.

    Parameters
    ----------
    probabilities : array-like
        Raw probabilities from model.predict_proba()[:, 1].
    original_indices : array-like, optional
        Original DataFrame indices for tracking.

    Returns
    -------
    risk_df : pd.DataFrame
        DataFrame with risk scores and categories for all samples.
    """
    results = []
    for i, prob in enumerate(probabilities):
        score, category = classify_risk(prob)
        results.append({
            'Sample_Index': original_indices[i] if original_indices is not None else i,
            'Bankruptcy_Probability': prob,
            'Risk_Score': round(score, 2),
            'Risk_Category': category,
        })

    risk_df = pd.DataFrame(results)
    return risk_df


def print_risk_summary(risk_df, dataset_label="Test"):
    """
    Print a detailed summary of risk classification results.

    Parameters
    ----------
    risk_df : pd.DataFrame
        Output from classify_all_risks().
    dataset_label : str
        Label for the dataset (e.g., "Test", "Training").
    """
    print("=" * 70)
    print(f"  🏷️   RISK CLASSIFICATION SUMMARY ({dataset_label} Set)")
    print("=" * 70)

    total = len(risk_df)

    print()
    print("  ┌────────────────────┬──────────┬────────────┬───────────────┐")
    print("  │ Risk Category      │  Count   │ Percentage │ Visual        │")
    print("  ├────────────────────┼──────────┼────────────┼───────────────┤")

    for _, _, label in RISK_CATEGORIES:
        count = (risk_df['Risk_Category'] == label).sum()
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  │ {label:<18} │ {count:>6}   │ {pct:>8.1f}%  │ {bar:<13} │")

    print("  └────────────────────┴──────────┴────────────┴───────────────┘")

    # Statistics
    print()
    print(f"  📊  Risk Score Statistics:")
    print(f"       Mean:   {risk_df['Risk_Score'].mean():.2f}")
    print(f"       Median: {risk_df['Risk_Score'].median():.2f}")
    print(f"       Std:    {risk_df['Risk_Score'].std():.2f}")
    print(f"       Min:    {risk_df['Risk_Score'].min():.2f}")
    print(f"       Max:    {risk_df['Risk_Score'].max():.2f}")

    # Show some extreme cases
    print()
    print(f"  🔴  Top 5 Highest Risk Companies:")
    top_risk = risk_df.nlargest(5, 'Risk_Score')
    for _, row in top_risk.iterrows():
        print(f"       Sample #{row['Sample_Index']:<6} → "
              f"Score: {row['Risk_Score']:>6.2f}%  [{row['Risk_Category']}]")

    print()
    print(f"  🟢  Top 5 Safest Companies:")
    low_risk = risk_df.nsmallest(5, 'Risk_Score')
    for _, row in low_risk.iterrows():
        print(f"       Sample #{row['Sample_Index']:<6} → "
              f"Score: {row['Risk_Score']:>6.2f}%  [{row['Risk_Category']}]")

    print()
    return risk_df
