"""
=============================================================================
  SHAP ANALYZER — Bankruptcy Prediction System
=============================================================================
  Interpretable AI module using SHAP (Shapley Additive Explanations)
  to explain XGBoost model predictions.

  Generates:
    1. Global SHAP summary plot (beeswarm)
    2. SHAP feature importance bar plot (mean |SHAP|)
    3. Local waterfall explanation for a single company
    4. Dependence plots for key financial indicators
    5. Feature contribution CSV table
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from config import OUTPUT_DIR, FIGURE_DPI, TOP_N_FEATURES


# ──────────────────────────────────────────────────────────────────────────────
#  STYLING (match project theme)
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
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE: Compute SHAP Values
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_values(model, X_test_scaled, feature_names):
    """
    Initialize SHAP TreeExplainer and compute SHAP values for the test set.

    Parameters
    ----------
    model : XGBClassifier
        Trained XGBoost model.
    X_test_scaled : array-like
        Scaled test features.
    feature_names : list
        Feature names.

    Returns
    -------
    explainer : shap.TreeExplainer
    shap_values : shap.Explanation
    """
    print("=" * 70)
    print("  🔬  SHAP ANALYSIS — Computing Shapley Values")
    print("=" * 70)
    print()
    print("  ⏳  Initializing TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    print("  ⏳  Computing SHAP values for test dataset...")

    # Create a DataFrame with feature names for cleaner SHAP output
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    shap_values = explainer(X_test_df)

    print(f"  ✅  SHAP values computed for {X_test_df.shape[0]:,} samples "
          f"× {X_test_df.shape[1]} features")
    print()

    return explainer, shap_values


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 1: Global SHAP Summary Plot (Beeswarm)
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap_summary(shap_values, feature_names, save=True):
    """
    Generate a SHAP summary plot (beeswarm) showing the top 20 features
    and how their values impact the bankruptcy prediction.

    Each dot = one sample. Color = feature value (red=high, blue=low).
    Horizontal position = SHAP value (positive = pushes toward bankruptcy).
    """
    ensure_output_dir()

    print("  📊  Generating SHAP Summary Plot (Beeswarm)...")

    fig, ax = plt.subplots(figsize=(14, 10))

    shap.summary_plot(
        shap_values,
        max_display=TOP_N_FEATURES,
        show=False,
        plot_size=None,
    )

    # Style the current figure
    current_fig = plt.gcf()
    current_fig.set_facecolor('#0d1117')
    for a in current_fig.get_axes():
        a.set_facecolor('#161b22')
        a.set_title('SHAP Summary — Top 20 Features Driving Bankruptcy Risk',
                     fontweight='bold', fontsize=15, pad=20, color='#c9d1d9')
        a.set_xlabel('SHAP Value (Impact on Bankruptcy Prediction)',
                      fontweight='bold', fontsize=12, color='#c9d1d9')
        a.tick_params(colors='#8b949e')
        for spine in a.spines.values():
            spine.set_edgecolor('#30363d')

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, 'shap_summary_plot.png')
        current_fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight',
                            facecolor='#0d1117')
        print(f"  💾  Saved: {path}")
    plt.close('all')


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 2: SHAP Feature Importance Bar Plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap_feature_importance(shap_values, feature_names, save=True):
    """
    Create a bar chart ranking features based on mean absolute SHAP values.
    Shows which features have the greatest overall impact on predictions.
    """
    ensure_output_dir()

    print("  📊  Generating SHAP Feature Importance Bar Plot...")

    fig, ax = plt.subplots(figsize=(14, 10))

    shap.plots.bar(
        shap_values,
        max_display=TOP_N_FEATURES,
        show=False,
    )

    current_fig = plt.gcf()
    current_fig.set_facecolor('#0d1117')
    for a in current_fig.get_axes():
        a.set_facecolor('#161b22')
        a.set_title('SHAP Feature Importance — Mean |SHAP| Values',
                     fontweight='bold', fontsize=15, pad=20, color='#c9d1d9')
        a.set_xlabel('Mean |SHAP Value|',
                      fontweight='bold', fontsize=12, color='#c9d1d9')
        a.tick_params(colors='#8b949e')
        for spine in a.spines.values():
            spine.set_edgecolor('#30363d')

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, 'shap_feature_importance.png')
        current_fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight',
                            facecolor='#0d1117')
        print(f"  💾  Saved: {path}")
    plt.close('all')


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 3: Local Waterfall Explanation (Single Company)
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap_waterfall(shap_values, sample_index=0, save=True):
    """
    Generate a SHAP waterfall plot for a single company showing how
    each feature contributed to the predicted bankruptcy probability.

    Parameters
    ----------
    shap_values : shap.Explanation
    sample_index : int
        Index of the sample to explain.
    """
    ensure_output_dir()

    print(f"  📊  Generating SHAP Waterfall Plot (Sample #{sample_index})...")

    fig, ax = plt.subplots(figsize=(14, 10))

    shap.plots.waterfall(
        shap_values[sample_index],
        max_display=TOP_N_FEATURES,
        show=False,
    )

    current_fig = plt.gcf()
    current_fig.set_facecolor('#0d1117')
    for a in current_fig.get_axes():
        a.set_facecolor('#161b22')
        a.set_title(f'SHAP Waterfall — Feature Contributions for Sample #{sample_index}',
                     fontweight='bold', fontsize=14, pad=20, color='#c9d1d9')
        a.tick_params(colors='#8b949e')
        for spine in a.spines.values():
            spine.set_edgecolor('#30363d')

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, 'shap_waterfall_example.png')
        current_fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight',
                            facecolor='#0d1117')
        print(f"  💾  Saved: {path}")
    plt.close('all')


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 4: SHAP Dependence Plots for Key Financial Indicators
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap_dependence(shap_values, feature_names, save=True):
    """
    Create SHAP dependence plots for key financial indicators.
    Shows how variations in individual features influence bankruptcy risk.

    Target features:
      - Debt ratio %
      - ROA(A) before interest and % after tax
      - Cash Flow Per Share
      - Current Ratio
    """
    ensure_output_dir()

    print("  📊  Generating SHAP Dependence Plots...")

    # Target features for dependence analysis
    target_features = [
        "Debt ratio %",
        "ROA(A) before interest and % after tax",
        "Cash Flow Per Share",
        "Current Ratio",
    ]

    # Find which target features exist in our feature list
    available_features = []
    for feat in target_features:
        if feat in feature_names:
            available_features.append(feat)

    # If some are missing, add top important features as fallback
    if len(available_features) < 4:
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        for idx in sorted_indices:
            if feature_names[idx] not in available_features:
                available_features.append(feature_names[idx])
            if len(available_features) >= 4:
                break

    n_plots = len(available_features)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for i, feat in enumerate(available_features[:4]):
        feat_idx = feature_names.index(feat)

        ax = axes[i]
        scatter = ax.scatter(
            shap_values.data[:, feat_idx],
            shap_values.values[:, feat_idx],
            c=shap_values.values[:, feat_idx],
            cmap='coolwarm',
            alpha=0.5,
            s=8,
            edgecolors='none',
        )

        ax.axhline(y=0, color='#484f58', linewidth=1, linestyle='--', alpha=0.7)
        ax.set_xlabel(feat[:35], fontweight='bold', fontsize=10, color='#c9d1d9')
        ax.set_ylabel('SHAP Value', fontweight='bold', fontsize=10, color='#c9d1d9')
        ax.set_title(f'Dependence: {feat[:35]}',
                     fontweight='bold', fontsize=12, pad=10, color='#c9d1d9')
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

        plt.colorbar(scatter, ax=ax, label='SHAP Value', shrink=0.8)

    fig.suptitle('SHAP Dependence Plots — Key Financial Indicators',
                 fontweight='bold', fontsize=16, color='#c9d1d9', y=1.02)
    fig.set_facecolor('#0d1117')

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, 'shap_dependence_plots.png')
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight',
                    facecolor='#0d1117')
        print(f"  💾  Saved: {path}")
    plt.close('all')


# ══════════════════════════════════════════════════════════════════════════════
#  TABLE: Feature Contribution Table (CSV)
# ══════════════════════════════════════════════════════════════════════════════

def compute_feature_contributions(shap_values, feature_names, save=True):
    """
    Calculate mean absolute SHAP value for each feature and create
    a ranked table of the top 20 features.

    Returns
    -------
    contributions_df : pd.DataFrame
        Ranked table with feature names, mean |SHAP|, direction of effect.
    """
    print("  📊  Computing SHAP Feature Contribution Table...")

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    mean_shap = shap_values.values.mean(axis=0)

    contributions_df = pd.DataFrame({
        'Rank': range(1, len(feature_names) + 1),
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap,
        'Mean_SHAP': mean_shap,
        'Direction': ['Increases Risk' if m > 0 else 'Decreases Risk'
                      for m in mean_shap],
    }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

    # Reassign ranks after sorting
    contributions_df['Rank'] = range(1, len(contributions_df) + 1)

    # Print top 20
    print()
    print("  ┌──────┬──────────────────────────────────────────────┬────────────┬──────────────────┐")
    print("  │ Rank │ Feature                                      │ Mean|SHAP| │ Effect Direction │")
    print("  ├──────┼──────────────────────────────────────────────┼────────────┼──────────────────┤")
    for _, row in contributions_df.head(20).iterrows():
        print(f"  │ {row['Rank']:>4} │ {row['Feature'][:44]:<44} │ {row['Mean_Abs_SHAP']:>10.6f} │ {row['Direction']:<16} │")
    print("  └──────┴──────────────────────────────────────────────┴────────────┴──────────────────┘")
    print()

    # Insights
    risk_increasers = contributions_df[contributions_df['Direction'] == 'Increases Risk'].head(5)
    risk_decreasers = contributions_df[contributions_df['Direction'] == 'Decreases Risk'].head(5)

    print("  🔴  Top 5 Features that INCREASE Bankruptcy Risk:")
    for _, row in risk_increasers.iterrows():
        print(f"       #{row['Rank']:<3} {row['Feature'][:45]:<45} (SHAP: {row['Mean_Abs_SHAP']:.6f})")

    print()
    print("  🟢  Top 5 Features that DECREASE Bankruptcy Risk:")
    for _, row in risk_decreasers.iterrows():
        print(f"       #{row['Rank']:<3} {row['Feature'][:45]:<45} (SHAP: {row['Mean_Abs_SHAP']:.6f})")
    print()

    if save:
        ensure_output_dir()
        path = os.path.join(OUTPUT_DIR, 'shap_feature_contributions.csv')
        contributions_df.to_csv(path, index=False)
        print(f"  💾  Saved: {path}")

    return contributions_df


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER FUNCTION: Run All SHAP Analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_shap_analysis(model, X_test_scaled, feature_names, y_test=None):
    """
    Execute the complete SHAP analysis pipeline.

    Parameters
    ----------
    model : XGBClassifier
        Trained model.
    X_test_scaled : array-like
        Scaled test features.
    feature_names : list
        Feature names.
    y_test : array-like, optional
        True labels (used to select an interesting sample for waterfall).

    Returns
    -------
    shap_values : shap.Explanation
    contributions_df : pd.DataFrame
    """
    print()
    print("+" + "-" * 68 + "+")
    print("|" + " " * 68 + "|")
    print("|   SHAP EXPLAINABILITY ANALYSIS                                    |")
    print("|   Interpreting XGBoost Bankruptcy Predictions                      |")
    print("|" + " " * 68 + "|")
    print("+" + "-" * 68 + "+")
    print()

    # Step 1: Compute SHAP values
    explainer, shap_values = compute_shap_values(
        model, X_test_scaled, feature_names
    )

    # Step 2: Global summary plot (beeswarm)
    plot_shap_summary(shap_values, feature_names)

    # Step 3: Feature importance bar plot
    plot_shap_feature_importance(shap_values, feature_names)

    # Step 4: Waterfall for a single company
    # Select an interesting sample — preferably one predicted as bankrupt
    if y_test is not None:
        proba = model.predict_proba(X_test_scaled)[:, 1]
        # Find a sample with high bankruptcy probability for a compelling waterfall
        high_risk_indices = np.where(proba > 0.5)[0]
        if len(high_risk_indices) > 0:
            sample_idx = high_risk_indices[0]
        else:
            sample_idx = 0
    else:
        sample_idx = 0

    plot_shap_waterfall(shap_values, sample_index=sample_idx)

    # Step 5: Dependence plots for key indicators
    plot_shap_dependence(shap_values, feature_names)

    # Step 6: Feature contribution table
    contributions_df = compute_feature_contributions(shap_values, feature_names)

    print()
    print("=" * 70)
    print("  ✅  SHAP ANALYSIS COMPLETE — All explainability outputs saved")
    print("=" * 70)
    print()

    return shap_values, contributions_df
