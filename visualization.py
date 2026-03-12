"""
=============================================================================
  VISUALIZATION — Bankruptcy Prediction System
=============================================================================
  Publication-quality visualizations for model analysis and reporting.
  All plots are saved to the outputs/ directory.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from config import OUTPUT_DIR, FIGURE_DPI, TOP_N_FEATURES, RISK_CATEGORIES


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
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# Color palette
COLORS = {
    'primary': '#58a6ff',
    'secondary': '#f778ba',
    'success': '#3fb950',
    'warning': '#d29922',
    'danger': '#f85149',
    'info': '#79c0ff',
    'gradient_start': '#58a6ff',
    'gradient_end': '#f778ba',
}

RISK_COLORS = ['#3fb950', '#58a6ff', '#d29922', '#f0883e', '#f85149']


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_roc_curve(results, save=True):
    """
    Plot ROC-AUC curve for both training and test sets.
    """
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Training ROC
    ax.plot(results['train_fpr'], results['train_tpr'],
            color=COLORS['primary'], linewidth=2.5, alpha=0.9,
            label=f"Training (AUC = {results['train_roc_auc']:.4f})")

    # Test ROC
    ax.plot(results['test_fpr'], results['test_tpr'],
            color=COLORS['secondary'], linewidth=2.5, alpha=0.9,
            label=f"Test (AUC = {results['test_roc_auc']:.4f})")

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], color='#484f58', linewidth=1.5,
            linestyle='--', alpha=0.7, label='Random (AUC = 0.5000)')

    # Fill under curves
    ax.fill_between(results['train_fpr'], results['train_tpr'],
                    alpha=0.08, color=COLORS['primary'])
    ax.fill_between(results['test_fpr'], results['test_tpr'],
                    alpha=0.08, color=COLORS['secondary'])

    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=13)
    ax.set_title('ROC-AUC Curve — Bankruptcy Prediction Model',
                 fontweight='bold', fontsize=16, pad=20)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.3,
              edgecolor='#30363d', facecolor='#161b22')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, 'roc_auc_curve.png')
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  💾  Saved: {path}")
    plt.close(fig)


def plot_precision_recall_curve(results, save=True):
    """
    Plot Precision-Recall curve for both sets.
    """
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(results['train_pr_recall'], results['train_pr_precision'],
            color=COLORS['primary'], linewidth=2.5, alpha=0.9,
            label=f"Training (AP = {results['train_avg_precision']:.4f})")

    ax.plot(results['test_pr_recall'], results['test_pr_precision'],
            color=COLORS['secondary'], linewidth=2.5, alpha=0.9,
            label=f"Test (AP = {results['test_avg_precision']:.4f})")

    ax.fill_between(results['train_pr_recall'], results['train_pr_precision'],
                    alpha=0.08, color=COLORS['primary'])
    ax.fill_between(results['test_pr_recall'], results['test_pr_precision'],
                    alpha=0.08, color=COLORS['secondary'])

    ax.set_xlabel('Recall', fontweight='bold', fontsize=13)
    ax.set_ylabel('Precision', fontweight='bold', fontsize=13)
    ax.set_title('Precision-Recall Curve — Bankruptcy Prediction Model',
                 fontweight='bold', fontsize=16, pad=20)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.3,
              edgecolor='#30363d', facecolor='#161b22')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, 'precision_recall_curve.png')
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  💾  Saved: {path}")
    plt.close(fig)


def plot_confusion_matrix(results, save=True):
    """
    Plot styled confusion matrix heatmap.
    """
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(8, 7))

    cm = results['confusion_matrix']
    labels = ['Not Bankrupt', 'Bankrupt']

    # Normalize for color mapping
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=2, linecolor='#30363d',
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 18, 'weight': 'bold'},
                ax=ax)

    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=13)
    ax.set_ylabel('True Label', fontweight='bold', fontsize=13)
    ax.set_title('Confusion Matrix — Test Set',
                 fontweight='bold', fontsize=16, pad=20)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  💾  Saved: {path}")
    plt.close(fig)


def plot_feature_importance(importance_df, save=True):
    """
    Plot top N feature importances as a horizontal bar chart.
    """
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(12, 10))

    top = importance_df.head(TOP_N_FEATURES).sort_values('Importance')

    # Gradient colors
    n = len(top)
    colors = plt.cm.cool(np.linspace(0.2, 0.9, n))

    bars = ax.barh(range(n), top['Importance'], color=colors,
                   edgecolor='#30363d', linewidth=0.5, height=0.7)

    ax.set_yticks(range(n))
    ax.set_yticklabels([name[:40] for name in top['Feature']], fontsize=10)
    ax.set_xlabel('Importance Score', fontweight='bold', fontsize=13)
    ax.set_title(f'Top {TOP_N_FEATURES} Feature Importances — XGBoost',
                 fontweight='bold', fontsize=16, pad=20)

    # Add value labels on bars
    for bar, val in zip(bars, top['Importance']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, color='#c9d1d9')

    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim(0, top['Importance'].max() * 1.15)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  💾  Saved: {path}")
    plt.close(fig)


def plot_correlation_heatmap(corr_matrix, save=True):
    """
    Plot feature correlation heatmap (top correlated features).
    """
    ensure_output_dir()

    # Select top 20 features by absolute correlation variance
    top_features = corr_matrix.abs().mean().nlargest(21).index
    subset = corr_matrix.loc[top_features, top_features]

    fig, ax = plt.subplots(figsize=(16, 14))

    mask = np.triu(np.ones_like(subset, dtype=bool), k=1)

    sns.heatmap(subset, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0,
                linewidths=0.5, linecolor='#21262d',
                annot_kws={'size': 7},
                cbar_kws={'label': 'Correlation Coefficient',
                          'shrink': 0.8},
                vmin=-1, vmax=1,
                ax=ax)

    short_labels = [name[:25] for name in top_features]
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(short_labels, rotation=0, fontsize=8)

    ax.set_title('Feature Correlation Heatmap (Top 20 Features)',
                 fontweight='bold', fontsize=16, pad=20)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  💾  Saved: {path}")
    plt.close(fig)


def plot_risk_distribution(risk_df, save=True):
    """
    Plot risk category distribution as a styled pie chart and bar chart.
    """
    ensure_output_dir()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Count per category
    category_labels = [label for _, _, label in RISK_CATEGORIES]
    counts = [
        (risk_df['Risk_Category'] == label).sum()
        for label in category_labels
    ]

    # ── Pie Chart ──
    wedges, texts, autotexts = ax1.pie(
        counts, labels=None, colors=RISK_COLORS,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '',
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor='#0d1117', linewidth=2)
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    ax1.set_title('Risk Category Distribution',
                  fontweight='bold', fontsize=14, pad=20)

    # Legend
    legend_labels = [f'{label} ({count:,})' for label, count
                     in zip(category_labels, counts)]
    ax1.legend(legend_labels, loc='center left', bbox_to_anchor=(-0.1, 0.5),
               fontsize=9, framealpha=0.3, edgecolor='#30363d',
               facecolor='#161b22')

    # ── Bar Chart ──
    bars = ax2.bar(range(len(counts)), counts, color=RISK_COLORS,
                   edgecolor='#30363d', linewidth=1, width=0.6)

    ax2.set_xticks(range(len(category_labels)))
    short_labels = [l.split(' ', 1)[1] if ' ' in l else l
                    for l in category_labels]
    ax2.set_xticklabels(short_labels, rotation=15, ha='right', fontsize=10)
    ax2.set_ylabel('Number of Companies', fontweight='bold', fontsize=12)
    ax2.set_title('Companies per Risk Category',
                  fontweight='bold', fontsize=14, pad=20)

    for bar, count in zip(bars, counts):
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{count:,}', ha='center', va='bottom',
                     fontweight='bold', fontsize=11, color='#c9d1d9')

    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, 'risk_distribution.png')
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  💾  Saved: {path}")
    plt.close(fig)


def plot_probability_distribution(probabilities, save=True):
    """
    Plot the distribution of bankruptcy probabilities (risk scores).
    """
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(12, 7))

    risk_scores = probabilities * 100

    # Histogram
    n, bins, patches = ax.hist(risk_scores, bins=50, edgecolor='#30363d',
                                linewidth=0.5, alpha=0.85)

    # Color the bars by risk category
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 20:
            patch.set_facecolor(RISK_COLORS[0])
        elif left_edge < 40:
            patch.set_facecolor(RISK_COLORS[1])
        elif left_edge < 60:
            patch.set_facecolor(RISK_COLORS[2])
        elif left_edge < 80:
            patch.set_facecolor(RISK_COLORS[3])
        else:
            patch.set_facecolor(RISK_COLORS[4])

    # Add vertical lines for thresholds
    for threshold in [20, 40, 60, 80]:
        ax.axvline(x=threshold, color='#c9d1d9', linewidth=1,
                   linestyle='--', alpha=0.5)

    # Add zone labels
    zone_labels = ['Safe', 'Low Risk', 'Moderate', 'High Risk', 'Critical']
    zone_positions = [10, 30, 50, 70, 90]
    max_count = max(n) if len(n) > 0 else 1
    for pos, label in zip(zone_positions, zone_labels):
        ax.text(pos, max_count * 0.95, label, ha='center', va='top',
                fontsize=9, fontweight='bold', color='#c9d1d9', alpha=0.7)

    ax.set_xlabel('Bankruptcy Risk Score (%)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Number of Companies', fontweight='bold', fontsize=13)
    ax.set_title('Distribution of Bankruptcy Risk Scores',
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xlim(-2, 102)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, 'risk_score_distribution.png')
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  💾  Saved: {path}")
    plt.close(fig)


def plot_metrics_comparison(results, save=True):
    """
    Plot a comparison of Train vs Test metrics as a grouped bar chart.
    """
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(12, 7))

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    train_vals = [
        results['train_accuracy'], results['train_precision'],
        results['train_recall'], results['train_f1'],
        results['train_roc_auc']
    ]
    test_vals = [
        results['test_accuracy'], results['test_precision'],
        results['test_recall'], results['test_f1'],
        results['test_roc_auc']
    ]

    x = np.arange(len(metrics))
    width = 0.3

    bars1 = ax.bar(x - width/2, train_vals, width, label='Training',
                   color=COLORS['primary'], edgecolor='#30363d',
                   linewidth=0.5, alpha=0.9)
    bars2 = ax.bar(x + width/2, test_vals, width, label='Test',
                   color=COLORS['secondary'], edgecolor='#30363d',
                   linewidth=0.5, alpha=0.9)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=COLORS['primary'])
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=COLORS['secondary'])

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=13)
    ax.set_title('Model Performance — Training vs Test',
                 fontweight='bold', fontsize=16, pad=20)
    ax.legend(fontsize=12, framealpha=0.3, edgecolor='#30363d',
              facecolor='#161b22')
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, 'metrics_comparison.png')
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  💾  Saved: {path}")
    plt.close(fig)


def generate_all_plots(results, importance_df, corr_matrix, risk_df):
    """
    Generate all visualization plots.
    """
    print("=" * 70)
    print("  🎨  GENERATING VISUALIZATIONS")
    print("=" * 70)
    print()

    plot_roc_curve(results)
    plot_precision_recall_curve(results)
    plot_confusion_matrix(results)
    plot_feature_importance(importance_df)
    plot_correlation_heatmap(corr_matrix)
    plot_risk_distribution(risk_df)
    plot_probability_distribution(results['test_probabilities'])
    plot_metrics_comparison(results)

    print()
    print(f"  ✅  All {8} visualizations saved to: {OUTPUT_DIR}")
    print()
