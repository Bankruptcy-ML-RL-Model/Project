"""
=============================================================================

   ██████╗  █████╗ ███╗   ██╗██╗  ██╗██████╗ ██╗   ██╗██████╗ ████████╗ ██████╗██╗   ██╗
   ██╔══██╗██╔══██╗████╗  ██║██║ ██╔╝██╔══██╗██║   ██║██╔══██╗╚══██╔══╝██╔════╝╚██╗ ██╔╝
   ██████╔╝███████║██╔██╗ ██║█████╔╝ ██████╔╝██║   ██║██████╔╝   ██║   ██║      ╚████╔╝
   ██╔══██╗██╔══██║██║╚██╗██║██╔═██╗ ██╔══██╗██║   ██║██╔═══╝    ██║   ██║       ╚██╔╝
   ██████╔╝██║  ██║██║ ╚████║██║  ██╗██║  ██║╚██████╔╝██║        ██║   ╚██████╗   ██║
   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝        ╚═╝    ╚═════╝   ╚═╝

   ██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗
   ██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║
   ██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║   ██║██║   ██║██╔██╗ ██║
   ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║   ██║██║   ██║██║╚██╗██║
   ██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║
   ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

=============================================================================
  MAIN PIPELINE — Corporate Bankruptcy Prediction System (v2.0)
=============================================================================
  This script orchestrates the complete machine learning pipeline:

   1. Set Global Random Seeds (reproducibility)
   2. Data Loading & Exploration
   3. Feature Engineering & Analysis
   4. Train/Test Split (80/20 Stratified)
   5. Optional Preprocessing (StandardScaler)
   6. XGBoost Classifier Training (with class imbalance handling)
   7. Feature Importance Extraction
   8. Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
   9. Risk Classification (5-tier scoring system)
  10. Visualization Generation (12 publication-quality charts)
  11. SHAP Explainability Analysis
  12. Cross-Validation & Overfitting Diagnostics
  13. Optimal Threshold Selection (Youden's J)
  14. Probability Calibration (Platt Scaling)
  15. Robustness Analysis (learning curves, feature stability)
  16. Save Model & Results

  Dataset: Taiwan Economic Journal (1999–2009)
  Model:   XGBoost Gradient Boosting Classifier
  Target:  Binary — Bankrupt (1) / Not Bankrupt (0)
=============================================================================
"""

import os
import sys
import time
import random
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

# ── Project modules ──
from config import (
    DATA_PATH, OUTPUT_DIR, MODEL_DIR, DIAGNOSTICS_DIR,
    TARGET_COLUMN, TEST_SIZE, RANDOM_SEED,
    USE_FEATURE_SCALING, ENABLE_PROBABILITY_CALIBRATION,
    ENABLE_RL_SIMULATION, RL_TIMESTEPS
)
from data_loader import load_dataset, split_dataset
from feature_engineering import (
    analyze_features, get_correlation_matrix, preprocess_features
)
from model_trainer import train_xgboost, get_feature_importance
from model_evaluator import evaluate_model
from risk_classifier import classify_all_risks, print_risk_summary
from visualization import generate_all_plots
from shap_analyzer import run_shap_analysis
from model_diagnostics import run_all_diagnostics
from rl_strategy_optimizer import run_rl_simulation


def set_global_seeds(seed):
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"  🔒  Global random seeds set to {seed}")
    print(f"       numpy, random, PYTHONHASHSEED = {seed}")
    print()


def print_banner():
    """Print the project banner."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║   🏦  CORPORATE BANKRUPTCY PREDICTION SYSTEM  v2.0                ║")
    print("║   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                ║")
    print("║   Model: XGBoost Gradient Boosting Classifier                     ║")
    print("║   Split: 80% Training / 20% Test (Stratified)                     ║")
    print("║   Risk:  5-Tier Classification System                             ║")
    print("║   SHAP:  Shapley Additive Explanations                            ║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()


def print_final_summary(results, risk_df, training_time, total_time,
                        optimal_threshold=None, cv_mean_auc=None):
    """Print the final project summary."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║   ✅  PIPELINE COMPLETED SUCCESSFULLY                             ║")
    print("║" + " " * 68 + "║")
    print("╠" + "═" * 68 + "╣")
    print("║                                                                    ║")
    print("║   📊  FINAL RESULTS SUMMARY                                       ║")
    print("║   ─────────────────────────────────────────                        ║")
    print(f"║   Training Accuracy:   {results['train_accuracy']:.4f}  "
          f"({results['train_accuracy']*100:.2f}%)                       ║")
    print(f"║   Training Precision:  {results['train_precision']:.4f}  "
          f"({results['train_precision']*100:.2f}%)                       ║")
    print(f"║   Training Recall:     {results['train_recall']:.4f}  "
          f"({results['train_recall']*100:.2f}%)                       ║")
    print(f"║   Training F1 Score:   {results['train_f1']:.4f}  "
          f"({results['train_f1']*100:.2f}%)                       ║")
    print(f"║   Training ROC-AUC:    {results['train_roc_auc']:.4f}  "
          f"({results['train_roc_auc']*100:.2f}%)                       ║")
    print("║                                                                    ║")
    print(f"║   Test Accuracy:       {results['test_accuracy']:.4f}  "
          f"({results['test_accuracy']*100:.2f}%)                       ║")
    print(f"║   Test ROC-AUC:        {results['test_roc_auc']:.4f}  "
          f"({results['test_roc_auc']*100:.2f}%)                       ║")
    if cv_mean_auc is not None:
        print(f"║   CV Mean ROC-AUC:     {cv_mean_auc:.4f}  "
              f"({cv_mean_auc*100:.2f}%)                       ║")
    if optimal_threshold is not None:
        print(f"║   Optimal Threshold:   {optimal_threshold:.4f}"
              f"                                      ║")
    print("║                                                                    ║")
    print("║   ⏱️   Performance                                                 ║")
    print(f"║   Training Time:  {training_time:.2f}s                     "
          f"                    ║")
    print(f"║   Total Pipeline: {total_time:.2f}s                       "
          f"                   ║")
    print("║                                                                    ║")
    print("║   📁  Outputs saved to: outputs/                                  ║")
    print("║   💾  Model saved to:   saved_model/                              ║")
    print("║                                                                    ║")
    print("╚" + "═" * 68 + "╝")
    print()


def main():
    """
    Execute the complete bankruptcy prediction pipeline.
    """
    pipeline_start = time.time()
    print_banner()

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 1: SET GLOBAL RANDOM SEEDS
    # ══════════════════════════════════════════════════════════════════════
    set_global_seeds(RANDOM_SEED)

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2: LOAD DATASET
    # ══════════════════════════════════════════════════════════════════════
    df = load_dataset()

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 3: FEATURE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    feature_stats, target_correlations = analyze_features(df)

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 4: CORRELATION MATRIX
    # ══════════════════════════════════════════════════════════════════════
    corr_matrix = get_correlation_matrix(df)

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 5: SPLIT DATASET (80% Train / 20% Test)
    # ══════════════════════════════════════════════════════════════════════
    X_train, X_test, y_train, y_test = split_dataset(df)

    # Store feature names before any processing
    feature_names = X_train.columns.tolist()

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 6: PREPROCESS FEATURES (Optional Scaling)
    # ══════════════════════════════════════════════════════════════════════
    X_train_processed, X_test_processed, scaler = preprocess_features(
        X_train, X_test
    )

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 7: TRAIN XGBOOST MODEL
    # ══════════════════════════════════════════════════════════════════════
    model, training_time = train_xgboost(
        X_train_processed, y_train, feature_names
    )

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 8: FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════════════════
    importance_df = get_feature_importance(model, feature_names)

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 9: EVALUATE MODEL
    # ══════════════════════════════════════════════════════════════════════
    results = evaluate_model(model, X_train_processed, y_train,
                             X_test_processed, y_test)

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 10: RISK CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════════
    test_probabilities = results['test_probabilities']
    test_indices = X_test.index.tolist() if hasattr(X_test, 'index') else None

    risk_df = classify_all_risks(test_probabilities, test_indices)
    print_risk_summary(risk_df, dataset_label="Test")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 11: GENERATE VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════════
    generate_all_plots(results, importance_df, corr_matrix, risk_df)

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 12: SHAP EXPLAINABILITY ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    shap_values, shap_contributions = run_shap_analysis(
        model, X_test_processed, feature_names, y_test
    )

    # ══════════════════════════════════════════════════════════════════════
    #  STEPS 13-16: DIAGNOSTICS, CV, THRESHOLD, CALIBRATION, ROBUSTNESS
    # ══════════════════════════════════════════════════════════════════════
    from model_trainer import compute_scale_pos_weight
    scale_weight = compute_scale_pos_weight(y_train)

    cv_df, optimal_threshold, calibrated_proba = run_all_diagnostics(
        model, X_train_processed, y_train,
        X_test_processed, y_test,
        feature_names, importance_df, shap_contributions,
        scale_weight
    )

    cv_mean_auc = cv_df['Validation_ROC_AUC'].mean()

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 17: REINFORCEMENT LEARNING STRATEGY SIMULATION
    # ══════════════════════════════════════════════════════════════════════
    if ENABLE_RL_SIMULATION:
        rl_strategies = run_rl_simulation(total_timesteps=RL_TIMESTEPS)
    else:
        print("  ℹ️   RL Simulation DISABLED in config.")
        rl_strategies = None

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 18: SAVE MODEL & RESULTS
    # ══════════════════════════════════════════════════════════════════════
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save model
    model_path = os.path.join(MODEL_DIR, 'xgboost_bankruptcy_model.json')
    model.save_model(model_path)
    print(f"  💾  Model saved: {model_path}")

    # Save scaler (only if scaling was used)
    if scaler is not None:
        scaler_path = os.path.join(MODEL_DIR, 'feature_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"  💾  Scaler saved: {scaler_path}")
    else:
        print(f"  ℹ️   Scaler not saved (scaling disabled)")

    # Save risk classification results
    risk_path = os.path.join(OUTPUT_DIR, 'risk_classification_results.csv')
    risk_df.to_csv(risk_path, index=False)
    print(f"  💾  Risk results saved: {risk_path}")

    # Save feature importance
    importance_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"  💾  Feature importance saved: {importance_path}")

    # Save metrics summary
    metrics_path = os.path.join(OUTPUT_DIR, 'model_metrics_summary.csv')
    metrics_summary = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC',
                    'CV_Mean_AUC', 'Optimal_Threshold'],
        'Training': [
            results['train_accuracy'], results['train_precision'],
            results['train_recall'], results['train_f1'],
            results['train_roc_auc'], '', ''
        ],
        'Test': [
            results['test_accuracy'], results['test_precision'],
            results['test_recall'], results['test_f1'],
            results['test_roc_auc'], cv_mean_auc, optimal_threshold
        ]
    })
    metrics_summary.to_csv(metrics_path, index=False)
    print(f"  💾  Metrics summary saved: {metrics_path}")

    # ══════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    total_time = time.time() - pipeline_start
    print_final_summary(results, risk_df, training_time, total_time,
                        optimal_threshold, cv_mean_auc)


if __name__ == "__main__":
    main()
