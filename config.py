"""
=============================================================================
  CONFIGURATION — Bankruptcy Prediction System
=============================================================================
  Centralized configuration for paths, model parameters, risk categories,
  reproducibility seeds, and pipeline toggles.
=============================================================================
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "corporate_bankruptcy_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
DIAGNOSTICS_DIR = os.path.join(OUTPUT_DIR, "model_diagnostics")

# ──────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
RANDOM_STATE = RANDOM_SEED  # backward compatibility alias

# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────
TARGET_COLUMN = "Bankrupt?"
TEST_SIZE = 0.20

# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE TOGGLES
# ──────────────────────────────────────────────────────────────────────────────
USE_FEATURE_SCALING = False           # XGBoost is tree-based; scaling not required
ENABLE_PROBABILITY_CALIBRATION = True # Platt scaling for better probability estimates
CV_FOLDS = 5                         # Number of StratifiedKFold cross-validation folds
ENABLE_RL_SIMULATION = True           # Train RL agent after SHAP & diagnostics
RL_TIMESTEPS = 50000                  # PPO training timesteps

# ──────────────────────────────────────────────────────────────────────────────
# XGBOOST HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
    "use_label_encoder": False,
}

# ──────────────────────────────────────────────────────────────────────────────
# RISK CLASSIFICATION THRESHOLDS
# Probability × 100 → Risk Category
# ──────────────────────────────────────────────────────────────────────────────
RISK_CATEGORIES = [
    (0, 20, "🟢 Safe"),
    (20, 40, "🔵 Low Risk"),
    (40, 60, "🟡 Moderate Risk"),
    (60, 80, "🟠 High Risk"),
    (80, 100, "🔴 Critical Risk"),
]

# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────
FIGURE_DPI = 150
TOP_N_FEATURES = 20  # Number of top features to show in importance plot
