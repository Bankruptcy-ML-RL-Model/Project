"""
=============================================================================
  RISK SIMULATOR — Bankruptcy Prediction System
=============================================================================
  Provides a prediction interface for external systems (e.g. RL agents)
  to interact with the trained bankruptcy model.

  Usage:
      from risk_simulator import predict_bankruptcy_risk
      import numpy as np

      state = np.random.rand(48)  # 48 financial features
      probability = predict_bankruptcy_risk(state)
=============================================================================
"""

import os
import numpy as np
import joblib
from xgboost import XGBClassifier

from config import MODEL_DIR, USE_FEATURE_SCALING


# Cache loaded model and scaler
_cached_model = None
_cached_scaler = None


def _load_model():
    """Load the trained model and scaler from disk (cached)."""
    global _cached_model, _cached_scaler

    if _cached_model is not None:
        return _cached_model, _cached_scaler

    model_path = os.path.join(MODEL_DIR, 'xgboost_bankruptcy_model.json')
    scaler_path = os.path.join(MODEL_DIR, 'feature_scaler.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run main.py first to train and save the model."
        )

    model = XGBClassifier()
    model.load_model(model_path)
    _cached_model = model

    if USE_FEATURE_SCALING and os.path.exists(scaler_path):
        _cached_scaler = joblib.load(scaler_path)
    else:
        _cached_scaler = None

    return _cached_model, _cached_scaler


def predict_bankruptcy_risk(financial_state_vector):
    """
    Predict bankruptcy probability for a given financial state.

    This function serves as the interface for RL agents and external
    systems to query the trained model.

    Parameters
    ----------
    financial_state_vector : array-like, shape (48,) or (n, 48)
        A single 48-feature vector or batch of vectors representing
        a company's financial state.

    Returns
    -------
    probability : float or np.ndarray
        Bankruptcy probability (0.0 to 1.0).
        - Single input  → returns a float
        - Batch input   → returns np.ndarray of shape (n,)

    Example
    -------
    >>> from risk_simulator import predict_bankruptcy_risk
    >>> import numpy as np
    >>> state = np.zeros(48)
    >>> prob = predict_bankruptcy_risk(state)
    >>> print(f"Bankruptcy probability: {prob:.4f}")
    >>> risk_score = prob * 100  # Convert to 0-100 risk scale
    """
    model, scaler = _load_model()

    # Ensure 2D input
    X = np.array(financial_state_vector)
    if X.ndim == 1:
        X = X.reshape(1, -1)
        single_input = True
    else:
        single_input = False

    if X.shape[1] != 48:
        raise ValueError(
            f"Expected 48 features, got {X.shape[1]}. "
            "The model requires exactly 48 financial indicators."
        )

    # Apply scaling if enabled
    if scaler is not None:
        X = scaler.transform(X)

    # Predict
    probabilities = model.predict_proba(X)[:, 1]

    if single_input:
        return float(probabilities[0])
    return probabilities


def predict_risk_category(financial_state_vector):
    """
    Predict the risk category for a given financial state.

    Returns
    -------
    result : dict
        Contains 'probability', 'risk_score', 'risk_category'.
    """
    from config import RISK_CATEGORIES

    probability = predict_bankruptcy_risk(financial_state_vector)
    risk_score = probability * 100

    category = "Unknown"
    for low, high, label in RISK_CATEGORIES:
        if low <= risk_score < high or (high == 100 and risk_score == 100):
            category = label
            break

    return {
        'probability': probability,
        'risk_score': round(risk_score, 4),
        'risk_category': category,
    }
