import os
import json
import joblib
import xgboost as xgb

from config import MODEL_DIR, BASE_DIR, USE_FEATURE_SCALING

# Global caches
_model = None
_scaler = None
_feature_metadata = None

def get_model():
    """Load and cache the XGBoost model."""
    global _model
    if _model is None:
        model_path = os.path.join(MODEL_DIR, 'xgboost_bankruptcy_model.json')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        _model = xgb.XGBClassifier()
        _model.load_model(model_path)
    return _model

def get_scaler():
    """Load and cache the StandardScaler, if configured and it exists."""
    global _scaler
    if _scaler is None:
        scaler_path = os.path.join(MODEL_DIR, 'feature_scaler.pkl')
        if USE_FEATURE_SCALING and os.path.exists(scaler_path):
            _scaler = joblib.load(scaler_path)
        else:
            _scaler = False # Explicitly set to False to avoid re-checking
    return _scaler if _scaler is not False else None

def get_feature_metadata():
    """Load and cache feature metadata from JSON."""
    global _feature_metadata
    if _feature_metadata is None:
        metadata_path = os.path.join(BASE_DIR, 'feature_metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Feature metadata not found at {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            _feature_metadata = data['features']
    return _feature_metadata

def get_feature_names():
    """Get just the ordered list of feature names."""
    metadata = get_feature_metadata()
    return [f['name'] for f in metadata]
