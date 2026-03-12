from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np

from backend.model_loader import get_model, get_scaler, get_feature_metadata
from risk_classifier import classify_risk

router = APIRouter()

class PredictionRequest(BaseModel):
    features: List[float]

@router.get("/metadata")
async def get_metadata():
    return get_feature_metadata()

@router.post("/predict")
async def predict_bankruptcy(request: PredictionRequest):
    if len(request.features) != 48:
        raise HTTPException(status_code=400, detail="Expected exactly 48 features")
        
    model = get_model()
    scaler = get_scaler()
    
    # Convert to numpy array
    input_data = np.array(request.features).reshape(1, -1)
    
    # Scale if scaler exists
    if scaler is not None:
        input_data = scaler.transform(input_data)
        
    # Predict probability (class 1)
    probability = float(model.predict_proba(input_data)[0, 1])
    
    # Get risk classification
    risk_score, risk_category = classify_risk(probability)
    
    return {
        "bankruptcy_probability": probability,
        "risk_score": risk_score,
        "risk_category": risk_category
    }
