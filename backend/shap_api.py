from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import shap

from backend.model_loader import get_model, get_scaler, get_feature_names

router = APIRouter()

class ShapRequest(BaseModel):
    features: List[float]

@router.post("/explain")
async def explain_prediction(request: ShapRequest):
    if len(request.features) != 48:
        raise HTTPException(status_code=400, detail="Expected exactly 48 features")
        
    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()
    
    input_data = np.array(request.features).reshape(1, -1)
    if scaler is not None:
        input_data = scaler.transform(input_data)
        
    # Compute SHAP values for this single instance
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)[0] # get first instance
    
    # Pair feature names with their shap values
    contributions = []
    for i, name in enumerate(feature_names):
        val = float(shap_values[i])
        contributions.append({
            "feature": name,
            "value": val,
            "abs_value": abs(val)
        })
        
    # Sort by absolute impact
    contributions.sort(key=lambda x: x["abs_value"], reverse=True)
    
    # Separate into risk (positive SHAP) and protective (negative SHAP)
    top_risk_factors = [c for c in contributions if c["value"] > 0][:5]
    top_protective_factors = [c for c in contributions if c["value"] < 0][:5]
    
    return {
        "top_risk_factors": top_risk_factors,
        "top_protective_factors": top_protective_factors,
        "all_contributions": contributions[:15] # Top 15 overall
    }
