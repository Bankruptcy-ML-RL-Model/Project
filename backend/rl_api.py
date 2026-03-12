from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np

from rl_strategy_optimizer import FinancialRiskEnv
from stable_baselines3 import PPO
import os
from config import BASE_DIR
from backend.model_loader import get_feature_metadata, get_model, get_scaler

router = APIRouter()

class StratRequest(BaseModel):
    features: List[float]

BUSINESS_MAPPINGS = {
    "Liability to Equity": "Reduce overall leverage by improving equity financing",
    "Debt ratio %": "Reduce total debt burden to improve solvency",
    "Borrowing dependency": "Decrease reliance on external borrowing",
    "Interest Coverage Ratio (Interest expense to EBIT)": "Improve EBIT to cover interest expenses more comfortably",
    "Operating Profit Margin": "Increase core operating profitability",
    "Net Income Flag": "Ensure positive net income generation",
    "Cash Flow Per Share": "Boost operational cash flow generation",
    "Persistent EPS in the Last Four Seasons": "Drive consistent earnings per share growth",
    "Current Ratio": "Improve short-term liquidity and working capital",
    "Quick Ratio": "Improve immediate liquidity position"
}

def get_business_action(direction, feature):
    if feature in BUSINESS_MAPPINGS:
        return BUSINESS_MAPPINGS[feature]
    if "Net Value Per Share" in feature:
        return "Enhance book value and shareholder equity"
    if "Return on" in feature or "ROA" in feature or "ROE" in feature:
        return f"Improve profitability metrics ({feature})"
    return f"{direction} {feature}"

@router.post("/strategy")
async def generate_strategy(request: StratRequest):
    if len(request.features) != 48:
        raise HTTPException(status_code=400, detail="Expected exactly 48 features")
        
    initial_state = np.array(request.features).reshape(1, -1)
    
    # Check if Risk is already very low to save computation
    model = get_model()
    scaler = get_scaler()
    scaled_state = initial_state
    if scaler is not None:
        scaled_state = scaler.transform(scaled_state)
    initial_prob = float(model.predict_proba(scaled_state)[0, 1])
    
    if initial_prob < 0.05:
         return {
            "skip_rl": True,
            "message": "This company already has very low bankruptcy risk. Strategy optimization is unnecessary."
         }
         
    feature_metadata = get_feature_metadata()
    
    # Create an environment specifically seeded with this user's state
    env = FinancialRiskEnv(sample_states=initial_state, max_steps=10, feature_metadata=feature_metadata)
    
    # Train a quick model specifically optimized for this state
    model = PPO("MlpPolicy", env, learning_rate=5e-3, n_steps=64, batch_size=64, n_epochs=10, verbose=0)
    model.learn(total_timesteps=5000) # Quick 5k steps focused on this exact company
    
    # Now run the trained strategy
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
    history = env.history
    initial_prob = history[0]['probability']
    final_prob = history[-1]['probability']
    
    steps = []
    for h in history[1:]:
        adj = h.get('adjustment', 0)
        direction = "Increase" if adj > 0 else "Reduce"
        feature = h['action']
        prob = h['probability']
        
        business_action = get_business_action(direction, feature)
        h['action_str'] = business_action
        steps.append(f"{business_action} (Risk: {prob*100:.1f}%)")
        
    # Return formatted strategy
    return {
        "skip_rl": False,
        "initial_risk": initial_prob * 100,
        "final_risk": final_prob * 100,
        "history": [
            {
                "step": h['step'],
                "probability": h['probability'] * 100,
                "action_str": h.get('action_str', "Initial State")
            } for h in history
        ],
        "steps": steps
    }
