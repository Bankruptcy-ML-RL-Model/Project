from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from backend.agent_controller import FinancialRiskAgent

router = APIRouter()

class AdvisorRequest(BaseModel):
    features: List[float]

agent = FinancialRiskAgent()

@router.post("/advisor")
async def generate_advisory_report(request: AdvisorRequest):
    if len(request.features) != 48:
        raise HTTPException(status_code=400, detail="Expected exactly 48 features")
        
    report = await agent.analyze_company(request.features)
    
    if "error" in report:
        raise HTTPException(status_code=500, detail=report["error"])
        
    return report
