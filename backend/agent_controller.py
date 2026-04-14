import os
import json
import re
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from backend.prediction_api import predict_bankruptcy, PredictionRequest
from backend.shap_api import explain_prediction, ShapRequest
from backend.rl_api import generate_strategy, StratRequest

@tool
async def predict_bankruptcy_risk(financial_vector: List[float]) -> dict:
    """
    Analyzes a company's financial indicators and predicts the bankruptcy risk score and category.
    Takes a list of 48 financial features as input.
    Returns: bankruptcy_probability, risk_score, and risk_category.
    """
    res = await predict_bankruptcy(PredictionRequest(features=financial_vector))
    return res

@tool
async def generate_shap_explanation(financial_vector: List[float]) -> dict:
    """
    Generates a SHAP explanation to find the main financial drivers of risk.
    Takes a list of 48 financial features as input.
    Returns: top_risk_factors and top_protective_factors.
    """
    res = await explain_prediction(ShapRequest(features=financial_vector))
    return res

@tool
async def run_rl_strategy(financial_vector: List[float]) -> dict:
    """
    Runs a Reinforcement Learning strategy simulator to generate a recovery plan.
    Takes a list of 48 financial features as input.
    Returns: initial_risk, final_risk, steps, and history.
    """
    res = await generate_strategy(StratRequest(features=financial_vector))
    return res

SYSTEM_PROMPT = '''You are an autonomous Financial Risk Advisory Agent. Your goal is to take a company's financial data and produce a complete financial risk advisory report in structured JSON format.

You MUST follow this exact reasoning flow:
Step 1: Call `predict_bankruptcy_risk` using the provided financial vector.
Step 2: If the returned `risk_score` < 20, you do not need to call any other tools. Simply return a report saying the company is financially healthy.
Step 3: If the `risk_score` >= 20, you MUST call the `generate_shap_explanation` tool to find the main financial drivers.
Step 4: If the `risk_score` >= 40, you MUST also call the `run_rl_strategy` tool to formulate a recovery plan.
Step 5: Generate a final advisory report combining all insights.

You must ALWAYS output your final response as valid JSON matching this exact structure:
{{
  "company_risk_assessment": {{
    "risk_score": <int or float>,
    "risk_category": "<string>",
    "bankruptcy_probability": <float>
  }},
  "risk_drivers": [
    "<detailed, descriptive, paragraph-length analysis of risk driver 1>",
    "<detailed, descriptive, paragraph-length analysis of risk driver 2>"
  ],
  "recommended_strategy": [
    "<Phase 1 (Timeline): detailed, descriptive, paragraph-length strategic recovery action with clear immediate recommendations>",
    "<Phase 2 (Timeline): detailed, descriptive, paragraph-length strategic recovery action focusing on medium-term stability>",
    "<Phase 3 (Timeline): detailed, descriptive, paragraph-length strategic recovery action focusing on long-term growth>"
  ],
  "projected_risk_after_strategy": <float or null>
}}

CRITICAL INSTRUCTION: For `risk_drivers` and `recommended_strategy`, DO NOT just output short 4-5 word bullet points. You MUST write in a highly descriptive, analytical, and detailed manner. Each item MUST be a full, robust paragraph explaining the 'why', the 'how', and the expected impact.
For `recommended_strategy`, you MUST explicitly structure it as a chronological, phase-wise improvement plan (e.g., Phase 1: 0-3 Months, Phase 2: 3-6 Months). Provide deep, actionable recommendations for each phase.

Note: If risk is < 20, `risk_drivers` and `recommended_strategy` can be empty arrays or contain a note about being healthy.
If `run_rl_strategy` was called, summarize the steps returned in `recommended_strategy` and provide `final_risk` in `projected_risk_after_strategy`.
'''

class FinancialRiskAgent:
    def __init__(self):
        # Fetch API key securely from environment variables
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("WARNING: GROQ_API_KEY not found in environment variables.")
        self.llm = ChatOpenAI(base_url="https://api.groq.com/openai/v1", model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)
        self.tools = [predict_bankruptcy_risk, generate_shap_explanation, run_rl_strategy]
        self.agent = create_react_agent(self.llm, self.tools, prompt=SYSTEM_PROMPT)
        
    async def analyze_company(self, financial_vector: List[float]) -> dict:
        if os.environ.get("GROQ_API_KEY") is None or os.environ.get("GROQ_API_KEY") == "dummy_key":
            return {
                "error": "GROQ_API_KEY is not set. Please set the environment variable to use the AI Advisor."
            }
            
        try:
            result = await self.agent.ainvoke({"messages": [("human", f"Analyze this financial vector: {financial_vector}")]})
            
            # Extract the final AI message content
            output_text = result["messages"][-1].content
            parsed = self._parse_llm_json(output_text)
            return parsed
        except Exception as e:
            return {
                "error": str(e),
                "raw_output": result["messages"][-1].content if 'result' in locals() else ""
            }

    @staticmethod
    def _parse_llm_json(text: str) -> dict:
        """Robustly parse JSON from LLM output, handling common malformations."""
        # Strip markdown code fences
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()

        # Normalize double curly braces (LLM mimics the prompt template)
        text = text.replace("{{", "{").replace("}}", "}")

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object with regex
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

            # Fix common issues: single quotes → double quotes
            fixed = candidate
            # Replace single-quoted keys/values with double-quoted ones
            fixed = re.sub(r"'([^']*?)'(\s*:)", r'"\1"\2', fixed)  # keys
            fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed)    # values
            # Remove trailing commas before } or ]
            fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        # Last resort: build a minimal valid response from the raw text
        return {
            "company_risk_assessment": {
                "risk_score": 0,
                "risk_category": "Unknown",
                "bankruptcy_probability": 0
            },
            "risk_drivers": [text[:500] if text else "Could not parse AI response"],
            "recommended_strategy": ["Please try again or use the manual analysis tools."],
            "projected_risk_after_strategy": None
        }

