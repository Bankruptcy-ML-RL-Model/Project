from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from config import BASE_DIR

from backend.prediction_api import router as prediction_router
from backend.shap_api import router as shap_router
from backend.rl_api import router as rl_router
from backend.advisor_api import router as advisor_router

# Initialize the frontend path
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = FastAPI(
    title="Corporate Bankruptcy Risk Simulator",
    description="Financial Risk Intelligence Platform API",
    version="2.0.0"
)

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routers
app.include_router(prediction_router, prefix="/api")
app.include_router(shap_router, prefix="/api")
app.include_router(rl_router, prefix="/api")
app.include_router(advisor_router, prefix="/api")

# Ensure frontend directory exists
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Mount the frontend directory to serve static files (HTML, CSS, JS)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
