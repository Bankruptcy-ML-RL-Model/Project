# 🏦 Corporate Bankruptcy Prediction System — v3.0 (Full-Stack Edition)

> **An end-to-end Machine Learning web application that predicts corporate bankruptcy, explains the driving risk factors using SHAP, and generates step-by-step financial recovery strategies using a Reinforcement Learning (RL) AI simulator.**

---

## 📑 Table of Contents

- [Project Overview & Key Features](#-project-overview--key-features)
- [System Architecture (What We Built)](#-system-architecture-what-we-built)
  - [1. The Machine Learning Engine (XGBoost)](#1-the-machine-learning-engine-xgboost)
  - [2. The Explainable AI Layer (SHAP)](#2-the-explainable-ai-layer-shap)
  - [3. The Strategy Simulator (Reinforcement Learning)](#3-the-strategy-simulator-reinforcement-learning)
  - [4. The Backend Server (FastAPI)](#4-the-backend-server-fastapi)
  - [5. The Web Dashboard (Frontend)](#5-the-web-dashboard-frontend)
- [Project Directory Structure](#-project-directory-structure)
- [ Dataset Description](#-dataset-description)
- [How to Run the Application (End-to-End)](#-how-to-run-the-application-end-to-end)
- [Technologies & Stack](#-technologies--stack)

---

## 🎯 Project Overview & Key Features

This application evolved from a complex machine learning pipeline into a **modern, full-stack predictive financial dashboard**. Designed for financial analysts and risk managers, the system uses **48 key financial indicators** to assess whether a Taiwanese company is at risk of going bankrupt. 

Rather than just outputting a black-box probability, this platform provides **three pillars of intelligence**:
1. **Accurate Risk Scoring:** Powered by highly-tuned XGBoost trees, calculating a 5-tier risk metric (0-100%).
2. **Interpretability:** Uses SHAP to generate human-readable paragraphs explaining exactly *why* a company received its score (e.g., "Driven by dangerously low Cash Flow Per Share...").
3. **Actionable Recovery:** Deploys a Stable-Baselines3 Reinforcement Learning agent to simulate 10 quarters of financial adjustments, providing a prioritized, step-by-step strategy to save distressed companies from bankruptcy.

---

## 🏗️ System Architecture (What We Built)

The project is structured into distinct, modular layers to ensure maintainability and separation of concerns.

### 1. The Machine Learning Engine (XGBoost)
- Built on Taiwan Economic Journal (1999–2009) historical data (10,000 corporate records).
- Implements `XGBClassifier` with scale_pos_weight to handle the 9:1 data class imbalance.
- Features automatic threshold optimization, probability calibration, and robust cross-validation tracking.

### 2. The Explainable AI Layer (SHAP)
- Integrates TreeExplainer to map local feature contributions.
- Automatically isolates the strongest positive and negative force vectors driving the risk score.
- **New Feature:** Dynamically translates SHAP array matrices into conversational English paragraphs (e.g., "This strategy focuses primarily on enhancing book value...").

### 3. The Strategy Simulator (Reinforcement Learning)
- Built using an OpenAI `Gymnasium` environment (`FinancialRiskEnv`).
- Employs **PPO (Proximal Policy Optimization)** algorithms to allow an AI agent to "play" the financial environment over a simulated 10-quarter timeline.
- The agent learns which financial knobs to turn (e.g., "Increase Liability to Equity") to achieve the steepest drop in risk probability with the least aggressive business moves.

### 4. The Backend Server (FastAPI)
- Exposes standard RESTful API endpoints.
- Maintains the model and scaler in memory for lightning-fast inference.
- Handlers include `/api/metadata`, `/api/predict`, `/api/explain`, and `/api/strategy`.

### 5. The Web Dashboard (Frontend)
- A sleek, dark-themed vanilla HTML/JS/CSS frontend—no clunky framework dependencies.
- Features a "Demo Data" generator that programmatically crafts either a "High-Risk" or "Healthy" company profile at the click of a button.
- Renders elegant visualizations via `Chart.js` for Risk Trajectories and SHAP forces.

---

## 🗂️ Project Directory Structure

We have heavily structured the codebase so that data scientists, backend developers, and UX engineers can navigate the code without overlap.

```text
bankruptcy project/
│
├── 🧠 Machine Learning Core (Root Directory)
│   ├── main.py                          # Re-trains the entire ML pipeline from scratch
│   ├── config.py                        # Global pipeline toggles (Scaling, Seed, Splits)
│   ├── data_loader.py                   # Data ingestion and train/test splitting
│   ├── feature_engineering.py           # VIF feature analysis and dataset cleaning
│   ├── model_trainer.py                 # Core XGBoost fitting
│   ├── rl_strategy_optimizer.py         # The Gymnasium RL Environment training script
│   ├── shap_analyzer.py                 # SHAP TreeExplainer generation
│   ├── risk_simulator.py                # Wrapper used by the backend to fetch logic
│   ├── feature_metadata.json            # Configuration file setting realistic limits for RL Agent actions
│   └── corporate_bankruptcy_dataset.csv # The original Taiwan economic dataset
│
├── 🌐 Backend API (FastAPI)
│   ├── backend/
│   │   ├── app.py                       # The Uvicorn standard entrypoint and router
│   │   ├── model_loader.py              # Singletons ensuring XGBoost loads into memory only once
│   │   ├── prediction_api.py            # API Route for calculating Risk Probability
│   │   ├── shap_api.py                  # API Route for running the SHAP explainer
│   │   └── rl_api.py                    # API Route for kicking off the 10-quarter RL simulation
│
├── 💻 Frontend Application
│   ├── frontend/
│   │   ├── index.html                   # The Entry Form (48 fields)
│   │   ├── dashboard.html               # The Results Analytics Page
│   │   ├── style.css                    # Dark Fintech styling variables and aesthetics
│   │   └── app.js                       # Frontend Controller: DOM logic, Chart instances, Text generation
│
└── 📁 Generated Artifacts
    ├── saved_model/                     # The serialized .pkl objects for XGBoost & Scalers
    └── outputs/                         # Beautiful ROC-AUC, CV matrices, and SHAP plots generated during training
```

---

## 🚀 How to Run the Application (End-to-End)

Follow these steps to spin up the entire application locally on your machine.

### Prerequisites
Make sure you have Python 3.9+ installed. First, install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### Step 1: (Optional) Re-train the Machine Learning Pipeline
If you want to train the model from scratch (for example, if you change `config.py` logic), you can run the primary orchestrator. Note: The pre-trained model `xgboost_model.pkl` is already compiled, so this step is strictly optional.
```bash
python main.py
```

### Step 2: Start the FastAPI Backend Server
The Web UI relies on the backend to crunch numbers. Open a terminal in the root `bankruptcy project` folder and run the Uvicorn server:
```bash
uvicorn backend.app:app --reload
```
You should see `INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)`. Leave this terminal running in the background.

### Step 3: Open the Web Application
Because the backend serves the static frontend files natively via FastAPI's `StaticFiles` router, you can immediately access the dashboard without starting a separate frontend server!

Simply open your web browser and navigate to:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### Walkthrough
1. On the initial page, click the **"Fill Demo Data"** button to automatically generate a sample corporate profile.
2. Click **"Analyze Company Risk"**.
3. View the generated Risk Classification ring, the SHAP Explainer Chart + Paragraph, and review the Reinforcement Learning agent's 10-Quarter strategic rescue plan dynamically formatted onscreen.

---

## 🛠️ Technologies & Stack

- **Machine Learning:** `xgboost`, `scikit-learn`, `pandas`, `numpy`
- **Explainability:** `shap`
- **Reinforcement Learning:** `stable-baselines3`, `gymnasium`
- **Backend:** `FastAPI`, `Uvicorn`, `Pydantic`
- **Frontend UI:** `Vanilla ES6 Javascript`, `HTML5`, `CSS3 Variables`, `FontAwesome`
- **Data Visualization:** `Chart.js`, `matplotlib`, `seaborn`
