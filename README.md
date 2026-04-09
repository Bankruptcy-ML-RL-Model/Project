<p align="center">
  <img src="frontend/fec_logo.png" alt="FEC Risk AI Logo" height="80">
</p>

<h1 align="center">FEC Risk AI — Corporate Bankruptcy Prediction System</h1>

<p align="center">
  <b>An end-to-end ML + Reinforcement Learning + Agentic AI platform for corporate financial risk analysis</b><br>
  Built by <b>Finance and Economics Club, IIT Guwahati</b>
</p>

---

## 📌 Overview

FEC Risk AI is a full-stack intelligent system that predicts the probability of corporate bankruptcy using an **XGBoost classifier** trained on 95 financial indicators. It goes beyond simple prediction by providing:

1. **SHAP Explainability** — Feature-level attribution analysis showing *why* a company is at risk.
2. **Reinforcement Learning Strategy Optimizer** — A PPO-based RL agent that simulates 10-quarter recovery strategies to reduce bankruptcy risk.
3. **Agentic AI Financial Advisor** — An autonomous LLaMA-3.3 70B agent (via Groq) that orchestrates all three tools and generates a detailed, phase-wise recovery plan.

---

## 🏗️ Project Structure

```
Project/
│
├── backend/                        # FastAPI backend server
│   ├── app.py                      # Main FastAPI application & static file serving
│   ├── prediction_api.py           # /api/predict — XGBoost bankruptcy prediction endpoint
│   ├── shap_api.py                 # /api/shap — SHAP feature attribution endpoint
│   ├── rl_api.py                   # /api/rl-strategy — RL strategy simulation endpoint
│   ├── advisor_api.py              # /api/advisor — Agentic AI advisor endpoint
│   ├── agent_controller.py         # LangChain agent with Groq LLM orchestration
│   └── model_loader.py             # Loads saved XGBoost model & scaler
│
├── frontend/                       # Web dashboard (served as static files)
│   ├── index.html                  # Input form page
│   ├── dashboard.html              # Results dashboard with all 4 sections
│   ├── style.css                   # Glassmorphic premium UI theme
│   ├── app.js                      # Frontend logic, API calls, chart rendering
│   └── fec_logo.png                # FEC IIT Guwahati logo
│
├── saved_model/                    # Trained model artifacts
│   ├── xgboost_bankruptcy_model.json   # Trained XGBoost model
│   └── feature_scaler.pkl              # Feature scaler (StandardScaler)
│
├── outputs/                        # Generated analysis outputs (plots, CSVs)
│   ├── confusion_matrix.png
│   ├── roc_auc_curve.png
│   ├── shap_summary_plot.png
│   ├── feature_importance.png
│   └── ...                         # Other diagnostic plots & CSV reports
│
├── analysis/                       # Text-based analysis results
│   └── shap_analysis_results.txt
│
├── config.py                       # Centralized configuration (paths, hyperparams, thresholds)
├── data_loader.py                  # Dataset loading & preprocessing
├── feature_engineering.py          # Feature engineering pipeline
├── model_trainer.py                # XGBoost model training script
├── model_evaluator.py              # Model evaluation & metrics
├── model_diagnostics.py            # Advanced model diagnostics
├── risk_classifier.py              # Risk category classification logic
├── risk_simulator.py               # Risk simulation utilities
├── shap_analyzer.py                # SHAP analysis pipeline
├── rl_strategy_optimizer.py        # PPO Reinforcement Learning environment & training
├── visualization.py                # Plotting & visualization utilities
├── main.py                         # Master pipeline script (runs everything end-to-end)
├── feature_metadata.json           # Feature names & descriptions
├── corporate_bankruptcy_dataset.csv # Training dataset (95 features)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Quick Start — How to Run the Website

### Prerequisites
- **Python 3.10+** installed
- **Groq API Key** (free at [console.groq.com](https://console.groq.com))

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd Project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set Your Groq API Key
**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
```

**macOS/Linux:**
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

### Step 4: Start the Backend Server
```bash
python -m uvicorn backend.app:app --reload
```
The server will start at **http://127.0.0.1:8000**

### Step 5: Open the Website
Open your browser and navigate to:
```
http://127.0.0.1:8000/index.html
```

---

## 📖 How to Use

1. **Enter Financial Data** — On the input page, click **"Inject Demo Parameters"** to auto-fill sample data, or manually enter the 48 financial indicators.
2. **Analyze** — Click **"SYNTHESIZE DATA"** to run the XGBoost prediction.
3. **View Dashboard** — You will be redirected to the dashboard showing:
   - **Risk Score** — Circular animated gauge showing bankruptcy probability
   - **SHAP Analysis** — Horizontal bar chart of the top financial drivers
4. **Run RL Optimizer** — Click **"Run Deep Simulation"** to execute the PPO reinforcement learning agent and see the 10-quarter risk reduction strategy.
5. **Run AI Advisor** — Click **"RUN AUTONOMOUS ANALYSIS"** to trigger the Agentic AI, which autonomously calls all 3 tools and generates a comprehensive, phase-wise recovery strategy report.

---

## 🧠 Technical Architecture

### Machine Learning Pipeline
| Component | Technology | Description |
|---|---|---|
| **Classifier** | XGBoost | Binary classification on 95 financial features |
| **Explainability** | SHAP (TreeExplainer) | Feature-level contribution analysis |
| **Strategy Optimizer** | PPO (Stable-Baselines3) | 10-quarter bankruptcy risk reduction simulation |
| **Agentic AI** | LangChain + Groq (LLaMA 3.3 70B) | Autonomous multi-tool agent for holistic analysis |

### Web Stack
| Layer | Technology |
|---|---|
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS + Chart.js |
| **LLM Provider** | Groq Cloud (OpenAI-compatible API) |

### Agent Tool Flow
```
User Input → Agent (LLaMA-3.3)
                ├── Tool 1: predict_bankruptcy_risk(features) → XGBoost
                ├── Tool 2: generate_shap_explanation(features) → SHAP
                └── Tool 3: run_rl_strategy(features) → PPO RL
             → Agent synthesizes all outputs into a unified advisory report
```

---

## 🔄 Running the Offline ML Pipeline

To retrain the model from scratch or regenerate all analysis outputs:

```bash
python main.py
```

This runs the full end-to-end pipeline:
1. Data loading & preprocessing
2. Feature engineering
3. XGBoost training with cross-validation
4. Model evaluation & diagnostics
5. SHAP analysis & visualization
6. Risk classification
7. RL strategy optimization

All outputs are saved to the `outputs/` directory.

---

## 📊 Dataset

- **Source**: Taiwan Economic Journal (1999–2009)
- **Samples**: 6,819 companies
- **Features**: 95 financial indicators
- **Target**: Binary (1 = Bankrupt, 0 = Not Bankrupt)
- **Class Imbalance**: ~3.2% positive class (handled via SMOTE in training)

---

## ⚙️ Configuration

All configurable parameters are centralized in `config.py`:
- File paths and directories
- XGBoost hyperparameters
- Risk category thresholds
- RL training timesteps
- Cross-validation folds
- Visualization settings

---

## 👥 Credits

- **Finance and Economics Club (FEC)**, IIT Guwahati
- Built with XGBoost, SHAP, Stable-Baselines3, LangChain, and Groq

---

## 📄 License

This project is for academic and educational purposes. Please contact FEC IIT Guwahati for usage permissions.
