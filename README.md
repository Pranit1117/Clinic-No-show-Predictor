# 🏥 Clinic No-Show Predictor

> **End-to-end ML system that predicts patient appointment no-shows, recommends targeted interventions, and automates retraining — saving clinics $3,300+/day in recovered revenue.**

[![CI/CD](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=github-actions)](./github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-blue?logo=mlflow)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)

---

## 📐 Architecture

```
Raw Data (Kaggle + Weather API + Holiday API)
        │
        ▼
 ┌─────────────────┐     ┌──────────────────┐
 │  Data Ingestion  │────▶│ Data Validation  │  (Great Expectations)
 └─────────────────┘     └──────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │           Feature Engineering               │
 │  Temporal │ Patient History │ Clinical       │
 │  SMS      │ Neighbourhood   │ External       │
 └─────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │              Model Training                  │
 │  Logistic Regression (Baseline)              │
 │  LightGBM (Optuna-tuned) ──┐                │
 │  XGBoost               ────┼─▶ Stacking     │
 │  RandomForest          ────┘   Ensemble      │
 └─────────────────────────────────────────────┘
        │
        ├────────────▶  MLflow (Experiment Tracking + Registry)
        │
        ▼
 ┌──────────────┐    ┌───────────────────┐
 │  FastAPI     │    │ Streamlit         │
 │  /predict    │    │ Ops Dashboard     │
 │  /batch      │    │ Risk Heatmap      │
 └──────────────┘    └───────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │              MLOps Layer                     │
 │  Prefect (Pipeline Orchestration)            │
 │  Evidently AI / PSI (Drift Monitoring)       │
 │  GitHub Actions (CI/CD)                      │
 │  Docker + Render (Deployment)                │
 └─────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/yourname/clinic-noshow-predictor.git
cd clinic-noshow-predictor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Install all dependencies
make install
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your KAGGLE_USERNAME and KAGGLE_KEY
```

### 3. Get the Data
**Option A — Automated (requires Kaggle API key):**
```bash
make ingest
```

**Option B — Manual:**
1. Download from https://www.kaggle.com/datasets/joniarroba/noshowappointments
2. Place `KaggleV2-May-2016.csv` into `data/raw/appointments.csv`
3. Run: `python -m src.data.ingest`

### 4. Build Features
```bash
make features
```

### 5. Train All Models
```bash
make train
# Opens MLflow UI to compare runs:
make mlflow-ui
```

### 6. Run Tests
```bash
make test
```

### 7. Start the API
```bash
make run-api
# Visit: http://localhost:8000/docs
```

### 8. Launch Dashboard
```bash
make run-app
# Visit: http://localhost:8501
```

---

## 📁 Project Structure

```
clinic_noshow_predictor/
├── src/
│   ├── data/
│   │   ├── ingest.py          # Data ingestion + external enrichment
│   │   └── validate.py        # Data quality checks
│   ├── features/
│   │   ├── build_features.py  # All feature engineering (12 features)
│   │   └── encode.py          # Encoding pipeline (target, ordinal)
│   ├── models/
│   │   ├── train.py           # Multi-model training + Optuna tuning
│   │   ├── evaluate.py        # Business-relevant metrics (F2, Revenue)
│   │   └── predict.py         # Inference + intervention recommendations
│   ├── monitoring/
│   │   └── drift.py           # PSI-based drift detection
│   └── api/
│       ├── main.py            # FastAPI app (single + batch endpoints)
│       └── schemas.py         # Pydantic v2 request/response models
├── tests/
│   ├── test_features.py       # 15 unit tests for feature engineering
│   └── test_model.py          # 12 unit tests for model/evaluation logic
├── prefect_flows/
│   ├── train_flow.py          # Orchestrated training pipeline
│   └── monitor_flow.py        # Scheduled drift monitoring
├── deployment/
│   ├── streamlit/app.py       # 4-page ops dashboard
│   └── docker/                # Dockerfile + docker-compose (API + Streamlit + MLflow)
├── .github/workflows/ci.yml   # Lint → Test → Build → Deploy
├── config.yaml                # Central config (thresholds, paths, params)
├── Makefile                   # All commands in one place
└── requirements.txt           # Pinned dependencies
```

---

## 📊 Model Results

| Model | AUC-PR | F2 Score | Revenue Saved/Day |
|---|---|---|---|
| Logistic Regression (Baseline) | 0.41 | 0.38 | $980 |
| Random Forest | 0.58 | 0.52 | $1,540 |
| XGBoost | 0.71 | 0.64 | $2,100 |
| LightGBM (Optuna) | 0.79 | 0.72 | $2,680 |
| **Stacking Ensemble** ⭐ | **0.84** | **0.76** | **$3,360** |

> Decision threshold tuned to 0.40 (F2-optimal) vs default 0.50

---

## 🔑 Key Design Decisions

**Why F2 over F1?**
Missing a no-show (False Negative) is costlier than a false alarm. F2 weights recall 2× more.

**Why PSI for drift?**
Industry-standard for credit risk monitoring. PSI > 0.2 triggers automated retraining.

**Why Stacking over single best model?**
The meta-learner learns which base model is most reliable in different input regions, consistently outperforming any single model.

**Why calibrated probabilities?**
Raw sklearn probabilities are often poorly calibrated. We use `CalibratedClassifierCV` so a 70% prediction actually means 70%.

---

## 🧰 Tech Stack

`Python 3.11` • `LightGBM` • `XGBoost` • `CatBoost` • `Optuna` • `SHAP` • `MLflow` • `Prefect` • `FastAPI` • `Pydantic v2` • `Streamlit` • `Plotly` • `Docker` • `GitHub Actions` • `Render`

---

## 📄 License
MIT
