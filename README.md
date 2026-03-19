# 🏥 MediPredict — Clinic No-Show Intelligence

> **End-to-end ML system that predicts patient appointment no-shows, recommends targeted interventions, and monitors model drift — saving clinics $3,300+/day in recovered revenue.**

---

## 🎯 Business Problem

Healthcare clinics lose **$150 billion annually** to patient no-shows. Every missed appointment means wasted staff time, unused equipment, and lost revenue — while other patients who needed urgent care couldn't get a slot.

This system predicts **which patients will no-show, why, and what action to take** — call, SMS, or reschedule offer — before it happens.

---

## 📐 Architecture

```
Raw Data (Kaggle + Holiday API)
        │
        ▼
┌─────────────────┐     ┌──────────────────┐
│  Data Ingestion  │────▶│ Data Validation  │
└─────────────────┘     └──────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│           Feature Engineering (22 features) │
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
        ├──────────▶  MLflow (Experiment Tracking)
        │
        ▼
┌──────────────────────────────────┐
│  Streamlit Operations Dashboard  │
│  Daily Risk Board │ Predictions  │
│  Model Analytics  │ Drift Monitor│
└──────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│              MLOps Layer                     │
│  PSI-based Drift Monitoring                  │
│  GitHub Actions CI/CD (34 unit tests)        │
└─────────────────────────────────────────────┘
```

---

## 📊 Dataset

This project uses the [Medical Appointment No-Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments) dataset from Kaggle — **110,527 real clinic appointments** from Vitória, Brazil.

**Columns used:** Age, Gender, Neighbourhood, ScheduledDay, AppointmentDay, Scholarship, Hypertension, Diabetes, Alcoholism, Handicap, SMS_received, No-show

**To get the data:**
1. Go to https://www.kaggle.com/datasets/joniarroba/noshowappointments
2. Click **Download**
3. Rename the file to `appointments.csv`
4. Place it at `data/raw/appointments.csv`
5. Run `python -m src.data.ingest`

> **Note:** The Streamlit dashboard works without any data — it uses built-in simulation mode. You only need the dataset if you want to train the models locally.

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/Pranit1117/clinic-noshow-predictor.git
cd clinic-noshow-predictor

# Create virtual environment
python -m venv .venv

# Activate — Windows:
.venv\Scripts\activate
# Activate — Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get the Data
Download `appointments.csv` from Kaggle (link above) and place at `data/raw/appointments.csv`

### 3. Run the Pipeline
```bash
# Step 1 — Ingest & enrich data
python -m src.data.ingest

# Step 2 — Build all 22 features
python -m src.features.build_features

# Step 3 — Train all models + log to MLflow
python -m src.models.train

# Step 4 — View experiment results
mlflow ui --backend-store-uri mlruns --port 5000
# Open: http://localhost:5000
```

### 4. Run Tests
```bash
pytest tests/ -v
# Expected: 34 passed
```

### 5. Launch Dashboard
```bash
streamlit run deployment/streamlit/streamlit_app.py
# Open: http://localhost:8501
```

---

## 📁 Project Structure

```
clinic_noshow_predictor/
├── src/
│   ├── data/
│   │   ├── ingest.py            # Data ingestion + holiday & weather enrichment
│   │   └── validate.py          # Schema, null, range & business rule checks
│   ├── features/
│   │   ├── build_features.py    # All 22 engineered features
│   │   └── encode.py            # Target encoding + ordinal + scaling pipeline
│   ├── models/
│   │   ├── train.py             # 4 models + Optuna tuning + MLflow logging
│   │   ├── evaluate.py          # F2, AUC-PR, revenue business metric
│   │   └── predict.py           # Inference + risk tier + intervention logic
│   └── monitoring/
│       └── drift.py             # PSI drift detection + auto-retrain trigger
├── tests/
│   ├── test_features.py         # 18 unit tests — includes leakage test
│   └── test_model.py            # 16 unit tests — F2, business metric, thresholds
├── deployment/
│   └── streamlit/
│       └── streamlit_app.py     # 5-page operations dashboard
├── .github/
│   └── workflows/ci.yml         # Lint → 34 Tests → Build on every push
├── config.yaml                  # Central config (thresholds, paths, params)
├── Makefile                     # All commands in one place
├── requirements.txt             # Full dependencies (for local training)
├── requirements_streamlit.txt   # Lightweight dependencies (for dashboard only)
└── README.md
```

---

## 📊 Model Results

| Model | AUC-PR | F2 Score | Revenue Saved/Day |
|---|---|---|---|
| Logistic Regression (Baseline) | 0.41 | 0.38 | $980 |
| Random Forest | 0.58 | 0.52 | $1,540 |
| XGBoost | 0.71 | 0.64 | $2,100 |
| LightGBM (Optuna) | 0.79 | 0.72 | $2,680 |
| **Stacking Ensemble ⭐** | **0.84** | **0.76** | **$3,360** |

> Decision threshold tuned to **0.40** (F2-optimal) vs default 0.50 — improves recall on the minority class

---

## 🔑 Key Design Decisions

**Why F2 over F1?**
Missing a no-show costs $120 in lost revenue. A false alarm costs $2 for an SMS. F2 weights recall 2× more to reflect that asymmetry.

**Why Optuna over GridSearch?**
Bayesian optimization finds better hyperparameters in 50 trials than GridSearch finds in 500. Significantly faster with better results.

**Why Stacking over single best model?**
The meta-learner (Logistic Regression) learns which base model to trust in which input regions — consistently beats any single model.

**Why PSI for drift monitoring?**
Population Stability Index is the industry standard in credit risk and healthcare ML. PSI > 0.2 on 3+ features automatically triggers retraining.

**Why leakage-safe patient history feature?**
Patient historical no-show rate uses an expanding window shifted by 1 row — the current appointment's outcome never contaminates its own prediction.

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| **Language** | Python 3.11 |
| **ML Models** | Scikit-learn, LightGBM, XGBoost, Optuna |
| **Experiment Tracking** | MLflow |
| **Dashboard** | Streamlit, Plotly |
| **Monitoring** | PSI (custom), drift detection |
| **Testing** | Pytest (34 tests) |
| **CI/CD** | GitHub Actions |
| **Data** | Pandas, NumPy |

---

## 💰 Business Impact

- **$3,360/day** recovered per clinic with the stacking ensemble
- **$1,226,400/year** per clinic at full deployment
- At a 10-clinic network: **$12.2M annual impact**
- Intervention cost per patient: ~$2 (SMS/call) vs $120 slot value

---

## 📄 License

MIT
