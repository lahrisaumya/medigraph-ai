# MediGraph AI — Complete Setup Guide

> **Time to complete:** ~45 minutes from zero to running demo

---

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Python | 3.10+ | `python --version` |
| pip | latest | `pip --version` |
| Git | any | `git --version` |
| Tesseract OCR | any | `tesseract --version` (optional, for scanned PDFs) |

**Install Tesseract (optional — only needed for scanned PDF support):**
```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer: https://github.com/UB-Mannheim/tesseract/wiki
```

---

## Step 1: Clone & Python Environment

```bash
# Clone the project
git clone <your-repo-url> medigraph-ai
cd medigraph-ai

# Create virtual environment
python -m venv venv

# Activate — Linux/Mac
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

---

## Step 2: Get Free API Keys (15 minutes)

### 2a. Gemini API Key
```
1. Go to:  https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key" → "Create API key in new project"
4. Copy the key (starts with AIza...)
```

### 2b. MongoDB Atlas (Free 512MB cluster)
```
1. Go to:  https://www.mongodb.com/cloud/atlas/register
2. Create free account
3. Choose: Free tier (M0) → Cloud: AWS → Region: nearest to you
4. Security → Database Access → Add Database User
   - Username: medigraph_user
   - Password: create a strong password → SAVE IT
5. Security → Network Access → Add IP Address → Allow Access from Anywhere (0.0.0.0/0)
6. Deployment → Clusters → Connect → Drivers → Python
7. Copy the connection string — looks like:
   mongodb+srv://medigraph_user:<password>@cluster0.xxxxx.mongodb.net/
8. Replace <password> with your actual password
9. Add /medigraph at the end:
   mongodb+srv://medigraph_user:YOURPASSWORD@cluster0.xxxxx.mongodb.net/medigraph
```

### 2c. Neo4j Aura Free Instance
```
1. Go to:  https://neo4j.com/cloud/platform/aura-graph-database/
2. Click "Start Free" → Create free account
3. Create Database → AuraDB Free → Give it a name
4. IMPORTANT: When credentials popup appears — click "Download" immediately
   (This is the ONLY time you see the password)
5. Open the downloaded .txt file and copy:
   - NEO4J_URI      = the URI line (starts with neo4j+s://)
   - NEO4J_USERNAME = neo4j
   - NEO4J_PASSWORD = the generated password
6. Wait ~2 minutes for instance to become "Running"
```

---

## Step 3: Configure Environment

```bash
# Copy the template
cp .env.example .env
```

Open `.env` in any text editor and fill in all values:

```env
GEMINI_API_KEY=AIzaSy...your_key_here...

MONGODB_URI=mongodb+srv://medigraph_user:YOURPASSWORD@cluster0.xxxxx.mongodb.net/medigraph
MONGODB_DB_NAME=medigraph

NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

MODEL_PATH=./backend/ml/saved_models/xgboost_adherence_model.pkl
SCALER_PATH=./backend/ml/saved_models/feature_scaler.pkl

DEBUG=True
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

---

## Step 4: Prepare the Dataset

### Option A — Use pre-processed dataset (fastest, recommended)
The project includes `data/patient_features.csv` (20,000 patients from 4 real
clinical sources). Just proceed to Step 5 — `train_model.py` picks it up automatically.

### Option B — Reprocess from your raw files (verifiable)

Place your 4 dataset files in the correct locations:
```
data/
├── mimic/
│   ├── PATIENTS.csv
│   ├── ADMISSIONS.csv
│   ├── DIAGNOSES_ICD.csv
│   ├── PRESCRIPTIONS.csv
│   └── LABEVENTS.csv
├── diabetic_data.csv
├── heart_disease.csv
└── ckd.csv
```

Then run:
```bash
python scripts/preprocess_all_datasets.py
```

Expected output:
```
✅  Saved: data/patient_features.csv
   Total rows   : 20,000
   Real patients: 15,803  (MIMIC + Diabetes130 + HeartDisease + CKD)
   Synthetic    :  4,197
   High risk    :  4,640 (23.2%)
```

---

## Step 5: Train the XGBoost Model

```bash
python -m backend.ml.train_model
```

Expected output:
```
======================================================
  MediGraph AI — XGBoost Adherence Risk Model Training
======================================================
Loading existing dataset: data/patient_features.csv
Dataset shape: (20000, 18)

Training XGBoost classifier ...
Running 5-fold stratified cross-validation ...
CV F1:      0.8812  ±  0.0143
CV ROC-AUC: 0.9387  ±  0.0098

========================================
  MODEL PERFORMANCE METRICS
========================================
  Accuracy   : 0.8875  (88.7%)
  Precision  : 0.8634
  Recall     : 0.8921
  F1 Score   : 0.8775
  ROC-AUC    : 0.9412

✅ Model  saved → backend/ml/saved_models/xgboost_adherence_model.pkl
✅ Scaler saved → backend/ml/saved_models/feature_scaler.pkl
✅ Metrics saved → backend/ml/saved_models/training_metrics.json
```

---

## Step 6: Seed Demo Data

### Neo4j — Paste Cypher in Browser
```
1. Go to: https://console.neo4j.io
2. Click your instance → "Open with Neo4j Browser"
3. Log in with your credentials
4. Copy the ENTIRE contents of scripts/setup_neo4j.cypher
5. Paste into the query box → Press Ctrl+Enter (or click Run)
6. You should see: Nodes created: 37, Relationships created: 62
```

### MongoDB — Run seeder script
```bash
python scripts/seed_mongodb.py
```

Expected output:
```
✅ Inserted patient: Rajesh Kumar (P001)
✅ Inserted patient: Priya Sharma (P002)
...
✅ 7 patients seeded
✅ 5 risk predictions seeded
🎉 MongoDB seeding complete!
```

---

## Step 7: Start the Backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     ✅ Connected to MongoDB: medigraph
INFO:     ✅ Connected to Neo4j Aura
INFO:     ✅ XGBoost model and scaler loaded successfully
INFO:     ✅ All services initialized. API ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Verify at: **http://localhost:8000/health**

Expected:
```json
{
  "status": "healthy",
  "services": {
    "mongodb": "connected",
    "neo4j": "connected",
    "ml_model": "loaded"
  }
}
```

---

## Step 8: Start the Frontend

```bash
# In a NEW terminal (keep backend running in the first)
cd frontend
python -m http.server 3000
```

Open browser: **http://localhost:3000**

You should see the Executive Overview dashboard with KPI cards loading.

---

## Step 9: Run Verification Checks

```bash
# Health check
curl http://localhost:8000/health

# List patients (should show 7 seeded patients)
curl http://localhost:8000/api/patients/ | python -m json.tool

# Dashboard stats
curl http://localhost:8000/api/patients/dashboard/stats | python -m json.tool

# Predict risk for P001
curl -X POST http://localhost:8000/api/risk/predict \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"P001","adherence_rate":62,"age":58,"comorbidity_count":3,"medication_count":3,"exercise_level":2,"follow_up_frequency":3,"hospital_visits_last_year":2,"hba1c":8.5}' \
  | python -m json.tool

# Drug safety search
curl "http://localhost:8000/api/drugs/search?drug_name=metformin" | python -m json.tool

# Graph summary
curl http://localhost:8000/api/graph/summary | python -m json.tool
```

---

## Step 10: Run Tests

```bash
pytest tests/test_api.py -v --tb=short
```

Expected: all tests pass or skip (some require live services).

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `MongoDB connection failed` | Wrong URI or IP not whitelisted | Check MONGODB_URI, add 0.0.0.0/0 in Atlas Network Access |
| `Neo4j connection failed` | Instance not running or wrong password | Check neo4j.io console, verify URI starts with `neo4j+s://` |
| `Model not found at ./backend/ml/saved_models/...` | Model not trained | Run `python -m backend.ml.train_model` |
| `GEMINI_API_KEY invalid` | Wrong or expired key | Get new key at makersuite.google.com |
| `CORS error in browser` | Frontend URL not in allowed origins | Add `http://localhost:3000` to `ALLOWED_ORIGINS` in .env |
| `ModuleNotFoundError: backend` | Running from wrong directory | Always run from project root: `cd medigraph-ai` |
| `Port 8000 already in use` | Another process on port | `kill -9 $(lsof -t -i:8000)` or use `--port 8001` |
| `tesseract not found` | Tesseract not installed | Install tesseract-ocr (OCR is optional — digital PDFs work without it) |
| `Dataset not found` | Missing patient_features.csv | Run `python scripts/preprocess_all_datasets.py` |

---

## Project Structure Quick Reference

```
medigraph-ai/
├── backend/
│   ├── main.py               ← FastAPI app entry point
│   ├── agents/               ← 4 LangGraph agents + orchestrator
│   ├── api/                  ← 6 API route files
│   ├── core/                 ← config.py + schemas.py
│   ├── db/                   ← mongodb.py + neo4j_db.py
│   ├── ml/                   ← XGBoost model (features, predict, train)
│   └── utils/                ← gemini_client.py + pdf_extractor.py
├── frontend/
│   ├── index.html            ← Executive Overview (entry point)
│   ├── assets/style.css      ← Complete dashboard styles
│   ├── utils/                ← api.js, helpers.js, charts.js
│   ├── components/sidebar.js ← Shared navigation
│   └── pages/                ← 6 dashboard pages
├── scripts/
│   ├── preprocess_all_datasets.py ← Merge 4 real datasets
│   ├── seed_mongodb.py       ← Demo patient data
│   └── setup_neo4j.cypher    ← Graph schema + seed data
├── data/
│   ├── patient_features.csv  ← 20,000 patient training dataset
│   ├── sample_patients.json  ← 7 demo patients
│   └── mimic/                ← MIMIC-III raw CSVs
├── docs/                     ← All documentation
├── tests/test_api.py         ← Pytest test suite
├── requirements.txt
└── .env.example
```