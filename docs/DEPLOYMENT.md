# MediGraph AI — Deployment Guide

---

## Local Development (Default — Use This for Demo)

```bash
# Terminal 1 — Backend
source venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
cd frontend
python -m http.server 3000
```

Access: **http://localhost:3000**  
API Docs: **http://localhost:8000/docs**

---

## Pre-Demo Checklist

Run through this before every demo / viva presentation:

```bash
# 1. Verify all services healthy
curl http://localhost:8000/health

# 2. Confirm demo patients exist
curl http://localhost:8000/api/patients/ | python -m json.tool | grep patient_id

# 3. Confirm model loaded
curl http://localhost:8000/api/risk/model/metrics | python -m json.tool | grep accuracy

# 4. Confirm Neo4j has data
curl http://localhost:8000/api/graph/summary | python -m json.tool

# 5. Test drug search (needs internet)
curl "http://localhost:8000/api/drugs/search?drug_name=metformin" | python -m json.tool | head -20
```

If anything fails, check the `Troubleshooting` section in SETUP.md.

---

## Cloud Deployment for Sharing / Evaluation

### Option 1: Render.com (Recommended — Free Tier)

Render gives you a public HTTPS URL so your evaluator can access the project without running it locally.

```bash
# Step 1: Push to GitHub
git add .
git commit -m "MediGraph AI complete"
git push origin main

# Step 2: Go to https://render.com
# → New → Web Service → Connect GitHub repo

# Step 3: Configure:
#   Name:          medigraph-ai
#   Runtime:       Python 3
#   Build Command: pip install -r requirements.txt && python -m backend.ml.train_model
#   Start Command: uvicorn backend.main:app --host 0.0.0.0 --port $PORT

# Step 4: Environment Variables — add all from your .env file:
#   GEMINI_API_KEY    = your_key
#   MONGODB_URI       = your_atlas_uri
#   NEO4J_URI         = your_neo4j_uri
#   NEO4J_USERNAME    = neo4j
#   NEO4J_PASSWORD    = your_password
#   DEBUG             = False
#   ALLOWED_ORIGINS   = https://your-app.onrender.com

# Step 5: Deploy → get URL like https://medigraph-ai.onrender.com
```

**Frontend on Render (Static Site):**
```
→ New → Static Site → Same GitHub repo
→ Root Directory: frontend
→ Publish Directory: .
→ No build command needed
→ Set API_BASE environment variable to your backend URL
```

### Option 2: Railway.app (Simplest)

```bash
npm install -g @railway/cli
railway login
railway init
railway up

# Set environment variables:
railway variables set GEMINI_API_KEY=your_key
railway variables set MONGODB_URI=your_uri
# ... etc
```

### Option 3: Google Cloud Run (Free tier: 2M requests/month)

```bash
# Build Docker image
gcloud builds submit --tag gcr.io/YOUR_PROJECT/medigraph-api

# Deploy
gcloud run deploy medigraph-api \
  --image gcr.io/YOUR_PROJECT/medigraph-api \
  --platform managed \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --set-env-vars "GEMINI_API_KEY=your_key,MONGODB_URI=your_uri,..."
```

---

## Docker (Local or Cloud)

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF and Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-train model at build time (uses data/patient_features.csv)
RUN python -m backend.ml.train_model

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml (full stack)
```yaml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - MONGODB_URI=${MONGODB_URI}
      - NEO4J_URI=${NEO4J_URI}
      - NEO4J_USERNAME=${NEO4J_USERNAME}
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - DEBUG=False
      - ALLOWED_ORIGINS=http://localhost:3000
    volumes:
      - ./data:/app/data
      - ./backend/ml/saved_models:/app/backend/ml/saved_models
    restart: unless-stopped

  frontend:
    image: nginx:alpine
    ports:
      - "3000:80"
    volumes:
      - ./frontend:/usr/share/nginx/html:ro
    depends_on:
      - api
    restart: unless-stopped
```

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop
docker-compose down
```

---

## Production Security Checklist

Before deploying to any public URL:

- [ ] Set `DEBUG=False` in environment variables
- [ ] Change `SECRET_KEY` to a 32+ character random string
- [ ] Set `ALLOWED_ORIGINS` to your exact frontend URL (not `*`)
- [ ] Enable MongoDB Atlas IP Access List — remove `0.0.0.0/0`, add specific IPs
- [ ] Enable HTTPS — handled automatically by Render/Railway/Cloud Run
- [ ] Remove `uvicorn --reload` flag (only for development)
- [ ] Add rate limiting middleware (fastapi-limiter)
- [ ] Add request logging to external service (Datadog, Sentry)

---

## Updating the Model After New Data

```bash
# 1. Reprocess datasets (if raw data changed)
python scripts/preprocess_all_datasets.py

# 2. Retrain model
python -m backend.ml.train_model

# 3. Restart backend (model is lazy-loaded on first prediction call)
# Development:
uvicorn backend.main:app --reload

# Production (Render/Railway):
# Push new model files to repo → triggers automatic redeploy
git add backend/ml/saved_models/
git commit -m "Retrained model with updated dataset"
git push
```

---

## Environment Variables Reference

| Variable | Required | Example | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ | `AIzaSy...` | Google AI Studio API key |
| `MONGODB_URI` | ✅ | `mongodb+srv://...` | MongoDB Atlas connection string |
| `MONGODB_DB_NAME` | | `medigraph` | Database name (default: medigraph) |
| `NEO4J_URI` | ✅ | `neo4j+s://...` | Neo4j Aura instance URI |
| `NEO4J_USERNAME` | | `neo4j` | Default: neo4j |
| `NEO4J_PASSWORD` | ✅ | `yourpassword` | From Neo4j credentials file |
| `MODEL_PATH` | | `./backend/ml/saved_models/xgboost_adherence_model.pkl` | XGBoost model file |
| `SCALER_PATH` | | `./backend/ml/saved_models/feature_scaler.pkl` | StandardScaler file |
| `DEBUG` | | `True` | Set False in production |
| `SECRET_KEY` | | `change-me-32chars` | JWT signing key |
| `ALLOWED_ORIGINS` | | `http://localhost:3000` | Comma-separated CORS origins |
| `LOG_LEVEL` | | `INFO` | Logging verbosity |
| `OPENFDA_API_KEY` | | (optional) | Increases FDA API rate limit |