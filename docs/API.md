# MediGraph AI — API Reference

**Base URL:** `http://localhost:8000`  
**Interactive Docs:** http://localhost:8000/docs (Swagger UI — auto-generated)  
**ReDoc:** http://localhost:8000/redoc  
**Health Check:** http://localhost:8000/health

---

## Authentication

No authentication required in development (`DEBUG=True`).  
JWT middleware is scaffolded in `config.py` — enable by setting `DEBUG=False` and adding auth middleware in `main.py` for production.

---

## Standard Response Format

All endpoints return:
```json
{
  "success": true,
  "message": "OK",
  "data": { ... },
  "error": null
}
```

On error:
```json
{
  "success": false,
  "message": "Error description",
  "data": null,
  "error": "Detailed error string"
}
```

---

## Health

### GET /
```json
{ "name": "MediGraph AI", "version": "1.0.0", "status": "running", "docs": "/docs" }
```

### GET /health
Returns real-time status of all connected services.
```json
{
  "status": "healthy",
  "services": {
    "mongodb": "connected",
    "neo4j":   "connected",
    "ml_model":"loaded"
  }
}
```

---

## Patients `/api/patients`

### POST /api/patients/
Create a new patient in MongoDB + Neo4j.

**Request body:**
```json
{
  "patient_id":               "P001",
  "name":                     "Rajesh Kumar",
  "age":                      58,
  "gender":                   "Male",
  "conditions":               ["type 2 diabetes", "hypertension"],
  "medications":              ["metformin", "lisinopril"],
  "adherence_rate":           62.0,
  "exercise_level":           2,
  "follow_up_frequency":      3,
  "comorbidity_count":        2,
  "medication_count":         2,
  "hospital_visits_last_year":2,
  "hba1c":                    8.5,
  "bmi":                      28.4
}
```

**Returns:** `201` `{ "id": "mongo_id", "patient_id": "P001", "name": "Rajesh Kumar" }`  
**Errors:** `409` if patient_id already exists

### GET /api/patients/
List all patients. Query params: `limit` (default 100), `skip` (default 0), `risk_level` (LOW/MODERATE/HIGH/CRITICAL).

### GET /api/patients/dashboard/stats
Executive dashboard statistics — all KPIs in one call.
```json
{
  "total_patients":      7,
  "high_risk_patients":  3,
  "documents_processed": 12,
  "avg_adherence_rate":  63.4,
  "risk_distribution":   { "LOW": 2, "MODERATE": 1, "HIGH": 2, "CRITICAL": 2 },
  "kg_nodes":            87,
  "kg_relationships":    145,
  "model_metrics":       { "accuracy": 0.8875, "f1_score": 0.8775, ... }
}
```

### GET /api/patients/search?q=diabetes
Search by name or condition (case-insensitive partial match).

### GET /api/patients/{patient_id}
Full patient detail + latest risk prediction + document count.

### PUT /api/patients/{patient_id}
Partial update. Body: any subset of patient fields.
```json
{ "adherence_rate": 78.5, "exercise_level": 6, "hba1c": 7.8 }
```

### DELETE /api/patients/{patient_id}
Soft-delete (sets `deleted=True`). Record retained for audit.

---

## Document Intelligence `/api/documents`

### POST /api/documents/upload
Upload a PDF and run the full 4-agent LangGraph pipeline.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | PDF | ✅ | Healthcare PDF (max 10MB) |
| `patient_id` | string | ✅ | Patient identifier |
| `patient_name` | string | | Default: "Unknown" |
| `document_type` | string | | prescription \| lab_report \| medical_summary |
| `age` | int | | Default: 50 |
| `adherence_rate` | float | | 0-100, default: 80.0 |
| `exercise_level` | int | | 1-10, default: 5 |
| `follow_up_frequency` | int | | visits/year, default: 4 |
| `comorbidity_count` | int | | default: 0 |
| `medication_count` | int | | default: 1 |
| `hospital_visits` | int | | default: 0 |
| `conditions` | string | | comma-separated |
| `medications` | string | | comma-separated |
| `hba1c` | float | | optional |
| `bmi` | float | | optional |

**Response:**
```json
{
  "patient_id":    "P001",
  "document_type": "prescription",
  "raw_text":      "Patient: Rajesh Kumar...",
  "extracted_entities": {
    "diseases":     ["type 2 diabetes", "hypertension"],
    "medications":  ["metformin", "lisinopril"],
    "symptoms":     ["fatigue", "frequent urination"],
    "lab_tests":    ["hba1c", "blood pressure"],
    "lab_values":   { "hba1c": "8.5%", "blood pressure": "148/92 mmHg" },
    "risk_factors": ["poor medication adherence", "elevated hba1c"],
    "dosages":      { "metformin": "500mg twice daily" },
    "instructions": ["follow up in 4 weeks"]
  },
  "summary":            "Patient Rajesh Kumar presents with poorly controlled...",
  "processing_time_ms": 4523.5,
  "status":             "success"
}
```

### POST /api/documents/analyze-text
Analyze pasted text without PDF. Query params: `patient_id`, `text`, `document_type`.

### GET /api/documents/patient/{patient_id}?limit=20
All processed documents for a patient, newest-first.

### GET /api/documents/recent?limit=10
Most recent documents across all patients.

### GET /api/documents/{doc_id}
Single document by MongoDB `_id` — includes full raw text and intervention plan.

---

## Risk Prediction `/api/risk`

### POST /api/risk/predict
Run XGBoost inference + Gemini explanation + persist prediction.

**Request body:**
```json
{
  "patient_id":               "P001",
  "adherence_rate":           62.0,
  "age":                      58,
  "comorbidity_count":        2,
  "medication_count":         3,
  "exercise_level":           3,
  "follow_up_frequency":      4,
  "hospital_visits_last_year":2,
  "hba1c":                    8.5,
  "bmi":                      28.4
}
```

**Response:**
```json
{
  "patient_id":      "P001",
  "risk_score":      78.2,
  "risk_level":      "HIGH",
  "adherence_level": "POOR",
  "key_risk_factors": [
    "Poor medication adherence (62%)",
    "Elevated HbA1c (8.5%) — poor glycaemic control",
    "High comorbidity burden (2 conditions)"
  ],
  "explanation":    "This patient has a HIGH risk score of 78.2%, driven primarily by...",
  "recommendations":["Weekly check-in for 4 weeks", "Enrol in adherence programme"],
  "predicted_at":   "2024-03-15T10:30:00"
}
```

**Risk level thresholds:**

| Score | Level | Action |
|-------|-------|--------|
| 0 – 34% | LOW | Routine monitoring |
| 35 – 54% | MODERATE | Bi-weekly check-in |
| 55 – 74% | HIGH | Weekly follow-up |
| 75 – 100% | CRITICAL | Immediate intervention |

### POST /api/risk/predict/batch
Batch prediction for up to 20 patients. Body: array of RiskPredictionRequest objects.

### GET /api/risk/patient/{patient_id}/latest
Most recent risk prediction.

### GET /api/risk/patient/{patient_id}/history?limit=10
Prediction history for trend charting (newest-first).

### GET /api/risk/patient/{patient_id}/trend
Returns: `trend` (improving/worsening/stable), `first_score`, `latest_score`, `delta`, chart `series`.

### GET /api/risk/all/high-risk?include_critical_only=false
All patients with HIGH or CRITICAL risk (based on latest prediction).

### GET /api/risk/model/metrics
XGBoost training metrics: accuracy, precision, recall, F1, ROC-AUC, CV scores, feature importances.

---

## What-If Simulation `/api/simulation`

### POST /api/simulation/run
Run care simulation with auto-generated or custom scenarios.

**Request body:**
```json
{
  "patient_id": "P001",
  "base_data": {
    "patient_id":               "P001",
    "adherence_rate":           62.0,
    "age":                      58,
    "exercise_level":           3,
    "follow_up_frequency":      4,
    "comorbidity_count":        2,
    "medication_count":         3,
    "hospital_visits_last_year":2
  },
  "scenarios": null
}
```

Pass `"scenarios": null` to auto-generate 3 scenarios (Current/Improved/Poor).

**Response:**
```json
{
  "patient_id":   "P001",
  "baseline_risk": 78.2,
  "scenarios": [
    {
      "label":            "Current Behavior",
      "risk_score":       78.2,
      "risk_level":       "HIGH",
      "predicted_outcome":"🟠 High — Likely deterioration without active intervention",
      "ai_explanation":  "Maintaining current habits...",
      "adherence_rate":   62.0,
      "exercise_level":   3,
      "follow_up_frequency": 4
    },
    {
      "label":            "Improved Adherence",
      "risk_score":       42.7,
      "risk_level":       "MODERATE",
      "predicted_outcome":"🟡 Moderate — Increased monitoring recommended",
      "ai_explanation":  "Improving adherence to 82% significantly reduces...",
      "adherence_rate":   82.0,
      "exercise_level":   5,
      "follow_up_frequency": 8
    }
  ],
  "recommendation": "The simulation reveals a 35.5% risk differential...",
  "simulated_at":   "2024-03-15T10:35:00"
}
```

### POST /api/simulation/quick
Instant single-comparison (XGBoost only, no Gemini). Query params:
`patient_id`, `base_adherence`, `new_adherence`, `age`, `comorbidity_count`, etc.

### GET /api/simulation/scenarios/default
Returns 4 built-in scenario templates with delta values for frontend form population.

### GET /api/simulation/patient/{patient_id}?limit=5
Past simulation runs for a patient.

---

## Knowledge Graph `/api/graph`

All endpoints return the standard graph format:
```json
{
  "nodes": [
    { "id": "1234", "label": "Rajesh Kumar", "type": "Patient", "properties": {...} },
    { "id": "5678", "label": "type 2 diabetes", "type": "Disease", "properties": {...} }
  ],
  "relationships": [
    { "source_id": "1234", "target_id": "5678", "relationship_type": "HAS_DISEASE", "properties": {} }
  ],
  "node_count":         8,
  "relationship_count": 12
}
```

### GET /api/graph/patient/{patient_id}
Full patient subgraph including second-hop Disease→Medication links.

### GET /api/graph/summary
Aggregate counts: node types, relationship types, totals.

### GET /api/graph/high-risk
Graph of all patients with `adherence_rate < 70` and their diseases.

### GET /api/graph/disease/{disease_name}
All patients with a specific disease + their medications.

### GET /api/graph/medication/{med_name}
All patients taking a specific medication + their conditions.

### POST /api/graph/ingest
Manually ingest extracted entities into Neo4j.
```json
{
  "patient_id": "P001",
  "entities": {
    "diseases":    ["type 2 diabetes"],
    "medications": ["metformin"],
    "symptoms":    ["fatigue"],
    "lab_tests":   ["hba1c"],
    "lab_values":  { "hba1c": "8.5%" },
    "risk_factors":["poor medication adherence"],
    "dosages":     { "metformin": "500mg twice daily" },
    "instructions":[]
  }
}
```

### GET /api/graph/stats
Detailed analytics: top diseases by patient count, top medications, average connections per patient.

---

## Drug Safety `/api/drugs`

### GET /api/drugs/search?drug_name=metformin
Full drug safety profile from OpenFDA FAERS database.
```json
{
  "drug_name":   "metformin",
  "brand_names": ["Glucophage", "Fortamet"],
  "generic_name":"metformin hydrochloride",
  "manufacturer":"Bristol-Myers Squibb",
  "top_adverse_events": [
    { "term": "nausea",    "count": 15234 },
    { "term": "diarrhoea", "count": 12891 }
  ],
  "warnings":         ["Lactic acidosis risk..."],
  "contraindications":["Renal impairment (eGFR < 30)..."],
  "indications":      ["Type 2 diabetes mellitus management..."],
  "total_reports":    87450,
  "source":           "OpenFDA"
}
```

### GET /api/drugs/interactions?drug_a=metformin&drug_b=lisinopril
Co-occurrence signal from FAERS. Returns `signal_strength`: NONE/LOW/MODERATE/HIGH.

### GET /api/drugs/recalls?drug_name=metformin&limit=5
FDA enforcement / recall history.

### POST /api/drugs/patient-check
Batch safety check for all medications.
```json
{ "patient_id": "P001", "medications": ["metformin", "lisinopril", "atorvastatin"] }
```

### GET /api/drugs/ndc/{ndc}
Look up drug by NDC product code.

---

## Error Codes

| Code | Meaning | Common Cause |
|------|---------|-------------|
| 400 | Bad request | Invalid input, wrong document type |
| 404 | Not found | Patient or document ID doesn't exist |
| 409 | Conflict | Duplicate patient_id |
| 413 | Too large | PDF file exceeds 10MB |
| 422 | Validation error | Pydantic schema mismatch — check request body |
| 500 | Server error | Check backend logs |
| 502 | External API error | OpenFDA or Gemini unavailable |
| 504 | Timeout | External API took too long — retry |