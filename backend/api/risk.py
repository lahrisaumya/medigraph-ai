"""
medigraph-ai/backend/api/risk.py
FILE:    backend/api/risk.py
PURPOSE: Medication Adherence Risk Prediction API.
         Runs the XGBoost model via the LangGraph Risk + Intervention agents,
         persists predictions to MongoDB, and exposes query endpoints for
         prediction history, trending, and model performance metrics.

ENDPOINTS:
    POST /api/risk/predict                       — Predict risk for a patient
    POST /api/risk/predict/batch                 — Predict risk for multiple patients
    GET  /api/risk/patient/{patient_id}/latest   — Latest prediction for patient
    GET  /api/risk/patient/{patient_id}/history  — Prediction history (for trending)
    GET  /api/risk/patient/{patient_id}/trend    — Risk trend summary
    GET  /api/risk/all/high-risk                 — All high-risk patients
    GET  /api/risk/model/metrics                 — XGBoost training metrics

DEPENDENCIES:
    backend.agents.graph   → run_risk_only()
    backend.db.mongodb     → save_risk_prediction(), get_latest_risk() …
    backend.ml.predict     → get_training_metrics()
    backend.core.schemas   → RiskPredictionRequest, RiskPredictionResponse …
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.core.schemas import (
    RiskPredictionRequest,
    RiskPredictionResponse,
    RiskLevel,
    AdherenceLevel,
    APIResponse,
)
from backend.agents.graph import run_risk_only
from backend.db.mongodb import (
    save_risk_prediction,
    get_latest_risk,
    get_risk_history,
    get_db,
)
from backend.ml.predict import get_training_metrics

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/risk",
    tags=["Risk Prediction"],
)


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/risk/predict
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=RiskPredictionResponse,
    summary="Predict medication adherence risk for a patient",
)
async def predict_risk_endpoint(request: RiskPredictionRequest):
    """
    Run the full Risk + Intervention pipeline for a single patient.

    **Pipeline:**
    1. Agent 3 (Risk) — XGBoost model inference → risk_score, risk_level, risk_factors
    2. Agent 3 (Risk) — Gemini explanation → clinical narrative
    3. Agent 4 (Intervention) — Gemini intervention plan
    4. Persist prediction to MongoDB

    **Risk Score interpretation:**
    | Score   | Level    | Action                        |
    |---------|----------|-------------------------------|
    | 0–34%   | LOW      | Routine monitoring             |
    | 35–54%  | MODERATE | Bi-weekly check-in             |
    | 55–74%  | HIGH     | Weekly follow-up               |
    | 75–100% | CRITICAL | Immediate clinical intervention |
    """
    logger.info(
        f"[risk/predict] patient={request.patient_id} | "
        f"adherence={request.adherence_rate}% | age={request.age}"
    )

    # ── Build state for LangGraph ─────────────────────────────────────────
    state = request.model_dump()
    state.update({
        "conditions":       [],
        "medications":      [],
        "lab_values":       {},
        "current_symptoms": [],
    })

    # ── Run Risk + Intervention agents ────────────────────────────────────
    try:
        result = await run_risk_only(state)
    except Exception as exc:
        logger.error(f"[risk/predict] Pipeline error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Risk pipeline failed: {str(exc)}")

    # ── Extract outputs ───────────────────────────────────────────────────
    risk_score     = float(result.get("risk_score",  50.0))
    risk_level_str = result.get("risk_level",        "MODERATE")
    adherence_lvl  = result.get("adherence_level",   "GOOD")
    risk_factors   = result.get("risk_factors",      [])
    explanation    = result.get("risk_explanation",  "")
    recommendations = (
        result.get("intervention_plan", {}).get("lifestyle_recommendations", [])
    )

    # ── Persist prediction ────────────────────────────────────────────────
    prediction_record = {
        "patient_id":       request.patient_id,
        "risk_score":       risk_score,
        "risk_level":       risk_level_str,
        "adherence_level":  adherence_lvl,
        "key_risk_factors": risk_factors,
        "explanation":      explanation,
        "recommendations":  recommendations,
        "input_features": {
            "adherence_rate":            request.adherence_rate,
            "age":                       request.age,
            "comorbidity_count":         request.comorbidity_count,
            "medication_count":          request.medication_count,
            "exercise_level":            request.exercise_level,
            "follow_up_frequency":       request.follow_up_frequency,
            "hospital_visits_last_year": request.hospital_visits_last_year,
            "hba1c":                     request.hba1c,
            "bmi":                       request.bmi,
        },
        "predicted_at": datetime.utcnow(),
    }

    try:
        await save_risk_prediction(prediction_record)
    except Exception as exc:
        logger.warning(f"[risk/predict] MongoDB persist failed (non-fatal): {exc}")

    logger.info(
        f"[risk/predict] ✅ patient={request.patient_id} | "
        f"risk_score={risk_score:.1f}% | level={risk_level_str}"
    )

    return RiskPredictionResponse(
        patient_id=request.patient_id,
        risk_score=risk_score,
        risk_level=RiskLevel(risk_level_str),
        adherence_level=AdherenceLevel(adherence_lvl),
        key_risk_factors=risk_factors,
        explanation=explanation,
        recommendations=recommendations,
    )


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/risk/predict/batch
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/predict/batch",
    response_model=APIResponse,
    summary="Predict risk for multiple patients (synchronous batch)",
)
async def predict_risk_batch(requests: List[RiskPredictionRequest]):
    """
    Run risk prediction for up to 20 patients in one call.

    Each patient is processed sequentially (Gemini rate limits prevent
    true parallelism). Returns a list of results in the same order as
    the input list.

    Maximum batch size: 20 patients.
    """
    if len(requests) > 20:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(requests)} exceeds maximum of 20",
        )

    logger.info(f"[risk/predict/batch] Processing {len(requests)} patients")

    results = []
    errors  = []

    for req in requests:
        try:
            state = req.model_dump()
            state.update({"conditions": [], "medications": [], "lab_values": {}, "current_symptoms": []})
            result = await run_risk_only(state)

            results.append({
                "patient_id":       req.patient_id,
                "risk_score":       result.get("risk_score", 50.0),
                "risk_level":       result.get("risk_level", "MODERATE"),
                "adherence_level":  result.get("adherence_level", "GOOD"),
                "key_risk_factors": result.get("risk_factors", []),
                "status":           "success",
            })

            # Persist each result
            await save_risk_prediction({
                "patient_id":  req.patient_id,
                "risk_score":  result.get("risk_score",  50.0),
                "risk_level":  result.get("risk_level",  "MODERATE"),
                "adherence_level": result.get("adherence_level", "GOOD"),
                "key_risk_factors": result.get("risk_factors", []),
                "predicted_at": datetime.utcnow(),
            })

        except Exception as exc:
            logger.error(f"[risk/predict/batch] Failed for {req.patient_id}: {exc}")
            errors.append({"patient_id": req.patient_id, "error": str(exc)})
            results.append({
                "patient_id": req.patient_id,
                "status":     "error",
                "error":      str(exc),
            })

    return APIResponse(
        message=f"Batch complete: {len(results) - len(errors)} success, {len(errors)} errors",
        data={
            "results":       results,
            "total":         len(requests),
            "success_count": len(results) - len(errors),
            "error_count":   len(errors),
            "errors":        errors,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/risk/patient/{patient_id}/latest
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/patient/{patient_id}/latest",
    response_model=APIResponse,
    summary="Get the most recent risk prediction for a patient",
)
async def get_latest_prediction(patient_id: str):
    """
    Returns the most recently stored risk prediction for the given patient,
    including risk score, level, risk factors, explanation, and recommendations.
    """
    logger.info(f"[risk/latest] patient_id={patient_id}")

    prediction = await get_latest_risk(patient_id)
    if not prediction:
        raise HTTPException(
            status_code=404,
            detail=f"No risk prediction found for patient '{patient_id}'. Run /predict first.",
        )

    return APIResponse(
        message="Latest prediction retrieved",
        data=prediction,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/risk/patient/{patient_id}/history
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/patient/{patient_id}/history",
    response_model=APIResponse,
    summary="Get risk prediction history for a patient (for chart trending)",
)
async def get_prediction_history(
    patient_id: str,
    limit: int = Query(default=10, ge=1, le=50, description="Number of historical records"),
):
    """
    Return a time-ordered list of risk predictions for a patient.
    Newest first. Use this to power the risk-trend line chart on the dashboard.

    Each record includes: risk_score, risk_level, predicted_at, key_risk_factors.
    """
    logger.info(f"[risk/history] patient_id={patient_id} limit={limit}")

    history = await get_risk_history(patient_id, limit=limit)

    return APIResponse(
        message=f"Returned {len(history)} prediction(s) for patient '{patient_id}'",
        data={
            "patient_id": patient_id,
            "history":    history,
            "count":      len(history),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/risk/patient/{patient_id}/trend
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/patient/{patient_id}/trend",
    response_model=APIResponse,
    summary="Risk trend summary — improving, worsening, or stable",
)
async def get_risk_trend(patient_id: str):
    """
    Analyse the patient's risk prediction history and return:
    - trend direction: "improving" | "worsening" | "stable" | "insufficient_data"
    - first and latest risk score
    - change in risk score
    - last N predictions as a chart-ready series

    Trend logic: compare average of oldest 2 vs newest 2 predictions.
    """
    history = await get_risk_history(patient_id, limit=10)

    if len(history) < 2:
        return APIResponse(
            message="Insufficient data for trend analysis (need ≥ 2 predictions)",
            data={
                "patient_id":  patient_id,
                "trend":       "insufficient_data",
                "count":       len(history),
                "series":      history,
            },
        )

    # Newest first → reverse for chronological order
    chronological = list(reversed(history))

    scores       = [h.get("risk_score", 50.0) for h in chronological]
    first_score  = scores[0]
    latest_score = scores[-1]
    delta        = round(latest_score - first_score, 1)

    if delta < -3:
        trend = "improving"
    elif delta > 3:
        trend = "worsening"
    else:
        trend = "stable"

    # Chart-ready series
    series = [
        {
            "date":       h.get("predicted_at", ""),
            "risk_score": h.get("risk_score",   50.0),
            "risk_level": h.get("risk_level",   "MODERATE"),
        }
        for h in chronological
    ]

    return APIResponse(
        message=f"Risk trend for patient '{patient_id}': {trend}",
        data={
            "patient_id":   patient_id,
            "trend":        trend,
            "first_score":  first_score,
            "latest_score": latest_score,
            "delta":        delta,
            "series":       series,
            "count":        len(history),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/risk/all/high-risk
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/all/high-risk",
    response_model=APIResponse,
    summary="Get all patients currently classified as HIGH or CRITICAL risk",
)
async def get_all_high_risk_patients(
    include_critical_only: bool = Query(
        default=False,
        description="If true, return only CRITICAL risk patients",
    ),
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    Returns patients whose most recent risk prediction is HIGH or CRITICAL.
    Useful for the 'Priority Patients' panel on the Executive Overview dashboard.

    Each record includes: patient_id, name, risk_score, risk_level, predicted_at.
    """
    logger.info(f"[risk/all/high-risk] critical_only={include_critical_only}")

    try:
        db = get_db()

        levels = ["CRITICAL"] if include_critical_only else ["HIGH", "CRITICAL"]

        # Aggregate: latest prediction per patient, filter by risk level
        pipeline = [
            {"$sort":  {"predicted_at": -1}},
            {"$group": {
                "_id":         "$patient_id",
                "risk_score":  {"$first": "$risk_score"},
                "risk_level":  {"$first": "$risk_level"},
                "predicted_at":{"$first": "$predicted_at"},
            }},
            {"$match": {"risk_level": {"$in": levels}}},
            {"$sort":  {"risk_score": -1}},
            {"$limit": limit},
        ]

        high_risk_list = []
        async for doc in db.risk_predictions.aggregate(pipeline):
            patient_id = doc["_id"]
            # Enrich with patient name from patients collection
            patient = await get_db().patients.find_one(
                {"patient_id": patient_id},
                {"name": 1, "age": 1, "conditions": 1, "_id": 0},
            )
            high_risk_list.append({
                "patient_id":   patient_id,
                "name":         patient.get("name",       "Unknown") if patient else "Unknown",
                "age":          patient.get("age",        0)         if patient else 0,
                "conditions":   patient.get("conditions", [])        if patient else [],
                "risk_score":   round(doc.get("risk_score",  0), 1),
                "risk_level":   doc.get("risk_level",  ""),
                "predicted_at": str(doc.get("predicted_at", "")),
            })

    except Exception as exc:
        logger.error(f"[risk/all/high-risk] Error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return APIResponse(
        message=f"Found {len(high_risk_list)} high-risk patient(s)",
        data={
            "patients": high_risk_list,
            "count":    len(high_risk_list),
            "levels":   levels,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/risk/model/metrics
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/model/metrics",
    response_model=APIResponse,
    summary="XGBoost model performance metrics from last training run",
)
async def model_metrics():
    """
    Return the saved performance metrics from the most recent training run:
    - Accuracy, Precision, Recall, F1 Score, ROC-AUC
    - Cross-validation F1 mean ± std
    - Feature importances (ranked)
    - Training sample counts

    Requires the model to have been trained:
    ```bash
    python -m backend.ml.train_model
    ```
    """
    metrics = get_training_metrics()

    if not metrics:
        raise HTTPException(
            status_code=404,
            detail=(
                "Model metrics not found. "
                "Train the model first: python -m backend.ml.train_model"
            ),
        )

    return APIResponse(
        message="Model performance metrics retrieved",
        data=metrics,
    )
