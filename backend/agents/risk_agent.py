"""
================================================================================
FILE:    backend/agents/risk_agent.py
AGENT:   Agent 3 — Risk Prediction Agent
PURPOSE: Runs the trained XGBoost adherence-risk model against the current
         patient's clinical profile, then calls Gemini to produce a human-
         readable explanation of WHY the patient received that score.

         This agent works whether or not Agent 1 (Document Agent) ran.
         If a PDF was uploaded, it uses enriched state from Agent 1/2.
         If skip_document=True, it uses the raw patient fields from state.

INPUTS  (from LangGraph state):
    patient_id               (str)         : patient identifier
    age                      (int)         : patient age
    adherence_rate           (float)       : current medication adherence %
    comorbidity_count        (int)         : number of concurrent conditions
    medication_count         (int)         : total medications prescribed
    exercise_level           (int, 1-10)   : physical activity level
    follow_up_frequency      (int)         : clinic visits per year
    hospital_visits_last_year(int)         : hospitalisations last 12 months
    hba1c                    (float|None)  : glycated haemoglobin %
    bmi                      (float|None)  : body mass index
    conditions               (list[str])   : active diagnoses
    medications              (list[str])   : active medications

OUTPUTS (added to LangGraph state):
    risk_score         (float)     : 0-100 adherence risk percentage
    risk_level         (str)       : "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
    adherence_level    (str)       : "EXCELLENT" | "GOOD" | "POOR" | "CRITICAL"
    risk_factors       (list[str]) : top human-readable risk factors
    risk_explanation   (str)       : 2-3 sentence Gemini narrative
    risk_agent_status  (str)       : "success" | "error: <message>"
================================================================================
"""

import logging
from typing import Dict, Any, List

from backend.ml.predict import predict_risk
from backend.utils.gemini_client import explain_risk_score

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# AGENT NODE
# ──────────────────────────────────────────────────────────────────────────────

async def risk_prediction_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node — Risk Prediction Agent.

    Workflow:
        1. Assemble patient feature dict from LangGraph state
           (merges document-extracted conditions/medications if available)
        2. Run XGBoost model → risk_score, risk_level, adherence_level, risk_factors
        3. Call Gemini → 2-3 sentence clinical explanation
        4. Return updated state

    Error handling:
        On model failure, returns a MODERATE default (50%) so the pipeline
        does not block the Intervention Agent.  The error is surfaced via
        risk_agent_status and logged for debugging.
    """
    patient_id = state.get("patient_id", "UNKNOWN")
    logger.info(f"[RiskAgent] ▶ Starting risk prediction for patient={patient_id}")

    try:
        # ── Step 1: Build patient feature dict ────────────────────────────────
        patient_data = _build_patient_features(state)
        _log_feature_summary(patient_id, patient_data)

        # ── Step 2: XGBoost inference ─────────────────────────────────────────
        logger.info("[RiskAgent] Running XGBoost inference ...")
        result = predict_risk(patient_data)

        risk_score     = result["risk_score"]      # float 0-100
        risk_level     = result["risk_level"]      # str  e.g. "HIGH"
        adherence_level= result["adherence_level"] # str  e.g. "POOR"
        risk_factors   = result["risk_factors"]    # list[str]

        logger.info(
            f"[RiskAgent] Model output → "
            f"score={risk_score:.1f}% | level={risk_level} | "
            f"adherence_level={adherence_level}"
        )

        # ── Step 3: Gemini explanation ────────────────────────────────────────
        logger.info("[RiskAgent] Calling Gemini for risk explanation ...")
        risk_explanation = await explain_risk_score(
            patient_data = patient_data,
            risk_score   = risk_score,
            risk_factors = risk_factors,
        )

        logger.info(f"[RiskAgent] ✅ Completed for patient={patient_id}")

        return {
            **state,
            "risk_score":        risk_score,
            "risk_level":        risk_level,
            "adherence_level":   adherence_level,
            "risk_factors":      risk_factors,
            "risk_explanation":  risk_explanation,
            "risk_agent_status": "success",
        }

    except Exception as exc:
        logger.error(f"[RiskAgent] ❌ Error: {exc}", exc_info=True)

        return {
            **state,
            "risk_score":        50.0,
            "risk_level":        "MODERATE",
            "adherence_level":   "GOOD",
            "risk_factors":      ["Risk calculation unavailable — model error"],
            "risk_explanation":  "Risk prediction is temporarily unavailable. Please retry.",
            "risk_agent_status": f"error: {str(exc)}",
        }


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _build_patient_features(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assemble the feature dict that predict_risk() and engineer_features() expect.

    Merges:
      • Direct state fields (age, adherence_rate, exercise_level, etc.)
      • Conditions and medications from state (user-entered OR extracted from docs)
      • HbA1c and BMI (optional, default to None → feature engineering uses safe defaults)
    """
    # Conditions: prefer document-extracted (enriched) over user-entered if available
    state_conditions  = state.get("conditions",  []) or []
    doc_diseases      = state.get("extracted_entities", {}).get("diseases", [])
    merged_conditions = list(dict.fromkeys(state_conditions + doc_diseases))  # dedup, preserve order

    # Medications: same merge strategy
    state_medications = state.get("medications", []) or []
    doc_medications   = state.get("extracted_entities", {}).get("medications", [])
    merged_medications= list(dict.fromkeys(state_medications + doc_medications))

    return {
        "age":                       int(state.get("age", 50)),
        "adherence_rate":            float(state.get("adherence_rate", 80.0)),
        "comorbidity_count":         int(state.get("comorbidity_count", 0)),
        "medication_count":          max(int(state.get("medication_count", 1)), len(merged_medications), 1),
        "exercise_level":            int(state.get("exercise_level", 5)),
        "follow_up_frequency":       int(state.get("follow_up_frequency", 4)),
        "hospital_visits_last_year": int(state.get("hospital_visits_last_year", 0)),
        "hba1c":                     state.get("hba1c"),    # None is fine — feature engineering handles it
        "bmi":                       state.get("bmi"),      # None is fine
        "conditions":                merged_conditions,
        "medications":               merged_medications,
    }


def _log_feature_summary(patient_id: str, features: Dict[str, Any]) -> None:
    """Log a compact feature summary for observability."""
    logger.info(
        f"[RiskAgent] Features for patient={patient_id} | "
        f"age={features['age']} | "
        f"adherence={features['adherence_rate']:.0f}% | "
        f"comorbidities={features['comorbidity_count']} | "
        f"medications={features['medication_count']} | "
        f"exercise={features['exercise_level']}/10 | "
        f"follow_up={features['follow_up_frequency']}/yr | "
        f"hba1c={features['hba1c']} | "
        f"conditions={features['conditions']}"
    )