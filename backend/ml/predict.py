"""
================================================================================
FILE:    backend/ml/predict.py
PURPOSE: XGBoost model inference module.
         Loads the trained model + scaler on first call (lazy loading),
         caches them as module-level singletons, and exposes the public
         predict_risk() function used by Risk Agent and Simulation API.

PUBLIC API:
    predict_risk(patient_data)         — single-patient inference
    simulate_scenario(base, override)  — what-if override inference
    get_training_metrics()             — load saved training_metrics.json

RISK LEVEL THRESHOLDS:
    0  – 34.9  →  LOW
    35 – 54.9  →  MODERATE
    55 – 74.9  →  HIGH
    75 – 100   →  CRITICAL

ADHERENCE LEVEL THRESHOLDS:
    ≥ 90%  →  EXCELLENT
    75–89% →  GOOD
    50–74% →  POOR
    < 50%  →  CRITICAL

LAZY LOADING:
    The model is loaded from disk on the first call to predict_risk().
    Subsequent calls reuse the cached objects (no disk I/O).
    Call _load_model() explicitly in app startup to fail fast on missing model.

FALLBACK:
    If the model file is not found, predict_risk() falls back to the
    rule-based heuristic in features.compute_risk_score_heuristic()
    so the API never crashes — it just returns a note in risk_factors.
================================================================================
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from backend.core.config import settings
from backend.core.schemas import AdherenceLevel, RiskLevel
from backend.ml.features import (
    FEATURE_COLUMNS,
    compute_risk_score_heuristic,
    engineer_features,
    identify_risk_factors,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETONS (lazy loaded)
# ──────────────────────────────────────────────────────────────────────────────

_model  = None   # XGBClassifier
_scaler = None   # StandardScaler


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────

def _load_model() -> None:
    """
    Load the XGBoost model and StandardScaler from disk into module singletons.
    Called automatically on the first predict_risk() call (lazy loading).
    Can also be called explicitly at app startup for fail-fast behaviour.

    Raises:
        FileNotFoundError: If model or scaler .pkl files do not exist.
                           Run: python -m backend.ml.train_model
    """
    global _model, _scaler

    model_path  = Path(settings.model_path)
    scaler_path = Path(settings.scaler_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"XGBoost model not found at '{model_path}'. "
            f"Train it first:\n  python -m backend.ml.train_model"
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Feature scaler not found at '{scaler_path}'. "
            f"Train it first:\n  python -m backend.ml.train_model"
        )

    logger.info(f"[predict] Loading model from {model_path} ...")
    _model  = joblib.load(model_path)
    _scaler = joblib.load(scaler_path)
    logger.info("✅ [predict] XGBoost model and scaler loaded successfully")


def _get_model():
    """Return (model, scaler) — loading from disk on first call."""
    global _model, _scaler
    if _model is None or _scaler is None:
        _load_model()
    return _model, _scaler


def is_model_loaded() -> bool:
    """Return True if the model has been loaded into memory."""
    return _model is not None and _scaler is not None


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _classify_risk_level(risk_score: float) -> str:
    """Map a 0–100 risk score to a RiskLevel string."""
    if risk_score >= 75:
        return RiskLevel.CRITICAL.value
    elif risk_score >= 55:
        return RiskLevel.HIGH.value
    elif risk_score >= 35:
        return RiskLevel.MODERATE.value
    else:
        return RiskLevel.LOW.value


def _classify_adherence_level(adherence_rate: float) -> str:
    """Map a 0–100 adherence percentage to an AdherenceLevel string."""
    if adherence_rate >= 90:
        return AdherenceLevel.EXCELLENT.value
    elif adherence_rate >= 75:
        return AdherenceLevel.GOOD.value
    elif adherence_rate >= 50:
        return AdherenceLevel.POOR.value
    else:
        return AdherenceLevel.CRITICAL.value


# ──────────────────────────────────────────────────────────────────────────────
# MAIN INFERENCE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def predict_risk(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict medication adherence risk for a single patient.

    Workflow:
        1. Call engineer_features() to build the 17-column feature DataFrame
        2. Apply StandardScaler transformation
        3. Run XGBClassifier.predict_proba() → probability of high risk
        4. Classify risk score into level and adherence level
        5. Identify human-readable risk factors (rule-based)
        6. Return structured result dict

    Falls back to the rule-based heuristic if the model is unavailable,
    returning an identical dict shape with is_heuristic=True flag.

    Args:
        patient_data: Dict with clinical features. All fields optional:
            age, adherence_rate, comorbidity_count, medication_count,
            exercise_level, follow_up_frequency, hospital_visits_last_year,
            hba1c, bmi, conditions (List[str])

    Returns:
        Dict with keys:
            risk_score       (float)      — 0-100 risk percentage
            risk_level       (str)        — LOW | MODERATE | HIGH | CRITICAL
            adherence_level  (str)        — EXCELLENT | GOOD | POOR | CRITICAL
            risk_factors     (List[str])  — top human-readable risk drivers
            probability      (float)      — raw model probability 0-1
            is_heuristic     (bool)       — True if model was unavailable

    Example:
        result = predict_risk({"age": 58, "adherence_rate": 62.0, "hba1c": 8.5})
        print(result["risk_score"])   # e.g. 78.4
        print(result["risk_level"])   # "HIGH"
    """
    adherence_rate = float(patient_data.get("adherence_rate", 80.0))

    # ── Try XGBoost inference ─────────────────────────────────────────────
    try:
        model, scaler = _get_model()

        # Feature engineering → shape (1, 17)
        X = engineer_features(patient_data)

        # Validate column order
        missing = set(FEATURE_COLUMNS) - set(X.columns)
        if missing:
            logger.warning(f"[predict] Missing features: {missing}. Using defaults.")
            for col in missing:
                X[col] = 0
        X = X[FEATURE_COLUMNS]

        # Scale features
        X_scaled = scaler.transform(X)

        # Model inference — probability of class 1 (high risk)
        probabilities = model.predict_proba(X_scaled)
        probability   = float(probabilities[0][1])
        risk_score    = round(probability * 100, 1)
        is_heuristic  = False

        logger.debug(
            f"[predict] XGBoost output: prob={probability:.4f} → "
            f"score={risk_score} | adherence={adherence_rate:.1f}%"
        )

    except FileNotFoundError as exc:
        # Model file missing — use heuristic fallback
        logger.warning(f"[predict] Model unavailable — using heuristic fallback: {exc}")
        risk_score    = compute_risk_score_heuristic(patient_data)
        probability   = risk_score / 100
        is_heuristic  = True

    except Exception as exc:
        # Unexpected error — use heuristic fallback
        logger.error(f"[predict] Inference error — using heuristic fallback: {exc}", exc_info=True)
        risk_score    = compute_risk_score_heuristic(patient_data)
        probability   = risk_score / 100
        is_heuristic  = True

    # ── Classify outputs ──────────────────────────────────────────────────
    risk_level      = _classify_risk_level(risk_score)
    adherence_level = _classify_adherence_level(adherence_rate)

    # ── Identify risk factors (rule-based for explainability) ─────────────
    risk_factors = identify_risk_factors(patient_data)

    if is_heuristic:
        risk_factors.insert(
            0,
            "⚠️ Using rule-based estimate (ML model not trained yet — "
            "run python -m backend.ml.train_model for full accuracy)"
        )

    result = {
        "risk_score":      risk_score,
        "risk_level":      risk_level,
        "adherence_level": adherence_level,
        "risk_factors":    risk_factors,
        "probability":     round(probability, 4),
        "is_heuristic":    is_heuristic,
    }

    logger.info(
        f"[predict] Result: score={risk_score:.1f}% | "
        f"level={risk_level} | adherence_level={adherence_level} | "
        f"heuristic={is_heuristic}"
    )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# WHAT-IF SCENARIO INFERENCE
# ──────────────────────────────────────────────────────────────────────────────

def simulate_scenario(
    base_data: Dict[str, Any],
    override:  Dict[str, Any],
) -> float:
    """
    Run XGBoost inference with specific parameter overrides for What-If simulation.

    Creates a copy of base_data, applies override values, then calls predict_risk().
    Returns only the risk_score (float 0–100).

    Args:
        base_data: Patient's current clinical feature dict.
        override:  Dict of parameters to change, e.g.:
                   {"adherence_rate": 90.0, "exercise_level": 8}

    Returns:
        risk_score as float 0–100.

    Example:
        current_score  = simulate_scenario(base, {})
        improved_score = simulate_scenario(base, {"adherence_rate": 90.0, "exercise_level": 7})
    """
    scenario_data = {**base_data, **override}
    result = predict_risk(scenario_data)
    return result["risk_score"]


def predict_risk_batch(patients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run predict_risk() on a list of patients.
    Uses the same model instance for all — efficient for batch scoring.

    Args:
        patients: List of patient data dicts.

    Returns:
        List of result dicts in same order as input.
        Each dict has the same keys as predict_risk() output,
        plus 'patient_id' if present in the input dict.
    """
    results = []
    for patient in patients:
        result = predict_risk(patient)
        # Preserve patient_id in result for traceability
        if "patient_id" in patient:
            result["patient_id"] = patient["patient_id"]
        results.append(result)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# MODEL METADATA
# ──────────────────────────────────────────────────────────────────────────────

def get_training_metrics() -> Optional[Dict[str, Any]]:
    """
    Load and return the training metrics saved by train_model.py.
    Returns None if the metrics file does not exist.

    Returns dict with keys:
        accuracy, precision, recall, f1_score, roc_auc,
        cv_f1_mean, cv_f1_std,
        train_samples, test_samples,
        feature_importances, top_features
    """
    metrics_path = Path("backend/ml/saved_models/training_metrics.json")

    if not metrics_path.exists():
        logger.warning(
            f"[predict] Training metrics not found at {metrics_path}. "
            f"Train the model first: python -m backend.ml.train_model"
        )
        return None

    try:
        with open(metrics_path, "r") as fh:
            metrics = json.load(fh)
        logger.debug(f"[predict] Loaded training metrics: F1={metrics.get('f1_score')}")
        return metrics
    except Exception as exc:
        logger.error(f"[predict] Failed to load metrics: {exc}")
        return None


def get_model_info() -> Dict[str, Any]:
    """
    Return metadata about the currently loaded model.
    Useful for the /api/risk/model/metrics endpoint.
    """
    metrics = get_training_metrics()
    model_path = Path(settings.model_path)

    return {
        "model_type":    "XGBoost",
        "model_loaded":  is_model_loaded(),
        "model_path":    str(model_path),
        "model_exists":  model_path.exists(),
        "feature_count": len(FEATURE_COLUMNS),
        "features":      FEATURE_COLUMNS,
        "risk_thresholds": {
            "LOW":      "0 – 34.9%",
            "MODERATE": "35 – 54.9%",
            "HIGH":     "55 – 74.9%",
            "CRITICAL": "75 – 100%",
        },
        "training_metrics": metrics or {},
    }