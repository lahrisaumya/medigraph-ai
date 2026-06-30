"""
================================================================================
FILE:    backend/ml/features.py
PURPOSE: Feature engineering pipeline for the XGBoost medication adherence
         risk model.  Converts raw patient data dicts into a model-ready
         numeric DataFrame with exactly 17 features.

FEATURE SET (17 features in FEATURE_COLUMNS order):
    Continuous (7):
        age, adherence_rate, comorbidity_count, medication_count,
        exercise_level, follow_up_frequency, hospital_visits_last_year

    Normalised lab values (2):
        hba1c_normalized   — HbA1c % ÷ 14  (capped at 1.0)
        bmi_normalized     — BMI ÷ 50      (capped at 1.0)

    One-hot age groups (3):
        age_group_young    — age < 40
        age_group_middle   — 40 ≤ age < 65
        age_group_senior   — age ≥ 65

    Binary risk flags (2):
        low_adherence_flag      — adherence_rate < 70
        high_comorbidity_flag   — comorbidity_count ≥ 3

    Binary disease flags (3):
        diabetes_flag      — any condition contains 'diabet'
        hypertension_flag  — any condition contains 'hypertension' or 'blood pressure'
        cardiac_flag       — any condition contains 'heart', 'cardiac', or 'coronary'

SAFE DEFAULTS:
    All optional fields (hba1c, bmi, conditions) fall back to clinically
    neutral values so inference never fails on incomplete patient data.

USAGE:
    from backend.ml.features import engineer_features, identify_risk_factors

    df = engineer_features(patient_dict)          # returns shape (1, 17)
    factors = identify_risk_factors(patient_dict)  # returns List[str]
================================================================================
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMN ORDER  — must match training exactly
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_COLUMNS: List[str] = [
    # Continuous
    "age",
    "adherence_rate",
    "comorbidity_count",
    "medication_count",
    "exercise_level",
    "follow_up_frequency",
    "hospital_visits_last_year",
    # Normalised lab values
    "hba1c_normalized",
    "bmi_normalized",
    # Age group one-hot
    "age_group_young",
    "age_group_middle",
    "age_group_senior",
    # Risk flags
    "low_adherence_flag",
    "high_comorbidity_flag",
    # Disease flags
    "diabetes_flag",
    "hypertension_flag",
    "cardiac_flag",
]

# Safe defaults for missing / None fields
_DEFAULTS: Dict[str, float] = {
    "age":                       50,
    "adherence_rate":            80.0,
    "comorbidity_count":         0,
    "medication_count":          1,
    "exercise_level":            5,
    "follow_up_frequency":       4,
    "hospital_visits_last_year": 0,
    "hba1c":                     5.5,   # normal HbA1c
    "bmi":                       25.0,  # healthy BMI midpoint
}

# Normalisation denominators (values above these are capped at 1.0)
_HBA1C_MAX = 14.0
_BMI_MAX   = 50.0


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — SINGLE PATIENT
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(patient_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert one patient data dict into a single-row feature DataFrame
    ready for XGBoost inference or StandardScaler transformation.

    Args:
        patient_data: Any subset of the following keys (all optional with defaults):
            age, adherence_rate, comorbidity_count, medication_count,
            exercise_level, follow_up_frequency, hospital_visits_last_year,
            hba1c, bmi, conditions (List[str])

    Returns:
        pd.DataFrame of shape (1, 17) with columns in FEATURE_COLUMNS order.
        All values are numeric (int or float), no NaNs.

    Raises:
        Nothing — all exceptions are caught and defaults applied.
    """
    try:
        # ── Scalar clinical features ──────────────────────────────────────
        age                       = _safe_int(patient_data.get("age"),                       _DEFAULTS["age"])
        adherence_rate            = _safe_float(patient_data.get("adherence_rate"),           _DEFAULTS["adherence_rate"])
        comorbidity_count         = _safe_int(patient_data.get("comorbidity_count"),          _DEFAULTS["comorbidity_count"])
        medication_count          = _safe_int(patient_data.get("medication_count"),           _DEFAULTS["medication_count"])
        exercise_level            = _safe_int(patient_data.get("exercise_level"),             _DEFAULTS["exercise_level"])
        follow_up_frequency       = _safe_int(patient_data.get("follow_up_frequency"),        _DEFAULTS["follow_up_frequency"])
        hospital_visits_last_year = _safe_int(patient_data.get("hospital_visits_last_year"),  _DEFAULTS["hospital_visits_last_year"])

        # ── Lab values (with safe defaults for missing) ───────────────────
        hba1c = _safe_float(patient_data.get("hba1c"), _DEFAULTS["hba1c"])
        bmi   = _safe_float(patient_data.get("bmi"),   _DEFAULTS["bmi"])

        # ── Conditions: normalise to lowercase list ───────────────────────
        raw_conditions = patient_data.get("conditions") or []
        if isinstance(raw_conditions, str):
            raw_conditions = [raw_conditions]
        conditions = [str(c).lower().strip() for c in raw_conditions if c]

        # ── Derived / engineered features ─────────────────────────────────

        # Normalised lab values (0 – 1 scale)
        hba1c_normalized = min(hba1c / _HBA1C_MAX, 1.0)
        bmi_normalized   = min(bmi   / _BMI_MAX,   1.0)

        # Age group one-hot (mutually exclusive)
        age_group_young  = 1 if age < 40           else 0
        age_group_middle = 1 if 40 <= age < 65     else 0
        age_group_senior = 1 if age >= 65           else 0

        # Binary risk flags
        low_adherence_flag    = 1 if adherence_rate    < 70 else 0
        high_comorbidity_flag = 1 if comorbidity_count >= 3  else 0

        # Disease presence flags (substring matching)
        diabetes_flag     = 1 if _has_condition(conditions, ["diabet"])                              else 0
        hypertension_flag = 1 if _has_condition(conditions, ["hypertension", "blood pressure"])      else 0
        cardiac_flag      = 1 if _has_condition(conditions, ["heart", "cardiac", "coronary"])        else 0

        # ── Assemble row dict ─────────────────────────────────────────────
        row = {
            "age":                       age,
            "adherence_rate":            adherence_rate,
            "comorbidity_count":         comorbidity_count,
            "medication_count":          medication_count,
            "exercise_level":            exercise_level,
            "follow_up_frequency":       follow_up_frequency,
            "hospital_visits_last_year": hospital_visits_last_year,
            "hba1c_normalized":          hba1c_normalized,
            "bmi_normalized":            bmi_normalized,
            "age_group_young":           age_group_young,
            "age_group_middle":          age_group_middle,
            "age_group_senior":          age_group_senior,
            "low_adherence_flag":        low_adherence_flag,
            "high_comorbidity_flag":     high_comorbidity_flag,
            "diabetes_flag":             diabetes_flag,
            "hypertension_flag":         hypertension_flag,
            "cardiac_flag":              cardiac_flag,
        }

        # Return as DataFrame with columns in canonical order
        return pd.DataFrame([row])[FEATURE_COLUMNS]

    except Exception as exc:
        logger.error(f"[features] engineer_features error: {exc}", exc_info=True)
        # Return all-zero safe fallback (model will return near-50% probability)
        return pd.DataFrame([{col: 0 for col in FEATURE_COLUMNS}])[FEATURE_COLUMNS]


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — BATCH
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features_batch(patients: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of patient dicts into a multi-row feature DataFrame.
    Each row corresponds to one patient in the same order as the input list.

    Args:
        patients: List of patient data dicts.

    Returns:
        pd.DataFrame of shape (N, 17).
    """
    if not patients:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    frames = [engineer_features(p) for p in patients]
    result = pd.concat(frames, ignore_index=True)
    logger.debug(f"[features] Batch engineered: {len(result)} rows × {len(result.columns)} cols")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# RISK FACTOR IDENTIFICATION (rule-based, for explanation)
# ──────────────────────────────────────────────────────────────────────────────

def identify_risk_factors(patient_data: Dict[str, Any]) -> List[str]:
    """
    Apply clinical decision rules to identify the patient's key risk drivers.
    Returns a human-readable list of risk factor strings used in the
    Gemini explanation prompt and the dashboard Risk Factors panel.

    Rules are ordered from highest clinical impact to lowest.
    If no individual factors are flagged, returns a composite risk note.

    Args:
        patient_data: Patient feature dict (same format as engineer_features input).

    Returns:
        List of plain-English risk factor strings (1 – 8 items typically).
    """
    factors: List[str] = []

    adherence    = _safe_float(patient_data.get("adherence_rate"),           80.0)
    age          = _safe_int(patient_data.get("age"),                         50)
    comorbidities= _safe_int(patient_data.get("comorbidity_count"),            0)
    meds         = _safe_int(patient_data.get("medication_count"),             1)
    exercise     = _safe_int(patient_data.get("exercise_level"),               5)
    follow_up    = _safe_int(patient_data.get("follow_up_frequency"),          4)
    hosp_visits  = _safe_int(patient_data.get("hospital_visits_last_year"),    0)
    hba1c        = _safe_float(patient_data.get("hba1c"),                    None)  # None = not measured
    bmi          = _safe_float(patient_data.get("bmi"),                      None)

    conditions   = [str(c).lower() for c in (patient_data.get("conditions") or [])]

    # ── Adherence (highest weight feature) ───────────────────────────────
    if adherence < 50:
        factors.append(f"⚠️ Critical medication adherence ({adherence:.0f}%) — severe non-compliance")
    elif adherence < 65:
        factors.append(f"Poor medication adherence ({adherence:.0f}%) — significant gap from target ≥85%")
    elif adherence < 75:
        factors.append(f"Below-target adherence ({adherence:.0f}%) — monitoring required")

    # ── HbA1c / glycaemic control ─────────────────────────────────────────
    if hba1c is not None:
        if hba1c > 9.0:
            factors.append(f"Severely elevated HbA1c ({hba1c:.1f}%) — very poor glycaemic control")
        elif hba1c > 7.5:
            factors.append(f"Elevated HbA1c ({hba1c:.1f}%) — suboptimal glycaemic control")
        elif hba1c > 7.0:
            factors.append(f"Borderline HbA1c ({hba1c:.1f}%) — target is <7.0%")

    # ── Comorbidity burden ────────────────────────────────────────────────
    if comorbidities >= 5:
        factors.append(f"Very high comorbidity burden ({comorbidities} concurrent conditions)")
    elif comorbidities >= 3:
        factors.append(f"High comorbidity count ({comorbidities} conditions) — polypharmacy risk")

    # ── Hospitalisation frequency ─────────────────────────────────────────
    if hosp_visits >= 4:
        factors.append(f"Frequent hospitalisations ({hosp_visits} in past year) — indicates poor disease control")
    elif hosp_visits >= 2:
        factors.append(f"Multiple hospitalisations ({hosp_visits} in past year)")

    # ── Age ───────────────────────────────────────────────────────────────
    if age >= 75:
        factors.append(f"Advanced age ({age} years) — elevated baseline risk and polypharmacy complexity")
    elif age >= 65:
        factors.append(f"Senior age group ({age} years) — higher adherence barrier risk")

    # ── Polypharmacy ──────────────────────────────────────────────────────
    if meds >= 6:
        factors.append(f"Polypharmacy ({meds} medications) — complexity increases non-adherence likelihood")

    # ── Physical inactivity ───────────────────────────────────────────────
    if exercise <= 1:
        factors.append("Sedentary lifestyle (exercise level 1/10) — independent cardiovascular risk factor")
    elif exercise <= 2:
        factors.append(f"Low physical activity (exercise level {exercise}/10)")

    # ── Follow-up gaps ────────────────────────────────────────────────────
    if follow_up == 0:
        factors.append("No scheduled follow-up visits — critical care gap")
    elif follow_up == 1:
        factors.append(f"Very infrequent follow-up ({follow_up} visit/year) — insufficient monitoring")
    elif follow_up < 3:
        factors.append(f"Infrequent follow-up ({follow_up} visits/year) — below recommended frequency")

    # ── BMI ───────────────────────────────────────────────────────────────
    if bmi is not None:
        if bmi >= 35:
            factors.append(f"Severe obesity (BMI {bmi:.1f}) — compounding metabolic risk")
        elif bmi >= 30:
            factors.append(f"Obesity (BMI {bmi:.1f}) — associated with reduced adherence and worse outcomes")

    # ── Disease-specific flags ────────────────────────────────────────────
    if _has_condition(conditions, ["heart failure"]):
        factors.append("Heart failure — requires strict sodium, fluid, and medication adherence")
    if _has_condition(conditions, ["chronic kidney", "ckd"]):
        factors.append("Chronic kidney disease — requires close medication dose monitoring")

    # ── If no factors flagged ─────────────────────────────────────────────
    if not factors:
        factors.append(
            "No single dominant risk factor — elevated risk reflects composite of multiple "
            "mild-to-moderate clinical indicators"
        )

    return factors


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE STATISTICS (for dashboard / model explainability)
# ──────────────────────────────────────────────────────────────────────────────

def get_feature_statistics(patients: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute basic statistics across a population of patients.
    Used by the dashboard Risk Prediction analytics page.

    Args:
        patients: List of patient data dicts.

    Returns:
        Dict with mean, median, std, min, max per feature.
    """
    if not patients:
        return {}

    df = engineer_features_batch(patients)
    stats = {}
    for col in FEATURE_COLUMNS:
        series = df[col]
        stats[col] = {
            "mean":   round(float(series.mean()),   2),
            "median": round(float(series.median()), 2),
            "std":    round(float(series.std()),    2),
            "min":    round(float(series.min()),    2),
            "max":    round(float(series.max()),    2),
        }
    return stats


def compute_risk_score_heuristic(patient_data: Dict[str, Any]) -> float:
    """
    Fast rule-based risk score (0–100) without the ML model.
    Used as a fallback when the model is not loaded.
    NOT a replacement for XGBoost — accuracy is lower.

    Weights are calibrated to approximate the trained model's outputs.
    """
    adherence    = _safe_float(patient_data.get("adherence_rate"),           80.0)
    age          = _safe_int(patient_data.get("age"),                         50)
    comorbidities= _safe_int(patient_data.get("comorbidity_count"),            0)
    exercise     = _safe_int(patient_data.get("exercise_level"),               5)
    follow_up    = _safe_int(patient_data.get("follow_up_frequency"),          4)
    hosp_visits  = _safe_int(patient_data.get("hospital_visits_last_year"),    0)
    hba1c        = _safe_float(patient_data.get("hba1c"),                    5.5)
    bmi          = _safe_float(patient_data.get("bmi"),                      25.0)

    score = (
        (100 - adherence) * 0.35            # adherence gap (highest weight)
        + max(0, hba1c - 5.5) * 2.5         # HbA1c above normal
        + comorbidities * 3.5               # each comorbidity
        + hosp_visits   * 5.0               # each hospitalisation
        + max(0, age - 40) * 0.3            # age above 40
        + max(0, 5 - exercise) * 2.0        # below median exercise
        + max(0, 4 - follow_up) * 1.5       # below target follow-up
        + max(0, bmi - 25) * 0.5            # BMI above healthy
    )
    return round(min(max(score, 0), 100), 1)


# ──────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _safe_float(value: Any, default: float) -> float:
    """Convert value to float, returning default on None or error."""
    if value is None:
        return float(default)
    try:
        return float(value)
    except (ValueError, TypeError):
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    """Convert value to int, returning default on None or error."""
    if value is None:
        return int(default)
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return int(default)


def _has_condition(conditions: List[str], keywords: List[str]) -> bool:
    """
    Check if any condition in the list matches any of the keywords.
    Case-insensitive substring match.
    """
    return any(
        kw in cond
        for cond in conditions
        for kw  in keywords
    )