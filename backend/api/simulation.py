"""
medigraph-ai/backend/api/simulation.py
FILE:    backend/api/simulation.py
PURPOSE: What-If Care Simulation Engine API.
         Predicts patient health outcomes under different care scenarios by
         re-running the XGBoost model with overridden behavioral parameters.
         Gemini generates a plain-language explanation for each scenario.

ENDPOINTS:
    POST /api/simulation/run               — Run what-if simulation
    POST /api/simulation/quick             — Quick single-scenario prediction
    GET  /api/simulation/scenarios/default — Default scenario templates
    GET  /api/simulation/patient/{id}      — Past simulations for a patient
    GET  /api/simulation/patient/{id}/best — Best achievable scenario

HOW IT WORKS:
    1. Accept patient baseline clinical data
    2. Build N scenarios (default 3: Current / Improved / Poor)
    3. For each scenario: override adherence, exercise, follow-up in features
    4. Run XGBoost → risk_score
    5. Call Gemini → 2-sentence AI explanation of the change
    6. Return all results with outcome narratives
    7. Persist simulation record to MongoDB

DEPENDENCIES:
    backend.ml.predict           → predict_risk() [synchronous XGBoost inference]
    backend.utils.gemini_client  → explain_simulation_scenario()
    backend.db.mongodb           → save_simulation()
    backend.core.schemas         → SimulationRequest, SimulationResponse …
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.core.schemas import (
    SimulationRequest,
    SimulationResponse,
    SimulationScenario,
    ScenarioResult,
    RiskLevel,
    RiskPredictionRequest,
    APIResponse,
)
from backend.ml.predict import predict_risk
from backend.utils.gemini_client import explain_simulation_scenario
from backend.db.mongodb import save_simulation, get_db

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/simulation",
    tags=["What-If Simulation"],
)


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/simulation/run
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/run",
    response_model=SimulationResponse,
    summary="Run a What-If Care Simulation for a patient",
)
async def run_simulation(request: SimulationRequest):
    """
    Simulate patient health outcomes under different care scenarios.

    **Default scenarios (auto-generated if not provided):**

    | Label              | Adherence Change | Exercise Change | Follow-up Change |
    |--------------------|-----------------|-----------------|-----------------|
    | Current Behavior   | ±0              | ±0              | ±0               |
    | Improved Adherence | +20%            | +2 pts          | +4 visits/yr    |
    | Poor Adherence     | −25%            | −2 pts          | −2 visits/yr    |

    **Custom scenarios** can be passed in the request body.

    **For each scenario the system returns:**
    - Predicted risk score (XGBoost)
    - Risk level classification
    - Predicted health outcome narrative
    - AI explanation of why the change occurs (Gemini)

    **Use cases:**
    - Clinician planning: "What if I increase follow-up from 4 to 12 visits/year?"
    - Patient education: "What happens if you stop taking your medication?"
    - Intervention ROI: Quantify the risk reduction of a programme
    """
    base = request.base_data

    # ── Build base feature dict ───────────────────────────────────────────
    base_features = _extract_base_features(base)

    # ── Baseline prediction ───────────────────────────────────────────────
    logger.info(
        f"[simulation/run] patient={request.patient_id} | "
        f"baseline adherence={base.adherence_rate}%"
    )

    try:
        baseline_pred = predict_risk(base_features)
        baseline_risk = baseline_pred["risk_score"]
    except Exception as exc:
        logger.error(f"[simulation/run] Baseline prediction failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Baseline prediction failed: {str(exc)}")

    logger.info(f"[simulation/run] Baseline risk: {baseline_risk:.1f}%")

    # ── Build scenario list ───────────────────────────────────────────────
    scenarios: List[SimulationScenario] = (
        request.scenarios if request.scenarios
        else _generate_default_scenarios(base)
    )

    # ── Run each scenario ─────────────────────────────────────────────────
    scenario_results: List[ScenarioResult] = []

    for scenario in scenarios:
        result = await _run_single_scenario(
            scenario      = scenario,
            base_features = base_features,
            baseline_risk = baseline_risk,
        )
        scenario_results.append(result)
        logger.info(
            f"[simulation/run] Scenario '{scenario.label}': "
            f"{result.risk_score:.1f}% ({result.risk_level})"
        )

    # ── Overall recommendation ────────────────────────────────────────────
    recommendation = _build_overall_recommendation(baseline_risk, scenario_results)

    # ── Persist simulation to MongoDB ─────────────────────────────────────
    try:
        await save_simulation({
            "patient_id":    request.patient_id,
            "baseline_risk": baseline_risk,
            "scenarios":     [s.model_dump() for s in scenario_results],
            "recommendation": recommendation,
        })
    except Exception as exc:
        logger.warning(f"[simulation/run] MongoDB persist failed (non-fatal): {exc}")

    return SimulationResponse(
        patient_id=request.patient_id,
        baseline_risk=baseline_risk,
        scenarios=scenario_results,
        recommendation=recommendation,
        simulated_at=datetime.utcnow(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/simulation/quick
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/quick",
    response_model=APIResponse,
    summary="Quick single-point prediction — how does one parameter change affect risk?",
)
async def quick_simulation(
    patient_id: str,
    base_adherence: float = Query(..., ge=0, le=100, description="Current adherence %"),
    new_adherence:  float = Query(..., ge=0, le=100, description="Hypothetical adherence %"),
    age: int              = Query(default=50, ge=1, le=120),
    comorbidity_count: int = Query(default=1, ge=0),
    medication_count: int  = Query(default=1, ge=1),
    exercise_level: int    = Query(default=5, ge=1, le=10),
    follow_up_frequency: int = Query(default=4, ge=0),
    hospital_visits: int   = Query(default=0, ge=0),
    hba1c: Optional[float] = Query(default=None),
    bmi: Optional[float]   = Query(default=None),
):
    """
    Fast single-comparison endpoint: current adherence vs one hypothetical value.

    Useful for the frontend slider ("drag adherence to 90% — what happens?").
    Does not call Gemini — pure XGBoost inference, sub-second response.
    """
    base_features = {
        "age":                       age,
        "adherence_rate":            base_adherence,
        "comorbidity_count":         comorbidity_count,
        "medication_count":          medication_count,
        "exercise_level":            exercise_level,
        "follow_up_frequency":       follow_up_frequency,
        "hospital_visits_last_year": hospital_visits,
        "hba1c":                     hba1c,
        "bmi":                       bmi,
        "conditions":                [],
        "medications":               [],
    }

    try:
        current_pred  = predict_risk(base_features)
        modified_pred = predict_risk({**base_features, "adherence_rate": new_adherence})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    current_score  = current_pred["risk_score"]
    modified_score = modified_pred["risk_score"]
    delta          = round(modified_score - current_score, 1)
    pct_change     = round((delta / current_score) * 100, 1) if current_score else 0

    return APIResponse(
        message="Quick simulation complete",
        data={
            "patient_id":      patient_id,
            "current_score":   current_score,
            "current_level":   current_pred["risk_level"],
            "modified_score":  modified_score,
            "modified_level":  modified_pred["risk_level"],
            "delta":           delta,
            "percent_change":  pct_change,
            "direction":       "improving" if delta < 0 else "worsening" if delta > 0 else "unchanged",
            "base_adherence":  base_adherence,
            "new_adherence":   new_adherence,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/simulation/scenarios/default
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/scenarios/default",
    response_model=APIResponse,
    summary="Return default scenario templates for the frontend",
)
async def get_default_scenario_templates():
    """
    Returns the 4 built-in scenario templates with their delta values.
    The frontend uses these to pre-populate the simulation form.
    Each delta is relative to the patient's current baseline values.
    """
    templates = [
        {
            "label":           "Current Behavior",
            "description":     "Maintaining current habits with no change",
            "adherence_delta": 0,
            "exercise_delta":  0,
            "followup_delta":  0,
            "color":           "#6B7280",
            "icon":            "→",
        },
        {
            "label":           "Improved Adherence",
            "description":     "Achieves ≥90% medication adherence with pharmacist support",
            "adherence_delta": +20,
            "exercise_delta":  +2,
            "followup_delta":  +4,
            "color":           "#10B981",
            "icon":            "↑",
        },
        {
            "label":           "Poor Adherence",
            "description":     "Adherence declines due to side effects or barriers",
            "adherence_delta": -25,
            "exercise_delta":  -2,
            "followup_delta":  -2,
            "color":           "#EF4444",
            "icon":            "↓",
        },
        {
            "label":           "Lifestyle Overhaul",
            "description":     "Joins a structured diet, exercise, and adherence programme",
            "adherence_delta": +15,
            "exercise_delta":  +4,
            "followup_delta":  +6,
            "color":           "#3B82F6",
            "icon":            "★",
        },
    ]

    return APIResponse(
        message="Default scenario templates",
        data={"scenarios": templates, "count": len(templates)},
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/simulation/patient/{patient_id}
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/patient/{patient_id}",
    response_model=APIResponse,
    summary="Get all past simulations for a patient",
)
async def get_patient_simulations(
    patient_id: str,
    limit: int = Query(default=5, ge=1, le=20),
):
    """
    Retrieve the simulation history for a patient — useful for comparing
    how care planning has evolved over time.
    """
    try:
        db = get_db()
        cursor = (
            db.simulations
            .find({"patient_id": patient_id})
            .sort("simulated_at", -1)
            .limit(limit)
        )
        sims = []
        async for doc in cursor:
            doc["id"] = str(doc.pop("_id"))
            sims.append(doc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return APIResponse(
        message=f"Found {len(sims)} simulation(s) for patient '{patient_id}'",
        data={"patient_id": patient_id, "simulations": sims, "count": len(sims)},
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/simulation/patient/{patient_id}/best
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/patient/{patient_id}/best",
    response_model=APIResponse,
    summary="Find the best achievable risk score across parameter space",
)
async def find_best_scenario(
    patient_id: str,
    age: int              = Query(..., ge=1, le=120),
    comorbidity_count: int = Query(default=1, ge=0),
    medication_count: int  = Query(default=1, ge=1),
    hospital_visits: int   = Query(default=0, ge=0),
    hba1c: Optional[float] = Query(default=None),
    bmi: Optional[float]   = Query(default=None),
    current_adherence: float = Query(default=70.0, ge=0, le=100),
    current_exercise: int    = Query(default=4, ge=1, le=10),
    current_followup: int    = Query(default=4, ge=0),
):
    """
    Compute the theoretical minimum risk score if all modifiable factors
    are optimised: adherence=95%, exercise=9, follow_up=12.

    Also returns the risk reduction gap between current and optimal.
    Used for the 'Potential Improvement' panel on the Simulation dashboard.
    """
    base = {
        "age":                       age,
        "comorbidity_count":         comorbidity_count,
        "medication_count":          medication_count,
        "hospital_visits_last_year": hospital_visits,
        "hba1c":                     hba1c,
        "bmi":                       bmi,
        "conditions":                [],
        "medications":               [],
    }

    try:
        current_pred = predict_risk({
            **base,
            "adherence_rate":      current_adherence,
            "exercise_level":      current_exercise,
            "follow_up_frequency": current_followup,
        })
        optimal_pred = predict_risk({
            **base,
            "adherence_rate":      95.0,
            "exercise_level":      9,
            "follow_up_frequency": 12,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    current_score = current_pred["risk_score"]
    optimal_score = optimal_pred["risk_score"]
    gap           = round(current_score - optimal_score, 1)

    return APIResponse(
        message="Best achievable scenario computed",
        data={
            "patient_id":         patient_id,
            "current_risk":       current_score,
            "current_level":      current_pred["risk_level"],
            "optimal_risk":       optimal_score,
            "optimal_level":      optimal_pred["risk_level"],
            "reduction_possible": gap,
            "optimal_parameters": {
                "adherence_rate":      95.0,
                "exercise_level":      9,
                "follow_up_frequency": 12,
            },
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _extract_base_features(base: RiskPredictionRequest) -> dict:
    """Convert RiskPredictionRequest to the dict expected by predict_risk()."""
    return {
        "age":                       base.age,
        "adherence_rate":            base.adherence_rate,
        "comorbidity_count":         base.comorbidity_count,
        "medication_count":          base.medication_count,
        "exercise_level":            base.exercise_level,
        "follow_up_frequency":       base.follow_up_frequency,
        "hospital_visits_last_year": base.hospital_visits_last_year,
        "hba1c":                     base.hba1c,
        "bmi":                       base.bmi,
        "conditions":                [],
        "medications":               [],
    }


def _generate_default_scenarios(base: RiskPredictionRequest) -> List[SimulationScenario]:
    """Generate 3 standard scenarios: Current / Improved / Poor."""
    return [
        SimulationScenario(
            label="Current Behavior",
            adherence_rate=base.adherence_rate,
            exercise_level=base.exercise_level,
            follow_up_frequency=base.follow_up_frequency,
            description="Maintaining current habits without any intervention",
        ),
        SimulationScenario(
            label="Improved Adherence",
            adherence_rate=min(95.0, base.adherence_rate + 20),
            exercise_level=min(10,   base.exercise_level  + 2),
            follow_up_frequency=min(24, base.follow_up_frequency + 4),
            description="Improved medication adherence and increased physical activity",
        ),
        SimulationScenario(
            label="Poor Adherence",
            adherence_rate=max(10.0, base.adherence_rate  - 25),
            exercise_level=max(1,    base.exercise_level   - 2),
            follow_up_frequency=max(0, base.follow_up_frequency - 2),
            description="Adherence declines without active clinical support",
        ),
    ]


async def _run_single_scenario(
    scenario: SimulationScenario,
    base_features: dict,
    baseline_risk: float,
) -> ScenarioResult:
    """
    Run XGBoost for one scenario and get Gemini explanation.
    Returns a fully populated ScenarioResult.
    """
    # Override scenario-specific parameters
    scenario_features = {
        **base_features,
        "adherence_rate":      scenario.adherence_rate,
        "exercise_level":      scenario.exercise_level,
        "follow_up_frequency": scenario.follow_up_frequency,
    }

    # XGBoost inference
    pred       = predict_risk(scenario_features)
    risk_score = pred["risk_score"]
    risk_level = pred["risk_level"]

    # Gemini explanation (with fallback)
    try:
        ai_explanation = await explain_simulation_scenario(
            scenario_label = scenario.label,
            original_risk  = baseline_risk,
            new_risk       = risk_score,
            adherence      = scenario.adherence_rate,
            exercise       = scenario.exercise_level,
            follow_up      = scenario.follow_up_frequency,
        )
    except Exception as exc:
        logger.warning(f"[simulation] Gemini explanation failed for '{scenario.label}': {exc}")
        direction = "decreases" if risk_score < baseline_risk else "increases"
        delta     = abs(risk_score - baseline_risk)
        ai_explanation = (
            f"Under the '{scenario.label}' scenario, predicted risk {direction} "
            f"by {delta:.1f} percentage points due to changes in adherence and lifestyle."
        )

    return ScenarioResult(
        label=scenario.label,
        description=scenario.description,
        adherence_rate=scenario.adherence_rate,
        exercise_level=scenario.exercise_level,
        follow_up_frequency=scenario.follow_up_frequency,
        risk_score=risk_score,
        risk_level=RiskLevel(risk_level),
        predicted_outcome=_outcome_narrative(risk_score),
        ai_explanation=ai_explanation,
    )


def _outcome_narrative(risk_score: float) -> str:
    """Map a risk score to a clinical outcome plain-text statement."""
    if risk_score < 25:
        return "✅ Excellent — Very low likelihood of adverse health events"
    elif risk_score < 40:
        return "🟢 Good — Stable health trajectory expected"
    elif risk_score < 55:
        return "🟡 Moderate — Increased monitoring recommended"
    elif risk_score < 70:
        return "🟠 High — Likely deterioration without active intervention"
    elif risk_score < 85:
        return "🔴 Critical — High hospital readmission risk"
    else:
        return "🚨 Severe — Immediate clinical intervention required"


def _build_overall_recommendation(
    baseline_risk: float,
    results: List[ScenarioResult],
) -> str:
    """Generate a one-paragraph overall simulation recommendation."""
    if not results:
        return "No scenarios to compare."

    best  = min(results, key=lambda x: x.risk_score)
    worst = max(results, key=lambda x: x.risk_score)
    span  = round(worst.risk_score - best.risk_score, 1)

    return (
        f"The simulation reveals a {span:.1f}% risk differential across all scenarios. "
        f"The '{best.label}' scenario achieves the lowest predicted risk at "
        f"{best.risk_score:.1f}% (vs current baseline of {baseline_risk:.1f}%). "
        f"Prioritising medication adherence and increasing follow-up frequency "
        f"delivers the greatest measurable risk reduction for this patient."
    )

