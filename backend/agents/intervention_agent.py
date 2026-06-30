"""
================================================================================
FILE:    backend/agents/intervention_agent.py
AGENT:   Agent 4 — Intervention Agent
PURPOSE: Uses the risk score and clinical context produced by Agent 3 to call
         Gemini and generate a fully structured, priority-tiered intervention
         plan with lifestyle recommendations, follow-up schedule, and an overall
         clinical narrative.

         This is the final agent in the LangGraph pipeline. Its output is what
         the dashboard surfaces as "AI Recommendations".

INPUTS  (from LangGraph state — set by Agents 1, 2, 3):
    patient_id        (str)        : patient identifier
    risk_score        (float)      : 0-100 from risk_agent
    risk_level        (str)        : "LOW"|"MODERATE"|"HIGH"|"CRITICAL"
    risk_factors      (list[str])  : top risk drivers from risk_agent
    risk_explanation  (str)        : Gemini narrative from risk_agent
    conditions        (list[str])  : merged active diagnoses
    medications       (list[str])  : merged active medications
    adherence_rate    (float)      : current adherence %
    hba1c             (float|None) : HbA1c value if available
    lab_values        (dict)       : any lab values from extracted_entities
    current_symptoms  (list[str])  : symptoms from extracted_entities or state
    exercise_level    (int)        : 1-10 physical activity level
    follow_up_frequency(int)       : clinic visits per year

OUTPUTS (added to LangGraph state):
    intervention_plan         (dict) : structured plan — see schema below
    intervention_agent_status (str)  : "success" | "error: <message>"

    intervention_plan schema:
    {
        "risk_summary":   str,
        "interventions": [
            {
                "priority":        "immediate"|"short_term"|"long_term",
                "action":          str,
                "rationale":       str,
                "expected_impact": str
            }, ...
        ],
        "lifestyle_recommendations": [str, ...],
        "follow_up_schedule":        str,
        "ai_narrative":              str
    }
================================================================================
"""

import logging
from typing import Dict, Any, List

from backend.utils.gemini_client import generate_interventions

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# AGENT NODE
# ──────────────────────────────────────────────────────────────────────────────

async def intervention_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node — Intervention Agent.

    Workflow:
        1. Assemble the full clinical context from state
        2. Call Gemini → structured JSON intervention plan
        3. Validate and sanitise Gemini output (fill missing keys with safe defaults)
        4. Return updated state

    Error handling:
        On Gemini failure, a hard-coded safe default plan is returned so the
        dashboard always has something meaningful to display.
    """
    patient_id = state.get("patient_id", "UNKNOWN")
    risk_score = state.get("risk_score", 50.0)
    risk_level = state.get("risk_level", "MODERATE")

    logger.info(
        f"[InterventionAgent] ▶ Starting for patient={patient_id} | "
        f"risk_score={risk_score:.1f}% | risk_level={risk_level}"
    )

    try:
        # ── Step 1: Build clinical context ────────────────────────────────────
        patient_data = _build_clinical_context(state)
        _log_context_summary(patient_id, patient_data, risk_score, risk_level)

        # ── Step 2: Generate intervention plan (Gemini) ───────────────────────
        logger.info("[InterventionAgent] Calling Gemini for intervention plan ...")
        plan = await generate_interventions(
            patient_data = patient_data,
            risk_score   = risk_score,
            risk_level   = risk_level,
        )

        # ── Step 3: Validate and sanitise ────────────────────────────────────
        plan = _sanitise_plan(plan, risk_score, risk_level)

        intervention_count = len(plan.get("interventions", []))
        logger.info(
            f"[InterventionAgent] ✅ Plan generated | "
            f"interventions={intervention_count} | "
            f"lifestyle_tips={len(plan.get('lifestyle_recommendations',[]))}"
        )

        return {
            **state,
            "intervention_plan":         plan,
            "intervention_agent_status": "success",
        }

    except Exception as exc:
        logger.error(f"[InterventionAgent] ❌ Error: {exc}", exc_info=True)
        fallback_plan = _fallback_plan(risk_score, risk_level)

        return {
            **state,
            "intervention_plan":         fallback_plan,
            "intervention_agent_status": f"error: {str(exc)}",
        }


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _build_clinical_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge all clinical signals from state into a single dict for Gemini.
    Uses merged conditions/medications (state + document-extracted).
    """
    # Merge conditions
    state_conditions = state.get("conditions", []) or []
    doc_diseases     = state.get("extracted_entities", {}).get("diseases", [])
    merged_conditions = list(dict.fromkeys(state_conditions + doc_diseases))

    # Merge medications
    state_medications = state.get("medications", []) or []
    doc_medications   = state.get("extracted_entities", {}).get("medications", [])
    merged_medications = list(dict.fromkeys(state_medications + doc_medications))

    # Lab values: prefer from extracted_entities, fallback to state
    lab_values = {
        **state.get("extracted_entities", {}).get("lab_values", {}),
        **state.get("lab_values", {}),
    }

    # Symptoms: combine extracted + state
    extracted_symptoms = state.get("extracted_entities", {}).get("symptoms", [])
    state_symptoms     = state.get("current_symptoms", []) or []
    merged_symptoms    = list(dict.fromkeys(extracted_symptoms + state_symptoms))

    return {
        "conditions":        merged_conditions,
        "medications":       merged_medications,
        "adherence_rate":    float(state.get("adherence_rate", 80.0)),
        "hba1c":             state.get("hba1c"),
        "bmi":               state.get("bmi"),
        "age":               int(state.get("age", 50)),
        "exercise_level":    int(state.get("exercise_level", 5)),
        "follow_up_frequency": int(state.get("follow_up_frequency", 4)),
        "lab_values":        lab_values,
        "current_symptoms":  merged_symptoms,
        "risk_factors":      state.get("risk_factors", []),
        "risk_explanation":  state.get("risk_explanation", ""),
    }


def _sanitise_plan(plan: Dict[str, Any], risk_score: float, risk_level: str) -> Dict[str, Any]:
    """
    Ensure the plan returned by Gemini has all required keys.
    Fills in safe defaults for any missing fields so the frontend never breaks.
    """
    # Ensure interventions list exists with correct structure
    if "interventions" not in plan or not isinstance(plan["interventions"], list):
        plan["interventions"] = _default_interventions(risk_level)
    else:
        # Ensure each intervention has all required keys
        for item in plan["interventions"]:
            item.setdefault("priority",        "short_term")
            item.setdefault("action",          "Follow up with physician")
            item.setdefault("rationale",       "Clinical assessment required")
            item.setdefault("expected_impact", "Improved care coordination")

    plan.setdefault("lifestyle_recommendations", [
        "Maintain regular medication schedule",
        "Aim for 30 minutes of moderate exercise daily",
        "Follow prescribed dietary guidelines",
    ])
    plan.setdefault("follow_up_schedule", _default_follow_up(risk_score))
    plan.setdefault("ai_narrative",
                    f"Patient has a {risk_level} risk profile with a score of {risk_score:.1f}%. "
                    f"Prioritising adherence improvement and regular monitoring is recommended.")
    plan.setdefault("risk_summary",
                    f"{risk_level} risk detected ({risk_score:.1f}%). Intervention plan generated.")

    return plan


def _default_interventions(risk_level: str) -> List[Dict[str, str]]:
    """Generate safe default interventions based on risk level."""
    if risk_level in ("HIGH", "CRITICAL"):
        immediate_action = "Schedule urgent physician review within 48 hours"
    elif risk_level == "MODERATE":
        immediate_action = "Schedule physician review within 2 weeks"
    else:
        immediate_action = "Continue current care plan with scheduled follow-up"

    return [
        {
            "priority": "immediate",
            "action": immediate_action,
            "rationale": f"Risk level is {risk_level} — timely clinical assessment is essential",
            "expected_impact": "Early identification and management of risk factors",
        },
        {
            "priority": "short_term",
            "action": "Enrol in medication adherence support programme",
            "rationale": "Low adherence is the primary modifiable risk driver",
            "expected_impact": "Target ≥85% adherence within 4 weeks",
        },
        {
            "priority": "long_term",
            "action": "Develop a structured lifestyle modification plan with dietitian",
            "rationale": "Diet and exercise changes reduce long-term disease burden",
            "expected_impact": "10-15% reduction in risk score over 6 months",
        },
    ]


def _default_follow_up(risk_score: float) -> str:
    """Suggest follow-up frequency based on risk score."""
    if risk_score >= 75:
        return "Weekly check-in for 4 weeks, then fortnightly for 2 months"
    elif risk_score >= 55:
        return "Bi-weekly follow-up for 6 weeks, then monthly review"
    elif risk_score >= 35:
        return "Monthly follow-up for 3 months, then quarterly"
    else:
        return "Quarterly routine review; patient may self-monitor"


def _fallback_plan(risk_score: float, risk_level: str) -> Dict[str, Any]:
    """
    Complete hard-coded fallback plan used when Gemini fails entirely.
    Ensures the dashboard always renders something useful.
    """
    return {
        "risk_summary": (
            f"Automated plan (Gemini unavailable). "
            f"Risk level: {risk_level} ({risk_score:.1f}%)."
        ),
        "interventions": _default_interventions(risk_level),
        "lifestyle_recommendations": [
            "Take medications at the same time each day",
            "Use a pill organiser or medication reminder app",
            "Walk for 30 minutes at least 5 days a week",
            "Reduce sodium intake to under 2g per day",
            "Monitor blood pressure at home weekly",
        ],
        "follow_up_schedule": _default_follow_up(risk_score),
        "ai_narrative": (
            f"This patient shows a {risk_level} adherence risk score of {risk_score:.1f}%. "
            f"Immediate focus should be on improving medication consistency and increasing "
            f"clinical contact frequency to prevent adverse outcomes."
        ),
    }


def _log_context_summary(patient_id: str, ctx: Dict[str, Any],
                          risk_score: float, risk_level: str) -> None:
    """Log compact context summary for observability."""
    logger.info(
        f"[InterventionAgent] Context for patient={patient_id} | "
        f"risk={risk_score:.1f}% ({risk_level}) | "
        f"conditions={len(ctx.get('conditions',[]))} | "
        f"medications={len(ctx.get('medications',[]))} | "
        f"symptoms={len(ctx.get('current_symptoms',[]))} | "
        f"adherence={ctx.get('adherence_rate',80):.0f}%"
    )