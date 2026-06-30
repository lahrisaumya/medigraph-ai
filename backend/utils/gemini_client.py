"""
backend/utils/gemini_client.py
Purpose: Wrapper around Google Gemini API using the NEW google-genai package.
         The old google.generativeai package is deprecated as of 2025.
         This version uses google.genai (google-genai package).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ── Client initialisation (new SDK style) ─────────────────────────────────────
_client: Optional[genai.Client] = None
_MODEL  = "gemini-1.5-flash"


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
        logger.info(f"✅ [Gemini] Client initialised — model: {_MODEL}")
    return _client


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _parse_json_safe(raw: str, default: Any) -> Any:
    try:
        return json.loads(_strip_json_fences(raw))
    except json.JSONDecodeError as e:
        logger.warning(f"[Gemini] JSON parse failed: {e}. Raw: {raw[:200]}")
        return default


async def _generate(prompt: str, temperature: float = 0.2, max_tokens: int = 1500) -> str:
    """Core async generation using new google-genai SDK."""
    client = _get_client()
    try:
        response = await client.aio.models.generate_content(
            model=_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text.strip()
    except Exception as exc:
        logger.warning(f"[Gemini] First attempt failed: {exc}. Retrying...")
        try:
            response = await client.aio.models.generate_content(
                model=_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text.strip()
        except Exception as exc2:
            logger.error(f"[Gemini] Both attempts failed: {exc2}")
            raise


# ── Public functions ──────────────────────────────────────────────────────────

async def extract_medical_entities(text: str) -> Dict[str, Any]:
    """Extract structured clinical entities from raw document text."""
    truncated = text[:8000] if len(text) > 8000 else text
    prompt = f"""You are a clinical NLP expert. Extract healthcare entities from the medical document below.

Return ONLY a valid JSON object with EXACTLY these keys:
{{
  "diseases":     ["list of disease names, lowercase"],
  "symptoms":     ["list of symptom names, lowercase"],
  "medications":  ["list of drug names, lowercase"],
  "lab_tests":    ["list of lab test names, lowercase"],
  "lab_values":   {{"test_name": "value with unit"}},
  "risk_factors": ["list of risk factors"],
  "dosages":      {{"medication_name": "dosage string"}},
  "instructions": ["list of clinical instructions"]
}}

Return ONLY the JSON. No markdown. No explanation.

Medical Document:
\"\"\"
{truncated}
\"\"\"
"""
    _default = {
        "diseases": [], "symptoms": [], "medications": [],
        "lab_tests": [], "lab_values": {}, "risk_factors": [],
        "dosages": {}, "instructions": [],
    }
    try:
        raw    = await _generate(prompt, temperature=0.1, max_tokens=1800)
        result = _parse_json_safe(raw, _default)
        for key in ("diseases", "symptoms", "medications", "lab_tests", "risk_factors", "instructions"):
            if not isinstance(result.get(key), list):
                result[key] = []
            result[key] = [str(v).lower().strip() for v in result[key] if v]
        for key in ("lab_values", "dosages"):
            if not isinstance(result.get(key), dict):
                result[key] = {}
        return result
    except Exception as exc:
        logger.error(f"[Gemini/extract] Failed: {exc}")
        return _default


async def summarize_medical_document(text: str, doc_type: str = "prescription") -> str:
    """Generate a 3-4 sentence clinical summary of a healthcare document."""
    truncated = text[:3500] if len(text) > 3500 else text
    guidance  = {
        "prescription":   "key diagnoses, medications prescribed with doses, and follow-up instructions",
        "lab_report":     "abnormal values, reference ranges, and clinical significance",
        "medical_summary":"primary diagnosis, treatment plan, findings, and recommendations",
    }.get(doc_type, "key clinical findings and action items")

    prompt = f"""Write a professional 3-4 sentence clinical summary of this {doc_type}.
Focus on: {guidance}. Use clinical language. Reference actual values.
No bullet points, no headers.

Document:
\"\"\"
{truncated}
\"\"\"

Summary:"""
    try:
        return await _generate(prompt, temperature=0.3, max_tokens=350)
    except Exception as exc:
        logger.error(f"[Gemini/summary] Failed: {exc}")
        return f"{doc_type.replace('_',' ').title()} processed. Clinical details extracted."


async def explain_risk_score(
    patient_data: Dict[str, Any],
    risk_score: float,
    risk_factors: List[str],
) -> str:
    """Generate a 2-3 sentence clinical explanation of the risk score."""
    risk_level = (
        "CRITICAL" if risk_score >= 75 else
        "HIGH"     if risk_score >= 55 else
        "MODERATE" if risk_score >= 35 else "LOW"
    )
    conditions  = patient_data.get("conditions",  []) or []
    medications = patient_data.get("medications", []) or []

    prompt = f"""You are a clinical AI explaining a medication adherence risk score to a physician.

Patient:
- Age: {patient_data.get('age', 'N/A')}
- Conditions: {', '.join(conditions) or 'None recorded'}
- Medications: {', '.join(medications) or 'None recorded'}
- Adherence: {patient_data.get('adherence_rate', 'N/A')}%
- Exercise Level: {patient_data.get('exercise_level', 'N/A')}/10
- HbA1c: {patient_data.get('hba1c', 'N/A')}
- Hospital Visits Last Year: {patient_data.get('hospital_visits_last_year', 'N/A')}

Risk Score: {risk_score:.1f}% ({risk_level})
Primary Risk Factors: {'; '.join(risk_factors[:4])}

Write a 2-3 sentence clinical explanation of WHY this patient has this risk score.
Reference specific values. Professional clinical language. No bullet points."""
    try:
        return await _generate(prompt, temperature=0.3, max_tokens=300)
    except Exception as exc:
        logger.error(f"[Gemini/explain_risk] Failed: {exc}")
        top = risk_factors[0] if risk_factors else "multiple clinical indicators"
        return (f"This patient has a {risk_level} risk score of {risk_score:.1f}%, "
                f"primarily driven by {top.lower()}. "
                f"Clinical review and targeted intervention are recommended.")


async def generate_interventions(
    patient_data: Dict[str, Any],
    risk_score: float,
    risk_level: str,
) -> Dict[str, Any]:
    """Generate a structured priority-tiered intervention plan."""
    conditions = patient_data.get("conditions", []) or []
    medications = patient_data.get("medications", []) or []

    prompt = f"""You are a clinical care manager. Create a personalised intervention plan.

Patient:
- Conditions: {', '.join(conditions) or 'None'}
- Medications: {', '.join(medications) or 'None'}
- Adherence: {patient_data.get('adherence_rate', 80)}%
- Risk Score: {risk_score:.1f}% ({risk_level})
- HbA1c: {patient_data.get('hba1c', 'N/A')}
- Exercise Level: {patient_data.get('exercise_level', 'N/A')}/10

Return ONLY valid JSON (no markdown):
{{
  "risk_summary": "1-2 sentence risk context",
  "interventions": [
    {{"priority":"immediate","action":"specific action within 48 hours","rationale":"clinical reason","expected_impact":"measurable outcome"}},
    {{"priority":"short_term","action":"action within 2-4 weeks","rationale":"reason","expected_impact":"improvement"}},
    {{"priority":"long_term","action":"action over 3-6 months","rationale":"reason","expected_impact":"long-term outcome"}}
  ],
  "lifestyle_recommendations": ["change 1","change 2","change 3","change 4"],
  "follow_up_schedule": "specific follow-up recommendation",
  "ai_narrative": "2-3 sentence overall care narrative"
}}"""

    _default = {
        "risk_summary": f"{risk_level} risk ({risk_score:.1f}%). Review recommended.",
        "interventions": [
            {"priority":"immediate","action":"Schedule physician review",
             "rationale":f"Risk score {risk_score:.1f}% requires assessment",
             "expected_impact":"Early intervention"},
            {"priority":"short_term","action":"Enrol in adherence support programme",
             "rationale":"Adherence is primary modifiable risk factor",
             "expected_impact":"Target ≥85% adherence within 4 weeks"},
            {"priority":"long_term","action":"Structured lifestyle modification",
             "rationale":"Diet and exercise reduce long-term risk",
             "expected_impact":"10-15% risk reduction over 6 months"},
        ],
        "lifestyle_recommendations": [
            "Take medications at the same time daily",
            "Use a pill organiser or reminder app",
            "Walk 30 minutes at least 5 days per week",
            "Follow dietary guidelines for your conditions",
        ],
        "follow_up_schedule": "Monthly for 3 months, then quarterly",
        "ai_narrative": (f"Patient has {risk_level} risk at {risk_score:.1f}%. "
                         f"Focus on medication adherence and regular monitoring."),
    }
    try:
        raw    = await _generate(prompt, temperature=0.4, max_tokens=1500)
        result = _parse_json_safe(raw, _default)
        if not isinstance(result.get("interventions"), list) or not result["interventions"]:
            result["interventions"] = _default["interventions"]
        result.setdefault("follow_up_schedule", _default["follow_up_schedule"])
        result.setdefault("ai_narrative",       _default["ai_narrative"])
        result.setdefault("risk_summary",       _default["risk_summary"])
        result.setdefault("lifestyle_recommendations", _default["lifestyle_recommendations"])
        return result
    except Exception as exc:
        logger.error(f"[Gemini/interventions] Failed: {exc}")
        return _default


async def explain_simulation_scenario(
    scenario_label: str,
    original_risk: float,
    new_risk: float,
    adherence: float,
    exercise: int,
    follow_up: int,
) -> str:
    """Generate a 2-sentence explanation of a What-If simulation result."""
    direction = "decrease" if new_risk < original_risk else "increase"
    change    = abs(new_risk - original_risk)

    prompt = f"""Explain this care simulation result in EXACTLY 2 sentences.

Scenario: "{scenario_label}"
- Adherence: {adherence:.0f}%
- Exercise Level: {exercise}/10
- Follow-up Visits/Year: {follow_up}
- Original Risk: {original_risk:.1f}%
- New Risk: {new_risk:.1f}% ({direction} of {change:.1f} points)

Sentence 1: WHY these changes cause this risk {direction}.
Sentence 2: What this means clinically for the patient.
No bullet points."""
    try:
        return await _generate(prompt, temperature=0.3, max_tokens=200)
    except Exception as exc:
        logger.error(f"[Gemini/simulation] Failed: {exc}")
        return (f"Under the '{scenario_label}' scenario, changing adherence to {adherence:.0f}% "
                f"leads to a {direction} in risk by {change:.1f} percentage points. "
                f"This {'improvement' if new_risk < original_risk else 'deterioration'} "
                f"highlights the clinical impact of consistent medication adherence.")


async def generate_drug_safety_narrative(
    drug_name: str,
    adverse_events: List[Dict[str, Any]],
    warnings: List[str],
) -> str:
    """Generate a plain-language patient safety summary for a drug."""
    top_ae = [ae.get("term","") for ae in adverse_events[:5] if ae.get("term")]
    warning_text = warnings[0][:200] if warnings else "None identified"

    prompt = f"""Write a 2-sentence drug safety summary for a healthcare professional.

Drug: {drug_name}
Top adverse effects (FDA FAERS): {', '.join(top_ae) if top_ae else 'No significant data'}
Key FDA warning: {warning_text}

Sentence 1: Most clinically significant adverse effects to monitor.
Sentence 2: Specific actionable safety recommendation.
Factual and professional. No bullet points."""
    try:
        return await _generate(prompt, temperature=0.3, max_tokens=200)
    except Exception as exc:
        logger.error(f"[Gemini/drug_safety] Failed: {exc}")
        if top_ae:
            return (f"Common adverse effects for {drug_name} include {', '.join(top_ae[:3])}. "
                    f"Patients should report any new or worsening symptoms to their physician promptly.")
        return (f"Always take {drug_name} exactly as prescribed. "
                f"Report unexpected side effects to your healthcare provider immediately.")