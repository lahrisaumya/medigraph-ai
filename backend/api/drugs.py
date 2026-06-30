"""
medigraph-ai/backend/api/drugs.py
FILE:    backend/api/drugs.py
PURPOSE: Drug Safety Center — OpenFDA API integration.
         Fetches real-time drug adverse events, label warnings,
         contraindications, recalls, and drug-drug interaction signals.
         Gemini enriches raw FDA data with a plain-language safety narrative.

ENDPOINTS:
    GET  /api/drugs/search           — Drug safety profile (adverse events + label)
    GET  /api/drugs/interactions     — Drug-drug interaction signal (co-reports)
    GET  /api/drugs/recalls          — Drug recall history
    GET  /api/drugs/ndc/{ndc}        — Lookup by NDC product code
    POST /api/drugs/patient-check    — Check all of a patient's medications

DATA SOURCE:
    OpenFDA (https://open.fda.gov) — No API key required.
    - /drug/event.json   → adverse event reports (FAERS database)
    - /drug/label.json   → FDA drug labels (warnings, contraindications, …)
    - /drug/enforcement.json → drug recalls

DEPENDENCIES:
    httpx                       → async HTTP client for OpenFDA
    backend.utils.gemini_client → generate_drug_safety_narrative()
    backend.core.schemas        → DrugSafetyResponse, DrugAdverseEvent, APIResponse
"""

import asyncio
import logging
from typing import Optional, List

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.core.schemas import (
    DrugSafetyResponse,
    DrugAdverseEvent,
    APIResponse,
)
from backend.utils.gemini_client import generate_drug_safety_narrative

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/drugs",
    tags=["Drug Safety"],
)

# ── OpenFDA constants ─────────────────────────────────────────────────────────
OPENFDA_BASE    = "https://api.fda.gov"
HTTP_TIMEOUT    = 12.0          # seconds
MAX_LABEL_CHARS = 400           # truncate long label fields
MAX_AE_RESULTS  = 10            # adverse events per drug query


# ──────────────────────────────────────────────────────────────────────────────
# OPENFDA HTTP HELPERS
# ──────────────────────────────────────────────────────────────────────────────

async def _fda_get(endpoint: str, params: dict) -> dict:
    """
    Execute a single GET request against the OpenFDA API.
    Returns {} on 404 (drug not found).
    Raises HTTPException(502) on other HTTP errors.
    """
    url = f"{OPENFDA_BASE}{endpoint}"
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(url, params=params)
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"OpenFDA API timed out ({HTTP_TIMEOUT}s). Please retry.",
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach OpenFDA API: {str(exc)}",
        )

    if resp.status_code == 404:
        return {}                    # Drug not found — not an error

    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"OpenFDA error {resp.status_code}: {resp.text[:200]}",
        )

    return resp.json()


async def _fetch_adverse_events(drug_name: str, limit: int = MAX_AE_RESULTS) -> List[DrugAdverseEvent]:
    """
    Query FAERS (FDA Adverse Event Reporting System) for the top reported
    adverse reactions for a drug.
    Returns a list of DrugAdverseEvent(term, count) sorted by count desc.
    """
    params = {
        "search": f'patient.drug.medicinalproduct:"{drug_name}"',
        "count":  "patient.reaction.reactionmeddrapt.exact",
        "limit":  limit,
    }
    data = await _fda_get("/drug/event.json", params)

    if not data or "results" not in data:
        return []

    return [
        DrugAdverseEvent(term=row["term"], count=row["count"])
        for row in data["results"]
    ]


async def _fetch_drug_label(drug_name: str) -> dict:
    """
    Fetch the FDA structured product label for a drug.
    Searches by brand name OR generic name.
    Returns the first matching label dict, or {} if not found.
    """
    params = {
        "search": (
            f'openfda.brand_name:"{drug_name}" '
            f'OR openfda.generic_name:"{drug_name}"'
        ),
        "limit": 1,
    }
    data = await _fda_get("/drug/label.json", params)

    if not data or "results" not in data or not data["results"]:
        return {}

    return data["results"][0]


async def _fetch_total_reports(drug_name: str) -> int:
    """Return the total number of FAERS adverse event reports for a drug."""
    params = {
        "search": f'patient.drug.medicinalproduct:"{drug_name}"',
        "limit":  1,
    }
    data = await _fda_get("/drug/event.json", params)
    return data.get("meta", {}).get("results", {}).get("total", 0)


def _extract_label_text(label: dict, field: str, max_chars: int = MAX_LABEL_CHARS) -> List[str]:
    """
    Safely extract a text field from an FDA label dict.
    Truncates to max_chars and appends "…" if the text is long.
    Returns [] if the field is absent or empty.
    """
    value = label.get(field, [])
    if not isinstance(value, list) or not value:
        return []
    text = value[0].strip()
    if not text:
        return []
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return [text]


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/drugs/search
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/search",
    response_model=DrugSafetyResponse,
    summary="Drug safety profile: adverse events, warnings, contraindications",
)
async def drug_safety_search(
    drug_name: str = Query(
        ...,
        description="Drug name — brand or generic (e.g. 'metformin', 'Glucophage')",
        min_length=2,
    ),
    patient_id: Optional[str] = Query(
        default=None,
        description="Optional patient ID (for logging / future audit trail)",
    ),
    include_narrative: bool = Query(
        default=True,
        description="If true, generate a Gemini AI safety narrative",
    ),
):
    """
    Full drug safety profile from OpenFDA.

    Returns:
    - **top_adverse_events**: Top reported side effects from FAERS (with counts)
    - **warnings**: FDA label warnings section
    - **contraindications**: Conditions / drugs to avoid
    - **indications**: Approved uses
    - **brand_names**: Known brand names
    - **total_reports**: Total FAERS adverse event report count
    - **ai_narrative**: Plain-language Gemini safety summary (if enabled)
    """
    logger.info(
        f"[drugs/search] drug='{drug_name}' | "
        f"patient={patient_id} | narrative={include_narrative}"
    )

    # ── Parallel OpenFDA calls ────────────────────────────────────────────
    try:
        adverse_events_task = _fetch_adverse_events(drug_name, limit=MAX_AE_RESULTS)
        label_task          = _fetch_drug_label(drug_name)
        total_task          = _fetch_total_reports(drug_name)

        adverse_events, label, total_reports = await asyncio.gather(
            adverse_events_task,
            label_task,
            total_task,
            return_exceptions=False,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[drugs/search] OpenFDA fetch error: {exc}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"OpenFDA API error: {str(exc)}")

    # ── Parse label fields ────────────────────────────────────────────────
    openfda      = label.get("openfda", {})
    brand_names  = openfda.get("brand_name",      [])[:3]
    generic_name = (openfda.get("generic_name",   [""])[0]
                    if openfda.get("generic_name") else "")
    manufacturer = (openfda.get("manufacturer_name", [""])[0]
                    if openfda.get("manufacturer_name") else "")

    warnings         = _extract_label_text(label, "warnings")
    contraindications = _extract_label_text(label, "contraindications")
    indications      = _extract_label_text(label, "indications_and_usage")

    # ── Gemini AI narrative ───────────────────────────────────────────────
    ai_narrative = ""
    if include_narrative and (adverse_events or warnings):
        try:
            ai_narrative = await generate_drug_safety_narrative(
                drug_name=drug_name,
                adverse_events=[ae.model_dump() for ae in adverse_events],
                warnings=warnings,
            )
        except Exception as exc:
            logger.warning(f"[drugs/search] Gemini narrative failed: {exc}")
            ae_terms    = ", ".join(ae.term for ae in adverse_events[:5])
            ai_narrative = (
                f"Common adverse effects reported for {drug_name} include: {ae_terms}. "
                f"Always follow your physician's guidance and report any new symptoms promptly."
            )

    logger.info(
        f"[drugs/search] ✅ '{drug_name}' | "
        f"AE count={len(adverse_events)} | total_reports={total_reports:,}"
    )

    return DrugSafetyResponse(
        drug_name=drug_name,
        brand_names=brand_names,
        generic_name=generic_name,
        manufacturer=manufacturer,
        top_adverse_events=adverse_events,
        warnings=warnings,
        contraindications=contraindications,
        indications=indications,
        total_reports=total_reports,
        source="OpenFDA",
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/drugs/interactions
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/interactions",
    response_model=APIResponse,
    summary="Drug-drug interaction signal via co-occurrence in FAERS reports",
)
async def drug_interactions(
    drug_a: str = Query(..., description="First drug name"),
    drug_b: str = Query(..., description="Second drug name"),
):
    """
    Detect potential drug-drug interactions by querying how often both
    drugs appear together in the same FAERS adverse event report.

    **Important caveat:** Co-occurrence in reports is a signal, not proof
    of causation. Always verify with clinical pharmacist or drug interaction
    databases (e.g. Drugs.com, Lexicomp).

    Returns:
    - co_report_count: total reports mentioning both drugs together
    - co_reported_events: top adverse events in those reports
    - signal_strength: LOW | MODERATE | HIGH based on report count
    """
    logger.info(f"[drugs/interactions] {drug_a} + {drug_b}")

    params = {
        "search": (
            f'patient.drug.medicinalproduct:"{drug_a}" '
            f'AND patient.drug.medicinalproduct:"{drug_b}"'
        ),
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": 8,
    }

    try:
        data = await _fda_get("/drug/event.json", params)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    results       = data.get("results", [])
    total_co      = data.get("meta", {}).get("results", {}).get("total", 0)

    # Signal strength heuristic
    if total_co >= 1000:
        signal_strength = "HIGH"
    elif total_co >= 100:
        signal_strength = "MODERATE"
    elif total_co > 0:
        signal_strength = "LOW"
    else:
        signal_strength = "NONE"

    return APIResponse(
        message=f"Drug interaction signal between '{drug_a}' and '{drug_b}'",
        data={
            "drug_a":              drug_a,
            "drug_b":              drug_b,
            "co_report_count":     total_co,
            "signal_strength":     signal_strength,
            "co_reported_events":  results[:8],
            "disclaimer": (
                "Co-occurrence in FAERS reports is a pharmacovigilance signal only. "
                "It does not confirm a causal interaction. "
                "Consult a clinical pharmacist before making prescribing decisions."
            ),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/drugs/recalls
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/recalls",
    response_model=APIResponse,
    summary="Check FDA drug recall history for a medication",
)
async def drug_recalls(
    drug_name: str = Query(..., description="Drug or product name to check"),
    limit: int     = Query(default=5, ge=1, le=20, description="Maximum recalls to return"),
):
    """
    Query the FDA Drug Enforcement database for recalls, market withdrawals,
    and safety alerts related to a specific drug or product name.

    Returns: recall number, product description, reason, status, and date.
    """
    logger.info(f"[drugs/recalls] drug='{drug_name}' limit={limit}")

    params = {
        "search": f'product_description:"{drug_name}"',
        "limit":  limit,
    }

    try:
        data = await _fda_get("/drug/enforcement.json", params)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    raw_results = data.get("results", [])

    formatted = [
        {
            "recall_number": r.get("recall_number",         "N/A"),
            "product":       r.get("product_description",   "")[:250],
            "reason":        r.get("reason_for_recall",     "")[:350],
            "status":        r.get("status",                "Unknown"),
            "classification": r.get("classification",       ""),  # Class I/II/III
            "date":          r.get("recall_initiation_date",""),
            "firm":          r.get("recalling_firm",        ""),
        }
        for r in raw_results
    ]

    active_recalls = [f for f in formatted if f["status"] == "Ongoing"]

    return APIResponse(
        message=f"Found {len(formatted)} recall record(s) for '{drug_name}'",
        data={
            "drug_name":     drug_name,
            "recalls":       formatted,
            "count":         len(formatted),
            "active_count":  len(active_recalls),
            "has_active":    len(active_recalls) > 0,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/drugs/patient-check
# ──────────────────────────────────────────────────────────────────────────────

class PatientMedCheckRequest(BaseModel):
    patient_id: str
    medications: List[str]


@router.post(
    "/patient-check",
    response_model=APIResponse,
    summary="Safety check for all of a patient's medications",
)
async def patient_medication_check(request: PatientMedCheckRequest):
    """
    Run an OpenFDA safety check on every medication in a patient's list.

    For each drug returns:
    - Total FAERS report count
    - Top 3 adverse events
    - Whether any active recalls exist

    **Note:** This calls OpenFDA once per medication — response time scales
    linearly. For >5 medications expect 5-15 second response times.
    """
    medications = request.medications[:10]   # cap at 10 to avoid timeouts
    logger.info(
        f"[drugs/patient-check] patient={request.patient_id} | "
        f"medications={medications}"
    )

    results = []
    for med in medications:
        try:
            ae_list = await _fetch_adverse_events(med, limit=3)
            total   = await _fetch_total_reports(med)

            # Quick recall check
            recall_data = await _fda_get(
                "/drug/enforcement.json",
                {"search": f'product_description:"{med}"', "limit": 1}
            )
            has_recall = bool(recall_data.get("results"))

            results.append({
                "medication":       med,
                "total_ae_reports": total,
                "top_adverse_events": [
                    {"term": ae.term, "count": ae.count}
                    for ae in ae_list
                ],
                "has_recall":  has_recall,
                "status":      "ok",
            })
        except Exception as exc:
            logger.warning(f"[drugs/patient-check] Failed for '{med}': {exc}")
            results.append({
                "medication": med,
                "status":     "error",
                "error":      str(exc),
            })

    flagged = [r for r in results if r.get("has_recall") or r.get("total_ae_reports", 0) > 50000]

    return APIResponse(
        message=f"Medication safety check complete for patient '{request.patient_id}'",
        data={
            "patient_id":  request.patient_id,
            "results":     results,
            "count":       len(results),
            "flagged":     len(flagged),
            "flagged_medications": [r["medication"] for r in flagged],
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/drugs/ndc/{ndc}
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/ndc/{ndc}",
    response_model=APIResponse,
    summary="Look up drug by NDC (National Drug Code)",
)
async def lookup_by_ndc(ndc: str):
    """
    Retrieve drug product information using an NDC code (e.g. '0378-1805').
    Useful for looking up exact formulations from prescription labels.
    """
    logger.info(f"[drugs/ndc] ndc={ndc}")

    params = {
        "search": f'openfda.package_ndc:"{ndc}"',
        "limit":  1,
    }

    try:
        data = await _fda_get("/drug/label.json", params)
    except HTTPException:
        raise

    if not data or not data.get("results"):
        raise HTTPException(
            status_code=404,
            detail=f"No drug found for NDC '{ndc}'",
        )

    label    = data["results"][0]
    openfda  = label.get("openfda", {})

    return APIResponse(
        message=f"Drug found for NDC '{ndc}'",
        data={
            "ndc":                ndc,
            "brand_name":         openfda.get("brand_name",       [""])[0],
            "generic_name":       openfda.get("generic_name",     [""])[0],
            "manufacturer":       openfda.get("manufacturer_name",[""])[0],
            "route":              openfda.get("route",            []),
            "dosage_form":        openfda.get("dosage_form",      [""])[0],
            "substance_name":     openfda.get("substance_name",   []),
            "purpose":            _extract_label_text(label, "purpose"),
            "warnings":           _extract_label_text(label, "warnings"),
            "indications":        _extract_label_text(label, "indications_and_usage"),
        },
    )

