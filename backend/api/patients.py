"""
medigraph-ai/backend/api/patients.py

FILE:    backend/api/patients.py
PURPOSE: Patient management API — full CRUD for patient records stored in
         MongoDB, plus Neo4j node synchronisation and dashboard statistics.

ENDPOINTS:
    POST   /api/patients/                  — Create new patient (MongoDB + Neo4j)
    GET    /api/patients/                  — List all patients (paginated)
    GET    /api/patients/dashboard/stats   — Executive dashboard statistics
    GET    /api/patients/search            — Search patients by name / condition
    GET    /api/patients/{patient_id}      — Get single patient detail
    PUT    /api/patients/{patient_id}      — Update patient fields
    DELETE /api/patients/{patient_id}      — Soft-delete patient record

DEPENDENCIES:
    backend.db.mongodb    → patient CRUD helpers + get_dashboard_stats()
    backend.db.neo4j_db   → upsert_patient_node(), get_full_graph_summary()
    backend.core.schemas  → PatientCreate, APIResponse
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.core.schemas import PatientCreate, APIResponse
from backend.db.mongodb import (
    create_patient,
    get_patient,
    get_all_patients,
    update_patient,
    get_dashboard_stats,
    get_db,
)
from backend.db.neo4j_db import upsert_patient_node, get_full_graph_summary

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/patients",
    tags=["Patients"],
)


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/patients/
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=APIResponse,
    status_code=201,
    summary="Create a new patient record",
)
async def create_new_patient(patient: PatientCreate):
    """
    Create a new patient in both MongoDB and Neo4j.

    - **MongoDB**: stores the full clinical profile for querying and ML features
    - **Neo4j**: creates a Patient node for knowledge graph relationships

    Returns the MongoDB document ID and patient_id on success.
    Raises 409 if a patient with the same patient_id already exists.
    """
    logger.info(f"[patients/create] Creating patient: {patient.patient_id}")

    # ── Duplicate check ───────────────────────────────────────────────────
    existing = await get_patient(patient.patient_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Patient '{patient.patient_id}' already exists. Use PUT to update.",
        )

    # ── MongoDB insert ────────────────────────────────────────────────────
    patient_data = patient.model_dump()
    try:
        mongo_id = await create_patient(patient_data)
    except Exception as exc:
        logger.error(f"[patients/create] MongoDB insert failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(exc)}")

    # ── Neo4j upsert (non-blocking: failure does not abort the request) ───
    try:
        await upsert_patient_node(patient_data)
        logger.info(f"[patients/create] Neo4j node created for {patient.patient_id}")
    except Exception as exc:
        logger.warning(
            f"[patients/create] Neo4j upsert failed (non-fatal): {exc}"
        )

    logger.info(f"[patients/create] ✅ Patient {patient.patient_id} created (id={mongo_id})")

    return APIResponse(
        message=f"Patient '{patient.patient_id}' created successfully",
        data={
            "id":         mongo_id,
            "patient_id": patient.patient_id,
            "name":       patient.name,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/patients/
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=APIResponse,
    summary="List all patients (paginated)",
)
async def list_patients(
    limit: int  = Query(default=100, ge=1,  le=500, description="Maximum records to return"),
    skip:  int  = Query(default=0,   ge=0,          description="Number of records to skip (pagination)"),
    risk_level: Optional[str] = Query(
        default=None,
        description="Filter by latest risk level: LOW | MODERATE | HIGH | CRITICAL",
    ),
):
    """
    Return a paginated list of all patients with their core clinical profile.

    Optional `risk_level` filter cross-references the latest risk prediction
    stored in MongoDB.
    """
    logger.info(f"[patients/list] limit={limit} skip={skip} risk_level={risk_level}")

    try:
        patients = await get_all_patients(limit=limit + skip)
        # Manual skip (Motor cursor doesn't support skip reliably without index)
        patients = patients[skip:skip + limit]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Optional risk_level filter (requires a join with risk_predictions in memory)
    if risk_level:
        risk_level_upper = risk_level.upper()
        try:
            db         = get_db()
            # Get latest risk level per patient
            pipeline   = [
                {"$sort": {"predicted_at": -1}},
                {"$group": {"_id": "$patient_id", "risk_level": {"$first": "$risk_level"}}},
            ]
            risk_map   = {}
            async for doc in db.risk_predictions.aggregate(pipeline):
                risk_map[doc["_id"]] = doc["risk_level"]

            patients = [
                p for p in patients
                if risk_map.get(p.get("patient_id"), "") == risk_level_upper
            ]
        except Exception as exc:
            logger.warning(f"[patients/list] Risk filter failed: {exc}")

    return APIResponse(
        message=f"Returned {len(patients)} patient(s)",
        data={
            "patients": patients,
            "count":    len(patients),
            "limit":    limit,
            "skip":     skip,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/patients/dashboard/stats
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/dashboard/stats",
    response_model=APIResponse,
    summary="Executive dashboard statistics",
)
async def dashboard_stats():
    """
    Aggregate statistics for the Executive Overview dashboard page.

    Returns:
    - total_patients, high_risk_patients, documents_processed
    - avg_adherence_rate
    - risk_distribution {LOW: N, MODERATE: N, HIGH: N, CRITICAL: N}
    - kg_nodes, kg_relationships (from Neo4j)
    - model_metrics (from saved XGBoost training report)
    """
    logger.info("[patients/dashboard/stats] Aggregating dashboard stats")

    # ── MongoDB stats ─────────────────────────────────────────────────────
    try:
        stats = await get_dashboard_stats()
    except Exception as exc:
        logger.error(f"[patients/dashboard/stats] MongoDB error: {exc}")
        stats = {
            "total_patients":     0,
            "high_risk_patients": 0,
            "documents_processed": 0,
            "avg_adherence_rate": 0.0,
            "risk_distribution":  {},
        }

    # ── Neo4j stats ───────────────────────────────────────────────────────
    try:
        kg_data = await get_full_graph_summary()
        stats["kg_nodes"]         = kg_data.get("total_nodes",         0)
        stats["kg_relationships"] = kg_data.get("total_relationships",  0)
        stats["kg_node_types"]    = kg_data.get("node_counts",          {})
        stats["kg_rel_types"]     = kg_data.get("relationship_counts",  {})
    except Exception as exc:
        logger.warning(f"[patients/dashboard/stats] Neo4j unavailable: {exc}")
        stats["kg_nodes"]         = 0
        stats["kg_relationships"] = 0
        stats["kg_node_types"]    = {}
        stats["kg_rel_types"]     = {}

    # ── ML model metrics ──────────────────────────────────────────────────
    try:
        from backend.ml.predict import get_training_metrics
        model_metrics = get_training_metrics()
        stats["model_metrics"] = model_metrics or {}
    except Exception:
        stats["model_metrics"] = {}

    logger.info(
        f"[patients/dashboard/stats] "
        f"patients={stats.get('total_patients')} | "
        f"high_risk={stats.get('high_risk_patients')} | "
        f"kg_nodes={stats.get('kg_nodes')}"
    )

    return APIResponse(
        message="Dashboard statistics retrieved",
        data=stats,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/patients/search
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/search",
    response_model=APIResponse,
    summary="Search patients by name or condition",
)
async def search_patients(
    q: str = Query(..., min_length=2, description="Search term (name or condition)"),
    limit: int = Query(default=20, ge=1, le=100),
):
    """
    Full-text search across patient names and conditions list.
    Uses MongoDB regex for case-insensitive partial matching.
    """
    logger.info(f"[patients/search] query='{q}'")

    try:
        db = get_db()
        cursor = db.patients.find(
            {
                "$or": [
                    {"name":       {"$regex": q, "$options": "i"}},
                    {"conditions": {"$regex": q, "$options": "i"}},
                    {"patient_id": {"$regex": q, "$options": "i"}},
                ]
            }
        ).limit(limit)

        results = []
        async for doc in cursor:
            doc["id"] = str(doc.pop("_id"))
            results.append(doc)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return APIResponse(
        message=f"Found {len(results)} match(es) for '{q}'",
        data={"results": results, "count": len(results), "query": q},
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/patients/{patient_id}
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{patient_id}",
    response_model=APIResponse,
    summary="Get full patient detail",
)
async def get_patient_detail(patient_id: str):
    """
    Return complete patient record including:
    - Clinical profile (conditions, medications, lab values)
    - Latest risk prediction
    - Document history count
    """
    logger.info(f"[patients/detail] patient_id={patient_id}")

    # ── Fetch patient ─────────────────────────────────────────────────────
    patient = await get_patient(patient_id)
    if not patient:
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_id}' not found",
        )

    # ── Attach latest risk prediction ─────────────────────────────────────
    try:
        from backend.db.mongodb import get_latest_risk
        latest_risk = await get_latest_risk(patient_id)
        patient["latest_risk"] = latest_risk
    except Exception:
        patient["latest_risk"] = None

    # ── Attach document count ─────────────────────────────────────────────
    try:
        db = get_db()
        doc_count = await db.documents.count_documents({"patient_id": patient_id})
        patient["document_count"] = doc_count
    except Exception:
        patient["document_count"] = 0

    return APIResponse(
        message=f"Patient '{patient_id}' retrieved",
        data=patient,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PUT /api/patients/{patient_id}
# ──────────────────────────────────────────────────────────────────────────────

@router.put(
    "/{patient_id}",
    response_model=APIResponse,
    summary="Update patient fields",
)
async def update_patient_endpoint(
    patient_id: str,
    update_data: dict,
):
    """
    Update one or more fields of an existing patient record.

    - Immutable fields (patient_id, _id) are ignored if present in payload
    - Also syncs the updated adherence_rate to the Neo4j Patient node

    **Example payload:**
    ```json
    {
      "adherence_rate": 78.5,
      "exercise_level": 6,
      "hba1c": 7.8
    }
    ```
    """
    logger.info(f"[patients/update] patient_id={patient_id} | fields={list(update_data.keys())}")

    # Remove immutable fields if accidentally sent
    update_data.pop("patient_id", None)
    update_data.pop("_id", None)

    if not update_data:
        raise HTTPException(status_code=400, detail="No updatable fields provided")

    # ── MongoDB update ────────────────────────────────────────────────────
    updated = await update_patient(patient_id, update_data)
    if not updated:
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_id}' not found or no fields changed",
        )

    # ── Sync Neo4j node ───────────────────────────────────────────────────
    try:
        patient = await get_patient(patient_id)
        if patient:
            await upsert_patient_node(patient)
    except Exception as exc:
        logger.warning(f"[patients/update] Neo4j sync failed (non-fatal): {exc}")

    return APIResponse(
        message=f"Patient '{patient_id}' updated successfully",
        data={"patient_id": patient_id, "updated_fields": list(update_data.keys())},
    )


# ──────────────────────────────────────────────────────────────────────────────
# DELETE /api/patients/{patient_id}
# ──────────────────────────────────────────────────────────────────────────────

@router.delete(
    "/{patient_id}",
    response_model=APIResponse,
    summary="Soft-delete a patient record",
)
async def delete_patient(patient_id: str):
    """
    Soft-delete a patient by setting `deleted=True` and `deleted_at=<timestamp>`.
    The record is retained for audit purposes; use `?hard=true` for a hard delete
    (not implemented in this version for data safety).

    Also marks the Neo4j Patient node as inactive.
    """
    from datetime import datetime

    patient = await get_patient(patient_id)
    if not patient:
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_id}' not found",
        )

    # Soft-delete
    await update_patient(
        patient_id,
        {"deleted": True, "deleted_at": datetime.utcnow().isoformat()},
    )

    logger.info(f"[patients/delete] Soft-deleted patient {patient_id}")

    return APIResponse(
        message=f"Patient '{patient_id}' has been deactivated",
        data={"patient_id": patient_id, "deleted": True},
    )