
"""
medigraph-ai/backend/api/documents.py
FILE:    backend/api/documents.py
PURPOSE: Document Intelligence API — receives PDF uploads, triggers the full
         LangGraph 4-agent pipeline, persists results, and exposes query
         endpoints for processed document history.

ENDPOINTS:
    POST /api/documents/upload          — Upload PDF → full pipeline
    POST /api/documents/analyze-text    — Analyze raw text (no PDF)
    GET  /api/documents/patient/{id}    — All documents for a patient
    GET  /api/documents/{doc_id}        — Single document by MongoDB _id
    GET  /api/documents/recent          — Most recently processed documents

PIPELINE TRIGGERED ON UPLOAD:
    Agent 1 (Document)   → PyMuPDF extraction + Gemini entity extraction
    Agent 2 (KG)         → Neo4j node + relationship ingestion
    Agent 3 (Risk)       → XGBoost risk prediction
    Agent 4 (Intervention)→ Gemini intervention plan

DEPENDENCIES:
    backend.agents.graph   → run_full_pipeline()
    backend.db.mongodb     → save_document_analysis(), get_patient_documents()
    backend.core.schemas   → DocumentAnalysisResponse, ExtractedEntity, APIResponse
"""

import time
import logging
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse

from backend.core.schemas import DocumentAnalysisResponse, ExtractedEntity, APIResponse
from backend.agents.graph import run_full_pipeline
from backend.db.mongodb import (
    save_document_analysis,
    get_patient_documents,
    get_db,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/documents",
    tags=["Document Intelligence"],
)

# ── Constants ─────────────────────────────────────────────────────────────────
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}
MAX_FILE_SIZE_MB      = 10
MAX_FILE_SIZE_BYTES   = MAX_FILE_SIZE_MB * 1024 * 1024

VALID_DOC_TYPES = {"prescription", "lab_report", "medical_summary"}


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/documents/upload
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=DocumentAnalysisResponse,
    summary="Upload a healthcare PDF and run the full AI pipeline",
    response_description="Structured entities, risk score, and intervention plan",
)
async def upload_document(
    # ── File ──────────────────────────────────────────────────────────────
    file: UploadFile = File(
        ...,
        description="Healthcare PDF (prescription, lab report, or medical summary)",
    ),
    # ── Patient identity ──────────────────────────────────────────────────
    patient_id: str = Form(
        ...,
        description="Unique patient identifier, e.g. 'P001'",
    ),
    patient_name: str = Form(
        default="Unknown",
        description="Patient full name",
    ),
    document_type: str = Form(
        default="prescription",
        description="Document type: prescription | lab_report | medical_summary",
    ),
    # ── Clinical baseline ─────────────────────────────────────────────────
    adherence_rate: float = Form(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Current medication adherence percentage (0–100)",
    ),
    age: int = Form(
        default=50,
        ge=1,
        le=120,
        description="Patient age in years",
    ),
    gender: str = Form(
        default="Unknown",
        description="Patient gender",
    ),
    conditions: str = Form(
        default="",
        description="Comma-separated known conditions, e.g. 'type 2 diabetes,hypertension'",
    ),
    medications: str = Form(
        default="",
        description="Comma-separated current medications, e.g. 'metformin,lisinopril'",
    ),
    exercise_level: int = Form(
        default=5,
        ge=1,
        le=10,
        description="Physical activity level: 1 (sedentary) to 10 (very active)",
    ),
    follow_up_frequency: int = Form(
        default=4,
        ge=0,
        description="Number of clinic visits per year",
    ),
    comorbidity_count: int = Form(
        default=0,
        ge=0,
        description="Total number of concurrent chronic conditions",
    ),
    medication_count: int = Form(
        default=1,
        ge=1,
        description="Total number of medications currently prescribed",
    ),
    hospital_visits: int = Form(
        default=0,
        ge=0,
        description="Number of hospital visits / admissions in the past 12 months",
    ),
    hba1c: Optional[float] = Form(
        default=None,
        description="Most recent HbA1c value (%), e.g. 8.5",
    ),
    bmi: Optional[float] = Form(
        default=None,
        description="Body Mass Index, e.g. 27.5",
    ),
):
    """
    Upload a healthcare PDF and execute the full LangGraph pipeline.

    **Processing steps:**
    1. Validate file type and size
    2. Agent 1 — PyMuPDF text extraction (+ OCR fallback for scanned pages)
    3. Agent 1 — Gemini clinical entity extraction (diseases, medications, labs …)
    4. Agent 2 — Neo4j knowledge graph ingestion
    5. Agent 3 — XGBoost medication adherence risk prediction
    6. Agent 4 — Gemini personalised intervention plan generation
    7. Persist full record to MongoDB
    8. Return structured response

    **Typical processing time:** 3–8 seconds (dominated by Gemini API calls).
    """

    # ── Validate file type ────────────────────────────────────────────────
    is_pdf_content_type = file.content_type in ALLOWED_CONTENT_TYPES
    is_pdf_filename     = (file.filename or "").lower().endswith(".pdf")
    if not (is_pdf_content_type or is_pdf_filename):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are accepted. Received content-type: {file.content_type}",
        )

    # ── Validate document type ────────────────────────────────────────────
    if document_type not in VALID_DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"document_type must be one of: {sorted(VALID_DOC_TYPES)}",
        )

    # ── Read and size-check file ──────────────────────────────────────────
    file_bytes = await file.read()
    size_bytes = len(file_bytes)
    size_mb    = size_bytes / (1024 * 1024)

    if size_bytes == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if size_bytes > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File size {size_mb:.1f} MB exceeds maximum of {MAX_FILE_SIZE_MB} MB",
        )

    logger.info(
        f"[documents/upload] patient={patient_id} | file={file.filename} | "
        f"size={size_mb:.2f}MB | doc_type={document_type}"
    )

    # ── Parse comma-separated lists ───────────────────────────────────────
    parsed_conditions  = [c.strip() for c in conditions.split(",")  if c.strip()]
    parsed_medications = [m.strip() for m in medications.split(",") if m.strip()]

    # ── Build LangGraph initial state ─────────────────────────────────────
    initial_state = {
        # Identity
        "patient_id":    patient_id,
        "patient_name":  patient_name,
        # Document
        "file_bytes":    file_bytes,
        "filename":      file.filename or "upload.pdf",
        "document_type": document_type,
        # Clinical baseline
        "age":                       age,
        "gender":                    gender,
        "adherence_rate":            adherence_rate,
        "conditions":                parsed_conditions,
        "medications":               parsed_medications,
        "exercise_level":            exercise_level,
        "follow_up_frequency":       follow_up_frequency,
        "comorbidity_count":         comorbidity_count,
        "medication_count":          medication_count,
        "hospital_visits_last_year": hospital_visits,
        "hba1c":                     hba1c,
        "bmi":                       bmi,
        "lab_values":                {},
        "current_symptoms":          [],
        # Pipeline control
        "skip_document": False,
    }

    # ── Run pipeline ──────────────────────────────────────────────────────
    pipeline_start = time.time()
    try:
        result = await run_full_pipeline(initial_state)
    except Exception as exc:
        logger.error(f"[documents/upload] Pipeline failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(exc)}")

    pipeline_ms = round((time.time() - pipeline_start) * 1000, 2)
    logger.info(f"[documents/upload] Pipeline complete in {pipeline_ms} ms")

    # ── Extract entity dict safely ────────────────────────────────────────
    raw_entities = result.get("extracted_entities") or {}
    entity_obj = ExtractedEntity(
        diseases=raw_entities.get("diseases",    []),
        symptoms=raw_entities.get("symptoms",    []),
        medications=raw_entities.get("medications", []),
        lab_tests=raw_entities.get("lab_tests",  []),
        lab_values=raw_entities.get("lab_values", {}),
        risk_factors=raw_entities.get("risk_factors", []),
        dosages=raw_entities.get("dosages",      {}),
        instructions=raw_entities.get("instructions", []),
    )

    # ── Persist to MongoDB ────────────────────────────────────────────────
    doc_record = {
        "patient_id":          patient_id,
        "patient_name":        patient_name,
        "filename":            file.filename,
        "document_type":       document_type,
        "raw_text":            result.get("raw_text", ""),
        "extracted_entities":  raw_entities,
        "summary":             result.get("document_summary", ""),
        "risk_score":          result.get("risk_score"),
        "risk_level":          result.get("risk_level"),
        "risk_explanation":    result.get("risk_explanation", ""),
        "intervention_plan":   result.get("intervention_plan", {}),
        "kg_nodes_created":    result.get("kg_nodes_created", 0),
        "doc_agent_status":    result.get("doc_agent_status", ""),
        "kg_agent_status":     result.get("kg_agent_status", ""),
        "risk_agent_status":   result.get("risk_agent_status", ""),
        "intervention_status": result.get("intervention_agent_status", ""),
        "processing_time_ms":  pipeline_ms,
        "file_size_bytes":     size_bytes,
    }
    await save_document_analysis(doc_record)

    # ── Build and return response ─────────────────────────────────────────
    return DocumentAnalysisResponse(
        patient_id=patient_id,
        document_type=document_type,
        raw_text=result.get("raw_text", ""),
        extracted_entities=entity_obj,
        summary=result.get("document_summary", ""),
        processing_time_ms=pipeline_ms,
        status=result.get("doc_agent_status", "success"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/documents/analyze-text
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/analyze-text",
    response_model=APIResponse,
    summary="Analyze pasted clinical text without a PDF",
)
async def analyze_raw_text(
    patient_id: str   = Query(..., description="Patient identifier"),
    text: str         = Query(..., description="Raw clinical text to analyze"),
    document_type: str = Query(default="medical_summary", description="Document type"),
    save_result: bool  = Query(default=True,  description="Persist result to MongoDB"),
):
    """
    Extract clinical entities and generate a summary from plain text.

    Useful for:
    - Testing the entity extraction pipeline without a PDF
    - Analyzing text copied from an EHR system
    - Demo purposes

    **Does NOT trigger the KG or Risk agents** — use /upload for the full pipeline.
    """
    from backend.utils.gemini_client import (
        extract_medical_entities,
        summarize_medical_document,
    )

    if not text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    if document_type not in VALID_DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"document_type must be one of: {sorted(VALID_DOC_TYPES)}",
        )

    logger.info(f"[documents/analyze-text] patient={patient_id} | chars={len(text)}")

    try:
        entities = await extract_medical_entities(text)
        summary  = await summarize_medical_document(text, document_type)
    except Exception as exc:
        logger.error(f"[documents/analyze-text] Gemini error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(exc)}")

    result_data = {
        "patient_id":         patient_id,
        "document_type":      document_type,
        "extracted_entities": entities,
        "summary":            summary,
        "char_count":         len(text),
    }

    if save_result:
        try:
            await save_document_analysis({
                **result_data,
                "raw_text":      text,
                "filename":      "text_analysis",
                "risk_score":    None,
                "risk_level":    None,
            })
        except Exception as exc:
            logger.warning(f"[documents/analyze-text] MongoDB save failed: {exc}")

    return APIResponse(
        message="Text analysis complete",
        data=result_data,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/documents/patient/{patient_id}
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/patient/{patient_id}",
    response_model=APIResponse,
    summary="Get all processed documents for a patient",
)
async def get_patient_documents_endpoint(
    patient_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Maximum documents to return"),
    doc_type: Optional[str] = Query(default=None, description="Filter by document type"),
):
    """
    Retrieve the full processing history for a patient — all uploaded PDFs,
    their extracted entities, risk scores, and intervention plans.

    Results are sorted newest-first.
    """
    docs = await get_patient_documents(patient_id)

    # Optional filter by document type
    if doc_type:
        docs = [d for d in docs if d.get("document_type") == doc_type]

    # Apply limit
    docs = docs[:limit]

    logger.info(f"[documents/patient] patient={patient_id} → {len(docs)} docs returned")

    return APIResponse(
        message=f"Found {len(docs)} document(s) for patient {patient_id}",
        data={
            "patient_id": patient_id,
            "documents":  docs,
            "count":      len(docs),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/documents/recent
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/recent",
    response_model=APIResponse,
    summary="Get the most recently processed documents across all patients",
)
async def get_recent_documents(
    limit: int = Query(default=10, ge=1, le=50, description="Number of recent documents"),
):
    """
    Returns the most recently uploaded and processed documents across the
    entire system. Useful for the dashboard 'Recent Activity' feed.
    """
    try:
        db     = get_db()
        cursor = db.documents.find({}).sort("uploaded_at", -1).limit(limit)
        docs   = []
        async for doc in cursor:
            doc["id"] = str(doc.pop("_id"))
            # Trim raw_text for listing view (keep entities + summary)
            doc.pop("raw_text", None)
            docs.append(doc)

        return APIResponse(
            message=f"Returned {len(docs)} recent document(s)",
            data={"documents": docs, "count": len(docs)},
        )
    except Exception as exc:
        logger.error(f"[documents/recent] Error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/documents/{doc_id}
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{doc_id}",
    response_model=APIResponse,
    summary="Get a single processed document by its MongoDB ID",
)
async def get_document_by_id(doc_id: str):
    """
    Retrieve the full detail of one processed document — including raw text,
    all extracted entities, risk score, and the intervention plan generated
    by the pipeline.
    """
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        oid = ObjectId(doc_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail=f"Invalid document ID format: {doc_id}")

    try:
        db  = get_db()
        doc = await db.documents.find_one({"_id": oid})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    doc["id"] = str(doc.pop("_id"))
    return APIResponse(data=doc)
