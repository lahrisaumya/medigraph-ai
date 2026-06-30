"""
================================================================================
FILE:    backend/agents/document_agent.py
AGENT:   Agent 1 — Document Analysis Agent
PURPOSE: Receives raw PDF bytes from the LangGraph state, extracts text using
         PyMuPDF (with OCR fallback for scanned pages), then calls the Gemini
         API to extract structured clinical entities (diseases, medications,
         lab values, symptoms, risk factors) and generate a document summary.

INPUTS  (from LangGraph state):
    file_bytes     (bytes | str) : raw PDF bytes, or plain text for testing
    filename       (str)         : original file name, e.g. "prescription.pdf"
    patient_id     (str)         : patient identifier
    document_type  (str)         : "prescription" | "lab_report" | "medical_summary"

OUTPUTS (added to LangGraph state):
    raw_text           (str)   : full extracted text from the PDF
    extracted_entities (dict)  : structured clinical entities from Gemini
    document_summary   (str)   : 3-4 sentence clinical summary from Gemini
    processing_time_ms (float) : total agent wall-clock time in milliseconds
    doc_agent_status   (str)   : "success" | "error: <message>"
================================================================================
"""

import time
import logging
from typing import Dict, Any

from backend.utils.pdf_extractor import extract_text_from_pdf, clean_extracted_text
from backend.utils.gemini_client import extract_medical_entities, summarize_medical_document

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# AGENT NODE
# ──────────────────────────────────────────────────────────────────────────────

async def document_analysis_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node — Document Analysis Agent.

    Workflow:
        1. Read file_bytes from state
        2. If bytes → PyMuPDF extraction (+ OCR for scanned pages)
           If str   → treat directly as raw text (useful for unit tests)
        3. Clean extracted text
        4. Gemini: extract structured clinical entities as JSON
        5. Gemini: generate 3-4 sentence clinical summary
        6. Return updated state with all outputs

    Error handling:
        Any exception is caught, logged, and written back to state as
        doc_agent_status = "error: <message>".  Empty defaults are set so
        downstream agents (KG Agent, Risk Agent) can still run safely.
    """
    start_time = time.time()
    patient_id = state.get("patient_id", "UNKNOWN")
    logger.info(f"[DocAgent] ▶ Starting for patient={patient_id}")

    try:
        # ── Step 1: Get document content ──────────────────────────────────────
        file_bytes = state.get("file_bytes")
        filename   = state.get("filename", "document.pdf")

        if file_bytes is None:
            raise ValueError("No file_bytes found in state. Did you set skip_document=True by mistake?")

        # ── Step 2: Extract text ──────────────────────────────────────────────
        if isinstance(file_bytes, bytes):
            logger.info(f"[DocAgent] Extracting text from PDF bytes ({len(file_bytes):,} bytes)")
            raw_text, page_count, is_scanned = extract_text_from_pdf(file_bytes)
            logger.info(
                f"[DocAgent] Extracted {len(raw_text):,} chars | "
                f"pages={page_count} | scanned={is_scanned}"
            )
        elif isinstance(file_bytes, str):
            # Plain text passed directly (e.g. from /analyze-text endpoint or tests)
            raw_text   = file_bytes
            page_count = 1
            is_scanned = False
            logger.info(f"[DocAgent] Received plain text ({len(raw_text):,} chars) — skipping PDF extraction")
        else:
            raise TypeError(f"file_bytes must be bytes or str, got {type(file_bytes)}")

        # ── Step 3: Clean text ────────────────────────────────────────────────
        cleaned_text = clean_extracted_text(raw_text)

        if len(cleaned_text.strip()) < 20:
            logger.warning("[DocAgent] Extracted text is very short — document may be empty or unreadable")

        # ── Step 4: Entity extraction (Gemini) ───────────────────────────────
        logger.info("[DocAgent] Calling Gemini for entity extraction ...")
        extracted_entities = await extract_medical_entities(cleaned_text)

        # Log what was found for observability
        _log_entity_summary(extracted_entities)

        # ── Step 5: Document summary (Gemini) ───────────────────────────────
        doc_type = state.get("document_type", "medical document")
        logger.info(f"[DocAgent] Calling Gemini for {doc_type} summary ...")
        document_summary = await summarize_medical_document(cleaned_text, doc_type)

        # ── Step 6: Timing ───────────────────────────────────────────────────
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"[DocAgent] ✅ Completed in {elapsed_ms} ms")

        return {
            **state,
            "raw_text":           cleaned_text,
            "extracted_entities": extracted_entities,
            "document_summary":   document_summary,
            "processing_time_ms": elapsed_ms,
            "doc_agent_status":   "success",
        }

    except Exception as exc:
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(f"[DocAgent] ❌ Failed after {elapsed_ms} ms: {exc}", exc_info=True)

        return {
            **state,
            "raw_text":           "",
            "extracted_entities": _empty_entities(),
            "document_summary":   "Document processing failed — see server logs.",
            "processing_time_ms": elapsed_ms,
            "doc_agent_status":   f"error: {str(exc)}",
        }


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _empty_entities() -> Dict[str, Any]:
    """Return a safe empty entity dict so downstream agents never KeyError."""
    return {
        "diseases":       [],
        "symptoms":       [],
        "medications":    [],
        "lab_tests":      [],
        "lab_values":     {},
        "risk_factors":   [],
        "dosages":        {},
        "instructions":   [],
    }


def _log_entity_summary(entities: Dict[str, Any]) -> None:
    """Log a one-line summary of extracted entities for debugging."""
    counts = {
        "diseases":     len(entities.get("diseases",     [])),
        "medications":  len(entities.get("medications",  [])),
        "symptoms":     len(entities.get("symptoms",     [])),
        "lab_tests":    len(entities.get("lab_tests",    [])),
        "risk_factors": len(entities.get("risk_factors", [])),
    }
    logger.info(
        "[DocAgent] Entities found — " +
        " | ".join(f"{k}={v}" for k, v in counts.items())
    )