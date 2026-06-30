"""
================================================================================
FILE:    backend/agents/graph.py
PURPOSE: LangGraph multi-agent orchestration for MediGraph AI.
         Defines the shared state schema, all routing functions, and the
         compiled StateGraph that wires the 4 agents into a directed workflow.

PIPELINE FLOW:

    ┌─────────────────────────────────────────────────────────────────┐
    │                        entry (router)                           │
    └────────────────────┬───────────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  file_bytes given?  │
              └──────────┬──────────┘
               YES       │        NO (skip_document=True)
               │         │         │
    ┌──────────▼──┐      │  ┌──────▼──────────────────┐
    │  doc_agent  │      │  │       risk_agent         │
    │  (Agent 1)  │      │  │       (Agent 3)          │
    └──────┬──────┘      │  └──────────────┬───────────┘
           │             │                 │
    success│   error─────┘                 │
           │                               │
    ┌──────▼──────┐                        │
    │   kg_agent  │                        │
    │   (Agent 2) │                        │
    └──────┬──────┘                        │
           │                               │
    ┌──────▼──────────────────────────────▼──┐
    │           risk_agent (Agent 3)          │
    └──────────────────────┬─────────────────┘
                           │
    ┌──────────────────────▼─────────────────┐
    │       intervention_agent (Agent 4)      │
    └──────────────────────┬─────────────────┘
                           │
                          END

USAGE:
    from backend.agents.graph import run_full_pipeline, run_risk_only

    # Full pipeline (with PDF upload):
    result = await run_full_pipeline({
        "patient_id": "P001",
        "patient_name": "Rajesh Kumar",
        "file_bytes": <bytes>,
        "filename": "prescription.pdf",
        "document_type": "prescription",
        "adherence_rate": 62.0,
        "age": 58,
        ...
    })

    # Risk + Intervention only (no PDF):
    result = await run_risk_only({
        "patient_id": "P001",
        "adherence_rate": 62.0,
        "age": 58,
        ...
    })
================================================================================
"""

import logging
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END

from backend.agents.document_agent    import document_analysis_agent
from backend.agents.kg_agent          import knowledge_graph_agent
from backend.agents.risk_agent        import risk_prediction_agent
from backend.agents.intervention_agent import intervention_agent

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# SHARED STATE SCHEMA
# ──────────────────────────────────────────────────────────────────────────────

class MediGraphState(TypedDict, total=False):
    """
    Central state object passed between all LangGraph nodes.
    total=False → every key is optional; agents read what they need and
    write what they produce.

    Sections:
        INPUT          — fields set by the caller before pipeline starts
        PATIENT DATA   — clinical profile fields
        AGENT 1 OUTPUT — document_analysis_agent writes these
        AGENT 2 OUTPUT — knowledge_graph_agent writes these
        AGENT 3 OUTPUT — risk_prediction_agent writes these
        AGENT 4 OUTPUT — intervention_agent writes these
        PIPELINE CTRL  — routing control flags
    """

    # ── INPUT ──────────────────────────────────────────────────────────────
    patient_id:    str          # Required: e.g. "P001"
    patient_name:  str          # Full name, e.g. "Rajesh Kumar"
    file_bytes:    Any          # Raw PDF bytes (bytes) or plain text (str)
    filename:      str          # Original filename, e.g. "lab_report.pdf"
    document_type: str          # "prescription" | "lab_report" | "medical_summary"

    # ── PATIENT CLINICAL DATA ──────────────────────────────────────────────
    age:                        int
    gender:                     str
    adherence_rate:             float          # 0-100 %
    exercise_level:             int            # 1 (sedentary) → 10 (very active)
    follow_up_frequency:        int            # clinic visits per year
    comorbidity_count:          int            # number of concurrent conditions
    medication_count:           int            # total medications prescribed
    hospital_visits_last_year:  int            # hospitalisations last 12 months
    hba1c:                      Optional[float]  # glycated haemoglobin %
    bmi:                        Optional[float]  # body mass index
    conditions:                 List[str]      # active diagnoses (user-entered)
    medications:                List[str]      # active medications (user-entered)
    lab_values:                 Dict[str, Any] # any extra lab results
    current_symptoms:           List[str]      # reported symptoms

    # ── AGENT 1: document_analysis_agent OUTPUT ────────────────────────────
    raw_text:           str                # cleaned text extracted from PDF
    extracted_entities: Dict[str, Any]    # {diseases, medications, symptoms, ...}
    document_summary:   str               # 3-4 sentence Gemini summary
    processing_time_ms: float             # agent wall-clock time
    doc_agent_status:   str               # "success" | "error: ..."

    # ── AGENT 2: knowledge_graph_agent OUTPUT ──────────────────────────────
    kg_agent_status:  str   # "success" | "error: ..."
    kg_nodes_created: int   # number of graph nodes upserted

    # ── AGENT 3: risk_prediction_agent OUTPUT ──────────────────────────────
    risk_score:        float      # 0-100 adherence risk %
    risk_level:        str        # "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
    adherence_level:   str        # "EXCELLENT" | "GOOD" | "POOR" | "CRITICAL"
    risk_factors:      List[str]  # top human-readable risk drivers
    risk_explanation:  str        # Gemini 2-3 sentence narrative
    risk_agent_status: str        # "success" | "error: ..."

    # ── AGENT 4: intervention_agent OUTPUT ────────────────────────────────
    intervention_plan:         Dict[str, Any]  # structured care plan from Gemini
    intervention_agent_status: str             # "success" | "error: ..."

    # ── PIPELINE CONTROL ──────────────────────────────────────────────────
    skip_document: bool          # True → skip doc_agent + kg_agent
    error:         Optional[str] # top-level pipeline error (rarely set)


# ──────────────────────────────────────────────────────────────────────────────
# ROUTING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _route_after_entry(state: MediGraphState) -> str:
    """
    Decide first step after the entry node.

    Rules:
        • skip_document=True  → jump straight to risk_agent (no PDF processing)
        • file_bytes missing  → jump straight to risk_agent
        • file_bytes present  → go to doc_agent (full pipeline)
    """
    skip  = state.get("skip_document", False)
    has_file = bool(state.get("file_bytes"))

    if skip or not has_file:
        logger.info("[Graph] Route: entry → risk_agent (skip_document or no file)")
        return "risk_agent"

    logger.info("[Graph] Route: entry → doc_agent (PDF present)")
    return "doc_agent"


def _route_after_doc(state: MediGraphState) -> str:
    """
    Decide next step after document_analysis_agent.

    Rules:
        • doc_agent succeeded → go to kg_agent (ingest entities into Neo4j)
        • doc_agent errored   → skip kg_agent, go directly to risk_agent
    """
    status = state.get("doc_agent_status", "")

    if status.startswith("error"):
        logger.warning(f"[Graph] Route: doc_agent → risk_agent (doc failed: {status})")
        return "risk_agent"

    logger.info("[Graph] Route: doc_agent → kg_agent")
    return "kg_agent"


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY NODE (no-op)
# ──────────────────────────────────────────────────────────────────────────────

async def _entry_node(state: MediGraphState) -> MediGraphState:
    """
    No-operation entry node required by LangGraph to have a named start.
    Routing logic lives in _route_after_entry (conditional edge).
    """
    logger.info(
        f"[Graph] Pipeline started | "
        f"patient_id={state.get('patient_id','?')} | "
        f"skip_document={state.get('skip_document', False)} | "
        f"has_file={bool(state.get('file_bytes'))}"
    )
    return state


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.

    Nodes:
        entry             → _entry_node            (router, no-op)
        doc_agent         → document_analysis_agent (Agent 1)
        kg_agent          → knowledge_graph_agent   (Agent 2)
        risk_agent        → risk_prediction_agent   (Agent 3)
        intervention_agent→ intervention_agent      (Agent 4)

    Edges:
        entry     →(conditional)→ doc_agent | risk_agent
        doc_agent →(conditional)→ kg_agent  | risk_agent
        kg_agent  →(fixed)→ risk_agent
        risk_agent→(fixed)→ intervention_agent
        intervention_agent → END
    """
    builder = StateGraph(MediGraphState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("entry",              _entry_node)
    builder.add_node("doc_agent",          document_analysis_agent)
    builder.add_node("kg_agent",           knowledge_graph_agent)
    builder.add_node("risk_agent",         risk_prediction_agent)
    builder.add_node("intervention_agent", intervention_agent)

    # ── Entry point ───────────────────────────────────────────────────────
    builder.set_entry_point("entry")

    # ── Conditional: entry → doc_agent OR risk_agent ──────────────────────
    builder.add_conditional_edges(
        "entry",
        _route_after_entry,
        {
            "doc_agent":  "doc_agent",
            "risk_agent": "risk_agent",
        }
    )

    # ── Conditional: doc_agent → kg_agent OR risk_agent ──────────────────
    builder.add_conditional_edges(
        "doc_agent",
        _route_after_doc,
        {
            "kg_agent":   "kg_agent",
            "risk_agent": "risk_agent",
        }
    )

    # ── Fixed edges ───────────────────────────────────────────────────────
    builder.add_edge("kg_agent",           "risk_agent")
    builder.add_edge("risk_agent",         "intervention_agent")
    builder.add_edge("intervention_agent", END)

    logger.info("[Graph] StateGraph compiled successfully")
    return builder.compile()


# Compile once at module load — reused as a singleton by all API calls
_medigraph = _build_graph()


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC INVOCATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

async def run_full_pipeline(initial_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the complete 4-agent pipeline.

    Call this when a PDF is uploaded (from /api/documents/upload).
    Sets skip_document=False so all 4 agents run.

    Args:
        initial_state: Must include at minimum:
            patient_id, patient_name, file_bytes, document_type,
            adherence_rate, age, conditions, medications, exercise_level,
            follow_up_frequency, comorbidity_count, medication_count,
            hospital_visits_last_year

    Returns:
        Final LangGraph state dict with outputs from all 4 agents.
    """
    state = {**initial_state, "skip_document": False}
    logger.info(f"[Graph] run_full_pipeline → patient={state.get('patient_id')}")

    result = await _medigraph.ainvoke(state)
    return dict(result)


async def run_risk_only(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute only Agent 3 (Risk) + Agent 4 (Intervention).
    Skips document upload and Knowledge Graph ingestion.

    Call this from:
        • /api/risk/predict  (direct risk API)
        • /api/simulation/run (what-if simulation)

    Args:
        patient_data: Must include at minimum:
            patient_id, adherence_rate, age, comorbidity_count,
            medication_count, exercise_level, follow_up_frequency,
            hospital_visits_last_year

    Returns:
        Final LangGraph state dict with risk + intervention outputs.
    """
    state = {**patient_data, "skip_document": True}
    logger.info(f"[Graph] run_risk_only → patient={state.get('patient_id')}")

    result = await _medigraph.ainvoke(state)
    return dict(result)