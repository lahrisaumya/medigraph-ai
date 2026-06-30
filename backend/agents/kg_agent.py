"""
================================================================================
FILE:    backend/agents/kg_agent.py
AGENT:   Agent 2 — Knowledge Graph Agent
PURPOSE: Reads extracted clinical entities from the LangGraph state (produced by
         Agent 1) and upserts every entity as a typed node in Neo4j Aura, then
         creates all clinical relationships between them.

         Node types created:
             Patient, Disease, Medication, Symptom, LabTest, RiskFactor

         Relationship types created:
             HAS_DISEASE, TAKES_MEDICATION, SHOWS_SYMPTOM,
             UNDERWENT_TEST, HAS_RISK, TREATED_WITH

INPUTS  (from LangGraph state):
    patient_id          (str)   : patient identifier
    patient_name        (str)   : full name  (default "Unknown")
    age                 (int)   : patient age
    gender              (str)   : patient gender
    adherence_rate      (float) : current adherence %
    extracted_entities  (dict)  : output from document_analysis_agent
                                  Keys: diseases, medications, symptoms,
                                        lab_tests, lab_values, risk_factors, dosages

OUTPUTS (added to LangGraph state):
    kg_agent_status   (str) : "success" | "error: <message>"
    kg_nodes_created  (int) : estimated count of nodes upserted this run
================================================================================
"""

import logging
from typing import Dict, Any

from backend.db.neo4j_db import ingest_patient_to_graph

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# AGENT NODE
# ──────────────────────────────────────────────────────────────────────────────

async def knowledge_graph_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node — Knowledge Graph Agent.

    Workflow:
        1. Pull patient metadata and extracted entities from state
        2. Call ingest_patient_to_graph() which:
               a. MERGE Patient node
               b. MERGE Disease nodes  → HAS_DISEASE relationships
               c. MERGE Medication nodes → TAKES_MEDICATION relationships
               d. MERGE Symptom nodes  → SHOWS_SYMPTOM relationships
               e. MERGE LabTest nodes  → UNDERWENT_TEST relationships
               f. MERGE RiskFactor nodes → HAS_RISK relationships
               g. MERGE Disease→Medication TREATED_WITH relationships
        3. Calculate approximate node count for state
        4. Return updated state

    Error handling:
        If Neo4j is unavailable (e.g. no credentials in .env), the error is
        caught and written to state.  Downstream agents (Risk + Intervention)
        will still execute normally — the graph is enrichment, not a blocker.
    """
    patient_id = state.get("patient_id", "UNKNOWN")
    logger.info(f"[KGAgent] ▶ Starting graph ingestion for patient={patient_id}")

    try:
        # ── Step 1: Collect inputs ────────────────────────────────────────────
        entities     = state.get("extracted_entities", {})
        patient_name = state.get("patient_name", "Unknown")
        adherence    = float(state.get("adherence_rate", 80.0))

        # Build the entity dict that neo4j_db.ingest_patient_to_graph expects
        # It needs: diseases, medications, symptoms, lab_tests, lab_values,
        #           risk_factors, dosages, age, gender
        enriched_entities = {
            "diseases":     entities.get("diseases",     []),
            "medications":  entities.get("medications",  []),
            "symptoms":     entities.get("symptoms",     []),
            "lab_tests":    entities.get("lab_tests",    []),
            "lab_values":   entities.get("lab_values",   {}),
            "risk_factors": entities.get("risk_factors", []),
            "dosages":      entities.get("dosages",      {}),
            "age":          state.get("age",    0),
            "gender":       state.get("gender", "Unknown"),
        }

        _log_ingest_plan(patient_id, enriched_entities)

        # ── Step 2: Neo4j ingestion ───────────────────────────────────────────
        await ingest_patient_to_graph(
            patient_id   = patient_id,
            patient_name = patient_name,
            entities     = enriched_entities,
            adherence    = adherence,
        )

        # ── Step 3: Calculate node count ─────────────────────────────────────
        node_count = _count_nodes(enriched_entities)
        logger.info(f"[KGAgent] ✅ Upserted ~{node_count} nodes for patient={patient_id}")

        return {
            **state,
            "kg_agent_status":  "success",
            "kg_nodes_created": node_count,
        }

    except Exception as exc:
        logger.error(f"[KGAgent] ❌ Neo4j ingestion failed: {exc}", exc_info=True)
        return {
            **state,
            "kg_agent_status":  f"error: {str(exc)}",
            "kg_nodes_created": 0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _count_nodes(entities: Dict[str, Any]) -> int:
    """
    Estimate how many graph nodes were created/updated.
    Counts one Patient node + one node per extracted entity.
    """
    return (
        1                                        # Patient node
        + len(entities.get("diseases",     []))
        + len(entities.get("medications",  []))
        + len(entities.get("symptoms",     []))
        + len(entities.get("lab_tests",    []))
        + len(entities.get("risk_factors", []))
    )


def _log_ingest_plan(patient_id: str, entities: Dict[str, Any]) -> None:
    """Log what will be written to Neo4j for observability."""
    logger.info(
        f"[KGAgent] Ingesting for patient={patient_id} | "
        f"diseases={len(entities.get('diseases',[]))} | "
        f"medications={len(entities.get('medications',[]))} | "
        f"symptoms={len(entities.get('symptoms',[]))} | "
        f"lab_tests={len(entities.get('lab_tests',[]))} | "
        f"risk_factors={len(entities.get('risk_factors',[]))}"
    )