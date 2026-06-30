"""
================================================================================
FILE:    backend/api/graph.py
PURPOSE: Healthcare Knowledge Graph API — exposes Neo4j graph data for the
         dashboard visualization page and graph analytics.

ENDPOINTS:
    GET  /api/graph/patient/{patient_id}     — Patient subgraph (nodes + rels)
    GET  /api/graph/summary                  — Full graph node/rel counts
    GET  /api/graph/high-risk                — High-risk patient graph
    GET  /api/graph/disease/{disease_name}   — All patients with a disease
    GET  /api/graph/medication/{med_name}    — All patients on a medication
    POST /api/graph/ingest                   — Manually ingest entities → graph
    GET  /api/graph/stats                    — Detailed graph analytics

VISUALIZATION FORMAT:
    All graph endpoints return a standard format:
    {
      "nodes":         [{ "id", "label", "type", "properties" }, ...],
      "relationships": [{ "source_id", "target_id", "relationship_type", "properties" }, ...],
      "node_count":    int,
      "relationship_count": int
    }
    This format is directly consumable by:
    - D3.js force-directed graphs
    - Neovis.js Neo4j visualizer
    - Plotly network graphs

DEPENDENCIES:
    backend.db.neo4j_db   → all Neo4j query functions
    backend.core.schemas  → APIResponse, KGIngestRequest
================================================================================
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.db.neo4j_db import (
    get_patient_subgraph,
    get_full_graph_summary,
    get_high_risk_patients_graph,
    ingest_patient_to_graph,
    get_driver,
)
from backend.core.schemas import APIResponse, KGIngestRequest

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/graph",
    tags=["Knowledge Graph"],
)


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/graph/patient/{patient_id}
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/patient/{patient_id}",
    response_model=APIResponse,
    summary="Get full patient subgraph — all nodes and relationships",
)
async def patient_graph(patient_id: str):
    """
    Return the complete Neo4j subgraph for a patient including:
    - Patient node
    - All connected Disease, Medication, Symptom, LabTest, RiskFactor nodes
    - All typed relationships (HAS_DISEASE, TAKES_MEDICATION, etc.)
    - Second-hop nodes (e.g. Diseases connected to the patient's Medications)

    **Used by:** Dashboard Knowledge Graph page — patient-focused view.

    Returns standard graph format (nodes + relationships arrays) ready for
    D3.js or Neovis.js rendering.
    """
    logger.info(f"[graph/patient] Fetching subgraph for patient={patient_id}")

    try:
        data = await get_patient_subgraph(patient_id)
    except Exception as exc:
        logger.error(f"[graph/patient] Neo4j error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Graph query failed: {str(exc)}")

    if data["node_count"] == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No graph data found for patient '{patient_id}'. "
                f"Upload a document first to populate the knowledge graph."
            ),
        )

    logger.info(
        f"[graph/patient] ✅ patient={patient_id} | "
        f"nodes={data['node_count']} | rels={data['relationship_count']}"
    )

    return APIResponse(
        message=f"Subgraph retrieved for patient '{patient_id}'",
        data=data,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/graph/summary
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/summary",
    response_model=APIResponse,
    summary="Full graph statistics — node and relationship type counts",
)
async def graph_summary():
    """
    Return aggregate counts across the entire Neo4j knowledge graph.

    Response includes:
    - **node_counts**: count per node type (Patient, Disease, Medication, …)
    - **relationship_counts**: count per relationship type (HAS_DISEASE, …)
    - **total_nodes**: sum of all nodes
    - **total_relationships**: sum of all relationships

    **Used by:** Executive Overview dashboard — Knowledge Graph KPI cards.
    """
    logger.info("[graph/summary] Fetching full graph summary")

    try:
        data = await get_full_graph_summary()
    except Exception as exc:
        logger.error(f"[graph/summary] Neo4j error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Graph summary failed: {str(exc)}")

    return APIResponse(
        message="Graph summary retrieved",
        data=data,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/graph/high-risk
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/high-risk",
    response_model=APIResponse,
    summary="Graph of all high-risk patients and their primary conditions",
)
async def high_risk_graph(
    adherence_threshold: float = Query(
        default=70.0,
        ge=0.0,
        le=100.0,
        description="Patients with adherence below this % are considered high-risk",
    ),
):
    """
    Return a graph showing all patients whose adherence rate is below the
    threshold, connected to their primary disease nodes.

    **Used by:** Risk Prediction dashboard — population-level graph view.

    The threshold defaults to 70% (clinically accepted poor adherence cutoff).
    """
    logger.info(f"[graph/high-risk] threshold={adherence_threshold}%")

    try:
        data = await get_high_risk_patients_graph()
    except Exception as exc:
        logger.error(f"[graph/high-risk] Neo4j error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"High-risk graph failed: {str(exc)}")

    return APIResponse(
        message=f"High-risk patient graph (adherence < {adherence_threshold}%)",
        data=data,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/graph/disease/{disease_name}
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/disease/{disease_name}",
    response_model=APIResponse,
    summary="All patients diagnosed with a specific disease",
)
async def disease_subgraph(
    disease_name: str,
    limit: int = Query(default=30, ge=1, le=100),
):
    """
    Return a graph centred on one disease node, showing all connected patients
    and the medications used to treat that disease.

    **Use case:** "Show me all patients with type 2 diabetes and what they're taking."

    Useful for population health analytics and care gap identification.
    """
    logger.info(f"[graph/disease] disease='{disease_name}' limit={limit}")

    query = """
    MATCH (p:Patient)-[:HAS_DISEASE]->(d:Disease {name: $disease_name})
    OPTIONAL MATCH (d)-[:TREATED_WITH]->(m:Medication)
    OPTIONAL MATCH (p)-[:TAKES_MEDICATION]->(m2:Medication)
    RETURN p, d, m, m2
    LIMIT $limit
    """

    try:
        driver = get_driver()
        nodes  = {}
        rels   = []

        async with driver.session() as session:
            result = await session.run(
                query,
                {"disease_name": disease_name.lower(), "limit": limit},
            )
            async for record in result:
                _add_node(nodes, record["p"], "Patient")
                _add_node(nodes, record["d"], "Disease")
                if record["m"]:
                    _add_node(nodes, record["m"], "Medication")
                    rels.append(_make_rel(record["d"], record["m"], "TREATED_WITH"))
                if record["m2"]:
                    _add_node(nodes, record["m2"], "Medication")
                    rels.append(_make_rel(record["p"], record["m2"], "TAKES_MEDICATION"))
                rels.append(_make_rel(record["p"], record["d"], "HAS_DISEASE"))

    except Exception as exc:
        logger.error(f"[graph/disease] Neo4j error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    if not nodes:
        raise HTTPException(
            status_code=404,
            detail=f"No patients found with disease '{disease_name}'",
        )

    data = {
        "nodes":               list(nodes.values()),
        "relationships":       _dedup_rels(rels),
        "node_count":          len(nodes),
        "relationship_count":  len(rels),
        "disease_name":        disease_name,
    }

    return APIResponse(
        message=f"Disease subgraph for '{disease_name}'",
        data=data,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/graph/medication/{med_name}
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/medication/{med_name}",
    response_model=APIResponse,
    summary="All patients taking a specific medication",
)
async def medication_subgraph(
    med_name: str,
    limit: int = Query(default=30, ge=1, le=100),
):
    """
    Return a graph centred on one medication node, showing all patients
    currently taking it and the diseases it treats.

    **Use case:** "Show me all patients on metformin and their conditions."

    The adherence_rate property on TAKES_MEDICATION edges encodes
    each patient's individual adherence to that specific drug.
    """
    logger.info(f"[graph/medication] med='{med_name}' limit={limit}")

    query = """
    MATCH (p:Patient)-[r:TAKES_MEDICATION]->(m:Medication {name: $med_name})
    OPTIONAL MATCH (d:Disease)-[:TREATED_WITH]->(m)
    OPTIONAL MATCH (p)-[:HAS_DISEASE]->(d2:Disease)
    RETURN p, r, m, d, d2
    LIMIT $limit
    """

    try:
        driver = get_driver()
        nodes  = {}
        rels   = []

        async with driver.session() as session:
            result = await session.run(
                query,
                {"med_name": med_name.lower(), "limit": limit},
            )
            async for record in result:
                _add_node(nodes, record["p"], "Patient")
                _add_node(nodes, record["m"], "Medication")
                rels.append(_make_rel(record["p"], record["m"], "TAKES_MEDICATION",
                                      dict(record["r"])))
                if record["d"]:
                    _add_node(nodes, record["d"], "Disease")
                    rels.append(_make_rel(record["d"], record["m"], "TREATED_WITH"))
                if record["d2"]:
                    _add_node(nodes, record["d2"], "Disease")
                    rels.append(_make_rel(record["p"], record["d2"], "HAS_DISEASE"))

    except Exception as exc:
        logger.error(f"[graph/medication] Neo4j error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    if not nodes:
        raise HTTPException(
            status_code=404,
            detail=f"No patients found taking medication '{med_name}'",
        )

    data = {
        "nodes":              list(nodes.values()),
        "relationships":      _dedup_rels(rels),
        "node_count":         len(nodes),
        "relationship_count": len(rels),
        "medication_name":    med_name,
    }

    return APIResponse(
        message=f"Medication subgraph for '{med_name}'",
        data=data,
    )


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/graph/ingest
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=APIResponse,
    summary="Manually ingest extracted entities into the knowledge graph",
)
async def ingest_entities(request: KGIngestRequest):
    """
    Directly ingest a set of clinical entities into Neo4j without
    going through the full document upload pipeline.

    **Use case:**
    - Seeding the graph from external data sources
    - Manually correcting or enriching extracted entities
    - Testing graph ingestion independently

    Accepts the same entity format as the document pipeline output.
    """
    logger.info(f"[graph/ingest] patient={request.patient_id}")

    entities = request.entities.model_dump()

    try:
        await ingest_patient_to_graph(
            patient_id   = request.patient_id,
            patient_name = entities.get("patient_name", "Unknown"),
            entities     = entities,
            adherence    = 80.0,
        )
    except Exception as exc:
        logger.error(f"[graph/ingest] Ingestion failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Graph ingestion failed: {str(exc)}")

    node_count = (
        1
        + len(entities.get("diseases",     []))
        + len(entities.get("medications",  []))
        + len(entities.get("symptoms",     []))
        + len(entities.get("lab_tests",    []))
        + len(entities.get("risk_factors", []))
    )

    return APIResponse(
        message=f"Entities ingested for patient '{request.patient_id}'",
        data={
            "patient_id":    request.patient_id,
            "nodes_created": node_count,
            "entity_counts": {
                "diseases":     len(entities.get("diseases",     [])),
                "medications":  len(entities.get("medications",  [])),
                "symptoms":     len(entities.get("symptoms",     [])),
                "lab_tests":    len(entities.get("lab_tests",    [])),
                "risk_factors": len(entities.get("risk_factors", [])),
            },
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/graph/stats
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/stats",
    response_model=APIResponse,
    summary="Detailed graph analytics — density, connectivity, top entities",
)
async def graph_analytics():
    """
    Return detailed analytics about the knowledge graph:
    - Node and relationship counts per type
    - Most connected diseases (by patient count)
    - Most prescribed medications (by patient count)
    - Average connections per patient node
    - Graph density metric

    **Used by:** Knowledge Graph dashboard — analytics panels.
    """
    logger.info("[graph/stats] Computing graph analytics")

    try:
        driver = get_driver()

        # Top diseases by patient count
        disease_q = """
        MATCH (p:Patient)-[:HAS_DISEASE]->(d:Disease)
        RETURN d.name AS disease, count(p) AS patient_count
        ORDER BY patient_count DESC LIMIT 10
        """

        # Top medications by patient count
        med_q = """
        MATCH (p:Patient)-[r:TAKES_MEDICATION]->(m:Medication)
        RETURN m.name AS medication, count(p) AS patient_count,
               avg(r.adherence_rate) AS avg_adherence
        ORDER BY patient_count DESC LIMIT 10
        """

        # Avg connections per patient
        avg_q = """
        MATCH (p:Patient)-[r]->()
        RETURN p.patient_id AS patient_id, count(r) AS connections
        ORDER BY connections DESC
        """

        top_diseases    = []
        top_medications = []
        patient_connectivity = []

        async with driver.session() as session:
            r1 = await session.run(disease_q)
            async for row in r1:
                top_diseases.append({
                    "disease":       row["disease"],
                    "patient_count": row["patient_count"],
                })

            r2 = await session.run(med_q)
            async for row in r2:
                top_medications.append({
                    "medication":    row["medication"],
                    "patient_count": row["patient_count"],
                    "avg_adherence": round(row["avg_adherence"] or 0, 1),
                })

            r3 = await session.run(avg_q)
            async for row in r3:
                patient_connectivity.append({
                    "patient_id":  row["patient_id"],
                    "connections": row["connections"],
                })

        summary = await get_full_graph_summary()
        total_nodes = summary.get("total_nodes", 0)
        total_rels  = summary.get("total_relationships", 0)
        n_patients  = summary.get("node_counts", {}).get("Patient", 0)
        avg_conn    = (
            round(sum(p["connections"] for p in patient_connectivity) / n_patients, 1)
            if n_patients else 0
        )

    except Exception as exc:
        logger.error(f"[graph/stats] Neo4j error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return APIResponse(
        message="Graph analytics computed",
        data={
            "total_nodes":            total_nodes,
            "total_relationships":    total_rels,
            "node_type_counts":       summary.get("node_counts",        {}),
            "relationship_type_counts": summary.get("relationship_counts", {}),
            "top_diseases":           top_diseases,
            "top_medications":        top_medications,
            "patient_connectivity":   patient_connectivity[:20],
            "avg_connections_per_patient": avg_conn,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _add_node(nodes: dict, neo_node, default_type: str) -> None:
    """Add a Neo4j node to the nodes dict if not already present."""
    if neo_node is None:
        return
    node_id = str(neo_node.id)
    if node_id not in nodes:
        props = dict(neo_node)
        nodes[node_id] = {
            "id":         node_id,
            "label":      props.get("name", props.get("patient_id", node_id)),
            "type":       list(neo_node.labels)[0] if neo_node.labels else default_type,
            "properties": props,
        }


def _make_rel(src, tgt, rel_type: str, props: dict = None) -> dict:
    """Build a relationship dict from two Neo4j nodes."""
    return {
        "source_id":         str(src.id),
        "target_id":         str(tgt.id),
        "relationship_type": rel_type,
        "properties":        props or {},
    }


def _dedup_rels(rels: list) -> list:
    """Remove duplicate relationships (same source, target, type)."""
    seen = set()
    unique = []
    for r in rels:
        key = (r["source_id"], r["target_id"], r["relationship_type"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique