"""
================================================================================
FILE:    backend/db/neo4j_db.py
PURPOSE: Neo4j Aura async driver — all Cypher queries, node/relationship
         creation, full patient ingestion, and graph visualisation data export.

GRAPH SCHEMA:
    Nodes:         Patient  Disease  Medication  Symptom  LabTest  RiskFactor
    Relationships: HAS_DISEASE  TAKES_MEDICATION  SHOWS_SYMPTOM
                   UNDERWENT_TEST  HAS_RISK  TREATED_WITH

    All nodes use MERGE (upsert) so re-processing a document is idempotent.

CONSTRAINTS (created on connect):
    Patient.patient_id          UNIQUE
    Disease.name                UNIQUE
    Medication.name             UNIQUE
    Symptom.name                UNIQUE
    LabTest.name                UNIQUE
    RiskFactor.name             UNIQUE

USAGE:
    from backend.db.neo4j_db import connect_neo4j, ingest_patient_to_graph

    # Startup:
    await connect_neo4j()

    # Ingest entities from document agent:
    await ingest_patient_to_graph("P001", "Rajesh Kumar", entities, adherence=62.0)

    # Get graph for frontend:
    graph = await get_patient_subgraph("P001")
================================================================================
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver

from backend.core.config import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# MODULE SINGLETON
# ──────────────────────────────────────────────────────────────────────────────

_driver: Optional[AsyncDriver] = None


# ──────────────────────────────────────────────────────────────────────────────
# CONNECTION LIFECYCLE
# ──────────────────────────────────────────────────────────────────────────────

async def connect_neo4j() -> None:
    """
    Initialise the Neo4j async driver and verify connectivity.
    Call once from FastAPI lifespan startup.

    Raises RuntimeError if the Aura instance is unreachable.
    After connecting, creates all required schema constraints.
    """
    global _driver
    logger.info(f"[Neo4j] Connecting to {settings.neo4j_uri[:40]}...")

    try:
        # neo4j driver 6.x: simplified constructor, no deprecated kwargs
        uri = settings.neo4j_uri.replace("neo4j+s://", "neo4j+ssc://")
        _driver = AsyncGraphDatabase.driver(
            uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        await _driver.verify_connectivity()
        logger.info("✅ [Neo4j] Connected to Aura instance")
        await _setup_constraints()

    except Exception as exc:
        logger.error(f"❌ [Neo4j] Connection failed: {exc}")
        raise


async def disconnect_neo4j() -> None:
    """Close the Neo4j async driver. Call from FastAPI lifespan shutdown."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
        logger.info("[Neo4j] Driver closed.")


def get_driver() -> AsyncDriver:
    """
    Return the active Neo4j async driver.
    Raises RuntimeError if connect_neo4j() has not been called.
    """
    if _driver is None:
        raise RuntimeError(
            "Neo4j driver not initialised. "
            "Call connect_neo4j() first (runs automatically on app startup)."
        )
    return _driver


# ──────────────────────────────────────────────────────────────────────────────
# SCHEMA CONSTRAINTS
# ──────────────────────────────────────────────────────────────────────────────

async def _setup_constraints() -> None:
    """
    Create uniqueness constraints for all node types.
    Uses IF NOT EXISTS so safe to run on every startup.
    """
    driver = get_driver()
    constraints = [
        "CREATE CONSTRAINT patient_id   IF NOT EXISTS FOR (p:Patient)    REQUIRE p.patient_id IS UNIQUE",
        "CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease)    REQUIRE d.name       IS UNIQUE",
        "CREATE CONSTRAINT med_name     IF NOT EXISTS FOR (m:Medication) REQUIRE m.name       IS UNIQUE",
        "CREATE CONSTRAINT sym_name     IF NOT EXISTS FOR (s:Symptom)    REQUIRE s.name       IS UNIQUE",
        "CREATE CONSTRAINT lab_name     IF NOT EXISTS FOR (l:LabTest)    REQUIRE l.name       IS UNIQUE",
        "CREATE CONSTRAINT risk_name    IF NOT EXISTS FOR (r:RiskFactor) REQUIRE r.name       IS UNIQUE",
    ]
    async with driver.session() as session:
        for cypher in constraints:
            try:
                await session.run(cypher)
            except Exception as exc:
                # Constraint may already exist — not an error
                logger.debug(f"[Neo4j] Constraint note: {exc}")

    logger.info("✅ [Neo4j] Schema constraints verified.")


# ──────────────────────────────────────────────────────────────────────────────
# NODE UPSERTS  (MERGE = create-or-update, fully idempotent)
# ──────────────────────────────────────────────────────────────────────────────

async def upsert_patient_node(patient_data: Dict[str, Any]) -> None:
    """
    Create or update a Patient node.
    Only writes fields that are present and non-None.
    """
    driver = get_driver()
    cypher = """
    MERGE (p:Patient {patient_id: $patient_id})
    SET   p.name           = $name,
          p.age            = $age,
          p.gender         = $gender,
          p.adherence_rate = $adherence_rate,
          p.updated_at     = $updated_at
    RETURN p
    """
    async with driver.session() as session:
        await session.run(cypher, {
            "patient_id":    patient_data.get("patient_id", "UNKNOWN"),
            "name":          patient_data.get("name",          "Unknown"),
            "age":           patient_data.get("age",           0),
            "gender":        patient_data.get("gender",        "Unknown"),
            "adherence_rate":float(patient_data.get("adherence_rate", 0.0)),
            "updated_at":    _now(),
        })


async def upsert_disease_node(name: str) -> None:
    """Create or update a Disease node with auto-classified category."""
    _CHRONIC = {"diabetes", "hypertension", "copd", "asthma", "heart failure",
                "chronic kidney", "ckd", "coronary", "hyperlipidemia", "obesity"}
    category = "chronic" if any(c in name.lower() for c in _CHRONIC) else "acute"

    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            "MERGE (d:Disease {name: $name}) SET d.category = $category RETURN d",
            {"name": name.lower(), "category": category},
        )


async def upsert_medication_node(name: str, dosage: str = "") -> None:
    """Create or update a Medication node."""
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            "MERGE (m:Medication {name: $name}) SET m.dosage = $dosage RETURN m",
            {"name": name.lower(), "dosage": dosage},
        )


async def upsert_symptom_node(name: str) -> None:
    """Create or update a Symptom node."""
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            "MERGE (s:Symptom {name: $name}) RETURN s",
            {"name": name.lower()},
        )


async def upsert_lab_test_node(name: str, last_value: str = "") -> None:
    """Create or update a LabTest node, storing the most recent value."""
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            "MERGE (l:LabTest {name: $name}) SET l.last_value = $value RETURN l",
            {"name": name.lower(), "value": str(last_value)},
        )


async def upsert_risk_factor_node(name: str) -> None:
    """Create or update a RiskFactor node."""
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            "MERGE (r:RiskFactor {name: $name}) RETURN r",
            {"name": name.lower()},
        )


# ──────────────────────────────────────────────────────────────────────────────
# RELATIONSHIP CREATION  (all use MERGE — idempotent)
# ──────────────────────────────────────────────────────────────────────────────

async def create_patient_disease_rel(patient_id: str, disease_name: str) -> None:
    """MERGE (Patient)-[:HAS_DISEASE]->(Disease)"""
    driver = get_driver()
    async with driver.session() as session:
        await session.run("""
            MATCH (p:Patient  {patient_id: $pid})
            MATCH (d:Disease  {name:       $name})
            MERGE (p)-[r:HAS_DISEASE]->(d)
            SET   r.since = $since
        """, {"pid": patient_id, "name": disease_name.lower(), "since": _now()})


async def create_patient_medication_rel(
    patient_id: str, med_name: str, adherence: float = 80.0
) -> None:
    """MERGE (Patient)-[:TAKES_MEDICATION {adherence_rate}]->(Medication)"""
    driver = get_driver()
    async with driver.session() as session:
        await session.run("""
            MATCH (p:Patient    {patient_id: $pid})
            MATCH (m:Medication {name:       $name})
            MERGE (p)-[r:TAKES_MEDICATION]->(m)
            SET   r.adherence_rate = $adherence,
                  r.started        = $started
        """, {"pid": patient_id, "name": med_name.lower(),
               "adherence": adherence, "started": _now()})


async def create_patient_symptom_rel(patient_id: str, symptom_name: str) -> None:
    """MERGE (Patient)-[:SHOWS_SYMPTOM]->(Symptom)"""
    driver = get_driver()
    async with driver.session() as session:
        await session.run("""
            MATCH (p:Patient {patient_id: $pid})
            MATCH (s:Symptom {name:       $name})
            MERGE (p)-[r:SHOWS_SYMPTOM]->(s)
            SET   r.reported_at = $ts
        """, {"pid": patient_id, "name": symptom_name.lower(), "ts": _now()})


async def create_patient_lab_rel(
    patient_id: str, test_name: str, value: str = ""
) -> None:
    """MERGE (Patient)-[:UNDERWENT_TEST {value}]->(LabTest)"""
    driver = get_driver()
    async with driver.session() as session:
        await session.run("""
            MATCH (p:Patient {patient_id: $pid})
            MATCH (l:LabTest {name:       $name})
            MERGE (p)-[r:UNDERWENT_TEST]->(l)
            SET   r.value     = $value,
                  r.tested_at = $ts
        """, {"pid": patient_id, "name": test_name.lower(),
               "value": str(value), "ts": _now()})


async def create_patient_risk_rel(
    patient_id: str, risk_factor: str, score: float = 0.0
) -> None:
    """MERGE (Patient)-[:HAS_RISK {score}]->(RiskFactor)"""
    driver = get_driver()
    async with driver.session() as session:
        await session.run("""
            MATCH (p:Patient    {patient_id: $pid})
            MATCH (r:RiskFactor {name:       $name})
            MERGE (p)-[rel:HAS_RISK]->(r)
            SET   rel.score       = $score,
                  rel.assessed_at = $ts
        """, {"pid": patient_id, "name": risk_factor.lower(),
               "score": score, "ts": _now()})


async def create_disease_medication_rel(disease_name: str, med_name: str) -> None:
    """MERGE (Disease)-[:TREATED_WITH]->(Medication)"""
    driver = get_driver()
    async with driver.session() as session:
        await session.run("""
            MATCH (d:Disease    {name: $dname})
            MATCH (m:Medication {name: $mname})
            MERGE (d)-[:TREATED_WITH]->(m)
        """, {"dname": disease_name.lower(), "mname": med_name.lower()})


# ──────────────────────────────────────────────────────────────────────────────
# FULL PATIENT INGESTION
# ──────────────────────────────────────────────────────────────────────────────

async def ingest_patient_to_graph(
    patient_id:   str,
    patient_name: str,
    entities:     Dict[str, Any],
    adherence:    float = 80.0,
) -> None:
    """
    Perform a complete graph ingestion for one patient's extracted entities.
    All operations use MERGE so this is fully idempotent — safe to call
    multiple times for the same patient.

    Steps:
        1. Upsert Patient node
        2. Upsert + link Disease nodes      (HAS_DISEASE)
        3. Upsert + link Medication nodes   (TAKES_MEDICATION)
        4. Upsert + link Symptom nodes      (SHOWS_SYMPTOM)
        5. Upsert + link LabTest nodes      (UNDERWENT_TEST)
        6. Upsert + link RiskFactor nodes   (HAS_RISK)
        7. Cross-link Disease → Medication  (TREATED_WITH)

    Args:
        patient_id:   Patient identifier, e.g. "P001"
        patient_name: Full name for the node label
        entities:     Dict from extract_medical_entities() — must have keys:
                      diseases, medications, symptoms, lab_tests, lab_values,
                      risk_factors, dosages  (all optional — missing → skipped)
        adherence:    Current medication adherence % (written to TAKES_MEDICATION edge)
    """
    logger.info(f"[Neo4j/ingest] Starting ingestion for patient={patient_id}")

    # 1. Patient node
    await upsert_patient_node({
        "patient_id":    patient_id,
        "name":          patient_name,
        "age":           entities.get("age",    0),
        "gender":        entities.get("gender", "Unknown"),
        "adherence_rate": adherence,
    })

    # 2. Diseases
    for disease in entities.get("diseases", []):
        if disease:
            await upsert_disease_node(disease)
            await create_patient_disease_rel(patient_id, disease)

    # 3. Medications
    dosages = entities.get("dosages", {})
    for med in entities.get("medications", []):
        if med:
            dosage = dosages.get(med, dosages.get(med.lower(), ""))
            await upsert_medication_node(med, dosage)
            await create_patient_medication_rel(patient_id, med, adherence)

    # 4. Symptoms
    for symptom in entities.get("symptoms", []):
        if symptom:
            await upsert_symptom_node(symptom)
            await create_patient_symptom_rel(patient_id, symptom)

    # 5. Lab Tests
    lab_values = entities.get("lab_values", {})
    for test in entities.get("lab_tests", []):
        if test:
            value = lab_values.get(test, lab_values.get(test.lower(), ""))
            await upsert_lab_test_node(test, str(value))
            await create_patient_lab_rel(patient_id, test, str(value))

    # 6. Risk Factors
    for risk in entities.get("risk_factors", []):
        if risk:
            await upsert_risk_factor_node(risk)
            await create_patient_risk_rel(patient_id, risk)

    # 7. Cross-link Disease → Medication (TREATED_WITH)
    diseases    = entities.get("diseases",    [])
    medications = entities.get("medications", [])
    for disease in diseases:
        for med in medications:
            if disease and med:
                await create_disease_medication_rel(disease, med)

    logger.info(
        f"[Neo4j/ingest] ✅ patient={patient_id} | "
        f"diseases={len(diseases)} | medications={len(medications)} | "
        f"symptoms={len(entities.get('symptoms',[]))} | "
        f"lab_tests={len(entities.get('lab_tests',[]))}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH QUERY FUNCTIONS  (for API endpoints and dashboard)
# ──────────────────────────────────────────────────────────────────────────────

async def get_patient_subgraph(patient_id: str) -> Dict[str, Any]:
    """
    Fetch all nodes and relationships centred on one patient.
    Includes:
        - Direct connections: Disease, Medication, Symptom, LabTest, RiskFactor
        - Second-hop: Medications connected to the patient's diseases (TREATED_WITH)

    Returns a dict in the standard graph format:
    {
        "nodes": [ {id, label, type, properties}, ... ],
        "relationships": [ {source_id, target_id, relationship_type, properties}, ... ],
        "node_count": int,
        "relationship_count": int
    }
    """
    driver = get_driver()

    # Fetch direct patient connections
    cypher_direct = """
    MATCH (p:Patient {patient_id: $patient_id})-[r]->(n)
    RETURN p, r, n, type(r) AS rel_type, labels(n) AS node_labels
    """
    # Fetch second-hop: disease → medication (TREATED_WITH)
    cypher_second_hop = """
    MATCH (p:Patient {patient_id: $patient_id})-[:HAS_DISEASE]->(d:Disease)-[r2:TREATED_WITH]->(m:Medication)
    RETURN d AS p, r2 AS r, m AS n, type(r2) AS rel_type, labels(m) AS node_labels
    """

    nodes         = {}
    relationships = []

    async with driver.session() as session:
        for cypher in (cypher_direct, cypher_second_hop):
            result = await session.run(cypher, {"patient_id": patient_id})
            async for record in result:
                src    = record["p"]
                tgt    = record["n"]
                rel    = record["r"]
                src_id = str(src.id)
                tgt_id = str(tgt.id)

                if src_id not in nodes:
                    nodes[src_id] = _format_node(src)

                if tgt_id not in nodes:
                    props = dict(tgt)
                    label_list = record.get("node_labels") or []
                    nodes[tgt_id] = {
                        "id":         tgt_id,
                        "label":      props.get("name", tgt_id),
                        "type":       label_list[0] if label_list else "Unknown",
                        "properties": props,
                    }

                relationships.append({
                    "source_id":         src_id,
                    "target_id":         tgt_id,
                    "relationship_type": record["rel_type"],
                    "properties":        dict(rel),
                })

    return {
        "nodes":               list(nodes.values()),
        "relationships":       relationships,
        "node_count":          len(nodes),
        "relationship_count":  len(relationships),
    }


async def get_full_graph_summary() -> Dict[str, Any]:
    """
    Return aggregate counts of all nodes and relationships in the graph.
    Used by the dashboard overview and graph analytics pages.
    """
    driver = get_driver()

    node_counts = {}
    rel_counts  = {}

    async with driver.session() as session:
        # Node counts per label
        r1 = await session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC"
        )
        async for row in r1:
            node_counts[row["label"]] = row["cnt"]

        # Relationship counts per type
        r2 = await session.run(
            "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt"
        )
        async for row in r2:
            rel_counts[row["rel_type"]] = row["cnt"]

    return {
        "node_counts":          node_counts,
        "relationship_counts":  rel_counts,
        "total_nodes":          sum(node_counts.values()),
        "total_relationships":  sum(rel_counts.values()),
    }


async def get_high_risk_patients_graph() -> Dict[str, Any]:
    """
    Return a graph of patients with adherence_rate < 70 and their diseases.
    Used by the risk prediction dashboard — population view.
    """
    driver = get_driver()
    cypher = """
    MATCH (p:Patient)-[:HAS_DISEASE]->(d:Disease)
    WHERE p.adherence_rate < 70
    RETURN p, d
    LIMIT 60
    """
    nodes         = {}
    relationships = []

    async with driver.session() as session:
        result = await session.run(cypher)
        async for record in result:
            p    = record["p"]
            d    = record["d"]
            p_id = str(p.id)
            d_id = str(d.id)

            if p_id not in nodes:
                nodes[p_id] = _format_node(p)
            if d_id not in nodes:
                nodes[d_id] = _format_node(d)

            relationships.append({
                "source_id":         p_id,
                "target_id":         d_id,
                "relationship_type": "HAS_DISEASE",
                "properties":        {},
            })

    return {
        "nodes":               list(nodes.values()),
        "relationships":       relationships,
        "node_count":          len(nodes),
        "relationship_count":  len(relationships),
    }


async def run_custom_cypher(
    cypher: str, params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Execute any read-only Cypher query and return results as a list of dicts.
    For use by advanced analytics endpoints.

    Args:
        cypher: Read-only Cypher query string.
        params: Query parameters dict.

    Returns:
        List of record dicts (one per row).
    """
    driver = get_driver()
    results = []
    async with driver.session() as session:
        result = await session.run(cypher, params or {})
        async for record in result:
            results.append(dict(record))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    """Return current UTC time as ISO 8601 string for Neo4j property values."""
    return datetime.utcnow().isoformat()


def _format_node(neo_node) -> Dict[str, Any]:
    """
    Convert a Neo4j node object to the standard frontend-ready dict format.
    Handles both Patient nodes (use patient_id as label) and named nodes
    (use name property as label).
    """
    props    = dict(neo_node)
    label    = props.get("name", props.get("patient_id", str(neo_node.id)))
    node_type = list(neo_node.labels)[0] if neo_node.labels else "Unknown"
    return {
        "id":         str(neo_node.id),
        "label":      label,
        "type":       node_type,
        "properties": props,
    }