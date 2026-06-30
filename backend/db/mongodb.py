"""
backend/db/mongodb.py
Purpose: Async MongoDB connection using Motor driver.
         Provides database client, collection accessors, and CRUD helpers.
"""
import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING
from bson import ObjectId
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────

_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
_db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None


async def connect_db():
    """Initialize MongoDB connection. Call on app startup."""
    global _client, _db
    try:
        _client = motor.motor_asyncio.AsyncIOMotorClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=5000
        )
        _db = _client[settings.mongodb_db_name]
        # Test connection
        await _client.admin.command("ping")
        logger.info(f"✅ Connected to MongoDB: {settings.mongodb_db_name}")

        # Create indexes
        await _create_indexes()
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        raise


async def disconnect_db():
    """Close MongoDB connection. Call on app shutdown."""
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB connection closed.")


def get_db() -> motor.motor_asyncio.AsyncIOMotorDatabase:
    """Return current database instance."""
    if _db is None:
        raise RuntimeError("Database not initialized. Call connect_db() first.")
    return _db


# ─────────────────────────────────────────────
# INDEXES
# ─────────────────────────────────────────────

async def _create_indexes():
    """Create MongoDB indexes for optimal query performance."""
    db = get_db()

    # Patients
    await db.patients.create_index([("patient_id", ASCENDING)], unique=True)
    await db.patients.create_index([("name", ASCENDING)])

    # Documents
    await db.documents.create_index([("patient_id", ASCENDING)])
    await db.documents.create_index([("uploaded_at", DESCENDING)])

    # Risk predictions
    await db.risk_predictions.create_index([("patient_id", ASCENDING)])
    await db.risk_predictions.create_index([("predicted_at", DESCENDING)])

    # Simulations
    await db.simulations.create_index([("patient_id", ASCENDING)])

    logger.info("✅ MongoDB indexes created.")


# ─────────────────────────────────────────────
# PATIENT CRUD
# ─────────────────────────────────────────────

async def create_patient(patient_data: Dict[str, Any]) -> str:
    """Insert a new patient document. Returns inserted _id as string."""
    db = get_db()
    patient_data["created_at"] = datetime.utcnow()
    patient_data["updated_at"] = datetime.utcnow()
    result = await db.patients.insert_one(patient_data)
    return str(result.inserted_id)


async def get_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    """Fetch patient by patient_id field."""
    db = get_db()
    doc = await db.patients.find_one({"patient_id": patient_id})
    if doc:
        doc["id"] = str(doc.pop("_id"))
    return doc


async def get_all_patients(limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch all patients."""
    db = get_db()
    cursor = db.patients.find({}).limit(limit).sort("created_at", DESCENDING)
    patients = []
    async for doc in cursor:
        doc["id"] = str(doc.pop("_id"))
        patients.append(doc)
    return patients


async def update_patient(patient_id: str, update_data: Dict[str, Any]) -> bool:
    """Update patient fields."""
    db = get_db()
    update_data["updated_at"] = datetime.utcnow()
    result = await db.patients.update_one(
        {"patient_id": patient_id},
        {"$set": update_data}
    )
    return result.modified_count > 0


# ─────────────────────────────────────────────
# DOCUMENT CRUD
# ─────────────────────────────────────────────

async def save_document_analysis(doc_data: Dict[str, Any]) -> str:
    """Save extracted document data."""
    db = get_db()
    doc_data["uploaded_at"] = datetime.utcnow()
    result = await db.documents.insert_one(doc_data)
    return str(result.inserted_id)


async def get_patient_documents(patient_id: str) -> List[Dict[str, Any]]:
    """Fetch all documents for a patient."""
    db = get_db()
    cursor = db.documents.find({"patient_id": patient_id}).sort("uploaded_at", DESCENDING)
    docs = []
    async for doc in cursor:
        doc["id"] = str(doc.pop("_id"))
        docs.append(doc)
    return docs


# ─────────────────────────────────────────────
# RISK PREDICTION CRUD
# ─────────────────────────────────────────────

async def save_risk_prediction(prediction: Dict[str, Any]) -> str:
    """Save a risk prediction result."""
    db = get_db()
    prediction["predicted_at"] = datetime.utcnow()
    result = await db.risk_predictions.insert_one(prediction)
    return str(result.inserted_id)


async def get_latest_risk(patient_id: str) -> Optional[Dict[str, Any]]:
    """Get the most recent risk prediction for a patient."""
    db = get_db()
    doc = await db.risk_predictions.find_one(
        {"patient_id": patient_id},
        sort=[("predicted_at", DESCENDING)]
    )
    if doc:
        doc["id"] = str(doc.pop("_id"))
    return doc


async def get_risk_history(patient_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get risk prediction history for trending."""
    db = get_db()
    cursor = db.risk_predictions.find(
        {"patient_id": patient_id}
    ).sort("predicted_at", DESCENDING).limit(limit)
    history = []
    async for doc in cursor:
        doc["id"] = str(doc.pop("_id"))
        history.append(doc)
    return history


# ─────────────────────────────────────────────
# SIMULATION CRUD
# ─────────────────────────────────────────────

async def save_simulation(simulation: Dict[str, Any]) -> str:
    """Save a what-if simulation result."""
    db = get_db()
    simulation["simulated_at"] = datetime.utcnow()
    result = await db.simulations.insert_one(simulation)
    return str(result.inserted_id)


# ─────────────────────────────────────────────
# DASHBOARD STATS
# ─────────────────────────────────────────────

async def get_dashboard_stats() -> Dict[str, Any]:
    """Aggregate statistics for the executive overview dashboard."""
    db = get_db()

    total_patients = await db.patients.count_documents({})
    docs_processed = await db.documents.count_documents({})

    # Risk distribution from latest predictions per patient
    pipeline = [
        {"$sort": {"predicted_at": -1}},
        {"$group": {
            "_id": "$patient_id",
            "risk_level": {"$first": "$risk_level"},
            "risk_score": {"$first": "$risk_score"}
        }},
        {"$group": {
            "_id": "$risk_level",
            "count": {"$sum": 1},
            "avg_score": {"$avg": "$risk_score"}
        }}
    ]
    risk_agg = await db.risk_predictions.aggregate(pipeline).to_list(length=10)
    risk_distribution = {item["_id"]: item["count"] for item in risk_agg if item["_id"]}
    high_risk = risk_distribution.get("HIGH", 0) + risk_distribution.get("CRITICAL", 0)

    # Avg adherence
    adherence_agg = await db.patients.aggregate([
        {"$group": {"_id": None, "avg": {"$avg": "$adherence_rate"}}}
    ]).to_list(length=1)
    avg_adherence = round(adherence_agg[0]["avg"], 1) if adherence_agg else 0.0

    return {
        "total_patients": total_patients,
        "high_risk_patients": high_risk,
        "documents_processed": docs_processed,
        "avg_adherence_rate": avg_adherence,
        "risk_distribution": risk_distribution
    }