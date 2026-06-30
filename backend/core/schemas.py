"""
backend/core/schemas.py
Purpose: All Pydantic models (request/response schemas) used across the project.
         Single source of truth for data shapes.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AdherenceLevel(str, Enum):
    EXCELLENT = "EXCELLENT"       # >= 90%
    GOOD = "GOOD"                 # 75-89%
    POOR = "POOR"                 # 50-74%
    CRITICAL = "CRITICAL"         # < 50%


# ─────────────────────────────────────────────
# PATIENT
# ─────────────────────────────────────────────

class PatientBase(BaseModel):
    patient_id: str
    name: str
    age: int = Field(..., ge=0, le=120)
    gender: str
    conditions: List[str] = []
    medications: List[str] = []


class PatientCreate(PatientBase):
    adherence_rate: float = Field(default=80.0, ge=0, le=100)
    exercise_level: int = Field(default=3, ge=1, le=10, description="1=sedentary, 10=very active")
    follow_up_frequency: int = Field(default=4, ge=0, description="visits per year")
    hba1c: Optional[float] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    cholesterol: Optional[float] = None
    bmi: Optional[float] = None
    comorbidity_count: int = 0
    medication_count: int = 0
    hospital_visits_last_year: int = 0
    age_group: Optional[str] = None


class PatientResponse(PatientBase):
    id: Optional[str] = None
    adherence_rate: float
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────
# DOCUMENT INTELLIGENCE
# ─────────────────────────────────────────────

class ExtractedEntity(BaseModel):
    diseases: List[str] = []
    symptoms: List[str] = []
    medications: List[str] = []
    lab_tests: List[str] = []
    lab_values: Dict[str, Any] = {}
    risk_factors: List[str] = []
    dosages: Dict[str, str] = {}
    instructions: List[str] = []


class DocumentAnalysisRequest(BaseModel):
    patient_id: str
    document_type: str = Field(default="prescription", description="prescription|lab_report|medical_summary")


class DocumentAnalysisResponse(BaseModel):
    patient_id: str
    document_type: str
    raw_text: str
    extracted_entities: ExtractedEntity
    summary: str
    processing_time_ms: float
    status: str = "success"


# ─────────────────────────────────────────────
# RISK PREDICTION
# ─────────────────────────────────────────────

class RiskPredictionRequest(BaseModel):
    patient_id: str
    adherence_rate: float = Field(..., ge=0, le=100)
    age: int
    comorbidity_count: int = 0
    medication_count: int = 1
    exercise_level: int = Field(default=5, ge=1, le=10)
    follow_up_frequency: int = Field(default=4, ge=0)
    hba1c: Optional[float] = None
    hospital_visits_last_year: int = 0
    bmi: Optional[float] = None


class RiskPredictionResponse(BaseModel):
    patient_id: str
    risk_score: float = Field(..., ge=0, le=100, description="0-100 risk percentage")
    risk_level: RiskLevel
    adherence_level: AdherenceLevel
    key_risk_factors: List[str]
    explanation: str
    recommendations: List[str]
    predicted_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# WHAT-IF SIMULATION
# ─────────────────────────────────────────────

class SimulationScenario(BaseModel):
    label: str
    adherence_rate: float
    exercise_level: int
    follow_up_frequency: int
    description: str


class SimulationRequest(BaseModel):
    patient_id: str
    base_data: RiskPredictionRequest
    scenarios: Optional[List[SimulationScenario]] = None  # auto-generate if None


class ScenarioResult(BaseModel):
    label: str
    description: str
    adherence_rate: float
    exercise_level: int
    follow_up_frequency: int
    risk_score: float
    risk_level: RiskLevel
    predicted_outcome: str
    ai_explanation: str


class SimulationResponse(BaseModel):
    patient_id: str
    baseline_risk: float
    scenarios: List[ScenarioResult]
    recommendation: str
    simulated_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# KNOWLEDGE GRAPH
# ─────────────────────────────────────────────

class KGNodeBase(BaseModel):
    id: str
    label: str
    type: str
    properties: Dict[str, Any] = {}


class KGRelationship(BaseModel):
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any] = {}


class KGGraphData(BaseModel):
    nodes: List[KGNodeBase]
    relationships: List[KGRelationship]
    node_count: int
    relationship_count: int


class KGIngestRequest(BaseModel):
    patient_id: str
    entities: ExtractedEntity


# ─────────────────────────────────────────────
# DRUG SAFETY
# ─────────────────────────────────────────────

class DrugSafetyRequest(BaseModel):
    drug_name: str
    patient_id: Optional[str] = None


class DrugAdverseEvent(BaseModel):
    term: str
    count: int


class DrugWarning(BaseModel):
    warning_type: str
    description: str


class DrugSafetyResponse(BaseModel):
    drug_name: str
    brand_names: List[str] = []
    generic_name: str = ""
    manufacturer: str = ""
    top_adverse_events: List[DrugAdverseEvent] = []
    warnings: List[str] = []
    contraindications: List[str] = []
    indications: List[str] = []
    total_reports: int = 0
    source: str = "OpenFDA"


# ─────────────────────────────────────────────
# AI INTERVENTION RECOMMENDATIONS
# ─────────────────────────────────────────────

class InterventionRequest(BaseModel):
    patient_id: str
    risk_score: float
    risk_level: str
    conditions: List[str]
    medications: List[str]
    adherence_rate: float
    lab_values: Dict[str, Any] = {}
    current_symptoms: List[str] = []


class InterventionPlan(BaseModel):
    priority: str  # immediate | short_term | long_term
    action: str
    rationale: str
    expected_impact: str


class InterventionResponse(BaseModel):
    patient_id: str
    risk_summary: str
    interventions: List[InterventionPlan]
    lifestyle_recommendations: List[str]
    follow_up_schedule: str
    ai_narrative: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# DASHBOARD OVERVIEW
# ─────────────────────────────────────────────

class DashboardStats(BaseModel):
    total_patients: int
    high_risk_patients: int
    documents_processed: int
    kg_nodes: int
    kg_relationships: int
    avg_adherence_rate: float
    risk_distribution: Dict[str, int]  # {LOW: N, MODERATE: N, HIGH: N, CRITICAL: N}


# ─────────────────────────────────────────────
# GENERIC API RESPONSE
# ─────────────────────────────────────────────

class APIResponse(BaseModel):
    success: bool = True
    message: str = "OK"
    data: Optional[Any] = None
    error: Optional[str] = None