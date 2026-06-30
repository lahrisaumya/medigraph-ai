"""
================================================================================
FILE:    scripts/seed_mongodb.py
PURPOSE: Seed MongoDB Atlas with realistic demo data for MediGraph AI.
         Creates patients, risk predictions, simulation records, and
         document processing history so the dashboard has meaningful
         data to display on first run.

USAGE:
    # Run from the project root directory
    python scripts/seed_mongodb.py

    # To reseed (clears existing data first):
    python scripts/seed_mongodb.py --reset

WHAT GETS SEEDED:
    Collection          Records   Description
    ─────────────────────────────────────────────────────
    patients                7     Full clinical profiles (P001–P007)
    risk_predictions       14     2 predictions per patient (for trend chart)
    simulations             3     What-If simulation runs for high-risk patients
    documents               5     Sample processed document records

PATIENTS SEEDED:
    P001  Rajesh Kumar    58M  HIGH      62% adherence — diabetes + HTN + lipids
    P002  Priya Sharma    45F  LOW       88% adherence — asthma only
    P003  Amit Verma      67M  CRITICAL  45% adherence — cardiac + heart failure
    P004  Sunita Patel    52F  MODERATE  75% adherence — diabetes + CKD
    P005  Vikram Singh    71M  CRITICAL  35% adherence — 4 conditions, polypharmacy
    P006  Ananya Reddy    34F  LOW       92% adherence — asthma, very healthy
    P007  Suresh Nair     63M  HIGH      55% adherence — diabetes + HTN + HF
================================================================================
"""

import asyncio
import os
import random
import sys
from datetime import datetime, timedelta

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from backend.db.mongodb import connect_db, get_db, disconnect_db


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_PATIENTS = [
    {
        # ── P001: High Risk — Diabetic with poor adherence ────────────────────
        "patient_id":               "P001",
        "name":                     "Rajesh Kumar",
        "age":                      58,
        "gender":                   "Male",
        "conditions":               ["type 2 diabetes", "hypertension", "hyperlipidemia"],
        "medications":              ["metformin", "lisinopril", "atorvastatin"],
        "adherence_rate":           62.0,
        "exercise_level":           2,
        "follow_up_frequency":      3,
        "comorbidity_count":        3,
        "medication_count":         3,
        "hospital_visits_last_year":2,
        "hba1c":                    8.5,
        "blood_pressure_systolic":  148,
        "blood_pressure_diastolic": 92,
        "cholesterol":              215.0,
        "bmi":                      28.4,
        "age_group":                "senior",
        "deleted":                  False,
    },
    {
        # ── P002: Low Risk — Well-controlled asthma ────────────────────────────
        "patient_id":               "P002",
        "name":                     "Priya Sharma",
        "age":                      45,
        "gender":                   "Female",
        "conditions":               ["asthma"],
        "medications":              ["salbutamol inhaler"],
        "adherence_rate":           88.0,
        "exercise_level":           7,
        "follow_up_frequency":      6,
        "comorbidity_count":        1,
        "medication_count":         1,
        "hospital_visits_last_year":0,
        "hba1c":                    5.4,
        "blood_pressure_systolic":  118,
        "blood_pressure_diastolic": 76,
        "cholesterol":              165.0,
        "bmi":                      22.1,
        "age_group":                "middle",
        "deleted":                  False,
    },
    {
        # ── P003: Critical Risk — Cardiac failure, very poor adherence ─────────
        "patient_id":               "P003",
        "name":                     "Amit Verma",
        "age":                      67,
        "gender":                   "Male",
        "conditions":               ["coronary artery disease", "heart failure", "hypertension"],
        "medications":              ["aspirin", "furosemide", "amlodipine"],
        "adherence_rate":           45.0,
        "exercise_level":           1,
        "follow_up_frequency":      2,
        "comorbidity_count":        3,
        "medication_count":         3,
        "hospital_visits_last_year":4,
        "hba1c":                    6.1,
        "blood_pressure_systolic":  162,
        "blood_pressure_diastolic": 98,
        "cholesterol":              240.0,
        "bmi":                      31.2,
        "age_group":                "senior",
        "deleted":                  False,
    },
    {
        # ── P004: Moderate Risk — Diabetic CKD, managed but CKD progressing ────
        "patient_id":               "P004",
        "name":                     "Sunita Patel",
        "age":                      52,
        "gender":                   "Female",
        "conditions":               ["type 2 diabetes", "chronic kidney disease"],
        "medications":              ["metformin", "insulin glargine"],
        "adherence_rate":           75.0,
        "exercise_level":           4,
        "follow_up_frequency":      8,
        "comorbidity_count":        2,
        "medication_count":         2,
        "hospital_visits_last_year":1,
        "hba1c":                    7.2,
        "blood_pressure_systolic":  130,
        "blood_pressure_diastolic": 84,
        "cholesterol":              188.0,
        "bmi":                      26.8,
        "age_group":                "senior",
        "deleted":                  False,
    },
    {
        # ── P005: Critical Risk — Polypharmacy, very high HbA1c ──────────────
        "patient_id":               "P005",
        "name":                     "Vikram Singh",
        "age":                      71,
        "gender":                   "Male",
        "conditions":               [
            "hypertension", "coronary artery disease",
            "hyperlipidemia", "type 2 diabetes"
        ],
        "medications":              [
            "lisinopril", "aspirin", "atorvastatin",
            "metformin", "insulin glargine"
        ],
        "adherence_rate":           35.0,
        "exercise_level":           1,
        "follow_up_frequency":      1,
        "comorbidity_count":        4,
        "medication_count":         5,
        "hospital_visits_last_year":6,
        "hba1c":                    9.8,
        "blood_pressure_systolic":  175,
        "blood_pressure_diastolic": 105,
        "cholesterol":              268.0,
        "bmi":                      33.7,
        "age_group":                "elderly",
        "deleted":                  False,
    },
    {
        # ── P006: Low Risk — Young, healthy, excellent adherence ───────────────
        "patient_id":               "P006",
        "name":                     "Ananya Reddy",
        "age":                      34,
        "gender":                   "Female",
        "conditions":               ["asthma"],
        "medications":              ["salbutamol inhaler"],
        "adherence_rate":           92.0,
        "exercise_level":           8,
        "follow_up_frequency":      4,
        "comorbidity_count":        1,
        "medication_count":         1,
        "hospital_visits_last_year":0,
        "hba1c":                    5.1,
        "blood_pressure_systolic":  112,
        "blood_pressure_diastolic": 72,
        "cholesterol":              142.0,
        "bmi":                      20.5,
        "age_group":                "young",
        "deleted":                  False,
    },
    {
        # ── P007: High Risk — Triple comorbidity, declining adherence ──────────
        "patient_id":               "P007",
        "name":                     "Suresh Nair",
        "age":                      63,
        "gender":                   "Male",
        "conditions":               ["type 2 diabetes", "hypertension", "heart failure"],
        "medications":              ["metformin", "amlodipine", "furosemide"],
        "adherence_rate":           55.0,
        "exercise_level":           2,
        "follow_up_frequency":      3,
        "comorbidity_count":        3,
        "medication_count":         3,
        "hospital_visits_last_year":3,
        "hba1c":                    8.1,
        "blood_pressure_systolic":  155,
        "blood_pressure_diastolic": 95,
        "cholesterol":              210.0,
        "bmi":                      29.6,
        "age_group":                "senior",
        "deleted":                  False,
    },
]


# ── Risk predictions: 2 per patient (creates trend data for the chart) ────────
def _build_risk_predictions(now: datetime) -> list:
    """
    Build 2 risk prediction records per patient.
    The first prediction (older) is slightly different from the second (recent)
    to create a visible trend on the dashboard risk trend chart.
    """
    records = [
        # ── P001: HIGH — worsening trend ──────────────────────────────────────
        {
            "patient_id":       "P001",
            "risk_score":       71.8,
            "risk_level":       "HIGH",
            "adherence_level":  "POOR",
            "key_risk_factors": ["Poor adherence (62%)", "Elevated HbA1c (8.5%)",
                                  "High comorbidity count (3 conditions)"],
            "explanation":      "Patient P001 shows declining adherence trend combined with uncontrolled HbA1c, placing them at HIGH risk.",
            "recommendations":  ["Weekly pharmacist check-in", "HbA1c retest in 6 weeks", "Diet counselling referral"],
            "predicted_at":     now - timedelta(days=14),
        },
        {
            "patient_id":       "P001",
            "risk_score":       78.2,
            "risk_level":       "HIGH",
            "adherence_level":  "POOR",
            "key_risk_factors": ["Poor adherence (62%)", "Elevated HbA1c (8.5%)",
                                  "2 hospital visits in past year"],
            "explanation":      "Risk has increased from 71.8% to 78.2% over 2 weeks due to continued poor adherence and recent hospitalisation.",
            "recommendations":  ["Immediate follow-up appointment", "Medication blister pack review", "Home BP monitoring"],
            "predicted_at":     now - timedelta(days=1),
        },
        # ── P002: LOW — stable ────────────────────────────────────────────────
        {
            "patient_id":       "P002",
            "risk_score":       21.4,
            "risk_level":       "LOW",
            "adherence_level":  "EXCELLENT",
            "key_risk_factors": ["No major risk factors identified"],
            "explanation":      "Excellent adherence, single condition, active lifestyle with exercise level 7/10.",
            "recommendations":  ["Continue current care plan", "Annual review sufficient"],
            "predicted_at":     now - timedelta(days=30),
        },
        {
            "patient_id":       "P002",
            "risk_score":       18.5,
            "risk_level":       "LOW",
            "adherence_level":  "EXCELLENT",
            "key_risk_factors": ["No major risk factors identified"],
            "explanation":      "Risk stable at LOW level. Patient maintains excellent adherence and active lifestyle.",
            "recommendations":  ["Maintain current inhaler technique", "Seasonal allergy review"],
            "predicted_at":     now - timedelta(days=2),
        },
        # ── P003: CRITICAL — worsening ────────────────────────────────────────
        {
            "patient_id":       "P003",
            "risk_score":       84.6,
            "risk_level":       "CRITICAL",
            "adherence_level":  "CRITICAL",
            "key_risk_factors": ["Critical adherence (45%)", "4 hospital admissions in past year",
                                  "Cardiac condition — heart failure"],
            "explanation":      "Patient presents with CRITICAL risk due to severely compromised cardiac function combined with very poor medication adherence.",
            "recommendations":  ["Urgent cardiology referral", "Daily medication check", "Consider supervised dispensing"],
            "predicted_at":     now - timedelta(days=21),
        },
        {
            "patient_id":       "P003",
            "risk_score":       91.4,
            "risk_level":       "CRITICAL",
            "adherence_level":  "CRITICAL",
            "key_risk_factors": ["Critical adherence (45%)", "Advanced age (67)", "Heart failure — high decompensation risk"],
            "explanation":      "Risk escalated to 91.4% — deterioration from 84.6%. Immediate clinical intervention is required.",
            "recommendations":  ["48-hour clinical review", "Medication compliance programme enrolment", "Family/carer involvement in medication management"],
            "predicted_at":     now - timedelta(days=1),
        },
        # ── P004: MODERATE — improving ────────────────────────────────────────
        {
            "patient_id":       "P004",
            "risk_score":       51.3,
            "risk_level":       "MODERATE",
            "adherence_level":  "GOOD",
            "key_risk_factors": ["CKD progression risk", "Borderline HbA1c (7.2%)", "Diabetes management complexity"],
            "explanation":      "Moderate risk driven by CKD and suboptimal glycaemic control. Good adherence partially mitigates risk.",
            "recommendations":  ["Nephrology follow-up", "eGFR monitoring every 3 months", "Dietary protein restriction counselling"],
            "predicted_at":     now - timedelta(days=28),
        },
        {
            "patient_id":       "P004",
            "risk_score":       42.1,
            "risk_level":       "MODERATE",
            "adherence_level":  "GOOD",
            "key_risk_factors": ["CKD monitoring required", "Borderline HbA1c"],
            "explanation":      "Risk improved from 51.3% to 42.1% following increased follow-up frequency. Continue current plan.",
            "recommendations":  ["Continue 8 visits/year frequency", "HbA1c target <7.0%", "BP target <130/80"],
            "predicted_at":     now - timedelta(days=3),
        },
        # ── P005: CRITICAL — consistently critical ────────────────────────────
        {
            "patient_id":       "P005",
            "risk_score":       91.2,
            "risk_level":       "CRITICAL",
            "adherence_level":  "CRITICAL",
            "key_risk_factors": ["Critical adherence (35%)", "Polypharmacy (5 medications)",
                                  "Advanced age (71)", "6 hospital admissions"],
            "explanation":      "Highest risk profile in the cohort. Polypharmacy combined with critically low adherence and frequent hospitalisations.",
            "recommendations":  ["Medication reconciliation review", "Geriatric assessment", "Carer support programme"],
            "predicted_at":     now - timedelta(days=10),
        },
        {
            "patient_id":       "P005",
            "risk_score":       94.7,
            "risk_level":       "CRITICAL",
            "adherence_level":  "CRITICAL",
            "key_risk_factors": ["Critical adherence (35%)", "HbA1c 9.8% — very poor glycaemic control",
                                  "Polypharmacy — 5 concurrent medications"],
            "explanation":      "Risk has worsened to 94.7%. Requires immediate multi-disciplinary team intervention.",
            "recommendations":  ["Immediate MDT case conference", "Blister pack dispensing", "Daily nurse/carer visit for 4 weeks"],
            "predicted_at":     now - timedelta(days=1),
        },
        # ── P006: LOW — stable, very low ─────────────────────────────────────
        {
            "patient_id":       "P006",
            "risk_score":       12.3,
            "risk_level":       "LOW",
            "adherence_level":  "EXCELLENT",
            "key_risk_factors": ["No significant risk factors"],
            "explanation":      "Young patient with excellent adherence, single condition, and high physical activity level.",
            "recommendations":  ["Annual review", "Continue current inhaler as needed"],
            "predicted_at":     now - timedelta(days=60),
        },
        {
            "patient_id":       "P006",
            "risk_score":       11.8,
            "risk_level":       "LOW",
            "adherence_level":  "EXCELLENT",
            "key_risk_factors": ["No significant risk factors"],
            "explanation":      "Continued LOW risk — stable trend. Patient self-managing effectively.",
            "recommendations":  ["6-monthly check-in", "Maintain exercise routine"],
            "predicted_at":     now - timedelta(days=5),
        },
        # ── P007: HIGH — worsening ────────────────────────────────────────────
        {
            "patient_id":       "P007",
            "risk_score":       62.4,
            "risk_level":       "HIGH",
            "adherence_level":  "POOR",
            "key_risk_factors": ["Poor adherence (55%)", "Triple comorbidity", "Elevated HbA1c (8.1%)"],
            "explanation":      "Patient with three concurrent chronic conditions and declining adherence presents at HIGH risk.",
            "recommendations":  ["Bi-weekly pharmacist review", "Furosemide fluid monitoring", "Cardiac function check"],
            "predicted_at":     now - timedelta(days=20),
        },
        {
            "patient_id":       "P007",
            "risk_score":       71.9,
            "risk_level":       "HIGH",
            "adherence_level":  "POOR",
            "key_risk_factors": ["Poor adherence (55%)", "3 hospitalisations past year",
                                  "Heart failure — fluid retention risk"],
            "explanation":      "Escalating risk from 62.4% to 71.9%. Combination of heart failure and diabetes with poor adherence is clinically dangerous.",
            "recommendations":  ["Weekly weight monitoring", "Salt restriction education", "Cardiology outpatient referral"],
            "predicted_at":     now - timedelta(days=2),
        },
    ]
    return records


# ── Simulation records ────────────────────────────────────────────────────────
def _build_simulations(now: datetime) -> list:
    return [
        {
            "patient_id":    "P001",
            "baseline_risk": 78.2,
            "scenarios": [
                {
                    "label":             "Current Behavior",
                    "description":       "Maintaining current habits without intervention",
                    "adherence_rate":    62.0,
                    "exercise_level":    2,
                    "follow_up_frequency": 3,
                    "risk_score":        78.2,
                    "risk_level":        "HIGH",
                    "predicted_outcome": "🟠 High — Likely deterioration without active intervention",
                    "ai_explanation":    "Maintaining the current 62% adherence rate with sedentary lifestyle continues the worsening trend seen over the past 2 weeks.",
                },
                {
                    "label":             "Improved Adherence",
                    "description":       "Achieves 85% adherence with pharmacist support programme",
                    "adherence_rate":    85.0,
                    "exercise_level":    4,
                    "follow_up_frequency": 6,
                    "risk_score":        43.7,
                    "risk_level":        "MODERATE",
                    "predicted_outcome": "🟡 Moderate — Increased monitoring recommended",
                    "ai_explanation":    "Improving adherence to 85% with increased follow-up reduces risk by 34.5 points, moving from HIGH to MODERATE — a clinically significant improvement.",
                },
                {
                    "label":             "Poor Adherence",
                    "description":       "Adherence declines further due to side effects",
                    "adherence_rate":    40.0,
                    "exercise_level":    1,
                    "follow_up_frequency": 1,
                    "risk_score":        89.1,
                    "risk_level":        "CRITICAL",
                    "predicted_outcome": "🔴 Critical — High hospital readmission risk",
                    "ai_explanation":    "Further decline in adherence to 40% with reduced follow-up escalates risk to CRITICAL (89.1%), making hospital readmission highly likely within 30 days.",
                },
            ],
            "recommendation": "The simulation reveals a 45.4% risk differential across scenarios. Improving adherence to 85% with pharmacist support delivers the greatest risk reduction for this patient.",
            "simulated_at": now - timedelta(days=1),
        },
        {
            "patient_id":    "P003",
            "baseline_risk": 91.4,
            "scenarios": [
                {
                    "label":             "Current Behavior",
                    "adherence_rate":    45.0,
                    "exercise_level":    1,
                    "follow_up_frequency": 2,
                    "risk_score":        91.4,
                    "risk_level":        "CRITICAL",
                    "predicted_outcome": "🚨 Severe — Immediate clinical intervention required",
                    "ai_explanation":    "Current trajectory is unsustainable — immediate decompensation risk is very high.",
                },
                {
                    "label":             "Supervised Medication Programme",
                    "adherence_rate":    78.0,
                    "exercise_level":    2,
                    "follow_up_frequency": 10,
                    "risk_score":        58.3,
                    "risk_level":        "HIGH",
                    "predicted_outcome": "🟠 High — Requires continued intensive management",
                    "ai_explanation":    "Supervised dispensing bringing adherence to 78% with 10 follow-ups/year reduces risk from CRITICAL to HIGH — preventing likely hospitalisation.",
                },
                {
                    "label":             "Full Intervention Programme",
                    "adherence_rate":    88.0,
                    "exercise_level":    3,
                    "follow_up_frequency": 16,
                    "risk_score":        38.9,
                    "risk_level":        "MODERATE",
                    "predicted_outcome": "🟡 Moderate — Achievable with structured care",
                    "ai_explanation":    "Full MDT intervention achieving 88% adherence reduces risk by 52.5 points — demonstrating that intensive care coordination can move this patient from CRITICAL to MODERATE risk.",
                },
            ],
            "recommendation": "Full intervention programme delivers a 52.5% risk reduction for P003. Priority action: enrol in supervised medication programme immediately.",
            "simulated_at": now - timedelta(days=2),
        },
        {
            "patient_id":    "P005",
            "baseline_risk": 94.7,
            "scenarios": [
                {
                    "label":             "Current Behavior",
                    "adherence_rate":    35.0,
                    "exercise_level":    1,
                    "follow_up_frequency": 1,
                    "risk_score":        94.7,
                    "risk_level":        "CRITICAL",
                    "predicted_outcome": "🚨 Severe — Immediate clinical intervention required",
                    "ai_explanation":    "Highest risk patient in cohort. Current behaviour makes adverse event within 2 weeks highly probable.",
                },
                {
                    "label":             "Blister Pack + Carer Support",
                    "adherence_rate":    72.0,
                    "exercise_level":    2,
                    "follow_up_frequency": 8,
                    "risk_score":        65.8,
                    "risk_level":        "HIGH",
                    "predicted_outcome": "🔴 Critical → High — Significant improvement but sustained monitoring needed",
                    "ai_explanation":    "Blister pack dispensing and daily carer visits bringing adherence to 72% achieves a 28.9-point risk reduction despite complex polypharmacy.",
                },
                {
                    "label":             "Geriatric MDT Programme",
                    "adherence_rate":    82.0,
                    "exercise_level":    3,
                    "follow_up_frequency": 12,
                    "risk_score":        51.2,
                    "risk_level":        "MODERATE",
                    "predicted_outcome": "🟡 Moderate — Best achievable with full geriatric support",
                    "ai_explanation":    "Comprehensive geriatric assessment with MDT coordination, carer support, and simplified medication regimen achieves a 43.5-point reduction — the best realistic outcome for this patient.",
                },
            ],
            "recommendation": "Geriatric MDT programme is the optimal pathway for P005, achieving a 43.5% risk reduction. Polypharmacy review to simplify the 5-medication regimen is the first priority.",
            "simulated_at": now - timedelta(days=1),
        },
    ]


# ── Document processing records ────────────────────────────────────────────────
def _build_documents(now: datetime) -> list:
    return [
        {
            "patient_id":    "P001",
            "patient_name":  "Rajesh Kumar",
            "filename":      "rajesh_kumar_prescription_jan2024.pdf",
            "document_type": "prescription",
            "raw_text":      "Patient: Rajesh Kumar | DOB: 15/03/1966 | Diagnosis: Type 2 Diabetes Mellitus, Hypertension | Medications: Metformin 500mg BD, Lisinopril 10mg OD, Atorvastatin 20mg HS | HbA1c: 8.5% | BP: 148/92 mmHg | Follow up in 4 weeks.",
            "extracted_entities": {
                "diseases":     ["type 2 diabetes", "hypertension"],
                "medications":  ["metformin", "lisinopril", "atorvastatin"],
                "symptoms":     ["fatigue", "frequent urination"],
                "lab_tests":    ["hba1c", "blood pressure"],
                "lab_values":   {"hba1c": "8.5%", "blood pressure": "148/92 mmHg"},
                "risk_factors": ["poor glycaemic control", "elevated blood pressure"],
                "dosages":      {"metformin": "500mg twice daily", "lisinopril": "10mg once daily", "atorvastatin": "20mg at bedtime"},
                "instructions": ["Follow up in 4 weeks", "Fasting blood glucose monitoring daily"],
            },
            "summary":           "Rajesh Kumar presents with poorly controlled Type 2 Diabetes (HbA1c 8.5%) and hypertension (BP 148/92 mmHg). Current regimen of Metformin, Lisinopril, and Atorvastatin requires adherence reinforcement. Follow-up in 4 weeks with repeat HbA1c recommended.",
            "risk_score":        78.2,
            "risk_level":        "HIGH",
            "risk_explanation":  "High HbA1c and declining adherence with multiple comorbidities elevate risk to HIGH level.",
            "kg_nodes_created":  9,
            "doc_agent_status":  "success",
            "kg_agent_status":   "success",
            "risk_agent_status": "success",
            "intervention_status": "success",
            "processing_time_ms": 4235.8,
            "file_size_bytes":   52480,
            "uploaded_at":       now - timedelta(days=3),
        },
        {
            "patient_id":    "P003",
            "patient_name":  "Amit Verma",
            "filename":      "amit_verma_lab_report_feb2024.pdf",
            "document_type": "lab_report",
            "raw_text":      "Patient: Amit Verma | Investigation Report | Troponin: 0.02 ng/mL (normal) | BNP: 485 pg/mL (elevated — reference <100) | Creatinine: 1.4 mg/dL | eGFR: 52 mL/min | Echocardiogram: EF 35% (reduced) | Impression: Decompensated heart failure, AKI stage 1.",
            "extracted_entities": {
                "diseases":     ["heart failure", "acute kidney injury"],
                "medications":  ["furosemide", "aspirin"],
                "symptoms":     ["shortness of breath", "swollen ankles"],
                "lab_tests":    ["troponin", "bnp", "creatinine", "egfr"],
                "lab_values":   {"troponin": "0.02 ng/mL", "bnp": "485 pg/mL", "creatinine": "1.4 mg/dL", "egfr": "52 mL/min"},
                "risk_factors": ["reduced ejection fraction", "elevated BNP"],
                "dosages":      {},
                "instructions": ["Urgent cardiology review", "Fluid restriction 1.5L/day", "Daily weight monitoring"],
            },
            "summary":           "Lab report for Amit Verma reveals decompensated heart failure with BNP markedly elevated at 485 pg/mL and reduced ejection fraction of 35% on echocardiogram. Stage 1 AKI is also noted. Urgent cardiology review and fluid management are indicated.",
            "risk_score":        91.4,
            "risk_level":        "CRITICAL",
            "risk_explanation":  "Critical cardiac decompensation markers combined with critically low adherence place this patient at severe risk of adverse events.",
            "kg_nodes_created":  11,
            "doc_agent_status":  "success",
            "kg_agent_status":   "success",
            "risk_agent_status": "success",
            "intervention_status": "success",
            "processing_time_ms": 5124.3,
            "file_size_bytes":   38912,
            "uploaded_at":       now - timedelta(days=1),
        },
        {
            "patient_id":    "P004",
            "patient_name":  "Sunita Patel",
            "filename":      "sunita_patel_medical_summary.pdf",
            "document_type": "medical_summary",
            "raw_text":      "Medical Summary — Sunita Patel | CKD Stage 3a, Type 2 Diabetes | eGFR: 48 mL/min (stable) | HbA1c: 7.2% | Creatinine: 1.5 mg/dL | On: Metformin 500mg BD (dose may need review at eGFR <45), Insulin Glargine 20 units nocte | Diabetic nephropathy monitoring active | Next review: 3 months.",
            "extracted_entities": {
                "diseases":     ["chronic kidney disease stage 3a", "type 2 diabetes", "diabetic nephropathy"],
                "medications":  ["metformin", "insulin glargine"],
                "symptoms":     [],
                "lab_tests":    ["egfr", "hba1c", "creatinine"],
                "lab_values":   {"egfr": "48 mL/min", "hba1c": "7.2%", "creatinine": "1.5 mg/dL"},
                "risk_factors": ["ckd progression", "metformin dose review needed at egfr <45"],
                "dosages":      {"metformin": "500mg twice daily", "insulin glargine": "20 units at night"},
                "instructions": ["Review metformin dose if eGFR falls below 45", "Next review in 3 months"],
            },
            "summary":           "Sunita Patel has stable CKD Stage 3a with eGFR 48 mL/min alongside Type 2 Diabetes with HbA1c 7.2%. Current medications include Metformin and Insulin Glargine — note that Metformin dose will require review if eGFR falls below 45 mL/min. Diabetic nephropathy monitoring is active with 3-monthly reviews.",
            "risk_score":        42.1,
            "risk_level":        "MODERATE",
            "kg_nodes_created":  10,
            "doc_agent_status":  "success",
            "kg_agent_status":   "success",
            "risk_agent_status": "success",
            "intervention_status": "success",
            "processing_time_ms": 3891.6,
            "file_size_bytes":   29696,
            "uploaded_at":       now - timedelta(days=5),
        },
        {
            "patient_id":    "P005",
            "patient_name":  "Vikram Singh",
            "filename":      "vikram_singh_prescription_feb2024.pdf",
            "document_type": "prescription",
            "raw_text":      "Patient: Vikram Singh | Complex multi-morbidity | Diagnosis: HTN, CAD, Hyperlipidaemia, T2DM | Medications: Lisinopril 20mg OD, Aspirin 75mg OD, Atorvastatin 40mg HS, Metformin 1000mg BD, Insulin Glargine 28 units nocte | HbA1c: 9.8% | BP: 175/105 mmHg | Risk: Very high cardiovascular risk | Adherence concern documented.",
            "extracted_entities": {
                "diseases":     ["hypertension", "coronary artery disease", "hyperlipidaemia", "type 2 diabetes"],
                "medications":  ["lisinopril", "aspirin", "atorvastatin", "metformin", "insulin glargine"],
                "symptoms":     ["dizziness", "fatigue"],
                "lab_tests":    ["hba1c", "blood pressure"],
                "lab_values":   {"hba1c": "9.8%", "blood pressure": "175/105 mmHg"},
                "risk_factors": ["very high cardiovascular risk", "polypharmacy", "documented adherence concern"],
                "dosages":      {"lisinopril": "20mg once daily", "aspirin": "75mg once daily", "atorvastatin": "40mg at bedtime", "metformin": "1000mg twice daily", "insulin glargine": "28 units at night"},
                "instructions": ["Adherence support programme referral", "BP target <130/80", "Diabetologist review urgently"],
            },
            "summary":           "Vikram Singh presents with complex multi-morbidity requiring five concurrent medications. HbA1c is critically elevated at 9.8% and blood pressure severely uncontrolled at 175/105 mmHg. Adherence concerns are formally documented. Urgent diabetologist review and adherence support programme referral are required.",
            "risk_score":        94.7,
            "risk_level":        "CRITICAL",
            "kg_nodes_created":  14,
            "doc_agent_status":  "success",
            "kg_agent_status":   "success",
            "risk_agent_status": "success",
            "intervention_status": "success",
            "processing_time_ms": 6782.1,
            "file_size_bytes":   61440,
            "uploaded_at":       now - timedelta(days=2),
        },
        {
            "patient_id":    "P007",
            "patient_name":  "Suresh Nair",
            "filename":      "suresh_nair_prescription_jan2024.pdf",
            "document_type": "prescription",
            "raw_text":      "Patient: Suresh Nair | T2DM, HTN, Heart Failure | HbA1c: 8.1% | BP: 155/95 | Weight: 82kg (baseline 79kg — weight gain noted) | Medications: Metformin 1g BD, Amlodipine 5mg OD, Furosemide 40mg OD | Weight gain of 3kg over 4 weeks — possible fluid retention | Review furosemide dose.",
            "extracted_entities": {
                "diseases":     ["type 2 diabetes", "hypertension", "heart failure"],
                "medications":  ["metformin", "amlodipine", "furosemide"],
                "symptoms":     ["weight gain", "possible fluid retention"],
                "lab_tests":    ["hba1c", "blood pressure", "weight"],
                "lab_values":   {"hba1c": "8.1%", "blood pressure": "155/95 mmHg", "weight": "82kg"},
                "risk_factors": ["fluid retention signal", "poor glycaemic control", "rising weight trend"],
                "dosages":      {"metformin": "1g twice daily", "amlodipine": "5mg once daily", "furosemide": "40mg once daily"},
                "instructions": ["Daily weight monitoring", "Alert if weight gain >2kg in 3 days", "Review furosemide dose next visit"],
            },
            "summary":           "Suresh Nair presents with Type 2 Diabetes (HbA1c 8.1%), Hypertension (BP 155/95), and Heart Failure. A 3kg weight gain over 4 weeks is noted, raising concern for fluid retention requiring furosemide dose review. Daily weight monitoring and early warning thresholds have been advised.",
            "risk_score":        71.9,
            "risk_level":        "HIGH",
            "kg_nodes_created":  12,
            "doc_agent_status":  "success",
            "kg_agent_status":   "success",
            "risk_agent_status": "success",
            "intervention_status": "success",
            "processing_time_ms": 4567.9,
            "file_size_bytes":   44032,
            "uploaded_at":       now - timedelta(days=4),
        },
    ]


# ══════════════════════════════════════════════════════════════════════════════
# SEEDER
# ══════════════════════════════════════════════════════════════════════════════

async def seed(reset: bool = True):
    print("=" * 55)
    print("  MediGraph AI — MongoDB Data Seeder")
    print("=" * 55)

    await connect_db()
    db = get_db()

    now = datetime.utcnow()

    # ── Optionally clear existing collections ─────────────────────────────────
    if reset:
        print("\n[1/5] Clearing existing collections...")
        await db.patients.delete_many({})
        await db.risk_predictions.delete_many({})
        await db.simulations.delete_many({})
        await db.documents.delete_many({})
        print("  ✅ All collections cleared")
    else:
        print("\n[1/5] Skipping clear (--no-reset mode)")

    # ── Insert patients ───────────────────────────────────────────────────────
    print("\n[2/5] Seeding patients...")
    for patient in SAMPLE_PATIENTS:
        patient["created_at"] = now - timedelta(days=random.randint(30, 120))
        patient["updated_at"] = now
        try:
            await db.patients.insert_one(dict(patient))
            print(f"  ✅ {patient['patient_id']} — {patient['name']} "
                  f"({patient['age']}{patient['gender'][0]}) "
                  f"adherence={patient['adherence_rate']}%")
        except Exception as e:
            print(f"  ⚠️  {patient['patient_id']} — {e}")

    print(f"\n  Total: {len(SAMPLE_PATIENTS)} patients seeded")

    # ── Insert risk predictions ───────────────────────────────────────────────
    print("\n[3/5] Seeding risk predictions (2 per patient for trend charts)...")
    predictions = _build_risk_predictions(now)
    for pred in predictions:
        try:
            await db.risk_predictions.insert_one(dict(pred))
        except Exception as e:
            print(f"  ⚠️  {pred['patient_id']} — {e}")
    print(f"  ✅ {len(predictions)} risk predictions seeded")

    # ── Insert simulations ────────────────────────────────────────────────────
    print("\n[4/5] Seeding simulation runs...")
    simulations = _build_simulations(now)
    for sim in simulations:
        try:
            await db.simulations.insert_one(dict(sim))
            print(f"  ✅ Simulation for {sim['patient_id']} — baseline {sim['baseline_risk']}%")
        except Exception as e:
            print(f"  ⚠️  {sim['patient_id']} — {e}")

    # ── Insert document records ───────────────────────────────────────────────
    print("\n[5/5] Seeding document processing records...")
    documents = _build_documents(now)
    for doc in documents:
        try:
            await db.documents.insert_one(dict(doc))
            print(f"  ✅ {doc['filename']} → risk={doc['risk_score']}% ({doc['risk_level']})")
        except Exception as e:
            print(f"  ⚠️  {doc['filename']} — {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  🎉  MongoDB Seeding Complete!")
    print("=" * 55)
    print(f"  Patients:         {await db.patients.count_documents({})}")
    print(f"  Risk predictions: {await db.risk_predictions.count_documents({})}")
    print(f"  Simulations:      {await db.simulations.count_documents({})}")
    print(f"  Documents:        {await db.documents.count_documents({})}")
    print("\n  Next: Start the backend and open the dashboard")
    print("  uvicorn backend.main:app --reload --port 8000")
    print("=" * 55 + "\n")

    await disconnect_db()


if __name__ == "__main__":
    reset_flag = "--no-reset" not in sys.argv
    asyncio.run(seed(reset=reset_flag))