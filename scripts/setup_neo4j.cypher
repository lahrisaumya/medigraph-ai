// ============================================================
// FILE:    scripts/setup_neo4j.cypher
// PURPOSE: Neo4j Aura schema + complete healthcare knowledge graph.
//          Includes complex multi-hop relationships, treatment
//          pathways, drug interactions, comorbidity links, and
//          temporal disease progression chains.
//
// HOW TO RUN:
//   1. Go to https://console.neo4j.io
//   2. Open Query tab for your medigraph instance
//   3. Copy entire file → paste → Ctrl+Enter
//
// COMPLEX RELATIONSHIPS ADDED:
//   COMPLICATES      — Disease → Disease (disease progression)
//   CONTRAINDICATED  — Medication → Disease (drug contraindications)
//   INTERACTS_WITH   — Medication → Medication (drug interactions)
//   CAUSES_SYMPTOM   — Disease → Symptom (disease-symptom mapping)
//   MONITORS         — LabTest → Disease (test-disease monitoring)
//   WORSENED_BY      — Disease → RiskFactor (risk amplification)
//   CO_OCCURS_WITH   — Disease → Disease (comorbidity pattern)
//
// EXPECTED RESULT:
//   Nodes:         ~65
//   Relationships: ~130+ (was 93)
// ============================================================


// ─────────────────────────────────────────────────────────────
// STEP 1: CONSTRAINTS
// ─────────────────────────────────────────────────────────────

CREATE CONSTRAINT patient_id_unique IF NOT EXISTS
  FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE;
CREATE CONSTRAINT disease_name_unique IF NOT EXISTS
  FOR (d:Disease) REQUIRE d.name IS UNIQUE;
CREATE CONSTRAINT medication_name_unique IF NOT EXISTS
  FOR (m:Medication) REQUIRE m.name IS UNIQUE;
CREATE CONSTRAINT symptom_name_unique IF NOT EXISTS
  FOR (s:Symptom) REQUIRE s.name IS UNIQUE;
CREATE CONSTRAINT lab_test_name_unique IF NOT EXISTS
  FOR (l:LabTest) REQUIRE l.name IS UNIQUE;
CREATE CONSTRAINT risk_factor_name_unique IF NOT EXISTS
  FOR (r:RiskFactor) REQUIRE r.name IS UNIQUE;


// ─────────────────────────────────────────────────────────────
// STEP 2: DISEASE NODES
// ─────────────────────────────────────────────────────────────

MERGE (:Disease {name:"type 2 diabetes",        category:"chronic",    icd9:"25000", severity:"moderate", prevalence:"high"});
MERGE (:Disease {name:"hypertension",            category:"chronic",    icd9:"4019",  severity:"moderate", prevalence:"very_high"});
MERGE (:Disease {name:"chronic kidney disease",  category:"chronic",    icd9:"5856",  severity:"high",     prevalence:"moderate"});
MERGE (:Disease {name:"coronary artery disease", category:"chronic",    icd9:"41401", severity:"high",     prevalence:"high"});
MERGE (:Disease {name:"asthma",                  category:"chronic",    icd9:"4939",  severity:"moderate", prevalence:"high"});
MERGE (:Disease {name:"heart failure",           category:"chronic",    icd9:"4280",  severity:"critical", prevalence:"moderate"});
MERGE (:Disease {name:"hyperlipidemia",          category:"chronic",    icd9:"2724",  severity:"low",      prevalence:"very_high"});
MERGE (:Disease {name:"obesity",                 category:"chronic",    icd9:"2780",  severity:"moderate", prevalence:"high"});
MERGE (:Disease {name:"atrial fibrillation",     category:"chronic",    icd9:"42731", severity:"high",     prevalence:"moderate"});
MERGE (:Disease {name:"diabetic nephropathy",    category:"complication",icd9:"25040",severity:"high",     prevalence:"moderate"});


// ─────────────────────────────────────────────────────────────
// STEP 3: MEDICATION NODES
// ─────────────────────────────────────────────────────────────

MERGE (:Medication {name:"metformin",          dosage:"500mg-1g twice daily", class:"biguanide",                route:"oral",      cost_tier:"low"});
MERGE (:Medication {name:"lisinopril",         dosage:"10-20mg once daily",   class:"ace_inhibitor",            route:"oral",      cost_tier:"low"});
MERGE (:Medication {name:"amlodipine",         dosage:"5mg once daily",       class:"calcium_channel_blocker",  route:"oral",      cost_tier:"low"});
MERGE (:Medication {name:"atorvastatin",       dosage:"20-40mg at bedtime",   class:"statin",                   route:"oral",      cost_tier:"low"});
MERGE (:Medication {name:"aspirin",            dosage:"75mg once daily",      class:"antiplatelet",             route:"oral",      cost_tier:"very_low"});
MERGE (:Medication {name:"insulin glargine",   dosage:"20-28 units at night", class:"insulin",                  route:"injection", cost_tier:"high"});
MERGE (:Medication {name:"furosemide",         dosage:"40mg once daily",      class:"loop_diuretic",            route:"oral",      cost_tier:"low"});
MERGE (:Medication {name:"salbutamol inhaler", dosage:"2 puffs as needed",    class:"bronchodilator",           route:"inhaled",   cost_tier:"moderate"});
MERGE (:Medication {name:"warfarin",           dosage:"dose per INR target",  class:"anticoagulant",            route:"oral",      cost_tier:"low"});
MERGE (:Medication {name:"bisoprolol",         dosage:"2.5-5mg once daily",   class:"beta_blocker",             route:"oral",      cost_tier:"low"});


// ─────────────────────────────────────────────────────────────
// STEP 4: SYMPTOM NODES
// ─────────────────────────────────────────────────────────────

MERGE (:Symptom {name:"fatigue",              severity:"variable", onset:"gradual",  system:"general"});
MERGE (:Symptom {name:"chest pain",           severity:"high",     onset:"acute",    system:"cardiac"});
MERGE (:Symptom {name:"shortness of breath",  severity:"high",     onset:"variable", system:"respiratory"});
MERGE (:Symptom {name:"frequent urination",   severity:"moderate", onset:"gradual",  system:"renal"});
MERGE (:Symptom {name:"blurred vision",       severity:"moderate", onset:"gradual",  system:"ophthalmic"});
MERGE (:Symptom {name:"headache",             severity:"low",      onset:"variable", system:"neurological"});
MERGE (:Symptom {name:"swollen ankles",       severity:"moderate", onset:"gradual",  system:"cardiovascular"});
MERGE (:Symptom {name:"dizziness",            severity:"moderate", onset:"variable", system:"neurological"});
MERGE (:Symptom {name:"weight gain",          severity:"moderate", onset:"gradual",  system:"metabolic"});
MERGE (:Symptom {name:"nausea",               severity:"low",      onset:"acute",    system:"gastrointestinal"});


// ─────────────────────────────────────────────────────────────
// STEP 5: LAB TEST NODES
// ─────────────────────────────────────────────────────────────

MERGE (:LabTest {name:"hba1c",                normal_range:"4.0-5.6%",       unit:"%",       frequency:"3 monthly",  critical_high:"9.0"});
MERGE (:LabTest {name:"fasting blood glucose", normal_range:"70-100 mg/dL",  unit:"mg/dL",   frequency:"monthly",    critical_high:"300"});
MERGE (:LabTest {name:"creatinine",            normal_range:"0.7-1.2 mg/dL", unit:"mg/dL",   frequency:"3 monthly",  critical_high:"5.0"});
MERGE (:LabTest {name:"ldl cholesterol",       normal_range:"<100 mg/dL",    unit:"mg/dL",   frequency:"6 monthly",  critical_high:"190"});
MERGE (:LabTest {name:"blood pressure",        normal_range:"<120/80 mmHg",  unit:"mmHg",    frequency:"monthly",    critical_high:"180"});
MERGE (:LabTest {name:"egfr",                  normal_range:">60 mL/min",    unit:"mL/min",  frequency:"3 monthly",  critical_low:"15"});
MERGE (:LabTest {name:"bnp",                   normal_range:"<100 pg/mL",    unit:"pg/mL",   frequency:"3 monthly",  critical_high:"500"});
MERGE (:LabTest {name:"troponin",              normal_range:"<0.04 ng/mL",   unit:"ng/mL",   frequency:"as_needed",  critical_high:"1.0"});


// ─────────────────────────────────────────────────────────────
// STEP 6: RISK FACTOR NODES
// ─────────────────────────────────────────────────────────────

MERGE (:RiskFactor {name:"poor medication adherence", category:"behavioural",  modifiable:true,  weight:0.45});
MERGE (:RiskFactor {name:"sedentary lifestyle",        category:"behavioural",  modifiable:true,  weight:0.25});
MERGE (:RiskFactor {name:"poor glycaemic control",     category:"clinical",     modifiable:true,  weight:0.30});
MERGE (:RiskFactor {name:"polypharmacy",               category:"clinical",     modifiable:false, weight:0.20});
MERGE (:RiskFactor {name:"advanced age",               category:"demographic",  modifiable:false, weight:0.15});
MERGE (:RiskFactor {name:"no follow-up",               category:"behavioural",  modifiable:true,  weight:0.35});
MERGE (:RiskFactor {name:"obesity",                    category:"clinical",     modifiable:true,  weight:0.20});
MERGE (:RiskFactor {name:"fluid retention",            category:"clinical",     modifiable:true,  weight:0.25});
MERGE (:RiskFactor {name:"reduced ejection fraction",  category:"clinical",     modifiable:false, weight:0.40});


// ─────────────────────────────────────────────────────────────
// STEP 7: PATIENT NODES
// ─────────────────────────────────────────────────────────────

MERGE (p:Patient {patient_id:"P001"})
SET p.name="Rajesh Kumar", p.age=58, p.gender="Male",
    p.adherence_rate=62.0, p.risk_level="HIGH", p.risk_score=78.2,
    p.hba1c=8.5, p.bmi=28.4, p.blood_pressure="148/92",
    p.exercise_level=2, p.follow_up_frequency=3,
    p.created_at="2024-01-15", p.insurance="private";

MERGE (p:Patient {patient_id:"P002"})
SET p.name="Priya Sharma", p.age=45, p.gender="Female",
    p.adherence_rate=88.0, p.risk_level="LOW", p.risk_score=18.5,
    p.hba1c=5.4, p.bmi=22.1, p.blood_pressure="118/76",
    p.exercise_level=7, p.follow_up_frequency=6,
    p.created_at="2024-01-20", p.insurance="government";

MERGE (p:Patient {patient_id:"P003"})
SET p.name="Amit Verma", p.age=67, p.gender="Male",
    p.adherence_rate=45.0, p.risk_level="CRITICAL", p.risk_score=91.4,
    p.hba1c=6.1, p.bmi=31.2, p.blood_pressure="162/98",
    p.exercise_level=1, p.follow_up_frequency=2,
    p.created_at="2024-02-01", p.insurance="none";

MERGE (p:Patient {patient_id:"P004"})
SET p.name="Sunita Patel", p.age=52, p.gender="Female",
    p.adherence_rate=75.0, p.risk_level="MODERATE", p.risk_score=42.1,
    p.hba1c=7.2, p.bmi=26.8, p.blood_pressure="130/84",
    p.exercise_level=4, p.follow_up_frequency=8,
    p.created_at="2024-02-10", p.insurance="private";

MERGE (p:Patient {patient_id:"P005"})
SET p.name="Vikram Singh", p.age=71, p.gender="Male",
    p.adherence_rate=35.0, p.risk_level="CRITICAL", p.risk_score=94.7,
    p.hba1c=9.8, p.bmi=33.7, p.blood_pressure="175/105",
    p.exercise_level=1, p.follow_up_frequency=1,
    p.created_at="2024-02-15", p.insurance="government";

MERGE (p:Patient {patient_id:"P006"})
SET p.name="Ananya Reddy", p.age=34, p.gender="Female",
    p.adherence_rate=92.0, p.risk_level="LOW", p.risk_score=11.8,
    p.hba1c=5.1, p.bmi=20.5, p.blood_pressure="112/72",
    p.exercise_level=8, p.follow_up_frequency=4,
    p.created_at="2024-02-20", p.insurance="private";

MERGE (p:Patient {patient_id:"P007"})
SET p.name="Suresh Nair", p.age=63, p.gender="Male",
    p.adherence_rate=55.0, p.risk_level="HIGH", p.risk_score=71.9,
    p.hba1c=8.1, p.bmi=29.6, p.blood_pressure="155/95",
    p.exercise_level=2, p.follow_up_frequency=3,
    p.created_at="2024-02-22", p.insurance="private";


// ─────────────────────────────────────────────────────────────
// STEP 8: PATIENT → DISEASE (HAS_DISEASE)
// ─────────────────────────────────────────────────────────────

MATCH (p:Patient {patient_id:"P001"})
MATCH (d1:Disease {name:"type 2 diabetes"})
MATCH (d2:Disease {name:"hypertension"})
MATCH (d3:Disease {name:"hyperlipidemia"})
MERGE (p)-[:HAS_DISEASE {since:"2018-03-10", severity:"moderate", controlled:false}]->(d1)
MERGE (p)-[:HAS_DISEASE {since:"2019-07-22", severity:"moderate", controlled:false}]->(d2)
MERGE (p)-[:HAS_DISEASE {since:"2020-01-15", severity:"low",      controlled:true}]->(d3);

MATCH (p:Patient {patient_id:"P002"})
MATCH (d:Disease {name:"asthma"})
MERGE (p)-[:HAS_DISEASE {since:"2015-09-12", severity:"mild", controlled:true}]->(d);

MATCH (p:Patient {patient_id:"P003"})
MATCH (d1:Disease {name:"coronary artery disease"})
MATCH (d2:Disease {name:"heart failure"})
MATCH (d3:Disease {name:"hypertension"})
MERGE (p)-[:HAS_DISEASE {since:"2017-11-03", severity:"high",     controlled:false}]->(d1)
MERGE (p)-[:HAS_DISEASE {since:"2019-04-18", severity:"critical", controlled:false}]->(d2)
MERGE (p)-[:HAS_DISEASE {since:"2016-06-22", severity:"high",     controlled:false}]->(d3);

MATCH (p:Patient {patient_id:"P004"})
MATCH (d1:Disease {name:"type 2 diabetes"})
MATCH (d2:Disease {name:"chronic kidney disease"})
MATCH (d3:Disease {name:"diabetic nephropathy"})
MERGE (p)-[:HAS_DISEASE {since:"2016-08-14", severity:"moderate", controlled:true}]->(d1)
MERGE (p)-[:HAS_DISEASE {since:"2021-03-05", severity:"moderate", controlled:true}]->(d2)
MERGE (p)-[:HAS_DISEASE {since:"2021-06-20", severity:"moderate", controlled:true}]->(d3);

MATCH (p:Patient {patient_id:"P005"})
MATCH (d1:Disease {name:"hypertension"})
MATCH (d2:Disease {name:"coronary artery disease"})
MATCH (d3:Disease {name:"hyperlipidemia"})
MATCH (d4:Disease {name:"type 2 diabetes"})
MERGE (p)-[:HAS_DISEASE {since:"2010-04-22", severity:"critical", controlled:false}]->(d1)
MERGE (p)-[:HAS_DISEASE {since:"2014-09-18", severity:"high",     controlled:false}]->(d2)
MERGE (p)-[:HAS_DISEASE {since:"2012-11-30", severity:"moderate", controlled:false}]->(d3)
MERGE (p)-[:HAS_DISEASE {since:"2015-02-14", severity:"high",     controlled:false}]->(d4);

MATCH (p:Patient {patient_id:"P006"})
MATCH (d:Disease {name:"asthma"})
MERGE (p)-[:HAS_DISEASE {since:"2019-05-20", severity:"mild", controlled:true}]->(d);

MATCH (p:Patient {patient_id:"P007"})
MATCH (d1:Disease {name:"type 2 diabetes"})
MATCH (d2:Disease {name:"hypertension"})
MATCH (d3:Disease {name:"heart failure"})
MERGE (p)-[:HAS_DISEASE {since:"2017-06-10", severity:"moderate", controlled:false}]->(d1)
MERGE (p)-[:HAS_DISEASE {since:"2018-02-28", severity:"high",     controlled:false}]->(d2)
MERGE (p)-[:HAS_DISEASE {since:"2022-08-15", severity:"high",     controlled:false}]->(d3);


// ─────────────────────────────────────────────────────────────
// STEP 9: PATIENT → MEDICATION (TAKES_MEDICATION)
// ─────────────────────────────────────────────────────────────

MATCH (p:Patient {patient_id:"P001"})
MATCH (m1:Medication {name:"metformin"})
MATCH (m2:Medication {name:"lisinopril"})
MATCH (m3:Medication {name:"atorvastatin"})
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:62.0, started:"2018-04-01", dose_changes:1, missed_doses_per_week:3}]->(m1)
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:68.0, started:"2019-08-05", dose_changes:0, missed_doses_per_week:2}]->(m2)
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:55.0, started:"2020-02-10", dose_changes:1, missed_doses_per_week:4}]->(m3);

MATCH (p:Patient {patient_id:"P002"})
MATCH (m:Medication {name:"salbutamol inhaler"})
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:88.0, started:"2015-10-01", dose_changes:0, missed_doses_per_week:0}]->(m);

MATCH (p:Patient {patient_id:"P003"})
MATCH (m1:Medication {name:"aspirin"})
MATCH (m2:Medication {name:"furosemide"})
MATCH (m3:Medication {name:"amlodipine"})
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:45.0, started:"2017-12-01", dose_changes:2, missed_doses_per_week:4}]->(m1)
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:40.0, started:"2019-05-10", dose_changes:3, missed_doses_per_week:5}]->(m2)
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:50.0, started:"2016-07-01", dose_changes:2, missed_doses_per_week:4}]->(m3);

MATCH (p:Patient {patient_id:"P004"})
MATCH (m1:Medication {name:"metformin"})
MATCH (m2:Medication {name:"insulin glargine"})
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:75.0, started:"2016-09-01", dose_changes:1, missed_doses_per_week:1}]->(m1)
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:80.0, started:"2021-07-01", dose_changes:1, missed_doses_per_week:1}]->(m2);

MATCH (p:Patient {patient_id:"P005"})
MATCH (m1:Medication {name:"lisinopril"})
MATCH (m2:Medication {name:"aspirin"})
MATCH (m3:Medication {name:"atorvastatin"})
MATCH (m4:Medication {name:"metformin"})
MATCH (m5:Medication {name:"insulin glargine"})
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:38.0, started:"2010-05-01", dose_changes:4, missed_doses_per_week:5}]->(m1)
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:40.0, started:"2014-10-01", dose_changes:2, missed_doses_per_week:4}]->(m2)
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:32.0, started:"2013-01-15", dose_changes:3, missed_doses_per_week:6}]->(m3)
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:35.0, started:"2015-03-01", dose_changes:3, missed_doses_per_week:5}]->(m4)
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:30.0, started:"2020-06-01", dose_changes:4, missed_doses_per_week:6}]->(m5);

MATCH (p:Patient {patient_id:"P006"})
MATCH (m:Medication {name:"salbutamol inhaler"})
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:92.0, started:"2019-06-01", dose_changes:0, missed_doses_per_week:0}]->(m);

MATCH (p:Patient {patient_id:"P007"})
MATCH (m1:Medication {name:"metformin"})
MATCH (m2:Medication {name:"amlodipine"})
MATCH (m3:Medication {name:"furosemide"})
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:55.0, started:"2017-07-01", dose_changes:2, missed_doses_per_week:3}]->(m1)
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:60.0, started:"2018-03-10", dose_changes:1, missed_doses_per_week:3}]->(m2)
MERGE (p)-[:TAKES_MEDICATION {adherence_rate:50.0, started:"2022-09-01", dose_changes:2, missed_doses_per_week:4}]->(m3);


// ─────────────────────────────────────────────────────────────
// STEP 10: PATIENT → SYMPTOM (SHOWS_SYMPTOM)
// ─────────────────────────────────────────────────────────────

MATCH (p:Patient {patient_id:"P001"})
MATCH (s1:Symptom {name:"fatigue"})
MATCH (s2:Symptom {name:"frequent urination"})
MATCH (s3:Symptom {name:"blurred vision"})
MERGE (p)-[:SHOWS_SYMPTOM {reported_at:"2024-01-15", frequency:"daily",      severity:"moderate"}]->(s1)
MERGE (p)-[:SHOWS_SYMPTOM {reported_at:"2024-01-15", frequency:"daily",      severity:"moderate"}]->(s2)
MERGE (p)-[:SHOWS_SYMPTOM {reported_at:"2024-01-15", frequency:"occasional", severity:"mild"}]->(s3);

MATCH (p:Patient {patient_id:"P002"})
MATCH (s:Symptom {name:"shortness of breath"})
MERGE (p)-[:SHOWS_SYMPTOM {reported_at:"2024-01-20", frequency:"occasional", severity:"mild"}]->(s);

MATCH (p:Patient {patient_id:"P003"})
MATCH (s1:Symptom {name:"shortness of breath"})
MATCH (s2:Symptom {name:"swollen ankles"})
MATCH (s3:Symptom {name:"chest pain"})
MATCH (s4:Symptom {name:"fatigue"})
MERGE (p)-[:SHOWS_SYMPTOM {reported_at:"2024-02-01", frequency:"daily",    severity:"severe"}]->(s1)
MERGE (p)-[:SHOWS_SYMPTOM {reported_at:"2024-02-01", frequency:"daily",    severity:"moderate"}]->(s2)
MERGE (p)-[:SHOWS_SYMPTOM {reported_at:"2024-02-01", frequency:"frequent", severity:"moderate"}]->(s3)
MERGE (p)-[:SHOWS_SYMPTOM {reported_at:"2024-02-01", frequency:"daily",    severity:"moderate"}]->(s4);

MATCH (p:Patient {patient_id:"P005"})
MATCH (s1:Symptom {name:"dizziness"})
MATCH (s2:Symptom {name:"fatigue"})
MATCH (s3:Symptom {name:"blurred vision"})
MERGE (p)-[:SHOWS_SYMPTOM {reported_at:"2024-02-15", frequency:"daily",    severity:"moderate"}]->(s1)
MERGE (p)-[:SHOWS_SYMPTOM {reported_at:"2024-02-15", frequency:"daily",    severity:"severe"}]->(s2)
MERGE (p)-[:SHOWS_SYMPTOM {reported_at:"2024-02-15", frequency:"frequent", severity:"mild"}]->(s3);

MATCH (p:Patient {patient_id:"P007"})
MATCH (s1:Symptom {name:"swollen ankles"})
MATCH (s2:Symptom {name:"weight gain"})
MATCH (s3:Symptom {name:"fatigue"})
MERGE (p)-[:SHOWS_SYMPTOM {reported_at:"2024-02-22", frequency:"daily",    severity:"moderate"}]->(s1)
MERGE (p)-[:SHOWS_SYMPTOM {reported_at:"2024-02-22", frequency:"weekly",   severity:"moderate"}]->(s2)
MERGE (p)-[:SHOWS_SYMPTOM {reported_at:"2024-02-22", frequency:"daily",    severity:"mild"}]->(s3);


// ─────────────────────────────────────────────────────────────
// STEP 11: PATIENT → LAB TEST (UNDERWENT_TEST)
// ─────────────────────────────────────────────────────────────

MATCH (p:Patient {patient_id:"P001"})
MATCH (l1:LabTest {name:"hba1c"})
MATCH (l2:LabTest {name:"blood pressure"})
MATCH (l3:LabTest {name:"ldl cholesterol"})
MERGE (p)-[:UNDERWENT_TEST {value:"8.5%",        tested_at:"2024-01-10", flag:"abnormal", trend:"worsening"}]->(l1)
MERGE (p)-[:UNDERWENT_TEST {value:"148/92 mmHg", tested_at:"2024-01-10", flag:"abnormal", trend:"stable"}]->(l2)
MERGE (p)-[:UNDERWENT_TEST {value:"142 mg/dL",   tested_at:"2024-01-10", flag:"abnormal", trend:"improving"}]->(l3);

MATCH (p:Patient {patient_id:"P002"})
MATCH (l:LabTest {name:"egfr"})
MERGE (p)-[:UNDERWENT_TEST {value:">90 mL/min", tested_at:"2023-12-01", flag:"normal", trend:"stable"}]->(l);

MATCH (p:Patient {patient_id:"P003"})
MATCH (l1:LabTest {name:"bnp"})
MATCH (l2:LabTest {name:"troponin"})
MATCH (l3:LabTest {name:"egfr"})
MATCH (l4:LabTest {name:"blood pressure"})
MERGE (p)-[:UNDERWENT_TEST {value:"485 pg/mL",   tested_at:"2024-02-01", flag:"abnormal", trend:"worsening"}]->(l1)
MERGE (p)-[:UNDERWENT_TEST {value:"0.02 ng/mL",  tested_at:"2024-02-01", flag:"normal",   trend:"stable"}]->(l2)
MERGE (p)-[:UNDERWENT_TEST {value:"52 mL/min",   tested_at:"2024-02-01", flag:"abnormal", trend:"worsening"}]->(l3)
MERGE (p)-[:UNDERWENT_TEST {value:"162/98 mmHg", tested_at:"2024-02-01", flag:"abnormal", trend:"worsening"}]->(l4);

MATCH (p:Patient {patient_id:"P004"})
MATCH (l1:LabTest {name:"hba1c"})
MATCH (l2:LabTest {name:"egfr"})
MATCH (l3:LabTest {name:"creatinine"})
MERGE (p)-[:UNDERWENT_TEST {value:"7.2%",      tested_at:"2024-02-10", flag:"abnormal", trend:"improving"}]->(l1)
MERGE (p)-[:UNDERWENT_TEST {value:"48 mL/min", tested_at:"2024-02-10", flag:"abnormal", trend:"stable"}]->(l2)
MERGE (p)-[:UNDERWENT_TEST {value:"1.5 mg/dL", tested_at:"2024-02-10", flag:"abnormal", trend:"stable"}]->(l3);

MATCH (p:Patient {patient_id:"P005"})
MATCH (l1:LabTest {name:"hba1c"})
MATCH (l2:LabTest {name:"blood pressure"})
MERGE (p)-[:UNDERWENT_TEST {value:"9.8%",         tested_at:"2024-02-15", flag:"abnormal", trend:"worsening"}]->(l1)
MERGE (p)-[:UNDERWENT_TEST {value:"175/105 mmHg", tested_at:"2024-02-15", flag:"abnormal", trend:"worsening"}]->(l2);

MATCH (p:Patient {patient_id:"P007"})
MATCH (l1:LabTest {name:"hba1c"})
MATCH (l2:LabTest {name:"blood pressure"})
MERGE (p)-[:UNDERWENT_TEST {value:"8.1%",        tested_at:"2024-02-22", flag:"abnormal", trend:"worsening"}]->(l1)
MERGE (p)-[:UNDERWENT_TEST {value:"155/95 mmHg", tested_at:"2024-02-22", flag:"abnormal", trend:"stable"}]->(l2);


// ─────────────────────────────────────────────────────────────
// STEP 12: PATIENT → RISK FACTOR (HAS_RISK)
// ─────────────────────────────────────────────────────────────

MATCH (p:Patient {patient_id:"P001"})
MATCH (r1:RiskFactor {name:"poor medication adherence"})
MATCH (r2:RiskFactor {name:"poor glycaemic control"})
MERGE (p)-[:HAS_RISK {score:78.0, assessed_at:"2024-01-15", priority:"high"}]->(r1)
MERGE (p)-[:HAS_RISK {score:72.0, assessed_at:"2024-01-15", priority:"high"}]->(r2);

MATCH (p:Patient {patient_id:"P003"})
MATCH (r1:RiskFactor {name:"poor medication adherence"})
MATCH (r2:RiskFactor {name:"advanced age"})
MATCH (r3:RiskFactor {name:"reduced ejection fraction"})
MATCH (r4:RiskFactor {name:"fluid retention"})
MERGE (p)-[:HAS_RISK {score:91.4, assessed_at:"2024-02-01", priority:"critical"}]->(r1)
MERGE (p)-[:HAS_RISK {score:65.0, assessed_at:"2024-02-01", priority:"high"}]->(r2)
MERGE (p)-[:HAS_RISK {score:88.0, assessed_at:"2024-02-01", priority:"critical"}]->(r3)
MERGE (p)-[:HAS_RISK {score:75.0, assessed_at:"2024-02-01", priority:"high"}]->(r4);

MATCH (p:Patient {patient_id:"P004"})
MATCH (r:RiskFactor {name:"poor glycaemic control"})
MERGE (p)-[:HAS_RISK {score:42.1, assessed_at:"2024-02-10", priority:"moderate"}]->(r);

MATCH (p:Patient {patient_id:"P005"})
MATCH (r1:RiskFactor {name:"poor medication adherence"})
MATCH (r2:RiskFactor {name:"polypharmacy"})
MATCH (r3:RiskFactor {name:"advanced age"})
MATCH (r4:RiskFactor {name:"poor glycaemic control"})
MERGE (p)-[:HAS_RISK {score:94.7, assessed_at:"2024-02-15", priority:"critical"}]->(r1)
MERGE (p)-[:HAS_RISK {score:82.0, assessed_at:"2024-02-15", priority:"critical"}]->(r2)
MERGE (p)-[:HAS_RISK {score:70.0, assessed_at:"2024-02-15", priority:"high"}]->(r3)
MERGE (p)-[:HAS_RISK {score:88.0, assessed_at:"2024-02-15", priority:"critical"}]->(r4);

MATCH (p:Patient {patient_id:"P007"})
MATCH (r1:RiskFactor {name:"poor medication adherence"})
MATCH (r2:RiskFactor {name:"fluid retention"})
MATCH (r3:RiskFactor {name:"poor glycaemic control"})
MERGE (p)-[:HAS_RISK {score:71.9, assessed_at:"2024-02-22", priority:"high"}]->(r1)
MERGE (p)-[:HAS_RISK {score:65.0, assessed_at:"2024-02-22", priority:"high"}]->(r2)
MERGE (p)-[:HAS_RISK {score:58.0, assessed_at:"2024-02-22", priority:"moderate"}]->(r3);


// ─────────────────────────────────────────────────────────────
// STEP 13: DISEASE → MEDICATION (TREATED_WITH)
// Standard clinical treatment guidelines
// ─────────────────────────────────────────────────────────────

MATCH (d:Disease {name:"type 2 diabetes"})
MATCH (m1:Medication {name:"metformin"})
MATCH (m2:Medication {name:"insulin glargine"})
MERGE (d)-[:TREATED_WITH {evidence_level:"A", guideline:"ADA 2024",     first_line:true,  mechanism:"reduces hepatic glucose production"}]->(m1)
MERGE (d)-[:TREATED_WITH {evidence_level:"A", guideline:"ADA 2024",     first_line:false, mechanism:"replaces endogenous insulin"}]->(m2);

MATCH (d:Disease {name:"hypertension"})
MATCH (m1:Medication {name:"lisinopril"})
MATCH (m2:Medication {name:"amlodipine"})
MERGE (d)-[:TREATED_WITH {evidence_level:"A", guideline:"JNC8",         first_line:true,  mechanism:"ACE inhibition reduces vasoconstriction"}]->(m1)
MERGE (d)-[:TREATED_WITH {evidence_level:"A", guideline:"JNC8",         first_line:true,  mechanism:"calcium channel blockade reduces vascular resistance"}]->(m2);

MATCH (d:Disease {name:"hyperlipidemia"})
MATCH (m:Medication {name:"atorvastatin"})
MERGE (d)-[:TREATED_WITH {evidence_level:"A", guideline:"ACC/AHA 2019", first_line:true,  mechanism:"HMG-CoA reductase inhibition reduces LDL synthesis"}]->(m);

MATCH (d:Disease {name:"coronary artery disease"})
MATCH (m1:Medication {name:"aspirin"})
MATCH (m2:Medication {name:"atorvastatin"})
MATCH (m3:Medication {name:"bisoprolol"})
MERGE (d)-[:TREATED_WITH {evidence_level:"A", guideline:"ESC 2023",    first_line:true,  mechanism:"antiplatelet prevents thrombosis"}]->(m1)
MERGE (d)-[:TREATED_WITH {evidence_level:"A", guideline:"ESC 2023",    first_line:true,  mechanism:"statin stabilises atherosclerotic plaque"}]->(m2)
MERGE (d)-[:TREATED_WITH {evidence_level:"A", guideline:"ESC 2023",    first_line:true,  mechanism:"beta blockade reduces myocardial oxygen demand"}]->(m3);

MATCH (d:Disease {name:"heart failure"})
MATCH (m1:Medication {name:"furosemide"})
MATCH (m2:Medication {name:"bisoprolol"})
MATCH (m3:Medication {name:"lisinopril"})
MERGE (d)-[:TREATED_WITH {evidence_level:"A", guideline:"ESC HF 2023", first_line:true,  mechanism:"loop diuresis reduces preload and fluid overload"}]->(m1)
MERGE (d)-[:TREATED_WITH {evidence_level:"A", guideline:"ESC HF 2023", first_line:true,  mechanism:"beta blockade improves cardiac remodelling"}]->(m2)
MERGE (d)-[:TREATED_WITH {evidence_level:"A", guideline:"ESC HF 2023", first_line:true,  mechanism:"ACE inhibition reduces afterload and fibrosis"}]->(m3);

MATCH (d:Disease {name:"asthma"})
MATCH (m:Medication {name:"salbutamol inhaler"})
MERGE (d)-[:TREATED_WITH {evidence_level:"A", guideline:"GINA 2023",   first_line:true,  mechanism:"beta-2 agonist bronchodilation relieves bronchospasm"}]->(m);

MATCH (d:Disease {name:"atrial fibrillation"})
MATCH (m:Medication {name:"warfarin"})
MERGE (d)-[:TREATED_WITH {evidence_level:"A", guideline:"ESC 2023",    first_line:true,  mechanism:"vitamin K antagonist prevents thromboembolic stroke"}]->(m);


// ─────────────────────────────────────────────────────────────
// STEP 14: COMPLEX — DISEASE → DISEASE (COMPLICATES)
// Disease progression and complication pathways
// ─────────────────────────────────────────────────────────────

MATCH (d1:Disease {name:"type 2 diabetes"})
MATCH (d2:Disease {name:"chronic kidney disease"})
MATCH (d3:Disease {name:"diabetic nephropathy"})
MATCH (d4:Disease {name:"coronary artery disease"})
MATCH (d5:Disease {name:"hypertension"})
MERGE (d1)-[:COMPLICATES {mechanism:"hyperglycaemia damages glomerular filtration",   risk_multiplier:3.2, years_to_complication:"5-10"}]->(d2)
MERGE (d1)-[:COMPLICATES {mechanism:"advanced glycation end-products damage kidneys", risk_multiplier:4.1, years_to_complication:"7-15"}]->(d3)
MERGE (d1)-[:COMPLICATES {mechanism:"insulin resistance accelerates atherosclerosis", risk_multiplier:2.8, years_to_complication:"10-20"}]->(d4)
MERGE (d5)-[:COMPLICATES {mechanism:"chronic pressure overload damages cardiac muscle",risk_multiplier:2.5, years_to_complication:"5-15"}]->(d4);

MATCH (d1:Disease {name:"coronary artery disease"})
MATCH (d2:Disease {name:"heart failure"})
MATCH (d3:Disease {name:"atrial fibrillation"})
MERGE (d1)-[:COMPLICATES {mechanism:"ischaemic cardiomyopathy reduces ejection fraction", risk_multiplier:3.5, years_to_complication:"5-10"}]->(d2)
MERGE (d1)-[:COMPLICATES {mechanism:"ischaemia disrupts electrical conduction pathways",  risk_multiplier:2.1, years_to_complication:"3-8"}]->(d3);

MATCH (d1:Disease {name:"chronic kidney disease"})
MATCH (d2:Disease {name:"hypertension"})
MATCH (d3:Disease {name:"heart failure"})
MERGE (d1)-[:COMPLICATES {mechanism:"RAAS activation increases fluid retention and BP",   risk_multiplier:2.8, years_to_complication:"2-5"}]->(d2)
MERGE (d1)-[:COMPLICATES {mechanism:"fluid overload and anaemia increase cardiac workload",risk_multiplier:2.3, years_to_complication:"3-7"}]->(d3);


// ─────────────────────────────────────────────────────────────
// STEP 15: COMPLEX — DISEASE → DISEASE (CO_OCCURS_WITH)
// Comorbidity patterns from epidemiological evidence
// ─────────────────────────────────────────────────────────────

MATCH (d1:Disease {name:"type 2 diabetes"})
MATCH (d2:Disease {name:"hypertension"})
MERGE (d1)-[:CO_OCCURS_WITH {co_occurrence_rate:0.72, evidence:"WHO 2023 — 72% of T2DM patients have hypertension", shared_pathway:"insulin resistance"}]->(d2);

MATCH (d1:Disease {name:"coronary artery disease"})
MATCH (d2:Disease {name:"hyperlipidemia"})
MERGE (d1)-[:CO_OCCURS_WITH {co_occurrence_rate:0.85, evidence:"ACC/AHA — 85% of CAD patients have dyslipidaemia", shared_pathway:"atherosclerosis"}]->(d2);

MATCH (d1:Disease {name:"obesity"})
MATCH (d2:Disease {name:"type 2 diabetes"})
MERGE (d1)-[:CO_OCCURS_WITH {co_occurrence_rate:0.68, evidence:"IDF 2022 — obesity is primary driver of T2DM", shared_pathway:"insulin resistance and adipokines"}]->(d2);

MATCH (d1:Disease {name:"heart failure"})
MATCH (d2:Disease {name:"chronic kidney disease"})
MERGE (d1)-[:CO_OCCURS_WITH {co_occurrence_rate:0.55, evidence:"Cardiorenal syndrome — 55% of HF patients develop CKD", shared_pathway:"reduced renal perfusion"}]->(d2);


// ─────────────────────────────────────────────────────────────
// STEP 16: COMPLEX — MEDICATION → DISEASE (CONTRAINDICATED)
// Drug contraindications for patient safety
// ─────────────────────────────────────────────────────────────

MATCH (m:Medication {name:"metformin"})
MATCH (d:Disease {name:"chronic kidney disease"})
MERGE (m)-[:CONTRAINDICATED {reason:"lactic acidosis risk when eGFR < 30 mL/min", severity:"severe", egfr_threshold:"<30", guideline:"NICE 2022"}]->(d);

MATCH (m:Medication {name:"lisinopril"})
MATCH (d:Disease {name:"chronic kidney disease"})
MERGE (m)-[:CONTRAINDICATED {reason:"hyperkalaemia and acute kidney injury risk with severe CKD", severity:"moderate", egfr_threshold:"<15", guideline:"BNF 2023"}]->(d);

MATCH (m:Medication {name:"warfarin"})
MATCH (d:Disease {name:"chronic kidney disease"})
MERGE (m)-[:CONTRAINDICATED {reason:"unpredictable INR and bleeding risk in severe CKD", severity:"moderate", egfr_threshold:"<15", guideline:"ESC 2023"}]->(d);

MATCH (m:Medication {name:"bisoprolol"})
MATCH (d:Disease {name:"asthma"})
MERGE (m)-[:CONTRAINDICATED {reason:"beta-blockade causes bronchospasm in asthmatic patients", severity:"severe", egfr_threshold:"N/A", guideline:"GINA 2023"}]->(d);


// ─────────────────────────────────────────────────────────────
// STEP 17: COMPLEX — MEDICATION → MEDICATION (INTERACTS_WITH)
// Clinically significant drug-drug interactions
// ─────────────────────────────────────────────────────────────

MATCH (m1:Medication {name:"warfarin"})
MATCH (m2:Medication {name:"aspirin"})
MERGE (m1)-[:INTERACTS_WITH {interaction_type:"pharmacodynamic", severity:"high", effect:"additive anticoagulation increases major bleeding risk 3-fold", monitoring:"INR weekly, bleeding signs"}]->(m2);

MATCH (m1:Medication {name:"lisinopril"})
MATCH (m2:Medication {name:"furosemide"})
MERGE (m1)-[:INTERACTS_WITH {interaction_type:"pharmacodynamic", severity:"moderate", effect:"first-dose hypotension risk — furosemide reduces preload, ACE inhibitor reduces afterload simultaneously", monitoring:"BP on initiation"}]->(m2);

MATCH (m1:Medication {name:"atorvastatin"})
MATCH (m2:Medication {name:"amlodipine"})
MERGE (m1)-[:INTERACTS_WITH {interaction_type:"pharmacokinetic", severity:"low", effect:"amlodipine inhibits CYP3A4, increases atorvastatin exposure by 18% — monitor for myopathy", monitoring:"CK levels if muscle pain"}]->(m2);

MATCH (m1:Medication {name:"metformin"})
MATCH (m2:Medication {name:"furosemide"})
MERGE (m1)-[:INTERACTS_WITH {interaction_type:"pharmacokinetic", severity:"low", effect:"furosemide may increase metformin plasma levels by reducing renal clearance", monitoring:"renal function quarterly"}]->(m2);


// ─────────────────────────────────────────────────────────────
// STEP 18: COMPLEX — DISEASE → SYMPTOM (CAUSES_SYMPTOM)
// Evidence-based disease-symptom causal links
// ─────────────────────────────────────────────────────────────

MATCH (d:Disease {name:"type 2 diabetes"})
MATCH (s1:Symptom {name:"frequent urination"})
MATCH (s2:Symptom {name:"fatigue"})
MATCH (s3:Symptom {name:"blurred vision"})
MERGE (d)-[:CAUSES_SYMPTOM {mechanism:"osmotic diuresis from hyperglycaemia",         prevalence:0.65}]->(s1)
MERGE (d)-[:CAUSES_SYMPTOM {mechanism:"impaired glucose utilisation in cells",         prevalence:0.70}]->(s2)
MERGE (d)-[:CAUSES_SYMPTOM {mechanism:"osmotic lens changes from hyperglycaemia",      prevalence:0.40}]->(s3);

MATCH (d:Disease {name:"heart failure"})
MATCH (s1:Symptom {name:"shortness of breath"})
MATCH (s2:Symptom {name:"swollen ankles"})
MATCH (s3:Symptom {name:"fatigue"})
MATCH (s4:Symptom {name:"weight gain"})
MERGE (d)-[:CAUSES_SYMPTOM {mechanism:"pulmonary oedema from elevated left atrial pressure", prevalence:0.90}]->(s1)
MERGE (d)-[:CAUSES_SYMPTOM {mechanism:"peripheral oedema from elevated venous pressure",     prevalence:0.85}]->(s2)
MERGE (d)-[:CAUSES_SYMPTOM {mechanism:"reduced cardiac output limits tissue perfusion",       prevalence:0.80}]->(s3)
MERGE (d)-[:CAUSES_SYMPTOM {mechanism:"fluid retention increases body weight",                prevalence:0.75}]->(s4);

MATCH (d:Disease {name:"hypertension"})
MATCH (s1:Symptom {name:"headache"})
MATCH (s2:Symptom {name:"dizziness"})
MERGE (d)-[:CAUSES_SYMPTOM {mechanism:"increased cerebral perfusion pressure",  prevalence:0.30}]->(s1)
MERGE (d)-[:CAUSES_SYMPTOM {mechanism:"impaired cerebral autoregulation",       prevalence:0.25}]->(s2);

MATCH (d:Disease {name:"coronary artery disease"})
MATCH (s1:Symptom {name:"chest pain"})
MATCH (s2:Symptom {name:"shortness of breath"})
MERGE (d)-[:CAUSES_SYMPTOM {mechanism:"myocardial ischaemia triggers angina",   prevalence:0.70}]->(s1)
MERGE (d)-[:CAUSES_SYMPTOM {mechanism:"reduced cardiac output impairs exercise", prevalence:0.55}]->(s2);


// ─────────────────────────────────────────────────────────────
// STEP 19: COMPLEX — LAB TEST → DISEASE (MONITORS)
// Test-disease monitoring relationships
// ─────────────────────────────────────────────────────────────

MATCH (l:LabTest {name:"hba1c"})
MATCH (d:Disease {name:"type 2 diabetes"})
MERGE (l)-[:MONITORS {target:"<7.0% for most patients", frequency:"every 3 months", action_threshold:">9.0% requires immediate review", guideline:"ADA 2024"}]->(d);

MATCH (l:LabTest {name:"egfr"})
MATCH (d:Disease {name:"chronic kidney disease"})
MERGE (l)-[:MONITORS {target:">60 mL/min/1.73m²", frequency:"every 3 months", action_threshold:"<30 mL/min requires nephrology referral", guideline:"KDIGO 2022"}]->(d);

MATCH (l:LabTest {name:"bnp"})
MATCH (d:Disease {name:"heart failure"})
MERGE (l)-[:MONITORS {target:"<100 pg/mL", frequency:"every 3 months", action_threshold:">500 pg/mL indicates decompensation", guideline:"ESC HF 2023"}]->(d);

MATCH (l:LabTest {name:"blood pressure"})
MATCH (d:Disease {name:"hypertension"})
MERGE (l)-[:MONITORS {target:"<130/80 mmHg", frequency:"monthly until controlled", action_threshold:">180/120 mmHg is hypertensive crisis", guideline:"ESC/ESH 2023"}]->(d);

MATCH (l:LabTest {name:"ldl cholesterol"})
MATCH (d:Disease {name:"coronary artery disease"})
MERGE (l)-[:MONITORS {target:"<70 mg/dL for very high risk", frequency:"every 6 months", action_threshold:">100 mg/dL requires therapy escalation", guideline:"ACC/AHA 2019"}]->(d);

MATCH (l:LabTest {name:"troponin"})
MATCH (d:Disease {name:"coronary artery disease"})
MERGE (l)-[:MONITORS {target:"<0.04 ng/mL", frequency:"only when symptomatic", action_threshold:"any elevation requires urgent assessment", guideline:"ESC NSTEMI 2023"}]->(d);


// ─────────────────────────────────────────────────────────────
// STEP 20: COMPLEX — DISEASE → RISK FACTOR (WORSENED_BY)
// Risk amplification pathways
// ─────────────────────────────────────────────────────────────

MATCH (d:Disease {name:"type 2 diabetes"})
MATCH (r1:RiskFactor {name:"poor medication adherence"})
MATCH (r2:RiskFactor {name:"sedentary lifestyle"})
MATCH (r3:RiskFactor {name:"obesity"})
MERGE (d)-[:WORSENED_BY {mechanism:"missed doses cause glycaemic excursions",    risk_increase:0.45}]->(r1)
MERGE (d)-[:WORSENED_BY {mechanism:"inactivity impairs insulin sensitivity",     risk_increase:0.30}]->(r2)
MERGE (d)-[:WORSENED_BY {mechanism:"adipose tissue increases insulin resistance",risk_increase:0.35}]->(r3);

MATCH (d:Disease {name:"heart failure"})
MATCH (r1:RiskFactor {name:"poor medication adherence"})
MATCH (r2:RiskFactor {name:"fluid retention"})
MATCH (r3:RiskFactor {name:"no follow-up"})
MERGE (d)-[:WORSENED_BY {mechanism:"missed diuretics cause acute decompensation",  risk_increase:0.60}]->(r1)
MERGE (d)-[:WORSENED_BY {mechanism:"excess sodium/fluid intake overwhelms diuresis",risk_increase:0.50}]->(r2)
MERGE (d)-[:WORSENED_BY {mechanism:"undetected deterioration leads to crisis",      risk_increase:0.55}]->(r3);

MATCH (d:Disease {name:"hypertension"})
MATCH (r1:RiskFactor {name:"poor medication adherence"})
MATCH (r2:RiskFactor {name:"obesity"})
MERGE (d)-[:WORSENED_BY {mechanism:"missed antihypertensives cause BP surge",   risk_increase:0.50}]->(r1)
MERGE (d)-[:WORSENED_BY {mechanism:"adipose tissue activates RAAS pathway",     risk_increase:0.30}]->(r2);


// ─────────────────────────────────────────────────────────────
// STEP 21: VERIFICATION QUERY
// ─────────────────────────────────────────────────────────────

MATCH (n)
RETURN labels(n)[0] AS NodeType, count(n) AS Count
ORDER BY Count DESC;
