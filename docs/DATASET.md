# MediGraph AI — Dataset Guide

---

## Overview

The project uses four real clinical datasets combined into one training file.

| Source | Patients | Real/Synthetic | Download Required |
|--------|---------|---------------|------------------|
| **MIMIC-III Demo** | 100 | ✅ Real ICU data | ✅ (already downloaded) |
| **Diabetes 130-US** | 15,000 sampled | ✅ Real EHR data | ✅ (already downloaded) |
| **Cleveland Heart Disease** | 303 | ✅ Real clinical data | ✅ (already downloaded) |
| **CKD India Hospital** | 400 | ✅ Real clinical data | ✅ (already downloaded) |
| **Synthetic augmentation** | 4,197 | 🔧 Calibrated synthetic | ❌ Auto-generated |
| **OpenFDA drug data** | live API | ✅ Real FDA data | ❌ Live API, no download |

**Final training dataset: 20,000 rows × 18 columns, 0 missing values**

---

## File Locations

```
data/
├── patient_features.csv       ← FINAL training file (ready to use)
├── sample_patients.json       ← 7 demo patients for MongoDB seeding
├── diabetic_data.csv          ← Diabetes 130-US raw file
├── heart_disease.csv          ← Cleveland Heart Disease raw file
├── ckd.csv                    ← CKD India Hospital raw file
└── mimic/
    ├── PATIENTS.csv
    ├── ADMISSIONS.csv
    ├── DIAGNOSES_ICD.csv
    ├── PRESCRIPTIONS.csv
    └── LABEVENTS.csv
```

---

## Dataset 1: MIMIC-III Demo

**Source:** PhysioNet — Beth Israel Deaconess Medical Center, Boston  
**Patients:** 100 ICU patients (all deceased — MIMIC Demo only includes deceased patients)  
**Files:** 5 CSV files (already in `data/mimic/`)

### What was extracted from MIMIC
| Feature | Extraction Method |
|---------|------------------|
| `age` | Computed from DOB − admission time; 8 patients had DOB shifted +300 years (MIMIC privacy rule for age >89) → corrected to age 91 |
| `comorbidity_count` | Count of unique 3-digit ICD-9 prefixes per patient |
| `medication_count` | Count of unique oral (PO route) drugs per patient |
| `hospital_visits_last_year` | Count of ADMISSIONS per patient |
| `diabetes_flag` | ICD-9 code starts with "250" |
| `hypertension_flag` | ICD-9 code starts with "401"-"404" |
| `cardiac_flag` | ICD-9 code starts with "410"-"414", "428", "427" |
| `hba1c_normalized` | Lab itemid 50852: real for 17 patients; ADAG formula (glucose→HbA1c) for remaining 83 |

### What was synthesised (not in MIMIC)
| Feature | Synthesis Method |
|---------|-----------------|
| `adherence_rate` | Complexity model: base 75% reduced by comorbidity × 3.0, medication × 1.2, hospitalisations × 3.5 |
| `exercise_level` | Cardiac/age-based: cardiac or >75yr → 1-3/10; others → 2-7/10 |
| `follow_up_frequency` | Poisson(4) + comorbidity × 0.5 + cardiac × 2 |
| `bmi_normalized` | Normal distribution (mean 27.5, std 5.5) adjusted for diabetes/age |

---

## Dataset 2: Diabetes 130-US Hospitals (1999–2008)

**Source:** UCI Machine Learning Repository  
**Original size:** 101,766 rows × 50 columns (sampled to 15,000)  
**URL:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

### Key real features used
| Dataset Column | Project Feature | Notes |
|---------------|----------------|-------|
| `age` | `age` | Bucket midpoint: [60-70) → 65 |
| `num_medications` | `medication_count` | Clipped to 1-15 |
| `number_inpatient + number_emergency` | `hospital_visits_last_year` | Clipped to 0-10 |
| `number_outpatient` | `follow_up_frequency` | Actual outpatient visit count |
| `number_diagnoses` | `comorbidity_count` | Divided by 2, clipped 0-8 |
| `A1Cresult` | `hba1c_normalized` | Buckets: Norm→5.5%, >7→7.5%, >8→9.2% |
| `diag_1` | `diabetes_flag`, `cardiac_flag`, `hypertension_flag` | ICD-9 prefix matching |
| **`readmitted`** | **`high_risk`** | **REAL TARGET: "<30" = 1, else = 0** |

### Missing values handled
| Column | Missing | Fix |
|--------|---------|-----|
| `weight` | 96.9% | Not used — modelled from age/diagnosis |
| `payer_code` | 39.6% | Not used |
| `race` | 2.2% | Not used |

### Why the target is clinically valid
Readmission within 30 days is the gold-standard proxy for medication non-adherence in published literature (Jencks et al., 2009, NEJM). 11.2% readmission rate reflects the real-world prevalence of high-adherence-risk patients in this diabetic population.

---

## Dataset 3: Cleveland Heart Disease (UCI)

**Source:** UCI ML Repository — Cleveland Clinic Foundation, Ohio  
**Patients:** 303 patients × 14 features  
**URL:** https://archive.ics.uci.edu/dataset/45/heart+disease  
**Note:** File has no header row — column names assigned from UCI documentation

### Column mapping
| Column | Name | Project Feature |
|--------|------|----------------|
| 1 | age | `age` |
| 2 | sex | not used |
| 3 | cp (chest pain type) | not used |
| 4 | trestbps (resting BP) | `hypertension_flag` (BP > 130) |
| 5 | chol (cholesterol) | `bmi_normalized` proxy |
| 6 | fbs (fasting blood sugar >120) | `diabetes_flag`, `hba1c_normalized` proxy |
| 8 | thalach (max heart rate) | `exercise_level` proxy |
| 9 | **exang (exercise-induced angina)** | **`exercise_level`** (1=angina→level 1-3; 0→level 4-8) |
| 14 | **target** | **`high_risk`** (**REAL TARGET**: target > 0 = heart disease) |

### Missing values
- `ca` (coronary vessels): 4 rows with "?" → filled with median
- `thal`: 2 rows with "?" → filled with median

---

## Dataset 4: Chronic Kidney Disease — UCI (India Hospital)

**Source:** UCI ML Repository — Hospital in Tamil Nadu, India  
**Patients:** 400 rows (399 usable after cleaning)  
**URL:** https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

### Data quality issues found and fixed
| Issue | Fix |
|-------|-----|
| `extra_26` column: 399/400 values missing | Dropped |
| 1 row has `class='no'` instead of `'notckd'` | Corrected to `'notckd'` |
| `?` used for missing values throughout | Replaced with NaN |
| `rbc`: 152/400 missing (38%) | Filled with mode |
| `wbcc`: 106/400 missing (27%) | Filled with median |
| `rbcc`: 131/400 missing (33%) | Filled with median |

### Key real features used
| Column | Meaning | Project Feature |
|--------|---------|----------------|
| `age` | Patient age | `age` |
| `htn` | Hypertension (yes/no) | `hypertension_flag` |
| `dm` | Diabetes mellitus (yes/no) | `diabetes_flag` |
| `cad` | Coronary artery disease (yes/no) | `cardiac_flag` |
| `bgr` | Blood glucose random (mg/dL) | `hba1c_normalized` via ADAG formula |
| `appet` | Appetite (good/poor) | `adherence_rate` penalty (poor = sicker) |
| **`class`** | ckd / notckd | **`high_risk`** (**REAL TARGET**: ckd = 1) |

### Why Indian hospital data matters
This dataset adds Indian population representation to the training data, making the model more applicable to Indian healthcare contexts — directly relevant for an MBA project in India.

---

## Preprocessing Script

The unified preprocessor merges all 4 sources:

```bash
# Run from project root
python scripts/preprocess_all_datasets.py
```

Processing time: ~15 seconds  
Output: `data/patient_features.csv`

The script performs:
1. DOB-shift correction for MIMIC patients (age >89)
2. ADAG formula for HbA1c estimation from glucose
3. ICD-9 prefix matching for disease flags
4. Stratified sampling from Diabetes 130-US (15,000 from 101,766)
5. Missing value imputation using dataset-specific strategies
6. Calibrated synthetic augmentation (4,197 rows in 3 clinical cohorts)
7. Full validation: range checks, binary checks, missing value check

---

## OpenFDA — Drug Safety Data (Live API)

No download required. The Drug Safety Center page queries OpenFDA in real-time:

| Endpoint | Data |
|---------|------|
| `/drug/event.json` | FAERS adverse event reports |
| `/drug/label.json` | FDA structured product labels |
| `/drug/enforcement.json` | Drug recall history |

**Rate limits (no API key):** 240 requests/minute  
**To increase limits:** Register at https://open.fda.gov/apis/authentication/ (free)  
**Set in .env:** `OPENFDA_API_KEY=your_key` (optional)

---

## Dataset Statistics Summary

| Metric | Value |
|--------|-------|
| Total training rows | 20,000 |
| Real clinical patient rows | 15,803 (79.0%) |
| Calibrated synthetic rows | 4,197 (21.0%) |
| High risk (=1) | 4,640 (23.2%) |
| Low risk (=0) | 15,360 (76.8%) |
| Features | 17 + 1 target = 18 columns |
| Missing values | 0 |
| Age range | 18 – 95 years |
| Mean adherence rate | 52.3% |
| Top correlated feature with target | `follow_up_frequency` (r = 0.42) |