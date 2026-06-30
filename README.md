# 🏥 MediGraph AI

### Agentic Healthcare Knowledge Graph for Predictive Care Intelligence

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.138-009688)
![Neo4j](https://img.shields.io/badge/Neo4j-Aura-008CC1)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248)
![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-8E75B2)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Overview

MediGraph AI is an AI-powered clinical decision support system that predicts medication non-adherence risk, builds a healthcare knowledge graph, and generates personalised intervention plans — combining graph databases, machine learning, and generative AI into a unified clinical intelligence platform.

The system analyses healthcare documents, maps semantic relationships between patients, diseases, medications, symptoms, and lab results, predicts which patients are likely to become non-adherent before deterioration occurs, and recommends targeted interventions — moving healthcare from reactive to proactive care.

## ✨ Key Features

- 🧠 **Healthcare Knowledge Graph** — Neo4j graph with 13 relationship types modelling disease progression, drug interactions, contraindications, and comorbidity patterns
- 🤖 **4-Agent AI Pipeline** — LangGraph orchestration: Document → Knowledge Graph → Risk → Intervention agents, powered by Google Gemini 2.0 Flash
- 📊 **Predictive Risk Model** — XGBoost classifier trained on 20,000 patients from 4 real clinical datasets
- 📄 **Document Intelligence** — PDF/OCR extraction with Gemini AI for structured clinical entity extraction
- 🔮 **What-If Simulation** — Cross-sectional counterfactual risk modelling for care planning
- 💊 **Drug Safety Center** — Real-time OpenFDA integration for adverse events, recalls, and interactions
- 📈 **Executive Dashboard** — Real-time KPIs, risk distribution, and population health analytics
- 🗄️ **Dual Database Architecture** — MongoDB Atlas for document storage + Neo4j Aura for graph relationships

## 🏗️ Architecture

The system follows a layered architecture:

**Frontend Layer** — Vanilla JavaScript dashboard with D3.js force-directed graphs and Plotly.js charts, served as static files.

**API Layer** — FastAPI backend exposing 6 routers: Patients, Documents, Risk, Simulation, Graph, and Drugs — all with async request handling.

**Intelligence Layer** — A 4-agent LangGraph pipeline processes each document:
1. **Document Agent** extracts text via PyMuPDF/OCR and structures it using **Gemini 2.0 Flash**
2. **Knowledge Graph Agent** ingests entities into **Neo4j Aura**
3. **Risk Agent** runs XGBoost inference and explains results via **Gemini**
4. **Intervention Agent** generates a care plan via **Gemini**

**Data Layer** — **MongoDB Atlas** stores patient records, documents, risk predictions, and simulation history. **Neo4j Aura** stores the clinical knowledge graph (diseases, medications, symptoms, lab tests, risk factors, and their relationships).

## 🛠️ Tech Stack

| Layer | Technology |
|       |            |
| Backend Framework | FastAPI, LangGraph, LangChain |
| Machine Learning | XGBoost, scikit-learn, pandas, numpy |
| Graph Database | **Neo4j Aura** (free tier) |
| Document Database | **MongoDB Atlas** (free tier) |
| Generative AI | **Google Gemini 2.0 Flash** |
| PDF Processing | PyMuPDF, Tesseract OCR |
| Frontend | HTML5, Vanilla JavaScript, D3.js, Plotly.js |
| External APIs | OpenFDA (drug safety data) |

## 📊 Dataset

Training data combines **4 real clinical datasets** into 20,000 patient records:

| Source | Patients | Contribution |
|        |          |              |
| [MIMIC-III Demo](https://physionet.org/content/mimiciii-demo/1.4/) | 100 | Real ICU clinical structure |
| [Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/) | 15,000 sampled | Real diabetic EHR records |
| [Cleveland Heart Disease](https://archive.ics.uci.edu/dataset/45/) | 303 | Real cardiac clinical features |
| [CKD Dataset (India)](https://archive.ics.uci.edu/dataset/336/) | 400 | Real kidney function markers |
| Calibrated synthetic | 4,197 | Fills population gaps |

See [docs/DATASET.md](docs/DATASET.md) for full methodology.

## 🎯 Model Performance

| Metric | Score |
|        |       |
| Accuracy | 85.3% |
| Precision | 76.1% |
| **Recall** | **92.2%** |
| F1 Score | 83.3% |
| **ROC-AUC** | **95.5%** |
| CV F1 (5-fold) | 0.840 ± 0.004 |

17 engineered features including adherence rate, comorbidity count, HbA1c, hospital visits, and demographic factors.

## 🧠 Knowledge Graph

**54 nodes** across 6 types (Patient, Disease, Medication, Symptom, LabTest, RiskFactor) and **137 relationships** across **13 relationship types**, all stored in **Neo4j Aura**:

| Relationship | Models |
|              |        |
| `HAS_DISEASE` `TAKES_MEDICATION` `SHOWS_SYMPTOM` | Core patient clinical state |
| `COMPLICATES` | Disease progression pathways |
| `CO_OCCURS_WITH` | Comorbidity patterns |
| `CONTRAINDICATED` | Drug safety warnings |
| `INTERACTS_WITH` | Drug-drug interactions |
| `CAUSES_SYMPTOM` | Disease-symptom causal links |
| `MONITORS` | Lab test-disease monitoring |
| `WORSENED_BY` | Risk factor amplification |

## 🤖 Generative AI Integration

**Google Gemini 2.0 Flash** powers four clinical AI functions:

| Function | Purpose |
|          |         |
| Entity Extraction | Converts unstructured PDF text into structured diseases, medications, lab values |
| Clinical Summarisation | Generates 3-4 sentence summaries of prescriptions and lab reports |
| Risk Explanation | Produces physician-readable explanations for XGBoost risk scores |
| Intervention Planning | Creates priority-tiered care plans (immediate / short-term / long-term) |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/lahrisaumya/medigraph-ai.git
cd medigraph-ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment — add your MongoDB, Neo4j, and Gemini credentials
cp .env.example .env

# Train the ML model
python -m backend.ml.train_model

# Seed MongoDB with demo patients
python scripts/seed_mongodb.py

# Run scripts/setup_neo4j.cypher in Neo4j Browser to build the graph

# Start backend
uvicorn backend.main:app --reload --port 8000

# Start frontend (new terminal)
python serve.py
```

Open `http://localhost:3000` to view the dashboard.

**Full setup guide:** [docs/SETUP.md](docs/SETUP.md)

## 📁 Project Structure

```
medigraph-ai/
├── backend/
│   ├── agents/        4 LangGraph agents + orchestrator
│   ├── api/            6 FastAPI routers
│   ├── core/            Config + Pydantic schemas
│   ├── db/              MongoDB + Neo4j connectors
│   ├── ml/              XGBoost training + inference
│   └── utils/            Gemini client + PDF extraction
├── frontend/
│   ├── index.html       Executive Overview dashboard
│   ├── pages/             6 additional dashboard pages
│   ├── assets/             Dark theme CSS
│   └── utils/               API client + helpers + charts
├── scripts/
│   ├── preprocess_all_datasets.py   Dataset merger
│   ├── seed_mongodb.py               MongoDB demo data seeder
│   └── setup_neo4j.cypher            Neo4j graph schema + data
├── docs/                  Setup, API, dataset, viva docs
├── tests/                  Pytest test suite
└── data/                   Training datasets
```

## 📖 Documentation

| Document | Description |
|          |             |
| [SETUP.md](docs/SETUP.md) | Complete step-by-step setup guide including MongoDB and Neo4j configuration |
| [API.md](docs/API.md) | Full REST API reference |
| [DATASET.md](docs/DATASET.md) | Dataset sources and preprocessing methodology |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Cloud deployment instructions |
| [VIVA_QA.md](docs/VIVA_QA.md) | 24 anticipated viva questions with detailed answers |


## Acknowledgements

- **MIMIC-III Clinical Database** — PhysioNet / MIT Lab for Computational Physiology
- **UCI Machine Learning Repository** — Diabetes 130-US, Heart Disease, CKD datasets
- **OpenFDA** — FDA Adverse Event Reporting System (FAERS) data
- **MongoDB Atlas** — Document database infrastructure
- **Neo4j Aura** — Graph database infrastructure
- **Google Gemini** — Generative AI clinical intelligence