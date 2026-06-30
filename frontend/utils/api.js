/**
 * frontend/utils/api.js
 * Centralised API client for MediGraph AI backend.
 * All fetch calls pass through here — handles base URL, headers, and errors.
 */

const BASE_URL = window.__API_BASE__ || "http://localhost:8000";

const DEFAULT_HEADERS = { "Content-Type": "application/json" };

/** Core fetch wrapper — returns parsed JSON or throws {status, message} */
async function apiFetch(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  try {
    const res  = await fetch(url, { headers: DEFAULT_HEADERS, ...options });
    const data = await res.json();
    if (!res.ok) throw { status: res.status, message: data.detail || data.message || "API Error" };
    return data;
  } catch (err) {
    if (err.status) throw err;
    throw { status: 0, message: `Network error — is the backend running? (${err.message})` };
  }
}

/** Multipart form upload (for PDF documents) */
async function apiUpload(path, formData) {
  const url = `${BASE_URL}${path}`;
  const res  = await fetch(url, { method: "POST", body: formData });
  const data = await res.json();
  if (!res.ok) throw { status: res.status, message: data.detail || "Upload failed" };
  return data;
}

const API = {
  // Health
  health:         ()   => apiFetch("/health"),
  dashboardStats: ()   => apiFetch("/api/patients/dashboard/stats"),

  // Patients
  getPatients:   (limit = 100) => apiFetch(`/api/patients/?limit=${limit}`),
  getPatient:    (id)          => apiFetch(`/api/patients/${id}`),
  createPatient: (data)        => apiFetch("/api/patients/", { method: "POST", body: JSON.stringify(data) }),
  searchPatients:(q)           => apiFetch(`/api/patients/search?q=${encodeURIComponent(q)}`),
  updatePatient: (id, data)    => apiFetch(`/api/patients/${id}`, { method: "PUT", body: JSON.stringify(data) }),

  // Documents
  uploadDocument: (formData)               => apiUpload("/api/documents/upload", formData),
  getPatientDocs: (id, limit = 20)         => apiFetch(`/api/documents/patient/${id}?limit=${limit}`),
  getRecentDocs:  (limit = 10)             => apiFetch(`/api/documents/recent?limit=${limit}`),
  analyzeText:    (patientId, text, dtype) =>
    apiFetch(`/api/documents/analyze-text?patient_id=${patientId}&text=${encodeURIComponent(text)}&document_type=${dtype}`, { method: "POST" }),

  // Risk Prediction
  predictRisk:      (data)        => apiFetch("/api/risk/predict", { method: "POST", body: JSON.stringify(data) }),
  latestRisk:       (id)          => apiFetch(`/api/risk/patient/${id}/latest`),
  riskHistory:      (id, lim=10)  => apiFetch(`/api/risk/patient/${id}/history?limit=${lim}`),
  riskTrend:        (id)          => apiFetch(`/api/risk/patient/${id}/trend`),
  highRiskPatients: ()            => apiFetch("/api/risk/all/high-risk"),
  modelMetrics:     ()            => apiFetch("/api/risk/model/metrics"),

  // What-If Simulation
  runSimulation:     (data)   => apiFetch("/api/simulation/run", { method: "POST", body: JSON.stringify(data) }),
  quickSim:          (params) => apiFetch(`/api/simulation/quick?${new URLSearchParams(params)}`),
  scenarioTemplates: ()       => apiFetch("/api/simulation/scenarios/default"),

  // Knowledge Graph
  patientGraph:    (id)   => apiFetch(`/api/graph/patient/${id}`),
  graphSummary:    ()     => apiFetch("/api/graph/summary"),
  highRiskGraph:   ()     => apiFetch("/api/graph/high-risk"),
  graphStats:      ()     => apiFetch("/api/graph/stats"),
  diseaseGraph:    (name) => apiFetch(`/api/graph/disease/${encodeURIComponent(name)}`),
  medicationGraph: (name) => apiFetch(`/api/graph/medication/${encodeURIComponent(name)}`),

  // Drug Safety
  drugSearch:       (name)   => apiFetch(`/api/drugs/search?drug_name=${encodeURIComponent(name)}`),
  drugInteractions: (a, b)   => apiFetch(`/api/drugs/interactions?drug_a=${encodeURIComponent(a)}&drug_b=${encodeURIComponent(b)}`),
  drugRecalls:      (name)   => apiFetch(`/api/drugs/recalls?drug_name=${encodeURIComponent(name)}`),
  patientMedCheck:  (data)   => apiFetch("/api/drugs/patient-check", { method: "POST", body: JSON.stringify(data) }),
};

window.API = API;