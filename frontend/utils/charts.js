/**
 * frontend/utils/charts.js
 * Shared Plotly chart factory functions used across all dashboard pages.
 * Each function renders a chart into a given div ID.
 */

const COLORS = {
  LOW:      "#10B981",
  MODERATE: "#F59E0B",
  HIGH:     "#EF4444",
  CRITICAL: "#7C3AED",
  blue:     "#3B82F6",
  slate:    "#64748B",
  bg:       "#0F172A",
  surface:  "#1E293B",
  border:   "#334155",
  text:     "#F1F5F9",
  subtext:  "#94A3B8",
};

const PLOTLY_BASE = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor:  "rgba(0,0,0,0)",
  font:          { color: COLORS.text, family: "Inter, sans-serif", size: 12 },
  margin:        { t: 30, b: 40, l: 50, r: 20 },
  legend:        { bgcolor: "rgba(0,0,0,0)", font: { color: COLORS.text } },
};

/** Risk distribution donut chart */
function renderRiskDonut(divId, distribution) {
  const labels = Object.keys(distribution);
  const values = Object.values(distribution);
  const colorMap = { LOW: COLORS.LOW, MODERATE: COLORS.MODERATE, HIGH: COLORS.HIGH, CRITICAL: COLORS.CRITICAL };
  const colors = labels.map(l => colorMap[l] || COLORS.blue);

  Plotly.newPlot(divId, [{
    type: "pie", hole: 0.6,
    labels, values,
    marker: { colors, line: { color: COLORS.bg, width: 2 } },
    textinfo: "label+percent",
    textfont: { color: COLORS.text, size: 11 },
    hovertemplate: "<b>%{label}</b><br>Patients: %{value}<br>%{percent}<extra></extra>",
  }], {
    ...PLOTLY_BASE,
    showlegend: true,
    annotations: [{ text: "Risk\nLevels", x: 0.5, y: 0.5, showarrow: false, font: { size: 13, color: COLORS.text } }],
  }, { responsive: true, displayModeBar: false });
}

/** Risk trend line chart */
function renderRiskTrendLine(divId, series) {
  if (!series || !series.length) { document.getElementById(divId).innerHTML = `<p class="no-data">No trend data</p>`; return; }
  const x = series.map(s => new Date(s.date).toLocaleDateString());
  const y = series.map(s => s.risk_score);

  Plotly.newPlot(divId, [{
    x, y, type: "scatter", mode: "lines+markers",
    line:   { color: COLORS.HIGH, width: 2 },
    marker: { color: COLORS.HIGH, size: 6 },
    fill:   "tozeroy", fillcolor: "rgba(239,68,68,0.08)",
    hovertemplate: "<b>%{x}</b><br>Risk: %{y:.1f}%<extra></extra>",
    name: "Risk Score",
  }], {
    ...PLOTLY_BASE,
    xaxis: { gridcolor: COLORS.border, zeroline: false },
    yaxis: { gridcolor: COLORS.border, range: [0, 100], title: "Risk %" },
  }, { responsive: true, displayModeBar: false });
}

/** Horizontal bar chart for feature importances */
function renderFeatureImportance(divId, importances) {
  if (!importances) return;
  const entries = Object.entries(importances).sort((a,b) => b[1]-a[1]).slice(0, 8);
  const labels  = entries.map(e => e[0].replace(/_/g," "));
  const values  = entries.map(e => e[1]);

  Plotly.newPlot(divId, [{
    type: "bar", orientation: "h",
    x: values, y: labels,
    marker: { color: values.map(v => `rgba(59,130,246,${0.4 + v * 0.6})`), line: { width: 0 } },
    hovertemplate: "<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
  }], {
    ...PLOTLY_BASE,
    xaxis: { gridcolor: COLORS.border, title: "Importance" },
    yaxis: { gridcolor: COLORS.border },
    margin: { ...PLOTLY_BASE.margin, l: 180 },
  }, { responsive: true, displayModeBar: false });
}

/** Scatter plot: adherence vs risk score */
function renderAdherenceScatter(divId, patients, riskMap) {
  const x = [], y = [], text = [], colors = [];
  patients.forEach(p => {
    const risk = riskMap[p.patient_id];
    if (risk !== undefined) {
      x.push(p.adherence_rate || 0);
      y.push(risk);
      text.push(`${p.name} (${p.patient_id})`);
      colors.push(risk >= 75 ? COLORS.CRITICAL : risk >= 55 ? COLORS.HIGH : risk >= 35 ? COLORS.MODERATE : COLORS.LOW);
    }
  });

  Plotly.newPlot(divId, [{
    type: "scatter", mode: "markers",
    x, y, text,
    marker: { color: colors, size: 10, opacity: 0.85, line: { color: "#fff", width: 1 } },
    hovertemplate: "<b>%{text}</b><br>Adherence: %{x:.0f}%<br>Risk: %{y:.1f}%<extra></extra>",
  }], {
    ...PLOTLY_BASE,
    xaxis: { title: "Adherence Rate (%)", gridcolor: COLORS.border, range: [0, 105] },
    yaxis: { title: "Risk Score (%)",    gridcolor: COLORS.border, range: [0, 105] },
  }, { responsive: true, displayModeBar: false });
}

/** Grouped bar chart for simulation scenarios */
function renderSimulationBars(divId, scenarios, baseline) {
  const labels = scenarios.map(s => s.label);
  const scores = scenarios.map(s => s.risk_score);
  const colors = scores.map(s => s >= 75 ? COLORS.CRITICAL : s >= 55 ? COLORS.HIGH : s >= 35 ? COLORS.MODERATE : COLORS.LOW);

  Plotly.newPlot(divId, [
    {
      type: "bar", name: "Scenario Risk",
      x: labels, y: scores,
      marker: { color: colors, line: { width: 0 } },
      hovertemplate: "<b>%{x}</b><br>Risk: %{y:.1f}%<extra></extra>",
    },
    {
      type: "scatter", mode: "lines", name: "Baseline",
      x: labels, y: Array(labels.length).fill(baseline),
      line: { dash: "dot", color: COLORS.subtext, width: 2 },
      hoverinfo: "skip",
    },
  ], {
    ...PLOTLY_BASE,
    barmode: "group",
    yaxis: { gridcolor: COLORS.border, range: [0, 100], title: "Risk Score %" },
    xaxis: { gridcolor: COLORS.border },
    legend: { orientation: "h", y: -0.15 },
  }, { responsive: true, displayModeBar: false });
}

/** Adverse events bar chart (Drug Safety) */
function renderAdverseEvents(divId, events) {
  if (!events || !events.length) { document.getElementById(divId).innerHTML = `<p class="no-data">No adverse event data</p>`; return; }
  const top = events.slice(0, 10);
  Plotly.newPlot(divId, [{
    type: "bar", orientation: "h",
    x: top.map(e => e.count),
    y: top.map(e => e.term),
    marker: { color: "rgba(239,68,68,0.7)", line: { width: 0 } },
    hovertemplate: "<b>%{y}</b><br>Reports: %{x:,}<extra></extra>",
  }], {
    ...PLOTLY_BASE,
    xaxis: { gridcolor: COLORS.border, title: "Report Count" },
    yaxis: { gridcolor: COLORS.border },
    margin: { ...PLOTLY_BASE.margin, l: 180 },
  }, { responsive: true, displayModeBar: false });
}

/** Node type bar chart for knowledge graph summary */
function renderNodeTypeBars(divId, nodeCounts) {
  if (!nodeCounts) return;
  const entries = Object.entries(nodeCounts).sort((a,b) => b[1]-a[1]);
  const typeColors = { Patient: "#3B82F6", Disease: "#EF4444", Medication: "#10B981", Symptom: "#F59E0B", LabTest: "#8B5CF6", RiskFactor: "#EC4899" };

  Plotly.newPlot(divId, [{
    type: "bar",
    x: entries.map(e => e[0]),
    y: entries.map(e => e[1]),
    marker: { color: entries.map(e => typeColors[e[0]] || COLORS.blue), line: { width: 0 } },
    hovertemplate: "<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
  }], {
    ...PLOTLY_BASE,
    xaxis: { gridcolor: COLORS.border },
    yaxis: { gridcolor: COLORS.border, title: "Node Count" },
  }, { responsive: true, displayModeBar: false });
}

window.Charts = {
  renderRiskDonut,
  renderRiskTrendLine,
  renderFeatureImportance,
  renderAdherenceScatter,
  renderSimulationBars,
  renderAdverseEvents,
  renderNodeTypeBars,
  COLORS,
};