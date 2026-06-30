/**
 * frontend/utils/helpers.js
 * Shared UI helper functions: formatting, DOM shortcuts, toasts, loaders.
 */

// ── Formatting ────────────────────────────────────────────────────────────────

function riskBadge(level) {
  const map = {
    LOW:      { cls: "badge-low",      icon: "✅", label: "LOW" },
    MODERATE: { cls: "badge-moderate", icon: "🟡", label: "MODERATE" },
    HIGH:     { cls: "badge-high",     icon: "🟠", label: "HIGH" },
    CRITICAL: { cls: "badge-critical", icon: "🔴", label: "CRITICAL" },
  };
  const b = map[level] || { cls: "badge-moderate", icon: "❓", label: level };
  return `<span class="badge ${b.cls}">${b.icon} ${b.label}</span>`;
}

function adherenceBadge(level) {
  const map = {
    EXCELLENT: { cls: "badge-low",      label: "EXCELLENT" },
    GOOD:      { cls: "badge-moderate", label: "GOOD" },
    POOR:      { cls: "badge-high",     label: "POOR" },
    CRITICAL:  { cls: "badge-critical", label: "CRITICAL" },
  };
  const b = map[level] || { cls: "badge-moderate", label: level };
  return `<span class="badge ${b.cls}">${b.label}</span>`;
}

function formatScore(score) {
  if (score == null) return "–";
  const s = parseFloat(score);
  const color = s >= 75 ? "#7C3AED" : s >= 55 ? "#EF4444" : s >= 35 ? "#F59E0B" : "#10B981";
  return `<span style="color:${color};font-weight:700">${s.toFixed(1)}%</span>`;
}

function formatDate(dateStr) {
  if (!dateStr) return "–";
  try { return new Date(dateStr).toLocaleDateString("en-IN", { day:"2-digit", month:"short", year:"numeric" }); }
  catch { return dateStr; }
}

function formatList(arr) {
  if (!arr || !arr.length) return '<span class="text-muted">None</span>';
  return arr.map(i => `<span class="tag">${i}</span>`).join(" ");
}

function capitalize(str) {
  return str ? str.charAt(0).toUpperCase() + str.slice(1) : "";
}

// ── DOM helpers ───────────────────────────────────────────────────────────────

function el(id) { return document.getElementById(id); }

function setHTML(id, html) {
  const node = el(id);
  if (node) node.innerHTML = html;
}

function show(id) { const n = el(id); if (n) n.style.display = ""; }
function hide(id) { const n = el(id); if (n) n.style.display = "none"; }

function showLoader(id, msg = "Loading…") {
  setHTML(id, `<div class="loader-wrap"><div class="spinner"></div><p>${msg}</p></div>`);
}

function showError(id, msg) {
  setHTML(id, `<div class="error-box">⚠️ ${msg}</div>`);
}

// ── Toast notifications ───────────────────────────────────────────────────────

function toast(msg, type = "info", duration = 3500) {
  let container = el("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    document.body.appendChild(container);
  }
  const t = document.createElement("div");
  t.className = `toast toast-${type}`;
  t.innerHTML = `<span>${msg}</span>`;
  container.appendChild(t);
  setTimeout(() => { t.classList.add("fade-out"); setTimeout(() => t.remove(), 400); }, duration);
}

// ── KPI Card builder ──────────────────────────────────────────────────────────

function kpiCard(icon, label, value, subtext = "", color = "#3B82F6") {
  return `
  <div class="kpi-card">
    <div class="kpi-icon" style="color:${color}">${icon}</div>
    <div class="kpi-body">
      <div class="kpi-value">${value}</div>
      <div class="kpi-label">${label}</div>
      ${subtext ? `<div class="kpi-sub">${subtext}</div>` : ""}
    </div>
  </div>`;
}

// ── Table builder ─────────────────────────────────────────────────────────────

function buildTable(headers, rows, emptyMsg = "No data") {
  if (!rows || !rows.length) return `<p class="no-data">${emptyMsg}</p>`;
  const ths = headers.map(h => `<th>${h}</th>`).join("");
  const trs = rows.map(r => `<tr>${r.map(c => `<td>${c ?? "–"}</td>`).join("")}</tr>`).join("");
  return `<table class="data-table"><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;
}

// ── Debounce ──────────────────────────────────────────────────────────────────

function debounce(fn, delay = 300) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), delay); };
}

// ── Export ────────────────────────────────────────────────────────────────────

window.H = {
  riskBadge, adherenceBadge, formatScore, formatDate, formatList, capitalize,
  el, setHTML, show, hide, showLoader, showError,
  toast, kpiCard, buildTable, debounce,
};