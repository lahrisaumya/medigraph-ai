/**
 * frontend/components/sidebar.js
 * Persistent left sidebar — call Sidebar.init(activeId) once per page.
 * Auto-detects whether we're at root (index.html) or in /pages/.
 */

const NAV_ITEMS = [
  { id: "overview",        icon: "📊", label: "Executive Overview",    file: "/index.html"                   },
  { id: "documents",       icon: "📄", label: "Document Intelligence", file: "/pages/documents.html"         },
  { id: "knowledge-graph", icon: "🧠", label: "Knowledge Graph",       file: "/pages/knowledge_graph.html"   },
  { id: "risk",            icon: "📈", label: "Risk Prediction",       file: "/pages/risk.html"              },
  { id: "drugs",           icon: "💊", label: "Drug Safety Center",    file: "/pages/drugs.html"             },
  { id: "simulation",      icon: "🔮", label: "Care Simulation",       file: "/pages/simulation.html"        },
  { id: "recommendations", icon: "🎯", label: "AI Recommendations",   file: "/pages/recommendations.html"   },
];

const Sidebar = {
  init(activeId) {
    const sidebar = document.getElementById("sidebar");
    if (!sidebar) return;

    // Detect depth: pages inside /pages/ need ../ prefix to reach root assets
    const inSubfolder = window.location.pathname.includes("/pages/");
    const root = inSubfolder ? "../" : "";

    sidebar.innerHTML = `
      <div class="sidebar-brand">
        <div class="brand-icon">⚕</div>
        <div>
          <div class="brand-name">MediGraph AI</div>
          <div class="brand-sub">Healthcare Intelligence</div>
        </div>
      </div>
      <nav class="sidebar-nav">
        ${NAV_ITEMS.map(item => {
          const href = item.file;
          return `<a href="${href}" class="nav-item ${item.id === activeId ? "active" : ""}" data-id="${item.id}">
            <span class="nav-icon">${item.icon}</span>
            <span class="nav-label">${item.label}</span>
          </a>`;
        }).join("")}
      </nav>
      <div class="sidebar-footer">
        <div class="status-dot" id="status-dot"></div>
        <span id="status-text">Checking…</span>
      </div>`;

    // Health check
    fetch("http://localhost:8000/health")
      .then(r => r.json())
      .then(d => {
        const ok = d.status === "healthy";
        document.getElementById("status-dot").className = `status-dot ${ok ? "online" : "degraded"}`;
        document.getElementById("status-text").textContent = ok ? "All systems online" : "Degraded";
      })
      .catch(() => {
        document.getElementById("status-dot").className = "status-dot offline";
        document.getElementById("status-text").textContent = "Backend offline";
      });
  }
};

window.Sidebar = Sidebar;