"""
backend/main.py
Purpose: FastAPI application entry point.
         Configures middleware, mounts all routers, handles lifespan events.
"""
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import settings
from backend.db.mongodb import connect_db, disconnect_db
from backend.db.neo4j_db import connect_neo4j, disconnect_neo4j
from backend.api import documents, graph, risk, simulation, drugs, patients

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# LIFESPAN (startup / shutdown)
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage database connections on startup and shutdown."""
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")

    # Startup
    try:
        await connect_db()
    except Exception as e:
        logger.error(f"MongoDB startup failed: {e}. Continuing without MongoDB.")

    try:
        await connect_neo4j()
    except Exception as e:
        logger.error(f"Neo4j startup failed: {e}. Continuing without Neo4j.")

    # Ensure ML model is ready
    try:
        from backend.ml.predict import _load_model
        _load_model()
    except Exception as e:
        logger.warning(f"ML model not loaded at startup: {e}. Will load on first request.")

    logger.info("✅ All services initialized. API ready.")
    yield

    # Shutdown
    logger.info("Shutting down services...")
    await disconnect_db()
    await disconnect_neo4j()
    logger.info("Goodbye.")


# ─────────────────────────────────────────────
# APP FACTORY
# ─────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## MediGraph AI — Healthcare Knowledge Graph Platform

An agentic healthcare intelligence system combining:
- **LangGraph** multi-agent AI orchestration
- **Neo4j** knowledge graph
- **XGBoost** adherence risk prediction
- **Gemini AI** for natural language reasoning
- **OpenFDA** drug safety data

### Key Features
- 📄 Document Intelligence (PDF → structured entities)
- 🧠 Healthcare Knowledge Graph
- 📊 Medication Adherence Risk Prediction
- 🔮 What-If Care Simulation Engine
- 💊 Drug Safety Center (OpenFDA)
- 🎯 AI-Powered Intervention Recommendations
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# ─────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add X-Process-Time header to every response."""
    start_time = time.time()
    response = await call_next(request)
    process_time = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time"] = f"{process_time}ms"
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"← {response.status_code} {request.url.path}")
    return response


# ─────────────────────────────────────────────
# GLOBAL ERROR HANDLER
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc), "message": "Internal server error"}
    )


# ─────────────────────────────────────────────
# ROUTERS
# ─────────────────────────────────────────────

app.include_router(patients.router)
app.include_router(documents.router)
app.include_router(graph.router)
app.include_router(risk.router)
app.include_router(simulation.router)
app.include_router(drugs.router)


# ─────────────────────────────────────────────
# ROOT / HEALTH ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check for all connected services."""
    from backend.db.mongodb import get_db
    from backend.db.neo4j_db import get_driver

    health = {"status": "healthy", "services": {}}

    # MongoDB
    try:
        db = get_db()
        await db.command("ping")
        health["services"]["mongodb"] = "connected"
    except Exception as e:
        health["services"]["mongodb"] = f"error: {str(e)}"
        health["status"] = "degraded"

    # Neo4j
    try:
        driver = get_driver()
        await driver.verify_connectivity()
        health["services"]["neo4j"] = "connected"
    except Exception as e:
        health["services"]["neo4j"] = f"error: {str(e)}"
        health["status"] = "degraded"

    # ML Model
    try:
        from backend.ml.predict import is_model_loaded
        health["services"]["ml_model"] = "loaded" if is_model_loaded() else "not loaded"
    except Exception:
        health["services"]["ml_model"] = "unavailable"

    return health


# ─────────────────────────────────────────────
# DEV RUNNER
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )