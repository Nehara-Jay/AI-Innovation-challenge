"""
FastAPI application entry point.
Includes CORS middleware, router registration, health checks, and Swagger documentation.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from backend.config import settings
from backend.models import HealthResponse
from backend.dependencies import get_qdrant
from backend.routers.ask import router as ask_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifespan events."""
    print("=" * 60)
    print(f"Starting {settings.project_name} (v{settings.app_version})")
    print(f"Qdrant URL: {settings.qdrant_url}")
    print(f"Voyage AI Model: {settings.voyage_model}")
    print(f"LLM Reasoning Model: {settings.llm_model}")
    try:
        qdrant = get_qdrant()
        collections = [c.name for c in qdrant.get_collections().collections]
        if settings.qdrant_collection in collections:
            count = qdrant.get_collection(settings.qdrant_collection).points_count
            print(f"Qdrant Connected: collection '{settings.qdrant_collection}' has {count} points.")
        else:
            print(f"[Warning] Collection '{settings.qdrant_collection}' not found in Qdrant.")
    except Exception as e:
        print(f"[Warning] Could not connect to Qdrant at startup: {e}")
    print("=" * 60)
    yield
    print("Shutting down API server...")


app = FastAPI(
    title=settings.project_name,
    version=settings.app_version,
    description="Multimodal Multi-Hop RAG & Reasoning Engine for the Ashen Era Archive (SLIIT Codefest 2026).",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for Frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(ask_router, prefix="/api")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint providing service information."""
    return {
        "project": settings.project_name,
        "version": settings.app_version,
        "status": "online",
        "docs_url": "/docs",
        "api_endpoints": {
            "ask": "/api/ask",
            "search": "/api/search",
            "health": "/health",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System Health"],
    summary="Check system health and database connectivity",
)
async def health_check() -> HealthResponse:
    """Returns the operational status of Qdrant and AI models."""
    qdrant_connected = False
    total_vectors = 0

    try:
        qdrant = get_qdrant()
        col_info = qdrant.get_collection(settings.qdrant_collection)
        total_vectors = col_info.points_count or 0
        qdrant_connected = True
        qdrant_status_str = "connected"
    except Exception as e:
        qdrant_status_str = f"disconnected ({str(e)})"

    return HealthResponse(
        status="healthy" if qdrant_connected else "degraded",
        qdrant_status=qdrant_status_str,
        total_vectors=total_vectors,
        collection_name=settings.qdrant_collection,
        voyage_model=settings.voyage_model,
        llm_model=settings.llm_model,
    )


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
