"""
LocInsight ML Engine — FastAPI application entrypoint
=====================================================
Hosted on Hugging Face Spaces (Docker SDK).

Endpoints:
  GET  /health            → liveness probe (used by Vercel cron)
  GET  /                  → service info
  POST /predict           → site success probability score (0-100%)
  POST /scrape_bali       → trigger Bali scraping job (async)
  GET  /scrape_bali/{jid} → polling status for a scrape job
  POST /train             → retrain model from latest Supabase data
  GET  /model/info        → current active model metadata
  GET  /blank_spots       → recommended blank spot candidates in Bali

Security:
  All endpoints (except /health and /) require a custom Bearer token
  passed via the `X-LocInsight-Token` header. The token is read from
  the `LOCINSIGHT_API_TOKEN` env var (set as a HF Space secret).

Architecture:
  - FastAPI + uvicorn (single worker; HF Spaces CPU basic tier)
  - Scikit-learn for Gradient Boosting Regressor (GBR) site scoring
  - HDBSCAN optional for blank-spot clustering (fallback: KMeans)
  - Supabase as data source + result sink (via service_role key)
  - Background tasks via FastAPI BackgroundTasks (no Celery/Redis for PoC)
"""
from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import get_settings
from app.routes import predict, scrape, train, blank_spots, health

settings = get_settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("locinsight_ml")


# =============================================================
# Security: custom Bearer token via X-LocInsight-Token header
# =============================================================
api_key_header = APIKeyHeader(
    name="X-LocInsight-Token",
    auto_error=False,
    description="Custom bearer token securing this API (only Vercel should call).",
)


async def verify_token(request: Request, api_key: Optional[str] = Security(api_key_header)):
    """Skip auth for /health and / ; enforce for all other routes."""
    path = request.url.path.rstrip("/")
    if path in ("", "/health", "/docs", "/openapi.json", "/redoc"):
        return None
    if not settings.locinsight_api_token:
        # If no token configured (e.g., local dev), allow all (NOT for production)
        log.warning("LOCINSIGHT_API_TOKEN not set — running in UNSECURED mode (dev only)")
        return None
    if api_key != settings.locinsight_api_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-LocInsight-Token. Access restricted to LocInsight backend.",
        )
    return api_key


# =============================================================
# App lifespan: preload model on startup
# =============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("LocInsight ML Engine starting up...")
    log.info(f"  Supabase URL: {settings.supabase_url}")
    log.info(f"  HF Space:     {os.getenv('SPACE_HOST', 'local')}")
    log.info(f"  Auth enabled: {bool(settings.locinsight_api_token)}")
    # Preload model artifact (download from HF hub if not cached)
    try:
        from app.ml.model_loader import load_active_model
        model, metadata = load_active_model()
        app.state.model = model
        app.state.model_metadata = metadata
        log.info(f"  Active model: {metadata.get('name','?')} v{metadata.get('version','?')}")
    except Exception as e:
        log.warning(f"  Model preload failed (will use default): {e}")
        app.state.model = None
        app.state.model_metadata = None
    yield
    log.info("LocInsight ML Engine shutting down.")


# =============================================================
# FastAPI app
# =============================================================
app = FastAPI(
    title="LocInsight ML Engine",
    description="Site selection scoring + scraping worker for MAP Active Adiperkasa",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "health", "description": "Liveness & readiness probes"},
        {"name": "predict", "description": "Site success probability scoring"},
        {"name": "scrape", "description": "Bali scraping jobs (async)"},
        {"name": "train", "description": "Model retraining"},
        {"name": "blank_spots", "description": "Recommended new-location candidates"},
    ],
)

# CORS — only allow Vercel app origin (read from env)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount route modules
app.include_router(health.router, tags=["health"])
app.include_router(predict.router, tags=["predict"], dependencies=[Security(verify_token)])
app.include_router(scrape.router, tags=["scrape"], dependencies=[Security(verify_token)])
app.include_router(train.router, tags=["train"], dependencies=[Security(verify_token)])
app.include_router(blank_spots.router, tags=["blank_spots"], dependencies=[Security(verify_token)])


@app.get("/", tags=["health"])
async def root():
    """Service info — no auth required (used for quick public check)."""
    return {
        "service": "LocInsight ML Engine",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs",
        "endpoints": [
            "GET  /health",
            "POST /predict",
            "POST /scrape_bali",
            "GET  /scrape_bali/{job_id}",
            "POST /train",
            "GET  /model/info",
            "GET  /blank_spots",
        ],
    }
