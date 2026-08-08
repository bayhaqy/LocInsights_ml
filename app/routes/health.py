"""Health endpoints — no auth required (used by Vercel cron)."""
from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    supabase_configured: bool
    model_loaded: bool


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Lightweight liveness probe — Vercel cron hits this every 15 min.

    Returns 200 even if Supabase or model is not yet configured, so that
    Hugging Face Spaces doesn't sleep. The cron job itself does a separate
    SELECT 1 on Supabase via the Vercel API route.
    """
    model = getattr(request.app.state, "model", None)
    settings = request.app.state.__dict__ if hasattr(request.app.state, "__dict__") else {}
    from app.config import get_settings
    s = get_settings()
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0",
        supabase_configured=bool(s.supabase_url and s.supabase_service_role_key),
        model_loaded=model is not None,
    )


@router.get("/model/info")
async def model_info(request: Request):
    """Return metadata about the currently-loaded model."""
    meta = getattr(request.app.state, "model_metadata", None)
    if not meta:
        from app.config import get_settings
        s = get_settings()
        return {
            "name": s.model_name,
            "version": s.model_version,
            "algorithm": "gradient_boosting_regressor",
            "status": "default_fallback",
            "features": [
                "competitor_density_1km",
                "competitor_density_3km",
                "poi_density_1km",
                "mall_distance_m",
                "income_index",
                "population_density",
                "tourist_index",
                "transport_index",
                "is_coastal",
                "is_in_mall",
            ],
        }
    return meta
