"""POST /predict — Site Success Probability Score."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.ml.scoring import (
    FEATURE_NAMES,
    compute_features,
    fallback_score,
    gbr_predict,
)
from app.supabase_client import get_supabase

router = APIRouter()
log = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Candidate site latitude")
    lng: float = Field(..., ge=-180, le=180, description="Candidate site longitude")
    is_in_mall: bool = Field(False, description="True if candidate is inside a mall")
    kelurahan_id: Optional[str] = Field(None, description="Optional: kelurahan ID for demographic features")
    brand_id: Optional[str] = Field(None, description="Optional: brand ID (affects category fit)")
    save: bool = Field(False, description="If true, save prediction to Supabase predictions table")


class PredictResponse(BaseModel):
    score: float = Field(..., ge=0, le=1, description="Success probability 0-1")
    score_pct: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    model: str
    model_version: str
    features_used: Dict[str, float]
    feature_contributions: Dict[str, float]
    recommendation: str
    is_blank_spot: bool


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request):
    """Compute Store Success Probability Score for a candidate site.

    The model is a Gradient Boosting Regressor trained on:
      - competitor density (1km + 3km radius)
      - POI density (1km radius)
      - distance to nearest mall
      - kelurahan-level income / population density / tourist / transport indices
      - coastal flag, mall flag

    If no trained artifact is loaded, falls back to a transparent weighted heuristic.
    """
    # 1. Pull data from Supabase around the candidate site
    sb = get_supabase()
    try:
        competitors = await sb.select(
            "competitor_stores",
            columns="name,lat,lng,brand_name",
            limit=2000,
        )
        pois = await sb.select("pois", columns="name,lat,lng,type", limit=2000)
        malls = await sb.select("malls", columns="name,lat,lng", limit=500)
        kelurahan = None
        if req.kelurahan_id:
            kl = await sb.select("kelurahan", columns="*", limit=1, filters={"id": req.kelurahan_id})
            kelurahan = kl[0] if kl else None
    except Exception as e:
        log.warning(f"Supabase read failed, using empty feature context: {e}")
        competitors, pois, malls, kelurahan = [], [], [], None

    # 2. Compute the 10 canonical features
    features = compute_features(
        req.lat, req.lng, competitors, pois, malls, kelurahan, req.is_in_mall
    )

    # 3. Run model (or fallback)
    model = getattr(request.app.state, "model", None)
    meta = getattr(request.app.state, "model_metadata", {}) or {}
    if model is not None:
        score, breakdown = gbr_predict(model, features)
        model_name = meta.get("name", "gbr_site_scoring")
        model_version = meta.get("version", "1.0.0")
    else:
        score, breakdown = fallback_score(features)
        model_name = "fallback_heuristic"
        model_version = "0.1.0"

    score_pct = round(score * 100, 1)
    # Confidence: higher when more data context is available
    confidence = min(1.0, 0.4 + 0.1 * (len(competitors) > 50) + 0.1 * (len(pois) > 30) + 0.2 * (kelurahan is not None) + 0.2 * (model is not None))

    # 4. Recommendation logic
    if score >= 0.75:
        rec = "STRONG GO — High-priority expansion candidate. Proceed to field survey."
    elif score >= 0.55:
        rec = "GO — Promising candidate. Validate with field survey + lease negotiation."
    elif score >= 0.35:
        rec = "HOLD — Marginal. Re-evaluate with different brand format or wait for market maturation."
    else:
        rec = "NO-GO — Low probability of success. Avoid this location."

    # 5. Blank spot = high score + no existing MAA store nearby
    is_blank_spot = score >= 0.65 and features["competitor_density_1km"] == 0

    # 6. Optionally persist
    if req.save:
        try:
            await sb.insert("predictions", [{
                "model_id": meta.get("id", "fallback"),
                "target_type": "candidate_site",
                "target_id": req.kelurahan_id or f"{req.lat:.4f},{req.lng:.4f}",
                "target_name": f"Candidate site @ {req.lat:.4f}, {req.lng:.4f}",
                "lat": req.lat,
                "lng": req.lng,
                "prediction": score,
                "confidence": confidence,
                "explanation": {
                    "features": features,
                    "contributions": breakdown,
                    "model": model_name,
                },
                "is_blank_spot": is_blank_spot,
            }])
        except Exception as e:
            log.warning(f"Failed to save prediction to Supabase: {e}")

    return PredictResponse(
        score=round(score, 4),
        score_pct=score_pct,
        confidence=round(confidence, 3),
        model=model_name,
        model_version=model_version,
        features_used=features,
        feature_contributions=breakdown,
        recommendation=rec,
        is_blank_spot=is_blank_spot,
    )
