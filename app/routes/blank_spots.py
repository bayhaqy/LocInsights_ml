"""GET /blank_spots — Recommended new-location candidates in Bali."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.ml.scoring import haversine_m, compute_features, fallback_score, gbr_predict
from app.supabase_client import get_supabase

router = APIRouter()
log = logging.getLogger(__name__)


class BlankSpotRequest(BaseModel):
    """Filter blank-spot search."""
    kab_filter: Optional[List[str]] = Field(None, description="Restrict to kabupaten codes (e.g., ['5175'])")
    tier_filter: Optional[List[str]] = Field(None, description="Restrict to kelurahan tiers ('1','2','3')")
    min_score: float = Field(0.6, ge=0, le=1, description="Minimum score threshold")
    limit: int = Field(20, ge=1, le=100)
    radius_m: int = Field(1000, description="Minimum distance from existing MAA stores")


class BlankSpot(BaseModel):
    kelurahan_id: str
    kelurahan_name: str
    kab: str
    city: str
    lat: float
    lng: float
    score: float
    score_pct: float
    is_coastal: bool
    tier: str
    nearest_maa_store_m: float
    nearest_competitor_m: float
    recommendation: str


class BlankSpotResponse(BaseModel):
    count: int
    spots: List[BlankSpot]
    criteria: Dict[str, Any]


@router.get("/blank_spots", response_model=BlankSpotResponse)
async def find_blank_spots(req: BlankSpotRequest, request: Request):
    """Find recommended blank-spot areas in Bali.

    A "blank spot" is a kelurahan centroid that:
      - Has NO existing MAA store within `radius_m` meters
      - Has a model score >= `min_score`
      - Optionally filtered by kabupaten or tier

    Returns top candidates sorted by score (desc).
    """
    sb = get_supabase()

    # 1. Fetch data
    try:
        kelurahan = await sb.select(
            "kelurahan",
            columns="id,name,kab_name,city,lat,lng,tier,is_coastal,density,income_index,tourist_index,transport_index",
            limit=5000,
        )
        stores = await sb.select("stores", columns="id,lat,lng,brand_id", limit=2000)
        competitors = await sb.select("competitor_stores", columns="name,lat,lng", limit=5000)
        pois = await sb.select("pois", columns="name,lat,lng,type", limit=2000)
        malls = await sb.select("malls", columns="name,lat,lng", limit=500)
    except Exception as e:
        log.warning(f"Supabase read failed: {e}")
        return BlankSpotResponse(count=0, spots=[], criteria=req.dict())

    # 2. Filter kelurahan
    if req.kab_filter:
        kelurahan = [k for k in kelurahan if k.get("kab_name") in req.kab_filter]
    if req.tier_filter:
        kelurahan = [k for k in kelurahan if str(k.get("tier")) in req.tier_filter]

    log.info(f"Evaluating {len(kelurahan)} kelurahan for blank spots")

    # 3. Score each kelurahan
    model = getattr(request.app.state, "model", None)
    spots: List[BlankSpot] = []

    for k in kelurahan:
        lat, lng = float(k["lat"]), float(k["lng"])

        # Check distance to nearest MAA store
        nearest_store = min(
            (haversine_m(lat, lng, s["lat"], s["lng"]) for s in stores),
            default=99999,
        )
        if nearest_store < req.radius_m:
            continue  # Too close to existing store

        nearest_comp = min(
            (haversine_m(lat, lng, c["lat"], c["lng"]) for c in competitors),
            default=99999,
        )

        # Compute features + score
        features = compute_features(lat, lng, competitors, pois, malls, k, False)
        if model is not None:
            score, _ = gbr_predict(model, features)
        else:
            score, _ = fallback_score(features)

        if score < req.min_score:
            continue

        if score >= 0.75:
            rec = "STRONG GO"
        elif score >= 0.65:
            rec = "GO"
        else:
            rec = "HOLD"

        spots.append(BlankSpot(
            kelurahan_id=k["id"],
            kelurahan_name=k.get("name", ""),
            kab=k.get("kab_name", ""),
            city=k.get("city", ""),
            lat=lat,
            lng=lng,
            score=round(score, 4),
            score_pct=round(score * 100, 1),
            is_coastal=bool(k.get("is_coastal")),
            tier=str(k.get("tier", "")),
            nearest_maa_store_m=round(nearest_store, 0),
            nearest_competitor_m=round(nearest_comp, 0),
            recommendation=rec,
        ))

    # 4. Sort + limit
    spots.sort(key=lambda s: -s.score)
    spots = spots[: req.limit]

    return BlankSpotResponse(
        count=len(spots),
        spots=spots,
        criteria={
            "kab_filter": req.kab_filter,
            "tier_filter": req.tier_filter,
            "min_score": req.min_score,
            "radius_m": req.radius_m,
            "evaluated": len(kelurahan),
        },
    )
