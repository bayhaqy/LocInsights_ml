"""
LocInsight ML Engine — PyScript (browser-based Python via Pyodide)
==================================================================

This module runs entirely in the user's browser using Pyodide (WebAssembly)
via PyScript. No server-side compute is required — compatible with HF Spaces
free tier (static SDK).

Capabilities:
  1. Health check — service info + connectivity to Supabase
  2. Predict site score — 0-100% success probability for a candidate location
  3. Find blank spots — top recommended new-location candidates in Bali
  4. Train GBR model — in-browser Gradient Boosting Regressor on synthetic data
  5. Data explorer — view raw master data pulled from Supabase

Data source: Supabase REST API (PostgREST) using the publishable (anon) key.
RLS policy allows anon read on master tables but blocks all writes.

Maintained by: Achmad Bayhaqy — Data Team, MAP Active Adiperkasa (MAA)
Last updated: 2026-08-08
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pyodide.http import pyfetch


# =============================================================
# Configuration — Supabase publishable (anon) key is safe to expose
# =============================================================
SUPABASE_URL = "https://fcyhrzzfvdsghtummizv.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_qoO6_bu4mcgG1fmjsH3Gug_BMTXtCZf"

FEATURE_NAMES: List[str] = [
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
]

# In-memory model (populated by train_gbr_model)
_TRAINED_MODEL: Any = None
_MODEL_META: Dict[str, Any] = {}


# =============================================================
# Supabase REST client (browser-based, uses pyfetch)
# =============================================================
async def sb_select(table: str, columns: str = "*", limit: int = 1000) -> List[Dict[str, Any]]:
    """SELECT rows from a Supabase table via PostgREST."""
    url = f"{SUPABASE_URL}/rest/v1/{table}?select={columns}&limit={limit}"
    r = await pyfetch(
        url,
        method="GET",
        headers={
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Accept": "application/json",
        },
    )
    text = await r.string()
    return json.loads(text)


async def sb_ping() -> Dict[str, Any]:
    """Lightweight connectivity check — fetch 1 row from brands table."""
    try:
        rows = await sb_select("brands", columns="id,name", limit=1)
        return {
            "supabase_reachable": True,
            "url": SUPABASE_URL,
            "sample_row_count": len(rows),
            "auth_mode": "anon (publishable key, RLS-enforced)",
        }
    except Exception as e:
        return {
            "supabase_reachable": False,
            "url": SUPABASE_URL,
            "error": str(e),
        }


# =============================================================
# Geospatial utilities
# =============================================================
def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Distance in meters between two lat/lng points (Haversine)."""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def compute_features(
    lat: float,
    lng: float,
    competitors: List[Dict[str, Any]],
    pois: List[Dict[str, Any]],
    malls: List[Dict[str, Any]],
    kelurahan: Optional[Dict[str, Any]] = None,
    is_in_mall: bool = False,
) -> Dict[str, float]:
    """Compute the 10 canonical features for a candidate site."""
    comp_1km = sum(1 for c in competitors if haversine_m(lat, lng, c["lat"], c["lng"]) <= 1000)
    comp_3km = sum(1 for c in competitors if haversine_m(lat, lng, c["lat"], c["lng"]) <= 3000)
    poi_1km = sum(1 for p in pois if haversine_m(lat, lng, p["lat"], p["lng"]) <= 1000)
    mall_d = min((haversine_m(lat, lng, m["lat"], m["lng"]) for m in malls), default=5000.0)

    k = kelurahan or {}
    return {
        "competitor_density_1km": float(comp_1km),
        "competitor_density_3km": float(comp_3km),
        "poi_density_1km": float(poi_1km),
        "mall_distance_m": float(mall_d),
        "income_index": float(k.get("income_index", 50) or 50),
        "population_density": float(k.get("density", 1000) or 1000),
        "tourist_index": float(k.get("tourist_index", 50) or 50),
        "transport_index": float(k.get("transport_index", 50) or 50),
        "is_coastal": 1.0 if k.get("is_coastal") else 0.0,
        "is_in_mall": 1.0 if is_in_mall else 0.0,
    }


# =============================================================
# Scoring models
# =============================================================
def fallback_score(features: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Transparent weighted heuristic — used when no trained model is available."""
    weights = {
        "competitor_density_1km": -0.18,
        "competitor_density_3km": -0.05,
        "poi_density_1km": 0.15,
        "mall_distance_m": -0.10,
        "income_index": 0.18,
        "population_density": 0.12,
        "tourist_index": 0.10,
        "transport_index": 0.07,
        "is_coastal": 0.03,
        "is_in_mall": 0.05,
    }

    norm = {}
    norm["competitor_density_1km"] = 1.0 - min(features.get("competitor_density_1km", 0) / 20.0, 1.0)
    norm["competitor_density_3km"] = 1.0 - min(features.get("competitor_density_3km", 0) / 80.0, 1.0)
    norm["poi_density_1km"] = min(features.get("poi_density_1km", 0) / 30.0, 1.0)
    norm["mall_distance_m"] = 1.0 - min(features.get("mall_distance_m", 5000) / 5000.0, 1.0)
    norm["income_index"] = (features.get("income_index", 50) or 50) / 100.0
    norm["population_density"] = min((features.get("population_density", 0) or 0) / 5000.0, 1.0)
    norm["tourist_index"] = (features.get("tourist_index", 50) or 50) / 100.0
    norm["transport_index"] = (features.get("transport_index", 50) or 50) / 100.0
    norm["is_coastal"] = 1.0 if features.get("is_coastal") else 0.0
    norm["is_in_mall"] = 1.0 if features.get("is_in_mall") else 0.0

    score = 0.0
    breakdown: Dict[str, float] = {}
    for k, w in weights.items():
        contribution = w * norm[k]
        score += contribution
        breakdown[k] = round(contribution, 4)

    score = max(0.0, min(1.0, (score + 0.2) / 1.05))
    return score, breakdown


def gbr_predict(model: Any, features: Dict[str, float]) -> Tuple[float, Optional[Dict[str, float]]]:
    """Run prediction using a loaded scikit-learn GBR model."""
    X = np.array([[features.get(k, 0.0) or 0.0 for k in FEATURE_NAMES]])
    raw = float(model.predict(X)[0])
    score = max(0.0, min(1.0, raw))

    breakdown: Optional[Dict[str, float]] = None
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        norm = []
        for k in FEATURE_NAMES:
            v = features.get(k, 0.0) or 0.0
            if k == "competitor_density_1km":
                norm.append(1.0 - min(v / 20.0, 1.0))
            elif k == "competitor_density_3km":
                norm.append(1.0 - min(v / 80.0, 1.0))
            elif k == "mall_distance_m":
                norm.append(1.0 - min(v / 5000.0, 1.0))
            elif k in ("is_coastal", "is_in_mall"):
                norm.append(1.0 if v else 0.0)
            elif k == "population_density":
                norm.append(min(v / 5000.0, 1.0))
            elif k in ("income_index", "tourist_index", "transport_index"):
                norm.append((v or 50) / 100.0)
            else:
                norm.append(min(v / 30.0, 1.0))
        breakdown = {k: round(float(imps[i]) * norm[i], 4) for i, k in enumerate(FEATURE_NAMES)}

    return score, breakdown


# =============================================================
# In-browser model training
# scikit-learn is pre-loaded via the <script type="py" config> tag
# in index.html (packages = ["numpy", "scikit-learn"]). No micropip needed.
# =============================================================
async def train_gbr_model(n_samples: int = 500, n_estimators: int = 80, max_depth: int = 3) -> Dict[str, Any]:
    """Train a Gradient Boosting Regressor in-browser on synthetic data."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        return {
            "error": f"scikit-learn not loaded yet: {str(e)[:200]}",
            "fallback": "Heuristic model still available for Predict and Blank Spots.",
            "hint": "Refresh the page — scikit-learn loads automatically via Pyodide packages config.",
        }

    rng = np.random.default_rng(42)
    X = rng.uniform(low=[0, 0, 0, 0, 30, 500, 20, 20, 0, 0],
                    high=[20, 80, 30, 5000, 90, 5000, 95, 95, 1, 1],
                    size=(n_samples, 10))
    y = (
        0.45
        - 0.012 * X[:, 0] - 0.002 * X[:, 1] + 0.010 * X[:, 2] - 0.00006 * X[:, 3]
        + 0.004 * X[:, 4] + 0.00004 * X[:, 5] + 0.003 * X[:, 6] + 0.002 * X[:, 7]
        + 0.040 * X[:, 8] + 0.050 * X[:, 9]
    )
    y = np.clip(y + rng.normal(0, 0.08, n_samples), 0.02, 0.98)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=0.1, subsample=0.8, random_state=42,
    )
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    rmse = float(np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)))

    global _TRAINED_MODEL, _MODEL_META
    _TRAINED_MODEL = model
    _MODEL_META = {
        "name": "gbr_site_scoring",
        "version": "1.0.0",
        "n_samples": n_samples,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "feature_importances": {
            FEATURE_NAMES[i]: round(float(model.feature_importances_[i]), 4)
            for i in range(len(FEATURE_NAMES))
        },
    }

    return {
        "status": "trained",
        **_MODEL_META,
        "note": "Model trained in-browser on synthetic data. Active for this session.",
    }


def get_active_model() -> Tuple[Any, Dict[str, Any]]:
    return _TRAINED_MODEL, _MODEL_META


# =============================================================
# PyScript-exposed handlers (called from JavaScript)
# =============================================================
async def health_check() -> str:
    sb = await sb_ping()
    model, meta = get_active_model()
    info = {
        "service": "LocInsight ML Engine (PyScript)",
        "version": "2.1.0",
        "runtime": "Pyodide (WebAssembly) — browser-side Python",
        "sdk": "HF Spaces static SDK (free tier compatible)",
        "supabase": sb,
        "model_loaded": model is not None,
        "model_name": meta.get("name", "fallback_heuristic"),
        "model_version": meta.get("version", "0.1.0"),
    }
    return json.dumps(info, indent=2, default=str)


async def predict_site(lat: float, lng: float, is_in_mall: bool, kelurahan_id: str) -> str:
    try:
        competitors = await sb_select("competitor_stores", columns="name,lat,lng,brand_name", limit=2000)
        pois = await sb_select("pois", columns="name,lat,lng,type", limit=2000)
        malls = await sb_select("malls", columns="name,lat,lng", limit=500)
    except Exception as e:
        return json.dumps({"error": f"Supabase fetch failed: {e}"}, indent=2)

    kelurahan = None
    if kelurahan_id.strip():
        try:
            kl = await sb_select("kelurahan", columns="*", limit=1)
            kelurahan = kl[0] if kl else None
        except Exception:
            pass

    features = compute_features(lat, lng, competitors, pois, malls, kelurahan, is_in_mall)

    model, meta = get_active_model()
    if model is not None:
        score, breakdown = gbr_predict(model, features)
        model_name = meta.get("name", "gbr_site_scoring")
        model_version = meta.get("version", "1.0.0")
    else:
        score, breakdown = fallback_score(features)
        model_name = "fallback_heuristic"
        model_version = "0.1.0"

    score_pct = round(score * 100, 1)
    confidence = min(1.0, 0.4 + 0.1 * (len(competitors) > 50) + 0.1 * (len(pois) > 30) + 0.2 * (kelurahan is not None) + 0.2 * (model is not None))

    if score >= 0.75:
        rec = "STRONG GO — High-priority expansion candidate."
    elif score >= 0.55:
        rec = "GO — Promising candidate."
    elif score >= 0.35:
        rec = "HOLD — Marginal."
    else:
        rec = "NO-GO — Low probability."

    is_blank = score >= 0.65 and features["competitor_density_1km"] == 0

    return json.dumps({
        "score": round(score, 4),
        "score_pct": score_pct,
        "confidence": round(confidence, 3),
        "model": model_name,
        "model_version": model_version,
        "features_used": features,
        "feature_contributions": breakdown,
        "recommendation": rec,
        "is_blank_spot": is_blank,
    }, indent=2, default=str)


async def find_blank_spots(min_score: float, limit: int, radius_m: int) -> str:
    try:
        kelurahan = await sb_select(
            "kelurahan",
            columns="id,name,kab_name,city,lat,lng,tier,is_coastal,density,income_index,tourist_index,transport_index",
            limit=5000,
        )
        stores = await sb_select("stores", columns="id,lat,lng,brand_id", limit=2000)
        competitors = await sb_select("competitor_stores", columns="name,lat,lng", limit=5000)
        pois = await sb_select("pois", columns="name,lat,lng,type", limit=2000)
        malls = await sb_select("malls", columns="name,lat,lng", limit=500)
    except Exception as e:
        return json.dumps({"error": f"Supabase fetch failed: {e}"}, indent=2)

    model, _ = get_active_model()
    spots = []
    for k in kelurahan:
        try:
            lat, lng = float(k["lat"]), float(k["lng"])
        except (KeyError, TypeError, ValueError):
            continue

        nearest_store = min(
            (haversine_m(lat, lng, s["lat"], s["lng"]) for s in stores if s.get("lat") and s.get("lng")),
            default=99999,
        )
        if nearest_store < radius_m:
            continue

        features = compute_features(lat, lng, competitors, pois, malls, k, False)
        if model is not None:
            score, _ = gbr_predict(model, features)
        else:
            score, _ = fallback_score(features)

        if score < min_score:
            continue

        nearest_comp = min(
            (haversine_m(lat, lng, c["lat"], c["lng"]) for c in competitors if c.get("lat") and c.get("lng")),
            default=99999,
        )

        if score >= 0.75:
            rec = "STRONG GO"
        elif score >= 0.65:
            rec = "GO"
        else:
            rec = "HOLD"

        spots.append({
            "kelurahan_id": k.get("id", ""),
            "kelurahan_name": k.get("name", ""),
            "kab": k.get("kab_name", ""),
            "city": k.get("city", ""),
            "lat": lat, "lng": lng,
            "score": round(score, 4),
            "score_pct": round(score * 100, 1),
            "is_coastal": bool(k.get("is_coastal")),
            "tier": str(k.get("tier", "")),
            "nearest_maa_store_m": round(nearest_store, 0),
            "nearest_competitor_m": round(nearest_comp, 0),
            "recommendation": rec,
        })

    spots.sort(key=lambda s: -s["score"])
    spots = spots[:limit]

    return json.dumps({
        "count": len(spots),
        "spots": spots,
        "criteria": {"min_score": min_score, "radius_m": radius_m, "evaluated": len(kelurahan)},
    }, indent=2, default=str)


async def train_model(n_samples: int, n_estimators: int, max_depth: int) -> str:
    result = await train_gbr_model(n_samples=n_samples, n_estimators=n_estimators, max_depth=max_depth)
    return json.dumps(result, indent=2, default=str)


def model_info() -> str:
    _, meta = get_active_model()
    if not meta:
        return json.dumps({
            "status": "no_model_loaded",
            "message": "Using fallback heuristic. Click Train to train a real model.",
        }, indent=2)
    return json.dumps(meta, indent=2, default=str)


async def data_explorer(table: str, limit: int) -> str:
    allowed = ["brands", "stores", "malls", "pois", "kelurahan", "kabupaten", "kecamatan", "competitor_stores"]
    if table not in allowed:
        return json.dumps({"error": f"Unknown table. Allowed: {allowed}"}, indent=2)
    try:
        rows = await sb_select(table, columns="*", limit=limit)
        return json.dumps({"table": table, "count": len(rows), "rows": rows[:limit]}, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


# =============================================================
# Expose functions to JavaScript (PyScript 2025.x API)
# =============================================================
# In PyScript 2025.x, the recommended way to call Python from JS is to
# assign functions to the `window` object via `from pyscript import window`.
# JS can then call them directly: window.health_check()
try:
    from pyscript import window
    window.health_check = health_check
    window.predict_site = predict_site
    window.find_blank_spots = find_blank_spots
    window.train_model = train_model
    window.model_info = model_info
    window.data_explorer = data_explorer
    print("[LocInsight] Functions exposed to JavaScript window object")
except Exception as e:
    print(f"[LocInsight] Failed to expose functions to window: {e}")
