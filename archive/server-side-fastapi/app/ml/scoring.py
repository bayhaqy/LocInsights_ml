"""
Site Selection Scoring — Gradient Boosting Regressor.

Algorithm: Gradient Boosting Regressor (sklearn) trained on historical
MAA store performance + competitor density + POI density + demographic
features. Outputs a "Store Success Probability Score" 0-100%.

If no trained model artifact exists, falls back to a transparent weighted
heuristic (documented in `fallback_score`).

Features used (10 features):
  1. competitor_density_1km   — count of competitor stores within 1km
  2. competitor_density_3km   — count of competitor stores within 3km
  3. poi_density_1km          — count of POIs within 1km
  4. mall_distance_m          — distance to nearest mall (meters)
  5. income_index             — 0-100 income index of kelurahan
  6. population_density       — people per km²
  7. tourist_index            — 0-100 tourist attractiveness
  8. transport_index          — 0-100 transport connectivity
  9. is_coastal               — 1 if coastal, 0 otherwise
 10. is_in_mall               — 1 if candidate is in a mall, 0 otherwise
"""
from __future__ import annotations

import math
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Feature names in canonical order (must match training)
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

# Default GBR hyperparameters (used when no artifact loaded)
DEFAULT_HYPERPARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 42,
}


def fallback_score(features: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Transparent weighted heuristic — used when no trained model is available.

    Returns (score_0_to_1, contribution_breakdown).
    """
    # Weights (sum to 1.0)
    weights = {
        "competitor_density_1km": -0.18,   # closer competitors = lower score
        "competitor_density_3km": -0.05,   # broader competition less impactful
        "poi_density_1km":        0.15,    # POIs attract footfall
        "mall_distance_m":       -0.10,    # closer to mall = better (negative coef)
        "income_index":           0.18,    # higher income = better
        "population_density":     0.12,    # denser population = more customers
        "tourist_index":          0.10,
        "transport_index":        0.07,
        "is_coastal":             0.03,    # coastal premium (tourism)
        "is_in_mall":             0.05,    # mall location is safer bet
    }

    # Normalize raw features to 0-1 range
    norm = {}
    norm["competitor_density_1km"] = 1.0 - min(features.get("competitor_density_1km", 0) / 20.0, 1.0)
    norm["competitor_density_3km"] = 1.0 - min(features.get("competitor_density_3km", 0) / 80.0, 1.0)
    norm["poi_density_1km"]        = min(features.get("poi_density_1km", 0) / 30.0, 1.0)
    norm["mall_distance_m"]        = 1.0 - min(features.get("mall_distance_m", 5000) / 5000.0, 1.0)
    norm["income_index"]           = (features.get("income_index", 50) or 50) / 100.0
    norm["population_density"]     = min((features.get("population_density", 0) or 0) / 5000.0, 1.0)
    norm["tourist_index"]          = (features.get("tourist_index", 50) or 50) / 100.0
    norm["transport_index"]        = (features.get("transport_index", 50) or 50) / 100.0
    norm["is_coastal"]             = 1.0 if features.get("is_coastal") else 0.0
    norm["is_in_mall"]             = 1.0 if features.get("is_in_mall") else 0.0

    score = 0.0
    breakdown: Dict[str, float] = {}
    for k, w in weights.items():
        contribution = w * norm[k]
        score += contribution
        breakdown[k] = round(contribution, 4)

    # Shift to 0-1 range (raw sum is roughly -0.2 to +0.85, so scale)
    score = max(0.0, min(1.0, (score + 0.2) / 1.05))
    return score, breakdown


def gbr_predict(model: Any, features: Dict[str, float]) -> Tuple[float, Optional[Dict[str, float]]]:
    """Run prediction using a loaded GBR model.

    Returns (score_0_to_1, feature_importance_breakdown_or_None).
    """
    X = np.array([[features.get(k, 0.0) or 0.0 for k in FEATURE_NAMES]])
    raw = float(model.predict(X)[0])
    score = max(0.0, min(1.0, raw))

    # Feature importance breakdown (if model exposes feature_importances_)
    breakdown: Optional[Dict[str, float]] = None
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        # Approximate per-feature contribution = importance × normalized feature value
        norm = []
        for k in FEATURE_NAMES:
            v = features.get(k, 0.0) or 0.0
            if k in ("competitor_density_1km", "competitor_density_3km", "mall_distance_m"):
                # Inverse-normalize these (high = bad)
                if k == "competitor_density_1km":
                    norm.append(1.0 - min(v / 20.0, 1.0))
                elif k == "competitor_density_3km":
                    norm.append(1.0 - min(v / 80.0, 1.0))
                else:
                    norm.append(1.0 - min(v / 5000.0, 1.0))
            elif k in ("is_coastal", "is_in_mall"):
                norm.append(1.0 if v else 0.0)
            elif k == "population_density":
                norm.append(min(v / 5000.0, 1.0))
            elif k in ("income_index", "tourist_index", "transport_index"):
                norm.append((v or 50) / 100.0)
            else:  # poi_density_1km
                norm.append(min(v / 30.0, 1.0))
        breakdown = {k: round(float(imps[i]) * norm[i], 4) for i, k in enumerate(FEATURE_NAMES)}

    return score, breakdown


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
    competitors: List[Dict[str, float]],
    pois: List[Dict[str, float]],
    malls: List[Dict[str, float]],
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


def save_model_artifact(model: Any, path: str, metadata: Dict[str, Any]) -> None:
    """Persist model + metadata as JSON (lightweight, no cloud storage needed)."""
    import pickle, base64
    blob = base64.b64encode(pickle.dumps(model)).decode("ascii")
    payload = {"metadata": metadata, "model_b64": blob}
    with open(path, "w") as f:
        json.dump(payload, f)


def load_model_artifact(path: str) -> Tuple[Any, Dict[str, Any]]:
    """Load model + metadata from JSON artifact."""
    import pickle, base64
    with open(path, "r") as f:
        payload = json.load(f)
    model = pickle.loads(base64.b64decode(payload["model_b64"]))
    return model, payload["metadata"]
