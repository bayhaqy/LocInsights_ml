"""POST /train — Retrain GBR model from latest Supabase data."""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel, Field

from app.ml.scoring import FEATURE_NAMES, DEFAULT_HYPERPARAMS, save_model_artifact
from app.supabase_client import get_supabase

router = APIRouter()
log = logging.getLogger(__name__)


class TrainRequest(BaseModel):
    """Trigger a model retraining run."""
    algorithm: str = Field("gradient_boosting", description="Algorithm: 'gradient_boosting' | 'random_forest'")
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    random_state: int = Field(42)
    save_artifact: bool = Field(True, description="If true, persist trained model to artifacts/")


class TrainResponse(BaseModel):
    job_id: str
    status: str
    started_at: float
    message: str


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str
    dataset_size: int
    metrics: Dict[str, float]
    feature_importance: List[Dict[str, float]]
    train_duration_ms: int
    error: Optional[str] = None


# In-memory job store (single-worker; for multi-worker use Redis)
_jobs: Dict[str, Dict[str, Any]] = {}


@router.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    """Trigger async model retraining.

    Pulls latest stores + competitors + POIs + kelurahan from Supabase,
    constructs feature vectors, trains a GBR (or RF), saves artifact.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "pending",
        "started_at": time.time(),
        "dataset_size": 0,
        "metrics": {},
        "feature_importance": [],
        "train_duration_ms": 0,
        "error": None,
        "request": req.dict(),
    }
    background_tasks.add_task(_run_train, job_id, req)
    return TrainResponse(
        job_id=job_id,
        status="pending",
        started_at=_jobs[job_id]["started_at"],
        message=f"Training job queued. Poll GET /train/{job_id} for status.",
    )


@router.get("/train/{job_id}", response_model=TrainStatusResponse)
async def get_train(job_id: str):
    if job_id not in _jobs:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    j = _jobs[job_id]
    return TrainStatusResponse(
        job_id=job_id,
        status=j["status"],
        dataset_size=j["dataset_size"],
        metrics=j["metrics"],
        feature_importance=j["feature_importance"],
        train_duration_ms=j["train_duration_ms"],
        error=j["error"],
    )


async def _run_train(job_id: str, req: TrainRequest):
    """Background training task."""
    j = _jobs[job_id]
    j["status"] = "running"
    start = time.time()
    try:
        # 1. Fetch training data from Supabase
        sb = get_supabase()
        stores = await sb.select("stores", columns="id,brand_id,lat,lng,kec,kab,confirmed", limit=2000)
        competitors = await sb.select("competitor_stores", columns="name,lat,lng", limit=5000)
        pois = await sb.select("pois", columns="name,lat,lng,type", limit=2000)
        malls = await sb.select("malls", columns="name,lat,lng", limit=500)
        kelurahan = await sb.select("kelurahan",
            columns="id,lat,lng,density,income_index,tourist_index,transport_index,is_coastal", limit=5000)

        if not stores or not kelurahan:
            j["status"] = "failed"
            j["error"] = "Insufficient training data (need stores + kelurahan)"
            return

        # 2. Build feature matrix
        from app.ml.scoring import compute_features
        import numpy as np

        X_list = []
        y_list = []
        for s in stores:
            # Find nearest kelurahan
            nearest_k = None
            min_d = float("inf")
            for k in kelurahan:
                d = (s["lat"] - k["lat"]) ** 2 + (s["lng"] - k["lng"]) ** 2
                if d < min_d:
                    min_d = d
                    nearest_k = k

            features = compute_features(
                s["lat"], s["lng"], competitors, pois, malls, nearest_k, False
            )
            X_list.append([features[k] for k in FEATURE_NAMES])
            # Synthetic target: confirmed stores get 0.8, unconfirmed get 0.5
            # (In production, replace with actual revenue/footfall data)
            y_list.append(0.8 if s.get("confirmed") else 0.5)

        X = np.array(X_list)
        y = np.array(y_list)
        j["dataset_size"] = len(X)

        # 3. Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=req.test_size, random_state=req.random_state
        )

        # 4. Train model
        if req.algorithm == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=100, max_depth=6, random_state=req.random_state
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(**DEFAULT_HYPERPARAMS)

        model.fit(X_train, y_train)

        # 5. Evaluate
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        y_pred = model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
            "r2": float(r2_score(y_test, y_pred)),
            "mape": float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 0.01))) * 100),
        }
        j["metrics"] = metrics

        # Feature importance
        if hasattr(model, "feature_importances_"):
            imps = model.feature_importances_
            j["feature_importance"] = [
                {"feature": FEATURE_NAMES[i], "importance": float(imps[i])}
                for i in range(len(FEATURE_NAMES))
            ]
            j["feature_importance"].sort(key=lambda x: -x["importance"])

        # 6. Save artifact
        if req.save_artifact:
            artifact_path = "artifacts/gbr_site_scoring.json"
            metadata = {
                "id": f"model-{job_id[:8]}",
                "name": "gbr_site_scoring",
                "version": f"1.{int(time.time()) % 1000}.0",
                "algorithm": req.algorithm,
                "trained_at": time.time(),
                "features": FEATURE_NAMES,
                "hyperparameters": DEFAULT_HYPERPARAMS,
                "metrics": metrics,
            }
            save_model_artifact(model, artifact_path, metadata)
            log.info(f"Model artifact saved to {artifact_path}")

            # Persist to Supabase
            try:
                await sb.insert("ml_models", [{
                    "id": metadata["id"],
                    "name": metadata["name"],
                    "version": metadata["version"],
                    "type": "site_scoring",
                    "algorithm": req.algorithm,
                    "features": FEATURE_NAMES,
                    "hyperparameters": DEFAULT_HYPERPARAMS,
                    "metrics": metrics,
                    "status": "active",
                }])
            except Exception as e:
                log.warning(f"Failed to persist ml_model to Supabase: {e}")

        j["train_duration_ms"] = int((time.time() - start) * 1000)
        j["status"] = "completed"
        log.info(f"Training job {job_id} completed: {metrics}")

    except Exception as e:
        log.exception("Training job failed")
        j["status"] = "failed"
        j["error"] = str(e)
        j["train_duration_ms"] = int((time.time() - start) * 1000)
