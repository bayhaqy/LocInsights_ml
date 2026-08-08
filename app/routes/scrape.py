"""Bali scraping endpoints — async background tasks."""
from __future__ import annotations

import logging
import uuid
import time
import asyncio
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from app.supabase_client import get_supabase

router = APIRouter()
log = logging.getLogger(__name__)

# In-memory job status (HF Space single-worker; for multi-worker use Redis)
_jobs: Dict[str, Dict[str, Any]] = {}


class ScrapeRequest(BaseModel):
    """Trigger a Bali scraping job."""
    target: str = Field("all", description="What to scrape: 'all' | 'map_stores' | 'competitors' | 'malls' | 'mall_tenants'")
    brand_filter: Optional[List[str]] = Field(None, description="Restrict MAP brands (e.g., ['nike','adidas'])")
    kab_filter: Optional[List[str]] = Field(None, description="Restrict kabupaten (e.g., ['5175'])")
    save_to_staging: bool = Field(True, description="If true, results go to staging_* tables for review")


class ScrapeJobResponse(BaseModel):
    job_id: str
    status: str
    started_at: float
    target: str
    message: str


class ScrapeStatusResponse(BaseModel):
    job_id: str
    status: str  # pending | running | success | failed | partial
    found_count: int
    saved_count: int
    elapsed_seconds: float
    error: Optional[str] = None
    sample_results: List[Dict[str, Any]] = []


@router.post("/scrape_bali", response_model=ScrapeJobResponse)
async def start_scrape(req: ScrapeRequest, background_tasks: BackgroundTasks, request: Request):
    """Start an async Bali scraping job.

    Returns immediately with a job_id. Poll /scrape_bali/{job_id} for status.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "pending",
        "started_at": time.time(),
        "found_count": 0,
        "saved_count": 0,
        "target": req.target,
        "brand_filter": req.brand_filter,
        "kab_filter": req.kab_filter,
        "save_to_staging": req.save_to_staging,
        "error": None,
        "sample_results": [],
    }
    background_tasks.add_task(_run_scrape, job_id, req)
    return ScrapeJobResponse(
        job_id=job_id,
        status="pending",
        started_at=_jobs[job_id]["started_at"],
        target=req.target,
        message=f"Scrape job queued. Poll GET /scrape_bali/{job_id} for updates.",
    )


@router.get("/scrape_bali/{job_id}", response_model=ScrapeStatusResponse)
async def get_scrape(job_id: str):
    if job_id not in _jobs:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    j = _jobs[job_id]
    return ScrapeStatusResponse(
        job_id=job_id,
        status=j["status"],
        found_count=j["found_count"],
        saved_count=j["saved_count"],
        elapsed_seconds=round(time.time() - j["started_at"], 1),
        error=j["error"],
        sample_results=j["sample_results"][:5],
    )


# =============================================================
# Scraping logic — uses Overpass API (OpenStreetMap) for POIs/stores
# and HTTP scraping for MAP brand lists.
# =============================================================
BALI_BBOX = (-8.83, 114.44, -8.06, 115.71)  # south, west, north, east

MAP_BRAND_OSM_TAGS = {
    "nike":        {"shop": "shoes", "brand": "Nike"},
    "adidas":      {"shop": "shoes", "brand": "Adidas"},
    "skechers":    {"shop": "shoes", "brand": "Skechers"},
    "starbucks":   {"amenity": "cafe", "brand": "Starbucks"},
    "zara":        {"shop": "clothes", "brand": "Zara"},
    "gap":         {"shop": "clothes", "brand": "Gap"},
}

COMPETITOR_BRAND_OSM_TAGS = {
    "Indomaret":     {"shop": "convenience", "brand": "Indomaret"},
    "Alfamart":      {"shop": "convenience", "brand": "Alfamart"},
    "Alfamidi":      {"shop": "convenience", "brand": "Alfamidi"},
    "McDonald's":    {"amenity": "fast_food", "brand": "McDonald's"},
    "KFC":           {"amenity": "fast_food", "brand": "KFC"},
    "Aw":            {"amenity": "fast_food", "brand": "AW"},
    "Havaianas":     {"shop": "clothes", "brand": "Havaianas"},
    "Informa":       {"shop": "furniture", "brand": "Informa"},
    "ACE Hardware":  {"shop": "doityourself", "brand": "Ace Hardware"},
    "Trans Mart":    {"shop": "mall", "brand": "Trans Mart"},
}


def _build_overpass_query(tags_dict: Dict[str, Dict[str, str]], bbox: tuple) -> str:
    """Build an Overpass QL query for a set of brand tags within a bbox."""
    s, w, n, e = bbox
    parts = []
    for brand, tags in tags_dict.items():
        for k, v in tags.items():
            parts.append(f'node["{k}"="{v}"]({s},{w},{n},{e});')
            parts.append(f'way["{k}"="{v}"]({s},{w},{n},{e});')
    return "[" + ",".join(parts) + "];out center 200;"


async def _query_overpass(query: str) -> List[Dict[str, Any]]:
    """Run an Overpass API query, return parsed elements."""
    settings = get_settings()
    async with httpx.AsyncClient(timeout=settings.scrape_request_timeout) as client:
        try:
            r = await client.post(settings.overpass_endpoint, data={"data": query})
            r.raise_for_status()
            data = r.json()
            results = []
            for el in data.get("elements", []):
                lat = el.get("lat") or el.get("center", {}).get("lat")
                lng = el.get("lon") or el.get("center", {}).get("lon")
                if lat is None or lng is None:
                    continue
                tags = el.get("tags", {})
                results.append({
                    "name": tags.get("name", ""),
                    "brand": tags.get("brand", ""),
                    "lat": float(lat),
                    "lng": float(lng),
                    "address": tags.get("addr:full", ""),
                    "source": "osm",
                    "raw_tags": tags,
                })
            return results
        except Exception as e:
            log.warning(f"Overpass query failed: {e}")
            return []


def _classify_competitor(name: str) -> str:
    n = name.lower()
    if "indomaret" in n: return "convenience_store"
    if "alfamart" in n or "alfamidi" in n: return "convenience_store"
    if "mcdonald" in n or "kfc" in n or "aw " in n: return "fast_food"
    if "starbucks" in n or "coffee" in n: return "coffee"
    if "havaianas" in n: return "fashion"
    if "informa" in n or "ace" in n: return "department_store"
    return "other"


async def _run_scrape(job_id: str, req: ScrapeRequest):
    """Background task: actually run the scraping."""
    j = _jobs[job_id]
    j["status"] = "running"
    try:
        all_results: List[Dict[str, Any]] = []

        if req.target in ("all", "map_stores"):
            brands = req.brand_filter or list(MAP_BRAND_OSM_TAGS.keys())
            tags = {b: MAP_BRAND_OSM_TAGS[b] for b in brands if b in MAP_BRAND_OSM_TAGS}
            if tags:
                q = _build_overpass_query(tags, BALI_BBOX)
                results = await _query_overpass(q)
                for r in results:
                    r["target_table"] = "staging_stores"
                    r["parent"] = "MAA"
                all_results.extend(results)

        if req.target in ("all", "competitors"):
            tags = COMPETITOR_BRAND_OSM_TAGS
            q = _build_overpass_query(tags, BALI_BBOX)
            results = await _query_overpass(q)
            for r in results:
                r["target_table"] = "staging_competitors"
                r["brand_category"] = _classify_competitor(r.get("name", "") + " " + r.get("brand", ""))
            all_results.extend(results)

        if req.target in ("all", "malls"):
            # OSM mall query for Bali
            q = '["shop"="mall"](-8.83,114.44,-8.06,115.71);out center 200;'
            results = await _query_overpass(q)
            for r in results:
                r["target_table"] = "staging_malls"
            all_results.extend(results)

        j["found_count"] = len(all_results)
        j["sample_results"] = all_results[:10]

        if req.save_to_staging and all_results:
            sb = get_supabase()
            batch_id = job_id
            now_iso = None  # Supabase will set default

            # Split by target table
            stores_rows = [{
                "batch_id": batch_id, "source": "osm", "source_url": "overpass-api.de",
                "brand_name": r.get("brand") or r.get("name", "Unknown"),
                "name": r.get("name", f"Store @ {r['lat']:.4f},{r['lng']:.4f}"),
                "lat": r["lat"], "lng": r["lng"],
                "address": r.get("address", ""),
                "is_in_mall": False,
                "review_status": "pending",
            } for r in all_results if r.get("target_table") == "staging_stores"]

            comp_rows = [{
                "batch_id": batch_id, "source": "osm", "source_url": "overpass-api.de",
                "brand_name": r.get("brand") or r.get("name", "Unknown"),
                "brand_category": r.get("brand_category", "other"),
                "name": r.get("name", f"Outlet @ {r['lat']:.4f},{r['lng']:.4f}"),
                "lat": r["lat"], "lng": r["lng"],
                "address": r.get("address", ""),
                "is_in_mall": False,
                "review_status": "pending",
            } for r in all_results if r.get("target_table") == "staging_competitors"]

            mall_rows = [{
                "batch_id": batch_id, "source": "osm", "source_url": "overpass-api.de",
                "name": r.get("name", f"Mall @ {r['lat']:.4f},{r['lng']:.4f}"),
                "lat": r["lat"], "lng": r["lng"],
                "address": r.get("address", ""),
                "review_status": "pending",
            } for r in all_results if r.get("target_table") == "staging_malls"]

            saved = 0
            for table, rows in [
                ("staging_stores", stores_rows),
                ("staging_competitors", comp_rows),
                ("staging_malls", mall_rows),
            ]:
                if rows:
                    try:
                        await sb.insert(table, rows)
                        saved += len(rows)
                    except Exception as e:
                        log.warning(f"Insert into {table} failed: {e}")
            j["saved_count"] = saved

        # Insert scraper_run log
        try:
            sb = get_supabase()
            await sb.insert("scraper_runs", [{
                "id": batch_id if req.save_to_staging else job_id,
                "query": f"target={req.target}",
                "source": "osm",
                "status": "success" if all_results else "partial",
                "found_count": len(all_results),
                "saved_count": j["saved_count"],
                "result_json": all_results[:50],
                "finished_at": "now()",
            }])
        except Exception as e:
            log.warning(f"Failed to log scraper_run: {e}")

        j["status"] = "success" if all_results else "partial"
        if not all_results:
            j["error"] = "No results found (Overpass rate limit or empty bbox?)"

    except Exception as e:
        log.exception("Scrape job failed")
        j["status"] = "failed"
        j["error"] = str(e)
