"""Centralized configuration — all secrets from environment variables."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List


class Settings:
    """Application settings loaded from env vars (HF Space secrets)."""

    # Supabase
    supabase_url: str
    supabase_service_role_key: str

    # API security
    locinsight_api_token: str

    # CORS
    cors_allowed_origins: List[str]

    # ML
    model_artifact_path: str
    model_name: str
    model_version: str

    # Scraper
    nominatim_user_agent: str
    overpass_endpoint: str
    scrape_request_timeout: int

    def __init__(self) -> None:
        self.supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        self.locinsight_api_token = os.getenv("LOCINSIGHT_API_TOKEN", "")

        cors_raw = os.getenv("CORS_ALLOWED_ORIGINS", "*")
        self.cors_allowed_origins = (
            [o.strip() for o in cors_raw.split(",") if o.strip()]
            if cors_raw != "*"
            else ["*"]
        )

        self.model_artifact_path = os.getenv("MODEL_ARTIFACT_PATH", "artifacts/gbr_site_scoring.json")
        self.model_name = os.getenv("MODEL_NAME", "gbr_site_scoring")
        self.model_version = os.getenv("MODEL_VERSION", "1.0.0")

        self.nominatim_user_agent = os.getenv("NOMINATIM_USER_AGENT", "LocInsight/1.0 (bayhaqy@mapactive.id)")
        self.overpass_endpoint = os.getenv("OVERPASS_ENDPOINT", "https://overpass-api.de/api/interpreter")
        self.scrape_request_timeout = int(os.getenv("SCRAPE_REQUEST_TIMEOUT", "30"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
