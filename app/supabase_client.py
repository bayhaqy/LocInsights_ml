"""Supabase client wrapper — used to read features and write predictions back."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)


class SupabaseClient:
    """Thin wrapper around Supabase REST (PostgREST).

    Uses service_role key — bypasses RLS. ONLY use this server-side.
    """

    def __init__(self) -> None:
        s = get_settings()
        self.base_url = s.supabase_url
        self.headers = {
            "apikey": s.supabase_service_role_key,
            "Authorization": f"Bearer {s.supabase_service_role_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        self.timeout = 30.0

    async def select(
        self,
        table: str,
        columns: str = "*",
        limit: int = 1000,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """SELECT rows from a table."""
        params: List[str] = [f"select={columns}", f"limit={limit}"]
        if filters:
            for k, v in filters.items():
                params.append(f"{k}=eq.{v}")
        url = f"{self.base_url}/rest/v1/{table}?" + "&".join(params)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(url, headers=self.headers)
            r.raise_for_status()
            return r.json()

    async def insert(self, table: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """INSERT rows into a table."""
        url = f"{self.base_url}/rest/v1/{table}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, headers=self.headers, json=rows)
            r.raise_for_status()
            return r.json() if r.content else []

    async def rpc(self, fn: str, params: Dict[str, Any]) -> Any:
        """Call a Postgres function via RPC."""
        url = f"{self.base_url}/rest/v1/rpc/{fn}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, headers=self.headers, json=params)
            r.raise_for_status()
            return r.json()

    async def ping(self) -> bool:
        """Lightweight connectivity check (SELECT 1)."""
        try:
            url = f"{self.base_url}/rest/v1/rpc/merge_staging_store"
            # Use a deliberate 404 to verify auth + connectivity (function requires args)
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(url, headers=self.headers, json={})
                # 400 = auth OK but missing args = connectivity OK
                return r.status_code in (400, 404)
        except Exception as e:
            log.warning(f"Supabase ping failed: {e}")
            return False


_client: Optional[SupabaseClient] = None


def get_supabase() -> SupabaseClient:
    global _client
    if _client is None:
        _client = SupabaseClient()
    return _client
