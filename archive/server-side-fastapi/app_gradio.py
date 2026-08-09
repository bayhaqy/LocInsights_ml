"""
LocInsight ML Engine — Gradio entry point for Hugging Face Spaces (Free Tier)
============================================================================

HF Spaces free tier does NOT support Docker. This module wraps the existing
FastAPI application inside a Gradio Blocks app using `gr.mount_gradio_app`,
so all original endpoints (/health, /predict, /scrape_bali, /train, /blank_spots)
continue to work, AND we get a Gradio UI for interactive testing.

Deploy steps:
  1. HF Space → Settings → SDK = "Gradio" (NOT Docker)
  2. Push this file as `app.py` in the Space root
  3. Set Space secrets: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,
     LOCINSIGHT_API_TOKEN, CORS_ALLOWED_ORIGINS
  4. Space auto-launches on port 7860

Vercel backend calls:
  - GET  {SPACE_URL}/health                → liveness probe (cron)
  - POST {SPACE_URL}/predict               → site success score
  - POST {SPACE_URL}/scrape_bali           → trigger scrape job
  - GET  {SPACE_URL}/scrape_bali/{job_id}  → poll job status
  - POST {SPACE_URL}/train                 → retrain GBR model
  - GET  {SPACE_URL}/model/info            → model metadata
  - GET  {SPACE_URL}/blank_spots           → recommended new locations
"""
from __future__ import annotations

import os
import logging
import json
from typing import Any, Dict, Optional

import gradio as gr

# Import the existing FastAPI app (all routes, auth, CORS, lifespan are preserved)
from app.main import app as fastapi_app

log = logging.getLogger("locinsight_gradio")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


# =============================================================
# Gradio UI — interactive test panel (also serves as a demo)
# =============================================================
def gradio_predict(lat: float, lng: float, is_in_mall: bool, kelurahan_id: str) -> str:
    """Call the FastAPI /predict endpoint internally and return a formatted result."""
    import asyncio
    from app.routes.predict import PredictRequest, predict
    from app.config import get_settings

    class _FakeRequest:
        """Minimal stand-in so predict() can read app.state.model."""
        class _App:
            class _State:
                model = getattr(fastapi_app.state, "model", None)
                model_metadata = getattr(fastapi_app.state, "model_metadata", {}) or {}
            state = _State()
        app = _App()

    async def _run():
        req = PredictRequest(
            lat=float(lat),
            lng=float(lng),
            is_in_mall=bool(is_in_mall),
            kelurahan_id=kelurahan_id.strip() or None,
        )
        result = await predict(req, _FakeRequest())
        return result.model_dump_json(indent=2)

    try:
        return asyncio.run(_run())
    except Exception as e:
        return f"Error: {e}"


def gradio_health() -> str:
    """Quick health check — shows config + model status."""
    from app.config import get_settings
    s = get_settings()
    model = getattr(fastapi_app.state, "model", None)
    meta = getattr(fastapi_app.state, "model_metadata", {}) or {}
    info = {
        "service": "LocInsight ML Engine",
        "version": "1.0.0",
        "status": "online",
        "supabase_configured": bool(s.supabase_url and s.supabase_service_role_key),
        "auth_enabled": bool(s.locinsight_api_token),
        "model_loaded": model is not None,
        "model_name": meta.get("name", "fallback_heuristic"),
        "model_version": meta.get("version", "0.1.0"),
        "huggingface_space": os.getenv("SPACE_HOST", "local"),
    }
    return json.dumps(info, indent=2)


def gradio_blank_spots(limit: int) -> str:
    """Fetch blank spot recommendations."""
    import asyncio
    from app.routes.blank_spots import blank_spots
    from app.ml.scoring import FEATURE_NAMES

    class _FakeRequest:
        class _App:
            class _State:
                model = getattr(fastapi_app.state, "model", None)
                model_metadata = getattr(fastapi_app.state, "model_metadata", {}) or {}
            state = _State()
        app = _App()

    async def _run():
        result = await blank_spots(_FakeRequest(), limit=int(limit))
        if hasattr(result, "body"):
            return result.body.decode("utf-8")
        return json.dumps(result, indent=2, default=str)

    try:
        return asyncio.run(_run())
    except Exception as e:
        return f"Error: {e}"


# =============================================================
# Build the Gradio Blocks UI
# =============================================================
with gr.Blocks(
    title="LocInsight ML Engine",
) as demo:
    gr.HTML("""
    <div class="header">
        <h1>LocInsight ML Engine</h1>
        <p>Site selection scoring + Bali scraping worker for MAP Active Adiperkasa</p>
        <p style="font-size: 12px; opacity: 0.7;">Hugging Face Space (Gradio SDK) · FastAPI mounted at /</p>
    </div>
    """)

    with gr.Tab("Health Check"):
        gr.Markdown("### Service Status\nQuick liveness + configuration check. The Vercel cron hits `/health` every 15 min to prevent auto-sleep.")
        health_btn = gr.Button("Check Health", variant="primary")
        health_out = gr.Code(label="Response", language="json")
        health_btn.click(fn=gradio_health, outputs=health_out)

    with gr.Tab("Predict Site Score"):
        gr.Markdown("### Store Success Probability Score\nInput a candidate site coordinate to get a 0-100% score based on competitor density, POI density, demographics, and mall proximity.")
        with gr.Row():
            p_lat = gr.Number(label="Latitude", value=-8.6705, info="Bali centroid default")
            p_lng = gr.Number(label="Longitude", value=115.2126)
        with gr.Row():
            p_mall = gr.Checkbox(label="Is in mall?", value=False)
            p_kel = gr.Textbox(label="Kelurahan ID (optional)", value="", placeholder="e.g., 5101010001")
        p_btn = gr.Button("Predict Score", variant="primary")
        p_out = gr.Code(label="Prediction Result", language="json")
        p_btn.click(fn=gradio_predict, inputs=[p_lat, p_lng, p_mall, p_kel], outputs=p_out)

    with gr.Tab("Blank Spots"):
        gr.Markdown("### Recommended Blank Spot Areas\nHigh-score candidate sites with no existing MAA store nearby. Highlighted in green on the LocInsight map.")
        bs_limit = gr.Slider(label="Limit", minimum=5, maximum=100, value=20, step=5)
        bs_btn = gr.Button("Find Blank Spots", variant="primary")
        bs_out = gr.Code(label="Recommendations", language="json")
        bs_btn.click(fn=gradio_blank_spots, inputs=[bs_limit], outputs=bs_out)

    with gr.Tab("API Reference"):
        gr.Markdown("""
### REST API Endpoints (FastAPI mounted)

All endpoints below are accessible via HTTP. Protected routes require the `X-LocInsight-Token` header.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET    | `/health` | No | Liveness probe (used by Vercel cron) |
| GET    | `/` | No | Service info |
| GET    | `/docs` | No | OpenAPI Swagger UI |
| POST   | `/predict` | Yes | Site success probability score |
| POST   | `/scrape_bali` | Yes | Trigger Bali scraping job (async) |
| GET    | `/scrape_bali/{job_id}` | Yes | Poll scrape job status |
| POST   | `/train` | Yes | Retrain GBR model from Supabase data |
| GET    | `/model/info` | Yes | Active model metadata |
| GET    | `/blank_spots` | Yes | Recommended new-location candidates |

### Example: Call /predict from Vercel backend

```typescript
const res = await fetch(`${process.env.ML_API_URL}/predict`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-LocInsight-Token': process.env.ML_API_TOKEN,
  },
  body: JSON.stringify({ lat: -8.67, lng: 115.21, is_in_mall: false }),
})
const { score_pct, recommendation, is_blank_spot } = await res.json()
```
        """)


# =============================================================
# Mount Gradio UI onto the existing FastAPI app
# =============================================================
# This preserves all FastAPI routes (/health, /predict, /scrape_bali, etc.)
# AND adds the Gradio UI at the root path.
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")


# =============================================================
# Entry point — HF Spaces Gradio SDK calls this
# =============================================================
if __name__ == "__main__":
    # HF Spaces Gradio SDK automatically manages the server.
    # We just need to launch the combined app on port 7860.
    port = int(os.getenv("PORT", "7860"))
    log.info(f"Starting LocInsight ML Engine on port {port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        prevent_thread_lock=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
        .header { background: linear-gradient(135deg, #7A0A1A 0%, #C8102E 100%); color: white; padding: 20px; border-radius: 8px; }
        .header h1 { margin: 0; font-size: 24px; }
        .header p { margin: 5px 0 0 0; opacity: 0.9; }
        """,
    )
