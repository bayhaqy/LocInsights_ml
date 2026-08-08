# LocInsight ML Engine

FastAPI service that powers **site selection scoring** and **Bali scraping** for the [LocInsight](https://github.com/bayhaqy/LocInsights) location intelligence system.

Hosted on **Hugging Face Spaces** (Docker SDK) and called by the Vercel-hosted Next.js frontend via custom Bearer token auth.

## Architecture

```
┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│  Vercel (Next)  │──────▶│  HF Spaces (ML)  │──────▶│  Supabase (DB)  │
│  Frontend+API   │ HTTPS │  FastAPI+sklearn │ HTTPS │  PostgreSQL+GIS │
└─────────────────┘       └──────────────────┘       └─────────────────┘
       │                          │                         ▲
       │  Vercel Cron (15 min)    │  /health ping           │
       └──────────────────────────┼─────────────────────────┘
                                  ▼
                          Anti-sleep pipeline
```

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET  | `/health` | None | Liveness probe (Vercel cron hits this) |
| GET  | `/` | None | Service info |
| GET  | `/docs` | None | OpenAPI Swagger UI |
| POST | `/predict` | Bearer | Site success probability score (0-100%) |
| POST | `/scrape_bali` | Bearer | Start async Bali scraping job |
| GET  | `/scrape_bali/{job_id}` | Bearer | Poll scrape job status |
| POST | `/train` | Bearer | Retrain GBR model from latest Supabase data |
| GET  | `/train/{job_id}` | Bearer | Poll training job status |
| GET  | `/blank_spots` | Bearer | Recommended new-location candidates |
| GET  | `/model/info` | Bearer | Currently-loaded model metadata |

## Security

- All protected endpoints require `X-LocInsight-Token` header matching `LOCINSIGHT_API_TOKEN` env var
- Only the Vercel backend (server-side) holds this token — never exposed to the browser
- `/health` and `/` are public (used for liveness probes)
- CORS restricted to `CORS_ALLOWED_ORIGINS` (set to Vercel app URL in production)

## ML Algorithm

**Gradient Boosting Regressor** (scikit-learn) trained on:
- Competitor density (1km + 3km radius)
- POI density (1km radius)
- Distance to nearest mall
- Kelurahan-level income / population density / tourist / transport indices
- Coastal flag, mall flag

**Output**: Store Success Probability Score (0-100%)

If no trained model artifact exists, falls back to a transparent weighted heuristic documented in `app/ml/scoring.py::fallback_score`.

## Local Development

```bash
# 1. Clone
git clone https://github.com/bayhaqy/LocInsights_ml.git
cd LocInsights_ml

# 2. Install deps
pip install -r requirements.txt

# 3. Configure env
cp .env.example .env
# Edit .env to add SUPABASE_SERVICE_ROLE_KEY + LOCINSIGHT_API_TOKEN

# 4. Run
uvicorn app.main:app --reload --port 8000

# 5. Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "X-LocInsight-Token: $LOCINSIGHT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"lat": -8.6705, "lng": 115.2126, "save": true}'
```

## Deployment to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space
   - SDK: **Docker**
   - License: Apache 2.0
2. Connect the Space to this GitHub repo (Settings > Repository > Connect)
3. Add Secrets (Settings > Secrets):
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `LOCINSIGHT_API_TOKEN`
   - `CORS_ALLOWED_ORIGINS` (set to your Vercel app URL)
4. The Space auto-builds on every push to `main`

## Maintained By

**Achmad Bayhaqy** — Data Team, MAP Active Adiperkasa (MAA)
Last updated: 2026-08-08
