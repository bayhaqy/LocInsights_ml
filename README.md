---
title: LocInsights ML Engine
emoji: 🗺️
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 4.44.1
app_port: 7860
pinned: false
license: apache-2.0
shortDescription: "ML engine for LocInsight — site selection scoring + Bali scraping worker for MAP Active Adiperkasa"
---

# LocInsights ML Engine

FastAPI service (mounted inside Gradio) that powers **site selection scoring** and **Bali scraping** for the LocInsight location intelligence system.

## Architecture

This Space uses the **Gradio SDK** (free tier compatible) with FastAPI mounted via `gr.mount_gradio_app`. This preserves all REST API endpoints while adding an interactive UI at `/ui`.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET  | `/health` | None | Liveness probe (Vercel cron hits this) |
| GET  | `/` | None | Service info |
| GET  | `/docs` | None | OpenAPI Swagger UI |
| GET  | `/ui` | None | Gradio interactive test panel |
| POST | `/predict` | Bearer | Site success probability score (0-100%) |
| POST | `/scrape_bali` | Bearer | Start async Bali scraping job |
| GET  | `/scrape_bali/{job_id}` | Bearer | Poll scrape job status |
| POST | `/train` | Bearer | Retrain GBR model from latest Supabase data |
| GET  | `/blank_spots` | Bearer | Recommended new-location candidates |
| GET  | `/model/info` | Bearer | Currently-loaded model metadata |

## Security

All endpoints (except `/health`, `/`, `/docs`, `/ui`) require a custom Bearer token via the `X-LocInsight-Token` header. Only the Vercel backend has this token.

## Space Secrets (configure in Settings)

| Secret | Description |
|--------|-------------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role key (bypasses RLS) |
| `LOCINSIGHT_API_TOKEN` | Custom bearer token for API auth |
| `CORS_ALLOWED_ORIGINS` | Comma-separated allowed origins (Vercel URL) |

## Source Code
Full source code: https://github.com/bayhaqy/LocInsights_ml

## Maintained By
**Achmad Bayhaqy** — Data Team, MAP Active Adiperkasa (MAA)
Last updated: 2026-08-08
