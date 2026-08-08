---
title: LocInsights ML Engine
emoji: 🗺️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
shortDescription: "ML engine for LocInsight — site selection scoring + Bali scraping worker for MAP Active Adiperkasa"
---

# LocInsights ML Engine

FastAPI service that powers **site selection scoring** and **Bali scraping** for the LocInsight location intelligence system.

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

## Source Code
Full source code: https://github.com/bayhaqy/LocInsights_ml

## Maintained By
**Achmad Bayhaqy** — Data Team, MAP Active Adiperkasa (MAA)
Last updated: 2026-08-08
