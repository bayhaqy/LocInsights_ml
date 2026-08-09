---
title: LocInsights ML Engine
emoji: 🗺️
colorFrom: indigo
colorTo: red
sdk: static
app_port: 7860
pinned: false
license: apache-2.0
shortDescription: "ML engine for LocInsight — GBR site selection scoring in Python (Gradio Lite / Pyodide)"
---

# LocInsights ML Engine (Gradio Lite)

ML engine for the **LocInsight** location intelligence system. Runs entirely in
the browser using **Gradio Lite** + **Pyodide** — no server-side Python required.
This makes it compatible with Hugging Face Spaces free tier (static SDK).

## How it works

1. The page loads `@gradio/lite` from a CDN
2. Pyodide boots up in a Web Worker and installs `scikit-learn`, `numpy`, `pandas`
3. The user interface is built with Gradio Blocks (Python code)
4. The Gradient Boosting Regressor (GBR) is trained **in-browser** on synthetic
   Bali data calibrated to the LocInsight scoring engine
5. Predictions are computed locally — no API calls, no cold start, no auth needed

## Features

- **Train GBR Model**: Click to train a Gradient Boosting Regressor on synthetic
  Bali kelurahan × brand data. Shows training metrics (RMSE, R²) and feature
  importance.
- **Predict Site Score**: Input lat/lng + brand to get a **Store Success
  Probability Score (0-100%)** based on competitor density, POI density, mall
  proximity, and demographics.
- **Find Blank Spots**: Identify high-score candidate sites with no existing
  MAA store nearby — highlighted in green on the LocInsight map.

## Integration with LocInsight web app

The LocInsight Next.js frontend embeds this Space via an `<iframe>` in the
**ML / AI Engine** page. All ML computation happens client-side, so the Vercel
backend doesn't need to proxy requests — the iframe loads directly from HF.

## Source Code

Full source: https://github.com/bayhaqy/LocInsights_ml

## Maintained By

**Achmad Bayhaqy** — Data Team, MAP Active Adiperkasa (MAA)
Last updated: 2026-08-09
