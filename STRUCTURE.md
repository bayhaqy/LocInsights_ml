# LocInsights ML Engine — File Structure

```
LocInsights_ml/
├── index.html              # Main entry — Gradio Lite + Pyodide + scikit-learn GBR
├── README.md               # This Space's documentation (HF metadata in frontmatter)
├── .gitignore              # Python + IDE ignores
├── .gitattributes          # Line ending normalization
└── archive/
    └── server-side-fastapi/  # Previous FastAPI implementation (archived)
        ├── app.py            # Original entry point
        ├── app_gradio.py     # Gradio SDK wrapper
        ├── app/              # FastAPI routes (predict, scrape, train, health)
        ├── requirements.txt  # Python deps (no longer needed for static)
        ├── .env.example      # Env var template
        └── artifacts/        # Model artifacts (gitignored)
```

## Why Gradio Lite (static SDK)?

Hugging Face Spaces free tier supports `static` SDK which serves HTML/CSS/JS
directly — no Python runtime on the server. We use `@gradio/lite` to run
Python **in the user's browser** via Pyodide (WebAssembly).

### Benefits:
- ✅ No server cost (runs in browser)
- ✅ No cold start (cached after first load)
- ✅ No API auth needed (all client-side)
- ✅ scikit-learn + numpy + pandas available via Pyodide
- ✅ Compatible with HF Spaces free tier

### Trade-offs:
- ⚠️ First load ~10-20s (Pyodide + scikit-learn install)
- ⚠️ Cannot connect to Supabase directly (browser CORS + no service role key)
- ⚠️ Cannot run background scraping jobs

For heavy server-side work (scraping, DB writes), the LocInsight Next.js backend
on Vercel handles those tasks via API routes.
