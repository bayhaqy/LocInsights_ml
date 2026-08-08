# =============================================================
# LocInsight ML Engine — Dockerfile (Hugging Face Spaces)
# Uses Python 3.11 slim base; uvicorn + FastAPI
# =============================================================
FROM python:3.11-slim

LABEL org.opencontainers.image.title="LocInsight ML Engine"
LABEL org.opencontainers.image.description="Site selection scoring + Bali scraping worker for MAP Active Adiperkasa"
LABEL org.opencontainers.image.source="https://github.com/bayhaqy/LocInsights_ml"

# System deps for psycopg2, geospatial libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create artifacts directory for trained models
RUN mkdir -p /app/artifacts /data

# Expose port (HF Spaces expects 7860)
ENV PORT=7860
EXPOSE 7860

# Health check (every 5 min)
HEALTHCHECK --interval=300s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fs http://localhost:7860/health || exit 1

# Run with uvicorn (single worker for HF Spaces CPU basic tier)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--log-level", "info"]
