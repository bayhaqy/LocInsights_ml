"""
LocInsight ML Engine — HF Spaces entry point
============================================

This file is what Hugging Face Spaces (Gradio SDK) looks for.
It imports the combined FastAPI + Gradio app from app_gradio.py
and launches it on port 7860.

The app exposes:
  - All FastAPI endpoints (/health, /predict, /scrape_bali, /train, /blank_spots)
  - A Gradio UI at /ui for interactive testing
  - The root / serves the FastAPI service info (backwards compatible)
"""
import os
import sys

# Ensure the app package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the combined app (FastAPI + Gradio mounted)
from app_gradio import app, demo  # noqa: F401

# HF Spaces Gradio SDK auto-detects the `demo` variable and launches it.
# The FastAPI app is mounted via gr.mount_gradio_app, so all REST endpoints
# are preserved at their original paths.

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        prevent_thread_lock=False,
        show_error=True,
    )
