"""Model loader — loads active GBR model from artifact file (or HF hub)."""
from __future__ import annotations

import os
import logging
from typing import Any, Dict, Tuple

from app.config import get_settings
from app.ml.scoring import load_model_artifact

log = logging.getLogger(__name__)


def load_active_model() -> Tuple[Any, Dict]:
    """Load the active model.

    Order of precedence:
      1. Local artifact at MODEL_ARTIFACT_PATH (set by /train endpoint)
      2. Bundled default artifact (if present in artifacts/)
      3. None (use fallback heuristic in scoring.fallback_score)
    """
    settings = get_settings()
    candidates = [
        settings.model_artifact_path,
        os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "gbr_site_scoring.json"),
        "/data/artifacts/gbr_site_scoring.json",  # HF Spaces persistent storage
    ]

    for path in candidates:
        path = os.path.normpath(path)
        if os.path.exists(path):
            try:
                model, meta = load_model_artifact(path)
                log.info(f"Loaded model from {path}")
                return model, meta
            except Exception as e:
                log.warning(f"Failed to load model from {path}: {e}")

    log.warning("No model artifact found — will use fallback heuristic")
    return None, {}
