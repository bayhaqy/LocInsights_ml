"""Routes package — exposes APIRouter instances from each module."""
from . import health, predict, scrape, train, blank_spots

__all__ = ["health", "predict", "scrape", "train", "blank_spots"]
