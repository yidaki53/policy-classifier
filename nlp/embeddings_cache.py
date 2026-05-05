"""Compute and cache category embeddings for faster semantic matching.

This module provides simple helpers to compute category embeddings using the
`EmbeddingMatcher` and persist them to disk (pickle). It's intentionally
lightweight and defensive: if sentence-transformers isn't available, callers
should use the fallback matcher behavior instead of cached embeddings.
"""

from pathlib import Path
from typing import Dict
import pickle
import logging

from swedish_parliament_policy_classifier.models.models import CategoryDef
from typing import Any

LOG = logging.getLogger(__name__)


def compute_category_embeddings(categories: Dict[str, CategoryDef], matcher: Any) -> Dict[str, object]:
    """Return a mapping category_name -> embedding (numpy array) using matcher.

    Raises ValueError if the matcher has no model loaded.
    """
    if matcher is None:
        raise ValueError("matcher must be provided")
    if getattr(matcher, "model", None) is None:
        raise ValueError("Embedding matcher has no loaded model; cannot compute embeddings")

    return matcher.build_category_embeddings(categories)


def save_cached_embeddings(path: Path, embeddings: Dict[str, object]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)
    LOG.info("Saved %d category embeddings to %s", len(embeddings), path)


def load_cached_embeddings(path: Path) -> Dict[str, object]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data
