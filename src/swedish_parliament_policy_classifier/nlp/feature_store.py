"""Small feature store wrapper for cached artifacts like category embeddings
and topic distributions.

This provides a single place callers can use to load or compute cached
features without duplicating cache path logic.
"""
from pathlib import Path
from typing import Dict, Any, Optional

from swedish_parliament_policy_classifier.nlp import embeddings_cache
from swedish_parliament_policy_classifier.nlp import topic_modeler
from swedish_parliament_policy_classifier.io import loader


def get_category_embeddings(categories: Dict[str, Any], matcher, cache_path: Optional[Path] = None, prefer_cache: bool = True) -> Dict[str, Any]:
    if cache_path is None:
        try:
            cache_path = Path(__file__).resolve().parents[3] / "data" / "category_embeddings.pkl"
        except Exception:
            cache_path = Path("data/category_embeddings.pkl")

    if prefer_cache and cache_path.exists():
        try:
            return embeddings_cache.load_cached_embeddings(cache_path)
        except Exception:
            # fall through to recompute
            pass

    # Compute and persist
    embs = embeddings_cache.compute_category_embeddings(categories, matcher)
    try:
        embeddings_cache.save_cached_embeddings(cache_path, embs)
    except Exception:
        pass
    return embs


def get_topic_distributions(topics_path: Optional[Path] = None) -> Dict[str, Any]:
    return topic_modeler.load_topic_distributions(topics_path)
