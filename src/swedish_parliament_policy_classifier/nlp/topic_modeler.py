"""BERTopic-based topic modeling for the Swedish parliamentary motion corpus.

Party-agnostic by design: we never pass party labels into the topic model.
Uses the same sentence-transformer embeddings as the semantic matcher.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from swedish_parliament_policy_classifier.io import loader

import numpy as np

LOG = logging.getLogger(__name__)


def _get_default_model_path() -> Path:
    try:
        return Path(__file__).resolve().parents[3] / "data" / "bertopic_model.pkl"
    except Exception:
        return Path("data/bertopic_model.pkl")


def _get_default_topics_path() -> Path:
    try:
        return Path(__file__).resolve().parents[3] / "data" / "motion_topics.json.zst"
    except Exception:
        return Path("data/motion_topics.json.zst")


def fit_topic_model(
    texts: List[str],
    motion_ids: List[str],
    embedding_model_name: str = "KBLab/sentence-bert-swedish-cased",
    model_path: Optional[Path] = None,
    topics_path: Optional[Path] = None,
    n_topics: Optional[int] = None,
    min_topic_size: int = 50,
) -> Tuple[object, Dict[str, List[float]]]:
    """Train a BERTopic model on motion texts and return topic distributions.

    Args:
        texts: List of motion texts (already preprocessed/extracted policy text).
        motion_ids: Parallel list of motion IDs.
        embedding_model_name: Sentence transformer model name.
        model_path: Where to save the fitted BERTopic model.
        topics_path: Where to save the per-motion topic distributions JSON.
        n_topics: Target number of topics (None = let HDBSCAN decide).
        min_topic_size: Minimum documents per topic for HDBSCAN.

    Returns:
        (topic_model, motion_topics_dict)
    """
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN

    LOG.info("Training BERTopic on %d motions...", len(texts))

    # Use same embedding model as the semantic matcher
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(embedding_model_name, device=device)

    # Configure UMAP and HDBSCAN for reproducibility
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True,
        nr_topics=n_topics,
        calculate_probabilities=True,
    )

    # Fit and transform
    topics, probs = topic_model.fit_transform(texts)

    # Build per-motion topic distribution dict
    # probs shape: (n_docs, n_topics) — probability of belonging to each topic
    motion_topics: Dict[str, List[float]] = {}
    for i, mid in enumerate(motion_ids):
        if probs is not None and len(probs) > i:
            motion_topics[mid] = probs[i].tolist()
        else:
            # Fallback: one-hot based on assigned topic
            n_topics_found = len(topic_model.get_topic_info())
            vec = [0.0] * n_topics_found
            if topics[i] >= 0 and topics[i] < n_topics_found:
                vec[topics[i]] = 1.0
            motion_topics[mid] = vec

    # Save model
    if model_path is None:
        model_path = _get_default_model_path()
    loader.save_pickle(model_path, topic_model)
    LOG.info("Saved BERTopic model to %s", model_path)

    # Save topic distributions
    if topics_path is None:
        topics_path = _get_default_topics_path()
    loader.save_json(topics_path, motion_topics)
    LOG.info("Saved topic distributions to %s", topics_path)

    return topic_model, motion_topics


def load_topic_distributions(topics_path: Optional[Path] = None) -> Dict[str, List[float]]:
    """Load cached per-motion topic distributions.

    Handles both list distributions (probabilities per topic) and float
    values (single-topic confidence) by normalising to a list.
    """
    if topics_path is None:
        topics_path = _get_default_topics_path()
    try:
        raw = loader.load_json(topics_path)
    except FileNotFoundError:
        return {}
    # Normalise floats to single-element lists so callers always get lists
    return {
        mid: ([v] if isinstance(v, (int, float)) else list(v))
        for mid, v in raw.items()
    }


def load_topic_model(model_path: Optional[Path] = None) -> Optional[object]:
    """Load a saved BERTopic model."""
    if model_path is None:
        model_path = _get_default_model_path()
    if not model_path.exists() and not Path(str(model_path) + ".zst").exists():
        return None
    return loader.load_pickle(model_path)


def get_topic_features(
    motion_id: str,
    topic_distributions: Optional[Dict[str, List[float]]] = None,
    topics_path: Optional[Path] = None,
) -> Optional[List[float]]:
    """Get topic distribution vector for a single motion."""
    if topic_distributions is None:
        topic_distributions = load_topic_distributions(topics_path)
    return topic_distributions.get(motion_id)
