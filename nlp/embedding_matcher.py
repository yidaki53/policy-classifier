"""Embedding-based semantic matcher with fallbacks.

If `sentence_transformers` is available we compute embeddings; otherwise a
keyword-overlap fallback is used. Category embeddings are computed from their
keywords/definitions.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

LOG = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np
except Exception:
    SentenceTransformer = None
    np = None

from swedish_parliament_policy_classifier.models.models import CategoryDef
from swedish_parliament_policy_classifier.nlp.embeddings_cache import load_cached_embeddings


class EmbeddingMatcher:
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        cache_path: Optional[Union[str, Path]] = None,
        prefer_cache: bool = True,
    ):
        self.model_name = model_name
        self.model = None
        self.cached_embeddings: Optional[Dict[str, object]] = None

        # Default cache path: repository/data/category_embeddings.pkl
        try:
            default_cache = Path(__file__).resolve().parents[2] / "data" / "category_embeddings.pkl"
        except Exception:
            default_cache = None

        if cache_path is None:
            cache_path = default_cache

        # Try loading cached embeddings first (fast path)
        if prefer_cache and cache_path is not None:
            try:
                p = Path(cache_path)
                if p.exists():
                    self.cached_embeddings = load_cached_embeddings(p)
                    LOG.info("Loaded cached category embeddings from %s", p)
            except Exception as e:
                LOG.warning("Failed to load cached embeddings from %s: %s", cache_path, e)

        # Still attempt to load a model if available to encode queries
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                LOG.warning("Failed to load SentenceTransformer '%s': %s", model_name, e)
                self.model = None

    def encode(self, texts: List[str]):
        if self.model is None:
            raise RuntimeError("SentenceTransformer not available; install sentence-transformers")
        return self.model.encode(texts, convert_to_numpy=True)

    def _mean_embedding(self, texts: List[str]):
        embs = self.encode(texts)
        return embs.mean(axis=0)

    def build_category_embeddings(self, cats: Dict[str, CategoryDef]) -> Dict[str, object]:
        embeddings = {}
        for name, cat in cats.items():
            parts = []
            parts.extend(cat.keywords or [])
            if cat.definition:
                parts.append(cat.definition)
            if self.model is not None and parts:
                embeddings[name] = self._mean_embedding(parts)
            else:
                embeddings[name] = parts
        return embeddings

    @staticmethod
    def _cosine(a, b):
        if np is None:
            raise RuntimeError("numpy required for cosine computation")
        a = a.astype(float)
        b = b.astype(float)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def match(self, text: str, category_embeddings: Dict[str, object], top_k: int = 3) -> List[Tuple[str, float]]:
        # If model available, compute embedding and cosine similarities
        if self.model is not None:
            q = self._mean_embedding([text])
            scores = []
            for name, emb in category_embeddings.items():
                if emb is None:
                    continue
                try:
                    s = self._cosine(q, emb)
                except Exception:
                    s = 0.0
                scores.append((name, s))
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

        # Fallback: keyword overlap score
        txt = text.lower()
        scores = []
        for name, parts in category_embeddings.items():
            # parts is a list of keywords/definition strings in fallback
            matches = 0
            total = max(1, len(parts))
            for p in parts:
                if p and p.lower() in txt:
                    matches += 1
            scores.append((name, matches / total))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
