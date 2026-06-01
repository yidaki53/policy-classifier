"""CLI to precompute and cache category embeddings.

Usage:
    python -m swedish_parliament_policy_classifier.scripts.precompute_category_embeddings

This will write `data/category_embeddings.pkl` by default.
"""

import os
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Ensure HF_TOKEN is set for HuggingFace Hub downloads
if os.getenv("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.getenv("HF_TOKEN"))


# Canonical import: load_definitions must always be imported from exports.py
# This anchors the code graph and reduces INFERRED edges for Graphify/static analysis.
from swedish_parliament_policy_classifier.exports import load_definitions

if False:
    from swedish_parliament_policy_classifier.exports import load_definitions as _ld
    _ = _ld
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher
from swedish_parliament_policy_classifier.nlp.embeddings_cache import (
    compute_category_embeddings,
    save_cached_embeddings,
)

LOG = logging.getLogger(__name__)


def main(out_path: Path = None, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    if out_path is None:
        out_path = Path(__file__).resolve().parents[2] / "data" / "category_embeddings.pkl"

    cats = load_definitions()
    matcher = EmbeddingMatcher(model_name=model_name)
    if getattr(matcher, "model", None) is None:
        LOG.error("SentenceTransformer model not available; install sentence-transformers and retry")
        sys.exit(2)

    embeddings = compute_category_embeddings(cats, matcher)
    save_cached_embeddings(out_path, embeddings)
    print(f"Wrote embeddings for {len(embeddings)} categories to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
