"""Package-local NLP helpers."""

from .preprocess import preprocess_text, init_spacy
from .embedding_matcher import EmbeddingMatcher
from .embeddings_cache import compute_category_embeddings, save_cached_embeddings, load_cached_embeddings

__all__ = [
	"preprocess_text",
	"init_spacy",
	"EmbeddingMatcher",
	"compute_category_embeddings",
	"save_cached_embeddings",
	"load_cached_embeddings",
]
