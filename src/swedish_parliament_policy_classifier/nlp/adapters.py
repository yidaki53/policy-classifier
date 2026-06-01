"""Adapter interfaces for ML models used by the pipeline.

Define minimal adapter interfaces so callers can depend on small, testable
contracts instead of heavyweight concrete implementations.
"""
from typing import List, Dict, Any
from abc import ABC, abstractmethod


class EmbeddingAdapter(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> Any:
        """Encode a list of texts into a numeric embedding array."""

    @abstractmethod
    def build_category_embeddings(self, categories: Dict[str, Any]) -> Dict[str, Any]:
        """Return a mapping category_name -> embedding object"""


class ZeroShotAdapter(ABC):
    @abstractmethod
    def score(self, text: str) -> Dict[str, float]:
        """Return per-category entailment / scores for the provided text."""


class TransformerAdapter(ABC):
    @abstractmethod
    def predict_proba(self, text: str) -> Dict[str, float]:
        """Return a mapping category -> probability from a transformer classifier."""


class LLMAdapter(ABC):
    @abstractmethod
    def classify(self, text: str, **kwargs) -> Dict[str, Any]:
        """Return a classification dict (e.g., {'category':..., 'score':..., 'reason':...})."""
