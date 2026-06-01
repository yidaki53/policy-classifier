"""Deep scoring facade that hides pipeline wiring behind one call boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from swedish_parliament_policy_classifier.models.models import CategoryDef, ClassificationResult
from swedish_parliament_policy_classifier.classifier.pipeline import score_motion


@dataclass
class DeepScoringService:
    categories: Dict[str, CategoryDef]
    embedding_matcher: Optional[object] = None
    topic_distributions: Optional[dict] = None
    meta_clf: Optional[dict] = None

    def classify(self, item_id: str, text: str) -> list[ClassificationResult]:
        return score_motion(
            motion_id=item_id,
            text=text,
            categories=self.categories,
            embedding_matcher=self.embedding_matcher,
            topic_distributions=self.topic_distributions,
            meta_clf=self.meta_clf,
            use_zero_shot=True,
            skip_policy_extraction=True,
            use_speech_preprocessing=True,
            use_ollama=False,
        )
