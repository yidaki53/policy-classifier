"""Classifier core facade providing clearer boundaries between extraction,
signal computation, and combination.

This module provides a small `ClassifierCore` class which acts as a
single entrypoint for classification-related operations. For now it delegates
to the existing `score_motion` implementation while offering a stable
surface for future refactors (signal plugins, pipeline composition, DI).
"""
from typing import List, Dict, Any, Optional

from swedish_parliament_policy_classifier.classifier.pipeline import score_motion
from swedish_parliament_policy_classifier.models.models import ClassificationResult


class ClassifierCore:
    def __init__(self, adapters: Optional[Dict[str, object]] = None):
        self.adapters = adapters or {}

    def classify(
        self,
        motion_id: str,
        text: str,
        categories: Dict[str, Any],
        mode: str = "motion",
        **kwargs,
    ) -> List[ClassificationResult]:
        """Classify a motion or speech.

        This thin façade currently delegates to `score_motion` to preserve
        existing behaviour while offering a single place to evolve the
        pipeline into smaller units (extractor/signals/combiner).
        """
        # Map 'mode' to the existing scorer flags
        if mode == "speech":
            kwargs.setdefault("skip_policy_extraction", True)
            kwargs.setdefault("use_speech_preprocessing", True)
        return score_motion(motion_id=motion_id, text=text, categories=categories, **kwargs)

    def compute_signals(self, *args, **kwargs):
        """Placeholder for computing and returning individual signals.

        Currently a thin wrapper around `classify` that returns raw signal
        dictionaries constructed from `ClassificationResult` objects.
        """
        results = self.classify(*args, **kwargs)
        sigs = {r.category: {"raw_score": r.raw_score, "matched_rules": r.matched_rules} for r in results}
        return sigs
