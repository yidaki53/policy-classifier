"""Classifier core facade providing clearer boundaries between extraction,
signal computation, and combination.

This module provides a small `ClassifierCore` class which acts as a
single entrypoint for classification-related operations. For now it delegates
to the existing `score_motion` implementation while offering a stable
surface for future refactors (signal plugins, pipeline composition, DI).
"""
from collections.abc import Callable
from typing import List, Dict, Any, Optional

from swedish_parliament_policy_classifier.classifier.scorer import score_motion, score_speech
from swedish_parliament_policy_classifier.classifier.signal_executor import (
    SignalContext,
    SignalExecutor,
    SignalProvider,
    SignalRun,
)
from swedish_parliament_policy_classifier.models.models import ClassificationResult


class ClassifierCore:
    def __init__(
        self,
        adapters: Optional[Dict[str, object]] = None,
        motion_scorer: Callable[..., List[ClassificationResult]] = score_motion,
        speech_scorer: Callable[..., List[ClassificationResult]] = score_speech,
        signal_executor: Optional[SignalExecutor] = None,
    ):
        self.adapters = adapters or {}
        self.motion_scorer = motion_scorer
        self.speech_scorer = speech_scorer
        self.signal_executor = signal_executor or SignalExecutor()

    def classify(
        self,
        motion_id: str,
        text: str,
        categories: Dict[str, Any],
        mode: str = "motion",
        **kwargs,
    ) -> List[ClassificationResult]:
        """Classify a motion or speech.

        Dispatches explicitly to the motion or speech scorer while preserving
        the existing public result contract.
        """
        if mode == "speech":
            return self.speech_scorer(
                speech_id=motion_id,
                text=text,
                categories=categories,
                **kwargs,
            )
        if mode != "motion":
            raise ValueError(f"Unsupported classification mode: {mode!r}")
        return self.motion_scorer(
            motion_id=motion_id,
            text=text,
            categories=categories,
            **kwargs,
        )

    def compute_signals(
        self,
        item_id: str,
        text: str,
        categories: Dict[str, Any],
        *,
        providers: Optional[List[SignalProvider]] = None,
        required: frozenset[str] = frozenset(),
    ) -> SignalRun:
        """Execute injected signals with explicit degradation semantics."""
        configured = providers or [
            provider
            for provider in self.adapters.values()
            if hasattr(provider, "compute") and hasattr(provider, "name")
        ]
        return self.signal_executor.run(
            configured,
            SignalContext(item_id=item_id, text=text, categories=categories),
            required=required,
        )
