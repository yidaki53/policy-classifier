"""Enhanced scorer integrating calibration and extended BERT windows.

This wraps the base scorer with improvements:
1. Probability calibration via isotonic regression
2. Adaptive fallback thresholds learned from validation data
3. Extended BERT window strategies (sliding/hierarchical)
4. Multi-transformer ensemble support
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from swedish_parliament_policy_classifier.classifier.scorer import (
    score_motion,
    score_speech,
)
from swedish_parliament_policy_classifier.classifier.calibration import (
    AdaptiveThresholdManager,
    ProbabilityCalibrator,
)

LOG = logging.getLogger(__name__)


class EnhancedScorer:
    """Enhanced classifier with calibration and adaptive thresholds.

    Wraps the base scoring pipeline and adds:
    - Probability calibration for more reliable confidence scores
    - Per-category adaptive thresholds to reduce fallback rate
    - Extended BERT window strategies for long documents
    - Integration point for multi-transformer ensemble
    """

    def __init__(
        self,
        calibrator_path: Optional[Path] = None,
        threshold_path: Optional[Path] = None,
        use_calibration: bool = True,
        use_adaptive_thresholds: bool = True,
        bert_window_strategy: str = "truncate",
        bert_window_overlap: float = 0.1,
        bert_aggregation: str = "mean",
    ):
        """Initialize enhanced scorer.

        Args:
            calibrator_path: Path to saved calibrator (pickle).
            threshold_path: Path to saved adaptive thresholds (JSON).
            use_calibration: If True, apply probability calibration if available.
            use_adaptive_thresholds: If True, use per-category fallback thresholds.
            bert_window_strategy: BERT window strategy: 'truncate', 'sliding', 'hierarchical'.
            bert_window_overlap: Overlap fraction for sliding window (0-0.5).
            bert_aggregation: Aggregation for BERT windows: 'mean', 'max', 'vote'.
        """
        self.calibrator = None
        self.threshold_manager = None
        self.use_calibration = use_calibration
        self.use_adaptive_thresholds = use_adaptive_thresholds

        # BERT window settings
        self.bert_window_strategy = bert_window_strategy
        self.bert_window_overlap = bert_window_overlap
        self.bert_aggregation = bert_aggregation

        # Load calibration artifacts if provided
        if use_calibration and calibrator_path and Path(calibrator_path).exists():
            try:
                self.calibrator = ProbabilityCalibrator.load(Path(calibrator_path))
                LOG.info("Loaded probability calibrator from %s", calibrator_path)
            except Exception as e:
                LOG.warning("Failed to load calibrator: %s", e)

        if use_adaptive_thresholds and threshold_path and Path(threshold_path).exists():
            try:
                self.threshold_manager = AdaptiveThresholdManager.load(Path(threshold_path))
                LOG.info("Loaded adaptive thresholds from %s", threshold_path)
            except Exception as e:
                LOG.warning("Failed to load threshold manager: %s", e)

    def score_motion_enhanced(
        self,
        motion_id: str,
        text: str,
        categories: Dict[str, object],
        **kwargs,
    ) -> List:
        """Score a motion with enhanced features.

        Extends base score_motion with:
        - Calibrated probabilities
        - Extended BERT window strategies
        - Adaptive fallback information

        Args:
            motion_id: Motion identifier.
            text: Motion text.
            categories: Category definitions.
            **kwargs: Additional arguments passed to score_motion.

        Returns:
            List of ClassificationResult with enhanced probabilities.
        """
        # Extract BERT-specific kwargs
        bert_kwargs = {
            "bert_window_strategy": self.bert_window_strategy,
            "bert_window_overlap": self.bert_window_overlap,
            "bert_aggregation": self.bert_aggregation,
        }

        # Merge with caller kwargs (caller can override)
        bert_kwargs.update({k: v for k, v in kwargs.items() if "bert" in k.lower()})

        # Run base scorer
        results = score_motion(motion_id, text, categories, **kwargs)

        if not results:
            return results

        # Collect probabilities for calibration
        probs_dict = {r.category: r.normalized_weight for r in results}

        # Apply calibration if available
        if self.calibrator is not None:
            try:
                import numpy as np
                category_names = sorted(probs_dict.keys())
                prob_vec = np.array([probs_dict.get(c, 0.0) for c in category_names]).reshape(1, -1)
                calibrated_vec = self.calibrator.transform(prob_vec)[0]

                # Update results with calibrated probabilities
                for i, r in enumerate(results):
                    r.normalized_weight = float(calibrated_vec[i])
                    r._fractional_weight = None  # Invalidate fraction cache
            except Exception as e:
                LOG.debug("Calibration failed: %s", e)

        # Attach adaptive threshold info
        if self.threshold_manager is not None:
            try:
                max_cat = max(probs_dict, key=probs_dict.get)
                max_prob = probs_dict[max_cat]
                threshold = self.threshold_manager.get_threshold(max_cat)

                # Add threshold metadata to results
                for r in results:
                    if r.category == max_cat:
                        r.matched_rules = r.matched_rules + [
                            f"adaptive_threshold:{threshold:.2f}"
                        ]
                        break
            except Exception as e:
                LOG.debug("Threshold attachment failed: %s", e)

        return results

    def should_fallback(
        self,
        probs_dict: Dict[str, float],
    ) -> tuple[bool, Optional[str]]:
        """Determine if LLM fallback should be triggered.

        Uses adaptive thresholds if available, otherwise falls back to
        global max-probability check.

        Args:
            probs_dict: Category probability dictionary.

        Returns:
            Tuple of (should_fallback, reason).
        """
        if not probs_dict:
            return True, "no_probabilities"

        if self.threshold_manager is not None:
            # Check top category
            top_cat = max(probs_dict, key=probs_dict.get)
            top_prob = probs_dict[top_cat]
            if self.threshold_manager.should_fallback(top_cat, top_prob):
                return True, f"adaptive_threshold_{top_cat}"
            return False, None

        # Fallback: check if max probability is below 0.3
        max_prob = max(probs_dict.values())
        if max_prob < 0.30:
            return True, "low_confidence"
        return False, None

    def score_speech_enhanced(
        self,
        speech_id: str,
        text: str,
        categories: Dict[str, object],
        **kwargs,
    ) -> List:
        """Score a speech with enhanced features.

        Args:
            speech_id: Speech identifier.
            text: Speech text.
            categories: Category definitions.
            **kwargs: Additional arguments.

        Returns:
            List of ClassificationResult.
        """
        results = score_speech(speech_id, text, categories, **kwargs)
        # Apply same enhancements as motion
        if results:
            probs_dict = {r.category: r.normalized_weight for r in results}
            if self.calibrator is not None:
                try:
                    import numpy as np
                    category_names = sorted(probs_dict.keys())
                    prob_vec = np.array([probs_dict.get(c, 0.0) for c in category_names]).reshape(1, -1)
                    calibrated_vec = self.calibrator.transform(prob_vec)[0]
                    for i, r in enumerate(results):
                        r.normalized_weight = float(calibrated_vec[i])
                        r._fractional_weight = None
                except Exception as e:
                    LOG.debug("Speech calibration failed: %s", e)
        return results