"""Probability calibration for ensemble predictions.

Provides isotonic regression and temperature scaling calibration to improve
prediction confidence reliability and reduce unnecessary LLM fallbacks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Calibrates ensemble probabilities using isotonic regression.

    Fits per-category calibrators on validation data to map raw ensemble
    probabilities to reliable confidence scores. Uncalibrated models often
    produce overconfident predictions, leading to unnecessary fallbacks.
    """

    def __init__(self, category_names: List[str]):
        self.category_names = category_names
        self.calibrators: Dict[str, object] = {}
        self._fitted = False

    def fit(self, y_true: np.ndarray, probs: np.ndarray) -> None:
        """Fit calibrators on validation data.

        Args:
            y_true: Integer-encoded true labels, shape (n_samples,)
            probs: Predicted probabilities, shape (n_samples, n_categories)
        """
        from sklearn.isotonic import IsotonicRegression

        n_categories = len(self.category_names)
        if probs.shape[1] != n_categories:
            raise ValueError(
                f"Probability shape {probs.shape} doesn't match "
                f"{n_categories} categories"
            )

        for i, cat in enumerate(self.category_names):
            # Create binary problem: is this category present?
            y_binary = (y_true == i).astype(int)
            prob_pos = probs[:, i]

            # Skip if only one class present
            if len(np.unique(y_binary)) < 2:
                LOG.warning(
                    "Category '%s' has only one class in validation data; "
                    "skipping calibration",
                    cat,
                )
                continue

            calibrator = IsotonicRegression(out_of_bounds='clip')
            try:
                calibrator.fit(prob_pos, y_binary)
                self.calibrators[cat] = calibrator
            except Exception as e:
                LOG.warning("Calibration failed for category '%s': %s", cat, e)

        self._fitted = len(self.calibrators) > 0
        if self._fitted:
            LOG.info(
                "Calibration fitted for %d/%d categories",
                len(self.calibrators),
                n_categories,
            )

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to probability matrix.

        Args:
            probs: Raw probabilities, shape (n_samples, n_categories)

        Returns:
            Calibrated probabilities (same shape, normalized to sum=1)
        """
        if not self._fitted:
            return probs

        calibrated = probs.copy()
        for i, cat in enumerate(self.category_names):
            if cat in self.calibrators:
                try:
                    calibrated[:, i] = self.calibrators[cat].predict(probs[:, i])
                except Exception as e:
                    LOG.debug("Calibration transform failed for '%s': %s", cat, e)

        # Normalize to sum=1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)
        calibrated = calibrated / row_sums

        return calibrated

    def fit_transform(self, y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """Fit calibrators and return calibrated probabilities.

        Args:
            y_true: Integer-encoded true labels
            probs: Raw predicted probabilities

        Returns:
            Calibrated probabilities
        """
        self.fit(y_true, probs)
        return self.transform(probs)

    def save(self, path: Path) -> None:
        """Save calibrators to disk.

        Args:
            path: Path to save calibrator state (pickle format)
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted; cannot save")

        import pickle
        state = {
            'category_names': self.category_names,
            'calibrators': self.calibrators,
            '_fitted': self._fitted,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        LOG.info("Saved calibrator to %s", path)

    @classmethod
    def load(cls, path: Path) -> 'ProbabilityCalibrator':
        """Load calibrator from disk.

        Args:
            path: Path to saved calibrator state

        Returns:
            Loaded ProbabilityCalibrator instance
        """
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)

        calibrator = cls(state['category_names'])
        calibrator.calibrators = state['calibrators']
        calibrator._fitted = state['_fitted']
        LOG.info("Loaded calibrator from %s", path)
        return calibrator


class AdaptiveThresholdManager:
    """Manages per-category fallback thresholds based on validation performance.

    Instead of a single global threshold, learns category-specific thresholds
    that balance recall vs. fallback rate. Hard categories get lower thresholds
    (more fallbacks), easy categories get higher thresholds (fewer fallbacks).
    """

    def __init__(
        self,
        category_names: List[str],
        default_threshold: float = 0.30,
        min_threshold: float = 0.15,
        max_threshold: float = 0.50,
    ):
        self.category_names = category_names
        self.default_threshold = default_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.thresholds: Dict[str, float] = {}
        self._fitted = False

    def fit(
        self,
        y_true: np.ndarray,
        probs: np.ndarray,
        target_fallback_rate: float = 0.15,
    ) -> None:
        """Learn per-category thresholds from validation data.

        Args:
            y_true: Integer-encoded true labels
            probs: Predicted probabilities
            target_fallback_rate: Overall target fallback rate (0-1)
        """
        n_categories = len(self.category_names)

        for i, cat in enumerate(self.category_names):
            y_binary = (y_true == i).astype(int)
            prob_pos = probs[:, i]

            # Compute precision at various thresholds
            precisions = []
            recalls = []
            fallback_rates = []

            thresholds = np.linspace(0.05, 0.7, 20)
            for thresh in thresholds:
                pred_pos = prob_pos >= thresh
                if pred_pos.sum() == 0:
                    precisions.append(0.0)
                    recalls.append(0.0)
                    fallback_rates.append(1.0)
                    continue

                tp = (pred_pos & (y_binary == 1)).sum()
                fp = (pred_pos & (y_binary == 0)).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / y_binary.sum() if y_binary.sum() > 0 else 0.0
                fallback_rate = 1.0 - pred_pos.mean()

                precisions.append(precision)
                recalls.append(recall)
                fallback_rates.append(fallback_rate)

            # Find threshold that achieves target fallback rate or maximizes F1
            precisions = np.array(precisions)
            recalls = np.array(recalls)
            fallback_rates = np.array(fallback_rates)

            # Prefer threshold close to target fallback rate
            idx = np.argmin(np.abs(fallback_rates - target_fallback_rate))
            best_thresh = thresholds[idx]

            # Ensure minimum precision
            if precisions[idx] < 0.5:
                # Relax threshold to improve precision
                high_prec_idx = np.where(precisions >= 0.5)[0]
                if len(high_prec_idx) > 0:
                    best_thresh = thresholds[high_prec_idx[0]]

            self.thresholds[cat] = np.clip(
                best_thresh, self.min_threshold, self.max_threshold
            )

        self._fitted = True
        LOG.info(
            "Learned adaptive thresholds for %d categories (target fallback=%.2f): %s",
            n_categories,
            target_fallback_rate,
            self.thresholds,
        )

    def get_threshold(self, category: str) -> float:
        """Get threshold for a specific category.

        Args:
            category: Category name

        Returns:
            Fallback threshold (0-1)
        """
        if not self._fitted:
            return self.default_threshold
        return self.thresholds.get(category, self.default_threshold)

    def should_fallback(self, category: str, prob: float) -> bool:
        """Determine if LLM fallback should be triggered.

        Args:
            category: Category name
            prob: Predicted probability for this category

        Returns:
            True if fallback should be triggered
        """
        threshold = self.get_threshold(category)
        return prob < threshold

    def get_expected_fallback_rate(self, probs: np.ndarray) -> float:
        """Estimate expected fallback rate on new data.

        Args:
            probs: Predicted probabilities (n_samples, n_categories)

        Returns:
            Estimated fraction of samples that would trigger fallback
        """
        if not self._fitted:
            # Use global threshold estimate
            max_probs = probs.max(axis=1)
            return (max_probs < self.default_threshold).mean()

        fallback_counts = []
        for i, cat in enumerate(self.category_names):
            thresh = self.thresholds.get(cat, self.default_threshold)
            fallback_counts.append(probs[:, i] < thresh)

        # A sample triggers fallback if any of its top categories are below threshold
        fallback_matrix = np.column_stack(fallback_counts)
        # Check top-2 categories per sample
        top2_idx = np.argsort(probs, axis=1)[:, -2:]
        fallback_for_top2 = fallback_matrix[
            np.arange(len(probs))[:, None], top2_idx
        ]
        return (fallback_for_top2.any(axis=1)).mean()

    def save(self, path: Path) -> None:
        """Save thresholds to disk.

        Args:
            path: Path to save threshold state (JSON format)
        """
        state = {
            'category_names': self.category_names,
            'thresholds': self.thresholds,
            'default_threshold': self.default_threshold,
            'min_threshold': self.min_threshold,
            'max_threshold': self.max_threshold,
            '_fitted': self._fitted,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            import json
            json.dump(state, f, indent=2)
        LOG.info("Saved adaptive thresholds to %s", path)

    @classmethod
    def load(cls, path: Path) -> 'AdaptiveThresholdManager':
        """Load thresholds from disk.

        Args:
            path: Path to saved threshold state

        Returns:
            Loaded AdaptiveThresholdManager instance
        """
        with open(path, 'r') as f:
            import json
            state = json.load(f)

        manager = cls(
            category_names=state['category_names'],
            default_threshold=state['default_threshold'],
            min_threshold=state['min_threshold'],
            max_threshold=state['max_threshold'],
        )
        manager.thresholds = state['thresholds']
        manager._fitted = state['_fitted']
        LOG.info("Loaded adaptive thresholds from %s", path)
        return manager