"""Multi-transformer ensemble for improved classification diversity.

Runs multiple transformer models in parallel and combines their predictions
to reduce model-specific biases and improve robustness.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

LOG = logging.getLogger(__name__)


class MultiTransformerEnsemble:
    """Ensemble of multiple transformer classifiers.

    Combines predictions from multiple BERT-based models using:
    - Mean probability aggregation
    - Rank-based ensemble
    - Learned weights (if calibration data available)
    """

    def __init__(
        self,
        model_dirs: Optional[List[str]] = None,
        device_map: Optional[Dict[str, str]] = None,
    ):
        """Initialize multi-transformer ensemble.

        Args:
            model_dirs: List of model directory paths. If None, uses default
                Swedish BERT models.
            device_map: Optional mapping of model name to device.
        """
        self.model_dirs = model_dirs or [
            "models/transformer_ideology_classifier/final",
        ]
        self.device_map = device_map or {}
        self.models: Dict[str, dict] = {}
        self._loaded = False

    def load_models(self) -> None:
        """Lazy-load all transformer models."""
        if self._loaded:
            return

        for model_dir in self.model_dirs:
            try:
                self._load_single_model(model_dir)
            except Exception as e:
                LOG.warning("Failed to load transformer model from %s: %s", model_dir, e)

        if not self.models:
            raise RuntimeError("No transformer models could be loaded")
        self._loaded = True
        LOG.info("Loaded %d transformer models", len(self.models))

    def _load_single_model(self, model_dir: str) -> None:
        """Load a single transformer model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_path = model_dir
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()

        # Determine device
        device = self.device_map.get(model_dir, "cpu")
        model = model.to(device)

        # Load label mapping
        id2label = {}
        parent_cfg = __import__('pathlib').Path(model_dir).parent / "config.json"
        if parent_cfg.exists():
            with open(parent_cfg) as f:
                import json
                cfg = json.load(f)
                id2label = {int(k): v for k, v in cfg.get("id2label", {}).items()}
        else:
            id2label = {int(k): v for k, v in model.config.id2label.items()}

        model_name = __import__('pathlib').Path(model_dir).name
        self.models[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
            "id2label": id2label,
            "device": device,
            "dir": model_dir,
        }

    def predict_proba(
        self,
        text: str,
        max_length: int = 512,
        aggregation: str = "mean",
    ) -> Dict[str, float]:
        """Predict category probabilities using ensemble of transformers.

        Args:
            text: Input text to classify.
            max_length: Maximum token length.
            aggregation: How to combine predictions: 'mean', 'max', 'vote',
                or 'weighted' (requires calibration).

        Returns:
            Dictionary mapping category names to probabilities.
        """
        if not self._loaded:
            self.load_models()

        all_probs = []
        all_categories = set()

        for model_name, model_dict in self.models.items():
            try:
                probs = self._predict_single(
                    text, model_dict, max_length
                )
                all_probs.append(probs)
                all_categories.update(probs.keys())
            except Exception as e:
                LOG.warning("Prediction failed for model '%s': %s", model_name, e)

        if not all_probs:
            return {}

        # Align all predictions to same category set
        aligned_probs = []
        for probs in all_probs:
            aligned = {cat: probs.get(cat, 0.0) for cat in all_categories}
            aligned_probs.append(aligned)

        # Aggregate
        category_names = sorted(all_categories)
        agg_probs = self._aggregate_predictions(
            aligned_probs, category_names, aggregation
        )

        return agg_probs

    def _predict_single(
        self,
        text: str,
        model_dict: dict,
        max_length: int,
    ) -> Dict[str, float]:
        """Run prediction on a single model."""
        import torch

        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        id2label = model_dict["id2label"]
        device = model_dict["device"]

        inputs = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return {id2label[i]: float(p) for i, p in enumerate(probs)}

    def _aggregate_predictions(
        self,
        prob_dicts: List[Dict[str, float]],
        category_names: List[str],
        method: str,
    ) -> Dict[str, float]:
        """Aggregate predictions from multiple models.

        Args:
            prob_dicts: List of probability dictionaries from each model.
            category_names: Canonical category names.
            method: Aggregation method ('mean', 'max', 'vote', 'weighted').

        Returns:
            Aggregated probability dictionary.
        """
        # Build matrix: (n_models, n_categories)
        n_models = len(prob_dicts)
        n_cats = len(category_names)
        prob_matrix = np.zeros((n_models, n_cats), dtype=np.float64)

        for i, probs in enumerate(prob_dicts):
            for j, cat in enumerate(category_names):
                prob_matrix[i, j] = probs.get(cat, 0.0)

        if method == "mean":
            final_probs = prob_matrix.mean(axis=0)
        elif method == "max":
            final_probs = prob_matrix.max(axis=0)
        elif method == "vote":
            # Hard voting
            votes = prob_matrix.argmax(axis=1)
            final_probs = np.zeros(n_cats)
            for v in votes:
                final_probs[v] += 1
            final_probs = final_probs / final_probs.sum()
        elif method == "weighted":
            # Inverse entropy weighting: more confident models get higher weight
            weights = np.zeros(n_models)
            for i in range(n_models):
                entropy = -np.sum(prob_matrix[i] * np.log(prob_matrix[i] + 1e-9))
                weights[i] = 1.0 / (1.0 + entropy)
            weights = weights / weights.sum()
            final_probs = (prob_matrix * weights[:, None]).sum(axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        # Normalize to sum=1
        s = final_probs.sum()
        if s > 0:
            final_probs = final_probs / s

        return {cat: float(p) for cat, p in zip(category_names, final_probs)}

    def get_model_count(self) -> int:
        """Return number of loaded models."""
        return len(self.models)