"""Predict ideology categories using the fine-tuned transformer classifier.

Loads ``models/transformer_ideology_classifier/final`` (a
``BertForSequenceClassification`` trained on the same 7-category label set)
and exposes a simple ``predict_proba`` function that returns per-category
probabilities.

The model and tokenizer are loaded lazily on first call and cached for the
process lifetime.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

LOG = logging.getLogger(__name__)

_model = None
_tokenizer = None
_id2label: Optional[Dict[int, str]] = None


def _default_model_dir() -> Path:
    try:
        return Path(__file__).resolve().parents[3] / "models" / "transformer_ideology_classifier" / "final"
    except Exception:
        return Path("models/transformer_ideology_classifier/final")


def _load(model_dir: Optional[Path] = None):
    global _model, _tokenizer, _id2label
    if _model is not None:
        return _model, _tokenizer, _id2label

    if model_dir is None:
        model_dir = _default_model_dir()

    if not model_dir.exists():
        raise FileNotFoundError(f"Transformer model not found at {model_dir}")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
    import torch

    LOG.info("Loading transformer ideology classifier from %s", model_dir)
    _tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    _model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    _model.eval()

    # Always use CPU to avoid GPU memory contention with the embedding
    # and zero-shot models that benefit more from GPU acceleration.
    _model = _model.to("cpu")
    LOG.info("Transformer classifier loaded on CPU (to preserve GPU for zero-shot)")

    # Read label mapping from parent config or model config
    parent_cfg = model_dir.parent / "config.json"
    if parent_cfg.exists():
        with open(parent_cfg) as f:
            cfg = json.load(f)
        _id2label = {int(k): v for k, v in cfg.get("id2label", {}).items()}
    else:
        _id2label = {int(k): v for k, v in _model.config.id2label.items()}

    return _model, _tokenizer, _id2label


def predict_proba(text: str, max_length: int = 512) -> Dict[str, float]:
    """Return ``{category: probability}`` for the input text."""
    import torch

    model, tokenizer, id2label = _load()
    device = next(model.parameters()).device

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
"""Predict ideology categories using the fine-tuned transformer classifier.

Loads ``models/transformer_ideology_classifier/final`` (a
``BertForSequenceClassification`` trained on the same 7-category label set)
and exposes a simple ``predict_proba`` function that returns per-category
probabilities.

The model and tokenizer are loaded lazily on first call and cached for the
process lifetime.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

LOG = logging.getLogger(__name__)

_model = None
_tokenizer = None
_id2label: Optional[Dict[int, str]] = None


def _default_model_dir() -> Path:
    try:
        return Path(__file__).resolve().parents[3] / "models" / "transformer_ideology_classifier" / "final"
    except Exception:
        return Path("models/transformer_ideology_classifier/final")


def _load(model_dir: Optional[Path] = None):
    global _model, _tokenizer, _id2label
    if _model is not None:
        return _model, _tokenizer, _id2label

    if model_dir is None:
        model_dir = _default_model_dir()

    if not model_dir.exists():
        raise FileNotFoundError(f"Transformer model not found at {model_dir}")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
    import torch

    LOG.info("Loading transformer ideology classifier from %s", model_dir)
    _tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    _model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    _model.eval()

    # Always use CPU to avoid GPU memory contention with the embedding
    # and zero-shot models that benefit more from GPU acceleration.
    _model = _model.to("cpu")
    LOG.info("Transformer classifier loaded on CPU (to preserve GPU for zero-shot)")

    # Read label mapping from parent config or model config
    parent_cfg = model_dir.parent / "config.json"
    if parent_cfg.exists():
        with open(parent_cfg) as f:
            cfg = json.load(f)
        _id2label = {int(k): v for k, v in cfg.get("id2label", {}).items()}
    else:
        _id2label = {int(k): v for k, v in _model.config.id2label.items()}

    return _model, _tokenizer, _id2label


def predict_proba(text: str, max_length: int = 512) -> Dict[str, float]:
    """Return ``{category: probability}`` for the input text."""
    import torch

    model, tokenizer, id2label = _load()
    device = next(model.parameters()).device

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
