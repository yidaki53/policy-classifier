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
import numpy as np
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


def predict_proba(
    text: str,
    max_length: int = 512,
    window_strategy: str = "truncate",
    window_overlap: float = 0.1,
    aggregation: str = "mean",
) -> Dict[str, float]:
    """Return ``{category: probability}`` for the input text.

    Args:
        text: Input text to classify.
        max_length: Maximum token length for BERT window.
        window_strategy: One of 'truncate', 'sliding', or 'hierarchical'.
            - 'truncate': Use first max_length tokens (default, original behavior).
            - 'sliding': Use overlapping windows for texts longer than max_length.
            - 'hierarchical': Sentence-level pooling then document-level aggregation.
        window_overlap: Fraction of window to overlap for sliding mode (0-0.5).
        aggregation: How to aggregate window predictions: 'mean', 'max', or 'vote'.

    Returns:
        Dictionary mapping category names to probabilities.
    """
    import torch

    model, tokenizer, id2label = _load()
    device = next(model.parameters()).device

    # Tokenize to check length
    tokens = tokenizer(text, truncation=False, return_tensors="pt")
    input_len = tokens["input_ids"].shape[1]

    if input_len <= max_length or window_strategy == "truncate":
        # Simple truncation (original behavior)
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

    # Extended window strategies for long texts
    if window_strategy == "sliding":
        return _predict_sliding_window(
            text, model, tokenizer, id2label, device,
            max_length, window_overlap, aggregation
        )
    elif window_strategy == "hierarchical":
        return _predict_hierarchical(
            text, model, tokenizer, id2label, device,
            max_length
        )
    else:
        raise ValueError(f"Unknown window_strategy: {window_strategy}")


def _predict_sliding_window(
    text: str,
    model,
    tokenizer,
    id2label: Dict[int, str],
    device,
    max_length: int,
    overlap: float,
    aggregation: str,
) -> Dict[str, float]:
    """Predict using sliding windows with overlap for long texts."""
    import torch

    # Tokenize without truncation
    tokens = tokenizer(text, truncation=False, return_tensors="pt")
    input_ids = tokens["input_ids"][0]
    n_tokens = len(input_ids)

    # Compute stride
    stride = int(max_length * (1 - overlap))
    if stride < 1:
        stride = 1

    # Generate windows
    windows = []
    start = 0
    while start < n_tokens:
        end = min(start + max_length, n_tokens)
        window_ids = input_ids[start:end]
        windows.append(window_ids)
        if end >= n_tokens:
            break
        start += stride

    # Predict for each window
    all_probs = []
    for window_ids in windows:
        inputs = {
            "input_ids": window_ids.unsqueeze(0).to(device),
            "attention_mask": torch.ones_like(window_ids.unsqueeze(0)).to(device),
        }
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        all_probs.append(probs)

    # Aggregate
    all_probs = np.array(all_probs)
    if aggregation == "mean":
        final_probs = all_probs.mean(axis=0)
    elif aggregation == "max":
        final_probs = all_probs.max(axis=0)
    elif aggregation == "vote":
        # Hard voting: pick class with most votes
        votes = all_probs.argmax(axis=1)
        final_probs = np.zeros(all_probs.shape[1])
        for v in votes:
            final_probs[v] += 1
        final_probs = final_probs / final_probs.sum()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return {id2label[i]: float(p) for i, p in enumerate(final_probs)}


def _predict_hierarchical(
    text: str,
    model,
    tokenizer,
    id2label: Dict[int, str],
    device,
    max_length: int,
) -> Dict[str, float]:
    """Hierarchical prediction: sentence-level then document-level pooling."""
    import torch
    import re

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        # Fallback to truncation
        return predict_proba(text, max_length=max_length, window_strategy="truncate")

    # Sentence-level predictions
    sentence_probs = []
    for sent in sentences:
        inputs = tokenizer(
            sent,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        sentence_probs.append(probs)

    # Document-level: mean pooling of sentence probabilities
    final_probs = np.mean(sentence_probs, axis=0)
    return {id2label[i]: float(p) for i, p in enumerate(final_probs)}