"""Deterministic, rule-first scorer for classifying motions into categories.

This is intentionally simple and transparent: keywords and regexes are loaded
from `definitions/political_spectrum.yaml`. Each matching keyword or regex adds
to a raw score for the category; scores are normalized to provide relative
weights. The function returns `ClassificationResult` objects.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional
from datetime import datetime

import yaml
import joblib
try:
    import numpy as np
except Exception:
    np = None

from swedish_parliament_policy_classifier.models.models import (
    CategoryDef,
    ClassificationResult,
)
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher
from swedish_parliament_policy_classifier.definitions.loader import load_verified_definitions

LOG = logging.getLogger(__name__)


def load_definitions(yaml_path: Union[str, Path, None] = None) -> Dict[str, CategoryDef]:
    """Load category definitions via the verified, immutable loader.

    When yaml_path is None the bundled political_spectrum.yaml is used and
    integrity is checked against its embedded checksum.  Passing an explicit
    path skips the checksum check (useful in tests).
    """
    if yaml_path is None:
        return load_verified_definitions()

    # Explicit path: load directly without integrity check (caller's responsibility).
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    categories: Dict[str, CategoryDef] = {}
    for entry in raw.get("categories", []):
        cat = CategoryDef(**{k: v for k, v in entry.items() if k in CategoryDef.model_fields})
        categories[cat.name] = cat

    return categories


def score_motion(
    motion_id: str,
    text: str,
    categories: Dict[str, CategoryDef],
    embedding_matcher: Optional[EmbeddingMatcher] = None,
    embedding_weight: float = 0.3,
    embedding_threshold: float = 0.0,
    supervised_model_dir: Optional[Union[str, Path]] = None,
    supervised_threshold: float = 0.5,
    supervised_trigger: float = 0.15,
    use_supervised: bool = True,
) -> List[ClassificationResult]:
    text_l = (text or "").lower()
    scores: Dict[str, float] = {}
    matches: Dict[str, List[str]] = {}

    for name, cat in categories.items():
        score = 0.0
        matched: List[str] = []

        for kw in cat.keywords or []:
            if kw and kw.lower() in text_l:
                score += 1.0
                matched.append(kw)

        for rx in cat.regexes or []:
            try:
                if rx and re.search(rx, text_l):
                    score += 1.0
                    matched.append(rx)
            except re.error:
                # ignore bad regex in definitions
                continue

        scores[name] = score
        matches[name] = matched

    total = sum(scores.values())
    # If embedding matcher provided, compute semantic scores and combine.
    emb_map: Dict[str, float] = {}
    if embedding_matcher is not None and embedding_weight > 0:
        try:
            cat_embs = embedding_matcher.build_category_embeddings(categories)
            # request scores for all categories
            emb_matches = embedding_matcher.match(text, cat_embs, top_k=len(categories))
            emb_map = {name: float(score) for name, score in emb_matches}
            # attach embedding notes to matched rules when above threshold
            for name, score in emb_map.items():
                if score >= embedding_threshold:
                    matches.setdefault(name, []).append(f"embedding:{score:.3f}")
        except Exception as e:  # defensive: embedding backend may fail
            LOG.warning("Embedding matcher failed: %s", e)

    # Normalize keyword/raw scores and embedding scores, then combine by weight.
    keyword_sum = sum(scores.values())
    keyword_norm = {k: (v / keyword_sum if keyword_sum > 0 else 0.0) for k, v in scores.items()}

    emb_sum = sum(emb_map.values()) if emb_map else 0.0
    emb_norm = {k: (emb_map.get(k, 0.0) / emb_sum if emb_sum > 0 else 0.0) for k in categories.keys()}

    combined_norm = {
        k: ( (1.0 - embedding_weight) * keyword_norm.get(k, 0.0) + embedding_weight * emb_norm.get(k, 0.0) )
        for k in categories.keys()
    }

    # renormalize to ensure weights sum to 1 (if anything positive)
    total_combined = sum(combined_norm.values())
    if total_combined > 0:
        for k in combined_norm:
            combined_norm[k] = combined_norm[k] / total_combined

    # Optional supervised fallback: only used when deterministic+embedding
    # confidence is low (max combined weight < supervised_trigger). The
    # supervised model is expected at `models/supervised_clf.joblib` and the
    # MultiLabelBinarizer at `models/supervised_mlb.joblib` under the
    # package root or a provided `supervised_model_dir`.
    if use_supervised:
        try:
            # resolve default model dir
            if supervised_model_dir is None:
                supervised_model_dir = Path(__file__).resolve().parents[1] / "models"
            model_dir = Path(supervised_model_dir)
            clf_path = model_dir / "supervised_clf.joblib"
            mlb_path = model_dir / "supervised_mlb.joblib"
            if clf_path.exists() and mlb_path.exists():
                clf = joblib.load(str(clf_path))
                mlb = joblib.load(str(mlb_path))

                # Predict probabilities for this single motion text
                try:
                    probs = clf.predict_proba([text])
                except Exception:
                    probs = None

                if probs is not None:
                    # handle array-like probs
                    try:
                        prob_vec = probs[0]
                    except Exception:
                        prob_vec = probs

                    # Build mapping label -> prob
                    try:
                        labels = list(mlb.classes_)
                    except Exception:
                        labels = list(range(len(prob_vec)))

                    sup_map = {str(l): float(p) for l, p in zip(labels, prob_vec)}

                    max_combined = max(combined_norm.values()) if combined_norm else 0.0
                    if max_combined < supervised_trigger:
                        # Use supervised predictions as fallback: select labels
                        # with probability >= supervised_threshold
                        selected = {k: v for k, v in sup_map.items() if v >= supervised_threshold}
                        if selected:
                            ssum = sum(selected.values())
                            # normalize selected probs to sum to 1 for weights
                            for k in selected:
                                selected[k] = selected[k] / ssum if ssum > 0 else 0.0

                            # override combined_norm to reflect supervised choices
                            combined_norm = {k: selected.get(k, 0.0) for k in categories.keys()}
                            # annotate matched_rules for traceability
                            for k, p in sup_map.items():
                                if p >= supervised_threshold:
                                    matches.setdefault(k, []).append(f"supervised:{p:.3f}")
                            # record that supervised was used in version
                            classifier_version += "+sup"
                            try:
                                classifier_version += f"({clf_path.name})"
                            except Exception:
                                classifier_version += "(unknown)"
        except Exception as e:
            LOG.warning("Supervised fallback failed: %s", e)

    results: List[ClassificationResult] = []
    base_version = "0.1.0"
    classifier_version = base_version
    if embedding_matcher is not None:
        classifier_version = base_version + "+emb"
        try:
            classifier_version += f"({embedding_matcher.model_name})"
        except Exception:
            classifier_version += "(fallback)"

    for name, raw_score in scores.items():
        # represent raw_score augmented slightly by embedding (for traceability)
        augmented_raw = float(raw_score) + (embedding_weight * float(emb_map.get(name, 0.0)))
        normalized = float(combined_norm.get(name, 0.0))
        results.append(
            ClassificationResult(
                motion_id=motion_id,
                category=name,
                raw_score=augmented_raw,
                normalized_weight=normalized,
                matched_rules=matches.get(name, []),
                classifier_version=classifier_version,
                created_at=datetime.utcnow(),
            )
        )

    return results
