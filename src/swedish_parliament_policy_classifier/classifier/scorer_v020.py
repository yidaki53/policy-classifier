"""Upgraded scorer: lemmatized keyword matching + semantic embeddings + stance awareness.

This scorer replaces the naive substring matcher with:
1. spaCy lemmatization for morphological matching (skatter -> skatt)
2. Sentence-transformer semantic similarity for synonym/paraphrase coverage
3. Stance-direction heuristics for concept+modifier pairs (höja vs sänka skatter)
4. Audit trail recording which signals fired for each classification
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from datetime import datetime, timezone

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
from swedish_parliament_policy_classifier.nlp.preprocess import init_spacy, preprocess_text

LOG = logging.getLogger(__name__)

# Lazy-loaded spaCy pipeline
_spacy_nlp = None


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        _spacy_nlp = init_spacy(model="sv_core_news_sm", install=False)
    return _spacy_nlp


# Stance direction rules: concept -> (left_indicators, right_indicators)
# When a concept keyword is found, check for directional modifiers nearby.
_STANCE_RULES: List[Tuple[str, List[str], List[str]]] = [
    # Tax direction
    ("skatt", ["höja", "höjda", "höjning", "progressiv", "öka", "mer", "högre", "tillbaka"],
              ["sänka", "sänkta", "sänkning", "lägre", "minska", "ned", "bort", "avskaffa"]),
    # Welfare direction
    ("välfärd", ["stärka", "bygga ut", "utbyggnad", "öka", "mer", "förbättra", "skydda", "rädda"],
                ["minska", "nedmontera", "privatisera", "marknadsorientera", "effektivisera bort", "spar"]),
    # Market direction
    ("marknad", ["reglera", "kontrollera", "begränsa", "övervaka", "tillsyn", "lagstifta", "styr"],
                ["avreglera", "frigöra", "liberalisera", "privatisera", "konkurrensutsätta", "marknadsöppna"]),
    # State size direction
    ("stat", ["stärka", "bygga ut", "förstärka", "mer", "ökad", "utvidga", "utöka"],
             ["minska", "krympa", "smalare", "effektivisera", "dereglera", "avveckla", "slimma"]),
    # Immigration direction
    ("invandr", ["öka", "fler", "öppna", "välkomna", "mottag", "asyl", "rätt", "human"],
                ["minska", "begränsa", "stoppa", "stänga", "återvandring", "utvisa", "färre", "stram"]),
    # Environment direction
    ("miljö", ["stärka", "skydda", "bevara", "klimat", "hållbar", "grön", "biologisk mångfald"],
              ["effektivisera", "förenkla", "avreglera", "tillväxt", "jobb", "konkurrenskraft"]),
]


def load_definitions(yaml_path: Union[str, Path, None] = None) -> Dict[str, CategoryDef]:
    """Load category definitions via the verified, immutable loader."""
    if yaml_path is None:
        return load_verified_definitions()

    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    categories: Dict[str, CategoryDef] = {}
    for entry in raw.get("categories", []):
        cat = CategoryDef(**{k: v for k, v in entry.items() if k in CategoryDef.model_fields})
        categories[cat.name] = cat

    return categories


def _build_lemma_kw_index(categories: Dict[str, CategoryDef]) -> Dict[str, List[Tuple[str, str]]]:
    """Build an index mapping lemma -> list of (category_name, original_keyword).

    This allows O(1) lookup of which categories care about a given lemma.
    """
    index: Dict[str, List[Tuple[str, str]]] = {}
    nlp = _get_spacy()

    for name, cat in categories.items():
        for kw in cat.keywords or []:
            if not kw:
                continue
            # Lemmatize keyword
            if nlp is not None:
                doc = nlp(kw.lower())
                lemmas = [t.lemma_.lower() for t in doc if not t.is_space and not t.is_punct]
                lemma_key = " ".join(lemmas)
            else:
                lemma_key = kw.lower()
            index.setdefault(lemma_key, []).append((name, kw))
    return index


def _detect_stance(text_lower: str, lemma_tokens: List[str]) -> Dict[str, float]:
    """Return stance adjustments: {category: delta_score}.

    For each stance rule, if the concept appears, scan nearby tokens for
    left/right directional indicators and apply a small boost to the matching side.
    """
    adjustments: Dict[str, float] = {}
    text = text_lower
    tokens = " ".join(lemma_tokens)

    for concept, left_inds, right_inds in _STANCE_RULES:
        if concept not in text and concept not in tokens:
            continue

        # Check for directional indicators anywhere in the text (simple heuristic)
        left_found = any(ind in text for ind in left_inds)
        right_found = any(ind in text for ind in right_inds)

        if left_found and not right_found:
            # Boost left-side categories
            adjustments["far_left"] = adjustments.get("far_left", 0.0) + 0.5
            adjustments["left"] = adjustments.get("left", 0.0) + 0.5
            adjustments["centre_left"] = adjustments.get("centre_left", 0.0) + 0.3
        elif right_found and not left_found:
            # Boost right-side categories
            adjustments["far_right"] = adjustments.get("far_right", 0.0) + 0.5
            adjustments["right"] = adjustments.get("right", 0.0) + 0.5
            adjustments["centre_right"] = adjustments.get("centre_right", 0.0) + 0.3

    return adjustments


def score_motion(
    motion_id: str,
    text: str,
    categories: Dict[str, CategoryDef],
    embedding_matcher: Optional[EmbeddingMatcher] = None,
    embedding_weight: float = 0.35,
    embedding_threshold: float = 0.0,
    stance_weight: float = 0.15,
    supervised_model_dir: Optional[Union[str, Path]] = None,
    supervised_threshold: float = 0.5,
    supervised_trigger: float = 0.15,
    use_supervised: bool = True,
) -> List[ClassificationResult]:
    """Classify a motion using lemmatized keywords, regex, semantic embeddings, and stance heuristics.

    Parameters
    ----------
    embedding_weight: weight of semantic embedding scores in the combined score.
    stance_weight: weight of stance-direction heuristics in the combined score.
    The remaining (1 - embedding_weight - stance_weight) goes to keyword+regex.
    """
    text_l = (text or "").lower()
    if not text_l:
        text_l = ""

    # Stage 0: spaCy preprocessing
    nlp = _get_spacy()
    if nlp is not None:
        preproc = preprocess_text(text, nlp=nlp, remove_stopwords=False, lemmatize=True, normalize=True)
        lemma_text = " ".join(preproc["lemmas"])
        lemma_tokens = preproc["lemmas"]
    else:
        lemma_text = text_l
        lemma_tokens = text_l.split()

    scores: Dict[str, float] = {}
    matches: Dict[str, List[str]] = {}

    # Stage 1a: Lemmatized keyword matching
    kw_index = _build_lemma_kw_index(categories)
    for lemma_key, cat_kw_pairs in kw_index.items():
        if lemma_key in lemma_text:
            for cat_name, orig_kw in cat_kw_pairs:
                scores[cat_name] = scores.get(cat_name, 0.0) + 1.0
                matches.setdefault(cat_name, []).append(f"lemma:{orig_kw}")

    # Stage 1b: Regex matching (on raw lowercased text)
    for name, cat in categories.items():
        for rx in cat.regexes or []:
            try:
                if rx and re.search(rx, text_l):
                    scores[name] = scores.get(name, 0.0) + 1.0
                    matches.setdefault(name, []).append(f"regex:{rx}")
            except re.error:
                continue

    # Stage 2: Stance-direction heuristics
    stance_scores = _detect_stance(text_l, lemma_tokens)
    for cat, delta in stance_scores.items():
        if cat in categories:
            scores[cat] = scores.get(cat, 0.0) + delta
            matches.setdefault(cat, []).append(f"stance:+{delta:.1f}")

    # Stage 3: Semantic embedding similarity
    emb_map: Dict[str, float] = {}
    if embedding_matcher is not None and embedding_weight > 0:
        try:
            cat_embs = embedding_matcher.build_category_embeddings(categories)
            emb_matches = embedding_matcher.match(text, cat_embs, top_k=len(categories))
            emb_map = {name: float(score) for name, score in emb_matches}
            for name, score in emb_map.items():
                if score >= embedding_threshold:
                    matches.setdefault(name, []).append(f"embedding:{score:.3f}")
        except Exception as e:
            LOG.warning("Embedding matcher failed: %s", e)

    # Normalize each signal independently
    keyword_sum = sum(scores.values())
    keyword_norm = {k: (v / keyword_sum if keyword_sum > 0 else 0.0) for k, v in scores.items()}

    emb_sum = sum(emb_map.values()) if emb_map else 0.0
    emb_norm = {k: (emb_map.get(k, 0.0) / emb_sum if emb_sum > 0 else 0.0) for k in categories.keys()}

    stance_sum = sum(stance_scores.values()) if stance_scores else 0.0
    stance_norm = {k: (stance_scores.get(k, 0.0) / stance_sum if stance_sum > 0 else 0.0) for k in categories.keys()}

    # Combine signals with weights
    kw_w = max(0.0, 1.0 - embedding_weight - stance_weight)
    combined_norm = {
        k: (
            kw_w * keyword_norm.get(k, 0.0)
            + embedding_weight * emb_norm.get(k, 0.0)
            + stance_weight * stance_norm.get(k, 0.0)
        )
        for k in categories.keys()
    }

    # Renormalize to sum to 1
    total_combined = sum(combined_norm.values())
    if total_combined > 0:
        for k in combined_norm:
            combined_norm[k] = combined_norm[k] / total_combined

    # Build classifier version string
    base_version = "0.2.0"
    classifier_version = base_version
    signals = []
    if nlp is not None:
        signals.append("spacy")
    if embedding_matcher is not None and emb_map:
        signals.append("emb")
        try:
            signals.append(f"({embedding_matcher.model_name})")
        except Exception:
            signals.append("(unknown)")
    if stance_scores:
        signals.append("stance")
    classifier_version += "+" + "+".join(signals) if signals else ""

    # Optional supervised fallback
    if use_supervised:
        try:
            if supervised_model_dir is None:
                supervised_model_dir = Path(__file__).resolve().parents[1] / "models"
            model_dir = Path(supervised_model_dir)
            clf_path = model_dir / "supervised_clf.joblib"
            mlb_path = model_dir / "supervised_mlb.joblib"
            if not clf_path.exists():
                alt = Path(__file__).resolve().parents[3] / "models"
                if alt.exists():
                    clf_path = alt / "supervised_clf.joblib"
                    mlb_path = alt / "supervised_mlb.joblib"

            if clf_path.exists() and mlb_path.exists():
                clf = joblib.load(str(clf_path))
                mlb = joblib.load(str(mlb_path))
                try:
                    probs = clf.predict_proba([text])
                except Exception:
                    probs = None

                if probs is not None:
                    try:
                        prob_vec = probs[0]
                    except Exception:
                        prob_vec = probs
                    try:
                        labels = list(mlb.classes_)
                    except Exception:
                        labels = list(range(len(prob_vec)))

                    sup_map = {str(l): float(p) for l, p in zip(labels, prob_vec)}
                    max_combined = max(combined_norm.values()) if combined_norm else 0.0
                    if max_combined < supervised_trigger:
                        selected = {k: v for k, v in sup_map.items() if v >= supervised_threshold}
                        if selected:
                            ssum = sum(selected.values())
                            for k in selected:
                                selected[k] = selected[k] / ssum if ssum > 0 else 0.0
                            combined_norm = {k: selected.get(k, 0.0) for k in categories.keys()}
                            for k, p in sup_map.items():
                                if p >= supervised_threshold:
                                    matches.setdefault(k, []).append(f"supervised:{p:.3f}")
                            classifier_version += "+sup"
                            try:
                                classifier_version += f"({clf_path.name})"
                            except Exception:
                                classifier_version += "(unknown)"
        except Exception as e:
            LOG.warning("Supervised fallback failed: %s", e)

    # Assemble results
    results: List[ClassificationResult] = []
    for name in categories.keys():
        raw_score = float(scores.get(name, 0.0))
        normalized = float(combined_norm.get(name, 0.0))
        results.append(
            ClassificationResult(
                motion_id=motion_id,
                category=name,
                raw_score=raw_score,
                normalized_weight=normalized,
                matched_rules=matches.get(name, []),
                classifier_version=classifier_version,
                created_at=datetime.now(timezone.utc),
            )
        )

    return results
