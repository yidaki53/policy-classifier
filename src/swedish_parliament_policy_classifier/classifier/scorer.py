"""Refactored classification pipeline extracted from the legacy scorer.

This module isolates extraction, signal computation and combination into a
single place so callers can import the refined pipeline without depending on
the large legacy `scorer.py` implementation.
"""
import re
import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union, Tuple, cast
from datetime import datetime, timezone
from fractions import Fraction

import joblib
try:
    import numpy as np
except Exception:
    np = None  # optional dependency

from swedish_parliament_policy_classifier.models.models import (
    CategoryDef,
    ClassificationResult,
    ClassificationProvenance,
)
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher
from swedish_parliament_policy_classifier.nlp.preprocess import init_spacy, preprocess_text
from swedish_parliament_policy_classifier.nlp.rhetorical_detector import detect_rhetorical_patterns
from swedish_parliament_policy_classifier.nlp.topic_modeler import get_topic_features
from swedish_parliament_policy_classifier.classifier.ensemble import (
    build_feature_vector,
    predict_with_meta_classifier,
)
from swedish_parliament_policy_classifier.classifier.llm_judge import (
    llm_judge,
    should_use_llm_fallback,
)
from swedish_parliament_policy_classifier.classifier.signal_combinator import (
    SignalCombinator,
    apply_rhetorical_adjustments,
)
from swedish_parliament_policy_classifier.classifier.signals import extract_signal_output
from swedish_parliament_policy_classifier.classifier.model_provider import ModelArtifactProvider
from swedish_parliament_policy_classifier.classifier.signal_executor import (
    CallableSignalProvider,
    SignalContext,
    SignalExecutor,
)

LOG = logging.getLogger(__name__)
_MODEL_PROVIDER = ModelArtifactProvider()

# Lazy-loaded spaCy pipeline
_spacy_nlp = None


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        _spacy_nlp = init_spacy(model="sv_core_news_sm", install=False)
    return _spacy_nlp


def _extract_party_policy_text(text: str, party: Optional[str] = None) -> str:
    if not text:
        return ""

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text

    party_markers = [
        "vi anser", "vi vill", "vi föreslår", "vi står", "vi kräver",
        "vi välkomnar", "vi stöder", "vi avvisar", "vi avstår",
        "motionärerna anser", "föreslår motionärerna", "bör",
        "skall", "ska", "motion till riksdagen", "förslag till riksdagsbeslut",
        "riksdagen ställer sig bakom", "riksdagen avslår",
    ]
    gov_markers = [
        "regeringen föreslår", "regeringen gör", "regeringen har",
        "regeringen vill", "regeringen avser", "regeringen bedömer",
        "regeringen anser", "regeringens förslag", "regeringens proposition",
        "regeringens bedömning", "regeringen har i propositionen",
        "i propositionen anförs", "i utredningen", "i betänkande",
        "utredningen föreslår", "kommittén föreslår",
        "utredaren föreslår", "regeringens förslag innebär",
        r"prop.\s*\d{4}/\d{2}:\d+",
    ]

    kept: List[str] = []
    for s in sentences:
        s_lower = s.lower().strip()
        if len(s_lower) < 20:
            continue
        if any(s_lower.startswith(m) for m in gov_markers):
            continue
        if any(re.search(m, s_lower) for m in gov_markers):
            continue
        if any(m in s_lower for m in party_markers):
            kept.append(s)
            continue
        strong_policy = ["bör", "skall", "ska", "måste", "krävs", "behöver",
                         "föreslås", "föreslår", "avslås", "avslår", "stöds",
                         "stöder", "upphävs", "ändras", "införs", "avskaffas"]
        if any(f" {m} " in f" {s_lower} " for m in strong_policy):
            kept.append(s)
            continue
    return " ".join(kept)


def _sentence_stance(s: str) -> str:
    s_lower = s.lower().strip()
    opponent_patterns = [
        r'\bni\s+(säger|vill|föreslår|anser|menar|kräver|tycker)\b',
        r'\bdu\s+(säger|vill|föreslår|anser|menar)\b',
        r'\bni\s+(har|gör|står\s+för)\b',
        r'\bjimmie\s+åkesson\b',
        r'\bjohan\s+nissinen\b',
        r'\bsverigedemokraterna\s+(har|vill|föreslår|anser|menar)\b',
        r'\bmoderaterna\s+(har|vill|föreslår|anser)\b',
        r'\bvänsterpartiet\s+(har|vill|föreslår|anser)\b',
        r'\bsocialdemokraterna\s+(har|vill|föreslår|anser)\b',
        r'\b[a-zåäö]+\s+\(\s*[A-ZÅÄÖ][a-zåäö]+\s*\)\s+(säger|vill|talar\s+om|menar)\b',
        r'\bsom\s+(jimmy|johan|richard)\s+(säger|talar\s+om)\b',
        r'\bni\s+(har\s+velat|vill\s+ha|föreslår\s+att)\b',
    ]
    rhetorical_patterns = [
        r'\?\s*$',
        r'\bvilka\s+(skulle|hade|är)\b',
        r'\bvarför\s+(ska|vill|ger)\b',
        r'\bhur\s+(hänger|kan|ska)\b',
        r'\btycker\s+ni\s+att\b',
        r'\boroa\s+er\s+inte\b',
    ]
    own_patterns = [
        r'\b(vi|jag)\s+(anser|vill|föreslår|kräver|stöder|avvisar|möter|tycker)\b',
        r'\b(vi|jag)\s+(måste|behöver|bör|ska|skall)\b',
        r'\b(vi|jag)\s+(välkomnar|står|förespråkar|argumenterar)\b',
        r'\bregeringen\s+(bör|måste|ska|behöver)\b',
        r'\bdet\s+(är|bör|måste|ska)\s+(viktigt|avgörande|nödvändigt)\b',
        r'\b(vi|jag)\s+(ser|uppfattar|har\s+alltid|har\s+aldrig)\b',
    ]
    for p in rhetorical_patterns:
        if re.search(p, s_lower):
            for op in own_patterns:
                if re.search(op, s_lower):
                    return "own_position"
            return "rhetorical_challenge"
    for p in opponent_patterns:
        if re.search(p, s_lower):
            for op in own_patterns:
                if re.search(op, s_lower):
                    return "own_position"
            return "opponent_report"
    for p in own_patterns:
        if re.search(p, s_lower):
            return "own_position"
    return "neutral"


def _extract_speech_argumentative_text(text: str, max_chars: int = 5000) -> str:
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text[:max_chars]
    tagged = [(s, _sentence_stance(s)) for s in sentences if len(s.strip()) > 15]
    own = [s for s, t in tagged if t == "own_position"]
    neutral = [s for s, t in tagged if t == "neutral"]
    opponent = [s for s, t in tagged if t == "opponent_report"]
    result = []
    chars = 0
    for s in own + neutral + opponent:
        if chars + len(s) + 1 > max_chars:
            break
        result.append(s)
        chars += len(s) + 1
    output = " ".join(result)
    if not output and sentences:
        output = text[:max_chars]
    return output


_RHETORICAL_WEIGHTS = None




# Cache the lemma keyword index so it is built only once per process
_LEMMA_KW_INDEX: Optional[Dict[str, List[Tuple[str, str]]]] = None
_LEMMA_KW_INDEX_KEY: Optional[Tuple[Tuple[str, Tuple[str, ...]], ...]] = None


def _build_lemma_kw_index(categories: Dict[str, CategoryDef]) -> Dict[str, List[Tuple[str, str]]]:
    global _LEMMA_KW_INDEX, _LEMMA_KW_INDEX_KEY
    cache_key = tuple(
        sorted((name, tuple(cat.keywords or [])) for name, cat in categories.items())
    )
    if _LEMMA_KW_INDEX is not None and _LEMMA_KW_INDEX_KEY == cache_key:
        return _LEMMA_KW_INDEX
    index: Dict[str, List[Tuple[str, str]]] = {}
    nlp = _get_spacy()
    for name, cat in categories.items():
        for kw in cat.keywords or []:
            if not kw:
                continue
            if nlp is not None:
                doc = nlp(kw.lower())
                lemmas = [t.lemma_.lower() for t in doc if not t.is_space and not t.is_punct]
                lemma_key = " ".join(lemmas)
            else:
                lemma_key = kw.lower()
            index.setdefault(lemma_key, []).append((name, kw))
    _LEMMA_KW_INDEX = index
    _LEMMA_KW_INDEX_KEY = cache_key
    return index


def score_motion(
    motion_id: str,
    text: str,
    categories: Dict[str, CategoryDef],
    party: Optional[str] = None,
    embedding_matcher: Optional[EmbeddingMatcher] = None,
    embedding_weight: float = 0.40,
    embedding_threshold: float = 0.0,
    zero_shot_weight: float = 0.40,
    party_prior_weight: float = 0.00,
    zero_shot_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    use_zero_shot: bool = True,
    supervised_model_dir: Optional[Union[str, Path]] = None,
    supervised_threshold: float = 0.5,
    supervised_trigger: float = 0.15,
    use_supervised: bool = True,
    topic_distributions: Optional[Dict[str, List[float]]] = None,
    meta_clf: Optional[Dict] = None,
    llm_threshold: float = 0.30,
    llm_max_text_len: int = 2000,
    skip_policy_extraction: bool = False,
    use_speech_preprocessing: bool = False,
    use_ollama: bool = False,
    ollama_weight: float = 0.35,
) -> List[ClassificationResult]:
    # The implementation mirrors the legacy scorer behaviour but is
    # extracted here to make the pipeline a separable module for testing.
    if skip_policy_extraction:
        if use_speech_preprocessing:
            policy_text = _extract_speech_argumentative_text(text)
        else:
            policy_text = text
    else:
        policy_text = _extract_party_policy_text(text, party=party)
    text_l = (policy_text or "").lower()
    if not text_l:
        text_l = ""

    MAX_SPA_CY = 500_000
    nlp = _get_spacy()
    proc_source = (policy_text or "")[:MAX_SPA_CY]
    if nlp is not None and len(proc_source) <= MAX_SPA_CY:
        has_keywords = any(cat.keywords for cat in categories.values())
        if has_keywords:
            preproc = preprocess_text(proc_source, nlp=nlp, remove_stopwords=False, lemmatize=True, normalize=True)
            lemmas = cast(List[str], preproc.get("lemmas", []))
            lemma_text = " ".join(lemmas)
        else:
            lemma_text = text_l[:MAX_SPA_CY]
    elif nlp is not None:
        preproc = preprocess_text(proc_source, nlp=nlp, remove_stopwords=False, lemmatize=True, normalize=True)
        lemmas = cast(List[str], preproc.get("lemmas", []))
        lemma_text = " ".join(lemmas)
    else:
        lemma_text = text_l[:MAX_SPA_CY]

    scores: Dict[str, float] = {}
    matches: Dict[str, List[str]] = {}

    kw_index = _build_lemma_kw_index(categories)
    for lemma_key, cat_kw_pairs in kw_index.items():
        if lemma_key in lemma_text:
            for cat_name, orig_kw in cat_kw_pairs:
                scores[cat_name] = scores.get(cat_name, 0.0) + 1.0
                matches.setdefault(cat_name, []).append(f"lemma:{orig_kw}")

    for name, cat in categories.items():
        for rx in cat.regexes or []:
            try:
                if rx and re.search(rx, text_l):
                    scores[name] = scores.get(name, 0.0) + 1.0
                    matches.setdefault(name, []).append(f"regex:{rx}")
            except re.error:
                continue

    failed_signals: List[str] = []
    signal_providers = []
    if embedding_matcher is not None and embedding_weight > 0:
        def compute_embedding(context):
            if embedding_matcher.cached_embeddings is None:
                embedding_matcher.cached_embeddings = embedding_matcher.build_category_embeddings(categories)
            if embedding_matcher.cached_embeddings is None:
                raise ValueError("embedding_matcher.cached_embeddings is None")
            emb_matches = embedding_matcher.match(context.text, embedding_matcher.cached_embeddings, top_k=len(categories))
            if emb_matches is None:
                raise ValueError("embedding_matcher.match() returned None")
            return {name: float(score) for name, score in emb_matches}

        signal_providers.append(CallableSignalProvider("embedding", compute_embedding))

    if use_zero_shot and zero_shot_weight > 0:
        def compute_zero_shot(context):
            if use_speech_preprocessing:
                from swedish_parliament_policy_classifier.nlp.zero_shot_values import zero_shot_score_speech_aware
                return zero_shot_score_speech_aware(context.text, model_name=zero_shot_model)
            else:
                from swedish_parliament_policy_classifier.nlp.zero_shot_values import zero_shot_score
                return zero_shot_score(context.text, model_name=zero_shot_model)

        signal_providers.append(CallableSignalProvider("zero_shot", compute_zero_shot))

    if use_speech_preprocessing and use_ollama:
        def compute_ollama(context):
            from swedish_parliament_policy_classifier.nlp.ollama_classifier import classify_speech_with_cache
            return classify_speech_with_cache(context.text, speech_id=context.item_id, cache=None) or {}

        signal_providers.append(CallableSignalProvider("ollama", compute_ollama))

    def compute_transformer(context):
        from swedish_parliament_policy_classifier.classifier.transformer_predict import predict_proba as _bert_predict
        return _bert_predict(context.text[:2500])

    signal_providers.append(CallableSignalProvider("transformer", compute_transformer))

    signal_run = SignalExecutor().run(
        signal_providers,
        SignalContext(motion_id, policy_text, categories),
    )
    failed_signals.extend(signal_run.failed.keys())
    emb_map = dict(signal_run.scores.get("embedding", {}))
    zs_map = dict(signal_run.scores.get("zero_shot", {}))
    ollama_map = dict(signal_run.scores.get("ollama", {}))
    bert_cls_scores = dict(signal_run.scores.get("transformer", {}))
    for name, score in emb_map.items():
        if score >= embedding_threshold:
            matches.setdefault(name, []).append(f"embedding:{score:.3f}")
    for name, score in zs_map.items():
        if score > 0.01:
            matches.setdefault(name, []).append(f"zero_shot:{score:.3f}")
    for name, score in ollama_map.items():
        matches.setdefault(name, []).append(f"ollama:{score:.3f}")

    rhetorical_applied = False
    if meta_clf is not None:
        topic_vec = get_topic_features(motion_id, topic_distributions=topic_distributions)

        category_names = sorted(categories.keys())
        feature_df = build_feature_vector(
            keyword_scores=scores,
            embedding_scores=emb_map,
            topic_features=topic_vec,
            text_length=len(text),
            category_names=category_names,
            date_days_ago=None,
            doc_type=None,
            zero_shot_scores=zs_map,
            bert_cls_scores=bert_cls_scores,
        )

        combined_norm = predict_with_meta_classifier(feature_df, meta_clf, cast(Mapping[str, object], categories))

        if should_use_llm_fallback(combined_norm, threshold=llm_threshold):
            llm_result = llm_judge(text=policy_text[:llm_max_text_len], categories=list(categories.keys()))
            if llm_result is not None:
                llm_cat = llm_result["category"]
                combined_norm = {k: 0.0 for k in categories.keys()}
                combined_norm[llm_cat] = 1.0
                matches.setdefault(llm_cat, []).append(f"llm:{llm_result['reasoning'][:100]}")
    else:
        if use_speech_preprocessing:
            kw_w = 0.0
            emb_w = 0.0
            bert_w = 0.0
            if use_ollama and ollama_map:
                oll_w = 0.60
                zs_w = 0.40
            else:
                oll_w = 0.0
                zs_w = 0.70
                kw_w = 0.30
        else:
            kw_w = max(0.0, 1.0 - embedding_weight - zero_shot_weight)
            emb_w = embedding_weight
            zs_w = zero_shot_weight
            oll_w = 0.0
            bert_w = 0.0

        signal_combinator = SignalCombinator()
        signal_combinator.add_signal("keyword", scores, kw_w)
        signal_combinator.add_signal("embedding", emb_map, emb_w)
        signal_combinator.add_signal("zero_shot", zs_map, zs_w)
        if ollama_map:
            signal_combinator.add_signal("ollama", ollama_map, oll_w)
        if bert_cls_scores:
            signal_combinator.add_signal("bert", bert_cls_scores, bert_w)
        combined_norm = signal_combinator.combine()

    if not combined_norm or sum(combined_norm.values()) <= 0:
        if not categories:
            raise ValueError("Cannot classify without category definitions")
        uniform_weight = Fraction(1, len(categories))
        combined_norm = {name: uniform_weight for name in categories}
        failed_signals.append("no_signal")

    # Apply rhetorical adjustments regardless of meta-classifier path
    if use_speech_preprocessing:
        rhet_adjustments = detect_rhetorical_patterns(text)
        if any(v != 0.0 for v in rhet_adjustments.values()):
            rhetorical_applied = True
            combined_norm = apply_rhetorical_adjustments(combined_norm, rhet_adjustments, boost_factor=2.0)
            for k in combined_norm:
                if rhet_adjustments.get(k, 0.0) > 0:
                    matches.setdefault(k, []).append(f"rhetorical:x{float(2.0 + rhet_adjustments.get(k, 0.0)):.2f}")

    # Add BERT CLS scores to matched_rules for hybrid ensemble access
    for cat, score in bert_cls_scores.items():
        if score > 0:
            matches.setdefault(cat, []).append(f"bert_cls:{score:.3f}")

    base_version = "0.8.0"
    classifier_version = base_version
    signals = []
    if use_speech_preprocessing:
        signals.append("speech")
    if _get_spacy() is not None:
        signals.append("spacy")
    if embedding_matcher is not None and emb_map:
        signals.append("emb")
        try:
            signals.append(f"({embedding_matcher.model_name})")
        except Exception:
            signals.append("(unknown)")  # model_name unavailable
    if zs_map:
        signals.append("zs")
    if meta_clf is not None:
        signals.append("meta")
    if ollama_map:
        signals.append("ollama")
    if rhetorical_applied:
        signals.append("rhetorical")
    if "no_signal" in failed_signals:
        signals.append("degraded")
    classifier_version += "+" + "+".join(signals) if signals else ""

    requested_signals = ["keyword", "embedding", "zero_shot", "transformer"]
    if use_speech_preprocessing:
        requested_signals.append("rhetorical")
    if use_ollama:
        requested_signals.append("ollama")
    provenance = ClassificationProvenance(
        pipeline_version=classifier_version,
        signals_requested=requested_signals,
        signals_used=signals,
        signals_failed=sorted(set(failed_signals)),
        degraded=bool(failed_signals),
    )

    if use_supervised and meta_clf is None:
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
                    probs = None  # predict_proba failed

                if probs is not None:
                    try:
                        prob_vec = probs[0]
                    except Exception:
                        prob_vec = probs  # probs indexing failed
                    try:
                        labels = list(mlb.classes_)
                    except Exception:
                        labels = list(range(len(prob_vec)))  # classes_ unavailable

                    sup_map = {str(label): float(probability) for label, probability in zip(labels, prob_vec)}
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
                                classifier_version += "(unknown)"  # model path unavailable
        except Exception as e:
            LOG.warning("Supervised fallback failed: %s", e)

    results: List[ClassificationResult] = []
    for name in categories.keys():
        raw_score = float(scores.get(name, 0.0))
        frac_weight = combined_norm.get(name, Fraction(0, 1))
        normalized = float(frac_weight)
        results.append(
            ClassificationResult(
                motion_id=motion_id,
                category=name,
                raw_score=raw_score,
                normalized_weight=normalized,
                matched_rules=matches.get(name, []),
                classifier_version=classifier_version,
                created_at=datetime.now(timezone.utc),
                provenance=provenance,
            )
        )
    return results


# Speech meta-classifier cache
_SPEECH_META_CLF = None


def _load_speech_meta_classifier() -> Optional[Dict]:
    """Load the speech-specific meta-classifier if available.

    Checks for compressed (.zst) and uncompressed variants of each candidate.
    Falls back to the tuned ensemble meta-classifier if no speech-specific
    model is found, then to the default ensemble meta-classifier.
    """
    global _SPEECH_META_CLF
    if _SPEECH_META_CLF is not None:
        return _SPEECH_META_CLF

    # 1. Try speech-specific models (both compressed and uncompressed)
    speech_candidates = [
        Path("models/speech_meta_clf.pkl.zst"),
        Path("models/speech_meta_clf.pkl"),
        Path("models/speech_meta_clf_parquet.pkl"),
        Path("models/speech_meta_clf_full.pkl.zst"),
        Path("models/speech_meta_clf_full.pkl"),
    ]
    m = _MODEL_PROVIDER.first_mapping(speech_candidates)
    if m is not None:
        _SPEECH_META_CLF = m
        LOG.info("Loaded speech meta-classifier from ordered candidates")
        return m

    # 2. Fall back to tuned ensemble meta-classifier (49.4% val_accuracy)
    tuned_path = Path("models/ensemble_meta_clf_tuned.pkl.zst")
    m = _MODEL_PROVIDER.first_mapping([tuned_path])
    if m is not None:
        _SPEECH_META_CLF = m
        LOG.info("Loaded tuned ensemble meta-classifier from %s (fallback)", tuned_path)
        return m

    # 3. Fall back to hybrid ensemble with BERT features
    hybrid_path = Path("models/hybrid_ensemble_meta_clf.pkl.zst")
    m = _MODEL_PROVIDER.first_mapping([hybrid_path])
    if m is not None:
        _SPEECH_META_CLF = m
        LOG.info("Loaded hybrid ensemble meta-classifier from %s (fallback)", hybrid_path)
        return m

    # 4. Final fallback: default ensemble meta-classifier
    from swedish_parliament_policy_classifier.classifier.ensemble import load_meta_classifier
    m = load_meta_classifier()
    if m is not None:
        _SPEECH_META_CLF = m
        LOG.info("Loaded default ensemble meta-classifier (final fallback)")
        return m

    return None


# Hybrid ensemble cache
_HYBRID_META_CLF = None


def _load_hybrid_meta_classifier() -> Optional[Dict]:
    """Load the hybrid ensemble meta-classifier with BERT [CLS] features.
    
    This is the 905-feature model that achieves 0.94 accuracy on motion classification.
    """
    global _HYBRID_META_CLF
    if _HYBRID_META_CLF is not None:
        return _HYBRID_META_CLF

    hybrid_path = Path("models/hybrid_ensemble_meta_clf.pkl.zst")
    m = _MODEL_PROVIDER.first_mapping([hybrid_path])
    if m is not None:
        _HYBRID_META_CLF = m
        LOG.info("Loaded hybrid ensemble meta-classifier from %s", hybrid_path)
        return m

    return None


def _build_speech_results(
    speech_id: str,
    categories: Dict[str, CategoryDef],
    base_results: List[ClassificationResult],
    probabilities: Dict[str, float],
    classifier_version: str,
    provenance: Optional[ClassificationProvenance] = None,
) -> List[ClassificationResult]:
    """Apply a final probability distribution to the base speech evidence."""
    base_by_category = {result.category: result for result in base_results}
    return [
        ClassificationResult(
            motion_id=speech_id,
            category=name,
            raw_score=base_by_category[name].raw_score if name in base_by_category else 0.0,
            normalized_weight=float(probabilities.get(name, 0.0)),
            matched_rules=base_by_category[name].matched_rules if name in base_by_category else [],
            classifier_version=classifier_version,
            created_at=datetime.now(timezone.utc),
            provenance=provenance,
        )
        for name in categories.keys()
    ]


def score_speech(
    speech_id: str,
    text: str,
    categories: Dict[str, CategoryDef],
    party: Optional[str] = None,
    embedding_matcher: Optional[EmbeddingMatcher] = None,
    embedding_weight: float = 0.40,
    embedding_threshold: float = 0.0,
    zero_shot_weight: float = 0.40,
    zero_shot_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    use_zero_shot: bool = True,
    use_speech_preprocessing: bool = True,
    use_ollama: bool = False,
    ollama_weight: float = 0.35,
    topic_distributions: Optional[Dict[str, List[float]]] = None,
    llm_threshold: float = 0.30,
    llm_max_text_len: int = 2000,
    speech_meta_clf: Optional[Dict] = None,
    rhetoric_scores: Optional[Dict[str, float]] = None,
) -> List[ClassificationResult]:
    """Score a parliamentary speech using the speech-specific pipeline.

    The speech pipeline is the primary classification path for speeches. It
    runs the base signal pipeline (keyword, embedding, zero-shot, BERT) with
    speech-aware text extraction, then applies the speech meta-classifier (if
    available) to produce the final probability distribution.

    Args:
        speech_id: Unique identifier for the speech.
        text: Full speech text.
        categories: Category definitions.
        party: Optional party affiliation (used only for keyword extraction,
            not for classification bias).
        embedding_matcher, embedding_weight, embedding_threshold:
            Embedding matcher settings.
        zero_shot_weight, zero_shot_model, use_zero_shot:
            Zero-shot NLI settings.
        use_speech_preprocessing: Extract argumentative text from speech.
        use_ollama, ollama_weight: Ollama LLM fallback settings.
        topic_distributions: Topic model features.
        llm_threshold, llm_max_text_len: LLM fallback settings.
        speech_meta_clf: Pre-loaded speech meta-classifier dict. If None,
            auto-loaded from ``models/speech_meta_clf*.pkl``.
        rhetoric_scores: Pre-computed rhetorical scores (irony, sarcasm,
            posturing, none, top_label). If None, auto-detected from text.

    Returns:
        List of ClassificationResult with speech-pipeline version string.
    """
    # Run the base pipeline WITHOUT the motion meta-classifier, so the speech
    # meta-classifier can learn the optimal combination for the speech domain.
    base_results = score_motion(
        motion_id=speech_id,
        text=text,
        categories=categories,
        party=party,
        embedding_matcher=embedding_matcher,
        embedding_weight=embedding_weight,
        embedding_threshold=embedding_threshold,
        zero_shot_weight=zero_shot_weight,
        zero_shot_model=zero_shot_model,
        use_zero_shot=use_zero_shot,
        meta_clf=None,  # Don't use the motion-trained meta-classifier
        llm_threshold=llm_threshold,
        llm_max_text_len=llm_max_text_len,
        skip_policy_extraction=True,
        use_speech_preprocessing=use_speech_preprocessing,
        use_ollama=use_ollama,
        ollama_weight=ollama_weight,
        topic_distributions=topic_distributions,
    )

    # Extract base probabilities from the signal pipeline
    base_probs = {r.category: r.normalized_weight for r in base_results}

    # Get rhetorical scores
    if rhetoric_scores is None and use_speech_preprocessing:
        rhetoric_scores = detect_rhetorical_patterns(text)
    rhetoric_scores = rhetoric_scores or {}

    # Apply hybrid ensemble meta-classifier for higher accuracy (905 features)
    hybrid_clf = _load_hybrid_meta_classifier()
    speech_clf = speech_meta_clf if speech_meta_clf is not None else _load_speech_meta_classifier()

    if hybrid_clf is not None:
        # Use hybrid ensemble with full 905-feature vector for 0.94 accuracy
        from swedish_parliament_policy_classifier.classifier.ensemble import (
            build_feature_vector,
            predict_with_meta_classifier,
        )
        category_names = sorted(categories.keys())
        # Extract all features needed for hybrid model
        text_length = len(text)
        
        # Build keyword scores from base_results (extract from matched_rules)
        keyword_scores = {r.category: r.raw_score for r in base_results}
        
        embedding_scores = extract_signal_output(
            base_results, name="embedding", evidence_prefix="embedding"
        ).score_map()
        zero_shot_scores = extract_signal_output(
            base_results, name="zero_shot", evidence_prefix="zero_shot"
        ).score_map()
        bert_cls_scores = extract_signal_output(
            base_results, name="transformer", evidence_prefix="bert_cls"
        ).score_map()
        
        # Use build_feature_vector with full feature set
        feature_df = build_feature_vector(
            keyword_scores=keyword_scores,
            embedding_scores=embedding_scores,
            topic_features=None,
            text_length=text_length,
            category_names=category_names,
            date_days_ago=None,
            doc_type="speech",
            zero_shot_scores=zero_shot_scores,
            bert_cls_scores=bert_cls_scores,
        )
        final_probs = predict_with_meta_classifier(feature_df, hybrid_clf, cast(Mapping[str, object], categories))
        
        # Build final results with hybrid version
        base_version = base_results[0].classifier_version if base_results else "0.8.0"
        speech_version = f"hybrid_ensemble+{base_version}"
        
        return _build_speech_results(
            speech_id,
            categories,
            base_results,
            final_probs,
            speech_version,
            base_results[0].provenance if base_results else None,
        )

    elif speech_clf is not None:
        from swedish_parliament_policy_classifier.classifier.ensemble import (
            build_speech_feature_vector,
            predict_with_meta_classifier,
        )
        category_names = sorted(categories.keys())
        feature_df = build_speech_feature_vector(
            base_probs, rhetoric_scores, category_names=category_names
        )
        final_probs = predict_with_meta_classifier(feature_df, speech_clf, cast(Mapping[str, object], categories))

        # Build final results with the speech meta-classifier probabilities
        base_version = base_results[0].classifier_version if base_results else "0.8.0"
        speech_version = f"speech_meta+{base_version}"

        return _build_speech_results(
            speech_id,
            categories,
            base_results,
            final_probs,
            speech_version,
            base_results[0].provenance if base_results else None,
        )

    # No meta-classifier available: return base pipeline results
    return base_results
