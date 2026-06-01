"""Refactored classification pipeline extracted from the legacy scorer.

This module isolates extraction, signal computation and combination into a
single place so callers can import the refined pipeline without depending on
the large legacy `scorer.py` implementation.
"""
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
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
from swedish_parliament_policy_classifier.nlp.preprocess import init_spacy, preprocess_text
from swedish_parliament_policy_classifier.nlp.topic_modeler import get_topic_features
from swedish_parliament_policy_classifier.classifier.ensemble import (
    build_feature_vector,
    predict_with_meta_classifier,
)
from swedish_parliament_policy_classifier.classifier.llm_judge import (
    llm_judge,
    should_use_llm_fallback,
)

LOG = logging.getLogger(__name__)

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
    if party:
        party_names = {
            "V": "vänsterpartiet",
            "S": "socialdemokraterna",
            "MP": "miljöpartiet",
            "C": "centerpartiet",
            "L": "liberalerna",
            "M": "moderaterna",
            "KD": "kristdemokraterna",
            "SD": "sverigedemokraterna",
        }
        pname = party_names.get(party, "")
        if pname:
            party_markers.append(pname)

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


def _detect_rhetorical_patterns(text: str) -> Dict[str, float]:
    if not text:
        return {}
    text_lower = text.lower()
    adjustments = {"far_left": 0.0, "left": 0.0, "centre_left": 0.0,
                   "centre": 0.0, "centre_right": 0.0, "right": 0.0, "far_right": 0.0}
    anti_eu_terms = [
        "eu-kritisk", "eu-kritik", "bryta med eu", "lämna eu",
        "europeiska unionen", "eu-skeptisk", "eus inflytande",
        "motståndare till eu", "eu-medlemskap", "budgetramar",
        "budgetdisciplin",
    ]
    fiscal_terms = [
        "budget", "budgetramar", "finansiella ramar", "budgetdisciplin",
        "ekonomiska ramar", "budgetansvar", "finanspolitik", "ekonomi",
    ]
    if any(t in text_lower for t in anti_eu_terms) and any(t in text_lower for t in fiscal_terms):
        adjustments["right"] += 1.40
        adjustments["far_right"] += 0.40
    env_terms = ["miljö", "natura 2000", "biologisk mångfald", "naturvård"]
    extraction_terms = ["gruva", "gruvdrift", "malmbrytning"]
    pro_industry = ["ja till", "positivt", "möjliggöra"]
    if any(t in text_lower for t in env_terms) and any(t in text_lower for t in extraction_terms):
        if any(t in text_lower for t in pro_industry):
            adjustments["right"] += 1.20
            adjustments["far_right"] += 0.30
            adjustments["centre_right"] += 0.30
    healthcare_terms = ["sjukvård", "vård", "missbruksvård"]
    privatization_terms = ["privat", "privata aktörer", "företag", "marknad"]
    if any(t in text_lower for t in healthcare_terms) and any(t in text_lower for t in privatization_terms):
        adjustments["right"] += 1.00
        adjustments["centre_right"] += 0.40
    return adjustments


# Cache the lemma keyword index so it is built only once per process
_LEMMA_KW_INDEX: Optional[Dict[str, List[Tuple[str, str]]]] = None


def _build_lemma_kw_index(categories: Dict[str, CategoryDef]) -> Dict[str, List[Tuple[str, str]]]:
    global _LEMMA_KW_INDEX
    if _LEMMA_KW_INDEX is not None:
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
            lemma_text = " ".join(preproc["lemmas"])
            lemma_tokens = preproc["lemmas"]
        else:
            lemma_text = text_l[:MAX_SPA_CY]
            lemma_tokens = lemma_text.split()
    elif nlp is not None:
        preproc = preprocess_text(proc_source, nlp=nlp, remove_stopwords=False, lemmatize=True, normalize=True)
        lemma_text = " ".join(preproc["lemmas"])
        lemma_tokens = preproc["lemmas"]
    else:
        lemma_text = text_l[:MAX_SPA_CY]
        lemma_tokens = lemma_text.split()

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

    emb_map: Dict[str, float] = {}
    if embedding_matcher is not None and embedding_weight > 0:
        try:
            if not hasattr(embedding_matcher, "_cached_cat_embs"):
                embedding_matcher._cached_cat_embs = embedding_matcher.build_category_embeddings(categories)
            emb_matches = embedding_matcher.match(policy_text, embedding_matcher._cached_cat_embs, top_k=len(categories))
            emb_map = {name: float(score) for name, score in emb_matches}
            for name, score in emb_map.items():
                if score >= embedding_threshold:
                    matches.setdefault(name, []).append(f"embedding:{score:.3f}")
        except Exception as e:
            LOG.warning("Embedding matcher failed: %s", e)

    zs_map: Dict[str, float] = {}
    if use_zero_shot and zero_shot_weight > 0:
        try:
            if use_speech_preprocessing:
                from swedish_parliament_policy_classifier.nlp.zero_shot_values import zero_shot_score_speech_aware
                zs_map = zero_shot_score_speech_aware(text, model_name=zero_shot_model)
            else:
                from swedish_parliament_policy_classifier.nlp.zero_shot_values import zero_shot_score
                zs_map = zero_shot_score(policy_text, model_name=zero_shot_model)
            for name, score in zs_map.items():
                if score > 0.01:
                    matches.setdefault(name, []).append(f"zero_shot:{score:.3f}")
        except Exception as e:
            LOG.warning("Zero-shot classification failed: %s", e)

    ollama_map: Dict[str, float] = {}
    if use_speech_preprocessing and use_ollama:
        try:
            from swedish_parliament_policy_classifier.nlp.ollama_classifier import classify_speech_with_cache
            ollama_map = classify_speech_with_cache(text, speech_id=motion_id, cache=None) or {}
            if ollama_map:
                for name, score in ollama_map.items():
                    matches.setdefault(name, []).append(f"ollama:{score:.3f}")
        except Exception as e:
            LOG.warning("Ollama classification failed: %s", e)

    bert_cls_scores: Dict[str, float] = {}
    try:
        from swedish_parliament_policy_classifier.classifier.transformer_predict import predict_proba as _bert_predict
        bert_cls_scores = _bert_predict(policy_text[:2500])
    except Exception as e:
        LOG.debug("Transformer predict unavailable for speech: %s", e)

    keyword_sum = sum(scores.values())
    keyword_norm = {k: (v / keyword_sum if keyword_sum > 0 else 0.0) for k, v in scores.items()}

    emb_sum = sum(emb_map.values()) if emb_map else 0.0
    emb_norm = {k: (emb_map.get(k, 0.0) / emb_sum if emb_sum > 0 else 0.0) for k in categories.keys()}

    zs_sum = sum(zs_map.values()) if zs_map else 0.0
    zs_norm = {k: (zs_map.get(k, 0.0) / zs_sum if zs_sum > 0 else 0.0) for k in categories.keys()}

    rhetorical_applied = False
    if meta_clf is not None:
        topic_vec = get_topic_features(motion_id, topic_distributions=topic_distributions)
        bert_cls_scores = {}
        try:
            from swedish_parliament_policy_classifier.classifier.transformer_predict import predict_proba as _bert_predict
            bert_cls_scores = _bert_predict(policy_text[:2500])
        except Exception as e:
            LOG.warning("Transformer predict unavailable: %s", e)

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

        combined_norm = predict_with_meta_classifier(feature_df, meta_clf, categories)

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
                bert_w = 0.25
                zs_w = 0.15
            else:
                oll_w = 0.0
                bert_w = 0.70
                zs_w = 0.30
        else:
            kw_w = max(0.0, 1.0 - embedding_weight - zero_shot_weight)
            emb_w = embedding_weight
            zs_w = zero_shot_weight
            oll_w = 0.0
            bert_w = 0.0

        combined_norm = {
            k: (
                kw_w * keyword_norm.get(k, 0.0)
                + emb_w * emb_norm.get(k, 0.0)
                + zs_w * zs_norm.get(k, 0.0)
                + oll_w * ollama_map.get(k, 0.0)
                + bert_w * bert_cls_scores.get(k, 0.0)
            )
            for k in categories.keys()
        }

        total_combined = sum(combined_norm.values())
        if total_combined > 0:
            for k in combined_norm:
                combined_norm[k] = combined_norm[k] / total_combined

        if use_speech_preprocessing:
            rhet_adjustments = _detect_rhetorical_patterns(text)
            if any(v != 0.0 for v in rhet_adjustments.values()):
                rhetorical_applied = True
                for k in combined_norm:
                    adj = rhet_adjustments.get(k, 0.0)
                    if adj > 0:
                        combined_norm[k] += adj
                        matches.setdefault(k, []).append(f"rhetorical:+{adj:.2f}")
                total_adj = sum(combined_norm.values())
                if total_adj > 0:
                    for k in combined_norm:
                        combined_norm[k] /= total_adj

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
            signals.append("(unknown)")
    if zs_map:
        signals.append("zs")
    if meta_clf is not None:
        signals.append("meta")
    if ollama_map:
        signals.append("ollama")
    if rhetorical_applied:
        signals.append("rhetorical")
    classifier_version += "+" + "+".join(signals) if signals else ""

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
