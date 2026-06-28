"""Refactored classification pipeline extracted from the legacy scorer.

This module isolates extraction, signal computation and combination into a
single place so callers can import the refined pipeline without depending on
the large legacy `scorer.py` implementation.
"""
import re
import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timezone
from fractions import Fraction
import decimal

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


_RHETORICAL_WEIGHTS = None


def _load_rhetorical_weights() -> dict:
    """Load tuned rhetorical weights from disk if available, else return defaults."""
    global _RHETORICAL_WEIGHTS
    if _RHETORICAL_WEIGHTS is not None:
        return _RHETORICAL_WEIGHTS

    default = {
        "base_far_left": 1.20, "inc_far_left": 0.25,
        "base_left": 1.00, "inc_left": 0.20,
        "base_centre_left": 0.80, "inc_centre_left": 0.15,
        "base_centre": 0.60, "inc_centre": 0.10,
        "base_centre_right": 0.80, "inc_centre_right": 0.15,
        "base_right": 1.00, "inc_right": 0.20,
        "base_far_right": 1.20, "inc_far_right": 0.25,
    }
    p = Path("models/rhetorical_weights_best.json")
    if p.exists():
        try:
            with open(p) as f:
                data = json.load(f)
            loaded = data.get("params", {})
            if loaded:
                default.update(loaded)
                print(f"Loaded tuned rhetorical weights from {p}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to load tuned weights: {e}, using defaults", file=sys.stderr)

    _RHETORICAL_WEIGHTS = default
    return default


def _detect_rhetorical_patterns(text: str) -> Dict[str, float]:
    """Detect ideological rhetorical patterns using 7-dimension Britannica-derived signals.

    Returns per-category adjustment floats that are added to the combined
    classification score when speech preprocessing is active.  The function is
    intentionally lightweight (no heavy ML) so it can run on every speech.
    Weights are loaded from ``models/rhetorical_weights_best.json`` if it exists,
    otherwise fall back to tuned defaults.
    """
    if not text:
        return {}
    text_lower = text.lower()
    adjustments = {"far_left": 0.0, "left": 0.0, "centre_left": 0.0,
                   "centre": 0.0, "centre_right": 0.0, "right": 0.0, "far_right": 0.0}

    w = _load_rhetorical_weights()

    def _apply(cat: str, signals: list[str]) -> None:
        count = sum(1 for s in signals if s in text_lower)
        base = w.get(f"base_{cat}", 0.0)
        inc = w.get(f"inc_{cat}", 0.0)
        if count >= 2:
            adjustments[cat] += base + (count * inc)
        elif count == 1:
            adjustments[cat] += base * 0.5

    # ─── FAR LEFT ───
    _apply("far_left", [
        "kapitalism", "kapitalistisk", "kapitalisterna", "borgarklass",
        "klasskamp", "klassamhälle", "profit", "vinstintresse", "marknadsfundamentalism",
        "nedrusta", "avveckla försvaret", "imperialism", "anti-imperialistisk",
        "kollektivt ägande", "samhälligt ägande", "företagsdemokrati", "demokratiskt ägande",
        "revolution", "radikal förändring", "omstörta", "systemkritisk", "systemfel",
    ])

    # ─── LEFT ───
    _apply("left", [
        "omfördelning", "progressiv beskattning", "höj skatten", "skattehöjning",
        "fackförening", "kollektivavtal", "anställningstrygghet", "las",
        "stärka välfärden", "bygga ut välfärden", "offentlig sektor",
        "stoppa vinster", "vinster i välfärden", "vinstförbud",
        "arbetstagare", "löntagare", "vanliga människor", "folkflertalet",
        "socioekonomisk", "fattigdom", "inkomstskillnader", "jämlikhet",
        "social rättvisa", "rättvisa", "solidaritet", "sammanhållning",
        "allmännytta", "allmännyttan", "bostad för alla",
    ])

    # ─── CENTRE LEFT ───
    _apply("centre_left", [
        "välfärd", "sociala tjänster", "hälso- och sjukvård", "sjukvård",
        "omsorg", "äldreomsorg", "barnomsorg", "förskola", "skola",
        "utbildning", "kompetensutveckling", "livslångt lärande",
        "miljö", "klimat", "hållbarhet", "biologisk mångfald",
        "socialdemokrati", "socialdemokratisk", "reform", "gradvis reform",
        "jämställdhet", "integration", "inkludering", "mänskliga rättigheter",
        "folkhälsa", "förebyggande", "demokrati", "jämlikhet",
        "aktiv arbetsmarknadspolitik", "arbetslöshetsbekämpning", "jobb för alla",
    ])

    # ─── CENTRE ───
    _apply("centre", [
        "båda sidor", "alla partier", "över blockgränsen", "samarbete",
        "kompromiss", "pragmatisk", "balanserad", "lagenlig", "rättssäker",
        "evidensbaserad", "fakta", "konkret förslag", "konkreta åtgärder",
        "effektiv", "resultat", "uppföljning", "utvärdering", "granskning",
        "riksrevisionen", "myndighet", "process", "beredning",
        "oberoende", "opartisk", "saklig", "trovärdig",
    ])

    # ─── CENTRE RIGHT ───
    _apply("centre_right", [
        "marknad", "marknadsekonomi", "marknadslösning", "privat", "privata aktörer",
        "företag", "företagare", "entreprenörskap", "näringsliv", "industri",
        "tillväxt", "kompetitivitet", "innovation", "effektivisera", "effektivitet",
        "skattepolitik", "skattenivå", "beskattning", "skattetryck",
        "budget", "budgetdisciplin", "finanspolitik", "statsfinanser",
        "reform", "modernisera", "förenkla", "färre regleringar",
        "arbetslinjen", "sysselsättning", "arbetskraftsdeltagande",
    ])

    # ─── RIGHT ───
    _apply("right", [
        "sänka skatter", "skattesänkning", "lägre skatt", "dereglering", "avreglering",
        "privatisera", "privatisering", "utförsäljning", "nedläggning",
        "försvar", "försvarsallians", "nato", "säkerhetspolitik", "försvarspolitik",
        "lag och ordning", "straff", "kriminalitet", "brott", "rättsväsende",
        "tradition", "kulturarv", "svenska värderingar", "jämställdhet traditionell",
        "familj", "föräldraskap", "uppfostran", "skolplikt", "disciplin",
        "suveränitet", "nationell", "nationellt självbestämmande", "självständig",
        "eu-kritisk", "eu-kritik", "bryta med eu", "lämna eu", "eu-skeptisk",
        "motståndare till eu", "eus inflytande", "budgetramar", "budgetdisciplin",
        "konservativ", "bevara", "värna", "tuffare", "strängare",
        "minska byråkrati", "minska regleringar", "minska staten",
        "tuffare tag", "ordning och reda", "svenska intressen",
        "egna intressen", "nationella intressen", "svensk suveränitet",
        "minska invandring", "minska migration", "minska asyl",
        "sverigedemokrat", "sverigedemokraterna", "sd",
    ])

    # ─── FAR RIGHT ───
    _apply("far_right", [
        "svenskhet", "svenska folket", "etnisk", "etnicitet", "kulturarv",
        "massinvandring", "invandring", "invandrare", "asyl", "migration",
        "integration misslyckad", "integration har misslyckats", "parallellsamhälle",
        "islam", "islamisering", "muslim", "muslimsk", "sharia", "extrem islam",
        "svenska värden", "västerländska värden", "jämställdhet hotad", "kvinnoförtryck",
        "gräns", "gränskontroll", "återvandring", "återvandra", "repatriering",
        "folkomröstning", "folkets vilja", "eliten", "etablissemanget", "pk-elit",
        "globalisering", "globalism", "internationalism", "fn", "förenta nationerna",
        "censur", "yttrandefrihet hotad", "demokrati i fara", "förrädare",
        "försvara sverige", "sverige först", "sverige åt svenskarna", "vårt land",
        "sverigedemokrat", "sverigedemokraterna", "sd",
    ])

    # ─── COMPOUND PATTERNS ───
    env_terms = ["miljö", "natura 2000", "biologisk mångfald", "naturvård"]
    extraction_terms = ["gruva", "gruvdrift", "malmbrytning"]
    pro_industry = ["ja till", "positivt", "möjliggöra"]
    if any(t in text_lower for t in env_terms) and any(t in text_lower for t in extraction_terms):
        if any(t in text_lower for t in pro_industry):
            adjustments["right"] += 0.80
            adjustments["far_right"] += 0.40
            adjustments["centre_right"] += 0.40
    healthcare_terms = ["sjukvård", "vård", "missbruksvård"]
    privatization_terms = ["privat", "privata aktörer", "företag", "marknad"]
    if any(t in text_lower for t in healthcare_terms) and any(t in text_lower for t in privatization_terms):
        adjustments["right"] += 0.60
        adjustments["centre_right"] += 0.30
    eu_budget_terms = ["eu-budget", "eu:s budget", "gemensam budget", "strukturfond", "budgetram", "budgetdisciplin"]
    conservative_terms = ["konservativ", "minska", "minskning", "effektivisera", "kritisk", "motsätter"]
    if any(t in text_lower for t in eu_budget_terms) and any(t in text_lower for t in conservative_terms):
        adjustments["right"] += 0.80
        adjustments["far_right"] += 0.40

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
            if not hasattr(embedding_matcher, "_cached_cat_embs") or embedding_matcher._cached_cat_embs is None:
                if categories is None:
                    LOG.error("Cannot build category embeddings: categories is None")
                    raise ValueError("categories is None when building embedding cache")
                embedding_matcher._cached_cat_embs = embedding_matcher.build_category_embeddings(categories)
            if embedding_matcher._cached_cat_embs is None:
                LOG.error("Embedding cache is None after build_category_embeddings")
                raise ValueError("embedding_matcher._cached_cat_embs is None")
            emb_matches = embedding_matcher.match(policy_text, embedding_matcher._cached_cat_embs, top_k=len(categories))
            if emb_matches is None:
                LOG.error("match() returned None for text: %s", policy_text[:100] if policy_text else "(empty)")
                raise ValueError("embedding_matcher.match() returned None")
            emb_map = {name: float(score) for name, score in emb_matches}
            for name, score in emb_map.items():
                if score >= embedding_threshold:
                    matches.setdefault(name, []).append(f"embedding:{score:.3f}")
        except Exception as e:
            LOG.warning("Embedding matcher failed: %s", e)
            import traceback
            LOG.debug("Embedding matcher traceback:\n%s", traceback.format_exc())

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

    # Compute exact normalised weights using Fractions to avoid recurring decimals
    keyword_sum = sum(scores.values())
    keyword_norm = {k: (Fraction(int(v), int(keyword_sum)) if keyword_sum > 0 else Fraction(0, 1)) for k, v in scores.items()}

    emb_sum = sum(emb_map.values()) if emb_map else 0.0
    if emb_sum > 1e-12:
        _emb_sum_frac = Fraction(emb_sum)
        emb_norm = {k: Fraction(emb_map.get(k, 0.0)) / _emb_sum_frac for k in categories.keys()}
    else:
        emb_norm = {k: Fraction(0, 1) for k in categories.keys()}

    zs_sum = sum(zs_map.values()) if zs_map else 0.0
    if zs_sum > 1e-12:
        _zs_sum_frac = Fraction(zs_sum)
        zs_norm = {k: Fraction(zs_map.get(k, 0.0)) / _zs_sum_frac for k in categories.keys()}
    else:
        zs_norm = {k: Fraction(0, 1) for k in categories.keys()}

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

        # Combine signals using Fraction for exact arithmetic
        _kw_w = Fraction(kw_w).limit_denominator(100)
        _emb_w = Fraction(emb_w).limit_denominator(100)
        _zs_w = Fraction(zs_w).limit_denominator(100)
        _oll_w = Fraction(oll_w).limit_denominator(100)
        _bert_w = Fraction(bert_w).limit_denominator(100)

        combined_norm = {
            k: (
                _kw_w * keyword_norm.get(k, Fraction(0, 1))
                + _emb_w * emb_norm.get(k, Fraction(0, 1))
                + _zs_w * zs_norm.get(k, Fraction(0, 1))
                + _oll_w * Fraction(ollama_map.get(k, 0.0)).limit_denominator(1000)
                + _bert_w * Fraction(bert_cls_scores.get(k, 0.0)).limit_denominator(1000)
            )
            for k in categories.keys()
        }

        total_combined = sum(combined_norm.values())
        if total_combined > 0:
            for k in combined_norm:
                combined_norm[k] = combined_norm[k] / total_combined

    # Apply rhetorical adjustments regardless of meta-classifier path
    if use_speech_preprocessing:
        rhet_adjustments = _detect_rhetorical_patterns(text)
        if any(v != 0.0 for v in rhet_adjustments.values()):
            rhetorical_applied = True
            for k in combined_norm:
                adj = rhet_adjustments.get(k, 0.0)
                if adj > 0:
                    # Aggressive multiplicative boost so speech-level rhetorical
                    # signals (the primary source of truth for speeches) can
                    # override motion-trained embedding/BERT bias.
                    combined_norm[k] *= Fraction(20, 10) + Fraction(adj).limit_denominator(100)
                    matches.setdefault(k, []).append(f"rhetorical:x{float(2.0 + adj):.2f}")
            total_adj = sum(combined_norm.values())
            if total_adj > 0:
                for k in combined_norm:
                    combined_norm[k] = combined_norm[k] / total_adj

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
                _fractional_weight=frac_weight,
            )
        )
    return results


# Speech meta-classifier cache
_SPEECH_META_CLF = None


def _load_speech_meta_classifier() -> Optional[Dict]:
    """Load the speech-specific meta-classifier if available."""
    global _SPEECH_META_CLF
    if _SPEECH_META_CLF is not None:
        return _SPEECH_META_CLF

    from swedish_parliament_policy_classifier.classifier.ensemble import load_meta_classifier
    from swedish_parliament_policy_classifier.io import loader
    import zstandard

    candidates = [
        Path("models/speech_meta_clf.pkl"),
        Path("models/speech_meta_clf_parquet.pkl"),
        Path("models/speech_meta_clf_full.pkl"),
    ]
    for cand in candidates:
        try:
            if cand.suffix == ".zst":
                with open(cand, "rb") as fh:
                    dctx = zstandard.ZstdDecompressor()
                    with dctx.stream_reader(fh) as reader:
                        m = pickle.load(reader)
            else:
                m = loader.load_pickle(cand)
            if m is not None and ("model" in m or "clf" in m):
                _SPEECH_META_CLF = m
                return m
        except Exception:
            continue
    return None


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
        rhetoric_scores = _detect_rhetorical_patterns(text)
    rhetoric_scores = rhetoric_scores or {}

    # Apply speech meta-classifier if available
    speech_clf = speech_meta_clf if speech_meta_clf is not None else _load_speech_meta_classifier()

    if speech_clf is not None:
        from swedish_parliament_policy_classifier.classifier.ensemble import (
            build_speech_feature_vector,
            predict_with_meta_classifier,
        )
        category_names = sorted(categories.keys())
        feature_df = build_speech_feature_vector(
            base_probs, rhetoric_scores, category_names=category_names
        )
        final_probs = predict_with_meta_classifier(feature_df, speech_clf, categories)

        # Build final results with the speech meta-classifier probabilities
        base_version = base_results[0].classifier_version if base_results else "0.8.0"
        speech_version = f"speech_meta+{base_version}"

        final_results = []
        for name in categories.keys():
            base_result = next((r for r in base_results if r.category == name), None)
            matched_rules = base_result.matched_rules if base_result else []
            raw_score = base_result.raw_score if base_result else 0.0
            final_results.append(
                ClassificationResult(
                    motion_id=speech_id,
                    category=name,
                    raw_score=raw_score,
                    normalized_weight=final_probs.get(name, 0.0),
                    matched_rules=matched_rules,
                    classifier_version=speech_version,
                    created_at=datetime.now(timezone.utc),
                )
            )
        return final_results

    # No speech meta-classifier available: return base pipeline results
    return base_results
