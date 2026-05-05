"""Text preprocessing utilities with graceful fallbacks.

Attempts to use spaCy (Swedish) when available, otherwise falls back to
lightweight tokenization and normalization. Functions are defensive so tests
can run without optional heavy deps installed.
"""

from typing import List, Optional, Dict
import re
import logging
import unicodedata

LOG = logging.getLogger(__name__)

_spacy_nlp = None

try:
    import spacy  # type: ignore
except Exception:
    spacy = None

# Expanded minimal Swedish stopword set as fallback (can be extended by loading spaCy stopwords)
_SW_STOPWORDS = set(
    [
        "och",
        "i",
        "att",
        "det",
        "som",
        "en",
        "på",
        "är",
        "av",
        "för",
        "med",
        "inte",
        "om",
        "till",
        "har",
        "var",
    ]
)


def init_spacy(model: str = "sv_core_news_sm", install: bool = False):
    """Initialize and return a spaCy language pipeline.

    If `install=True` the function will attempt to download the model using
    `spacy.cli.download` when the model is not present. This is optional and
    controlled by the caller, tests do not attempt downloads by default.
    """
    global _spacy_nlp
    if spacy is None:
        LOG.info("spaCy not installed; skipping spacy init")
        return None
    try:
        _spacy_nlp = spacy.load(model)
        return _spacy_nlp
    except Exception as e:
        LOG.debug("spaCy model load failed: %s", e)
        if install:
            try:
                import spacy.cli as spacy_cli  # type: ignore

                spacy_cli.download(model)
                _spacy_nlp = spacy.load(model)
                return _spacy_nlp
            except Exception as e2:
                LOG.warning("spaCy model auto-download failed: %s", e2)
        _spacy_nlp = None
        return None


def _simple_tokenize(text: str) -> List[str]:
    # Keep words, numbers, but drop isolated punctuation tokens
    tokens = re.findall(r"\w+|[^\t\n\r\f\v\w]+", text.lower(), flags=re.UNICODE)
    return [t for t in tokens if t.strip() and not re.fullmatch(r"\W+", t)]


def _normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\u2013", "-")
    t = t.replace("\u2014", "-")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _merge_stopwords(nlp) -> set:
    sw = set(_SW_STOPWORDS)
    if nlp is not None and hasattr(nlp, "Defaults"):
        try:
            spacy_sw = set([s.lower() for s in nlp.Defaults.stop_words])
            sw.update(spacy_sw)
        except Exception:
            pass
    return sw


def preprocess_text(
    text: str,
    nlp=None,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    normalize: bool = True,
) -> Dict[str, object]:
    """Return a dict with normalized text, tokens, and lemmas.

    The function is defensive: if spaCy is available and `nlp` is provided or
    previously initialized via `init_spacy`, it will use spaCy for tokenization
    and lemmatization. Otherwise it falls back to a lightweight tokenizer.
    """
    if not text:
        return {"text": "", "tokens": [], "lemmas": []}

    if normalize:
        text_norm = _normalize_text(text)
    else:
        text_norm = text

    if nlp is None:
        nlp = _spacy_nlp

    stopwords = _merge_stopwords(nlp)

    if nlp is not None:
        doc = nlp(text_norm)
        tokens = [t.text for t in doc if not t.is_space and not t.is_punct]
        if lemmatize:
            lemmas = [t.lemma_.lower() if t.lemma_ else t.text.lower() for t in doc if not t.is_space and not t.is_punct]
        else:
            lemmas = [t.text.lower() for t in doc if not t.is_space and not t.is_punct]
        if remove_stopwords:
            tokens = [t for t, l in zip(tokens, lemmas) if l not in stopwords]
            lemmas = [l for l in lemmas if l not in stopwords]
        return {"text": text_norm, "tokens": tokens, "lemmas": lemmas}

    # Fallback
    tokens = _simple_tokenize(text_norm)
    lemmas = tokens[:]  # no lemmatizer in fallback
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stopwords]
        lemmas = [l for l in lemmas if l not in stopwords]
    return {"text": text_norm.lower(), "tokens": tokens, "lemmas": lemmas}
