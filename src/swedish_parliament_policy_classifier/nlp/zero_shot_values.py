"""Zero-shot value-based classification using multilingual NLI.

This module adds a "meaning layer" to the scorer: instead of matching surface
keywords, it tests whether a motion's text *entails* value-laden hypotheses that
capture the core ideals of each ideological position.

The underlying model is a multilingual NLI classifier (mDeBERTa-v3-base-mnli-xnli).
All hypotheses are batched into a single forward pass per motion for GPU efficiency.
"""

from __future__ import annotations

# Preload nvidia/cu13 libs so PyTorch CUDA JIT can find libnvrtc-builtins
import swedish_parliament_policy_classifier.cuda_fix  # noqa: F401

import logging
import os
import re
from typing import Dict, List

# Reduce CUDA memory fragmentation when other processes (e.g. Ollama)
# already occupy most of the GPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LOG = logging.getLogger(__name__)

# Lazy import so the rest of the pipeline can start without the heavy model
_pipeline = None
_tokenizer = None

# Value-laden hypotheses in Swedish for each category.
# Derived from academic encyclopedic definitions (Britannica, Stanford Encyclopedia
# of Philosophy, political science textbooks) to ensure objectivity and cross-cultural
# applicability.  Each hypothesis tests whether the text *advocates* for a position,
# not merely mentions it.
#
# CRITICAL REDESIGN (v0.8.0): hypotheses are deliberately topic-agnostic.
# Instead of tying each category to specific policies (crime, immigration,
# feminism, etc.) they focus on five underlying ideological dimensions:
#   1. Economic system  (collective/state vs market/private)
#   2. Social values    (egalitarian/rights-based vs traditional/hierarchical)
#   3. Governance       (regulation/redistribution vs deregulation/low-tax)
#   4. Change           (radical/revolutionary vs reformist vs conservative)
#   5. Nationalism      (open/cooperative vs closed/sovereigntist)
# This prevents a speech ABOUT crime/policing from being mis-classified
# by topic overlap — a left-wing speaker discussing crime must still be
# recognised as left because their framing focuses on social causes
# (economic system) and rights (social values), not on harsh punishment
# or ethnic homogeneity.
CATEGORY_HYPOTHESES: Dict[str, List[str]] = {
    "far_left": [
        "Detta förespråkar att avskaffa kapitalismen och införa kollektivt ägande.",
        "Detta vill att samhällets resurser ska fördelas efter behov, inte efter arbetsinsats.",
        "Detta argumenterar för en radikal, revolutionär omstöpning av det ekonomiska systemet.",
        "Detta förespråkar arbetarklassens makt och klasskamp för att bryta de styrandes herravälde.",
        "Detta vill att staten eller de anställda ska äga och styra stora företag.",
    ],
    "left": [
        "Detta förespråkar att minska ekonomiska klyftor genom omfördelning.",
        "Detta vill stärka arbetares rättigheter och fackföreningar.",
        "Detta argumenterar att skatt på de rika ska höjas för att finansiera välfärd.",
        "Detta förespråkar att utbyggd allmän välfärd ska garantera alla ett drägligt liv.",
        "Detta vill att staten ska reglera ekonomin för att skydda arbetare och miljö.",
    ],
    "centre_left": [
        "Detta förespråkar en balans mellan marknadsekonomi och stark offentlig välfärd.",
        "Detta vill gradvis reformera samhället mot större jämlikhet utan att avskaffa kapitalismen.",
        "Detta argumenterar att staten ska reglera näringslivet och investera i utbildning och hälsa.",
        "Detta förespråkar individers rättigheter, inkludering och jämställdhet som centrala politiska mål.",
        "Detta vill behålla en blandekonomi med både privata företag och offentliga åtaganden.",
    ],
    "centre": [
        "Detta förespråkar att undvika både radikal förändring och konservativ status quo.",
        "Detta vill bedriva pragmatisk politik baserad på fakta snarare än ideologi.",
        "Detta argumenterar för samarbete över blockgränser och kompromiss.",
        "Detta förespråkar att balansera effektivitet med socialt ansvar utan extrema åtgärder.",
        "Detta vill att administrationen ska effektiviseras utan att förändra systemets grundvalar.",
    ],
    "centre_right": [
        "Detta förespråkar fri marknad, privat ägande och att staten ska lämna stora delar av ekonomin.",
        "Detta vill att individer själva ska ta ansvar för sitt liv och sina val.",
        "Detta argumenterar för lägre skatter, avreglering och privatisering av offentliga tjänster.",
        "Detta förespråkar att reformera samhället långsamt utan att bryta med traditionella institutioner.",
        "Detta vill att entreprenörskap och konkurrens ska driva samhällets utveckling.",
    ],
    "right": [
        "Detta förespråkar att bevara traditionella institutioner, seder och sociala hierarkier.",
        "Detta vill sänka skatter, krympa staten och avreglera marknaden.",
        "Detta argumenterar att individens frihet och familjens integritet ska skyddas från statlig inblandning.",
        "Detta förespråkar nationens säkerhet, militär styrka och strängare ordningsregler.",
        "Detta vill bevara landets kulturarv och traditionella värderingar mot förändring.",
    ],
    "far_right": [
        "Detta förespråkar att nationen ska stänga ute utlänningar och försvara en etniskt ren identitet.",
        "Detta vill ha ett auktoritärt, disciplinerat styre med starkt ledarskap.",
        "Detta argumenterar mot multikulturalism och för att kulturell homogenitet ska upprätthållas med tvång.",
        "Detta förespråkar att staten ska prioritera nationens säkerhet över individers rättigheter.",
        "Detta varnar att inhemsk kultur och nationellt oberoende håller på att förgöras av utländskt inflytande.",
    ],
}

# Critique hypotheses: test whether text *argues against* or *criticizes* a position.
# Also derived from academic definitions — each is the logical negation/critique of
# the corresponding advocacy position, following the same topic-agnostic
# ideological-dimension design as the advocacy hypotheses.
CATEGORY_HYPOTHESES_CRITIQUE: Dict[str, List[str]] = {
    "far_left": [
        "Detta kritiserar förslag om att avskaffa kapitalismen eller införa kollektivt ägande.",
        "Detta argumenterar emot att samhällets resurser ska fördelas efter behov snarare än marknadskrafter.",
        "Detta motsätter sig att arbetarklassen ska ta över makten genom revolution.",
    ],
    "left": [
        "Detta kritiserar omfördelning av välstånd och försvagande av marknadsekonomin.",
        "Detta argumenterar emot att arbetares rättigheter och fackföreningar ska stärkas.",
        "Detta motsätter sig att skatter ska höjas på de rika för att finansiera välfärd.",
    ],
    "centre_left": [
        "Detta kritiserar att staten ska reglera näringslivet eller driva omfattande välfärdsprogram.",
        "Detta argumenterar emot en blandekonomi och vill ha mer renodlad marknadsliberalism.",
        "Detta motsätter sig att jämställdhet, inkludering och individers rättigheter ska vara centrala mål.",
    ],
    "centre": [
        "Detta kritiserar pragmatism och mittenpositioner som efterliknar både höger och vänster.",
        "Detta argumenterar emot faktabaserad politik och kompromisslösningar.",
        "Detta motsätter sig samarbete över blockgränser och vill se tydliga ideologiska skillnader.",
    ],
    "centre_right": [
        "Detta kritiserar fri marknad, privat ägande och att staten ska dra sig tillbaka från ekonomin.",
        "Detta argumenterar emot att individuellt ansvar och entreprenörskap ska vara ledande principer.",
        "Detta motsätter sig att skatter ska sänkas, marknaden avregleras och offentliga tjänster privatiseras.",
    ],
    "right": [
        "Detta kritiserar traditionella institutioner och seder som hämmar social utveckling.",
        "Detta argumenterar emot att skatter ska sänkas, staten krympas och marknaden avregleras.",
        "Detta motsätter sig att individers och familjers frihet ska skyddas från statlig inblandning.",
    ],
    "far_right": [
        "Detta kritiserar att utlänningar ska stängas ute eller att etnisk homogenitet ska tvingas fram.",
        "Detta argumenterar emot auktoritärt styre och mot krav på hård disciplin och starkt ledarskap.",
        "Detta motsätter sig att multikulturalism ska bekämpas och att nationen ska isoleras från omvärlden.",
    ],
}


# Flatten hypotheses for batched processing
_FLAT_HYPOTHESES: List[tuple[str, str]] = []
for _cat, _hyps in CATEGORY_HYPOTHESES.items():
    for _h in _hyps:
        _FLAT_HYPOTHESES.append((_cat, _h))

_FLAT_HYPOTHESES_CRITIQUE: List[tuple[str, str]] = []
for _cat, _hyps in CATEGORY_HYPOTHESES_CRITIQUE.items():
    for _h in _hyps:
        _FLAT_HYPOTHESES_CRITIQUE.append((_cat, _h))


def _load_model(model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"):
    """Lazy-load the zero-shot NLI pipeline."""
    global _pipeline, _tokenizer
    if _pipeline is not None:
        return _pipeline, _tokenizer

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("transformers and torch required for zero-shot NLI") from exc

    LOG.info("Loading zero-shot NLI model: %s", model_name)
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _pipeline = AutoModelForSequenceClassification.from_pretrained(model_name)
    _pipeline.eval()

    # Pick device: if another GPU-heavy process (e.g. Ollama) is running,
    # free memory may be too fragmented for reliable batched inference.
    # In that case we keep the small NLI model on CPU and only run the
    # large generative LLM on GPU.
    device = "cpu"
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)
        # Need ~3 GB of contiguous free memory for this model with batching.
        if free_gb >= 3.0:
            device = "cuda"
            LOG.info("CUDA has %.1f / %.1f GB free — loading NLI model on CUDA", free_gb, total_gb)
        else:
            LOG.info(
                "CUDA only has %.1f / %.1f GB free — keeping NLI model on CPU to avoid OOM fragmentation",
                free_gb, total_gb,
            )
    _pipeline = _pipeline.to(device)
    LOG.info("Zero-shot model loaded on %s", device.upper())
    return _pipeline, _tokenizer


def _chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    """Split long text into semantically coherent chunks.

    Chunks are kept large (up to 1500 chars, ~250-300 words) so that each
    piece contains enough context for meaningful ideological analysis.
    Very short chunks (<100 chars) are dropped because they carry insufficient
    semantic signal.
    """
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: List[str] = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chars:
            current = current + " " + s if current else s
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks


def zero_shot_score(
    text: str,
    model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
) -> Dict[str, float]:
    """Return a mapping {category: entailment_score} for the given text.

    Scores are computed per-chunk and averaged so long documents do not dilute
    the signal.  All hypotheses are batched into a single forward pass per chunk
    for GPU efficiency.
    """
    if not text or not text.strip():
        return {cat: 0.0 for cat in CATEGORY_HYPOTHESES}

    model, tokenizer = _load_model(model_name)

    import torch
    import numpy as np

    chunks = _chunk_text(text, max_chars=1500)
    # Drop boilerplate chunks that are just headers/metadata
    # (keep only chunks with substantial content for meaningful analysis)
    keepers = []
    for ch in chunks:
        if len(ch) < 100:
            continue
        if ch.startswith("Motion till riksdagen") or ch.startswith("Förslag till riksdagsbeslut"):
            pass
        keepers.append(ch)
    if not keepers:
        keepers = chunks[:1]

    device = next(model.parameters()).device

    # Accumulate scores across chunks — process one chunk at a time
    # to minimise GPU memory pressure when other processes (e.g. Ollama)
    # already occupy most of the VRAM.
    all_scores: Dict[str, List[float]] = {cat: [] for cat in CATEGORY_HYPOTHESES}
    BATCH_SIZE = 1

    for batch_start in range(0, len(keepers), BATCH_SIZE):
        batch_chunks = keepers[batch_start:batch_start + BATCH_SIZE]

        # Build batched inputs: each premise repeated for every hypothesis
        premises: List[str] = []
        hypotheses: List[str] = []
        for premise in batch_chunks:
            premises.extend([premise] * len(_FLAT_HYPOTHESES))
            hypotheses.extend([h for _, h in _FLAT_HYPOTHESES])

        # Tokenize in batch
        inputs = tokenizer(
            premises,
            hypotheses,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits  # shape: (batch, 3)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            entailment_probs = probs[:, 0]  # index 0 = entailment for this model

        # Aggregate back per chunk per category
        n_hyps = len(_FLAT_HYPOTHESES)
        for chunk_idx, premise in enumerate(batch_chunks):
            start = chunk_idx * n_hyps
            end = start + n_hyps
            for (cat, _), ent_prob in zip(_FLAT_HYPOTHESES, entailment_probs[start:end]):
                all_scores[cat].append(float(ent_prob))

        # Free any fragmented allocations before the next chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Average across all chunks
    result = {cat: float(np.mean(vals)) for cat, vals in all_scores.items()}
    return result


def _tag_chunk_stance(chunk_text: str) -> str:
    """Tag a chunk's dominant stance for speech-aware NLI."""
    # Import stance logic from scorer
    from swedish_parliament_policy_classifier.classifier.scorer import _sentence_stance

    sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
    stances = [_sentence_stance(s) for s in sentences if len(s.strip()) > 15]
    if not stances:
        return "neutral"

    # Majority stance
    from collections import Counter
    c = Counter(stances)
    return c.most_common(1)[0][0]


def zero_shot_score_speech_aware(
    text: str,
    model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    critique_weight: float = 0.5,
) -> Dict[str, float]:
    """Return a mapping {category: net_entailment_score} for a parliamentary speech.

    Unlike the base zero_shot_score, this function:
    1. Tags each chunk as own_position, opponent_report, rhetorical_challenge, or neutral.
    2. For opponent-report and rhetorical-challenge chunks, runs BOTH advocacy
       hypotheses and critique hypotheses.
    3. Computes net_score = advocacy_score - (critique_score * critique_weight).
    4. For own-position chunks, only runs advocacy hypotheses (speaker's own position).
    5. Never uses party membership info.

    Args:
        text: The speech text (should already be passed through
              _extract_speech_argumentative_text or similar preprocessing).
        model_name: The NLI model to use.
        critique_weight: How much to weight critique scores when subtracting.
                         0.5 means critique counts half as much as advocacy.
    """
    if not text or not text.strip():
        return {cat: 0.0 for cat in CATEGORY_HYPOTHESES}

    model, tokenizer = _load_model(model_name)
    import torch
    import numpy as np

    chunks = _chunk_text(text, max_chars=1500)
    if not chunks:
        return {cat: 0.0 for cat in CATEGORY_HYPOTHESES}

    device = next(model.parameters()).device

    # Accumulate scores across chunks
    adv_scores: Dict[str, List[float]] = {cat: [] for cat in CATEGORY_HYPOTHESES}
    crit_scores: Dict[str, List[float]] = {cat: [] for cat in CATEGORY_HYPOTHESES}

    for chunk in chunks:
        if len(chunk) < 100:
            continue

        stance = _tag_chunk_stance(chunk)

        # Build inputs: run BOTH advocacy and critique for all chunks.
        # Stance only affects how much we weight the critique signal later.
        premises: List[str] = []
        hypotheses: List[str] = []
        # Advocacy hypotheses
        premises.extend([chunk] * len(_FLAT_HYPOTHESES))
        hypotheses.extend([h for _, h in _FLAT_HYPOTHESES])
        # Critique hypotheses
        premises.extend([chunk] * len(_FLAT_HYPOTHESES_CRITIQUE))
        hypotheses.extend([h for _, h in _FLAT_HYPOTHESES_CRITIQUE])

        # Tokenize
        inputs = tokenizer(
            premises,
            hypotheses,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            entailment_probs = probs[:, 0]

        # Aggregate advocacy scores
        n_hyps = len(_FLAT_HYPOTHESES)
        for i in range(n_hyps):
            cat = _FLAT_HYPOTHESES[i][0]
            adv_scores[cat].append(float(entailment_probs[i]))

        # Aggregate critique scores for ALL chunks.
        # Critique signals ("this criticizes X") are essential to distinguish
        # a speaker who argues *against* a right-wing policy (high critique,
        # low advocacy) from a speaker who advocates it (high advocacy).
        # Previously these were dropped for own_position chunks, which
        # caused topic-driven false positives: a left-wing speech on crime
        # kept its right-wing advocacy score because the right critique
        # score was thrown away.
        for i in range(len(_FLAT_HYPOTHESES_CRITIQUE)):
            cat = _FLAT_HYPOTHESES_CRITIQUE[i][0]
            idx = n_hyps + i
            crit_scores[cat].append(float(entailment_probs[idx]))

        # Free any fragmented allocations before the next chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Compute net scores
    result: Dict[str, float] = {}
    for cat in CATEGORY_HYPOTHESES:
        adv = float(np.mean(adv_scores[cat])) if adv_scores[cat] else 0.0
        crit = float(np.mean(crit_scores[cat])) if crit_scores[cat] else 0.0
        net = max(0.0, adv - crit * critique_weight)
        result[cat] = net

    return result
