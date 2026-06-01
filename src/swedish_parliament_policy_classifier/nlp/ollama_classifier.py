"""Local LLM-based ideological classifier using Ollama.

This module provides a fallback / enhancement layer for speech classification
that uses a locally-running LLM (qwen2.5-coder-14b via Ollama) to classify
speeches by ideological framing rather than surface topic overlap.

The key design principle: the LLM is given the full academic definitions
of all 7 ideological positions and instructed to classify based on WHICH
IDEOLOGICAL FRAMEWORK the speaker applies to the topic, not merely WHICH
topic is discussed.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

LOG = logging.getLogger(__name__)

# Default model — must be pulled and available locally via `ollama`
# Use llama3.1:8b (fast, ~8GB VRAM, good enough for classification)
# qwen2.5-coder-14b is higher quality but 6x slower on this hardware.
import os

# Default model — prefer environment override or the local high-quality Qwen model
# If `OLLAMA_MODEL` is set, use that; otherwise prefer the local `qwen2.5-coder-14b-32k:latest`.
_DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder-14b-32k:latest")

# Condensed category definitions in English (smaller prompt = faster inference).
_CATEGORY_DEFINITIONS = {
    "far_left": "abolish capitalism, collective ownership, revolution, classless society, redistribute by need",
    "left": "egalitarianism, redistribute wealth, progressive taxation, strong unions, welfare expansion, regulate economy for workers",
    "centre_left": "balance market and welfare, gradual reform, equality, inclusion, mixed economy",
    "centre": "pragmatic, compromise, evidence-based, cross-party cooperation, technocratic",
    "centre_right": "free market, lower taxes, privatization, entrepreneurship, individual responsibility, basic welfare safety net",
    "right": "tradition, family, law/order, military strength, lower taxes, smaller state, national security",
    "far_right": "ultranationalism, anti-immigration, ethnic homogeneity, authoritarian, anti-multiculturalism, nation over individual",
}

_CATEGORIES_ORDER = ["far_left", "left", "centre_left", "centre", "centre_right", "right", "far_right"]


def _build_prompt(speech_text: str, max_chars: int = 2500) -> str:
    """Construct a compact classification prompt for fast inference."""
    text = speech_text.strip()
    if len(text) > max_chars:
        idx = text.rfind(".", 0, max_chars)
        if idx > max_chars * 0.7:
            text = text[: idx + 1]
        else:
            text = text[:max_chars]

    defs_block = "\n".join(
        f"- {name}: {desc}"
        for name, desc in _CATEGORY_DEFINITIONS.items()
    )

    prompt = (
        "Classify this Swedish parliamentary speech by ideological framing, NOT by topic. "
        "Two speakers may discuss the same topic (e.g. crime, taxes, patents) from completely different ideological perspectives. "
        "What matters is WHICH SOLUTION they advocate and WHICH ideological framework they apply.\n\n"
        "Categories (left to right):\n"
        f"{defs_block}\n\n"
        "Rules:\n"
        "- A speaker criticising 'harsher punishment' and instead advocating rehabilitation and social investment is LEFT, even if the topic is crime.\n"
        "- A speaker advocating to relax drug patents so poor countries can access medicine is LEFT, even though patents are market-related.\n"
        "- A speaker advocating lower taxes, privatisation, and individual responsibility is CENTRE-RIGHT / RIGHT.\n"
        "- Return ONLY a JSON object with one score (0-100) per category. Total need not be 100.\n\n"
        "Speech text:\n"
        "---\n"
        f"{text}\n"
        "---\n\n"
        "JSON:"
    )
    return prompt


def _ollama_generate(prompt: str, model: str = _DEFAULT_MODEL, timeout: int = 120) -> str:
    """Send a prompt to the local Ollama API and return the raw response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 512,
        },
    }
    try:
        import requests  # type: ignore
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except Exception as req_err:
        # Fallback to CLI if HTTP fails
        LOG.warning("Ollama HTTP request failed (%s); trying CLI fallback", req_err)
        try:
            cmd = [
                "ollama", "run", model,
                "--format", "json",
            ]
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode == 0:
                return proc.stdout
            LOG.error("Ollama CLI failed: %s", proc.stderr)
            return ""
        except Exception as cli_err:
            LOG.error("Ollama CLI fallback also failed: %s", cli_err)
            return ""


def _parse_scores(raw: str) -> Optional[Dict[str, float]]:
    """Extract the JSON score object from LLM response text."""
    # Try to find JSON block
    # Look for pattern like {"far_left": 10, "left": 45, ...}
    match = re.search(r'\{[^{}]*"far_left"[^{}]*\}', raw, re.DOTALL)
    if not match:
        # Try broader search for any JSON object with our keys
        match = re.search(r'\{.*"left".*\}', raw, re.DOTALL)
    if not match:
        return None

    json_str = match.group(0)
    # Clean up common LLM artifacts
    json_str = json_str.replace("'", '"')
    json_str = re.sub(r",\s*\}", "}", json_str)  # trailing comma
    json_str = re.sub(r",\s*\]", "]", json_str)

    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        # Try extracting key:value pairs manually
        scores: Dict[str, float] = {}
        for cat in _CATEGORIES_ORDER:
            pattern = rf'"{cat}"\s*:\s*(\d+(?:\.\d+)?)'
            m = re.search(pattern, json_str)
            if m:
                scores[cat] = float(m.group(1))
        if scores:
            return scores
        return None

    scores = {}
    for cat in _CATEGORIES_ORDER:
        val = obj.get(cat)
        if val is None:
            # Try variant keys
            alt = cat.replace("_", " ")
            val = obj.get(alt)
        if val is not None:
            try:
                scores[cat] = float(val)
            except (ValueError, TypeError):
                pass
    return scores if scores else None


def classify_speech(
    text: str,
    model: str = _DEFAULT_MODEL,
    max_chars: int = 3500,
    timeout: int = 120,
) -> Optional[Dict[str, float]]:
    """Classify a speech using the local Ollama LLM.

    Returns a dict {category: normalized_score_0_to_1} or None on failure.
    """
    # Respect optional environment overrides to run the LLM on smaller
    # chunks (helpful when the local Ollama instance is memory-constrained).
    try:
        import os

        env_chunk = os.getenv("OLLAMA_CHUNK_CHARS")
        env_max_chunks = os.getenv("OLLAMA_MAX_CHUNKS")
        chunk_chars = int(env_chunk) if env_chunk is not None else None
        max_chunks = int(env_max_chunks) if env_max_chunks is not None else 6
    except Exception:
        chunk_chars = None
        max_chunks = 6

    if chunk_chars is None:
        chunk_chars = max_chars

    # If the speech fits within a single chunk, behave as before.
    if len(text) <= chunk_chars:
        prompt = _build_prompt(text, max_chars=max_chars)
        raw = _ollama_generate(prompt, model=model, timeout=timeout)
        if not raw:
            return None

        scores = _parse_scores(raw)
        if scores is None:
            LOG.warning("Could not parse Ollama response: %r", raw[:500])
            return None
    else:
        # Split into roughly `chunk_chars`-sized pieces (try to split at
        # sentence boundaries) and aggregate parsed scores across chunks.
        texts: List[str] = []
        s = text
        start = 0
        tlen = len(s)
        while start < tlen and len(texts) < max_chunks:
            end = min(start + chunk_chars, tlen)
            # Prefer splitting on sentence boundary when available
            idx = s.rfind('.', start, end)
            if idx > start + int(chunk_chars * 0.5):
                end = idx + 1
            part = s[start:end].strip()
            if not part:
                break
            texts.append(part)
            # advance
            if end <= start:
                start = start + chunk_chars
            else:
                start = end

        if not texts:
            return None

        agg_scores: Dict[str, float] = {}
        seen_chunks = 0
        for part in texts:
            prompt = _build_prompt(part, max_chars=max_chars)
            raw = _ollama_generate(prompt, model=model, timeout=timeout)
            if not raw:
                continue
            scores_part = _parse_scores(raw)
            if scores_part is None:
                LOG.debug("Could not parse Ollama chunk response (skipping chunk)")
                continue
            seen_chunks += 1
            for k, v in scores_part.items():
                agg_scores[k] = agg_scores.get(k, 0.0) + float(v)

        if seen_chunks == 0:
            return None

        # Average the aggregated chunk scores
        scores = {k: (v / seen_chunks) for k, v in agg_scores.items()}

    # Normalize to [0, 1]
    # LLM returns 0-100 usually, but sometimes 0-1
    max_val = max(scores.values()) if scores else 0
    if max_val > 1.5:  # clearly 0-100 scale
        for k in scores:
            scores[k] = scores[k] / 100.0
    else:
        # Already 0-1 or mixed; just clamp
        for k in scores:
            scores[k] = max(0.0, min(1.0, scores[k]))

    # Renormalize so they sum to 1 (softmax-like for ensemble blending)
    total = sum(scores.values())
    if total > 0:
        for k in scores:
            scores[k] = scores[k] / total
    else:
        return None

    return scores


def classify_speech_with_cache(
    text: str,
    speech_id: str,
    cache: Optional[Dict[str, Dict[str, float]]] = None,
    model: str = _DEFAULT_MODEL,
    max_chars: int = 3500,
    timeout: int = 120,
) -> Optional[Dict[str, float]]:
    """Classify with an in-memory cache keyed by speech_id."""
    if cache is not None and speech_id in cache:
        return cache[speech_id]
    result = classify_speech(text, model=model, max_chars=max_chars, timeout=timeout)
    if cache is not None and result is not None:
        cache[speech_id] = result
    return result
