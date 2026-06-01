"""LLM-as-judge fallback using local Ollama for low-confidence motions.

Structured prompt requesting JSON category + reasoning output.
Only invoked when the ensemble meta-classifier produces low confidence.
"""

import json
import logging
from typing import Dict, Optional, List

import ollama

LOG = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5-coder-14b-32k:latest"
DEFAULT_THRESHOLD = 0.30


def _build_prompt(text: str, categories: List[str], max_text_len: int = 2000) -> str:
    """Build a structured prompt for the LLM judge."""
    truncated = text[:max_text_len]
    cat_list = ", ".join(f'"{c}"' for c in categories)

    prompt = f"""Du är en expert på svensk politik och ideologisk analys.

Uppgift: Läs följande motion från Sveriges riksdag och placera den på en politisk skala.

Kategorier att välja mellan: {cat_list}

Instruktioner:
- Välj ENDAST en kategori.
- Motivet ska vara baserat på motionens POLICY-INNEHÅLL, inte på vilket parti som författat den.
- Förklara KORT varför du valde den kategorin.

Motionstext:
---
{truncated}
---

Svara ENDAST med ett JSON-objekt i exakt detta format:
{{"category": "<kategori>", "reasoning": "<motivering>"}}
"""
    return prompt


def llm_judge(
    text: str,
    categories: List[str],
    model: str = DEFAULT_MODEL,
    max_text_len: int = 2000,
    temperature: float = 0.1,
) -> Optional[Dict[str, str]]:
    """Query local Ollama model for category judgment.

    Returns dict with 'category' and 'reasoning', or None on failure.
    """
    prompt = _build_prompt(text, categories, max_text_len)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        content = response["message"]["content"]

        # Try to extract JSON from response
        # Ollama sometimes wraps JSON in markdown code fences
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        result = json.loads(content)

        # Validate
        if "category" not in result or "reasoning" not in result:
            LOG.warning("LLM response missing required fields: %s", result)
            return None

        if result["category"] not in categories:
            LOG.warning("LLM returned unknown category: %s", result["category"])
            return None

        return result

    except Exception as e:
        LOG.warning("LLM judge failed: %s", e)
        return None


def should_use_llm_fallback(
    ensemble_probs: Dict[str, float],
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """Determine if LLM fallback should be invoked based on low confidence."""
    max_prob = max(ensemble_probs.values()) if ensemble_probs else 0.0
    return max_prob < threshold
