#!/usr/bin/env python3
"""Targeted augmentation for underrepresented ideological classes.

Generates synthetic parliamentary motions for rare classes (right, far-right,
centre) using LLM prompting conditioned on category definitions and real
example motions. Optionally verifies synthetic motions via a second LLM pass
to ensure they match the target category.

Usage:
    uv run python3 scripts/augment_rare_classes.py --db data/swedish_parliament.db --target-per-class 150
"""

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher

# Use GPT-4o for higher-quality generation if OPENAI_API_KEY is available,
# otherwise fall back to local Ollama
try:
    import openai
    OPENAI_CLIENT = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    HAS_OPENAI = True
except Exception:
    OPENAI_CLIENT = None
    HAS_OPENAI = False

try:
    import ollama
    HAS_OLLAMA = True
except Exception:
    HAS_OLLAMA = False

DEFAULT_LOCAL_MODEL = "qwen2.5-coder-14b-32k:latest"
MAX_TEXT_LEN = 3000
DEFAULT_SYNTH_MODEL = DEFAULT_LOCAL_MODEL
DEFAULT_VERIFY_MODEL = DEFAULT_LOCAL_MODEL
SYNTH_SEED_BATCH = 5  # number of real examples to include in prompt

# Classes we consider "rare" based on test-set recall analysis
RARE_CLASSES = ["right", "far_right", "centre"]


def _build_synthetic_prompt(category: str, categories: Dict, examples: List[str], num_examples: int = SYNTH_SEED_BATCH) -> str:
    """Build a rich prompt for synthetic motion generation."""
    cat_def = categories[category].definition
    cat_keywords = categories[category].keywords or []
    sample_texts = "\n\n".join(
        f"Exempel {i+1}:\n---\n{e[:1200]}\n---"
        for i, e in enumerate(examples[:num_examples])
    )
    keyword_list = ", ".join(cat_keywords[:15])  # top 15 keywords

    prompt = f"""Du är en erfaren svensk riksdagsledamot som skriver en motion.

UPPGIFT: Skriv en NY, AUTENTISK riksdagsmotion som tydligt tillhör den ideologiska kategorin "{category}".

KATEGORIDEFINITION:
{cat_def}

VANLIGA TERMEROCH KONCEPT I DENNA KATEGORI:
{keyword_list}

RIKTLINJER:
- Motionen ska vara 4-12 meningar lång.
- Den ska innehålla konkreta politiska förslag eller ståndpunkter.
- Använd autentisk riksdagsterminologi ("vi föreslår", "riksdagen bör", "regeringen ska").
- Texten ska vara idéologiskt konsekvent med kategorin ovan.
- Undvik att upprepa exakt formulering från exemplen; skapa nytt innehåll.

{sample_texts}

Skriv ENDAST motionstexten, ingen introduktion eller avslutning."""
    return prompt


# Adjacent categories that are most likely to bleed into each other
_CONFUSABLE_NEIGHBORS = {
    "far_left": ["left", "centre_left"],
    "left": ["far_left", "centre_left"],
    "centre_left": ["left", "centre"],
    "centre": ["centre_left", "centre_right"],
    "centre_right": ["centre", "right"],
    "right": ["centre_right", "far_right"],
    "far_right": ["right", "centre_right"],
}


def embedding_gate(text: str, expected_category: str, categories: Dict, matcher: EmbeddingMatcher, min_margin: float = 0.05) -> bool:
    """Gate 2: synthetic motion must be semantically closer to the target category than its neighbors."""
    if matcher is None or getattr(matcher, "model", None) is None:
        print("  Embedding gate skipped: no embedding model available", file=sys.stderr)
        return True  # soft-fail: let other gates decide

    neighbors = _CONFUSABLE_NEIGHBORS.get(expected_category, [])
    candidates = {k: categories[k] for k in [expected_category] + neighbors if k in categories}
    if len(candidates) < 2:
        return True

    try:
        cat_embs = matcher.build_category_embeddings(candidates)
        q_emb = matcher.encode([text])[0]
        target_sim = matcher._cosine(q_emb, cat_embs[expected_category])
        for n in neighbors:
            if n not in cat_embs:
                continue
            n_sim = matcher._cosine(q_emb, cat_embs[n])
            if target_sim - n_sim < min_margin:
                print(f"  Embedding gate rejected: {expected_category}={target_sim:.3f} vs {n}={n_sim:.3f} (margin {target_sim - n_sim:.3f} < {min_margin})", file=sys.stderr)
                return False
        return True
    except Exception as e:
        print(f"  Embedding gate error: {e}", file=sys.stderr)
        return True  # soft-fail


def keyword_gate(text: str, expected_category: str, categories: Dict, min_keywords: int = 3) -> bool:
    """Gate 3: synthetic motion must contain enough target-category keywords or regexes."""
    text_lower = text.lower()
    cat = categories.get(expected_category)
    if cat is None:
        return False

    keywords = cat.keywords or []
    kw_hits = sum(1 for kw in keywords if kw.lower() in text_lower)

    regex_hits = 0
    for rx in cat.regexes or []:
        try:
            if re.search(rx, text_lower):
                regex_hits += 1
        except re.error:
            continue

    if kw_hits >= min_keywords or regex_hits >= 1:
        return True

    print(f"  Keyword gate rejected: {kw_hits} keywords, {regex_hits} regexes (need >= {min_keywords} keywords or >= 1 regex)", file=sys.stderr)
    return False


def _build_verification_prompt(text: str, expected_category: str, categories: Dict) -> str:
    """Build a strict verification prompt comparing target against confusable neighbors only.

    Instead of open classification among all 7 categories (which causes drift into
    adjacent positions), we force the LLM to score the motion against the target
    category and its two most confusable neighbors. A synthetic motion is only
    accepted if the target category scores highest by a clear margin.
    """
    neighbors = _CONFUSABLE_NEIGHBORS.get(expected_category, [])
    candidates = [expected_category] + neighbors

    defs_text = "\n\n".join(
        f"{name.upper()}:\n{categories[name].definition[:600]}"
        for name in candidates
    )

    prompt = f"""Du är en expert på svensk politisk ideologi. Din uppgift är att bedöma om en riksdagsmotion TYDIGT tillhör en specifik ideologisk kategori, eller om den "blöder" över i en närliggande position.

BEDÖM FÖLJANDE MOTION mot ENDAST dessa tre kategorier:

{defs_text}

MOTIONSTEXT:
---
{text[:2000]}
---

INSTRUKTIONER:
1. Ge en POÄNG (0-10) för hur väl motionen matchar VARJE av de tre kategorierna.
2. Förklara KORT varför du gav dessa poäng.
3. Om målkategorin ("{expected_category}") INTE har högst poäng, avvisa motionen.
4. Om målkategorin har högst poäng men bara med 1-2 poängs marginal, avvisa den (osäker gräns).

Svara ENDAST med ett JSON-objekt i exakt detta format:
{{"{expected_category}": <0-10>, "{neighbors[0] if neighbors else 'none'}": <0-10>, "{neighbors[1] if len(neighbors) > 1 else 'none'}": <0-10>, "accepted": true/false, "margin": <poäng>, "reasoning": "<kort förklaring>"}}"""
    return prompt


def generate_with_openai(prompt: str, temperature: float = 0.8, max_tokens: int = 800) -> Optional[str]:
    if not HAS_OPENAI or OPENAI_CLIENT is None:
        return None

    import concurrent.futures

    def _call():
        return OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call)
            response = future.result(timeout=60)
            return response.choices[0].message.content.strip()
    except concurrent.futures.TimeoutError:
        print("  OpenAI call timed out after 60s", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  OpenAI generation failed: {e}", file=sys.stderr)
        return None


def generate_with_ollama(prompt: str, model: str = DEFAULT_LOCAL_MODEL, temperature: float = 0.8) -> Optional[str]:
    if not HAS_OLLAMA:
        return None
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        text = response["message"]["content"].strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        return text[:MAX_TEXT_LEN]
    except Exception as e:
        print(f"  Ollama generation failed: {e}", file=sys.stderr)
        return None


def generate_synthetic_motion(category: str, categories: Dict, examples: List[str], temperature: float = 0.8, model: str = DEFAULT_SYNTH_MODEL) -> Optional[str]:
    """Generate a synthetic motion, preferring GPT-4o, falling back to Ollama."""
    prompt = _build_synthetic_prompt(category, categories, examples)
    text = generate_with_openai(prompt, temperature=temperature)
    if text is None:
        text = generate_with_ollama(prompt, temperature=temperature, model=model)
    return text


def verify_synthetic_motion(text: str, expected_category: str, categories: Dict, model: str = DEFAULT_VERIFY_MODEL) -> bool:
    """Gate 1: Verify that a synthetic motion matches its claimed category via LLM boundary check."""
    prompt = _build_verification_prompt(text, expected_category, categories)
    response = generate_with_openai(prompt, temperature=0.1)
    if response is None:
        response = generate_with_ollama(prompt, temperature=0.1, model=model)
    if response is None:
        return False

    try:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        result = json.loads(response)
        accepted = result.get("accepted", False)
        margin = result.get("margin", 0)
        reasoning = result.get("reasoning", "")

        # Strict acceptance criteria:
        # 1. The verification LLM explicitly accepted it (target category won with >=3pt margin)
        # 2. Margin is at least 3 (clear separation from neighbors)
        if accepted and margin >= 3:
            return True
        print(f"  Verification rejected: accepted={accepted}, margin={margin}, reason={reasoning[:80]}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  Verification parse failed: {e}", file=sys.stderr)
        return False


def get_train_distribution(conn: sqlite3.Connection) -> Dict[str, int]:
    """Count existing training samples per category."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT category, COUNT(*) FROM augmented_gold_labels
        WHERE split = 'train' OR split IS NULL
        GROUP BY category
        """
    )
    return {row[0]: row[1] for row in cur.fetchall()}


def get_example_motions(conn: sqlite3.Connection, category: str, n: int = SYNTH_SEED_BATCH) -> List[str]:
    """Fetch real example motions for a category to seed generation."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT a.text FROM augmented_gold_labels a
        WHERE a.category = ? AND (a.source = 'original' OR a.source = 'backtrans')
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (category, n),
    )
    return [row[0] for row in cur.fetchall() if row[0] and len(row[0]) > 100]


def insert_synthetic(
    conn: sqlite3.Connection,
    category: str,
    text: str,
    verified: bool,
    model_used: str,
):
    """Insert a synthetic motion into augmented_gold_labels."""
    cur = conn.cursor()
    synth_id = f"synth_rare_{category}_{int(time.time())}_{random.randint(1000,9999)}"
    source = "synthetic_rare_verified" if verified else "synthetic_rare_unverified"
    cur.execute(
        """
        INSERT OR IGNORE INTO augmented_gold_labels
        (motion_id, category, reasoning, source, parent_id, text, created_at, split)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            synth_id,
            category,
            f"LLM-generated rare-class synthetic ({model_used}) for {category}",
            source,
            None,
            text,
            datetime.now(timezone.utc).isoformat(),
            "train",
        ),
    )
    conn.commit()
    return cur.rowcount > 0


def augment_rare_classes(
    db_path: str = "data/swedish_parliament.db",
    target_per_class: int = 150,
    verify: bool = True,
    max_attempts_multiplier: float = 1.5,
    synth_model: str = DEFAULT_SYNTH_MODEL,
    verify_model: str = DEFAULT_VERIFY_MODEL,
    emb_margin: float = 0.05,
    min_keywords: int = 3,
):
    """Generate synthetic motions for rare classes until target count is reached.

    Uses a three-gate verification pipeline:
      1. LLM boundary check (target vs confusable neighbors)
      2. Embedding similarity gate (cosine margin vs neighbors)
      3. Keyword/regex gate (must match target category signals)
    """
    conn = init_db(db_path)
    defs = load_definitions()

    # Load embedding matcher for Gate 2 (soft-fail if unavailable)
    matcher = None
    try:
        matcher = EmbeddingMatcher()
        if matcher.model is None:
            matcher = None
    except Exception as e:
        print(f"Embedding matcher unavailable: {e}", file=sys.stderr)

    # Check current distribution
    dist = get_train_distribution(conn)
    print("Current training distribution:", file=sys.stderr)
    for cat, count in sorted(dist.items(), key=lambda x: x[1]):
        marker = " *** RARE ***" if cat in RARE_CLASSES else ""
        print(f"  {cat}: {count}{marker}", file=sys.stderr)

    total_inserted = 0
    total_verified = 0
    total_failed = 0

    for category in RARE_CLASSES:
        current = dist.get(category, 0)
        needed = max(0, target_per_class - current)
        if needed <= 0:
            print(f"\n{category}: already at {current} samples, skipping.", file=sys.stderr)
            continue

        print(f"\n{category}: {current} -> target {target_per_class} (need {needed})", file=sys.stderr)

        # Fetch seed examples
        examples = get_example_motions(conn, category, n=SYNTH_SEED_BATCH)
        if not examples:
            print(f"  WARNING: no real examples found for {category}, using definitions only", file=sys.stderr)

        # Generate with three-gate verification
        attempts = 0
        max_attempts = int(needed * max_attempts_multiplier)
        successes = 0

        while successes < needed and attempts < max_attempts:
            attempts += 1
            # Vary temperature to increase diversity
            temp = 0.7 + random.uniform(-0.15, 0.25)
            text = generate_synthetic_motion(category, defs, examples, temperature=temp, model=synth_model)
            if text is None or len(text) < 200:
                total_failed += 1
                continue

            model_used = "gpt-4o" if HAS_OPENAI else synth_model

            # Gate 1: LLM boundary check
            if verify:
                is_valid = verify_synthetic_motion(text, category, defs, model=verify_model)
                if not is_valid:
                    print(f"  Attempt {attempts}: Gate 1 (LLM) FAILED", file=sys.stderr)
                    total_failed += 1
                    continue
                total_verified += 1

            # Gate 2: Embedding similarity gate
            if not embedding_gate(text, category, defs, matcher, min_margin=emb_margin):
                print(f"  Attempt {attempts}: Gate 2 (embedding) FAILED", file=sys.stderr)
                total_failed += 1
                continue

            # Gate 3: Keyword/regex gate
            if not keyword_gate(text, category, defs, min_keywords=min_keywords):
                print(f"  Attempt {attempts}: Gate 3 (keyword) FAILED", file=sys.stderr)
                total_failed += 1
                continue

            inserted = insert_synthetic(conn, category, text, verify, model_used)
            if inserted:
                successes += 1
                total_inserted += 1
                print(f"  Attempt {attempts}: all gates passed ({successes}/{needed})", file=sys.stderr)
            else:
                total_failed += 1

        print(f"  {category} complete: {successes} inserted after {attempts} attempts", file=sys.stderr)

    # Final summary
    new_dist = get_train_distribution(conn)
    print(f"\n=== Rare-class augmentation complete ===", file=sys.stderr)
    print(f"Total inserted: {total_inserted}", file=sys.stderr)
    print(f"Total verified: {total_verified}", file=sys.stderr)
    print(f"Total failed/rejected: {total_failed}", file=sys.stderr)
    print(f"\nNew training distribution:", file=sys.stderr)
    for cat, count in sorted(new_dist.items(), key=lambda x: x[1]):
        marker = " *** RARE ***" if cat in RARE_CLASSES else ""
        print(f"  {cat}: {count}{marker}", file=sys.stderr)

    return total_inserted, total_verified, total_failed


def main():
    parser = argparse.ArgumentParser(description="Targeted augmentation for rare ideological classes")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--target-per-class", type=int, default=150, help="Target training samples per rare class")
    parser.add_argument("--no-verify", action="store_true", help="Skip LLM verification of synthetic motions")
    parser.add_argument("--max-attempts-multiplier", type=float, default=1.5, help="Max attempts = needed * multiplier")
    parser.add_argument("--synth-model", default=DEFAULT_SYNTH_MODEL, help="Ollama model for synthetic generation")
    parser.add_argument("--verify-model", default=DEFAULT_VERIFY_MODEL, help="Ollama model for LLM verification")
    parser.add_argument("--emb-margin", type=float, default=0.05, help="Minimum cosine margin for embedding gate")
    parser.add_argument("--min-keywords", type=int, default=3, help="Minimum keyword hits for keyword gate")
    args = parser.parse_args()

    augment_rare_classes(
        db_path=args.db,
        target_per_class=args.target_per_class,
        verify=not args.no_verify,
        max_attempts_multiplier=args.max_attempts_multiplier,
        synth_model=args.synth_model,
        verify_model=args.verify_model,
        emb_margin=args.emb_margin,
        min_keywords=args.min_keywords,
    )


if __name__ == "__main__":
    main()
