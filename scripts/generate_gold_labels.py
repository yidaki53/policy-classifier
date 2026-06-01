#!/usr/bin/env python3
"""Generate gold labels using local Ollama LLM-as-judge.

Samples motions stratified by doc_type and decade, sends each to Ollama
with full category definitions, and stores the gold label in the database.

Usage:
    uv run python3 scripts/generate_gold_labels.py --db data/swedish_parliament.db --sample 2000
"""

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

if os.getenv("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.getenv("HF_TOKEN"))

import ollama

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions


DEFAULT_MODEL = "qwen2.5-coder-14b-32k:latest"
MAX_TEXT_LEN = 3000


def _build_gold_prompt(text: str, categories: Dict, max_text_len: int = MAX_TEXT_LEN) -> str:
    """Build a prompt that presents the full definitions to the LLM."""
    truncated = text[:max_text_len]

    cat_blocks = []
    for name, cat in categories.items():
        definition = cat.definition or ""
        keywords = ", ".join(cat.keywords or [])[:200]
        cat_blocks.append(
            f'"{name}": {definition}\n  Nyckelord: {keywords}'
        )
    cat_text = "\n\n".join(cat_blocks)

    prompt = f"""Du är en expert på svensk politik och ideologisk analys med tillgång till följande exakta kategoridefinitioner.

Uppgift: Läs motionen nedan och placera den på EXAKT EN av de sex kategorierna. Ditt svar ska baseras uteslutande på motionens POLICY-INNEHÅLL, inte på vilket parti som författat den.

Kategorier (i strikt ordning från vänster till höger):

{cat_text}

Instruktioner:
- Välj ENDAST EN kategori från listan ovan.
- Använd definitionerna som exakta riktlinjer.
- Motivera KORT varför du valde den kategorin (2-3 meningar).
- Svara ENDAST med JSON i exakt detta format:
{{"category": "<kategorinamn>", "reasoning": "<motivering>"}}

Motionstext:
---
{truncated}
---

JSON-svar:"""
    return prompt


def _parse_llm_response(content: str, valid_categories: List[str]) -> Optional[Dict]:
    """Extract and validate JSON from LLM response."""
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        result = json.loads(content)
    except Exception:
        # Try to extract JSON object from surrounding text
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(content[start:end+1])
            except Exception:
                return None
        else:
            return None

    if "category" not in result or "reasoning" not in result:
        return None

    cat = result["category"].strip().lower()
    if cat not in [c.lower() for c in valid_categories]:
        # Try fuzzy match against valid names
        for valid in valid_categories:
            if valid.lower() == cat:
                result["category"] = valid
                return result
        return None

    result["category"] = cat
    return result


def _llm_judge(text: str, categories: Dict, model: str = DEFAULT_MODEL, temperature: float = 0.1) -> Optional[Dict]:
    prompt = _build_gold_prompt(text, categories)
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        content = response["message"]["content"]
        return _parse_llm_response(content, list(categories.keys()))
    except Exception as e:
        print(f"  LLM error: {e}", file=sys.stderr)
        return None


def _sample_motions(conn: sqlite3.Connection, sample_size: int, min_text_len: int = 200) -> List[Tuple]:
    """Stratified sample by doc_type and decade."""
    cur = conn.cursor()

    # Get distribution by doc_type + decade
    cur.execute(f"""
        SELECT SUBSTR(date, 1, 3) || '0s' AS decade, doc_type, COUNT(*)
        FROM normalized_motions
        WHERE date IS NOT NULL AND text IS NOT NULL AND LENGTH(text) > {min_text_len}
        GROUP BY decade, doc_type
    """)
    dist = cur.fetchall()

    total = sum(d[2] for d in dist)
    samples = []
    remaining = sample_size

    # Allocate proportionally, ensure at least 1 per stratum
    for decade, doc_type, count in dist:
        target = max(1, int(sample_size * count / total))
        samples.append((decade, doc_type, target))
        remaining -= target

    # Distribute remainder to largest strata
    if remaining > 0:
        samples.sort(key=lambda x: x[2], reverse=True)
        for i in range(remaining):
            idx = i % len(samples)
            decade, doc_type, target = samples[idx]
            samples[idx] = (decade, doc_type, target + 1)

    # Fetch actual rows
    selected = []
    for decade, doc_type, target in samples:
        decade_prefix = decade[:3]
        cur.execute("""
            SELECT id, text, date, party, doc_type
            FROM normalized_motions
            WHERE date LIKE ? AND doc_type = ? AND text IS NOT NULL AND LENGTH(text) > ?
            ORDER BY RANDOM()
            LIMIT ?
        """, (f"{decade_prefix}%", doc_type, min_text_len, target))
        rows = cur.fetchall()
        selected.extend(rows)

    random.shuffle(selected)
    return selected[:sample_size]


def _already_gold(conn: sqlite3.Connection, motion_id: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM gold_labels WHERE motion_id = ?", (motion_id,))
    return cur.fetchone() is not None


def generate_gold_labels(
    db_path: str = "data/swedish_parliament.db",
    sample_size: int = 2000,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    resume: bool = True,
):
    conn = init_db(db_path)
    defs = load_definitions()
    valid_categories = list(defs.keys())

    print(f"Sampling up to {sample_size} motions for gold-label generation...", file=sys.stderr)
    rows = _sample_motions(conn, sample_size)
    print(f"Selected {len(rows)} motions (stratified by decade/doc_type)", file=sys.stderr)

    if resume:
        rows = [r for r in rows if not _already_gold(conn, r[0])]
        print(f"After skipping already-gold: {len(rows)} remaining", file=sys.stderr)

    inserted = 0
    failed = 0
    start_time = time.time()

    for i, row in enumerate(rows):
        motion_id, text, date, party, doc_type = row
        result = _llm_judge(text or "", defs, model=model, temperature=temperature)

        if result is None:
            failed += 1
            continue

        cur = conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO gold_labels
               (motion_id, category, reasoning, prompt_version, model, temperature, raw_response, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                motion_id,
                result["category"],
                result.get("reasoning", "")[:1000],
                "gold-v1.0-full-defs",
                model,
                temperature,
                json.dumps(result, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        inserted += 1

        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(rows) - (i + 1)) / rate / 3600 if rate > 0 else 0
        if (i + 1) % 10 == 0:
            print(
                f"  {i+1}/{len(rows)}  inserted={inserted} failed={failed}  "
                f"rate={rate:.1f}m/s  ETA={eta:.1f}h",
                file=sys.stderr,
            )

    elapsed = time.time() - start_time
    print(
        f"\nDone: {inserted} gold labels inserted, {failed} failures in {elapsed:.1f}s",
        file=sys.stderr,
    )
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Generate gold labels using Ollama LLM")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--sample", type=int, default=2000, help="Target number of motions to label")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Re-label already processed motions")
    args = parser.parse_args()

    total = generate_gold_labels(
        db_path=args.db,
        sample_size=args.sample,
        model=args.model,
        temperature=args.temperature,
        resume=args.resume,
    )
    print(f"Total gold labels: {total}")


if __name__ == "__main__":
    main()
