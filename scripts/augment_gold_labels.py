#!/usr/bin/env python3
"""Augment gold labels using back-translation and synthetic LLM generation.

For rare classes (<50): generate synthetic motions via Ollama until ~80 samples.
For moderate rare (50-100): back-translate to double count.
For common (>=100): 20% back-translation.

Stores results in augmented_gold_labels table with 'source' column.

Usage:
    uv run python3 scripts/augment_gold_labels.py --db data/swedish_parliament.db
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
import torch
from transformers import MarianMTModel, MarianTokenizer, MarianConfig

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions

DEFAULT_MODEL = "qwen2.5-coder-14b-32k:latest"
MAX_TEXT_LEN = 3000

SYNTHETIC_TARGET = 80  # target samples for very rare classes via LLM
BACKTRANS_TARGET_MIN = 50  # minimum samples after back-translation for moderate rare

_sv_to_en_model = None
_en_to_sv_model = None
_sv_to_en_tokenizer = None
_en_to_sv_tokenizer = None
_device = None


def _get_translation_models():
    global _sv_to_en_model, _en_to_sv_model
    global _sv_to_en_tokenizer, _en_to_sv_tokenizer
    global _device

    if _sv_to_en_model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading translation models on {_device}...", file=sys.stderr)

        # Swedish -> English
        _sv_to_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-sv-en")
        _sv_to_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-sv-en").to(_device)

        # English -> Swedish
        _en_to_sv_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-sv")
        _en_to_sv_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-sv").to(_device)

        print(f"Translation models loaded.", file=sys.stderr)

    return _sv_to_en_model, _en_to_sv_model, _sv_to_en_tokenizer, _en_to_sv_tokenizer, _device


def back_translate(text: str, batch_size: int = 8) -> Optional[str]:
    """Back-translate Swedish text: sv->en->sv."""
    try:
        sv_en_model, en_sv_model, sv_en_tok, en_sv_tok, device = _get_translation_models()

        # Swedish -> English
        inputs = sv_en_tok([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            translated = sv_en_model.generate(**inputs, max_length=512)
        english = sv_en_tok.batch_decode(translated, skip_special_tokens=True)[0]

        # English -> Swedish
        inputs = en_sv_tok([english], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            translated = en_sv_model.generate(**inputs, max_length=512)
        swedish = en_sv_tok.batch_decode(translated, skip_special_tokens=True)[0]

        return swedish
    except Exception as e:
        print(f"  Back-translation failed: {e}", file=sys.stderr)
        return None


def _build_synthetic_prompt(category: str, categories: Dict, examples: List[str], num_examples: int = 3) -> str:
    """Build prompt for LLM to generate synthetic motion in a given category."""
    cat_def = categories[category].definition
    sample_texts = "\n\n".join(f"Exempel {i+1}:\n---\n{e[:1000]}\n---" for i, e in enumerate(examples[:num_examples]))

    prompt = f"""Du är en svensk riksdagsledamot som skriver en motion i kategorin "{category}".

Definition av kategorin:
{cat_def}

{sample_texts}

Uppgift: Skriv en NY, AUTENTISK motion på svenska som tydligt tillhör kategorin "{category}". Motionen ska vara 3-10 meningar lång och innehålla konkreta politiska förslag. Använd riksdagsterminologi.

Skriv ENDAST motionstexten, ingen introduktion eller avslutning."""
    return prompt


def generate_synthetic_motion(category: str, categories: Dict, examples: List[str], model: str = DEFAULT_MODEL, temperature: float = 0.8) -> Optional[str]:
    """Use Ollama to generate a synthetic parliamentary motion for a category."""
    prompt = _build_synthetic_prompt(category, categories, examples)
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        text = response["message"]["content"].strip()
        # Remove markdown fences if any
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        return text[:MAX_TEXT_LEN]
    except Exception as e:
        print(f"  Synthetic generation failed: {e}", file=sys.stderr)
        return None


def load_gold_labels(conn: sqlite3.Connection) -> List[Tuple]:
    cur = conn.cursor()
    cur.execute("SELECT motion_id, category, reasoning FROM gold_labels ORDER BY category")
    return cur.fetchall()


def get_motion_text(conn: sqlite3.Connection, motion_id: str) -> Optional[str]:
    cur = conn.cursor()
    cur.execute("SELECT text FROM normalized_motions WHERE id = ?", (motion_id,))
    row = cur.fetchone()
    return row[0] if row else None


def create_augmented_table(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS augmented_gold_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            motion_id TEXT NOT NULL,
            category TEXT NOT NULL,
            reasoning TEXT,
            source TEXT NOT NULL,  -- 'original', 'backtrans', 'synthetic'
            parent_id TEXT,        -- for backtrans: original motion_id; for synthetic: None
            text TEXT NOT NULL,
            created_at TEXT
        )
    """)
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_aug_gold ON augmented_gold_labels(motion_id)")
    conn.commit()


def augment_gold_labels(
    db_path: str = "data/swedish_parliament.db",
    model: str = DEFAULT_MODEL,
    backtrans_temperature: float = 0.8,
):
    conn = init_db(db_path)
    create_augmented_table(conn)
    defs = load_definitions()

    gold = load_gold_labels(conn)

    # Group by category
    by_cat: Dict[str, List[Tuple]] = {}
    for motion_id, cat, reasoning in gold:
        by_cat.setdefault(cat, []).append((motion_id, reasoning))

    print(f"Gold label distribution:", file=sys.stderr)
    for cat, items in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(items)}", file=sys.stderr)

    # Determine augmentation strategy per category
    targets = {}
    for cat, items in by_cat.items():
        count = len(items)
        if cat in ("far_left", "far_right"):
            # Extremely rare: full LLM synthetic generation
            targets[cat] = {"synthetic": max(0, SYNTHETIC_TARGET - count), "backtrans": 0}
        elif count < 100:
            # Moderate rare: back-translate all to ~double, synthetic only if still short
            bt_count = count
            synth_needed = max(0, BACKTRANS_TARGET_MIN - count * 2)
            targets[cat] = {"synthetic": synth_needed, "backtrans": bt_count}
        else:
            # Common: 20% back-translation only (no synthetic)
            targets[cat] = {"synthetic": 0, "backtrans": max(1, count // 5)}

    print(f"\nAugmentation plan:", file=sys.stderr)
    for cat, t in sorted(targets.items()):
        print(f"  {cat}: synthetic={t['synthetic']}, backtrans={t['backtrans']}", file=sys.stderr)

    # Process each category
    total_inserted = 0
    total_failed = 0

    for cat, items in by_cat.items():
        print(f"\nProcessing {cat} ({len(items)} originals)...", file=sys.stderr)

        # Get texts for originals
        originals_with_text = []
        for motion_id, reasoning in items:
            text = get_motion_text(conn, motion_id)
            if text and len(text) > 50:
                originals_with_text.append((motion_id, reasoning, text))

        # Insert originals first (if not already present)
        cur = conn.cursor()
        for motion_id, reasoning, text in originals_with_text:
            cur.execute("""
                INSERT OR IGNORE INTO augmented_gold_labels (motion_id, category, reasoning, source, parent_id, text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (motion_id, cat, reasoning, "original", None, text, datetime.now(timezone.utc).isoformat()))
        conn.commit()

        # Back-translation
        bt_target = targets[cat]["backtrans"]
        if bt_target > 0:
            # Sample which originals to back-translate
            sample = random.sample(originals_with_text, min(bt_target, len(originals_with_text)))
            print(f"  Back-translating {len(sample)} motions...", file=sys.stderr)
            for i, (motion_id, reasoning, text) in enumerate(sample):
                bt_text = back_translate(text[:1000])
                if bt_text:
                    bt_id = f"{motion_id}_bt{i}"
                    cur.execute("""
                        INSERT OR IGNORE INTO augmented_gold_labels (motion_id, category, reasoning, source, parent_id, text, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (bt_id, cat, reasoning, "backtrans", motion_id, bt_text, datetime.now(timezone.utc).isoformat()))
                    conn.commit()
                    total_inserted += 1
                else:
                    total_failed += 1

        # Synthetic generation
        synth_target = targets[cat]["synthetic"]
        if synth_target > 0:
            # Collect example texts for prompt
            examples = [text for _, _, text in originals_with_text[:5]]
            print(f"  Generating {synth_target} synthetic motions via LLM...", file=sys.stderr)
            for i in range(synth_target):
                synth_text = generate_synthetic_motion(cat, defs, examples, model=model)
                if synth_text:
                    synth_id = f"synth_{cat}_{i}_{int(time.time())}"
                    cur.execute("""
                        INSERT OR IGNORE INTO augmented_gold_labels (motion_id, category, reasoning, source, parent_id, text, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (synth_id, cat, f"LLM-generated synthetic for {cat}", "synthetic", None, synth_text, datetime.now(timezone.utc).isoformat()))
                    conn.commit()
                    total_inserted += 1
                else:
                    total_failed += 1

    # Summary
    cur = conn.cursor()
    cur.execute("SELECT category, source, COUNT(*) FROM augmented_gold_labels GROUP BY category, source ORDER BY category")
    summary = cur.fetchall()
    print(f"\n=== Augmented dataset summary ===", file=sys.stderr)
    for row in summary:
        print(f"  {row[0]} ({row[1]}): {row[2]}", file=sys.stderr)

    print(f"\nTotal inserted: {total_inserted}, failed: {total_failed}", file=sys.stderr)
    return total_inserted, total_failed


def main():
    parser = argparse.ArgumentParser(description="Augment gold labels")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    total, failed = augment_gold_labels(db_path=args.db, model=args.model)
    print(f"Augmented dataset: {total} new samples, {failed} failures")


if __name__ == "__main__":
    main()
