#!/usr/bin/env python3
"""Augment speech gold labels using back-translation and synthetic LLM generation.

Creates `augmented_speech_gold_labels` table and stores original, back-translated
and synthetic speech texts for use in training. Uses MarianMT for back-translation
and Ollama (if available) for synthetic speech generation.

Usage:
    uv run python3 scripts/augment_speech_gold_labels.py --db data/swedish_parliament.db
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

if os.getenv("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.getenv("HF_TOKEN"))

try:
    import ollama
except Exception:
    ollama = None

import torch
from transformers import MarianMTModel, MarianTokenizer

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher

DEFAULT_MODEL = "qwen2.5-coder-14b-32k:latest"
MAX_TEXT_LEN = 3000

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

        _sv_to_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-sv-en")
        _sv_to_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-sv-en").to(_device)

        _en_to_sv_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-sv")
        _en_to_sv_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-sv").to(_device)

        print(f"Translation models loaded.", file=sys.stderr)

    return _sv_to_en_model, _en_to_sv_model, _sv_to_en_tokenizer, _en_to_sv_tokenizer, _device


def back_translate(text: str) -> Optional[str]:
    try:
        sv_en_model, en_sv_model, sv_en_tok, en_sv_tok, device = _get_translation_models()

        inputs = sv_en_tok([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            translated = sv_en_model.generate(**inputs, max_length=512)
        english = sv_en_tok.batch_decode(translated, skip_special_tokens=True)[0]

        inputs = en_sv_tok([english], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            translated = en_sv_model.generate(**inputs, max_length=512)
        swedish = en_sv_tok.batch_decode(translated, skip_special_tokens=True)[0]
        return swedish
    except Exception as e:
        print(f"  Back-translation failed: {e}", file=sys.stderr)
        return None


def _build_speech_prompt(category: str, categories: Dict, examples: List[str], num_examples: int = 3) -> str:
    cat_def = categories[category].definition
    sample_texts = "\n\n".join(f"Exempel {i+1}:\n---\n{e[:1000]}\n---" for i, e in enumerate(examples[:num_examples]))

    prompt = f"""Du är en svensk riksdagsledamot som håller ett anförande i kammaren.\n\nDefinition av kategorin:\n{cat_def}\n\n{sample_texts}\n\nUppgift: Skriv ETT nytt, kort tal (3-8 meningar) på svenska som tydligt tillhör kategorin \"{category}\". Använd talspråk som är lämpligt i riksdagsanföranden och inkludera konkreta politiska budskap eller förslag. Skriv ENDAST talet."""
    return prompt


def generate_synthetic_speech(category: str, categories: Dict, examples: List[str], model: str = DEFAULT_MODEL, temperature: float = 0.8) -> Optional[str]:
    if ollama is None:
        print("Ollama not available; skipping synthetic generation.", file=sys.stderr)
        return None
    prompt = _build_speech_prompt(category, categories, examples)
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
        print(f"  Synthetic generation failed: {e}", file=sys.stderr)
        return None


def _list_speech_parquets() -> List[Path]:
    d = Path("data") / "speeches" / "parquet"
    return sorted(d.glob("*.parquet"))


def _load_speech_text(speech_id: str, parquet_files: List[Path]) -> Optional[str]:
    import pandas as pd
    for f in parquet_files:
        try:
            df = pd.read_parquet(f, columns=["anforande_id", "anforandetext"])
        except Exception:
            try:
                df = pd.read_parquet(f)
            except Exception:
                continue
        match = df[df["anforande_id"] == speech_id]
        if len(match) > 0:
            text = match.iloc[0]["anforandetext"]
            if text and isinstance(text, str):
                return text
    return None


def create_augmented_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS augmented_speech_gold_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            speech_id TEXT NOT NULL,
            category TEXT NOT NULL,
            reasoning TEXT,
            source TEXT NOT NULL,
            parent_id TEXT,
            text TEXT NOT NULL,
            created_at TEXT
        )
    """)
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_aug_speech ON augmented_speech_gold_labels(speech_id)")
    conn.commit()


def augment_speech_gold_labels(db_path: str = "data/swedish_parliament.db", model: str = DEFAULT_MODEL, manual_only: bool = False):
    conn = init_db(db_path)
    create_augmented_table(conn)
    defs = load_definitions()

    cur = conn.cursor()
    if manual_only:
        # restrict to manually curated stratified gold set where prompt_version is NULL
        cur.execute("SELECT speech_id, category, reasoning FROM speech_gold_labels WHERE prompt_version IS NULL ORDER BY category")
    else:
        cur.execute("SELECT speech_id, category, reasoning FROM speech_gold_labels ORDER BY category")
    gold = cur.fetchall()

    by_cat: Dict[str, List[Tuple]] = {}
    for sid, cat, reasoning in gold:
        by_cat.setdefault(cat, []).append((sid, reasoning))

    parquet_files = _list_speech_parquets()

    total_inserted = 0
    total_failed = 0

    for cat, items in by_cat.items():
        print(f"\nProcessing {cat} ({len(items)} originals)...", file=sys.stderr)

        originals_with_text = []
        for sid, reasoning in items:
            text = _load_speech_text(sid, parquet_files)
            if text and len(text) > 20:
                originals_with_text.append((sid, reasoning, text))

        # Insert originals into augmented table
        for sid, reasoning, text in originals_with_text:
            cur.execute("""
                INSERT OR IGNORE INTO augmented_speech_gold_labels (speech_id, category, reasoning, source, parent_id, text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (sid, cat, reasoning, "original", None, text, datetime.now(timezone.utc).isoformat()))
        conn.commit()

        # Back-translation: simple doubling strategy for small categories
        bt_count = min(len(originals_with_text), max(1, len(originals_with_text)))
        if bt_count > 0:
            sample = random.sample(originals_with_text, min(bt_count, len(originals_with_text)))
            print(f"  Back-translating {len(sample)} speeches...", file=sys.stderr)
            for i, (sid, reasoning, text) in enumerate(sample):
                bt_text = back_translate((text or "")[:1000])
                if bt_text:
                    bt_id = f"{sid}_bt{i}_{int(time.time())}"
                    cur.execute("""
                        INSERT OR IGNORE INTO augmented_speech_gold_labels (speech_id, category, reasoning, source, parent_id, text, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (bt_id, cat, reasoning, "backtrans", sid, bt_text, datetime.now(timezone.utc).isoformat()))
                    conn.commit()
                    total_inserted += 1
                else:
                    total_failed += 1

        # Synthetic generation for very small classes
        if len(originals_with_text) < 30 and ollama is not None:
            synth_target = max(0, 30 - len(originals_with_text))
            examples = [t for _, _, t in originals_with_text[:5]]
            print(f"  Generating {synth_target} synthetic speeches via LLM...", file=sys.stderr)
            for i in range(synth_target):
                synth_text = generate_synthetic_speech(cat, defs, examples, model=model)
                if synth_text:
                    synth_id = f"synth_speech_{cat}_{i}_{int(time.time())}"
                    cur.execute("""
                        INSERT OR IGNORE INTO augmented_speech_gold_labels (speech_id, category, reasoning, source, parent_id, text, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (synth_id, cat, f"LLM-generated synthetic for {cat}", "synthetic", None, synth_text, datetime.now(timezone.utc).isoformat()))
                    conn.commit()
                    total_inserted += 1
                else:
                    total_failed += 1

    # Summary
    cur.execute("SELECT category, source, COUNT(*) FROM augmented_speech_gold_labels GROUP BY category, source ORDER BY category")
    summary = cur.fetchall()
    print(f"\n=== Augmented speech dataset summary ===", file=sys.stderr)
    for row in summary:
        print(f"  {row[0]} ({row[1]}): {row[2]}", file=sys.stderr)

    print(f"\nTotal inserted: {total_inserted}, failed: {total_failed}", file=sys.stderr)
    return total_inserted, total_failed


def main():
    parser = argparse.ArgumentParser(description="Augment speech gold labels")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--manual-only", dest="manual_only", action="store_true", help="Only augment speech gold rows where prompt_version IS NULL (manual stratified set)")
    args = parser.parse_args()
    total, failed = augment_speech_gold_labels(db_path=args.db, model=args.model, manual_only=getattr(args, "manual_only", False))
    print(f"Augmented speech dataset: {total} new samples, {failed} failures")


if __name__ == "__main__":
    main()
