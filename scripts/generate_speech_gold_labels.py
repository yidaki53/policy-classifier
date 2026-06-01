#!/usr/bin/env python3
"""Generate gold labels for speeches using Scandi NLI zero-shot classification.

Samples speeches stratified by their current top-category classification,
uses alexandrainst/scandi-nli-large-v2 to assign ideology labels,
and stores results in speech_gold_labels.

Usage:
    uv run python scripts/generate_speech_gold_labels.py --db data/swedish_parliament.db --target-per-category 30
"""

import argparse
import gc
import json
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import pipeline

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions

DEFAULT_MODEL = "alexandrainst/scandi-nli-large-v2"
TARGET_PER_CATEGORY = 30
MAX_TEXT_LEN = 3000
CATEGORIES = ["far_left", "left", "centre_left", "centre", "centre_right", "right", "far_right"]


def _list_speech_parquets() -> list[Path]:
    d = Path("data/speeches/parquet")
    return sorted(d.glob("*.parquet"))


def _already_gold(conn: sqlite3.Connection, speech_id: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM speech_gold_labels WHERE speech_id = ?", (speech_id,))
    return cur.fetchone() is not None


def _load_classifier(model_name: str = DEFAULT_MODEL):
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading zero-shot classifier: {model_name} (device={device})", file=sys.stderr)
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )
    return classifier


def _build_hypothesis_template() -> str:
    """Build a stance-aware hypothesis template in Swedish."""
    return "Detta tal uttrycker {}."


def _build_candidate_labels(definitions: Dict) -> List[str]:
    """Build Swedish candidate labels from category definitions."""
    labels = []
    for cat in CATEGORIES:
        d = definitions.get(cat)
        if d:
            # Combine name with short definition
            labels.append(f"{cat}: {d.definition[:120]}")
        else:
            labels.append(cat)
    return labels


def _classify_speech(
    text: str,
    classifier,
    candidate_labels: List[str],
    max_text_len: int = MAX_TEXT_LEN,
) -> Tuple[str, Dict[str, float], str]:
    """Classify a single speech. Returns (category, scores_dict, reasoning)."""
    truncated = text[:max_text_len]
    
    result = classifier(
        truncated,
        candidate_labels=candidate_labels,
        hypothesis_template=_build_hypothesis_template(),
        multi_label=False,
    )
    
    # Map back from full label strings to category names
    scores = {}
    for label, score in zip(result["labels"], result["scores"]):
        cat = label.split(":")[0].strip()
        scores[cat] = float(score)
    
    top_cat = max(scores, key=scores.get)
    
    # Build a simple reasoning string
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    reasoning = f"Top 3: {', '.join(f'{c}={s:.3f}' for c, s in sorted_scores[:3])}"
    
    return top_cat, scores, reasoning


def _sample_speeches_by_category(
    conn: sqlite3.Connection,
    target_per_category: int,
) -> Dict[str, List[str]]:
    """Return speech IDs sampled per top-category, proportional with min floor."""
    cur = conn.cursor()
    
    # Get all classifications to find top category per speech
    cur.execute("""
        SELECT speech_id, category, normalized_weight
        FROM speech_classifications
    """)
    rows = cur.fetchall()
    
    # Find top category per speech
    speech_best = {}
    for speech_id, category, weight in rows:
        if speech_id not in speech_best or weight > speech_best[speech_id][1]:
            speech_best[speech_id] = (category, weight)
    
    # Group by category
    by_cat = {cat: [] for cat in CATEGORIES}
    for speech_id, (cat, weight) in speech_best.items():
        by_cat.setdefault(cat, []).append((speech_id, weight))
    
    # Calculate targets proportional with min floor
    totals = {cat: len(items) for cat, items in by_cat.items()}
    total_speeches = sum(totals.values())
    
    targets = {}
    for cat in CATEGORIES:
        proportional = int(target_per_category * 2.5 * totals.get(cat, 0) / total_speeches) if total_speeches > 0 else 0
        targets[cat] = max(target_per_category, proportional)
    
    # Sample
    sampled = {}
    for cat in CATEGORIES:
        items = by_cat.get(cat, [])
        if not items:
            sampled[cat] = []
            continue
        target = min(targets[cat], len(items))
        # Sample higher-confidence speeches (better top weights)
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        # Take top half by confidence, then random sample
        top_half = items_sorted[:max(1, len(items_sorted)//2)]
        selected = random.sample(top_half, min(target, len(top_half)))
        if len(selected) < target and len(items_sorted) > len(top_half):
            remaining = [i for i in items_sorted[len(top_half):] if i not in selected]
            if remaining:
                extra = random.sample(remaining, min(target - len(selected), len(remaining)))
                selected.extend(extra)
        sampled[cat] = [sid for sid, _ in selected]
    
    return sampled


def _load_speech_text(speech_id: str, parquet_files: List[Path]) -> Optional[str]:
    """Find speech text across parquet files."""
    # Build a lookup from the first file that contains this ID
    for f in parquet_files:
        df = pd.read_parquet(f, columns=["anforande_id", "anforandetext"])
        match = df[df["anforande_id"] == speech_id]
        if len(match) > 0:
            text = match.iloc[0]["anforandetext"]
            if text and isinstance(text, str):
                return text
    return None


def _insert_gold_label(
    conn: sqlite3.Connection,
    speech_id: str,
    category: str,
    reasoning: str,
    model: str,
    scores: Dict[str, float],
):
    cur = conn.cursor()
    cur.execute(
        """INSERT OR REPLACE INTO speech_gold_labels
           (speech_id, category, reasoning, prompt_version, model, temperature, raw_response, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            speech_id,
            category,
            reasoning,
            "scandi-nli-stance-v1.0",
            model,
            0.0,  # NLI models don't have temperature
            json.dumps(scores, ensure_ascii=False),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def generate_speech_gold_labels(
    db_path: str = "data/swedish_parliament.db",
    target_per_category: int = TARGET_PER_CATEGORY,
    model_name: str = DEFAULT_MODEL,
    resume: bool = True,
):
    conn = init_db(db_path)
    defs = load_definitions()
    
    # Load classifier
    classifier = _load_classifier(model_name)
    candidate_labels = _build_candidate_labels(defs)
    
    # Sample speeches
    print(f"Sampling speeches (target ~{target_per_category} per category)...", file=sys.stderr)
    sampled = _sample_speeches_by_category(conn, target_per_category)
    
    all_speech_ids = []
    for cat, ids in sampled.items():
        print(f"  {cat}: {len(ids)} samples", file=sys.stderr)
        all_speech_ids.extend(ids)
    
    if resume:
        all_speech_ids = [sid for sid in all_speech_ids if not _already_gold(conn, sid)]
        print(f"After skipping already-labeled: {len(all_speech_ids)} remaining", file=sys.stderr)
    
    random.shuffle(all_speech_ids)
    
    # Load parquet files list once
    parquet_files = _list_speech_parquets()
    if not parquet_files:
        print("No speech parquet files found!", file=sys.stderr)
        return 0
    
    inserted = 0
    failed = 0
    start_time = time.time()
    
    for i, speech_id in enumerate(all_speech_ids):
        text = _load_speech_text(speech_id, parquet_files)
        if text is None:
            failed += 1
            continue
        
        try:
            category, scores, reasoning = _classify_speech(
                text, classifier, candidate_labels
            )
        except Exception as e:
            print(f"  Classification failed for {speech_id}: {e}", file=sys.stderr)
            failed += 1
            continue
        
        _insert_gold_label(conn, speech_id, category, reasoning, model_name, scores)
        inserted += 1
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(all_speech_ids) - (i + 1)) / rate / 60 if rate > 0 else 0
            print(
                f"  {i+1}/{len(all_speech_ids)}  inserted={inserted} failed={failed}  "
                f"rate={rate:.1f}/s  ETA={eta:.1f}m",
                file=sys.stderr,
            )
            sys.stderr.flush()
        
        # Periodic GPU cleanup
        if (i + 1) % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    elapsed = time.time() - start_time
    print(
        f"\nDone: {inserted} speech gold labels inserted, {failed} failures in {elapsed:.1f}s",
        file=sys.stderr,
    )
    conn.close()
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Generate speech gold labels using Scandi NLI")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--target-per-category", type=int, default=TARGET_PER_CATEGORY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()
    
    total = generate_speech_gold_labels(
        db_path=args.db,
        target_per_category=args.target_per_category,
        model_name=args.model,
        resume=args.resume,
    )
    print(f"Total speech gold labels: {total}")


if __name__ == "__main__":
    main()
