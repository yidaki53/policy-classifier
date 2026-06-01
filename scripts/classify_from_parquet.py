#!/usr/bin/env python3
"""Batch classify motions from parquet using baseline ensemble, write to SQLite, and generate figures.

Reads normalized_motions.parquet directly (bypassing the 200k SQLite table),
classifies motions with text, writes classifications to SQLite, then generates
all manuscript figures.

Usage:
    uv run python scripts/classify_and_generate_figures.py --db data/swedish_parliament.db --out-dir figures/manuscript
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions, score_motion
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher
from swedish_parliament_policy_classifier.nlp.topic_modeler import load_topic_distributions
from swedish_parliament_policy_classifier.classifier.ensemble import load_meta_classifier

MAX_TEXT_LEN = 2500
BATCH_SIZE = 500


def classify_from_parquet(
    db_path: str,
    parquet_path: str = "data/parquet/normalized_motions.parquet",
    max_text_len: int = MAX_TEXT_LEN,
    batch_size: int = BATCH_SIZE,
    meta_classifier_path: Optional[str] = None,
):
    """Classify motions with text from parquet, write to SQLite."""
    conn = init_db(db_path)
    cur = conn.cursor()
    defs = load_definitions()

    # Load topic distributions
    topic_distributions = load_topic_distributions()
    print(f"Loaded topic distributions for {len(topic_distributions)} motions.", file=sys.stderr)

    # Embedding matcher
    matcher = None
    try:
        matcher = EmbeddingMatcher()
        if matcher.model is None:
            matcher = None
    except Exception as e:
        print(f"Embedding matcher unavailable: {e}", file=sys.stderr)

    # Read motions with text from parquet
    print(f"Reading {parquet_path}...", file=sys.stderr)
    df = pd.read_parquet(parquet_path)
    df = df[df["text"].notna() & (df["text"].str.len() > 100)].copy()
    total = len(df)
    print(f"Motions with text >100 chars: {total}", file=sys.stderr)

    # Check which are already classified
    cur.execute("SELECT DISTINCT motion_id FROM classifications")
    classified = {r[0] for r in cur.fetchall()}
    df = df[~df["id"].isin(classified)].copy()
    print(f"Unclassified motions: {len(df)}", file=sys.stderr)

    if len(df) == 0:
        print("All motions already classified.")
        return 0

    batch = []
    processed = 0
    start_time = time.time()

    for _, row in df.iterrows():
        motion_id = row["id"]
        text = (row.get("text") or "")[:max_text_len]
        party = row.get("party")

        results = score_motion(
            motion_id,
            text,
            defs,
            party=party,
            embedding_matcher=matcher,
            use_zero_shot=True,
            topic_distributions=topic_distributions if topic_distributions else None,
            use_meta_classifier=bool(topic_distributions),
            meta_classifier_path=meta_classifier_path,
        )

        for r in results:
            batch.append((
                r.motion_id,
                r.category,
                float(r.raw_score),
                float(r.normalized_weight),
                json.dumps(r.matched_rules, ensure_ascii=False),
                r.classifier_version,
                r.created_at.isoformat(),
            ))

        processed += 1
        if len(batch) >= batch_size * len(defs):
            _insert_batch(conn, cur, batch)
            batch.clear()
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(df) - processed) / rate / 3600 if rate > 0 else 0
            print(
                f"  {processed}/{len(df)}  ({processed/len(df)*100:.1f}%)  rate={rate:.1f} motions/s  ETA={eta:.1f}h",
                file=sys.stderr,
            )

    if batch:
        _insert_batch(conn, cur, batch)
        batch.clear()

    elapsed = time.time() - start_time
    print(f"\nDone: classified {processed} motions in {elapsed:.1f}s ({processed/elapsed:.1f} motions/s)", file=sys.stderr)
    return processed


def _insert_batch(conn, cur, batch: list):
    cur.executemany(
        """INSERT INTO classifications (
            motion_id, category, raw_score, normalized_weight, matched_rules,
            classifier_version, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        batch,
    )
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Classify motions and generate manuscript figures")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--parquet", default="data/parquet/normalized_motions.parquet")
    parser.add_argument("--out-dir", default="figures/manuscript")
    parser.add_argument("--meta-classifier", default=None, help="Path to custom meta-classifier (e.g., hybrid)")
    args = parser.parse_args()

    total = classify_from_parquet(
        db_path=args.db,
        parquet_path=args.parquet,
        meta_classifier_path=args.meta_classifier,
    )
    print(f"Classified {total} motions")


if __name__ == "__main__":
    main()
