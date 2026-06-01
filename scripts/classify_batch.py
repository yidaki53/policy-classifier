#!/usr/bin/env python3
"""Batch classify all motions with v0.4.0 scorer using batched DB inserts.

This is significantly faster than scripts/classify.py because it:
- Truncates text to MAX_TEXT_LEN (default 2500) chars
- Loads category embeddings once
- Uses executemany for batch INSERTs (one commit per batch)
- Skips per-row commit overhead
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.db.readers import fetch_unclassified_motions
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions, score_motion
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher
from swedish_parliament_policy_classifier.nlp.topic_modeler import load_topic_distributions
from swedish_parliament_policy_classifier.analysis.aggregate import compute_party_profiles
from swedish_parliament_policy_classifier.visualization.plot_party_profiles import plot_party_profiles

MAX_TEXT_LEN = 2500
BATCH_SIZE = 500


def classify_batch(
    db_path: str = "data/swedish_parliament.db",
    max_text_len: int = MAX_TEXT_LEN,
    batch_size: int = BATCH_SIZE,
    use_embeddings: bool = True,
    use_zero_shot: bool = True,
    meta_classifier_path: Optional[str] = None,
):
    conn = init_db(db_path)
    cur = conn.cursor()
    defs = load_definitions()

    matcher = None
    if use_embeddings:
        try:
            matcher = EmbeddingMatcher()
            if matcher.model is None:
                matcher = None
        except Exception as e:
            print(f"Embedding matcher unavailable: {e}", file=sys.stderr)
            matcher = None

    # Load topic distributions if available
    topic_distributions = load_topic_distributions()
    if topic_distributions:
        print(f"Loaded topic distributions for {len(topic_distributions)} motions.", file=sys.stderr)
    else:
        print("No topic distributions available; meta-classifier will not be used.", file=sys.stderr)

    # Fetch all unclassified motions (reader will prefer parquet if exported)
    rows = fetch_unclassified_motions(conn)
    total = len(rows)
    print(f"Found {total} unclassified motions", file=sys.stderr)
    if total == 0:
        print("All motions already classified. Skipping.")
        return 0

    batch: list[tuple] = []
    processed = 0
    start_time = time.time()

    for row in rows:
        # `rows` is a list of NormalizedMotion (Pydantic) or sqlite3.Row; handle both
        if hasattr(row, "id"):
            motion_id = row.id
            text = (row.text or "")[:max_text_len]
            party = row.party
        else:
            motion_id = row[0]
            text = (row[2] or "")[:max_text_len]
            party = row[4]

        results = score_motion(
            motion_id,
            text,
            defs,
            party=party,
            embedding_matcher=matcher,
            use_zero_shot=use_zero_shot,
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
            eta = (total - processed) / rate / 3600 if rate > 0 else 0
            print(
                f"  {processed}/{total}  ({processed/total*100:.1f}%)  rate={rate:.1f} motions/s  ETA={eta:.1f}h",
                file=sys.stderr,
            )

    # Flush remaining
    if batch:
        _insert_batch(conn, cur, batch)
        batch.clear()

    elapsed = time.time() - start_time
    print(f"\nDone: classified {processed} motions in {elapsed:.1f}s ({processed/elapsed:.1f} motions/s)", file=sys.stderr)

    # Recompute profiles and generate figure
    print("Computing party profiles...", file=sys.stderr)
    compute_party_profiles(conn)
    out = plot_party_profiles(conn)
    print(f"Figure saved: {out}", file=sys.stderr)
    return processed


def _insert_batch(conn, cur, batch: list[tuple]):
    cur.executemany(
        """INSERT INTO classifications (
            motion_id, category, raw_score, normalized_weight, matched_rules,
            classifier_version, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        batch,
    )
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Batch classify motions with batched DB inserts")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--max-text-len", type=int, default=MAX_TEXT_LEN, help="Truncate motion text to N chars")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Rows per INSERT batch")
    parser.add_argument("--no-embeddings", dest="use_embeddings", action="store_false")
    parser.add_argument("--no-zero-shot", dest="use_zero_shot", action="store_false")
    parser.add_argument("--meta-classifier", default=None, help="Path to custom meta-classifier (e.g., hybrid ensemble)")
    args = parser.parse_args()

    total = classify_batch(
        db_path=args.db,
        max_text_len=args.max_text_len,
        batch_size=args.batch_size,
        use_embeddings=args.use_embeddings,
        use_zero_shot=args.use_zero_shot,
        meta_classifier_path=args.meta_classifier,
    )
    print(f"Classified {total} motions")


if __name__ == "__main__":
    main()
