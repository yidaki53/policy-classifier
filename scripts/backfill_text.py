#!/usr/bin/env python3
"""Backfill normalized_motions text from raw_motions JSON html field.

Re-parses raw_motions JSON for all normalized rows with short/empty text,
extracts full text from the html field when available, and updates
normalized_motions in-place.

Usage:
    uv run python3 scripts/backfill_text.py --db data/swedish_parliament.db --min-len 50
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from swedish_parliament_policy_classifier.nlp.preprocess import extract_plain_text_from_html


def backfill(db_path: str = "data/swedish_parliament.db", min_len: int = 50, batch_size: int = 500):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT nm.id, nm.title, nm.text, rm.json
        FROM normalized_motions nm
        JOIN raw_motions rm ON nm.id = rm.id
        """
    )
    rows = cur.fetchall()
    total = len(rows)
    print(f"Processing {total} normalized motions (will update where text is missing or too short).", file=sys.stderr)
    if total == 0:
        conn.close()
        return 0, 0

    updated = 0
    failed = 0
    update_batch: list[tuple] = []

    for i, row in enumerate(rows, start=1):
        raw_json = row["json"]
        motion_id = row["id"]
        try:
            data = json.loads(raw_json)
        except Exception:
            failed += 1
            continue

        # Unwrap bulk dataset schema
        if isinstance(data, dict) and "dokumentstatus" in data:
            ds = data.get("dokumentstatus")
            if isinstance(ds, dict):
                inner = ds.get("dokument") or ds
                if isinstance(inner, dict):
                    data = inner

        # Try primary text fields first
        text = (
            data.get("doktext")
            or data.get("text")
            or data.get("body")
            or data.get("sammanfattning")
            or data.get("undertitel")
            or ""
        )

        # Fallback to HTML
        if not text or len(text) < min_len:
            html = data.get("html") or ""
            if html:
                text = extract_plain_text_from_html(html)

        # Fallback to title
        if not text:
            text = row["title"] or ""

        if text and len(text) >= min_len:
            # Only queue update if text actually changed
            existing = row["text"] or ""
            if text != existing:
                update_batch.append((text, motion_id))
                updated += 1
        else:
            failed += 1

        if len(update_batch) >= batch_size:
            cur.executemany(
                "UPDATE normalized_motions SET text = ? WHERE id = ?",
                update_batch,
            )
            conn.commit()
            update_batch.clear()

        if i % 1000 == 0 or i == total:
            print(
                f"  {i}/{total} processed — updated={updated}, failed={failed}",
                file=sys.stderr,
            )

    # Flush remaining batch
    if update_batch:
        cur.executemany(
            "UPDATE normalized_motions SET text = ? WHERE id = ?",
            update_batch,
        )
        conn.commit()

    conn.close()
    print(f"\nDone: updated={updated}, failed={failed} out of {total}", file=sys.stderr)
    return updated, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill normalized_motions text from raw_motions html")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    backfill(args.db, args.min_len, args.batch_size)
