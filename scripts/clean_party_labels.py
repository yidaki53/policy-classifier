#!/usr/bin/env python3
"""Normalize and clean party abbreviations in normalized_motions.

Some rows have mis-identified party labels from earlier text-extraction heuristics
(e.g. single Swedish words like "bör", "år", "är").  Others have valid parties in
mixed case or historical abbreviations.  This script normalizes known variants and
sets anything outside the valid set to NULL.

Usage:
    poetry run python scripts/clean_party_labels.py --db data/swedish_parliament.db
"""

import argparse
import sqlite3
import sys

VALID_PARTIES = {"S", "M", "C", "L", "KD", "V", "MP", "SD"}

# Map historical / lowercase variants to canonical forms
PARTY_MAP = {
    "s": "S", "m": "M", "c": "C", "v": "V", "mp": "MP", "kd": "KD",
    "fp": "L", "fp": "L",
    "vpk": "V",
    "apk": "S",
    "-": None,
}


def normalize_party(p: str | None) -> str | None:
    if not p:
        return None
    p = p.strip()
    if not p:
        return None
    # Direct mapping
    if p in PARTY_MAP:
        return PARTY_MAP[p]
    # Uppercase if already valid
    up = p.upper()
    if up in VALID_PARTIES:
        return up
    # Anything else is garbage
    return None


def clean_parties(db_path: str, batch_size: int = 500):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT id, party FROM normalized_motions WHERE party IS NOT NULL")
    rows = cur.fetchall()
    total = len(rows)
    print(f"Scanning {total} rows with non-NULL party...", file=sys.stderr)

    updated = 0
    unchanged = 0
    update_batch: list[tuple] = []

    for i, row in enumerate(rows, start=1):
        old = row["party"]
        new = normalize_party(old)
        if new != old:
            update_batch.append((new, row["id"]))
            updated += 1
        else:
            unchanged += 1

        if len(update_batch) >= batch_size:
            cur.executemany(
                "UPDATE normalized_motions SET party = ? WHERE id = ?",
                update_batch,
            )
            conn.commit()
            update_batch.clear()
            if i % 5000 == 0:
                pct = i / total * 100 if total else 0
                print(f"  {i}/{total} ({pct:.1f}%) — updated={updated}, unchanged={unchanged}", file=sys.stderr)

    if update_batch:
        cur.executemany(
            "UPDATE normalized_motions SET party = ? WHERE id = ?",
            update_batch,
        )
        conn.commit()

    conn.close()
    print(f"\nDone: updated={updated}, unchanged={unchanged} out of {total}", file=sys.stderr)
    return updated, unchanged


def main():
    parser = argparse.ArgumentParser(description="Clean and normalize party labels")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    clean_parties(args.db, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
