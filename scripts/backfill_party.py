#!/usr/bin/env python3
"""Backfill party affiliations from raw_motions JSON into normalized_motions.

Bulk dataset JSON stores party in ``dokumentstatus.dokintressent.intressent[].partibet``.
The original importer only checked ``dokumentstatus.dokument.parti`` (always NULL),
so ~194k motions have ``party=NULL``.  This script re-parses ``raw_motions.json``
for all rows where ``normalized_motions.party IS NULL``, extracts the party via
:classifier.party_extractor:`, and updates the table in batches.

Usage:
    poetry run python scripts/backfill_party.py --db data/swedish_parliament.db
"""

import argparse
import json
import sqlite3
import sys

from swedish_parliament_policy_classifier.classifier.party_extractor import extract_party_from_intressent


def backfill_parties(db_path: str, batch_size: int = 500):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Count how many rows need updating before we start
    cur.execute("SELECT COUNT(*) FROM normalized_motions WHERE party IS NULL")
    total_null = cur.fetchone()[0]
    print(f"Found {total_null} normalized motions with party=NULL", file=sys.stderr)
    if total_null == 0:
        conn.close()
        return 0, 0, 0

    # Pull raw JSON for every normalized motion that currently lacks a party
    cur.execute(
        """
        SELECT nm.id, rm.json
        FROM normalized_motions nm
        JOIN raw_motions rm ON nm.id = rm.id
        WHERE nm.party IS NULL
        """
    )
    rows = cur.fetchall()
    total = len(rows)
    assert total == total_null, "Mismatch between count and fetched rows"

    updated = 0
    failed = 0
    unchanged = 0
    update_batch: list[tuple] = []

    for i, row in enumerate(rows, start=1):
        raw_json = row["json"]
        motion_id = row["id"]

        try:
            data = json.loads(raw_json)
        except Exception:
            failed += 1
            continue

        party = extract_party_from_intressent(data)
        if party:
            update_batch.append((party, motion_id))
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
            pct = i / total * 100 if total else 0
            print(
                f"  {i}/{total} ({pct:.1f}%) — updated={updated}, unchanged={unchanged}, failed={failed}",
                file=sys.stderr,
            )

    if update_batch:
        cur.executemany(
            "UPDATE normalized_motions SET party = ? WHERE id = ?",
            update_batch,
        )
        conn.commit()

    conn.close()
    print(
        f"\nDone: updated={updated}, unchanged={unchanged}, failed={failed} out of {total}",
        file=sys.stderr,
    )
    return updated, unchanged, failed


def main():
    parser = argparse.ArgumentParser(description="Backfill party affiliations from raw JSON")
    parser.add_argument("--db", default="data/swedish_parliament.db", help="Path to SQLite database")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    backfill_parties(args.db, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
