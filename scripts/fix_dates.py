#!/usr/bin/env python3
"""Fix incorrect dates in normalized_motions where publicerad was used instead of datum.

For bulk-imported historical motions, the Riksdagen API sets publicerad to the
digitization date (e.g. 2014-10-10) rather than the actual document date.
This script corrects those by reading the true datum from raw_motions JSON.

Usage:
    uv run python scripts/fix_dates.py --db data/swedish_parliament.db
"""

import argparse
import json
import sqlite3
import sys


def fix_dates(db_path: str):
    conn = sqlite3.connect(db_path, timeout=60)
    cur = conn.cursor()

    # WAL mode + relaxed sync for bulk writes
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")

    # Find rows where date is 2014-10-10 (the digitization date)
    cur.execute("""
        SELECT nm.id, rm.json
        FROM normalized_motions nm
        JOIN raw_motions rm ON nm.id = rm.id
        WHERE nm.date = '2014-10-10 00:00:00'
    """)
    rows = cur.fetchall()
    print(f"Found {len(rows):,} rows with incorrect 2014-10-10 date", file=sys.stderr)

    if not rows:
        conn.close()
        return 0

    updates = []
    for row in rows:
        mid = row[0]
        d = json.loads(row[1])
        doc = d.get("dokumentstatus", {}).get("dokument", {})
        # Prefer datum (actual document date), fall back to publicerad
        correct_date = doc.get("datum") or doc.get("dok_datum") or doc.get("publicerad") or None
        if correct_date:
            updates.append((correct_date, mid))

    print(f"Updating {len(updates):,} rows in batches of 200", file=sys.stderr)

    BATCH = 200
    done = 0
    for i in range(0, len(updates), BATCH):
        batch = updates[i : i + BATCH]
        cur.executemany("UPDATE normalized_motions SET date = ? WHERE id = ?", batch)
        conn.commit()
        done += len(batch)
        print(f"  {done:,}/{len(updates):,}", file=sys.stderr)

    conn.close()
    print(f"Done: {done:,} rows updated", file=sys.stderr)
    return done


def main():
    parser = argparse.ArgumentParser(description="Fix incorrect dates in normalized_motions")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    args = parser.parse_args()
    fix_dates(args.db)


if __name__ == "__main__":
    main()
