#!/usr/bin/env python3
"""Incremental sync with the Riksdag open-data endpoint and idempotent insert into sqlite."""

import argparse
import json
from datetime import datetime, timezone
from typing import Optional

from swedish_parliament_policy_classifier.fetch import fetch_recent_motions
from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.persist import record_lineage


def sync(
    db_path: str = "data/swedish_parliament.db",
    limit: int = 100,
    dry_run: bool = False,
    query: Optional[str] = None,
    doktyps: Optional[list[str]] = None,
    parties: Optional[list[str]] = None,
):
    conn = init_db(db_path)
    cur = conn.cursor()
    if not doktyps:
        doktyps = ["mot"]

    inserted_total = 0
    for dok in doktyps:
        motions = fetch_recent_motions(sample=False, limit=limit, query=query, doktyp=dok, parties=parties)
        inserted = 0
        for m in motions:
            mid = m.get("id")
            if not mid:
                continue
            cur.execute("SELECT 1 FROM raw_motions WHERE id = ?", (mid,))
            if cur.fetchone():
                continue
            if dry_run:
                print(f"[dry-run] would insert ({dok})", mid)
                continue
            raw_json = json.dumps(m, ensure_ascii=False)
            retrieved_at = datetime.now(timezone.utc).isoformat()
            cur.execute(
                "INSERT OR IGNORE INTO raw_motions (id, json, retrieved_at, checksum, doc_type) VALUES (?, ?, ?, ?, ?)",
                (mid, raw_json, retrieved_at, None, dok),
            )
            if cur.rowcount:
                inserted += 1
                try:
                    record_lineage(conn, "raw_motions", mid, "sync", checksum=None)
                except Exception:
                    pass

        conn.commit()
        print(f"Inserted {inserted} new raw documents (type={dok}) into {db_path}")
        inserted_total += inserted

    print(f"Total inserted: {inserted_total}")


def main():
    parser = argparse.ArgumentParser(description="Sync recent Riksdag motions into sqlite")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--query", default=None)
    parser.add_argument("--doktyp", nargs="+", default=["mot"], help="Document types to fetch (e.g. mot, prop, votering)")
    parser.add_argument("--party", dest="parties", action="append", help="Party code to filter (repeatable)")
    args = parser.parse_args()
    sync(args.db, args.limit, args.dry_run, args.query, doktyps=args.doktyp, parties=args.parties)


if __name__ == "__main__":
    main()
