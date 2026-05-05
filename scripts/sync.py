#!/usr/bin/env python3
"""Incremental sync with the Riksdag open-data endpoint and idempotent insert into sqlite."""

import argparse
import json
from datetime import datetime
from typing import Optional

from swedish_parliament_policy_classifier.fetch import fetch_recent_motions
from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.persist import record_lineage


def sync(db_path: str = "data/swedish_parliament.db", limit: int = 100, dry_run: bool = False, query: Optional[str] = None):
    conn = init_db(db_path)
    cur = conn.cursor()
    motions = fetch_recent_motions(sample=False, limit=limit, query=query)
    inserted = 0
    for m in motions:
        mid = m.get("id")
        if not mid:
            continue
        cur.execute("SELECT 1 FROM raw_motions WHERE id = ?", (mid,))
        if cur.fetchone():
            continue
        if dry_run:
            print("[dry-run] would insert", mid)
            continue
        raw_json = json.dumps(m, ensure_ascii=False)
        retrieved_at = datetime.utcnow().isoformat()
        cur.execute(
            "INSERT OR IGNORE INTO raw_motions (id, json, retrieved_at, checksum) VALUES (?, ?, ?, ?)",
            (mid, raw_json, retrieved_at, None),
        )
        if cur.rowcount:
            inserted += 1
            record_lineage(conn, "raw_motions", mid, "sync")

    conn.commit()
    print(f"Inserted {inserted} new raw motions into {db_path}")


def main():
    parser = argparse.ArgumentParser(description="Sync recent Riksdag motions into sqlite")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--query", default=None)
    args = parser.parse_args()
    sync(args.db, args.limit, args.dry_run, args.query)


if __name__ == "__main__":
    main()
