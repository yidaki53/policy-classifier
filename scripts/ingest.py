#!/usr/bin/env python3
"""Simple ingest CLI: fetch motions (sample) and store raw JSON into sqlite."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from swedish_parliament_policy_classifier.fetch import fetch_recent_motions
from swedish_parliament_policy_classifier.db.schema import init_db, get_connection


def ingest(sample: bool = True, db_path: str | Path = "data/swedish_parliament.db"):
    conn = init_db(db_path)
    cur = conn.cursor()

    motions = fetch_recent_motions(sample=sample)
    inserted = 0
    for m in motions:
        motion_id = m.get("id")
        raw_json = json.dumps(m, ensure_ascii=False)
        retrieved_at = datetime.utcnow().isoformat()
        cur.execute(
            "INSERT OR IGNORE INTO raw_motions (id, json, retrieved_at, checksum) VALUES (?, ?, ?, ?)",
            (motion_id, raw_json, retrieved_at, None),
        )
        if cur.rowcount:
            inserted += 1

    conn.commit()
    print(f"Inserted {inserted} new raw motions into {db_path}")


def main():
    parser = argparse.ArgumentParser(description="Ingest Riksdag motions (sample mode)")
    parser.add_argument("--no-sample", dest="sample", action="store_false", help="Do not use the built-in sample dataset")
    parser.add_argument("--db", dest="db_path", default="data/swedish_parliament.db", help="Path to sqlite DB")
    args = parser.parse_args()
    ingest(sample=args.sample, db_path=args.db_path)


if __name__ == "__main__":
    main()
