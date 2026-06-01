#!/usr/bin/env python3
"""
Apply manual speech corrections (CSV) into `speech_gold_labels`.

CSV format: speech_id,corrected_category,annotator,notes

This script upserts rows into `speech_gold_labels` and records the change as lineage.
"""

import argparse
import sqlite3
import csv
from datetime import datetime, timezone
from pathlib import Path


def init_db_conn(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def upsert_speech_gold(conn, speech_id: str, category: str, annotator: str, notes: str):
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    # Upsert: delete existing then insert to keep schema simple
    cur.execute("DELETE FROM speech_gold_labels WHERE speech_id = ?", (speech_id,))
    cur.execute(
        "INSERT INTO speech_gold_labels (speech_id, category, reasoning, prompt_version, model, temperature, raw_response, created_at) VALUES (?, ?, ?, NULL, NULL, NULL, NULL, ?)",
        (speech_id, category, notes or None, now),
    )
    # lineage
    cur.execute(
        "INSERT INTO lineage (source_table, source_id, operation, timestamp, notes) VALUES (?, ?, ?, ?, ?)",
        ("speech_gold_labels", speech_id, "manual_upsert", now, f"annotator={annotator}"),
    )
    conn.commit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/swedish_parliament.db")
    p.add_argument("--csv", required=True, help="CSV with speech_id,corrected_category,annotator,notes")
    args = p.parse_args()

    conn = init_db_conn(args.db)
    Path(args.csv).expanduser()
    with open(args.csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            sid = r.get("speech_id") or r.get("id")
            cat = r.get("corrected_category") or r.get("category")
            annotator = r.get("annotator") or "manual"
            notes = r.get("notes") or ""
            if not sid or not cat:
                print(f"Skipping invalid row: {r}")
                continue
            upsert_speech_gold(conn, sid, cat, annotator, notes)
            print(f"Upserted {sid} -> {cat}")


if __name__ == "__main__":
    main()
