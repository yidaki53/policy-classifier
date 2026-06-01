#!/usr/bin/env python3
"""
Ingest rhetoric labels CSV into `speech_rhetoric_labels` table.

CSV expected columns: speech_id, irony, sarcasm, posturing, none, top_label, reasoning

Example:
  uv run python3 scripts/ingest_rhetoric_labels.py --csv logs/rhetoric_llm_labels_selected_10.csv --db data/swedish_parliament.db
"""

import argparse
import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


def upsert_rhetoric(conn, sid, irony, sarcasm, posturing, none, top_label, reasoning):
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    # delete existing then insert for idempotence
    cur.execute("DELETE FROM speech_rhetoric_labels WHERE speech_id = ?", (sid,))
    cur.execute(
        "INSERT INTO speech_rhetoric_labels (speech_id, irony, sarcasm, posturing, none, top_label, reasoning, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, float(irony or 0.0), float(sarcasm or 0.0), float(posturing or 0.0), float(none or 0.0), top_label, reasoning, now),
    )
    # lineage
    try:
        cur.execute(
            "INSERT INTO lineage (source_table, source_id, operation, timestamp, notes) VALUES (?, ?, ?, ?, ?)",
            ("speech_rhetoric_labels", sid, "ingest_zero_shot", now, "source=llm"),
        )
    except Exception:
        pass
    conn.commit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--db", default="data/swedish_parliament.db")
    args = p.parse_args()

    csvp = Path(args.csv)
    if not csvp.exists():
        raise SystemExit(f"CSV/Parquet not found: {csvp}")

    # Ensure DB schema is initialized / migrated
    from swedish_parliament_policy_classifier.db.schema import init_db

    conn = init_db(args.db)

    # Support both CSV and Parquet inputs
    if csvp.suffix.lower() == '.parquet':
        df = pd.read_parquet(csvp)
        for _, r in df.iterrows():
            sid = r.get("speech_id")
            irony = r.get("irony") or r.get("irony_score") or 0.0
            sarcasm = r.get("sarcasm") or 0.0
            posturing = r.get("posturing") or r.get("posturing_score") or 0.0
            none = r.get("none") or 0.0
            top_label = r.get("top_label") or ""
            reasoning = r.get("reasoning") or r.get("reason") or ""
            if not sid:
                print("Skipping row with no speech_id", r)
                continue
            upsert_rhetoric(conn, sid, irony, sarcasm, posturing, none, top_label, reasoning)
            print(f"Ingested rhetoric for {sid}")
    else:
        with open(csvp, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                sid = r.get("speech_id")
                irony = r.get("irony") or r.get("irony_score") or 0.0
                sarcasm = r.get("sarcasm") or 0.0
                posturing = r.get("posturing") or r.get("posturing") or r.get("posturing_score") or 0.0
                none = r.get("none") or 0.0
                top_label = r.get("top_label") or ""
                reasoning = r.get("reasoning") or r.get("reason") or ""
                if not sid:
                    print("Skipping row with no speech_id", r)
                    continue
                upsert_rhetoric(conn, sid, irony, sarcasm, posturing, none, top_label, reasoning)
                print(f"Ingested rhetoric for {sid}")


if __name__ == '__main__':
    main()
