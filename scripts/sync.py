#!/usr/bin/env python3
"""Incremental sync with the Riksdag open-data endpoint and idempotent insert into Parquet."""

import argparse
import json
from datetime import datetime
from typing import Optional
from pathlib import Path

import pandas as pd

from swedish_parliament_policy_classifier.fetch import fetch_recent_motions


def _append_lineage(lineage_out: str, table: str, subject_id: str, operation: str):
    out_p = lineage_out
    row = {"table": table, "subject_id": subject_id, "operation": operation, "timestamp": datetime.utcnow().isoformat()}
    try:
        prev = pd.read_parquet(out_p)
        out_df = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    except Exception:
        out_df = pd.DataFrame([row])
    # write
    out_df.to_parquet(out_p, index=False)


def sync_parquet(out_path: str = "data/parquet/raw_motions.parquet", lineage_out: str = "data/parquet/lineage.parquet", limit: int = 100, dry_run: bool = False, query: Optional[str] = None):
    motions = fetch_recent_motions(sample=False, limit=limit, query=query)
    out_p = out_path
    try:
        prev = pd.read_parquet(out_p)
        prev_ids = set(prev["id"].astype(str).unique()) if "id" in prev.columns else set()
    except Exception:
        prev = None
        prev_ids = set()

    inserted = 0
    rows = []
    for m in motions:
        mid = m.get("id")
        if not mid:
            continue
        if str(mid) in prev_ids:
            continue
        if dry_run:
            print("[dry-run] would insert", mid)
            continue
        raw_json = json.dumps(m, ensure_ascii=False)
        retrieved_at = datetime.utcnow().isoformat()
        rows.append({"id": mid, "json": raw_json, "retrieved_at": retrieved_at, "checksum": None})
        inserted += 1
        # record lineage row later

    if rows:
        chunk = pd.DataFrame(rows)
        out_dir = Path(out_p).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        if prev is not None:
            out_df = pd.concat([prev, chunk], ignore_index=True)
        else:
            out_df = chunk
        out_df.to_parquet(out_p, index=False)
        for r in rows:
            _append_lineage(lineage_out, "raw_motions", str(r["id"]), "sync")

    print(f"Inserted {inserted} new raw motions into {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Sync recent Riksdag motions into Parquet")
    parser.add_argument("--out", default="data/parquet/raw_motions.parquet")
    parser.add_argument("--lineage-out", default="data/parquet/lineage.parquet")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--query", default=None)
    args = parser.parse_args()
    sync_parquet(args.out, args.lineage_out, args.limit, args.dry_run, args.query)


if __name__ == "__main__":
    main()
