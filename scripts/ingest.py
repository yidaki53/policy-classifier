#!/usr/bin/env python3
"""Simple ingest CLI: fetch motions (sample) and store raw JSON into Parquet.

This replaces the legacy sqlite-backed `raw_motions` table with an appendable
Parquet file at `data/parquet/raw_motions.parquet` by default.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

from swedish_parliament_policy_classifier.fetch import fetch_recent_motions


def ingest_to_parquet(sample: bool = True, out_path: str | Path = "data/parquet/raw_motions.parquet") -> int:
    out = Path(out_path)
    motions = fetch_recent_motions(sample=sample)
    rows: List[dict] = []
    for m in motions:
        motion_id = m.get("id")
        if not motion_id:
            continue
        raw_json = json.dumps(m, ensure_ascii=False)
        retrieved_at = datetime.now(timezone.utc).isoformat()
        rows.append({"id": motion_id, "json": raw_json, "retrieved_at": retrieved_at, "checksum": None})

    if not rows:
        return 0

    chunk = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try:
            prev = pd.read_parquet(out)
            # Avoid duplicating existing motion ids
            prev_ids = set(prev["id"].astype(str).unique()) if "id" in prev.columns else set()
            new_chunk = chunk[~chunk["id"].astype(str).isin(prev_ids)]
            if new_chunk.empty:
                return 0
            out_df = pd.concat([prev, new_chunk], ignore_index=True)
        except Exception:
            out_df = pd.concat([chunk], ignore_index=True)
    else:
        out_df = chunk

    out_df.to_parquet(out, index=False, compression="zstd")
    return len(out_df) if out_df is not None else 0


def main():
    parser = argparse.ArgumentParser(description="Ingest Riksdag motions (sample mode) and persist to Parquet")
    parser.add_argument("--no-sample", dest="sample", action="store_false", help="Do not use the built-in sample dataset")
    parser.add_argument("--out", dest="out_path", default="data/parquet/raw_motions.parquet", help="Parquet output path")
    args = parser.parse_args()
    written = ingest_to_parquet(sample=args.sample, out_path=args.out_path)
    print(f"Wrote {written} raw motions to {args.out_path}")


if __name__ == "__main__":
    main()
