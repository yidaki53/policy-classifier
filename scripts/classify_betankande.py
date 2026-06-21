#!/usr/bin/env python3
"""Classify normalized betankande (committee reports) and persist results to Parquet.

Reads from data/parquet/betankande_normalized.parquet, classifies each row,
and appends classification rows to data/parquet/classifications.parquet.

Usage:
    uv run python scripts/classify_betankande.py
    uv run python scripts/classify_betankande.py --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from swedish_parliament_policy_classifier.exports import load_definitions, classify_motion


def classify_betankande_parquet(
    betankande_parquet: str | Path = "data/parquet/betankande_normalized.parquet",
    classifications_out: str | Path = "data/parquet/classifications.parquet",
    limit: Optional[int] = None,
) -> int:
    defs = load_definitions()
    bp = Path(betankande_parquet)
    out_p = Path(classifications_out)

    if not bp.exists():
        print(f"No betankande parquet found at {bp}")
        return 0

    bet = pd.read_parquet(bp)

    # Determine already classified betankande
    classified_ids: set[str] = set()
    if out_p.exists():
        try:
            prev = pd.read_parquet(out_p, columns=["motion_id"])
            classified_ids = set(prev["motion_id"].astype(str).unique()) if not prev.empty else set()
        except Exception:
            classified_ids = set()

    to_classify = bet[~bet["id"].astype(str).isin(classified_ids)].copy()
    if limit:
        to_classify = to_classify.head(limit)

    print(f"Classifying {len(to_classify)} of {len(bet)} betankande ...", file=sys.stderr)

    rows = []
    for _, r in to_classify.iterrows():
        bid = str(r.get("id"))
        text = (r.get("title") or "") + "\n" + (r.get("text") or "")
        try:
            results = classify_motion(motion_id=bid, text=text, categories=defs)
        except Exception as e:
            print(f"Failed to classify {bid}: {e}", file=sys.stderr)
            continue

        for rr in results:
            rows.append({
                "motion_id": rr.motion_id,
                "category": rr.category,
                "raw_score": float(rr.raw_score),
                "normalized_weight": float(rr.normalized_weight),
                "matched_rules": json.dumps(rr.matched_rules, ensure_ascii=False),
                "classifier_version": rr.classifier_version,
                "created_at": rr.created_at.isoformat(),
            })

    if not rows:
        return 0

    chunk = pd.DataFrame(rows)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        try:
            prev = pd.read_parquet(out_p)
            out_df = pd.concat([prev, chunk], ignore_index=True)
            if "created_at" in out_df.columns:
                out_df = out_df.sort_values("created_at").drop_duplicates(subset=["motion_id", "category"], keep="last")
            else:
                out_df = out_df.drop_duplicates(subset=["motion_id", "category"], keep="last")
        except Exception:
            out_df = chunk
    else:
        out_df = chunk

    out_df.to_parquet(out_p, index=False, compression="zstd")
    return len(chunk)


def main():
    parser = argparse.ArgumentParser(description="Classify betankande and persist to Parquet")
    parser.add_argument("--betankande-parquet", default="data/parquet/betankande_normalized.parquet")
    parser.add_argument("--classifications-out", default="data/parquet/classifications.parquet")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    classified = classify_betankande_parquet(
        args.betankande_parquet,
        args.classifications_out,
        limit=args.limit,
    )
    print(f"Appended {classified} classification rows to {args.classifications_out}")


if __name__ == "__main__":
    main()

if False:
    # Graphify hint: betankande classification uses the same classify_motion() pipeline as motions
    from swedish_parliament_policy_classifier.exports import load_definitions, classify_motion