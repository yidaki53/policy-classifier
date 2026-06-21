#!/usr/bin/env python3
"""Classify structured interpellation turns and persist results to Parquet.

Reads from data/parquet/interpellations.parquet, classifies each turn,
and appends classification rows to data/parquet/classifications.parquet.

Turn types:
  - ip_question: the asking party's statement (says)
  - ip_answer: the minister's response (says)
  - ip_followup: follow-up question (says)

Usage:
    uv run python scripts/classify_interpellations.py
    uv run python scripts/classify_interpellations.py --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from swedish_parliament_policy_classifier.exports import load_definitions, classify_motion


def classify_interpellations_parquet(
    ip_parquet: str | Path = "data/parquet/interpellations.parquet",
    classifications_out: str | Path = "data/parquet/classifications.parquet",
    limit: Optional[int] = None,
) -> int:
    defs = load_definitions()
    ip_p = Path(ip_parquet)
    out_p = Path(classifications_out)

    if not ip_p.exists():
        print(f"No interpellations parquet found at {ip_p}")
        return 0

    ip_df = pd.read_parquet(ip_p)

    # Determine already classified turns (check both motion_id and speech_id columns
    # because the shared parquet may have been written by different classifiers)
    classified_ids: set[str] = set()
    if out_p.exists():
        try:
            prev = pd.read_parquet(out_p)
            id_cols = [c for c in ("motion_id", "speech_id") if c in prev.columns]
            if id_cols:
                for col in id_cols:
                    classified_ids.update(prev[col].dropna().astype(str).unique())
        except Exception:
            classified_ids = set()

    to_classify = ip_df[~ip_df["id"].astype(str).isin(classified_ids)].copy()
    if limit:
        to_classify = to_classify.head(limit)

    q_count = len(to_classify[to_classify["turn_type"] == "ip_question"])
    a_count = len(to_classify[to_classify["turn_type"] == "ip_answer"])
    print(f"Classifying {len(to_classify)} turns (questions={q_count}, answers={a_count}) ...", file=sys.stderr)

    rows = []
    it = list(to_classify.iterrows())
    _last_report_ts = 0.0
    for idx, (_, r) in enumerate(it, start=1):
        turn_id = str(r.get("id"))
        text = str(r.get("text") or "")
        import time
        now = time.monotonic()
        if idx == 1 or idx % 50 == 0 or idx == len(it) or (now - _last_report_ts) > 30:
            msg = f"  classify_interpellations: {idx}/{len(it)} turns"
            print(msg, file=sys.stderr, flush=True)
            _last_report_ts = now
        try:
            results = classify_motion(motion_id=turn_id, text=text, categories=defs)
        except Exception as e:
            print(f"Failed to classify {turn_id}: {e}", file=sys.stderr)
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
    parser = argparse.ArgumentParser(description="Classify interpellation turns and persist to Parquet")
    parser.add_argument("--ip-parquet", default="data/parquet/interpellations.parquet")
    parser.add_argument("--classifications-out", default="data/parquet/classifications.parquet")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    classified = classify_interpellations_parquet(
        args.ip_parquet,
        args.classifications_out,
        limit=args.limit,
    )
    print(f"Appended {classified} classification rows to {args.classifications_out}")


if __name__ == "__main__":
    main()

if False:
    # Graphify hint: interpellation classification uses the same classify_motion() pipeline
    from swedish_parliament_policy_classifier.exports import load_definitions, classify_motion