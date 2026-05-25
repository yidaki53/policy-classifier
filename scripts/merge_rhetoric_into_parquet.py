#!/usr/bin/env python3
"""Merge `speech_rhetoric_labels.parquet` into `speech_gold_labels.parquet`.

Writes an augmented Parquet file with rhetoric numeric scores and `top_label`.

Usage:
    uv run python3 scripts/merge_rhetoric_into_parquet.py --parquet-dir data/parquet --out data/parquet/augmented_speech_gold_labels.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", default="data/parquet", help="Parquet directory")
    p.add_argument("--out", default=None, help="Output parquet path (defaults to augmented_speech_gold_labels.parquet)")
    args = p.parse_args()

    parquet_dir = Path(args.parquet_dir)
    out_path = Path(args.out) if args.out else parquet_dir / "augmented_speech_gold_labels.parquet"

    gold_path = parquet_dir / "speech_gold_labels.parquet"
    rhet_path = parquet_dir / "speech_rhetoric_labels.parquet"

    if not gold_path.exists():
        raise SystemExit(f"speech gold parquet not found: {gold_path}")
    if not rhet_path.exists():
        raise SystemExit(f"speech rhetoric parquet not found: {rhet_path}")

    gold = pd.read_parquet(gold_path)
    rhetoric = pd.read_parquet(rhet_path)

    if "speech_id" not in gold.columns:
        raise SystemExit("speech_id missing from gold parquet")
    if "speech_id" not in rhetoric.columns:
        raise SystemExit("speech_id missing from rhetoric parquet")

    # Merge rhetoric into gold (left join so all gold rows are preserved)
    merged = gold.merge(rhetoric, on="speech_id", how="left", suffixes=("", "_rhet"))

    # Ensure numeric rhetoric cols exist and fill missing with 0.0
    for c in ("irony", "sarcasm", "posturing", "none"):
        if c in merged.columns:
            merged[c] = merged[c].fillna(0.0)
        else:
            merged[c] = 0.0

    # If top_label exists, keep it; otherwise add <none>
    if "top_label" not in merged.columns:
        merged["top_label"] = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print("Wrote augmented parquet:", out_path)


if __name__ == "__main__":
    main()
