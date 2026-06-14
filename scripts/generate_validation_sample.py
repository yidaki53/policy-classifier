#!/usr/bin/env python3
"""Generate a random held-out validation sample of speech IDs.

Usage:
    uv run python scripts/generate_validation_sample.py --n 200 --out logs/validation_sample_ids.txt
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", default="logs/validation_sample_ids.txt")
    p.add_argument("--parquet-dir", default="data/speeches/parquet")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    parquet_dir = Path(args.parquet_dir)
    files = sorted(parquet_dir.glob("*.parquet"))
    all_ids = []
    for f in files:
        df = pd.read_parquet(f, columns=["anforande_id"])
        all_ids.extend(df["anforande_id"].astype(str).tolist())

    selected = random.sample(all_ids, min(args.n, len(all_ids)))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sid in selected:
            f.write(f"{sid}\n")

    print(f"Wrote {len(selected)} IDs to {out_path}")


if __name__ == "__main__":
    main()
