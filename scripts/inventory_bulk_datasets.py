#!/usr/bin/env python3
"""Inventory local Riksdag bulk datasets and summarize coverage."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _extract_year_tokens(name: str) -> list[int]:
    years = []
    for m in re.finditer(r"(19\d{2}|20\d{2})", name):
        try:
            years.append(int(m.group(1)))
        except Exception:
            continue
    return years


def main() -> None:
    p = argparse.ArgumentParser(description="Inventory local bulk datasets")
    p.add_argument("--bulk-dir", default="data/bulk_datasets")
    p.add_argument("--out", default="output/analysis/bulk_dataset_inventory.parquet")
    p.add_argument("--summary-out", default="output/analysis/bulk_dataset_inventory_summary.json")
    args = p.parse_args()

    bulk_dir = Path(args.bulk_dir)
    if not bulk_dir.exists():
        raise FileNotFoundError(f"Missing bulk dataset directory: {bulk_dir}")

    rows = []
    for f in sorted(bulk_dir.glob("*.zip")):
        stem = f.name.split("-")[0].lower() if "-" in f.name else f.stem.lower()
        years = _extract_year_tokens(f.name)
        rows.append(
            {
                "file": f.name,
                "kind": stem,
                "size_bytes": int(f.stat().st_size),
                "min_year_token": min(years) if years else None,
                "max_year_token": max(years) if years else None,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No bulk zip datasets found in {bulk_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="zstd")

    by_kind = (
        df.groupby("kind", as_index=False)
        .agg(
            n_files=("file", "size"),
            total_size_bytes=("size_bytes", "sum"),
            min_year=("min_year_token", "min"),
            max_year=("max_year_token", "max"),
        )
        .sort_values("kind")
    )

    summary = {
        "bulk_dir": str(bulk_dir),
        "n_files": int(len(df)),
        "kinds": by_kind.to_dict(orient="records"),
        "output": str(out_path),
    }

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
