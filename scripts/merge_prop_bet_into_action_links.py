#!/usr/bin/env python3
"""Merge proposition and betankande speech links into the main action links table.

Reads speech_action_links.parquet (motion/vote links) and
speech_prop_bet_links.parquet (proposition/betankande links), producing a
combined output that the axis alignment and contradiction scoring can use.

The existing links are kept as-is; prop/bet links are appended as new rows
with their respective action_type.

Usage:
    uv run python scripts/merge_prop_bet_into_action_links.py
    uv run python scripts/merge_prop_bet_into_action_links.py --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Merge prop/bet links into speech_action_links")
    p.add_argument("--existing-links", default="data/parquet/speech_action_links.parquet")
    p.add_argument("--prop-bet-links", default="data/parquet/speech_prop_bet_links.parquet")
    p.add_argument("--out", default="data/parquet/speech_action_links_with_prop_bet.parquet")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    existing_path = Path(args.existing_links)
    prop_bet_path = Path(args.prop_bet_links)
    out_path = Path(args.out)

    if out_path.exists() and not args.force:
        print(f"Output {out_path} exists. Use --force to overwrite.")
        return

    existing = pd.read_parquet(existing_path) if existing_path.exists() else pd.DataFrame()
    prop_bet = pd.read_parquet(prop_bet_path) if prop_bet_path.exists() else pd.DataFrame()

    if existing.empty:
        print("No existing links found.")
        existing = pd.DataFrame(columns=[
            "speech_id", "motion_id", "action_id", "action_type", "speech_party",
            "category", "speech_date", "action_date", "days_diff", "link_source",
        ])

    if prop_bet.empty:
        print("No prop/bet links to merge.")
        combined = existing.copy()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_path, index=False, compression="zstd")
        print(f"Copied existing links to {out_path} ({len(combined)} rows)")
        return

    # Normalize prop_bet columns to match existing link schema
    # existing has: speech_id, motion_id, action_id, action_type, speech_party, ...
    prop_bet_cols = {
        "speech_id": "speech_id",
        "motion_id": "action_id",  # rename action_id -> motion_id
        "action_id": "action_id",
        "action_type": "action_type",
        "speech_party": "speech_party",
        "category": "category",
        "speech_date": "speech_date",
        "motion_date": "action_date",  # rename action_date -> motion_date
        "days_diff": "days_diff",
        "link_source": "link_source",
        "motion_party": "action_party",  # rename action_party -> motion_party
    }

    # Build prop/bet rows that match the existing schema
    pb_rows = []
    for _, r in prop_bet.iterrows():
        pb_rows.append({
            "speech_id": str(r.get("speech_id", "")),
            "motion_id": str(r.get("action_id", "")),
            "action_id": f"{r.get('action_type', '')}:{r.get('action_id', '')}",
            "action_type": str(r.get("action_type", "")),
            "speech_party": str(r.get("speech_party", "")),
            "motion_party": str(r.get("action_party", "")),
            "category": str(r.get("category", "")),
            "speech_date": r.get("speech_date"),
            "motion_date": r.get("action_date"),
            "days_diff": r.get("days_diff"),
            "link_source": str(r.get("link_source", "")),
        })

    pb_df = pd.DataFrame(pb_rows) if pb_rows else pd.DataFrame(columns=existing.columns if not existing.empty else [
        "speech_id", "motion_id", "action_id", "action_type", "speech_party",
        "motion_party", "category", "speech_date", "motion_date", "days_diff", "link_source",
    ])

    # Fill missing columns in existing if needed
    for col in pb_df.columns:
        if col not in existing.columns:
            existing[col] = None

    combined = pd.concat([existing, pb_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["speech_id", "action_type", "motion_id"], keep="first")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False, compression="zstd")

    summary = {
        "output": str(out_path),
        "existing_rows": int(len(existing)),
        "prop_bet_rows": int(len(pb_df)),
        "total_rows": int(len(combined)),
        "action_types": sorted(combined["action_type"].unique().tolist()) if not combined.empty else [],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()