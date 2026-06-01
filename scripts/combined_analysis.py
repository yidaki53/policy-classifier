#!/usr/bin/env python3
"""Build combined votes × party × category from Parquet inputs.

Logic:
 - pick the top category per motion from `data/parquet/classifications.parquet`
 - map motions -> first_votering_id via `data/parquet/motion_votes.parquet`
 - stream `data/votering/parquet/*.parquet` and count +1 for every "Ja" vote
   cast by a party for a votering that matches a motion's first_votering_id

Outputs a Parquet file with columns: `party`, `category`, `votes_count`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, Set

import pandas as pd


JA_VALUES: Set[str] = {"ja", "j", "1", "yes", "y", "för", "for", "for"}


def load_top_category_per_motion(classifications_path: str) -> Dict[str, str]:
    df = pd.read_parquet(classifications_path, columns=["motion_id", "category", "normalized_weight"])
    df = df.dropna(subset=["motion_id"])  # be defensive
    # If there are multiple rows per motion, take argmax normalized_weight
    df_sorted = df.sort_values(["motion_id", "normalized_weight"], ascending=[True, False])
    top = df_sorted.groupby("motion_id", sort=False).first().reset_index()
    mapping = dict(zip(top["motion_id"].astype(str), top["category"].astype(str)))
    return mapping


def load_motion_votes_map(motion_votes_path: str) -> Dict[str, str]:
    df = pd.read_parquet(motion_votes_path, columns=["motion_id", "first_votering_id"]).fillna("")
    df = df[df["first_votering_id"] != ""]
    return dict(zip(df["first_votering_id"].astype(str), df["motion_id"].astype(str)))


def votering_files(votering_dir: str) -> Iterable[str]:
    pattern = os.path.join(votering_dir, "*.parquet")
    for p in sorted(glob.glob(pattern)):
        yield p


def is_ja(vote_val: str) -> bool:
    if not isinstance(vote_val, str):
        return False
    return vote_val.strip().lower() in JA_VALUES


def aggregate_votes(
    votering_dir: str,
    votering_to_motion: Dict[str, str],
    motion_to_category: Dict[str, str],
) -> Counter:
    counts = Counter()
    # Build votering_id -> category mapping for faster lookup
    votering_to_category: Dict[str, str] = {}
    for vot_id, motion_id in votering_to_motion.items():
        cat = motion_to_category.get(motion_id)
        if cat:
            votering_to_category[vot_id] = cat

    if not votering_to_category:
        return counts

    for p in votering_files(votering_dir):
        try:
            df = pd.read_parquet(p, columns=["votering_id", "parti", "rost"]) 
        except Exception:
            continue
        df = df[df["votering_id"].isin(votering_to_category.keys())]
        if df.empty:
            continue
        # Filter to 'Ja' votes
        df["_is_ja"] = df["rost"].astype(str).str.strip().str.lower().isin(JA_VALUES)
        df = df[df["_is_ja"]]
        # Group by party and votering_id
        grouped = df.groupby(["parti", "votering_id"]).size().reset_index(name="n")
        for _, row in grouped.iterrows():
            party = row["parti"]
            vot_id = row["votering_id"]
            cat = votering_to_category.get(vot_id)
            if not cat:
                continue
            counts[(party, cat)] += int(row["n"])

    return counts


def counts_to_frame(counts: Counter) -> pd.DataFrame:
    rows = [(party, cat, cnt) for (party, cat), cnt in counts.items()]
    if not rows:
        return pd.DataFrame(columns=["party", "category", "votes_count"])
    df = pd.DataFrame(rows, columns=["party", "category", "votes_count"])
    return df


def main():
    parser = argparse.ArgumentParser(description="Combined motions+votes aggregation (Parquet-only)")
    parser.add_argument("--classifications", default="data/parquet/classifications.parquet")
    parser.add_argument("--motion-votes", default="data/parquet/motion_votes.parquet")
    parser.add_argument("--votering-dir", default="data/votering/parquet")
    parser.add_argument("--out", default="data/parquet/combined_votes_party_category.parquet")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"Output {args.out} already exists. Use --force to overwrite.")
        return

    print("Loading top categories per motion...")
    motion_to_category = load_top_category_per_motion(args.classifications)
    print(f"Top categories for {len(motion_to_category)} motions loaded.")

    print("Loading motion -> first_votering mapping...")
    votering_to_motion = load_motion_votes_map(args.motion_votes)
    print(f"Found {len(votering_to_motion)} votering->motion mappings.")

    print("Aggregating votes from votering Parquets (this may take a while)...")
    counts = aggregate_votes(args.votering_dir, votering_to_motion, motion_to_category)
    df_out = counts_to_frame(counts)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df_out.to_parquet(args.out, index=False, compression="zstd")
    print(f"Wrote combined counts to {args.out} (rows={len(df_out)})")


if __name__ == "__main__":
    main()
