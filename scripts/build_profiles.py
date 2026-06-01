#!/usr/bin/env python3
"""Build per-party ideological profiles from motions, votes, and speeches (Parquet-only).

Outputs a Parquet table with columns: `party`, `modality`, `category`, `proportion`.
Recency weighting uses w = exp(-lambda * delta_years) relative to the latest date
found across motions, speeches and votes (configurable `--lam`, default 0.3).
"""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

from swedish_parliament_policy_classifier.analysis.speech_visualizations import (
    load_speech_classifications,
    load_speech_metadata,
)


def _parse_date(val):
    try:
        return pd.to_datetime(val, errors="coerce")
    except Exception:
        return pd.NaT


def _recency_weight(d, reference_date: pd.Timestamp, lam: float) -> float:
    if pd.isna(d) or lam == 0.0:
        return 1.0
    delta_years = max((reference_date - d).days / 365.25, 0.0)
    return math.exp(-lam * delta_years)


def compute_motion_profiles(parquet_dir: str, lam: float, reference_date: pd.Timestamp):
    cls_path = os.path.join(parquet_dir, "classifications.parquet")
    nm_path = os.path.join(parquet_dir, "normalized_motions.parquet")

    cls = pd.read_parquet(cls_path, columns=["motion_id", "category", "normalized_weight"]).copy()
    try:
        nm = pd.read_parquet(nm_path, columns=["id", "party", "date", "doc_type"]).copy()
    except Exception:
        # some parquet exports may omit optional columns; read full table and fill missing cols
        nm = pd.read_parquet(nm_path).copy()
        for _c in ("id", "party", "date", "doc_type"):
            if _c not in nm.columns:
                nm[_c] = None
    nm["id"] = nm["id"].astype(str)

    df = cls.merge(nm, left_on="motion_id", right_on="id", how="inner")
    df = df[df["party"].notna()].copy()
    df["normalized_weight"] = pd.to_numeric(df["normalized_weight"], errors="coerce").fillna(0.0)
    # date column may be missing or nested; be defensive and fall back to NaT
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    else:
        df["date_parsed"] = pd.NaT
    df["recency_w"] = df["date_parsed"].apply(lambda d: _recency_weight(d, reference_date, lam))
    df["w"] = df["normalized_weight"] * df["recency_w"]

    grp = df.groupby(["party", "category"], as_index=False)["w"].sum().rename(columns={"w": "weight"})
    totals = grp.groupby("party", as_index=False)["weight"].sum().rename(columns={"weight": "sum_w"})
    out = grp.merge(totals, on="party", how="left")
    out["proportion"] = out.apply(lambda r: float(r["weight"] / r["sum_w"]) if r["sum_w"] > 0 else 0.0, axis=1)
    out["modality"] = "motion"
    return out[["party", "modality", "category", "proportion"]]


def compute_vote_profiles(parquet_dir: str, votering_dir: str, lam: float, reference_date: pd.Timestamp):
    # Build mapping: first_votering_id -> motion_id -> category, and motion dates if available
    mv_path = os.path.join(parquet_dir, "motion_votes.parquet")
    cls_path = os.path.join(parquet_dir, "classifications.parquet")
    nm_path = os.path.join(parquet_dir, "normalized_motions.parquet")

    if not os.path.exists(mv_path):
        return pd.DataFrame(columns=["party", "modality", "category", "proportion"]) 
    mv = pd.read_parquet(mv_path, columns=["motion_id", "first_votering_id"]).fillna("")
    mv = mv[mv["first_votering_id"] != ""].copy()
    cls = pd.read_parquet(cls_path, columns=["motion_id", "category", "normalized_weight"]).copy()
    nm = pd.read_parquet(nm_path, columns=["id", "date"]).copy()
    nm["id"] = nm["id"].astype(str)
    if "date" in nm.columns:
        nm["date_parsed"] = pd.to_datetime(nm["date"], errors="coerce", utc=True)
    else:
        nm["date_parsed"] = pd.NaT

    # top category per motion
    cls_sorted = cls.sort_values(["motion_id", "normalized_weight"], ascending=[True, False])
    top = cls_sorted.groupby("motion_id", sort=False).first().reset_index()
    motion_to_cat = dict(zip(top["motion_id"].astype(str), top["category"].astype(str)))
    motion_to_date = dict(zip(nm["id"].astype(str), nm["date_parsed"]))

    votering_map = dict(zip(mv["first_votering_id"].astype(str), mv["motion_id"].astype(str)))

    # iterate votering files and sum recency-weighted vote counts by party/category
    counts = defaultdict(float)
    import glob

    pattern = os.path.join(votering_dir, "*.parquet")
    for p in sorted(glob.glob(pattern)):
        try:
            df = pd.read_parquet(p, columns=["votering_id", "parti", "rost", "datum"]) 
        except Exception:
            continue
        df = df[df["votering_id"].isin(votering_map.keys())]
        if df.empty:
            continue
        df = df[df["rost"].astype(str).str.strip().str.lower() == "ja"]
        if df.empty:
            continue
        for _, r in df.iterrows():
            vot_id = str(r["votering_id"])
            party = r["parti"]
            vot_date = pd.to_datetime(r.get("datum", None), errors="coerce", utc=True)
            motion_id = votering_map.get(vot_id)
            cat = motion_to_cat.get(motion_id)
            # prefer votering date for recency; fallback to motion date
            use_date = vot_date if not pd.isna(vot_date) else motion_to_date.get(motion_id)
            w = _recency_weight(use_date, reference_date, lam)
            if cat and party:
                counts[(party, cat)] += float(w)

    rows = [(party, cat, cnt) for (party, cat), cnt in counts.items()]
    if not rows:
        return pd.DataFrame(columns=["party", "modality", "category", "proportion"]) 
    df = pd.DataFrame(rows, columns=["party", "category", "weight"]) 
    totals = df.groupby("party", as_index=False)["weight"].sum().rename(columns={"weight": "sum_w"})
    out = df.merge(totals, on="party", how="left")
    out["proportion"] = out.apply(lambda r: float(r["weight"] / r["sum_w"]) if r["sum_w"] > 0 else 0.0, axis=1)
    out["modality"] = "vote"
    return out[["party", "modality", "category", "proportion"]]


def compute_speech_profiles(speech_classifications_path: str, speech_parquet_dir: str, lam: float, reference_date: pd.Timestamp):
    if not os.path.exists(speech_classifications_path):
        return pd.DataFrame(columns=["party", "modality", "category", "proportion"])
    cls = load_speech_classifications(speech_classifications_path)
    meta = load_speech_metadata(speech_parquet_dir)
    merged = cls.merge(meta, left_on="speech_id", right_on="speech_id", how="left")
    merged["date_parsed"] = pd.to_datetime(merged.get("date", None), errors="coerce", utc=True)
    merged["recency_w"] = merged["date_parsed"].apply(lambda d: _recency_weight(d, reference_date, lam))
    merged["w"] = merged["normalized_weight"] * merged["recency_w"]
    merged["party"] = merged["party"].fillna("Unknown")

    grp = merged.groupby(["party", "category"], as_index=False)["w"].sum().rename(columns={"w": "weight"})
    totals = grp.groupby("party", as_index=False)["weight"].sum().rename(columns={"weight": "sum_w"})
    out = grp.merge(totals, on="party", how="left")
    out["proportion"] = out.apply(lambda r: float(r["weight"] / r["sum_w"]) if r["sum_w"] > 0 else 0.0, axis=1)
    out["modality"] = "speech"
    return out[["party", "modality", "category", "proportion"]]


def main():
    parser = argparse.ArgumentParser(description="Build per-party profiles from Parquet with recency weighting")
    parser.add_argument("--parquet-dir", default="data/parquet")
    parser.add_argument("--speech-classifications", default="data/parquet/speech_classifications_with_rhetoric_full.parquet")
    parser.add_argument("--speech-parquet-dir", default="data/speeches/parquet")
    parser.add_argument("--votering-dir", default="data/votering/parquet")
    parser.add_argument("--lam", type=float, default=0.3)
    parser.add_argument("--out", default="data/parquet/party_profiles_recency.parquet")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"Output {args.out} exists. Use --force to overwrite.")
        return

    # Compute a conservative reference date: use latest normalized motion date if present
    nm_path = os.path.join(args.parquet_dir, "normalized_motions.parquet")
    nm = pd.read_parquet(nm_path, columns=["date"]) if os.path.exists(nm_path) else pd.DataFrame()
    motion_dates = pd.to_datetime(nm["date"], errors="coerce", utc=True) if not nm.empty else pd.Series([], dtype="datetime64[ns, UTC]")

    # speech latest date
    speech_meta = load_speech_metadata(args.speech_parquet_dir)
    speech_dates = pd.to_datetime(speech_meta["date"], errors="coerce", utc=True) if not speech_meta.empty else pd.Series([], dtype="datetime64[ns, UTC]")

    all_dates = pd.concat([motion_dates.dropna(), speech_dates.dropna()])
    reference_date = all_dates.max() if not all_dates.empty else pd.Timestamp.now(tz="UTC")

    print(f"Reference date for recency weighting: {reference_date}")

    motions = compute_motion_profiles(args.parquet_dir, args.lam, reference_date)
    votes = compute_vote_profiles(args.parquet_dir, args.votering_dir, args.lam, reference_date)
    speeches = compute_speech_profiles(args.speech_classifications, args.speech_parquet_dir, args.lam, reference_date)

    out = pd.concat([motions, votes, speeches], ignore_index=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_parquet(args.out, index=False, compression="zstd")
    print(f"Wrote party profiles to {args.out} (rows={len(out)})")


if __name__ == "__main__":
    main()
