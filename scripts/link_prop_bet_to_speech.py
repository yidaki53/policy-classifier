#!/usr/bin/env python3
"""Link speeches to proposition and betankande action targets for contradiction scoring.

This script creates speech->proposition and speech->betankande links by:
  1. Explicit document ID matching (rel_dok_id / betankande ref_dok_ids)
  2. Party+category+time window fallback

Outputs: data/parquet/speech_prop_bet_links.parquet

Usage:
    uv run python scripts/link_prop_bet_to_speech.py
    uv run python scripts/link_prop_bet_to_speech.py --force
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import pandas as pd

from swedish_parliament_policy_classifier.analysis.speech_visualizations import (
    load_speech_classifications,
    load_speech_metadata,
)


def _top_category_per_speech(speech_classifications_path: str) -> pd.DataFrame:
    df = pd.read_parquet(speech_classifications_path, columns=["speech_id", "category", "normalized_weight"]).copy()
    df["speech_id"] = df["speech_id"].astype(str)
    df["normalized_weight"] = pd.to_numeric(df["normalized_weight"], errors="coerce").fillna(0.0)
    df = df.sort_values(["speech_id", "normalized_weight"], ascending=[True, False])
    top = df.groupby("speech_id", sort=False).first().reset_index()
    return top[["speech_id", "category", "normalized_weight"]].rename(columns={"normalized_weight": "speech_weight"})


def _top_category_per_motion(classifications_path: str) -> pd.DataFrame:
    df = pd.read_parquet(classifications_path, columns=["motion_id", "category", "normalized_weight"]).copy()
    df = df.sort_values(["motion_id", "normalized_weight"], ascending=[True, False])
    top = df.groupby("motion_id", sort=False).first().reset_index()
    return top[["motion_id", "category", "normalized_weight"]].rename(columns={"normalized_weight": "action_weight"})


def main() -> None:
    p = argparse.ArgumentParser(description="Link speeches to propositions and betankande for contradiction scoring")
    p.add_argument("--speech-classifications", default="data/parquet/speech_classifications_with_rhetoric_full.parquet")
    p.add_argument("--speech-parquet-dir", default="data/speeches/parquet")
    p.add_argument("--classifications", default="data/parquet/classifications.parquet")
    p.add_argument("--normalized-motions", default="data/parquet/normalized_motions.parquet")
    p.add_argument("--betankande-parquet", default="data/parquet/betankande_normalized.parquet")
    p.add_argument("--out", default="data/parquet/speech_prop_bet_links.parquet")
    p.add_argument("--window-days", type=int, default=180, help="Time window for party+category fallback")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"Output {out_path} exists. Use --force to overwrite.")
        return

    print("Loading speech classifications and metadata...")
    speech_cls = load_speech_classifications(args.speech_classifications)
    speech_meta = load_speech_metadata(args.speech_parquet_dir)
    if speech_meta.empty or speech_cls.empty:
        print("No speech metadata or classifications found; exiting.")
        return

    speech_top = _top_category_per_speech(args.speech_classifications)
    speech_df = speech_top.merge(speech_meta, left_on="speech_id", right_on="speech_id", how="left")
    speech_df = speech_df.rename(columns={"date": "speech_date", "party": "party"})
    speech_df["speech_date_parsed"] = pd.to_datetime(speech_df.get("speech_date", None), errors="coerce", utc=True)
    speech_df = speech_df[speech_df["party"].notna() & speech_df["speech_date_parsed"].notna()].copy()
    speech_df["speech_id"] = speech_df["speech_id"].astype(str)

    # Build list of speech IDs that already have motion/vote links (skip those)
    existing_links_path = "data/parquet/speech_action_links.parquet"
    existing_speech_ids: set[str] = set()
    if os.path.exists(existing_links_path):
        try:
            existing = pd.read_parquet(existing_links_path, columns=["speech_id"])
            existing_speech_ids = set(existing["speech_id"].astype(str).unique())
        except Exception:
            pass
    print(f"Existing speech_action_links cover {len(existing_speech_ids)} speeches")

    motion_top = _top_category_per_motion(args.classifications)

    # Load propositions from normalized_motions
    print("Loading propositions (doc_type=prop)...")
    nm = pd.read_parquet(args.normalized_motions)
    nm = nm.rename(columns={"id": "action_id", "date": "action_date"})
    if "action_date" not in nm.columns:
        nm["action_date"] = pd.NaT
    nm["action_date_parsed"] = pd.to_datetime(nm.get("action_date", None), errors="coerce", utc=True)
    nm["doc_type"] = nm.get("doc_type", "").astype(str)
    props = nm[nm["doc_type"] == "prop"].copy()
    props = props[props["action_date_parsed"].notna()].copy()
    props = props.merge(motion_top, left_on="action_id", right_on="motion_id", how="left")
    # Do not drop propositions just because they lack a category classification;
    # keep them with empty/default category so they still contribute as "do" actions.
    if "category" not in props.columns:
        props["category"] = ""
    else:
        props["category"] = props["category"].fillna("").astype(str)
    print(f"  {len(props)} propositions loaded (before category/motion filters)")

    # Load normalized betankande
    print("Loading betankande...")
    bet = pd.read_parquet(args.betankande_parquet)
    if "id" not in bet.columns:
        print("  No 'id' column in betankande parquet")
        bet = pd.DataFrame()
    else:
        bet = bet.rename(columns={"id": "action_id", "date": "action_date", "party": "party"})
        if "action_date" not in bet.columns:
            bet["action_date"] = pd.NaT
        bet["action_date_parsed"] = pd.to_datetime(bet.get("action_date", None), errors="coerce", utc=True)
        bet = bet[bet["action_date_parsed"].notna()].copy()
        bet = bet.merge(motion_top, left_on="action_id", right_on="motion_id", how="left")
        bet["doc_type"] = "betankande"
        if "category" not in bet.columns:
            bet["category"] = ""
        else:
            bet["category"] = bet["category"].fillna("").astype(str)
        print(f"  {len(bet)} betankande loaded (before category/motion filters)")

    # Build quick lookup of speech metadata from parquet for explicit ID linking
    print("Scanning speech parquet for explicit document references...")
    speech_refs: dict[str, dict[str, str]] = {}
    for pf in sorted(glob.glob(os.path.join(args.speech_parquet_dir, "*.parquet"))):
        try:
            sdf = pd.read_parquet(pf)
        except Exception:
            continue
        sid_col = next((c for c in ("anforande_id", "speech_id") if c in sdf.columns), None)
        if not sid_col:
            continue
        ref_cols = [c for c in ("dok_id", "rel_dok_id", "dok_doc_id", "relaterat_id") if c in sdf.columns]
        if not ref_cols:
            continue
        for _, r in sdf.iterrows():
            sid = str(r.get(sid_col, ""))
            if not sid or sid not in speech_df["speech_id"].values:
                continue
            entry = speech_refs.setdefault(sid, {})
            for col in ref_cols:
                val = r.get(col)
                if pd.notna(val):
                    entry[col] = str(val).strip()

    # Build action ID maps
    prop_map: dict[str, dict] = {}
    for _, r in props.iterrows():
        aid = str(r["action_id"]).strip()
        if aid:
            prop_map[aid.lower()] = dict(r)
    bet_map: dict[str, dict] = {}
    for _, r in bet.iterrows():
        aid = str(r["action_id"]).strip()
        if aid:
            bet_map[aid.lower()] = dict(r)

    window = pd.Timedelta(days=args.window_days)
    rows: list[dict] = []
    it = speech_df.itertuples(index=False)
    try:
        from tqdm.auto import tqdm as _tqdm
        it = _tqdm(list(speech_df.itertuples(index=False)), desc="link_prop_bet", unit="speech")
    except Exception:
        it = speech_df.itertuples(index=False)

    # Process each speech: try prop linking first, then betankande
    for s in it:
        s = {k: getattr(s, k) for k in speech_df.columns}
        sid = str(s["speech_id"])
        sid = str(s["speech_id"])
        party = str(s["party"])
        cat = str(s["category"])
        sdate = s["speech_date_parsed"]
        sref = speech_refs.get(sid, {})

        # --- Proposition linking ---
        # First: explicit ID match from speech refs
        linked_prop = None
        prop_source = None
        for col in ("rel_dok_id", "dok_id", "relaterat_id"):
            val = sref.get(col)
            if val and val.lower() in prop_map:
                linked_prop = prop_map[val.lower()]
                prop_source = f"prop_explicit_{col}"
                break
        if not linked_prop:
            # Fallback: party + category + time window, but do not require category match
            candidates = props[props["party"].astype(str).str.strip() == party].copy()
            if cat:
                cat_candidates = candidates[candidates["category"].astype(str) == cat].copy()
                if not cat_candidates.empty:
                    candidates = cat_candidates
            # If no candidate remains after optional category filter, keep party-only candidates
            time_candidates = candidates[
                (candidates["action_date_parsed"] >= sdate - window)
                & (candidates["action_date_parsed"] <= sdate + window)
            ]
            if not time_candidates.empty:
                time_candidates["days_diff"] = (time_candidates["action_date_parsed"] - sdate).abs().dt.days
                best = time_candidates.nsmallest(1, "days_diff").iloc[0]
                linked_prop = dict(best)
                prop_source = "prop_fallback_party_category_time"

        if linked_prop:
            days_diff = None
            try:
                mdate = linked_prop.get("action_date_parsed", linked_prop.get("action_date"))
                if pd.notna(sdate) and pd.notna(mdate):
                    days_diff = int(abs((pd.Timestamp(mdate) - sdate).days))
            except Exception:
                days_diff = None
            rows.append({
                "speech_id": sid,
                "action_id": str(linked_prop.get("action_id", "")),
                "action_type": "proposition",
                "speech_party": party,
                "category": cat,
                "speech_date": sdate,
                "action_date": linked_prop.get("action_date_parsed", linked_prop.get("action_date")),
                "action_party": str(linked_prop.get("party", "")),
                "action_weight": float(linked_prop.get("action_weight", 0.0)),
                "days_diff": days_diff,
                "link_source": prop_source or "prop_fallback",
            })

        # --- Betankande linking ---
        linked_bet = None
        bet_source = None
        for col in ("rel_dok_id", "dok_id", "relaterat_id"):
            val = sref.get(col)
            if val and val.lower() in bet_map:
                linked_bet = bet_map[val.lower()]
                bet_source = f"bet_explicit_{col}"
                break
        if not linked_bet:
            # Fallback: use betankande's party field (organ proxy) or category+time
            # Betankande have no party field explicitly; use category + time window
            candidates = bet.copy()
            if cat:
                cat_candidates = bet[bet["category"].astype(str) == cat].copy()
                if not cat_candidates.empty:
                    candidates = cat_candidates
            candidates = candidates[
                (candidates["action_date_parsed"] >= sdate - window)
                & (candidates["action_date_parsed"] <= sdate + window)
            ]
            if not candidates.empty:
                candidates["days_diff"] = (candidates["action_date_parsed"] - sdate).abs().dt.days
                best = candidates.nsmallest(1, "days_diff").iloc[0]
                linked_bet = dict(best)
                bet_source = "bet_fallback_category_time"

        if linked_bet:
            days_diff = None
            try:
                mdate = linked_bet.get("action_date_parsed", linked_bet.get("action_date"))
                if pd.notna(sdate) and pd.notna(mdate):
                    days_diff = int(abs((pd.Timestamp(mdate) - sdate).days))
            except Exception:
                days_diff = None
            rows.append({
                "speech_id": sid,
                "action_id": str(linked_bet.get("action_id", "")),
                "action_type": "betankande",
                "speech_party": party,
                "category": cat,
                "speech_date": sdate,
                "action_date": linked_bet.get("action_date_parsed", linked_bet.get("action_date")),
                "action_party": str(linked_bet.get("party", "")),  # organ proxy
                "action_weight": float(linked_bet.get("action_weight", 0.0)),
                "days_diff": days_diff,
                "link_source": bet_source or "bet_fallback",
            })

    if not rows:
        print("No proposition/betankande links found. Writing empty output.")
        outdf = pd.DataFrame(columns=[
            "speech_id", "action_id", "action_type", "speech_party", "category",
            "speech_date", "action_date", "action_party", "action_weight",
            "days_diff", "link_source",
        ])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        outdf.to_parquet(out_path, index=False, compression="zstd")
        return

    outdf = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outdf.to_parquet(out_path, index=False, compression="zstd")

    summary = {
        "output": str(out_path),
        "rows": int(len(outdf)),
        "action_types": sorted(outdf["action_type"].unique().tolist()),
        "link_source_counts": outdf["link_source"].value_counts().to_dict(),
        "n_proposition_links": int((outdf["action_type"] == "proposition").sum()),
        "n_betankande_links": int((outdf["action_type"] == "betankande").sum()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()