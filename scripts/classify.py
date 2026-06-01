#!/usr/bin/env python3
"""Classify normalized motions and persist results to Parquet.

This script is a Parquet-first replacement for the legacy DB-backed `classify.py`.
It reads `data/parquet/raw_motions.parquet`, builds or updates
`data/parquet/normalized_motions.parquet` and appends classification rows to
`data/parquet/classifications.parquet`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from swedish_parliament_policy_classifier.exports import load_definitions, classify_motion


def _normalize_raw_to_parquet(raw_parquet: str | Path, normalized_out: str | Path) -> int:
    raw_p = Path(raw_parquet)
    out_p = Path(normalized_out)
    if not raw_p.exists():
        print("No raw motions parquet found at", raw_p)
        return 0

    raw = pd.read_parquet(raw_p)
    rows = []
    for _, r in raw.iterrows():
        mid = str(r.get("id")) if r.get("id") is not None else None
        if not mid:
            continue
        raw_json = r.get("json")
        try:
            data = json.loads(raw_json) if isinstance(raw_json, str) else (raw_json or {})
        except Exception:
            data = raw_json or {}
        title = data.get("title") or data.get("rubrik") or ""
        text = data.get("text") or data.get("body") or title or ""
        date = data.get("date") or data.get("datum") or None
        party = data.get("party") or data.get("parti") or None
        doc_type = data.get("doc_type") or data.get("dokumenttyp") or None
        metadata = {k: v for k, v in data.items() if k not in ("title", "text", "date", "party")}
        rows.append({"id": mid, "title": title, "text": text, "date": date, "party": party, "doc_type": doc_type, "metadata": json.dumps(metadata, ensure_ascii=False)})

    new_df = pd.DataFrame(rows)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        try:
            prev = pd.read_parquet(out_p)
            # keep previous normalized rows (do not overwrite existing normalized motions)
            combined = pd.concat([prev, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["id"], keep="first")
        except Exception:
            combined = new_df
    else:
        combined = new_df

    combined.to_parquet(out_p, index=False, compression="zstd")
    return len(combined)


def classify_parquet(normalized_parquet: str | Path = "data/parquet/normalized_motions.parquet", classifications_out: str | Path = "data/parquet/classifications.parquet", limit: Optional[int] = None) -> int:
    defs = load_definitions()
    nm_p = Path(normalized_parquet)
    out_p = Path(classifications_out)
    if not nm_p.exists():
        print("No normalized motions found; run ingest first to create normalized_motions.parquet")
        return 0

    nm = pd.read_parquet(nm_p)
    # ensure id column exists
    if "id" not in nm.columns:
        print("normalized_motions.parquet missing 'id' column")
        return 0

    # Determine already classified motions
    classified_ids = set()
    if out_p.exists():
        try:
            prev = pd.read_parquet(out_p, columns=["motion_id"]) if out_p.exists() else pd.DataFrame()
            classified_ids = set(prev["motion_id"].astype(str).unique()) if not prev.empty else set()
        except Exception:
            classified_ids = set()

    to_classify = nm[~nm["id"].astype(str).isin(classified_ids)].copy()
    if limit:
        to_classify = to_classify.head(limit)

    rows = []
    for _, r in to_classify.iterrows():
        mid = str(r.get("id"))
        text = (r.get("title") or "") + "\n" + (r.get("text") or "")
        try:
            results = classify_motion(motion_id=mid, text=text, categories=defs)
        except Exception as e:
            print(f"Failed to classify {mid}: {e}")
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
            # keep latest per (motion_id, category)
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
    parser = argparse.ArgumentParser(description="Classify motions and persist to Parquet")
    parser.add_argument("--raw", default="data/parquet/raw_motions.parquet", help="Raw motions parquet (input)")
    parser.add_argument("--normalized-out", default="data/parquet/normalized_motions.parquet", help="Normalized motions parquet (output)")
    parser.add_argument("--classifications-out", default="data/parquet/classifications.parquet", help="Classifications parquet (output)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    written_nm = _normalize_raw_to_parquet(args.raw, args.normalized_out)
    print(f"Normalized motions (rows) now: {written_nm}")
    classified = classify_parquet(args.normalized_out, args.classifications_out, limit=args.limit)
    print(f"Appended {classified} classification rows to {args.classifications_out}")


if __name__ == "__main__":
    main()

if False:
    # Graphify hint: classify() calls load_definitions() and score_motion()
    # Anchor the verified loader implementation directly (avoid classifier->scorer indirection)
    from definitions.loader import load_verified_definitions as _hint_load_verified_definitions
    from swedish_parliament_policy_classifier.models import CategoryDef as _hint_CategoryDef
