#!/usr/bin/env python3
"""Apply rhetorical adjustments to an existing classification parquet.

Reads existing speech_classifications_with_rhetoric_full.parquet and
speech parquet files, calculates rhetorical adjustments per speech using
tuned weights, and writes a new parquet with adjusted probabilities.

Usage:
    uv run python scripts/apply_rhetorical_adjustments.py \
        --classifications data/parquet/speech_classifications_with_rhetoric_full.parquet \
        --speeches data/speeches/parquet \
        --out data/parquet/speech_classifications_rhetorical_adjusted.parquet
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from fractions import Fraction
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from swedish_parliament_policy_classifier.classifier.scorer import _detect_rhetorical_patterns


def apply_rhetorical_to_probs(
    probs: dict[str, float],
    rhet_adjustments: dict[str, float],
) -> dict[str, float]:
    """Apply multiplicative rhetorical boost and renormalize."""
    from fractions import Fraction

    combined = {k: Fraction(probs.get(k, 0.0)) for k in probs}
    if any(v != 0.0 for v in rhet_adjustments.values()):
        for k in combined:
            adj = rhet_adjustments.get(k, 0.0)
            if adj > 0:
                combined[k] *= Fraction(20, 10) + Fraction(adj).limit_denominator(100)
        total_adj = sum(combined.values())
        if total_adj > 0:
            for k in combined:
                combined[k] = combined[k] / total_adj
    return {k: float(v) for k, v in combined.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classifications", default="data/parquet/speech_classifications_with_rhetoric_full.parquet")
    p.add_argument("--speeches", default="data/speeches/parquet")
    p.add_argument("--out", default="data/parquet/speech_classifications_rhetorical_adjusted.parquet")
    p.add_argument("--batch", type=int, default=5000, help="Speeches per batch")
    args = p.parse_args()

    # Load existing classifications (all rows for all speeches)
    print(f"Reading classifications from {args.classifications}", file=sys.stderr)
    class_df = pd.read_parquet(args.classifications)
    print(f"  Loaded {len(class_df)} rows, {class_df['speech_id'].nunique()} unique speeches", file=sys.stderr)

    # Build a speech_id -> probs map from all_category_probs_json
    speech_probs = {}
    for sid, grp in class_df.groupby("speech_id"):
        if grp["all_category_probs_json"].notna().any():
            speech_probs[sid] = json.loads(grp["all_category_probs_json"].iloc[0])
        else:
            # Fallback: build from individual rows
            speech_probs[sid] = {
                row["category"]: float(row["normalized_weight"])
                for _, row in grp.iterrows()
            }

    # Load speech texts from parquet shards
    speech_files = sorted(Path(args.speeches).glob("*.parquet"))
    speech_texts = {}
    for f in speech_files:
        df = pd.read_parquet(f, columns=["anforande_id", "anforandetext"])
        for _, row in df.iterrows():
            sid = str(row["anforande_id"]) if row.get("anforande_id") is not None else None
            if sid and sid in speech_probs and sid not in speech_texts:
                speech_texts[sid] = row.get("anforandetext") or ""

    print(f"  Found texts for {len(speech_texts)} speeches", file=sys.stderr)

    # Process in batches and write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    speech_ids = sorted(speech_probs.keys())
    pbar = tqdm(total=len(speech_ids), desc="applying rhetorical adjustments", unit="speeches")

    for sid in speech_ids:
        text = speech_texts.get(sid, "")
        rhet_adjustments = _detect_rhetorical_patterns(text)
        original_probs = speech_probs[sid]
        adjusted_probs = apply_rhetorical_to_probs(original_probs, rhet_adjustments)

        # Build new rows with adjusted probabilities
        for cat, prob in adjusted_probs.items():
            all_rows.append({
                "speech_id": sid,
                "category": cat,
                "normalized_weight": prob,
                "category_probability": prob,
                "all_category_probs_json": json.dumps(adjusted_probs, ensure_ascii=False, sort_keys=True),
                "rhetorical_adjusted": any(v != 0.0 for v in rhet_adjustments.values()),
                "classifier_version": "0.8.0+speech+meta+rhetorical",
                "label_source": "rhetorical_adjustment",
            })

        pbar.update(1)

    pbar.close()

    if all_rows:
        batch_df = pd.DataFrame(all_rows)
        batch_df.to_parquet(out_path, index=False, compression="zstd")

    print(f"Wrote adjusted classifications to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
