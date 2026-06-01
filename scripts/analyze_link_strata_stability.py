#!/usr/bin/env python3
"""Quantify robustness of say-vs-do distances across link-confidence strata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _mean_by_party(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        df.groupby("speech_party", as_index=False)
        .agg(n=(value_col, "size"), mean_value=(value_col, "mean"), std_value=(value_col, "std"))
        .rename(columns={"speech_party": "party"})
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze stability of axis distances across link-confidence strata")
    p.add_argument("--axis-scores", default="output/analysis/speech_action_axis_scores.parquet")
    p.add_argument("--link-strata", default="output/analysis/speech_action_link_confidence_strata.parquet")
    p.add_argument("--out", default="output/analysis/link_strata_stability_party.parquet")
    p.add_argument("--summary-out", default="output/analysis/link_strata_stability_summary.json")
    p.add_argument("--metric", default="axis_js_distance", choices=["axis_js_distance", "axis_cosine_distance"])
    p.add_argument("--min-n", type=int, default=100)
    args = p.parse_args()

    axis_path = Path(args.axis_scores)
    strata_path = Path(args.link_strata)
    if not axis_path.exists():
        raise FileNotFoundError(f"Missing axis scores: {axis_path}")
    if not strata_path.exists():
        raise FileNotFoundError(f"Missing link strata: {strata_path}")

    axis = pd.read_parquet(axis_path, columns=["speech_id", "motion_id", args.metric]).copy()
    strata = pd.read_parquet(
        strata_path,
        columns=["speech_id", "motion_id", "speech_party", "link_confidence_stratum", "link_confidence_score"],
    ).copy()

    df = axis.merge(strata, on=["speech_id", "motion_id"], how="inner")
    df = df[df[args.metric].notna()].copy()
    df["speech_party"] = df["speech_party"].fillna("Unknown").astype(str)

    full = _mean_by_party(df, args.metric).rename(
        columns={"n": "n_all", "mean_value": f"{args.metric}_all", "std_value": f"{args.metric}_all_std"}
    )

    rows = []
    for stratum, grp in df.groupby("link_confidence_stratum", dropna=False):
        g = _mean_by_party(grp, args.metric)
        g = g.rename(
            columns={
                "n": f"n_{stratum}",
                "mean_value": f"{args.metric}_{stratum}",
                "std_value": f"{args.metric}_{stratum}_std",
            }
        )
        rows.append(g)

    out = full
    for g in rows:
        out = out.merge(g, on="party", how="left")

    # Primary stability contrast: structural_high vs all
    structural_col = f"{args.metric}_structural_high"
    n_struct_col = "n_structural_high"
    if structural_col in out.columns:
        out[f"delta_{args.metric}_structural_minus_all"] = out[structural_col] - out[f"{args.metric}_all"]
        out["structural_available"] = out[n_struct_col].fillna(0).astype(int) >= int(args.min_n)
    else:
        out[f"delta_{args.metric}_structural_minus_all"] = np.nan
        out["structural_available"] = False

    out = out.sort_values("n_all", ascending=False).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, compression="zstd")

    summary = {
        "input_axis_scores": str(axis_path),
        "input_link_strata": str(strata_path),
        "metric": args.metric,
        "rows_input": int(len(df)),
        "parties_output": int(len(out)),
        "min_n": int(args.min_n),
        "output": str(out_path),
    }

    if structural_col in out.columns:
        avail = out[out["structural_available"]].copy()
        summary["structural_available_parties"] = int(len(avail))
        if not avail.empty:
            deltas = avail[f"delta_{args.metric}_structural_minus_all"].dropna().to_numpy(dtype=float)
            if deltas.size > 0:
                summary["delta_structural_minus_all_mean"] = float(np.mean(deltas))
                summary["delta_structural_minus_all_abs_mean"] = float(np.mean(np.abs(deltas)))
                summary["delta_structural_minus_all_abs_max"] = float(np.max(np.abs(deltas)))

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
