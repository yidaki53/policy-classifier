#!/usr/bin/env python3
"""Bootstrap uncertainty intervals for say-vs-do distance metrics by party.

Uses existing analysis artifacts and does not require rerunning classifier jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float) -> tuple[float, float, float]:
    if values.size == 0:
        return np.nan, np.nan, np.nan
    means = np.empty(n_boot, dtype=float)
    n = values.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(values[idx].mean())
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return float(values.mean()), lo, hi


def main() -> None:
    p = argparse.ArgumentParser(description="Bootstrap say-vs-do uncertainty intervals by party")
    p.add_argument("--axis-scores", default="output/analysis/speech_action_axis_scores.parquet")
    p.add_argument("--links", default="data/parquet/speech_action_links.parquet")
    p.add_argument("--out", default="output/analysis/say_do_uncertainty_intervals_party.parquet")
    p.add_argument("--summary-out", default="output/analysis/say_do_uncertainty_summary.json")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-party-n", type=int, default=200)
    args = p.parse_args()

    axis_path = Path(args.axis_scores)
    links_path = Path(args.links)
    if not axis_path.exists():
        raise FileNotFoundError(f"Missing axis scores: {axis_path}")
    if not links_path.exists():
        raise FileNotFoundError(f"Missing links parquet: {links_path}")

    axis = pd.read_parquet(axis_path, columns=["speech_id", "motion_id", "axis_js_distance", "axis_cosine_distance"])
    links = pd.read_parquet(links_path, columns=["speech_id", "motion_id", "speech_party"])
    merged = axis.merge(links, on=["speech_id", "motion_id"], how="left")
    merged["speech_party"] = merged["speech_party"].fillna("Unknown").astype(str)

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    for party, grp in merged.groupby("speech_party", dropna=False):
        if len(grp) < args.min_party_n:
            continue

        js_vals = grp["axis_js_distance"].to_numpy(dtype=float)
        cos_vals = grp["axis_cosine_distance"].to_numpy(dtype=float)

        js_mean, js_lo, js_hi = _bootstrap_mean_ci(js_vals, args.n_boot, rng, args.alpha)
        cos_mean, cos_lo, cos_hi = _bootstrap_mean_ci(cos_vals, args.n_boot, rng, args.alpha)

        rows.append(
            {
                "speech_party": str(party),
                "n": int(len(grp)),
                "axis_js_mean": js_mean,
                "axis_js_ci_low": js_lo,
                "axis_js_ci_high": js_hi,
                "axis_cosine_mean": cos_mean,
                "axis_cosine_ci_low": cos_lo,
                "axis_cosine_ci_high": cos_hi,
                "alpha": float(args.alpha),
                "n_boot": int(args.n_boot),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("n", ascending=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False, compression="zstd")

    summary = {
        "axis_scores": str(axis_path),
        "links": str(links_path),
        "rows_input": int(len(merged)),
        "parties_output": int(len(out_df)),
        "n_boot": int(args.n_boot),
        "alpha": float(args.alpha),
        "min_party_n": int(args.min_party_n),
        "output": str(out_path),
    }

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
