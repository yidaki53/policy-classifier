#!/usr/bin/env python3
"""Generate per-party trend visualizations (last ~15 years).

Produces:
  1. party_ideology_trends.png — Line chart of each party's mean ideological
     placement over time (year on x-axis, net left-right [-1,1] on y-axis).
  2. party_fulfillment_trends.png — Line chart of each party's fulfillment
     rate over time.

Usage:
    uv run python scripts/generate_party_trends.py \\
        --party-scores output/analysis/recency_weighted_party_scores.parquet \\
        --fulfillment output/analysis/promise_fulfillment_party_topic_year.parquet \\
        --out-dir figures/manuscript
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swedish_parliament_policy_classifier.visualization.style_config import (
    CURRENT_PARTIES,
    PARTY_LABELS,
    PARTY_COLORS_PLOT,
    CATEGORY_ORDER,
    add_figure_credits,
    compute_ideology_score_from_proportions,
)


def _ideology_score_from_proportions(proportions: dict) -> float:
    """Compute net left-right score [-1, 1] from category proportion dict."""
    return compute_ideology_score_from_proportions(proportions)


def plot_party_ideology_trends(
    party_scores_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot each party's ideological placement over the last ~15 years.

    Reads from recency_weighted_party_scores.parquet which contains
    party-year level ideology scores, or computes from category proportions.
    """
    # Check what columns are available
    cols = party_scores_df.columns.tolist()

    # Determine year column
    year_col = None
    for c in ["year", "år", "Year", "time_key"]:
        if c in cols:
            year_col = c
            break
    if year_col is None:
        raise ValueError(
            f"No year column found in party scores. Columns: {cols}\n"
            "Expected a party-year level parquet with at least 'party', 'year', and a score column."
        )

    # Ensure party column exists
    party_col = "party"
    if party_col not in cols:
        raise ValueError(f"Expected 'party' column. Found: {cols}")

    # Determine score column
    score_col = None
    for c in ["ideology_score", "latent_ideology_score", "speech_axis_mean", "ideology_uphold_v2"]:
        if c in cols:
            score_col = c
            break

    # If no direct score column, try computing from category proportions
    if score_col is None:
        cat_cols = [c for c in cols if c in CATEGORY_ORDER]
        if len(cat_cols) >= 3:
            # Compute ideology score from proportions
            rows = []
            for _, row in party_scores_df.iterrows():
                props = {c: float(row.get(c, 0.0)) for c in cat_cols}
                rows.append({
                    "party": row[party_col],
                    year_col: row[year_col],
                    "ideology_score": _ideology_score_from_proportions(props),
                })
            df = pd.DataFrame(rows)
        else:
            raise ValueError(
                f"No ideology score column and no category proportion columns found. Columns: {cols}"
            )
    else:
        df = party_scores_df[[party_col, year_col, score_col]].copy()
        df = df.rename(columns={score_col: "ideology_score"})

    # Cast year to numeric
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df.dropna(subset=[year_col, "ideology_score"])
    df[year_col] = df[year_col].astype(int)

    # Filter to current parties only
    df = df[df["party"].isin(CURRENT_PARTIES)].copy()

    # Filter to last ~15 years
    max_year = df[year_col].max()
    min_year = max_year - 15
    df = df[df[year_col] >= min_year].copy()

    if df.empty:
        print("WARNING: No party ideology trend data available after filtering.", file=sys.stderr)
        return

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for party in sorted(CURRENT_PARTIES):
        sub = df[df["party"] == party].sort_values(year_col)
        if sub.empty:
            continue
        color = PARTY_COLORS_PLOT.get(party, "#888888")
        label = PARTY_LABELS.get(party, party)
        ax.plot(
            sub[year_col], sub["ideology_score"],
            color=color, linewidth=2.5, marker="o", markersize=5,
            label=label,
        )

    ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Ideological placement (Left → Right)", fontsize=12)
    ax.set_title("Party Ideology Trends Over Time", fontsize=13)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    add_figure_credits(fig, n_parties=len(CURRENT_PARTIES), extra_text="Current 8 Riksdag parties")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_party_fulfillment_trends(
    fulfillment_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot each party's fulfillment rate over the last ~15 years."""
    cols = fulfillment_df.columns.tolist()

    # Determine year column
    year_col = None
    for c in ["year", "år", "Year", "time_key"]:
        if c in cols:
            year_col = c
            break
    if year_col is None:
        print(
            f"WARNING: No year column in fulfillment data. Columns: {cols}. Fulfillment trend skipped.",
            file=sys.stderr,
        )
        return

    # Ensure party column
    if "party" not in cols:
        print(
            f"WARNING: No 'party' column in fulfillment data. Columns: {cols}. Fulfillment trend skipped.",
            file=sys.stderr,
        )
        return

    # Determine fulfillment metric column
    fulfill_col = None
    for c in ["pct_speech_motion_vote", "fulfillment_rate", "fulfillment_score",
               "pct_speech_motion_vote_mean", "motion_pathway_fulfillment"]:
        if c in cols:
            fulfill_col = c
            break

    if fulfill_col is None:
        print(
            f"WARNING: No fulfillment metric column found. Columns: {cols}. Fulfillment trend skipped.",
            file=sys.stderr,
        )
        return

    df = fulfillment_df[["party", year_col, fulfill_col]].copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[fulfill_col] = pd.to_numeric(df[fulfill_col], errors="coerce")
    df = df.dropna(subset=[year_col, fulfill_col])
    df[year_col] = df[year_col].astype(int)

    # Filter to current parties only
    df = df[df["party"].isin(CURRENT_PARTIES)].copy()

    # Aggregate to party-year level (mean across topics if needed)
    df = df.groupby(["party", year_col], as_index=False)[fulfill_col].mean()

    # Filter to last ~15 years
    max_year = df[year_col].max()
    min_year = max_year - 15
    df = df[df[year_col] >= min_year].copy()

    if df.empty or len(df) < 2:
        print(
            "WARNING: Not enough fulfillment trend data after filtering. Skipping fulfillment trend.",
            file=sys.stderr,
        )
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for party in sorted(CURRENT_PARTIES):
        sub = df[df["party"] == party].sort_values(year_col)
        if sub.empty:
            continue
        color = PARTY_COLORS_PLOT.get(party, "#888888")
        label = PARTY_LABELS.get(party, party)
        ax.plot(
            sub[year_col], sub[fulfill_col],
            color=color, linewidth=2.5, marker="s", markersize=5,
            label=label,
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Fulfillment Rate", fontsize=12)
    ax.set_title("Party Fulfillment Trends Over Time", fontsize=13)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    add_figure_credits(fig, n_parties=len(CURRENT_PARTIES), extra_text="Speech→action pathway continuation rate")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-party ideology and fulfillment trend visualizations."
    )
    parser.add_argument(
        "--party-scores",
        default="output/analysis/recency_weighted_party_scores.parquet",
        help="Path to recency_weighted_party_scores.parquet (party-year ideology data).",
    )
    parser.add_argument(
        "--fulfillment",
        default="output/analysis/promise_fulfillment_party_topic_year.parquet",
        help="Path to promise_fulfillment_party_topic_year.parquet (fulfillment data).",
    )
    parser.add_argument(
        "--out-dir",
        default="figures/manuscript",
        help="Output directory for trend figures.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Ideology trends ---
    try:
        ideology_df = pd.read_parquet(args.party_scores)
        plot_party_ideology_trends(ideology_df, out_dir / "party_ideology_trends.png")
    except Exception as e:
        print(f"ERROR generating ideology trends: {e}", file=sys.stderr)

    # --- Fulfillment trends ---
    try:
        fulfillment_df = pd.read_parquet(args.fulfillment)
        plot_party_fulfillment_trends(fulfillment_df, out_dir / "party_fulfillment_trends.png")
    except Exception as e:
        print(f"ERROR generating fulfillment trends: {e}", file=sys.stderr)

    print(f"\nAll trend figures saved to {out_dir}")


if __name__ == "__main__":
    main()