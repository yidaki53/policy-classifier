#!/usr/bin/env python3
"""Pareto frontier visualization: speech-action consistency vs vote fidelity.

Shows the trade-off surface between two key accountability metrics:
- X-axis: Consistency score (how aligned speech, motion, and vote signals are)
- Y-axis: Vote fidelity (share of speech-linked pathways that reach vote action)

Party-year points are sized by recency weight (more recent = larger) and colored
by parliamentary era. The Pareto frontier connects points that are not dominated
on either axis, showing the best achievable combinations.

Usage:
    uv run python scripts/plot_pareto_frontier.py \
        --analysis-dir output/analysis \
        --figures-dir output/manuscript/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swedish_parliament_policy_classifier.visualization.style_config import add_figure_credits


# Parliamentary eras for coloring
ERA_BOUNDARIES = {
    "2014-2017": (2014, 2017),
    "2018-2021": (2018, 2021),
    "2022-2026": (2022, 2026),
}

ERA_COLORS = {
    "2014-2017": "#4393c3",  # Blue
    "2018-2021": "#f4a582",  # Orange
    "2022-2026": "#d6604d",  # Red
}


def _assign_era(year: int) -> str:
    for era, (start, end) in ERA_BOUNDARIES.items():
        if start <= year <= end:
            return era
    if year < 2014:
        return "2014-2017"
    return "2022-2026"


def _compute_recency_weight(year: int, reference_year: int, lam: float = 0.15) -> float:
    """Exponential recency weight: more recent years get higher weight."""
    delta = max(reference_year - year, 0)
    return np.exp(-lam * delta)


def compute_pareto_frontier(points: pd.DataFrame) -> pd.DataFrame:
    """Compute Pareto frontier: points not dominated on either axis.
    
    A point is dominated if another point has both higher consistency AND higher fidelity.
    """
    if points.empty:
        return points
    
    # Aggregate to party level for frontier computation
    party_agg = points.groupby("party").agg({
        "consistency_score": "mean",
        "pct_speech_motion_vote": "mean",
    }).reset_index()
    
    frontier_mask = []
    for idx, row in party_agg.iterrows():
        dominated = False
        for _, other in party_agg.iterrows():
            if idx == other.name:
                continue
            # Check if 'other' dominates 'row'
            if (other["consistency_score"] >= row["consistency_score"] and 
                other["pct_speech_motion_vote"] >= row["pct_speech_motion_vote"] and
                (other["consistency_score"] > row["consistency_score"] or 
                 other["pct_speech_motion_vote"] > row["pct_speech_motion_vote"])):
                dominated = True
                break
        frontier_mask.append(not dominated)
    
    return party_agg[frontier_mask].sort_values(["consistency_score", "pct_speech_motion_vote"])


def plot_pareto_frontier(
    analysis_dir: Path,
    figures_dir: Path,
    min_year: int = 2014,
) -> Path:
    """Generate Pareto frontier visualization.
    
    Args:
        analysis_dir: Directory with analysis parquet files
        figures_dir: Directory to save output figures
        min_year: Minimum year to include (speech data starts ~2014)
    
    Returns:
        Path to generated figure
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load party-year data
    party_year_path = analysis_dir / "consistency_fulfillment_party_year.parquet"
    if not party_year_path.exists():
        raise FileNotFoundError(f"Required file not found: {party_year_path}")
    
    df = pd.read_parquet(party_year_path)
    
    # Filter to relevant years
    df = df[df["year"] >= min_year].copy()
    
    # Filter out zero-fulfillment years (no speech-action linkage)
    df = df[df["pct_speech_motion_vote"] > 0].copy()
    
    if df.empty:
        raise ValueError("No data points after filtering")
    
    # Compute recency weights
    reference_year = df["year"].max()
    df["recency_weight"] = df["year"].apply(lambda y: _compute_recency_weight(y, reference_year))
    
    # Assign eras
    df["era"] = df["year"].apply(_assign_era)
    
    # Compute Pareto frontier at party level
    frontier = compute_pareto_frontier(df)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot points by era
    for era in sorted(ERA_COLORS.keys()):
        era_data = df[df["era"] == era]
        if era_data.empty:
            continue
        
        # Size points by recency weight (scale to reasonable marker sizes)
        sizes = era_data["recency_weight"] * 200 + 30  # Range: 30-230
        
        ax.scatter(
            era_data["consistency_score"],
            era_data["pct_speech_motion_vote"],
            s=sizes,
            c=ERA_COLORS[era],
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
            label=era,
            zorder=3,
        )
    
    # Plot Pareto frontier
    if not frontier.empty:
        # Sort by consistency for line drawing
        frontier_sorted = frontier.sort_values("consistency_score")
        
        # Draw frontier line
        ax.plot(
            frontier_sorted["consistency_score"],
            frontier_sorted["pct_speech_motion_vote"],
            color="black",
            linewidth=2,
            linestyle="--",
            alpha=0.7,
            label="Pareto frontier",
            zorder=4,
        )
        
        # Mark frontier points
        ax.scatter(
            frontier_sorted["consistency_score"],
            frontier_sorted["pct_speech_motion_vote"],
            s=150,
            c="gold",
            edgecolors="black",
            linewidths=2,
            marker="*",
            label="Frontier parties",
            zorder=5,
        )
        
        # Annotate frontier parties
        for _, row in frontier_sorted.iterrows():
            ax.annotate(
                row["party"],
                (row["consistency_score"], row["pct_speech_motion_vote"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )
    
    # Formatting
    ax.set_xlabel("Consistency Score\n(1 - JS divergence across modalities)", fontsize=11)
    ax.set_ylabel("Vote Fidelity\n(share of speech pathways reaching vote action)", fontsize=11)
    ax.set_title(
        "Pareto Frontier: Speech-Action Consistency vs Vote Fidelity\n"
        f"Party-year observations, {min_year}-{reference_year}",
        fontsize=12,
        fontweight="bold",
    )
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", framealpha=0.9)
    
    # Add interpretation guide
    ax.text(
        0.02, 0.98,
        "Higher is better on both axes\n"
        "Frontier = best achievable combinations",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    
    add_figure_credits(
        fig,
        n_total=len(df),
        n_parties=df["party"].nunique(),
        date_range=f"{min_year}-{reference_year}",
        source=str(party_year_path),
    )
    
    fig.tight_layout()
    
    out_path = figures_dir / "figure_pareto_frontier_consistency_fidelity.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=Path("output/analysis"))
    parser.add_argument("--figures-dir", type=Path, default=Path("output/manuscript/figures"))
    parser.add_argument("--min-year", type=int, default=2014)
    args = parser.parse_args()
    
    out_path = plot_pareto_frontier(args.analysis_dir, args.figures_dir, args.min_year)
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    main()
