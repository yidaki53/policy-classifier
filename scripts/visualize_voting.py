#!/usr/bin/env python3
"""Visualise voting-based ideology and cohesion analysis.

Reads votering Parquet files directly (the canonical vote store) and
produces academic figures for the "motions + votes" paper section.

Figures:
1. Party vote cohesion over time (fraction voting together per party/yr)
2. Cross-party agreement matrix (how often two parties vote the same way)
3. Vote distribution (ja/nej/avstår/frånvarande) by party
4. Motions filed vs motions voted on (timeline)
5. Committee issue areas voted on (distribution of beteckning prefixes)

Usage:
    uv run python scripts/visualize_voting.py --votering-parquet data/votering/parquet --out figures/voting
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime

from swedish_parliament_policy_classifier.visualization.style_config import (
    add_figure_credits,
)
from swedish_parliament_policy_classifier.provenance import write_run_provenance

PARTY_COLORS = {
    "S":  "#CC3333",
    "M":  "#3366AA",
    "C":  "#44AA44",
    "L":  "#FFAA00",
    "KD": "#4444AA",
    "V":  "#AA3333",
    "MP": "#33AA33",
    "SD": "#DDCC00",
}

VALID_PARTIES = list(PARTY_COLORS.keys())


def _load_all_votering(parquet_dir: str) -> pd.DataFrame:
    """Load all votering Parquet files into a single DataFrame."""
    path = Path(parquet_dir)
    files = sorted(path.glob("*.parquet"))
    print(f"Loading {len(files)} votering Parquet files ...", file=sys.stderr)
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["year"] = df["rm"].astype(str).str[:4].astype(int, errors="ignore")
    return df


def _party_cohesion(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["parti"].isin(VALID_PARTIES)].copy()
    df["rost_norm"] = df["rost"].astype(str).str.strip().str.lower()

    results = []
    for (year, party), group in df.groupby(["year", "parti"]):
        cohesion_scores = []
        for _, vgroup in group.groupby("votering_id"):
            counts = vgroup["rost_norm"].value_counts()
            total = counts.sum()
            if total == 0:
                continue
            max_agree = counts.max()
            cohesion = max_agree / total
            cohesion_scores.append(cohesion)
        if cohesion_scores:
            results.append({
                "year": int(year) if pd.notna(year) else None,
                "party": party,
                "cohesion_mean": np.mean(cohesion_scores),
                "cohesion_median": np.median(cohesion_scores),
                "n_votes": len(cohesion_scores),
            })
    return pd.DataFrame(results)


def plot_cohesion(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    coh = _party_cohesion(df)
    if coh.empty:
        print("WARNING: no cohesion data to plot", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(11, 6.4))
    year_min = int(coh["year"].min())
    year_max = int(coh["year"].max())
    for party in VALID_PARTIES:
        sub = coh[coh["party"] == party]
        if sub.empty:
            continue
        ax.plot(
            sub["year"],
            sub["cohesion_mean"],
            marker="o",
            markersize=4,
            label=party,
            color=PARTY_COLORS.get(party, "#333333"),
            linewidth=1.9,
            alpha=0.95,
        )

    y_min = float(pd.to_numeric(coh["cohesion_mean"], errors="coerce").min())
    if y_min > 0.55:
        lower = max(0.0, y_min - 0.06)
        ylabel = "Mean vote cohesion (share, 0-1; axis truncated for readability)"
    else:
        lower = 0.0
        ylabel = "Mean vote cohesion (share, 0-1)"

    ax.set_xlabel("Year (calendar year)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Party Vote Cohesion Over Time ({year_min}-{year_max})")
    ax.legend(title="Party", loc="lower left", ncol=4)
    ax.set_ylim(lower, 1.02)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="both", labelsize=10)
    add_figure_credits(fig, n_total=len(coh), date_range=f"{coh['year'].min()}-{coh['year'].max()}")
    out_path = out_dir / "party_cohesion_timeseries.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}", file=sys.stderr)


def _cross_party_agreement(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["parti"].isin(VALID_PARTIES)].copy()
    df["rost_norm"] = df["rost"].astype(str).str.strip().str.lower()
    df = df[df["rost_norm"].isin(["ja", "nej"])]

    party_pos = (
        df.groupby(["votering_id", "parti"])["rost_norm"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index(name="position")
    )

    wide = party_pos.pivot(index="votering_id", columns="parti", values="position")
    wide = wide.reindex(columns=VALID_PARTIES)

    matrix = pd.DataFrame(index=VALID_PARTIES, columns=VALID_PARTIES, dtype=float)
    for p1 in VALID_PARTIES:
        for p2 in VALID_PARTIES:
            if p1 == p2:
                matrix.loc[p1, p2] = 1.0
                continue
            both = wide[[p1, p2]].dropna()
            if len(both) == 0:
                matrix.loc[p1, p2] = np.nan
                continue
            agree = (both[p1] == both[p2]).sum()
            matrix.loc[p1, p2] = agree / len(both)

    return matrix.astype(float)


def plot_agreement_matrix(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    mat = _cross_party_agreement(df)
    if mat.empty:
        print("WARNING: no agreement data to plot", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    year_min = int(pd.to_numeric(df["year"], errors="coerce").dropna().min())
    year_max = int(pd.to_numeric(df["year"], errors="coerce").dropna().max())
    sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Agreement rate"},
        ax=ax,
    )
    ax.set_title(f"Cross-Party Voting Agreement (ja/nej only, {year_min}-{year_max})")
    add_figure_credits(fig, n_total=len(df), date_range=f"{df['year'].min()}-{df['year'].max()}")
    out_path = out_dir / "cross_party_agreement.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}", file=sys.stderr)


def plot_vote_distribution(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df[df["parti"].isin(VALID_PARTIES)].copy()
    df["rost_norm"] = df["rost"].astype(str).str.strip().str.lower()

    counts = df.groupby(["parti", "rost_norm"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["ja", "nej", "avstår", "frånvarande"], fill_value=0)
    counts_pct = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    year_min = int(pd.to_numeric(df["year"], errors="coerce").dropna().min())
    year_max = int(pd.to_numeric(df["year"], errors="coerce").dropna().max())
    counts_pct.plot(kind="bar", stacked=True, ax=ax,
                    color=["#2ca02c", "#d62728", "#ff7f0e", "#9467bd"])
    ax.set_xlabel("Party")
    ax.set_ylabel("Vote share (%)")
    ax.set_title(f"Vote Distribution by Party ({year_min}-{year_max})")
    ax.legend(title="Vote", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    add_figure_credits(fig, n_total=len(df), date_range=f"{df['year'].min()}-{df['year'].max()}")
    out_path = out_dir / "vote_distribution_by_party.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}", file=sys.stderr)


def plot_motions_vs_votes_timeline(df: pd.DataFrame, out_dir: Path,
                                   normalized_motions_path: str = "data/parquet/normalized_motions.parquet",
                                   motion_votes_path: str = "data/parquet/motion_votes.parquet"):
    """Figure 4: Motions filed vs motions voted on per year (Parquet-only)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        nm = pd.read_parquet(normalized_motions_path, columns=["date", "doc_type"])
        nm = nm[nm["doc_type"] == "mot"].dropna(subset=["date"])
        motion_years = nm["date"].dropna().apply(lambda d: int(str(d)[:4])).tolist()
        motion_counts = pd.Series(motion_years).value_counts().sort_index().reset_index()
        motion_counts.columns = ["year", "motions"]
    except Exception as e:
        print(f"WARNING: could not load motion counts: {e}", file=sys.stderr)
        motion_counts = pd.DataFrame(columns=["year", "motions"])

    try:
        mv = pd.read_parquet(motion_votes_path, columns=["rm"])
        mv = mv.dropna(subset=["rm"])
        matched_years = mv["rm"].apply(lambda r: int(str(r)[:4])).tolist()
        matched_counts = pd.Series(matched_years).value_counts().sort_index().reset_index()
        matched_counts.columns = ["year", "matched"]
    except Exception as e:
        print(f"WARNING: could not load matched motion counts: {e}", file=sys.stderr)
        matched_counts = pd.DataFrame(columns=["year", "matched"])

    vote_counts = df.groupby("year")["votering_id"].nunique().reset_index(name="votes")

    merged = motion_counts.merge(matched_counts, on="year", how="outer")
    merged = merged.merge(vote_counts, on="year", how="outer").sort_values("year")
    merged = merged.fillna(0).astype({"year": int, "motions": int, "matched": int, "votes": int})

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.25
    ax.bar(merged["year"] - width, merged["motions"], width=width, label="Motions filed", color="#3366AA", alpha=0.8)
    ax.bar(merged["year"], merged["matched"], width=width, label="Motions reaching vote", color="#CC3333", alpha=0.8)
    ax.bar(merged["year"] + width, merged["votes"], width=width, label="Roll-call votes", color="#44AA44", alpha=0.8)
    ax.set_xlabel("Year (calendar year)")
    ax.set_ylabel("Count (n)")
    ax.set_title(f"Motions Filed, Reaching Vote, and Roll-Call Votes ({merged['year'].min()}-{merged['year'].max()})")
    ax.legend()
    add_figure_credits(fig, n_total=int(merged["motions"].sum()), date_range=f"{merged['year'].min()}-{merged['year'].max()}")
    out_path = out_dir / "motions_vs_votes_timeline.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}", file=sys.stderr)


def plot_committee_distribution(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["committee"] = df["beteckning"].astype(str).str.extract(r"^([A-Za-z]+)", expand=False).str.upper()
    committee_counts = df.groupby("committee")["votering_id"].nunique().sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    year_min = int(pd.to_numeric(df["year"], errors="coerce").dropna().min())
    year_max = int(pd.to_numeric(df["year"], errors="coerce").dropna().max())
    committee_counts.plot(kind="barh", ax=ax, color="#3366AA")
    ax.set_xlabel("Number of roll-call votes (n)")
    ax.set_ylabel("Committee")
    ax.set_title(f"Roll-Call Votes by Committee ({year_min}-{year_max})")
    add_figure_credits(fig, n_total=int(committee_counts.sum()), date_range=f"{df['year'].min()}-{df['year'].max()}")
    out_path = out_dir / "committee_vote_distribution.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}", file=sys.stderr)


def generate_all(votering_parquet_dir: str, out_dir: str,
                 normalized_motions_path: str = "data/parquet/normalized_motions.parquet",
                 motion_votes_path: str = "data/parquet/motion_votes.parquet"):
    df = _load_all_votering(votering_parquet_dir)
    print(f"Loaded {len(df):,} vote records", file=sys.stderr)
    out = Path(out_dir)

    plot_cohesion(df, out)
    plot_agreement_matrix(df, out)
    plot_vote_distribution(df, out)
    plot_motions_vs_votes_timeline(df, out,
                                   normalized_motions_path=normalized_motions_path,
                                   motion_votes_path=motion_votes_path)
    plot_committee_distribution(df, out)

    provenance_path = write_run_provenance(
        script="scripts/visualize_voting.py",
        inputs={
            "votering_parquet": votering_parquet_dir,
            "normalized_motions": normalized_motions_path,
            "motion_votes": motion_votes_path,
        },
        outputs=[
            str(out / "party_cohesion_timeseries.pdf"),
            str(out / "party_cohesion_timeseries.png"),
            str(out / "cross_party_agreement.pdf"),
            str(out / "cross_party_agreement.png"),
            str(out / "vote_distribution_by_party.pdf"),
            str(out / "vote_distribution_by_party.png"),
            str(out / "motions_vs_votes_timeline.pdf"),
            str(out / "motions_vs_votes_timeline.png"),
            str(out / "committee_vote_distribution.pdf"),
            str(out / "committee_vote_distribution.png"),
        ],
        output_dir=out,
        metadata={
            "n_vote_rows": int(len(df)),
            "year_min": int(pd.to_numeric(df["year"], errors="coerce").dropna().min()),
            "year_max": int(pd.to_numeric(df["year"], errors="coerce").dropna().max()),
        },
    )
    print(f"Saved provenance: {provenance_path}", file=sys.stderr)

    print(f"\nAll figures written to {out}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Generate voting-based analysis figures (Parquet-only)")
    parser.add_argument("--votering-parquet", default="data/votering/parquet")
    parser.add_argument("--out", default="figures/voting")
    parser.add_argument("--normalized-motions", default="data/parquet/normalized_motions.parquet")
    parser.add_argument("--motion-votes", default="data/parquet/motion_votes.parquet")
    args = parser.parse_args()

    generate_all(args.votering_parquet, args.out,
                 normalized_motions_path=args.normalized_motions,
                 motion_votes_path=args.motion_votes)


if __name__ == "__main__":
    main()