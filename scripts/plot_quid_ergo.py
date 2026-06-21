#!/usr/bin/env python3
"""Quid Ergo visualization: what parties claim (speech) vs what they advance (vote action).

This directly addresses the manuscript's central "quid ergo" question: do parties
follow through on their rhetorical emphasis with institutional action? The visualization
shows a party-by-topic matrix where:
- Rows: parliamentary topics (ideological categories)
- Columns: parties
- Cell annotation: the gap between speech-side emphasis and action-side execution

For each party-topic, we compare:
- Speech-side: weighted ideological emphasis in plenary speeches
- Action-side: weighted ideology of motions/votes that continue from those speeches

A positive gap means rhetoric runs ahead of action. A negative gap means action
runs ahead of rhetoric (rare but possible).

Usage:
    uv run python scripts/plot_quid_ergo.py \
        --analysis-dir output/analysis \
        --figures-dir output/manuscript/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swedish_parliament_policy_classifier.visualization.style_config import (
    add_figure_credits,
    CATEGORY_ORDER,
    CATEGORY_LABELS,
)


# Map raw topic labels to ordered ideology categories
IDEOLOGY_TOPICS_ORDER = [
    "far_left",
    "left",
    "centre_left",
    "centre",
    "centre_right",
    "right",
    "far_right",
]


def _compute_speech_vs_action_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted ideology index for speech-side vs action-side evidence.

    Uses the topic weights (S=speech, M=motion, V=vote) from
    promise_fulfillment_party_topic_year to derive each party's ideological
    emphasis in speech as opposed to action channels.
    """
    rows = []
    topic_to_idx = {t: i for i, t in enumerate(IDEOLOGY_TOPICS_ORDER)}
    parties = sorted(df["party"].unique())

    for party in parties:
        party_data = df[df["party"] == party]
        # Speech-side: weighted by S (speech counts)
        speech_idx = (party_data["S"] * party_data["topic"].map(topic_to_idx)).sum()
        speech_total = party_data["S"].sum()

        # Action-side: weighted by M (motion) and V (vote) counts combined
        action_data = party_data.copy()
        action_data["action_count"] = action_data["M"].fillna(0) + action_data["V"].fillna(0)
        action_idx = (action_data["action_count"] * action_data["topic"].map(topic_to_idx)).sum()
        action_total = action_data["action_count"].sum()

        if speech_total > 0 and action_total > 0:
            speech_norm = speech_idx / speech_total
            action_norm = action_idx / action_total
            gap = action_norm - speech_norm  # Positive = action more right than speech

            rows.append({
                "party": party,
                "speech_idx": speech_norm,
                "action_idx": action_norm,
                "gap": gap,
                "speech_total": float(speech_total),
                "action_total": float(action_total),
            })
        else:
            rows.append({
                "party": party,
                "speech_idx": np.nan,
                "action_idx": np.nan,
                "gap": np.nan,
                "speech_total": float(speech_total),
                "action_total": float(action_total),
            })

    return pd.DataFrame(rows)


def _compute_topic_level_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute topic-level emphasis of speech vs action, per party.

    Returns a per (party, topic) gap indicating whether speech or action
    emphasizes that topic more strongly.
    """
    rows = []
    parties = sorted(df["party"].unique())
    topics = [t for t in IDEOLOGY_TOPICS_ORDER if t in df["topic"].unique()]

    for party in parties:
        for topic in topics:
            row = df[(df["party"] == party) & (df["topic"] == topic)]
            if row.empty:
                continue
            r = row.iloc[0]
            s = float(r.get("S", 0) or 0)
            action = float(r.get("M", 0) or 0) + float(r.get("V", 0) or 0)

            speech_topic_share = s  # raw count
            # Normalize within party for action share
            action_topic_share = action

            rows.append({
                "party": party,
                "topic": topic,
                "speech_count": s,
                "action_count": action,
            })

    return pd.DataFrame(rows)


def plot_quid_ergo(
    analysis_dir: Path,
    figures_dir: Path,
    min_year: int = 2014,
) -> Path:
    """Generate Quid Ergo visualization.

    Two-panel visualization:
    1. Party-level dot plot: speech ideology index vs action ideology index
    2. Topic-level bar chart: per-party speech vs action emphasis

    Args:
        analysis_dir: Directory with analysis parquet files
        figures_dir: Directory to save output figures
        min_year: Minimum year to include (speech data starts ~2014)

    Returns:
        Path to generated figure
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load topic-year data
    py_path = analysis_dir / "promise_fulfillment_party_topic_year.parquet"
    if not py_path.exists():
        raise FileNotFoundError(f"Required file not found: {py_path}")

    df = pd.read_parquet(py_path)

    # Filter to relevant years and aggregate across years
    df = df[df["year"] >= min_year].copy()
    df_agg = df.groupby(["party", "topic"], as_index=False).agg(
        S=("S", "sum"),
        M=("M", "sum"),
        V=("V", "sum"),
    )

    # Panel 1: Party-level speech vs action
    party_gap = _compute_speech_vs_action_gap(df_agg)

    # Panel 2: Topic emphasis per party (normalized shares)
    topic_data = df_agg.copy()
    party_totals_speech = topic_data.groupby("party")["S"].sum().rename("party_S")
    party_totals_action = topic_data.groupby("party").apply(
        lambda g: g["M"].fillna(0).sum() + g["V"].fillna(0).sum(), include_groups=False
    ).rename("party_action")
    topic_data = topic_data.merge(party_totals_speech, on="party")
    topic_data = topic_data.merge(party_totals_action, on="party")
    topic_data["speech_share"] = np.where(
        topic_data["party_S"] > 0,
        topic_data["S"] / topic_data["party_S"] * 100,
        0.0,
    )
    topic_data["action_share"] = np.where(
        topic_data["party_action"] > 0,
        (topic_data["M"].fillna(0) + topic_data["V"].fillna(0)) / topic_data["party_action"] * 100,
        0.0,
    )

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Party-level speech vs action ideology index
    ax = axes[0]
    pg = party_gap.dropna(subset=["speech_idx", "action_idx", "gap"]).copy()
    if not pg.empty:
        # Color: gap sign; positive gap shaded differently
        gap_colors = ["#d6604d" if g > 0 else "#4393c3" for g in pg["gap"]]

        ax.scatter(pg["speech_idx"], pg["action_idx"], c=gap_colors, s=180, alpha=0.8,
                   edgecolors="black", linewidths=1.0, zorder=3)
        # Diagonal reference line (speech == action)
        lim_min = 0
        lim_max = max(pg["speech_idx"].max(), pg["action_idx"].max()) * 1.05
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.4, label="Speech = Action")

        for _, r in pg.iterrows():
            ax.annotate(
                r["party"],
                (r["speech_idx"], r["action_idx"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_xlabel("Speech-side ideological index (left to right)", fontsize=10)
        ax.set_ylabel("Action-side ideological index (left to right)", fontsize=10)
        ax.set_title(
            "A. Quid Ergo: Speech ideology vs Action ideology by party\n"
            "(points above diagonal = action more right than speech)",
            fontsize=11,
        )
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.grid(alpha=0.3, linestyle=":")

        # Legend squares
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#4393c3", edgecolor="black", label="Speech more right than action"),
            Patch(facecolor="#d6604d", edgecolor="black", label="Action more right than speech"),
        ]
        ax.legend(handles=legend_elements + [plt.Line2D([0], [0], color="k", linestyle="--",
                                                        label="Speech = Action")],
                  loc="lower right", fontsize=8, framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)

    # Panel B: Per-party-topic speech vs action share (heatmap-style grouped bars)
    ax2 = axes[1]
    parties = sorted(topic_data["party"].unique())
    topics_present = [t for t in IDEOLOGY_TOPICS_ORDER if t in topic_data["topic"].unique()]
    n_parties = len(parties)
    n_topics = len(topics_present)
    width = 0.4
    x = np.arange(n_parties)

    # Plot each topic as a side-by-side pair
    import matplotlib.pyplot as _plt
    cmap = _plt.get_cmap("RdBu_r")
    palette = [cmap(i / max(n_topics - 1, 1)) for i in range(n_topics)]   

    for i, topic in enumerate(topics_present):
        sub = topic_data[topic_data["topic"] == topic].set_index("party")
        s_vals = [sub.loc[p, "speech_share"] if p in sub.index else 0 for p in parties]
        a_vals = [sub.loc[p, "action_share"] if p in sub.index else 0 for p in parties]
        offset = (i - n_topics / 2) * width + width / 2
        ax2.bar(x + offset, s_vals, width=width * 0.9, color=palette[i], alpha=0.5,
                edgecolor="white", linewidth=0.5)
        ax2.bar(x + offset, a_vals, width=width * 0.9, color=palette[i], alpha=1.0,
                edgecolor="black", linewidth=0.5, label=topic if i == 0 else None)

    ax2.set_xticks(x)
    ax2.set_xticklabels(parties, rotation=45, ha="right")
    ax2.set_xlabel("Party", fontsize=10)
    ax2.set_ylabel("Share of speech (light) vs action (solid) within party (%)", fontsize=10)
    ax2.set_title(
        "B. Within-party topic emphasis: speech (faded) vs action (solid)\n"
        "Color = ideological topic (red = left, blue = right)",
        fontsize=11,
    )
    ax2.grid(axis="y", alpha=0.3, linestyle=":")

    fig.suptitle(
        "Quid Ergo: What parties say vs what parties advance through parliamentary action",
        fontsize=13, fontweight="bold", y=1.02,
    )

    add_figure_credits(
        fig,
        n_total=len(df_agg),
        n_parties=n_parties,
        date_range=f"{min_year}-{df['year'].max()}",
        source=str(py_path),
    )

    fig.tight_layout()
    out_path = figures_dir / "figure_quid_ergo_speech_vs_action.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=Path("output/analysis"))
    parser.add_argument("--figures-dir", type=Path, default=Path("output/manuscript/figures"))
    parser.add_argument("--min-year", type=int, default=2014)
    args = parser.parse_args()

    out_path = plot_quid_ergo(args.analysis_dir, args.figures_dir, args.min_year)
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    main()
