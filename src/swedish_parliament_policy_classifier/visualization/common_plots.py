"""Common plotting utilities shared across analysis scripts.

These are the standard academic figure types used for:
  - Part 1: pure motion classifications
  - Part 2: combined motions+votes
  - Part 3: speeches vs motions vs votes

All functions accept a data dict with the same keys as generate_figures.prepare_data().
"""

import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swedish_parliament_policy_classifier.visualization.style_config import (
    CATEGORY_ORDER,
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    add_figure_credits,
)


def plot_pie_chart(data: Dict, stats: Dict, out_dir: Path, filename_base: str = "pie_chart_categories"):
    """Figure 1: Pie chart of overall category proportions."""
    counts = [data["cat_counts"][c] for c in CATEGORY_ORDER]
    labels = [CATEGORY_LABELS[c] for c in CATEGORY_ORDER]
    colors = [CATEGORY_COLORS[c] for c in CATEGORY_ORDER]
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(9, 9))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else "",
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in texts:
        t.set_fontsize(11)
    for t in autotexts:
        t.set_fontsize(10)
        t.set_color("white" if t.get_text() else "none")

    ax.set_title(
        f"Distribution by Ideological Category\n"
        f"(n = {total:,}, {stats.get('n_parties', '?')} parties, {stats.get('date_range', '')})",
        fontsize=12,
    )
    add_figure_credits(
        fig,
        n_total=total,
        n_parties=stats.get("n_parties"),
        date_range=stats.get("date_range"),
    )

    out_path = out_dir / f"{filename_base}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_party_motions(data: Dict, stats: Dict, out_dir: Path, normalized: bool = False, filename_base: str = "party_motions_stacked"):
    """Figure 2 & 3: Stacked bar chart of items per party by category."""
    parties = sorted(data["party_cat_counts"].keys(), key=lambda p: data["party_total"].get(p, 0), reverse=True)
    if not parties:
        print("No party data available. Skipping party bar chart.", file=sys.stderr)
        return

    n_parties = len(parties)
    bar_height = 0.6
    fig_height = max(6, 0.5 * n_parties)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_positions = np.arange(n_parties)
    lefts = np.zeros(n_parties)

    for cat in CATEGORY_ORDER:
        vals = np.array([data["party_cat_counts"][p].get(cat, 0) for p in parties], dtype=float)
        if normalized:
            totals = np.array([data["party_total"].get(p, 1) for p in parties], dtype=float)
            vals = np.divide(vals, totals, out=np.zeros_like(vals), where=totals > 0) * 100
        ax.barh(
            y_positions, vals, left=lefts,
            color=CATEGORY_COLORS[cat], edgecolor="white",
            linewidth=0.5, height=bar_height, label=CATEGORY_LABELS[cat],
        )
        lefts += vals

    ax.set_yticks(y_positions)
    ax.set_yticklabels(parties)
    ax.set_xlabel("Percentage" if normalized else "Number of Items", fontsize=12)
    title = "Ideological Distribution by Party"
    if normalized:
        title += " (Percentage)"
    ax.set_title(
        f"{title}\n(n = {stats.get('n_items', stats.get('n_motions', '?')):,}, {stats.get('date_range', '')})",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.invert_yaxis()
    ax.set_xlim(0, lefts.max() * 1.05)
    add_figure_credits(
        fig,
        n_total=stats.get("n_items", stats.get("n_motions")),
        n_parties=stats.get("n_parties"),
        date_range=stats.get("date_range"),
    )

    suffix = "_normalized" if normalized else ""
    out_path = out_dir / f"{filename_base}{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_ideology_timeline(data: Dict, stats: Dict, out_dir: Path, filename_base: str = "ideology_timeline"):
    """Figure 4: Category proportions over time."""
    years = sorted(data["year_cat_counts"].keys())
    if len(years) < 2:
        print("Not enough temporal data for timeline. Skipping.", file=sys.stderr)
        return

    # Filter to years with >50 items for stability
    years = [y for y in years if data["year_total"].get(y, 0) >= 50]
    if len(years) < 2:
        print("Not enough years with sufficient data. Skipping timeline.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for cat in CATEGORY_ORDER:
        proportions = []
        for year in years:
            total = data["year_total"].get(year, 1)
            count = data["year_cat_counts"][year].get(cat, 0)
            proportions.append(count / total * 100)
        ax.plot(
            years, proportions,
            color=CATEGORY_COLORS[cat], linewidth=2,
            marker="o", markersize=3, label=CATEGORY_LABELS[cat],
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Percentage", fontsize=12)
    ax.set_title(
        f"Ideological Category Proportions Over Time\n(n = {stats.get('n_items', stats.get('n_motions', '?')):,}, {stats.get('date_range', '')})",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    add_figure_credits(
        fig,
        n_total=stats.get("n_items", stats.get("n_motions")),
        n_parties=stats.get("n_parties"),
        date_range=stats.get("date_range"),
    )

    out_path = out_dir / f"{filename_base}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_party_ideology_heatmap(data: Dict, stats: Dict, out_dir: Path, filename_base: str = "party_ideology_heatmap"):
    """Figure 5: Heatmap of party vs category (normalized by party)."""
    parties = sorted(data["party_cat_counts"].keys(), key=lambda p: data["party_total"].get(p, 0), reverse=True)
    if not parties:
        print("No party data for heatmap. Skipping.", file=sys.stderr)
        return

    matrix = []
    for party in parties:
        total = data["party_total"].get(party, 1)
        row = [data["party_cat_counts"][party].get(cat, 0) / total * 100 for cat in CATEGORY_ORDER]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(parties) + 1)))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=0, vmax=matrix.max())

    ax.set_xticks(np.arange(len(CATEGORY_ORDER)))
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORY_ORDER], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(parties)))
    ax.set_yticklabels(parties)

    for i in range(len(parties)):
        for j in range(len(CATEGORY_ORDER)):
            text_color = "white" if matrix[i, j] > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color=text_color, fontsize=8)

    ax.set_title(
        f"Party Ideology Intensity (% of Party Items)\n"
        f"(n = {stats.get('n_items', stats.get('n_motions', '?')):,}, {stats.get('n_parties', '?')} parties, {stats.get('date_range', '')})",
        fontsize=12,
    )
    fig.colorbar(im, ax=ax, label="Percentage")
    add_figure_credits(
        fig,
        n_total=stats.get("n_items", stats.get("n_motions")),
        n_parties=stats.get("n_parties"),
        date_range=stats.get("date_range"),
    )

    out_path = out_dir / f"{filename_base}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
