#!/usr/bin/env python3
"""Generate all figures for the manuscript from classified motions.

Produces:
  1. pie_chart_categories.png — Overall ideological category distribution
  2. party_motions_stacked.png — Stacked bar: motions per party, by category
  3. party_motions_normalized.png — Normalized stacked bar (percentage)
  4. ideology_timeline.png — Category proportions over time (yearly)
  5. party_ideology_heatmap.png — Heatmap of party vs category intensity

Color scheme (ideological spectrum, left->right):
  far_left      dark red    #8B0000
  left          red         #CC3333
  centre_left   salmon      #FF7F7F
  centre        gray        #CCCCCC
  centre_right  light blue  #7FB3D5
  right         blue        #3366AA
  far_right     dark blue   #00008B

Usage:
    uv run python scripts/generate_figures.py --db data/swedish_parliament.db --out-dir figures/manuscript
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from swedish_parliament_policy_classifier.visualization.style_config import (
    CATEGORY_ORDER,
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    add_figure_credits,
    query_summary_stats,
)
from swedish_parliament_policy_classifier.provenance import write_run_provenance


def load_classifications(conn: sqlite3.Connection) -> List[Tuple]:
    """Load deterministic top-classification per motion.

    Uses explicit tie-breaking (category ASC) when normalized_weight ties occur,
    so figure inputs are stable across reruns.
    """
    cur = conn.cursor()
    cur.execute(
        """
        WITH ranked AS (
            SELECT
                c.motion_id,
                c.category,
                c.normalized_weight,
                ROW_NUMBER() OVER (
                    PARTITION BY c.motion_id
                    ORDER BY c.normalized_weight DESC, c.category ASC
                ) AS rn
            FROM classifications c
        )
        SELECT
            r.motion_id,
            r.category,
            r.normalized_weight,
            n.date,
            n.party,
            n.doc_type
        FROM ranked r
        LEFT JOIN normalized_motions n ON n.id = r.motion_id
        WHERE r.rn = 1
        """
    )
    return [tuple(row) for row in cur.fetchall()]


def prepare_data(rows: List[Tuple]) -> Dict:
    """Aggregate counts from classification rows."""
    cat_counts: Dict[str, int] = {c: 0 for c in CATEGORY_ORDER}
    party_cat_counts: Dict[str, Dict[str, int]] = {}
    party_total: Dict[str, int] = {}
    year_cat_counts: Dict[str, Dict[str, int]] = {}
    year_total: Dict[str, int] = {}

    for motion_id, category, weight, date, party, doc_type in rows:
        if category not in CATEGORY_ORDER:
            continue
        cat_counts[category] += 1

        if party and party != "NYD":
            party_cat_counts.setdefault(party, {c: 0 for c in CATEGORY_ORDER})[category] += 1
            party_total[party] = party_total.get(party, 0) + 1

        if date and len(str(date)) >= 4:
            year = str(date)[:4]
            year_cat_counts.setdefault(year, {c: 0 for c in CATEGORY_ORDER})[category] += 1
            year_total[year] = year_total.get(year, 0) + 1

    return {
        "cat_counts": cat_counts,
        "party_cat_counts": party_cat_counts,
        "party_total": party_total,
        "year_cat_counts": year_cat_counts,
        "year_total": year_total,
        "total": sum(cat_counts.values()),
    }


def plot_pie_chart(data: Dict, stats: Dict, out_dir: Path):
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
        f"Distribution of Parliamentary Motions by Ideological Category\n"
        f"(n = {total:,} classified motions, {stats['n_parties']} parties, {stats['date_range']})",
        fontsize=12,
    )
    add_figure_credits(fig, n_total=total, n_parties=stats["n_parties"], date_range=stats["date_range"])

    out_path = out_dir / "pie_chart_categories.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_party_motions(data: Dict, stats: Dict, out_dir: Path, normalized: bool = False):
    """Figure 2 & 3: Stacked bar chart of motions per party by category."""
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
        ax.barh(y_positions, vals, left=lefts, color=CATEGORY_COLORS[cat], edgecolor="white", linewidth=0.5, height=bar_height, label=CATEGORY_LABELS[cat])
        lefts += vals

    ax.set_yticks(y_positions)
    ax.set_yticklabels(parties)
    ax.set_xlabel("Percentage of Motions" if normalized else "Number of Motions", fontsize=12)
    title = "Ideological Distribution of Motions by Party"
    if normalized:
        title += " (Percentage)"
    ax.set_title(
        f"{title}\n(n = {stats['n_motions']:,} motions, {stats['date_range']})",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.invert_yaxis()
    ax.set_xlim(0, lefts.max() * 1.05)
    add_figure_credits(fig, n_total=stats["n_motions"], n_parties=stats["n_parties"], date_range=stats["date_range"])

    suffix = "_normalized" if normalized else ""
    out_path = out_dir / f"party_motions_stacked{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_ideology_timeline(data: Dict, stats: Dict, out_dir: Path):
    """Figure 4: Category proportions over time."""
    years = sorted(data["year_cat_counts"].keys())
    if len(years) < 2:
        print("Not enough temporal data for timeline. Skipping.", file=sys.stderr)
        return

    # Filter to years with >50 motions for stability
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
        ax.plot(years, proportions, color=CATEGORY_COLORS[cat], linewidth=2, marker="o", markersize=3, label=CATEGORY_LABELS[cat])

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Percentage of Motions", fontsize=12)
    ax.set_title(
        f"Ideological Category Proportions Over Time\n(n = {stats['n_motions']:,} motions, {stats['date_range']})",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    add_figure_credits(fig, n_total=stats["n_motions"], n_parties=stats["n_parties"], date_range=stats["date_range"])

    out_path = out_dir / "ideology_timeline.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_party_ideology_heatmap(data: Dict, stats: Dict, out_dir: Path):
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

    # Annotate cells
    for i in range(len(parties)):
        for j in range(len(CATEGORY_ORDER)):
            text_color = "white" if matrix[i, j] > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color=text_color, fontsize=8)

    ax.set_title(
        f"Party Ideology Intensity (% of Party Motions)\n"
        f"(n = {stats['n_motions']:,} motions, {stats['n_parties']} parties, {stats['date_range']})",
        fontsize=12,
    )
    fig.colorbar(im, ax=ax, label="Percentage of Party Motions")
    add_figure_credits(fig, n_total=stats["n_motions"], n_parties=stats["n_parties"], date_range=stats["date_range"])

    out_path = out_dir / "party_ideology_heatmap.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_all_figures(db_path: str, out_dir: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Loading classifications from database...", file=sys.stderr)
    rows = load_classifications(conn)
    print(f"Loaded {len(rows)} classified motions.", file=sys.stderr)

    if not rows:
        print("No classifications found. Run classify_batch.py first.", file=sys.stderr)
        sys.exit(1)

    data = prepare_data(rows)
    stats = query_summary_stats(conn)
    print(f"Total classified motions: {data['total']}", file=sys.stderr)
    print(f"Stats: {stats}", file=sys.stderr)

    print("Generating figures...", file=sys.stderr)
    plot_pie_chart(data, stats, out_path)
    plot_party_motions(data, stats, out_path, normalized=False)
    plot_party_motions(data, stats, out_path, normalized=True)
    plot_ideology_timeline(data, stats, out_path)
    plot_party_ideology_heatmap(data, stats, out_path)

    provenance_path = write_run_provenance(
        script="scripts/generate_figures.py",
        inputs={"db": db_path},
        outputs=[
            str(out_path / "pie_chart_categories.png"),
            str(out_path / "party_motions_stacked.png"),
            str(out_path / "party_motions_stacked_normalized.png"),
            str(out_path / "ideology_timeline.png"),
            str(out_path / "party_ideology_heatmap.png"),
        ],
        output_dir=out_path,
        metadata={
            "n_classified_motions": data["total"],
            "n_parties": stats.get("n_parties"),
            "date_range": stats.get("date_range"),
        },
    )
    print(f"Saved provenance: {provenance_path}", file=sys.stderr)

    print(f"\nAll figures saved to {out_path}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Generate manuscript figures from classifications")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--out-dir", default="figures/manuscript")
    args = parser.parse_args()

    generate_all_figures(args.db, args.out_dir)


if __name__ == "__main__":
    main()
