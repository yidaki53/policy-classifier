"""Additional visualisations and analysis for the classified corpus.

Produces:
- Category distribution pie/bar chart (overall)
- Ideological spectrum heatmap (parties vs categories)
- Party polarization index over time
"""

import os
import sys
import sqlite3
from typing import Optional, Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions
from swedish_parliament_policy_classifier.visualization.timeseries import compute_timeseries
from swedish_parliament_policy_classifier.visualization.style_config import (
    add_figure_credits,
    query_summary_stats,
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    CATEGORY_COLORS,
)


def plot_category_distribution(conn, out_dir: str = "figures") -> Optional[Tuple[str, str]]:
    """Bar chart of top-weighted categories across all motions."""
    os.makedirs(out_dir, exist_ok=True)
    cur = conn.cursor()

    cur.execute("""
    WITH ranked AS (
        SELECT motion_id, category, normalized_weight,
               ROW_NUMBER() OVER (PARTITION BY motion_id ORDER BY normalized_weight DESC) as rn
        FROM classifications
    )
    SELECT category, COUNT(*) as cnt
    FROM ranked
    WHERE rn = 1
    GROUP BY category
    ORDER BY cnt DESC
    """)

    cats, counts = [], []
    for row in cur.fetchall():
        cats.append(row[0])
        counts.append(row[1])

    total = sum(counts)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    colors = plt.cm.RdYlBu(np.linspace(0, 1, len(cats)))
    bars = ax1.barh(range(len(cats)), counts, color=colors)
    ax1.set_yticks(range(len(cats)))
    ax1.set_yticklabels(cats)
    ax1.invert_yaxis()
    ax1.set_xlabel("Motions (top-weighted category)")
    ax1.set_title("Category Distribution (All Motions)")

    # Add percentage labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        pct = 100 * count / total
        ax1.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%", va='center', fontsize=9)

    # Pie chart
    wedges, texts, autotexts = ax2.pie(counts, labels=cats, autopct="%1.1f%%",
                                        colors=colors, startangle=90)
    ax2.set_title("Category Proportions")

    stats = query_summary_stats(conn)
    add_figure_credits(fig, n_total=total, n_parties=stats["n_parties"], date_range=stats["date_range"], recency_weighted=True)

    plt.tight_layout()
    png = os.path.join(out_dir, "category_distribution.png")
    pdf = os.path.join(out_dir, "category_distribution.pdf")
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved category distribution to {png}", file=sys.stderr)
    return png, pdf


def plot_spectrum_heatmap(conn, out_dir: str = "figures") -> Optional[Tuple[str, str]]:
    """Heatmap: parties (rows) x categories (columns), showing mean normalized_weight."""
    os.makedirs(out_dir, exist_ok=True)
    cur = conn.cursor()
    defs = load_definitions()
    categories = list(defs.keys())

    cur.execute("""
    SELECT nm.party, c.category, AVG(c.normalized_weight) as avg_weight
    FROM classifications c
    JOIN normalized_motions nm ON c.motion_id = nm.id
    WHERE nm.party IS NOT NULL AND nm.party != ''
    GROUP BY nm.party, c.category
    """)

    # Build matrix: party -> category -> avg_weight
    data: Dict[str, Dict[str, float]] = {}
    for row in cur.fetchall():
        party, cat, weight = row[0], row[1], row[2] or 0.0
        if party not in data:
            data[party] = {}
        data[party][cat] = float(weight)

    # Filter to major parties
    major_parties = ["V", "S", "MP", "C", "KD", "L", "M", "SD"]
    party_list = [p for p in major_parties if p in data]
    if not party_list:
        party_list = sorted(data.keys())

    ordered_cats = ["far_left", "left", "centre_left", "centre", "centre_right", "right", "far_right"]
    cat_list = [c for c in ordered_cats if c in categories]
    if not cat_list:
        cat_list = categories

    # Build matrix
    mat = np.zeros((len(party_list), len(cat_list)))
    for i, party in enumerate(party_list):
        for j, cat in enumerate(cat_list):
            mat[i, j] = data.get(party, {}).get(cat, 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mat, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=mat.max())

    ax.set_xticks(range(len(cat_list)))
    ax.set_xticklabels(cat_list, rotation=45, ha='right')
    ax.set_yticks(range(len(party_list)))
    ax.set_yticklabels(party_list)

    # Add text annotations
    for i in range(len(party_list)):
        for j in range(len(cat_list)):
            val = mat[i, j]
            text = ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                          color="white" if val > mat.max() * 0.6 else "black", fontsize=8)

    stats = query_summary_stats(conn)
    ax.set_title(
        f"Mean Category Weight by Party (Ideological Spectrum)\n"
        f"(n = {stats['n_motions']:,} motions, {stats['n_parties']} parties, {stats['date_range']})",
        fontsize=12,
    )
    fig.colorbar(im, ax=ax, label="Mean Normalized Weight")
    plt.tight_layout()
    add_figure_credits(fig, n_total=stats["n_motions"], n_parties=stats["n_parties"], date_range=stats["date_range"], recency_weighted=True)

    png = os.path.join(out_dir, "spectrum_heatmap.png")
    pdf = os.path.join(out_dir, "spectrum_heatmap.pdf")
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved spectrum heatmap to {png}", file=sys.stderr)
    return png, pdf


def plot_polarization_index(conn, out_dir: str = "figures") -> Optional[Tuple[str, str]]:
    """Polarization index per parliament period: variance of category proportions
    weighted by ideological distance from centre.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = compute_timeseries(conn)

    # Ideological distance weights
    cat_distances = {
        "far_left": -3, "left": -2, "centre_left": -1,
        "centre": 0,
        "centre_right": 1, "right": 2, "far_right": 3,
    }

    from swedish_parliament_policy_classifier.visualization.timeseries import PARLIAMENT_PERIODS
    periods = [p[0] for p in PARLIAMENT_PERIODS]
    major_parties = ["V", "S", "MP", "C", "KD", "L", "M", "SD"]

    # Compute polarization: weighted std dev from centre across parties per period
    polarization = {period: [] for period in periods}

    for party in major_parties:
        if party not in ts:
            continue
        for period in periods:
            if period not in ts[party]:
                continue
            # Weighted average position
            position = 0.0
            for cat, weight in ts[party][period].items():
                position += cat_distances.get(cat, 0) * weight
            polarization[period].append(position)

    # Mean absolute deviation from centre per period
    periods_with_data = []
    pol_values = []
    for period in periods:
        vals = polarization[period]
        if vals:
            periods_with_data.append(period)
            # Inter-party polarization: std dev of positions
            pol_values.append(np.std(vals))

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(periods_with_data))
    ax.plot(x, pol_values, marker='o', linewidth=2, markersize=8, color='darkred')
    ax.fill_between(x, pol_values, alpha=0.3, color='darkred')
    ax.set_xticks(x)
    ax.set_xticklabels(periods_with_data, rotation=45, ha='right')
    ax.set_ylabel("Polarization Index (std dev of party positions)")
    ax.set_xlabel("Parliament Period")
    ax.set_title("Inter-Party Polarization Over Time")
    stats = query_summary_stats(conn)
    ax.axhline(y=np.mean(pol_values), color='gray', linestyle='--', alpha=0.5, label=f"Mean={np.mean(pol_values):.2f}")
    ax.legend()
    ax.set_title(
        f"Inter-Party Polarization Over Time\n"
        f"(n = {stats['n_motions']:,} motions, {stats['n_parties']} parties, {stats['date_range']})",
        fontsize=12,
    )
    plt.tight_layout()
    add_figure_credits(fig, n_total=stats["n_motions"], n_parties=stats["n_parties"], date_range=stats["date_range"], recency_weighted=True)

    png = os.path.join(out_dir, "polarization_timeseries.png")
    pdf = os.path.join(out_dir, "polarization_timeseries.pdf")
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved polarization timeseries to {png}", file=sys.stderr)
    return png, pdf


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/swedish_parliament.db"
    conn = init_db(db_path)

    print("Generating category distribution...", file=sys.stderr)
    plot_category_distribution(conn)

    print("Generating spectrum heatmap...", file=sys.stderr)
    plot_spectrum_heatmap(conn)

    print("Generating polarization timeseries...", file=sys.stderr)
    plot_polarization_index(conn)

    conn.close()
    print("Done.", file=sys.stderr)
