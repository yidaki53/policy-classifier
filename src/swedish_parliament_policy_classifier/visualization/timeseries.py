"""Timeseries visualisation of ideological shifts by party over time.

Generates a multi-panel figure showing each party's ideological category
proportions per parliament year. Uses recency-weighted smoothing within
5-year windows.
"""

import os
import sys
import math
import sqlite3
import json
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions
from swedish_parliament_policy_classifier.visualization.style_config import (
    add_figure_credits,
    query_summary_stats,
)


# Parliament periods and their years
PARLIAMENT_PERIODS = [
    ("1971-1979", 1971, 1979),
    ("1980-1989", 1980, 1989),
    ("1990-1997", 1990, 1997),
    ("1998-2001", 1998, 2001),
    ("2002-2005", 2002, 2005),
    ("2006-2009", 2006, 2009),
    ("2010-2013", 2010, 2013),
    ("2014-2017", 2014, 2017),
    ("2018-2021", 2018, 2021),
    ("2022-2025", 2022, 2025),
]


def _parse_date(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str.strip()[:10], "%Y-%m-%d").year
    except ValueError:
        try:
            return int(date_str.strip()[:4])
        except ValueError:
            return None


def _get_period(year: int) -> Optional[str]:
    for name, start, end in PARLIAMENT_PERIODS:
        if start <= year <= end:
            return name
    return None


def compute_timeseries(
    conn,
    parties: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return {party: {period: {category: proportion}}}"""
    cur = conn.cursor()
    defs = load_definitions()
    categories = list(defs.keys())

    # If no parties specified, use major Swedish parties
    if parties is None:
        parties = ["V", "MP", "S", "C", "KD", "L", "M", "SD"]

    # Fetch all classifications with dates
    cur.execute(
        """
        SELECT nm.party, c.category, c.normalized_weight, nm.date
        FROM classifications c
        JOIN normalized_motions nm ON c.motion_id = nm.id
        WHERE nm.party IS NOT NULL AND nm.party != ''
        """
    )

    # Aggregate into periods
    party_periods: Dict[str, Dict[str, Dict[str, float]]] = {}
    for party in parties:
        party_periods[party] = {}

    for row in cur.fetchall():
        party = row[0]
        if party not in parties:
            continue
        category = row[1]
        weight = row[2] or 0.0
        year = _parse_date(row[3])
        if year is None:
            continue

        period = _get_period(year)
        if period is None:
            continue

        if period not in party_periods[party]:
            party_periods[party][period] = {cat: 0.0 for cat in categories}
        party_periods[party][period][category] += float(weight)

    # Normalize per party per period
    for party in parties:
        for period, cat_weights in party_periods[party].items():
            total = sum(cat_weights.values())
            if total > 0:
                for cat in categories:
                    cat_weights[cat] /= total
            else:
                for cat in categories:
                    cat_weights[cat] = 0.0

    return party_periods


def plot_timeseries(
    conn,
    out_dir: str = "figures",
    parties: Optional[List[str]] = None,
) -> Optional[Tuple[str, str]]:
    os.makedirs(out_dir, exist_ok=True)

    ts = compute_timeseries(conn, parties=parties)
    if not ts:
        return None

    ordered_cats = [
        "far_left", "left", "centre_left",
        "centre", "centre_right", "right", "far_right",
    ]

    # Filter to categories present in data
    all_cats = set()
    for party_data in ts.values():
        for period_data in party_data.values():
            all_cats.update(period_data.keys())
    ordered = [c for c in ordered_cats if c in all_cats]
    if not ordered:
        return None

    # Ordered periods
    periods = [p[0] for p in PARLIAMENT_PERIODS]
    period_indices = {p: i for i, p in enumerate(periods)}

    # Ordered parties by left-right position
    party_list = sorted(ts.keys())

    # Create figure with subplots per party
    fig, axes = plt.subplots(len(party_list), 1, figsize=(12, 2.5 * len(party_list)), sharex=True)
    if len(party_list) == 1:
        axes = [axes]

    cmap = plt.get_cmap("RdYlBu")
    n = len(ordered)
    colors = [cmap(i / max(1, n - 1)) for i in range(n)]
    cat_index = {cat: i for i, cat in enumerate(ordered)}

    for ax, party in zip(axes, party_list):
        period_data = ts[party]

        # Prepare data arrays
        x = []
        y_data = {cat: [] for cat in ordered}

        for period in periods:
            if period in period_data:
                x.append(period_indices[period])
                for cat in ordered:
                    y_data[cat].append(period_data[period].get(cat, 0.0))

        # Plot stacked area
        bottom = np.zeros(len(x))
        for cat in ordered:
            vals = np.array(y_data[cat])
            color_idx = cat_index[cat]
            ax.fill_between(x, bottom, bottom + vals, color=colors[color_idx], label=cat, alpha=0.85)
            bottom += vals

        ax.set_ylim(0, 1)
        ax.set_ylabel(party)
        ax.set_yticks([0, 0.5, 1.0])
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

        # Add annotations for shifts
        if len(x) >= 2:
            left_sum = np.zeros(len(x))
            right_sum = np.zeros(len(x))
            for cat in ordered:
                if cat in ["far_left", "left", "centre_left"]:
                    left_sum += np.array(y_data[cat])
                elif cat in ["right", "far_right", "centre_right"]:
                    right_sum += np.array(y_data[cat])

            if len(x) > 1 and left_sum[-1] > right_sum[-1] and left_sum[0] <= right_sum[0]:
                ax.annotate("shifted left", xy=(x[-1], 0.5), fontsize=8, ha='right', color='red')
            elif len(x) > 1 and right_sum[-1] > left_sum[-1] and right_sum[0] <= left_sum[0]:
                ax.annotate("shifted right", xy=(x[-1], 0.5), fontsize=8, ha='right', color='blue')

    axes[-1].set_xticks(range(len(periods)))
    axes[-1].set_xticklabels(periods, rotation=45, ha='right')
    axes[-1].set_xlabel("Parliament Period")

    # Legend on top subplot
    handles = [plt.Rectangle((0,0),1,1, color=colors[cat_index[cat]]) for cat in ordered]
    axes[0].legend(handles, ordered, loc='upper left', bbox_to_anchor=(1.02, 1), ncol=1)

    stats = query_summary_stats(conn)
    fig.suptitle(
        f"Party Ideological Shifts Over Time (Parliament Periods)\n"
        f"(n = {stats['n_motions']:,} motions, {stats['n_parties']} parties, {stats['date_range']})",
        fontsize=12,
        y=0.995,
    )
    plt.tight_layout()
    add_figure_credits(fig, n_total=stats["n_motions"], n_parties=stats["n_parties"], date_range=stats["date_range"], recency_weighted=True)

    png_path = os.path.join(out_dir, "party_timeseries.png")
    pdf_path = os.path.join(out_dir, "party_timeseries.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved timeseries to {png_path} and {pdf_path}", file=sys.stderr)
    return png_path, pdf_path


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/swedish_parliament.db"
    conn = init_db(db_path)
    out = plot_timeseries(conn)
    conn.close()
    print(out)
