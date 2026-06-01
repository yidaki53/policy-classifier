"""Plotting utilities for party profiles."""

import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt

from swedish_parliament_policy_classifier.exports import load_definitions

# (removed Graphify import-hint)
from swedish_parliament_policy_classifier.analysis.aggregate import compute_party_profiles
from swedish_parliament_policy_classifier.visualization.style_config import (
    add_figure_credits,
    query_summary_stats,
)


def plot_party_profiles(conn, out_dir: str = "figures") -> Optional[Tuple[str, str]]:
    os.makedirs(out_dir, exist_ok=True)
    profiles = compute_party_profiles(conn)
    if not profiles:
        return None
    # Exclude historical party NYD
    profiles = {p: d for p, d in profiles.items() if p != "NYD"}

    # Prefer an explicit left->right ordering
    ordered_cats = [
        "far_left",
        "left",
        "centre_left",
        "centre",
        "centre_right",
        "right",
        "far_right",
    ]

    defs = load_definitions()
    # Fallback: use definitions order if explicit order incomplete
    all_defs = list(defs.keys())
    ordered = [c for c in ordered_cats if c in defs]
    if not ordered:
        ordered = all_defs

    # Compute party ordering by weighted average on the categorical index
    cat_index = {cat: i for i, cat in enumerate(ordered)}
    party_avgs = {}
    for party, d in profiles.items():
        props = d.get("proportions", {})
        avg = sum(props.get(cat, 0.0) * cat_index.get(cat, 0) for cat in ordered)
        party_avgs[party] = avg

    parties = sorted(profiles.keys(), key=lambda p: party_avgs[p])

    data = [[profiles[party].get("proportions", {}).get(cat, 0.0) for cat in ordered] for party in parties]

    fig, ax = plt.subplots(figsize=(8, max(2, 0.5 * len(parties))))
    cmap = plt.get_cmap("RdYlBu")
    n = len(ordered)
    colors = [cmap(i / max(1, n - 1)) for i in range(n)]

    y_positions = list(range(len(parties)))
    lefts = [0.0] * len(parties)

    for i, cat in enumerate(ordered):
        vals = [row[i] for row in data]
        ax.barh(y_positions, vals, left=lefts, color=colors[i], edgecolor="white", label=cat)
        lefts = [l + v for l, v in zip(lefts, vals)]

    ax.set_yticks(y_positions)
    ax.set_yticklabels(parties)
    ax.set_xlim(0, 1)
    stats = query_summary_stats(conn)
    ax.set_xlabel("Proportion across categories")
    ax.set_title(
        f"Party Policy Profile — Category Distribution\n"
        f"(n = {stats['n_motions']:,} motions, {stats['n_parties']} parties, {stats['date_range']})",
        fontsize=12,
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    add_figure_credits(fig, n_total=stats["n_motions"], n_parties=stats["n_parties"], date_range=stats["date_range"], recency_weighted=True)

    png_path = os.path.join(out_dir, "party_profiles.png")
    pdf_path = os.path.join(out_dir, "party_profiles.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path
