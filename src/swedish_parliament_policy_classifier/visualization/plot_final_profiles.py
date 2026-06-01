"""Combined final visualisation: ideological placement + heatmap of category distributions."""

import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from swedish_parliament_policy_classifier.exports import load_definitions

if False:
    from swedish_parliament_policy_classifier.exports import load_definitions as _ld
    _ = _ld
from swedish_parliament_policy_classifier.analysis.aggregate import compute_party_profiles, load_party_profiles
from swedish_parliament_policy_classifier.visualization.style_config import (
    add_figure_credits,
    query_summary_stats,
)


def plot_final_profiles(conn, out_dir: str = "figures", basename: str = "party_profiles_final") -> Optional[Tuple[str, str]]:
    os.makedirs(out_dir, exist_ok=True)

    # `conn` may be either a sqlite connection or a path to a parquet
    # `party_profiles` table. Support both: if a parquet path is provided
    # load and convert to the mapping expected by downstream code.
    profiles = None
    try:
        if isinstance(conn, str) and os.path.exists(conn) and conn.lower().endswith(".parquet"):
            df = pd.read_parquet(conn)
            # expected columns: party, modality, category, proportion
            if df.empty:
                return None
            grp = df.groupby(["party", "category"], as_index=False)["proportion"].sum()
            party_map = {}
            for party, g in grp.groupby("party"):
                total = float(g["proportion"].sum() or 0.0)
                mapping = {row["category"]: float(row["proportion"]) / total if total > 0 else 0.0 for _, row in g.iterrows()}
                party_map[party] = mapping
            profiles = party_map
        else:
            profiles = compute_party_profiles(conn)
            if not profiles:
                profiles = load_party_profiles(conn)
    except Exception:
        return None

    defs = load_definitions()

    # Preferred left->right ordering
    ordered_cats = [
        "far_left",
        "left",
        "centre_left",
        "centre",
        "centre_right",
        "right",
        "far_right",
    ]
    ordered = [c for c in ordered_cats if c in defs]
    if not ordered:
        ordered = list(defs.keys())

    n = len(ordered)
    cat_index = {cat: i for i, cat in enumerate(ordered)}

    # Exclude historical party NYD from visualisation
    profiles = {p: d for p, d in profiles.items() if p != "NYD"}

    # Compute normalized ideological score in [0,1]
    party_scores = {}
    for party, d in profiles.items():
        props = d.get("proportions", {})
        avg = sum(props.get(cat, 0.0) * cat_index.get(cat, 0) for cat in ordered)
        avg_norm = avg / (n - 1) if n > 1 else 0.5
        party_scores[party] = float(avg_norm)

    parties = sorted(list(profiles.keys()), key=lambda p: party_scores[p])
    data = np.array([[profiles[party].get("proportions", {}).get(cat, 0.0) for cat in ordered] for party in parties])

    fig = plt.figure(figsize=(12, max(4, 0.6 * len(parties))))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3])
    ax_top = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])
    # reserve space for long party names and rotated xlabels
    fig.subplots_adjust(left=0.25, right=0.92, top=0.92, bottom=0.18, hspace=0.35)

    # Top plot: ideological placement (scatter along [0,1])
    scores = [party_scores[p] for p in parties]
    y = np.arange(len(parties))
    cmap = plt.get_cmap("RdYlBu_r")
    colors = [cmap(s) for s in scores]
    ax_top.scatter(scores, y, c=colors, s=120, edgecolors="black")
    ax_top.set_yticks(y)
    ax_top.set_yticklabels(parties)
    ax_top.set_xlim(-0.02, 1.02)
    ax_top.set_xlabel("Ideological placement (Left=0 → Right=1)")
    ax_top.set_title("Party ideological placement — weighted average across categories", pad=8)
    # annotate numeric ideological score next to each point
    for i, s in enumerate(scores):
        ax_top.text(s + 0.02, y[i], f"{s:.2f}", va="center", fontsize=9)
    ax_top.invert_yaxis()

    # Error bars: horizontal ±1 SE on the placement score
    for i, party in enumerate(parties):
        se = profiles[party].get("placement_se", 0.0)
        if se > 0:
            ax_top.errorbar(scores[i], y[i], xerr=se, fmt="none", ecolor="black", alpha=0.5, capsize=3)

    # Bottom: heatmap of distributions
    im = ax_heat.imshow(data, aspect="auto", cmap="YlGnBu", interpolation="nearest", vmin=0, vmax=1)
    ax_heat.set_yticks(y)
    ax_heat.set_yticklabels(parties)
    ax_heat.set_xticks(np.arange(len(ordered)))
    ax_heat.set_xticklabels(ordered, rotation=45, ha="right")
    ax_heat.set_xlabel("Category")
    ax_heat.set_title("Party category distribution (proportions)", pad=8)

    # colorbar attached to the heatmap only
    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical", fraction=0.04, pad=0.02)
    cbar.set_label("Proportion")

    # annotate heatmap cells with values (if not too dense)
    nr, nc = data.shape
    if nr * nc <= 200:
        for i in range(nr):
            for j in range(nc):
                val = data[i, j]
                if val > 0:
                    txt_color = "white" if val > 0.45 else "black"
                    ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=8)

    # Compute caption stats. If `conn` is a sqlite connection use the DB query,
    # otherwise derive minimal stats from the in-memory `profiles` mapping.
    if hasattr(conn, "cursor"):
        stats = query_summary_stats(conn)
    else:
        stats = {"n_motions": None, "n_parties": len(profiles), "date_range": None}
    add_figure_credits(fig, n_total=stats["n_motions"], n_parties=stats["n_parties"], date_range=stats["date_range"], recency_weighted=True)

    png_path = os.path.join(out_dir, f"{basename}.png")
    pdf_path = os.path.join(out_dir, f"{basename}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path
