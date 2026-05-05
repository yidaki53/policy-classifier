"""Combined final visualisation: ideological placement + heatmap of category distributions."""

import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from swedish_parliament_policy_classifier.classifier.scorer import load_definitions
from swedish_parliament_policy_classifier.analysis.aggregate import compute_party_profiles, load_party_profiles


def plot_final_profiles(conn, out_dir: str = "figures", basename: str = "party_profiles_final") -> Optional[Tuple[str, str]]:
    os.makedirs(out_dir, exist_ok=True)

    # Ensure party profiles are up-to-date; compute_party_profiles persists results
    profiles = compute_party_profiles(conn)
    if not profiles:
        profiles = load_party_profiles(conn)
    if not profiles:
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

    # Compute normalized ideological score in [0,1]
    party_scores = {}
    for party, d in profiles.items():
        avg = sum(d.get(cat, 0.0) * cat_index.get(cat, 0) for cat in ordered)
        avg_norm = avg / (n - 1) if n > 1 else 0.5
        party_scores[party] = float(avg_norm)

    parties = sorted(list(profiles.keys()), key=lambda p: party_scores[p])
    data = np.array([[profiles[party].get(cat, 0.0) for cat in ordered] for party in parties])

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

    png_path = os.path.join(out_dir, f"{basename}.png")
    pdf_path = os.path.join(out_dir, f"{basename}.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)
    return png_path, pdf_path
