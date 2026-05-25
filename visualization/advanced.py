"""Advanced visualizations: weighted aggregation, clustering heatmap, PCA biplot."""

import os
import json
import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA

from swedish_parliament_policy_classifier.exports import load_definitions

if False:
    from swedish_parliament_policy_classifier.exports import load_definitions as _ld
    _ = _ld


def compute_weighted_party_profiles(conn, half_life_days: float = 365.0, strength_field: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """Compute party profiles weighted by recency and optional strength field.

    - half_life_days: half-life for exponential decay (in days). Set to 0 or None to disable decay.
    - strength_field: key in normalized_motions.metadata JSON to use as motion strength multiplier.
    """
    cur = conn.cursor()
    defs = load_definitions()
    categories = list(defs.keys())

    cur.execute("SELECT nm.id, nm.party, nm.date, nm.metadata, c.category, c.normalized_weight FROM classifications c JOIN normalized_motions nm ON c.motion_id = nm.id")
    rows = cur.fetchall()

    party_map: Dict[str, Dict[str, float]] = {}

    now = datetime.now(timezone.utc)

    for row in rows:
        motion_id = row[0]
        party = row[1] or "Unknown"
        date_raw = row[2]
        metadata_raw = row[3]
        category = row[4]
        normalized_weight = float(row[5] or 0.0)

        # compute time decay
        weight_time = 1.0
        if date_raw and half_life_days and half_life_days > 0:
            try:
                dt = datetime.fromisoformat(date_raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                days = max(0.0, (now - dt).total_seconds() / 86400.0)
                decay_rate = math.log(2) / half_life_days
                weight_time = math.exp(-decay_rate * days)
            except Exception:
                weight_time = 1.0

        # extract strength
        strength = 1.0
        if metadata_raw and strength_field:
            try:
                m = json.loads(metadata_raw)
                v = m.get(strength_field)
                if isinstance(v, (int, float)):
                    strength = float(v)
            except Exception:
                strength = 1.0

        motion_weight = weight_time * (strength or 1.0)

        if party not in party_map:
            party_map[party] = {cat: 0.0 for cat in categories}

        # accumulate weighted normalized weight
        if category not in party_map[party]:
            party_map[party][category] = 0.0
        party_map[party][category] += normalized_weight * motion_weight

    # normalize per party
    for party, d in party_map.items():
        total = sum(d.values())
        if total > 0:
            for cat in categories:
                d[cat] = float(d.get(cat, 0.0) / total)
        else:
            for cat in categories:
                d[cat] = 0.0

    # persist back to party_profiles if desired by caller
    return party_map


def plot_clustered_heatmap(conn, out_dir: str = "figures", basename: str = "party_clustered") -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    profiles = compute_weighted_party_profiles(conn)
    if not profiles:
        raise RuntimeError("No party profiles available to plot")

    defs = load_definitions()
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

    parties = list(profiles.keys())
    data = np.array([[profiles[p].get(c, 0.0) for c in ordered] for p in parties])

    # clustering on parties
    if len(parties) > 1:
        Z = linkage(data, method="ward")
        dendro = dendrogram(Z, labels=parties, orientation="left", no_plot=True)
        order = dendro["leaves"]
    else:
        order = [0]

    ordered_parties = [parties[i] for i in order]
    data_ord = data[order, :]

    fig = plt.figure(figsize=(10, max(4, 0.5 * len(parties))))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.25, 0.75], wspace=0.05)
    ax_dend = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    if len(parties) > 1:
        dendrogram(Z, labels=parties, orientation="left", ax=ax_dend, color_threshold=0)
    ax_dend.set_xticks([])
    ax_dend.set_yticks([])

    im = ax_heat.imshow(data_ord, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax_heat.set_yticks(np.arange(len(ordered_parties)))
    ax_heat.set_yticklabels(ordered_parties)
    ax_heat.set_xticks(np.arange(len(ordered)))
    ax_heat.set_xticklabels(ordered, rotation=45, ha="right")
    ax_heat.set_title("Clustered party-category heatmap")

    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical", fraction=0.04, pad=0.02)
    cbar.set_label("Proportion")

    png = os.path.join(out_dir, f"{basename}.png")
    pdf = os.path.join(out_dir, f"{basename}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_pca_biplot(conn, out_dir: str = "figures", basename: str = "party_pca") -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    profiles = compute_weighted_party_profiles(conn)
    if not profiles:
        raise RuntimeError("No party profiles available to plot")

    defs = load_definitions()
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

    parties = list(profiles.keys())
    data = np.array([[profiles[p].get(c, 0.0) for c in ordered] for p in parties])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(data)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(coords[:, 0], coords[:, 1], s=100, edgecolors="k")
    for i, p in enumerate(parties):
        ax.text(coords[i, 0] + 0.01, coords[i, 1], p, fontsize=9)

    # plot category loadings as arrows
    loadings = pca.components_.T
    for i, cat in enumerate(ordered):
        ax.arrow(0, 0, loadings[i, 0] * 2.0, loadings[i, 1] * 2.0, color="red", alpha=0.7, head_width=0.02)
        ax.text(loadings[i, 0] * 2.1, loadings[i, 1] * 2.1, cat, color="red", fontsize=9)

    ax.set_title("PCA biplot of party category distributions")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    png = os.path.join(out_dir, f"{basename}.png")
    pdf = os.path.join(out_dir, f"{basename}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png, pdf
