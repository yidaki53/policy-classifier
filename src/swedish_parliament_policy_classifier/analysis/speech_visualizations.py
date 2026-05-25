from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

IDEOLOGY_ORDER = [
    "far_left",
    "left",
    "centre_left",
    "centre",
    "centre_right",
    "right",
    "far_right",
]


def _pick_first(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_speech_metadata(speech_parquet_dir: str | Path) -> pd.DataFrame:
    root = Path(speech_parquet_dir)
    rows = []
    for p in sorted(root.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue

        sid_col = _pick_first(df, ["anforande_id", "speech_id"])
        party_col = _pick_first(df, ["parti", "party", "parti_kod"])
        date_col = _pick_first(df, ["datum", "date", "anforandedatum"])
        if not sid_col or not party_col:
            continue

        part = pd.DataFrame(
            {
                "speech_id": df[sid_col].astype(str),
                "party": df[party_col].astype(str),
                "date": pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT,
            }
        )
        rows.append(part)

    if not rows:
        return pd.DataFrame(columns=["speech_id", "party", "date"])

    out = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["speech_id"], keep="last")
    out["party"] = out["party"].fillna("Unknown")
    return out


def load_speech_classifications(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    req = {"speech_id", "category", "normalized_weight"}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in speech classifications: {sorted(missing)}")
    df = df[["speech_id", "category", "normalized_weight"]].copy()
    df["speech_id"] = df["speech_id"].astype(str)
    df["category"] = df["category"].astype(str)
    df["normalized_weight"] = pd.to_numeric(df["normalized_weight"], errors="coerce").fillna(0.0)
    return df


def build_speech_party_profiles(
    speech_classifications_path: str | Path,
    speech_parquet_dir: str | Path,
) -> pd.DataFrame:
    cls = load_speech_classifications(speech_classifications_path)
    meta = load_speech_metadata(speech_parquet_dir)
    merged = cls.merge(meta, on="speech_id", how="left")
    merged["party"] = merged["party"].fillna("Unknown")

    # normalize and filter out unknown/NYD parties for visualisations
    merged["party"] = merged["party"].astype(str).str.strip()
    merged = merged[~merged["party"].str.lower().isin({"unknown", "nyd", ""})].copy()

    grp = (
        merged.groupby(["party", "category"], as_index=False)["normalized_weight"].mean()
        .rename(columns={"normalized_weight": "weight"})
    )

    total = grp.groupby("party", as_index=False)["weight"].sum().rename(columns={"weight": "party_sum"})
    out = grp.merge(total, on="party", how="left")
    out["proportion"] = np.where(out["party_sum"] > 0, out["weight"] / out["party_sum"], 0.0)
    return out[["party", "category", "proportion"]]


def _profiles_to_matrix(profiles: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    parties = sorted(profiles["party"].unique().tolist())
    cats = IDEOLOGY_ORDER
    pivot = (
        profiles.pivot_table(index="party", columns="category", values="proportion", aggfunc="mean", fill_value=0.0)
        .reindex(index=parties, columns=cats, fill_value=0.0)
    )
    return parties, pivot.to_numpy(dtype=float)


def _ideology_score(v: np.ndarray) -> float:
    idx = np.arange(len(IDEOLOGY_ORDER), dtype=float)
    d = v.sum()
    if d <= 0:
        return 0.5
    return float((v @ idx) / d / max(1, len(IDEOLOGY_ORDER) - 1))


def plot_speech_profiles(
    speech_classifications_path: str | Path,
    speech_parquet_dir: str | Path,
    out_dir: str | Path = "figures/speeches",
    basename: str = "speech_profiles",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = build_speech_party_profiles(speech_classifications_path, speech_parquet_dir)
    parties, mat = _profiles_to_matrix(profiles)

    # 1) stacked bars
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(parties))))
    cmap = plt.get_cmap("RdYlBu")
    colors = [cmap(i / max(1, len(IDEOLOGY_ORDER) - 1)) for i in range(len(IDEOLOGY_ORDER))]
    y = np.arange(len(parties))
    left = np.zeros(len(parties))
    for i, cat in enumerate(IDEOLOGY_ORDER):
        vals = mat[:, i]
        ax.barh(y, vals, left=left, color=colors[i], edgecolor="white", label=cat)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(parties)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion")
    ax.set_title("Speech ideology profiles by party")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    stacked_png = out_dir / f"{basename}_stacked.png"
    fig.savefig(stacked_png, dpi=300)
    plt.close(fig)

    # 2) ideology + heatmap
    scores = np.array([_ideology_score(v) for v in mat])
    order = np.argsort(scores)
    parties_o = [parties[i] for i in order]
    mat_o = mat[order]
    scores_o = scores[order]

    fig = plt.figure(figsize=(12, max(4, 0.55 * len(parties_o))))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3])
    ax_top = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])
    fig.subplots_adjust(left=0.25, right=0.92, top=0.92, bottom=0.18, hspace=0.35)

    y = np.arange(len(parties_o))
    ax_top.scatter(scores_o, y, c=scores_o, cmap="RdYlBu_r", s=110, edgecolors="black")
    ax_top.set_yticks(y)
    ax_top.set_yticklabels(parties_o)
    ax_top.set_xlim(-0.02, 1.02)
    ax_top.set_xlabel("Ideological placement (Left=0 -> Right=1)")
    ax_top.set_title("Speech ideological placement")
    ax_top.invert_yaxis()

    im = ax_heat.imshow(mat_o, aspect="auto", cmap="YlGnBu", interpolation="nearest", vmin=0, vmax=1)
    ax_heat.set_yticks(y)
    ax_heat.set_yticklabels(parties_o)
    ax_heat.set_xticks(np.arange(len(IDEOLOGY_ORDER)))
    ax_heat.set_xticklabels(IDEOLOGY_ORDER, rotation=45, ha="right")
    ax_heat.set_xlabel("Category")
    ax_heat.set_title("Speech category distribution")
    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical", fraction=0.04, pad=0.02)
    cbar.set_label("Proportion")

    heat_png = out_dir / f"{basename}_heatmap.png"
    fig.savefig(heat_png, dpi=300)
    plt.close(fig)

    # 3) clustered heatmap
    order = list(range(len(parties_o)))
    if len(parties_o) > 1:
        z = linkage(mat_o, method="ward")
        d = dendrogram(z, labels=parties_o, orientation="left", no_plot=True)
        order = d["leaves"]
    parties_c = [parties_o[i] for i in order]
    mat_c = mat_o[order]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(parties_c))))
    im = ax.imshow(mat_c, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(parties_c)))
    ax.set_yticklabels(parties_c)
    ax.set_xticks(np.arange(len(IDEOLOGY_ORDER)))
    ax.set_xticklabels(IDEOLOGY_ORDER, rotation=45, ha="right")
    ax.set_title("Speech clustered ideology heatmap")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    cluster_png = out_dir / f"{basename}_clustered.png"
    fig.savefig(cluster_png, dpi=300)
    plt.close(fig)

    # 4) PCA biplot
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    if len(parties_o) >= 2:
        pca = PCA(n_components=2, random_state=42)
        xy = pca.fit_transform(mat_o)
    else:
        xy = np.zeros((len(parties_o), 2))

    for i, party in enumerate(parties_o):
        ax.scatter(xy[i, 0], xy[i, 1], color="tab:blue", s=70)
        ax.text(xy[i, 0] + 0.01, xy[i, 1] + 0.01, party, fontsize=9)

    for j, cat in enumerate(IDEOLOGY_ORDER):
        if len(parties_o) >= 2:
            vec = pca.components_[:2, j]
        else:
            vec = np.array([0.0, 0.0])
        ax.arrow(0, 0, vec[0], vec[1], color="tab:orange", alpha=0.8, head_width=0.02, length_includes_head=True)
        ax.text(vec[0] * 1.08, vec[1] * 1.08, cat, color="tab:orange", fontsize=9)

    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_title("Speech PCA biplot")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    pca_png = out_dir / f"{basename}_pca.png"
    fig.savefig(pca_png, dpi=300)
    plt.close(fig)

    summary = {
        "stacked": str(stacked_png),
        "heatmap": str(heat_png),
        "clustered": str(cluster_png),
        "pca": str(pca_png),
        "party_count": len(parties),
    }
    (out_dir / f"{basename}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
