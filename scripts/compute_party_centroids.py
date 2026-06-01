#!/usr/bin/env python3
"""Compute party-level 'says' (speeches) vs 'does' (motions) centroids.

Produces CSV summaries and a 2D projection plot (UMAP if available, PCA fallback).

Usage:
    uv run python3 scripts/compute_party_centroids.py --db data/swedish_parliament.db
"""

from pathlib import Path
import argparse
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt


def load_speech_probs(conn: sqlite3.Connection) -> pd.DataFrame:
    q = "SELECT speech_id, category, normalized_weight FROM speech_classifications"
    df = pd.read_sql_query(q, conn)
    if df.empty:
        return df
    pivot = df.pivot_table(index="speech_id", columns="category", values="normalized_weight", aggfunc="sum", fill_value=0.0)
    pivot.reset_index(inplace=True)
    return pivot


def load_motion_probs_with_party(conn: sqlite3.Connection) -> pd.DataFrame:
    q = """
    SELECT c.motion_id as motion_id, nm.party as party, c.category as category, c.normalized_weight as normalized_weight
    FROM classifications c
    JOIN normalized_motions nm ON nm.id = c.motion_id
    """
    df = pd.read_sql_query(q, conn)
    if df.empty:
        return df
    pivot = df.pivot_table(index=["motion_id", "party"], columns="category", values="normalized_weight", aggfunc="sum", fill_value=0.0)
    pivot.reset_index(inplace=True)
    return pivot


def load_speech_party_map(parquet_dir: Path) -> pd.DataFrame:
    parts = []
    for p in sorted(parquet_dir.glob("*.parquet")):
        try:
            parts.append(pd.read_parquet(p))
        except Exception:
            continue
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    return df[["anforande_id", "parti"]].rename(columns={"anforande_id": "speech_id", "parti": "party"})


def compute_party_centroids(db_path: str = "data/swedish_parliament.db") -> Path:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Speech-level probabilities
    speech_probs = load_speech_probs(conn)
    # Map speech -> party via parquet
    parquet_dir = Path("data/speeches/parquet")
    speech_party = load_speech_party_map(parquet_dir)
    if speech_party.empty:
        raise RuntimeError("No speech parquet files found to map speech->party")

    if speech_probs.empty:
        raise RuntimeError("No speech_classifications found in DB; run scripts/classify_speeches.py first")

    sp = speech_probs.merge(speech_party, on="speech_id", how="left")
    sp = sp.dropna(subset=[c for c in sp.columns if c not in ["speech_id", "party"]], how="all")

    # Per-party says centroid: mean over speeches
    cat_cols = [c for c in sp.columns if c not in ["speech_id", "party"]]
    says = sp.groupby("party")[cat_cols].mean().reset_index()

    # Motion-level probs with party
    motion_probs = load_motion_probs_with_party(conn)
    if motion_probs.empty:
        raise RuntimeError("No motion classifications found in DB")
    does = motion_probs.groupby("party")[ [c for c in motion_probs.columns if c not in ["motion_id","party"]] ].mean().reset_index()

    # Align columns
    all_cats = sorted(set(cat_cols) | set([c for c in does.columns if c not in ["party"]]))
    says = says.set_index("party").reindex(columns=all_cats, fill_value=0.0)
    does = does.set_index("party").reindex(columns=all_cats, fill_value=0.0)

    # Compute distances and save CSV
    from sklearn.metrics.pairwise import cosine_distances

    parties = sorted(set(says.index) | set(does.index))
    records = []
    for p in parties:
        v_s = says.loc[p].values if p in says.index else np.zeros(len(all_cats))
        v_d = does.loc[p].values if p in does.index else np.zeros(len(all_cats))
        cos_dist = float(cosine_distances(v_s.reshape(1, -1), v_d.reshape(1, -1))[0, 0])
        euclid = float(np.linalg.norm(v_s - v_d))
        records.append({"party": p, "cosine_distance": cos_dist, "euclidean_distance": euclid})

    meta_df = pd.DataFrame(records).sort_values("cosine_distance", ascending=False)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / f"party_centroids_says_vs_does_{ts}.parquet"
    meta_df.to_parquet(out_csv, index=False, compression='zstd')

    # Build combined matrix for projection: each party has two rows (says/does)
    rows = []
    for p in parties:
        v_s = says.loc[p].values if p in says.index else np.zeros(len(all_cats))
        v_d = does.loc[p].values if p in does.index else np.zeros(len(all_cats))
        rows.append({"party": p, "type": "says", **{c: float(v) for c, v in zip(all_cats, v_s)}})
        rows.append({"party": p, "type": "does", **{c: float(v) for c, v in zip(all_cats, v_d)}})
    comb = pd.DataFrame(rows)

    feat = comb[all_cats].values
    # Try UMAP, else PCA
    try:
        import umap
        reducer = umap.UMAP(random_state=42)
        proj = reducer.fit_transform(feat)
    except Exception:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        proj = reducer.fit_transform(feat)

    comb[["x", "y"]] = proj

    # KMeans clustering into 3 groups
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42).fit(feat)
        comb["cluster"] = kmeans.labels_
    except Exception:
        comb["cluster"] = 0

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    types = {"says": "o", "does": "s"}
    for t, marker in types.items():
        subset = comb[comb["type"] == t]
        ax.scatter(subset["x"], subset["y"], label=t, marker=marker, s=120)
        for _, r in subset.iterrows():
            ax.text(r["x"] + 0.01, r["y"] + 0.01, r["party"], fontsize=8)

    ax.set_title("Party centroids: says (speeches) vs does (motions)")
    ax.legend()
    plot_path = out_dir / f"party_centroids_projection_{ts}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # Save combined centroids and per-party vectors
    comb_out = out_dir / f"party_centroids_vectors_{ts}.parquet"
    comb.to_parquet(comb_out, index=False, compression='zstd')

    return out_csv, comb_out, plot_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/swedish_parliament.db")
    args = p.parse_args()
    out = compute_party_centroids(db_path=args.db)
    print("Wrote centroids metadata:", out[0])
    print("Wrote combined vectors:", out[1])
    print("Wrote projection plot:", out[2])


if __name__ == "__main__":
    main()
