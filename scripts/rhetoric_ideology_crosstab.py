#!/usr/bin/env python3
"""Produce rhetoric × ideology crosstabs and figures.

Reads:
 - `data/parquet/speech_rhetoric_labels.parquet` (speech-level rhetoric predictions)
 - `data/parquet/speech_classifications_with_rhetoric_full.parquet` (per-speech long-form category probs)
 - speech parquet files under `data/speeches/parquet` (to obtain `parti`/`date` metadata)

Writes:
 - `data/parquet/rhetoric_ideology_crosstab.parquet` (counts + proportions)
 - `figures/rhetoric/rhetoric_ideology_heatmaps.png`
 - `figures/rhetoric/rhetoric_ideology_tables.csv`
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from swedish_parliament_policy_classifier.analysis.speech_visualizations import (
    load_speech_metadata,
    IDEOLOGY_ORDER,
)


def top_category_from_long(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["speech_id", "category", "normalized_weight"]].copy()
    df["speech_id"] = df["speech_id"].astype(str)
    df["normalized_weight"] = pd.to_numeric(df["normalized_weight"], errors="coerce").fillna(0.0)
    if df.empty:
        return pd.DataFrame(columns=["speech_id", "top_category", "top_weight"])
    idx = df.groupby("speech_id")["normalized_weight"].idxmax()
    top = df.loc[idx, ["speech_id", "category", "normalized_weight"]].copy()
    top = top.rename(columns={"category": "top_category", "normalized_weight": "top_weight"})
    return top


def build_crosstabs(speech_classifications_path: str, rhetoric_parquet: str, speech_parquet_dir: str) -> pd.DataFrame:
    print("Loading rhetoric labels:", rhetoric_parquet)
    rhet = pd.read_parquet(rhetoric_parquet)
    if "speech_id" not in rhet.columns:
        raise ValueError("Rhetoric parquet missing 'speech_id' column")
    rhet["speech_id"] = rhet["speech_id"].astype(str)
    # single-label top label (kept for compatibility)
    rhet["top_label"] = rhet.get("top_label", None).fillna("none")

    print("Loading speech classifications (long) -> determining top category:", speech_classifications_path)
    cls = pd.read_parquet(speech_classifications_path)
    top = top_category_from_long(cls)

    print("Loading speech metadata from:", speech_parquet_dir)
    meta = load_speech_metadata(speech_parquet_dir)

    merged = top.merge(rhet, on="speech_id", how="left")
    merged = merged.merge(meta, on="speech_id", how="left")
    merged["party"] = merged.get("party", pd.Series()).fillna("Unknown")
    # normalize and drop unknown / NYD parties for visual outputs
    merged["party"] = merged["party"].astype(str).str.strip()
    merged = merged[~merged["party"].str.lower().isin({"unknown", "nyd", ""})].copy()
    merged["top_label"] = merged["top_label"].fillna("none")

    # 1) Single-label crosstab (top_label)
    ct_single = (
        merged.groupby(["party", "top_category", "top_label"]).size().reset_index().rename(columns={0: "count"})
    )
    if ct_single.empty:
        ct_single = pd.DataFrame()
    else:
        totals = ct_single.groupby(["party", "top_category"], as_index=False)["count"].sum().rename(columns={"count": "total"})
        ct_single = ct_single.merge(totals, on=["party", "top_category"], how="left")
        ct_single["prop"] = ct_single["count"] / ct_single["total"].replace({0: 1})

    # 2) Multi-label crosstab using pred_* binary columns (counts = number of speeches with label)
    # expected pred columns: pred_irony, pred_sarcasm, pred_posturing, pred_none
    pred_cols = [c for c in ["pred_irony", "pred_sarcasm", "pred_posturing", "pred_none"] if c in rhet.columns]
    if pred_cols:
        ml = merged.copy()
        # ensure binary integer
        for c in pred_cols:
            ml[c] = ml[c].fillna(0).astype(int)
        # total unique speeches per party+category
        total_speeches = ml.groupby(["party", "top_category"])['speech_id'].nunique().reset_index().rename(columns={'speech_id':'n_speeches'})
        # sum predicted flags per label
        sums = ml.groupby(["party", "top_category"] + []).agg({c: 'sum' for c in pred_cols}).reset_index()
        # melt to long form
        ml_long = sums.melt(id_vars=["party", "top_category"], value_vars=pred_cols, var_name="pred_col", value_name="count")
        ml_long["top_label"] = ml_long["pred_col"].str.replace("pred_", "")
        ml_long = ml_long.merge(total_speeches, on=["party", "top_category"], how="left")
        ml_long["prop"] = ml_long["count"] / ml_long["n_speeches"].replace({0: 1})
    else:
        ml_long = pd.DataFrame()

    # return both single-label and multi-label tables as a dict of DataFrames
    return {"single_label": ct_single, "multi_label": ml_long}


def save_outputs(cts: dict, out_parquet: str, out_dir: str):
    os.makedirs(Path(out_dir), exist_ok=True)
    results = {}
    # single-label
    single = cts.get("single_label") if isinstance(cts, dict) else None
    if single is not None and not single.empty:
        out_df = single.copy()
        single_parquet = out_parquet.replace('.parquet', '.single.parquet')
        out_df.to_parquet(single_parquet, index=False)
        csv_out = os.path.join(out_dir, "rhetoric_ideology_tables_single.csv")
        out_df.to_csv(csv_out, index=False)
        results.update({"single_parquet": single_parquet, "single_csv": csv_out})

    # multi-label
    multi = cts.get("multi_label") if isinstance(cts, dict) else None
    if multi is not None and not multi.empty:
        multi_parquet = out_parquet.replace('.parquet', '.multilabel.parquet')
        multi.to_parquet(multi_parquet, index=False)
        csv_out2 = os.path.join(out_dir, "rhetoric_ideology_tables_multilabel.csv")
        multi.to_csv(csv_out2, index=False)
        results.update({"multi_parquet": multi_parquet, "multi_csv": csv_out2})

    # produce heatmaps for multi-label proportions (one panel per label)
    labels = sorted(set(multi["top_label"].tolist())) if multi is not None and not multi.empty else []
    parties = sorted(set(single["party"].tolist())) if single is not None and not single.empty else []
    cats = IDEOLOGY_ORDER

    if labels:
        nlabels = max(1, len(labels))
        ncols = 2
        nrows = (nlabels + 1) // ncols
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, max(4, 0.35 * len(parties)) * nrows))
        axs = np.array(axs).reshape(-1)

        for i, lbl in enumerate(labels):
            ax = axs[i]
            sub = multi[multi["top_label"] == lbl]
            pivot = sub.pivot_table(index="party", columns="top_category", values="prop", fill_value=0)
            pivot = pivot.reindex(index=parties, columns=cats, fill_value=0)
            mat = pivot.to_numpy(dtype=float)
            im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
            ax.set_yticks(range(len(parties)))
            ax.set_yticklabels(parties)
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats, rotation=45, ha="right")
            ax.set_title(lbl)
            fig.colorbar(im, ax=ax, fraction=0.03)

        for j in range(i + 1, len(axs)):
            try:
                axs[j].axis("off")
            except Exception:
                pass

        out_png = os.path.join(out_dir, "rhetoric_ideology_heatmaps.png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        results.update({"heatmap": out_png})

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--speech-classifications", default="data/parquet/speech_classifications_with_rhetoric_full.parquet")
    p.add_argument("--rhetoric-parquet", default="data/parquet/speech_rhetoric_labels.parquet")
    p.add_argument("--speech-parquet-dir", default="data/speeches/parquet")
    p.add_argument("--out-parquet", default="data/parquet/rhetoric_ideology_crosstab.parquet")
    p.add_argument("--out-dir", default="figures/rhetoric")
    args = p.parse_args()

    ct = build_crosstabs(args.speech_classifications, args.rhetoric_parquet, args.speech_parquet_dir)
    out = save_outputs(ct, args.out_parquet, args.out_dir)
    print(out)


if __name__ == "__main__":
    main()
