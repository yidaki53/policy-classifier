#!/usr/bin/env python3
"""Orchestrate three-way comparison (speeches, motions, votes):
 - build profiles (uses scripts/build_profiles.py logic)
 - run paired tests per party×category
 - apply FDR correction
 - output CSV/Parquet and simple figures
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

from swedish_parliament_policy_classifier.analysis.statistical_tests import run_paired_test, apply_bh_correction
from swedish_parliament_policy_classifier.analysis.speech_visualizations import IDEOLOGY_ORDER


def load_profiles(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def pivot_profiles(df: pd.DataFrame) -> pd.DataFrame:
    # returns multi-index (party, modality) × categories
    pivot = df.pivot_table(index=["party", "modality"], columns="category", values="proportion", fill_value=0.0)
    # ensure ideology ordering
    for c in IDEOLOGY_ORDER:
        if c not in pivot.columns:
            pivot[c] = 0.0
    return pivot[IDEOLOGY_ORDER].reset_index()


def run_tests_on_profiles(pivot_df: pd.DataFrame) -> pd.DataFrame:
    parties = sorted(set(pivot_df["party"]))
    rows = []
    for party in parties:
        speech = pivot_df[(pivot_df["party"] == party) & (pivot_df["modality"] == "speech")]
        motion = pivot_df[(pivot_df["party"] == party) & (pivot_df["modality"] == "motion")]
        vote = pivot_df[(pivot_df["party"] == party) & (pivot_df["modality"] == "vote")]
        if speech.empty or motion.empty:
            continue
        svec = speech[IDEOLOGY_ORDER].to_numpy().ravel()
        mvec = motion[IDEOLOGY_ORDER].to_numpy().ravel()
        vvec = vote[IDEOLOGY_ORDER].to_numpy().ravel() if not vote.empty else np.zeros_like(svec)

        # per-category paired tests (speech vs combined(motion+vote))
        combined = (mvec + vvec) / 2.0
        for i, cat in enumerate(IDEOLOGY_ORDER):
            a = np.array([svec[i]])
            b = np.array([combined[i]])
            res = run_paired_test(a, b)
            rows.append({"party": party, "category": cat, **res})

    out = pd.DataFrame(rows)
    return out


def is_ja(val: str) -> bool:
    if not isinstance(val, str):
        return False
    return val.strip().lower() in {"ja", "j", "1", "yes", "y", "för", "for", "for"}


def load_motion_votes_map(parquet_path: str) -> dict:
    if not os.path.exists(parquet_path):
        return {}
    mv = pd.read_parquet(parquet_path, columns=["motion_id", "first_votering_id"]).fillna("")
    mv = mv[mv["first_votering_id"] != ""].copy()
    return dict(zip(mv["motion_id"].astype(str), mv["first_votering_id"].astype(str)))


def load_votes_for_votering_ids(votering_dir: str, votering_ids: set) -> dict:
    # returns mapping (votering_id, party) -> rost
    mapping = {}
    pattern = os.path.join(votering_dir, "*.parquet")
    for p in sorted(glob.glob(pattern)):
        try:
            df = pd.read_parquet(p, columns=["votering_id", "parti", "rost"]).copy()
        except Exception:
            continue
        df = df[df["votering_id"].isin(votering_ids)]
        if df.empty:
            continue
        for _, r in df.iterrows():
            mapping[(str(r["votering_id"]), str(r["parti"]))] = str(r["rost"])
    return mapping


def run_linked_pair_tests(speech_motions_path: str, motion_votes_parquet: str, votering_dir: str) -> pd.DataFrame:
    if not os.path.exists(speech_motions_path):
        return pd.DataFrame()
    sm = pd.read_parquet(speech_motions_path)
    if sm.empty:
        return pd.DataFrame()

    # build list of motion_ids to fetch votering ids for
    motion_ids = sm["motion_id"].astype(str).unique().tolist()
    motion_to_votering = load_motion_votes_map(motion_votes_parquet)
    votering_ids_needed = set([motion_to_votering.get(mid) for mid in motion_ids if motion_to_votering.get(mid)])
    votering_ids_needed = set([v for v in votering_ids_needed if v])

    votes_map = load_votes_for_votering_ids(votering_dir, votering_ids_needed) if votering_ids_needed else {}

    rows = []
    # group by party+category
    grouped = sm.groupby(["party", "category"])
    for (party, cat), g in grouped:
        speech_vals = g["speech_weight"].astype(float).to_numpy()
        motion_vals = g["motion_weight"].astype(float).to_numpy()
        # compute vote binary per row using votering map
        vote_vals = []
        for mid in g["motion_id"].astype(str):
            vid = motion_to_votering.get(mid)
            rost = votes_map.get((vid, party)) if vid else None
            vote_vals.append(1.0 if is_ja(rost) else 0.0)
        vote_vals = np.array(vote_vals, dtype=float)

        # combined metric
        combined = (motion_vals + vote_vals) / 2.0

        # paired tests
        res_sm = run_paired_test(speech_vals, motion_vals)
        res_sv = run_paired_test(speech_vals, vote_vals)
        res_sc = run_paired_test(speech_vals, combined)

        rows.append({"party": party, "category": cat, "comparison": "speech_vs_motion", **res_sm})
        rows.append({"party": party, "category": cat, "comparison": "speech_vs_vote", **res_sv})
        rows.append({"party": party, "category": cat, "comparison": "speech_vs_combined", **res_sc})

    out = pd.DataFrame(rows)
    return out


def annotate_and_correct(tests_df: pd.DataFrame) -> pd.DataFrame:
    pvals = tests_df["pvalue"].fillna(1.0).tolist()
    adjusted = apply_bh_correction(pvals)
    tests_df = tests_df.copy()
    tests_df["p_adj"] = adjusted
    tests_df["signif"] = tests_df["p_adj"].apply(lambda p: "ns" if p >= 0.05 else ("*" if p<0.05 else ""))
    return tests_df


def plot_divergence_matrix(pivot_df: pd.DataFrame, out_dir: str):
    parties = sorted(set(pivot_df["party"]))
    cats = IDEOLOGY_ORDER
    mat = []
    for p in parties:
        speech = pivot_df[(pivot_df["party"] == p) & (pivot_df["modality"] == "speech")]
        motion = pivot_df[(pivot_df["party"] == p) & (pivot_df["modality"] == "motion")]
        vote = pivot_df[(pivot_df["party"] == p) & (pivot_df["modality"] == "vote")]
        if speech.empty:
            mat.append([0.0] * len(cats))
            continue
        svec = speech[cats].to_numpy().ravel()
        mvec = motion[cats].to_numpy().ravel() if not motion.empty else np.zeros_like(svec)
        vvec = vote[cats].to_numpy().ravel() if not vote.empty else np.zeros_like(svec)
        combined = (mvec + vvec) / 2.0
        mat.append(np.abs(svec - combined))

    mat = np.vstack(mat)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(parties))))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_yticks(range(len(parties)))
    ax.set_yticklabels(parties)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.03)
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, "divergence_heatmap.png")
    fig.savefig(outp, dpi=200)
    plt.close(fig)
    return outp


def main():
    parser = argparse.ArgumentParser(description="Three-way speeches vs motions vs votes analysis")
    parser.add_argument("--profiles", default="data/parquet/party_profiles_recency.parquet")
    parser.add_argument("--out-dir", default="figures/three_way")
    parser.add_argument("--speech-motions", default="data/parquet/speech_motions.parquet")
    parser.add_argument("--motion-votes", default="data/parquet/motion_votes.parquet")
    parser.add_argument("--votering-dir", default="data/votering/parquet")
    args = parser.parse_args()

    profiles = load_profiles(args.profiles)
    pivot = pivot_profiles(profiles)
    tests = run_tests_on_profiles(pivot)
    tests = annotate_and_correct(tests)

    os.makedirs(args.out_dir, exist_ok=True)
    tests.to_parquet(os.path.join(args.out_dir, "paired_tests.parquet"), index=False)
    tests.to_csv(os.path.join(args.out_dir, "paired_tests.csv"), index=False)
    hm = plot_divergence_matrix(pivot, args.out_dir)
    print(json.dumps({"paired_tests": os.path.join(args.out_dir, "paired_tests.parquet"), "heatmap": hm}))

    # If speech->motion links exist, run linked-pair tests and produce effect-size tables + annotated heatmaps
    linked_tests = run_linked_pair_tests(args.speech_motions, args.motion_votes, args.votering_dir)
    if not linked_tests.empty:
        # apply BH per comparison group
        out_rows = []
        for comp, group in linked_tests.groupby("comparison"):
            pvals = group["pvalue"].fillna(1.0).tolist()
            adj = apply_bh_correction(pvals)
            group = group.copy()
            group["p_adj"] = adj
            # significance stars
            def star(p):
                if pd.isna(p):
                    return ""
                if p < 0.001:
                    return "***"
                if p < 0.01:
                    return "**"
                if p < 0.05:
                    return "*"
                return "ns"

            group["signif"] = group["p_adj"].apply(star)
            out_rows.append(group)

        eff_df = pd.concat(out_rows, ignore_index=True)
        eff_out_parquet = os.path.join(args.out_dir, "effect_size_table.parquet")
        eff_out_csv = os.path.join(args.out_dir, "effect_size_table.csv")
        eff_df.to_parquet(eff_out_parquet, index=False)
        eff_df.to_csv(eff_out_csv, index=False)

        # annotated heatmap for speech_vs_combined
        def plot_annotated_significance(pivot_df: pd.DataFrame, tests_df: pd.DataFrame, comparison: str, out_dir: str):
            parties = sorted(set(pivot_df["party"]))
            cats = IDEOLOGY_ORDER
            mat = []
            stars = []
            for p in parties:
                speech = pivot_df[(pivot_df["party"] == p) & (pivot_df["modality"] == "speech")]
                motion = pivot_df[(pivot_df["party"] == p) & (pivot_df["modality"] == "motion")]
                vote = pivot_df[(pivot_df["party"] == p) & (pivot_df["modality"] == "vote")]
                if speech.empty:
                    mat.append([0.0] * len(cats))
                    stars.append([""] * len(cats))
                    continue
                svec = speech[cats].to_numpy().ravel()
                mvec = motion[cats].to_numpy().ravel() if not motion.empty else np.zeros_like(svec)
                vvec = vote[cats].to_numpy().ravel() if not vote.empty else np.zeros_like(svec)
                combined = (mvec + vvec) / 2.0
                mat.append(np.abs(svec - combined))

                row_stars = []
                for cat in cats:
                    sel = tests_df[(tests_df["party"] == p) & (tests_df["category"] == cat) & (tests_df["comparison"] == comparison)]
                    if sel.empty:
                        row_stars.append("")
                    else:
                        val = sel.iloc[0].get("p_adj", None)
                        if pd.isna(val):
                            row_stars.append("")
                        elif val < 0.001:
                            row_stars.append("***")
                        elif val < 0.01:
                            row_stars.append("**")
                        elif val < 0.05:
                            row_stars.append("*")
                        else:
                            row_stars.append("ns")
                stars.append(row_stars)

            mat = np.vstack(mat)
            fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(parties))))
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
            ax.set_yticks(range(len(parties)))
            ax.set_yticklabels(parties)
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats, rotation=45, ha="right")
            # overlay stars
            for i in range(len(parties)):
                for j in range(len(cats)):
                    txt = stars[i][j]
                    if txt:
                        ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.03)
            os.makedirs(out_dir, exist_ok=True)
            outp = os.path.join(out_dir, f"divergence_{comparison}_significance.png")
            fig.savefig(outp, dpi=200)
            plt.close(fig)
            return outp

        try:
            sig_png = plot_annotated_significance(pivot, eff_df, "speech_vs_combined", args.out_dir)
            print(json.dumps({"effect_table": eff_out_parquet, "annotated_heatmap": sig_png}))
        except Exception:
            print("Linked tests present but failed to plot annotated heatmap.")


if __name__ == "__main__":
    main()
