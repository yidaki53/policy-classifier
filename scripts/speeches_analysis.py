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


DEFAULT_EXCLUDED_PARTIES = {"Unknown", "Moderaterna", "Vänsterpartiet", "X"}
VOTE_YES_VALUES = {"ja", "j", "1", "yes", "y", "för", "for"}
VOTE_NO_VALUES = {"nej", "no", "n", "0", "against", "emot"}


def _parse_excluded_parties(raw: str | None) -> set[str]:
    if not raw:
        return set(DEFAULT_EXCLUDED_PARTIES)
    return {p.strip() for p in raw.split(",") if p.strip()}


def _filter_excluded_parties(df: pd.DataFrame, excluded: set[str], party_col: str = "party") -> pd.DataFrame:
    if party_col not in df.columns:
        return df
    out = df.copy()
    out[party_col] = out[party_col].fillna("Unknown").astype(str)
    if not excluded:
        return out
    return out[~out[party_col].isin(excluded)].copy()


def _finite_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


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
    rows = []
    speech = pivot_df[pivot_df["modality"] == "speech"].copy()
    motion = pivot_df[pivot_df["modality"] == "motion"].copy()
    vote = pivot_df[pivot_df["modality"] == "vote"].copy()

    for cat in IDEOLOGY_ORDER:
        merged = speech[["party", cat]].rename(columns={cat: "speech"})
        merged = merged.merge(motion[["party", cat]].rename(columns={cat: "motion"}), on="party", how="inner")
        merged = merged.merge(vote[["party", cat]].rename(columns={cat: "vote"}), on="party", how="left")
        if merged.empty:
            continue

        merged["combined"] = (pd.to_numeric(merged["motion"], errors="coerce") + pd.to_numeric(merged["vote"], errors="coerce")) / 2.0
        speech_vals, motion_vals = _finite_pairs(
            pd.to_numeric(merged["speech"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(merged["motion"], errors="coerce").to_numpy(dtype=float),
        )
        speech_vote_vals, vote_vals = _finite_pairs(
            pd.to_numeric(merged["speech"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(merged["vote"], errors="coerce").to_numpy(dtype=float),
        )
        speech_combined_vals, combined_vals = _finite_pairs(
            pd.to_numeric(merged["speech"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(merged["combined"], errors="coerce").to_numpy(dtype=float),
        )

        rows.append({"party": "ALL_PARTIES", "category": cat, "comparison": "speech_vs_motion", **run_paired_test(speech_vals, motion_vals)})
        rows.append({"party": "ALL_PARTIES", "category": cat, "comparison": "speech_vs_vote", **run_paired_test(speech_vote_vals, vote_vals)})
        rows.append({"party": "ALL_PARTIES", "category": cat, "comparison": "speech_vs_combined", **run_paired_test(speech_combined_vals, combined_vals)})

    out = pd.DataFrame(rows)
    return out


def vote_signal(val: str) -> float:
    if not isinstance(val, str):
        return float("nan")
    norm = val.strip().lower()
    if norm in VOTE_YES_VALUES:
        return 1.0
    if norm in VOTE_NO_VALUES:
        return 0.0
    return float("nan")


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


def run_linked_pair_tests(
    speech_motions_path: str,
    motion_votes_parquet: str,
    votering_dir: str,
    excluded_parties: set[str] | None = None,
) -> pd.DataFrame:
    if not os.path.exists(speech_motions_path):
        return pd.DataFrame()
    sm = pd.read_parquet(speech_motions_path)
    sm = _filter_excluded_parties(sm, excluded_parties or set(), party_col="party")
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
            vote_vals.append(vote_signal(rost))
        vote_vals = np.array(vote_vals, dtype=float)

        # combined metric
        combined = np.where(np.isfinite(motion_vals) & np.isfinite(vote_vals), (motion_vals + vote_vals) / 2.0, np.nan)

        # paired tests
        res_sm = run_paired_test(*_finite_pairs(speech_vals, motion_vals))
        res_sv = run_paired_test(*_finite_pairs(speech_vals, vote_vals))
        res_sc = run_paired_test(*_finite_pairs(speech_vals, combined))

        rows.append({"party": party, "category": cat, "comparison": "speech_vs_motion", **res_sm})
        rows.append({"party": party, "category": cat, "comparison": "speech_vs_vote", **res_sv})
        rows.append({"party": party, "category": cat, "comparison": "speech_vs_combined", **res_sc})

    out = pd.DataFrame(rows)
    return out


def annotate_and_correct(tests_df: pd.DataFrame) -> pd.DataFrame:
    if tests_df.empty or "pvalue" not in tests_df.columns:
        tests_df = tests_df.copy()
        tests_df["p_adj"] = []
        tests_df["signif"] = []
        return tests_df
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
    mat = np.where(np.isfinite(mat), mat, np.nan)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.72 * len(parties))))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(range(len(parties)))
    ax.set_yticklabels(parties, fontsize=10)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("Speech vs combined divergence by party and category", fontsize=12)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, "divergence_heatmap.png")
    fig.savefig(outp, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return outp


def main():
    parser = argparse.ArgumentParser(description="Three-way speeches vs motions vs votes analysis")
    parser.add_argument("--profiles", default="data/parquet/party_profiles_recency.parquet")
    parser.add_argument("--out-dir", default="figures/three_way")
    parser.add_argument("--speech-motions", default="data/parquet/speech_motions.parquet")
    parser.add_argument("--motion-votes", default="data/parquet/motion_votes.parquet")
    parser.add_argument("--votering-dir", default="data/votering/parquet")
    parser.add_argument(
        "--exclude-parties",
        default=",".join(sorted(DEFAULT_EXCLUDED_PARTIES)),
        help="Comma-separated party labels to exclude from divergence outputs.",
    )
    args = parser.parse_args()

    excluded_parties = _parse_excluded_parties(args.exclude_parties)
    profiles = _filter_excluded_parties(load_profiles(args.profiles), excluded_parties)
    pivot = pivot_profiles(profiles)
    tests = run_tests_on_profiles(pivot)
    tests = annotate_and_correct(tests)

    os.makedirs(args.out_dir, exist_ok=True)
    tests.to_parquet(os.path.join(args.out_dir, "paired_tests.parquet"), index=False, compression="zstd")
    tests.to_csv(os.path.join(args.out_dir, "paired_tests.csv"), index=False)
    hm = plot_divergence_matrix(pivot, args.out_dir)
    print(json.dumps({"paired_tests": os.path.join(args.out_dir, "paired_tests.parquet"), "heatmap": hm}))

    # If speech->motion links exist, run linked-pair tests and produce effect-size tables + annotated heatmaps
    linked_tests = run_linked_pair_tests(
        args.speech_motions,
        args.motion_votes,
        args.votering_dir,
        excluded_parties=excluded_parties,
    )
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
        eff_df.to_parquet(eff_out_parquet, index=False, compression="zstd")
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
            fig, ax = plt.subplots(figsize=(12, max(6, 0.72 * len(parties))))
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
            ax.set_yticks(range(len(parties)))
            ax.set_yticklabels(parties, fontsize=10)
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=10)
            # overlay stars
            for i in range(len(parties)):
                for j in range(len(cats)):
                    txt = stars[i][j]
                    if txt:
                        ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.03)
            ax.set_title("Speech vs combined divergence with significance markers", fontsize=12)
            fig.tight_layout()
            os.makedirs(out_dir, exist_ok=True)
            outp = os.path.join(out_dir, f"divergence_{comparison}_significance.png")
            fig.savefig(outp, dpi=300, bbox_inches="tight", pad_inches=0.06)
            plt.close(fig)
            return outp

        try:
            sig_png = plot_annotated_significance(pivot, eff_df, "speech_vs_combined", args.out_dir)
            print(json.dumps({"effect_table": eff_out_parquet, "annotated_heatmap": sig_png}))
        except Exception:
            print("Linked tests present but failed to plot annotated heatmap.")


if __name__ == "__main__":
    main()
