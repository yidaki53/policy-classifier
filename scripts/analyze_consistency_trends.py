#!/usr/bin/env python3
"""Build integrated consistency and directional trend analyses.

Outputs (all parquet with zstd compression):
- output/analysis/consistency_score_party.parquet
- output/analysis/lead_lag_speech_to_action_party_year.parquet
- output/analysis/parliament_direction_over_time.parquet
- output/analysis/consistency_trends_summary.json

Figures:
- output/manuscript/figures/figure_consistency_vs_fulfillment.png
- output/manuscript/figures/figure_parliament_direction_over_time.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from swedish_parliament_policy_classifier.io.parquet_guard import ensure_min_rows_many
from swedish_parliament_policy_classifier.runtime.resources import apply_cpu_throttle, thermal_safe_defaults
from swedish_parliament_policy_classifier.runtime.experiment import ExperimentRun
from swedish_parliament_policy_classifier.visualization.style_config import add_figure_credits


IDEOLOGY_ORDER = [
    "far_left",
    "left",
    "centre_left",
    "centre",
    "centre_right",
    "right",
    "far_right",
]

# CHES trend file (1999-2024) direct CSV link from CHES Europe datasets page.
CHES_TREND_CSV_URL = "https://www.chesdata.eu/s/1999-2024_CHES_dataset_meansV2-3k4l.csv"
CHES_SWEDEN_COUNTRY_CODE = 16
CHES_SWEDEN_PARTY_MAP = {
    "C": "C",
    "KD": "KD",
    "L": "L",
    "FP": "L",  # Legacy Folkpartiet code in pre-2019 waves.
    "M": "M",
    "MP": "MP",
    "SD": "SD",
    "V": "V",
    "SAP": "S",
    "S/SAP": "S",
}


def build_consistency_score(
    gap_df: pd.DataFrame,
    pf_df: pd.DataFrame,
    speech_motion_weight: float = 0.5,
    fulfillment_fill_value: float = 0.5,
) -> pd.DataFrame:
    gap_df = gap_df[~gap_df["party"].astype(str).str.lower().isin({"unknown", "nyd", ""})].copy()
    pf_df = pf_df[~pf_df["party"].astype(str).str.lower().isin({"unknown", "nyd", ""})].copy()

    pivot = gap_df.pivot_table(index="party", columns="comparison", values="js_distance", aggfunc="mean")
    pivot = pivot.reset_index()

    for col in ["speech_vs_motion", "speech_vs_vote", "motion_vs_vote"]:
        if col not in pivot.columns:
            pivot[col] = np.nan

    w_sm = float(np.clip(speech_motion_weight, 0.0, 1.0))
    w_sv = float(1.0 - w_sm)
    pivot["consistency_score"] = 1.0 - (w_sm * pivot["speech_vs_motion"] + w_sv * pivot["speech_vs_vote"])
    pivot["action_cohesion_score"] = 1.0 - pivot["motion_vs_vote"]

    out = pivot.merge(pf_df, on="party", how="left")

    # Fairness guard: do not penalize parties for speeches with no available
    # motion pathway. Fulfillment is measured conditionally on speeches that do
    # have a motion pathway.
    vote = pd.to_numeric(out.get("pct_speech_motion_vote"), errors="coerce").fillna(0.0)
    motion_no_vote = pd.to_numeric(out.get("pct_speech_motion_no_vote"), errors="coerce").fillna(0.0)
    denom = vote + motion_no_vote
    out["pct_vote_given_motion_pathway"] = np.where(denom > 0, vote / denom, np.nan)

    fill_val = float(fulfillment_fill_value)
    out["consistency_x_fulfillment"] = out["consistency_score"] * out["pct_vote_given_motion_pathway"].fillna(fill_val)
    return out.sort_values("consistency_score", ascending=False).reset_index(drop=True)


def build_vote_alignment_fulfillment(edge_df: pd.DataFrame) -> pd.DataFrame:
    votes = edge_df.copy()
    votes = votes[votes["action_type"].astype(str) == "vote"].copy()
    votes = votes[~votes["speech_party"].astype(str).str.lower().isin({"unknown", "nyd", ""})].copy()
    if votes.empty:
        return pd.DataFrame(columns=["party", "vote_alignment_fulfillment", "vote_alignment_edges"])

    votes["alignment_score"] = (1.0 - pd.to_numeric(votes["contradiction_score_raw"], errors="coerce").fillna(1.0)).clip(0.0, 1.0)
    votes["edge_confidence_raw"] = pd.to_numeric(votes["edge_confidence_raw"], errors="coerce").fillna(1.0).clip(0.0, 1.0)

    def _weighted_mean(g: pd.DataFrame) -> float:
        w = g["edge_confidence_raw"].to_numpy(dtype=float)
        x = g["alignment_score"].to_numpy(dtype=float)
        den = float(w.sum())
        if den <= 0:
            return float(np.nanmean(x))
        return float((x * w).sum() / den)

    out = (
        votes.groupby("speech_party", as_index=False)
        .apply(lambda g: pd.Series({"vote_alignment_fulfillment": _weighted_mean(g), "vote_alignment_edges": int(len(g))}))
        .reset_index(drop=True)
        .rename(columns={"speech_party": "party"})
    )
    return out


def _build_yearly_modality_indices(topic_year_df: pd.DataFrame) -> pd.DataFrame:
    idx_map = {k: i for i, k in enumerate(IDEOLOGY_ORDER)}
    df = topic_year_df.copy()
    df = df[df["topic"].isin(idx_map)].copy()
    df["idx"] = df["topic"].map(idx_map).astype(float)

    agg = []
    for party, year in df[["party", "year"]].drop_duplicates().itertuples(index=False):
        sub = df[(df["party"] == party) & (df["year"] == year)]

        def _wmean(weight_col: str) -> float:
            w = sub[weight_col].astype(float).to_numpy()
            x = sub["idx"].astype(float).to_numpy()
            den = w.sum()
            if den <= 0:
                return float("nan")
            return float((x * w).sum() / den)

        agg.append(
            {
                "party": party,
                "year": int(year),
                "speech": _wmean("S"),
                "motion": _wmean("M"),
                "vote": _wmean("V"),
            }
        )

    piv = pd.DataFrame(agg)
    piv = piv[~piv["party"].astype(str).str.lower().isin({"unknown", "nyd", ""})].copy()
    piv["action_idx"] = piv[["motion", "vote"]].mean(axis=1)
    return piv


def build_lead_lag(topic_year_df: pd.DataFrame) -> pd.DataFrame:
    piv = _build_yearly_modality_indices(topic_year_df)

    rows = []
    for party, grp in piv.groupby("party"):
        grp = grp.sort_values("year").reset_index(drop=True)
        action_by_year = dict(zip(grp["year"], grp["action_idx"]))
        for _, r in grp.iterrows():
            y = r["year"]
            speech_idx = r["speech"]
            if pd.isna(speech_idx):
                continue
            nxt = action_by_year.get(y + 1)
            if pd.isna(nxt) or nxt is None:
                continue
            rows.append(
                {
                    "party": party,
                    "year": int(y),
                    "speech_idx_t": float(speech_idx),
                    "action_idx_t_plus_1": float(nxt),
                    "lead_shift_t_to_t1": float(nxt - speech_idx),
                    "abs_lead_shift_t_to_t1": float(abs(nxt - speech_idx)),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    med = out.groupby("year", as_index=False)["action_idx_t_plus_1"].median().rename(
        columns={"action_idx_t_plus_1": "parliament_median_action_t_plus_1"}
    )
    out = out.merge(med, on="year", how="left")
    out["action_vs_parliament_median_t_plus_1"] = (
        out["action_idx_t_plus_1"] - out["parliament_median_action_t_plus_1"]
    )
    return out.sort_values(["party", "year"]).reset_index(drop=True)


def build_parliament_direction(topic_year_df: pd.DataFrame) -> pd.DataFrame:
    piv = _build_yearly_modality_indices(topic_year_df)

    year_stats = (
        piv.groupby("year", as_index=False)
        .agg(
            parliament_median_action=("action_idx", "median"),
            parliament_mean_action=("action_idx", "mean"),
            parliament_median_speech=("speech", "median"),
            parliament_mean_speech=("speech", "mean"),
            n_parties=("party", "nunique"),
        )
        .sort_values("year")
    )

    return year_stats.reset_index(drop=True)


def _slope(x: pd.Series, y: pd.Series) -> float:
    m = (~x.isna()) & (~y.isna())
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(x[m].to_numpy(dtype=float), y[m].to_numpy(dtype=float), 1)[0])


def _year_range_label(series: pd.Series) -> str | None:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return None
    lo = int(vals.min())
    hi = int(vals.max())
    return f"{lo}-{hi}" if lo != hi else f"{lo}"


def plot_consistency_vs_fulfillment(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    x = df["consistency_score"].to_numpy(dtype=float)
    y = df["pct_speech_motion_vote"].to_numpy(dtype=float)
    parties = df["party"].astype(str).tolist()

    ax.scatter(x, y, alpha=0.85)
    for i, p in enumerate(parties):
        ax.text(x[i], y[i], p, fontsize=9, ha="left", va="bottom")

    ax.set_xlabel("Consistency score (unitless index)")
    ax.set_ylabel("Promise fulfillment (share, 0-1)")
    ax.set_title(f"Party consistency vs fulfillment (n={len(df)})")
    ax.grid(alpha=0.25)

    add_figure_credits(
        fig,
        n_total=len(df),
        n_parties=df["party"].nunique() if "party" in df.columns else None,
        source="output/analysis/consistency_score_party.parquet",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_parliament_direction(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(df["year"], df["parliament_median_action"], marker="o", label="median action")
    ax.plot(df["year"], df["parliament_median_speech"], marker="o", label="median speech")

    year_range = _year_range_label(df["year"])
    ax.set_xlabel("Year (calendar year)")
    ax.set_ylabel("Ideology index (category scale, unitless)")
    span = f" ({year_range})" if year_range else ""
    ax.set_title(f"Parliament directional movement over time{span}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    add_figure_credits(
        fig,
        n_total=len(df),
        date_range=year_range,
        source="output/analysis/parliament_direction_over_time.parquet",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_party_year_consistency_fulfillment(
    consistency_df: pd.DataFrame,
    topic_year_df: pd.DataFrame,
) -> pd.DataFrame:
    t = topic_year_df.copy()
    t = t[~t["party"].astype(str).str.lower().isin({"unknown", "nyd", ""})].copy()
    t["year"] = pd.to_numeric(t["year"], errors="coerce")
    t["pct_speech_motion_vote"] = pd.to_numeric(t["pct_speech_motion_vote"], errors="coerce")
    yearly = (
        t.groupby(["party", "year"], as_index=False)["pct_speech_motion_vote"]
        .mean()
        .dropna(subset=["year", "pct_speech_motion_vote"])
    )
    c = consistency_df[["party", "consistency_score"]].copy()
    out = yearly.merge(c, on="party", how="left")
    out["consistency_fulfillment_metric"] = out["consistency_score"] * out["pct_speech_motion_vote"]
    out["year"] = out["year"].astype(int)
    return out.sort_values(["party", "year"]).reset_index(drop=True)


def _load_external_party_year_benchmark(analysis_dir: Path, internal_df: pd.DataFrame) -> pd.DataFrame:
    preferred = analysis_dir / "external_party_benchmarks_party_year.csv"
    fallback = analysis_dir / "external_party_benchmarks.csv"

    if preferred.exists():
        ext = pd.read_csv(preferred)
    elif fallback.exists():
        base = pd.read_csv(fallback)
        keys = internal_df[["party", "year"]].drop_duplicates().copy()
        ext = keys.merge(base[["party", "benchmark_ideology_score"]], on="party", how="left")
    else:
        return pd.DataFrame(columns=["party", "year", "benchmark_ideology_score"])

    if "year" not in ext.columns:
        return pd.DataFrame(columns=["party", "year", "benchmark_ideology_score"])

    ext = ext[["party", "year", "benchmark_ideology_score"]].copy()
    ext["year"] = pd.to_numeric(ext["year"], errors="coerce")
    ext["benchmark_ideology_score"] = pd.to_numeric(ext["benchmark_ideology_score"], errors="coerce")
    ext = ext.dropna(subset=["year", "benchmark_ideology_score"])
    ext["year"] = ext["year"].astype(int)
    return ext.sort_values(["party", "year"]).reset_index(drop=True)


def _build_ches_party_year_benchmark(
    internal_df: pd.DataFrame,
    out_path: Path,
    ches_url: str,
    country_code: int,
) -> pd.DataFrame:
    ches = pd.read_csv(ches_url)
    required = {"year", "country", "party", "lrgen"}
    missing = required - set(ches.columns)
    if missing:
        raise ValueError(f"CHES schema missing required columns: {sorted(missing)}")

    sw = ches[ches["country"] == int(country_code)].copy()
    sw["party"] = sw["party"].astype(str).map(CHES_SWEDEN_PARTY_MAP)
    sw = sw.dropna(subset=["party"]).copy()
    sw["year"] = pd.to_numeric(sw["year"], errors="coerce")
    sw["benchmark_ideology_score"] = pd.to_numeric(sw["lrgen"], errors="coerce")
    sw = sw.dropna(subset=["year", "benchmark_ideology_score"])
    sw["year"] = sw["year"].astype(int)

    wave = (
        sw.groupby(["party", "year"], as_index=False)["benchmark_ideology_score"]
        .mean()
        .sort_values(["party", "year"])
    )

    target_keys = internal_df[["party", "year"]].drop_duplicates().copy()
    target_keys["year"] = pd.to_numeric(target_keys["year"], errors="coerce")
    target_keys = target_keys.dropna(subset=["year"])
    target_keys["year"] = target_keys["year"].astype(int)

    merged = target_keys.merge(
        wave.rename(columns={"year": "ches_wave_year", "benchmark_ideology_score": "wave_score"}),
        left_on=["party", "year"],
        right_on=["party", "ches_wave_year"],
        how="left",
    )

    merged = merged.sort_values(["party", "year"]).reset_index(drop=True)

    # LOCF then backward-fill within party to support yearly plotting/validation.
    merged["benchmark_ideology_score"] = (
        merged.groupby("party", group_keys=False)["wave_score"].ffill().bfill()
    )
    merged["benchmark_imputed"] = merged["wave_score"].isna()
    merged["benchmark_source"] = "CHES 1999-2024 trend file (lrgen, Sweden)"
    merged["benchmark_note"] = (
        "Wave years are CHES observations; non-wave years are party-wise LOCF/backfill from nearest available CHES wave."
    )

    out = merged[
        [
            "party",
            "year",
            "benchmark_ideology_score",
            "benchmark_source",
            "benchmark_note",
            "benchmark_imputed",
            "ches_wave_year",
        ]
    ].copy()
    out = out.dropna(subset=["benchmark_ideology_score"]).sort_values(["party", "year"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def _load_external_party_year_benchmark_with_source(
    analysis_dir: Path,
    internal_df: pd.DataFrame,
    benchmark_source: str,
    ches_url: str,
    ches_country_code: int,
) -> pd.DataFrame:
    preferred = analysis_dir / "external_party_benchmarks_party_year.csv"
    fallback = analysis_dir / "external_party_benchmarks.csv"

    if benchmark_source == "ches":
        ext = _build_ches_party_year_benchmark(internal_df, preferred, ches_url, ches_country_code)
    elif benchmark_source == "auto" and not preferred.exists():
        try:
            ext = _build_ches_party_year_benchmark(internal_df, preferred, ches_url, ches_country_code)
        except Exception:
            ext = None
    else:
        ext = None

    if ext is None:
        if preferred.exists():
            ext = pd.read_csv(preferred)
        elif fallback.exists():
            base = pd.read_csv(fallback)
            keys = internal_df[["party", "year"]].drop_duplicates().copy()
            ext = keys.merge(base[["party", "benchmark_ideology_score"]], on="party", how="left")
        else:
            return pd.DataFrame(columns=["party", "year", "benchmark_ideology_score"])

    if "year" not in ext.columns:
        return pd.DataFrame(columns=["party", "year", "benchmark_ideology_score"])

    ext = ext[["party", "year", "benchmark_ideology_score"]].copy()
    ext["year"] = pd.to_numeric(ext["year"], errors="coerce")
    ext["benchmark_ideology_score"] = pd.to_numeric(ext["benchmark_ideology_score"], errors="coerce")
    ext = ext.dropna(subset=["year", "benchmark_ideology_score"])
    ext["year"] = ext["year"].astype(int)
    return ext.sort_values(["party", "year"]).reset_index(drop=True)


def plot_metric_vs_benchmark_party_year(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parties = sorted(df["party"].astype(str).unique().tolist())
    n = len(parties)
    if n >= 8:
        ncols = 4
    elif n >= 5:
        ncols = 3
    elif n > 1:
        ncols = 2
    else:
        ncols = 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.1 * ncols, max(3.4, 2.7 * nrows)),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, party in enumerate(parties):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        sub = df[df["party"] == party].sort_values("year")

        mvals = pd.to_numeric(sub["consistency_fulfillment_metric"], errors="coerce")
        bvals = pd.to_numeric(sub["benchmark_ideology_score"], errors="coerce")
        y_min = float(min(mvals.min(), bvals.min()))
        y_max = float(max(mvals.max(), bvals.max()))
        y_pad = max(0.02, 0.12 * (y_max - y_min if y_max > y_min else 1.0))

        ax.plot(
            sub["year"],
            sub["consistency_fulfillment_metric"],
            marker="o",
            markersize=4.4,
            linewidth=2.1,
            label="Consistency x fulfillment",
        )
        ax.plot(
            sub["year"],
            sub["benchmark_ideology_score"],
            marker="s",
            markersize=4.2,
            linewidth=1.8,
            label="External benchmark",
        )
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_title(str(party), fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(alpha=0.24)

    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    year_range = _year_range_label(df["year"])
    fig.suptitle(
        f"Party-year consistency-fulfillment metric vs external benchmark"
        + (f" ({year_range})" if year_range else ""),
        fontsize=13,
    )
    fig.supxlabel("Year (calendar year)")
    fig.supylabel("Index value (party-scaled panel range)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=2, frameon=False, fontsize=9)

    add_figure_credits(
        fig,
        n_total=len(df),
        n_parties=len(parties),
        date_range=year_range,
        source="output/analysis/promise_fulfillment_party_topic_year.parquet + output/analysis/external_party_benchmarks_party_year.csv",
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    safe = thermal_safe_defaults("safe")
    p = argparse.ArgumentParser(description="Integrated consistency and directional trend analysis")
    p.add_argument("--analysis-dir", default="output/analysis")
    p.add_argument("--figures-dir", default="output/manuscript/figures")
    p.add_argument("--cpu-fraction", type=float, default=float(os.environ.get("CLASSIFIER_CPU_FRACTION", str(safe["cpu_fraction"]))))
    p.add_argument("--mlflow", action="store_true")
    p.add_argument("--mlflow-experiment", default="consistency-trends")
    p.add_argument("--mlflow-tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    p.add_argument("--speech-motion-weight", type=float, default=0.5)
    p.add_argument("--vote-alignment-weight", type=float, default=0.5)
    p.add_argument("--fulfillment-fill", type=float, default=0.0)
    p.add_argument("--expected-contradiction-fill", type=float, default=0.0)
    p.add_argument("--expected-uphold-fill", type=float, default=1.0)
    p.add_argument("--contradiction-penalty-power", type=float, default=1.0)
    p.add_argument(
        "--benchmark-source",
        choices=["auto", "local", "ches"],
        default="local",
        help="External benchmark source: local files only (default), CHES forced refresh, or auto-build CHES when missing.",
    )
    p.add_argument("--ches-url", default=CHES_TREND_CSV_URL)
    p.add_argument("--ches-country-code", type=int, default=CHES_SWEDEN_COUNTRY_CODE)
    args = p.parse_args()

    throttle = apply_cpu_throttle(cpu_fraction=args.cpu_fraction)
    run = ExperimentRun.start(
        enabled=args.mlflow,
        experiment_name=args.mlflow_experiment,
        run_name="analyze-consistency-trends",
        tracking_uri=args.mlflow_tracking_uri,
    )

    analysis_dir = Path(args.analysis_dir)
    figures_dir = Path(args.figures_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    input_counts = ensure_min_rows_many(
        [
            (analysis_dir / "ideological_gap_party.parquet", 20, "ideological_gap_party"),
            (analysis_dir / "promise_fulfillment_party_summary.parquet", 5, "promise_fulfillment_party_summary"),
            (analysis_dir / "promise_fulfillment_party_topic_year.parquet", 200, "promise_fulfillment_party_topic_year"),
        ]
    )

    gap = pd.read_parquet(analysis_dir / "ideological_gap_party.parquet")
    pf = pd.read_parquet(analysis_dir / "promise_fulfillment_party_summary.parquet")
    drift = pd.read_parquet(analysis_dir / "party_ideology_drift_by_modality_year.parquet")
    topic_year = pd.read_parquet(analysis_dir / "promise_fulfillment_party_topic_year.parquet")

    consistency = build_consistency_score(
        gap,
        pf,
        speech_motion_weight=args.speech_motion_weight,
        fulfillment_fill_value=args.fulfillment_fill,
    )

    expected_p = analysis_dir / "speech_action_expected_contradiction_party_topic_year.parquet"
    edges_p = analysis_dir / "speech_action_contradiction_edges.parquet"
    if expected_p.exists():
        exp = pd.read_parquet(expected_p)
        exp_party = (
            exp.groupby("party", as_index=False)[["expected_contradiction", "expected_uphold"]]
            .mean()
            .fillna({"expected_contradiction": 0.0, "expected_uphold": 1.0})
        )
        consistency = consistency.merge(exp_party, on="party", how="left")
        consistency["expected_contradiction"] = pd.to_numeric(
            consistency["expected_contradiction"], errors="coerce"
        ).fillna(float(args.expected_contradiction_fill))
        consistency["expected_uphold"] = pd.to_numeric(consistency["expected_uphold"], errors="coerce").fillna(
            float(args.expected_uphold_fill)
        )
    else:
        consistency["expected_contradiction"] = float(args.expected_contradiction_fill)
        consistency["expected_uphold"] = float(args.expected_uphold_fill)

    # Composite fulfillment signal: both motion pathway completion and
    # speech->vote alignment should matter.
    consistency["motion_pathway_fulfillment"] = pd.to_numeric(
        consistency.get("pct_vote_given_motion_pathway"), errors="coerce"
    ).fillna(float(args.fulfillment_fill))

    w_vote = float(np.clip(args.vote_alignment_weight, 0.0, 1.0))
    w_motion = float(1.0 - w_vote)

    if edges_p.exists():
        vote_alignment = build_vote_alignment_fulfillment(pd.read_parquet(edges_p))
        consistency = consistency.merge(vote_alignment, on="party", how="left")
        consistency["vote_alignment_fulfillment"] = pd.to_numeric(
            consistency["vote_alignment_fulfillment"], errors="coerce"
        ).fillna(float(args.fulfillment_fill))
        consistency["fulfillment_signal"] = (
            w_vote * consistency["vote_alignment_fulfillment"]
            + w_motion * consistency["motion_pathway_fulfillment"]
        )
    else:
        consistency["vote_alignment_fulfillment"] = np.nan
        consistency["fulfillment_signal"] = consistency["motion_pathway_fulfillment"]

    consistency["consistency_x_fulfillment"] = consistency["consistency_score"] * consistency["fulfillment_signal"]

    penalty = np.power(1.0 - consistency["expected_contradiction"], float(args.contradiction_penalty_power))

    consistency["contradiction_adjusted_consistency"] = (
        consistency["consistency_score"] * penalty
    )
    consistency["ideology_uphold_v2"] = (
        consistency["consistency_score"]
        * consistency["fulfillment_signal"]
        * penalty
    )

    lead_lag = build_lead_lag(topic_year)
    direction = build_parliament_direction(topic_year)

    consistency_p = analysis_dir / "consistency_score_party.parquet"
    lead_lag_p = analysis_dir / "lead_lag_speech_to_action_party_year.parquet"
    direction_p = analysis_dir / "parliament_direction_over_time.parquet"

    consistency.to_parquet(consistency_p, index=False, compression="zstd")
    lead_lag.to_parquet(lead_lag_p, index=False, compression="zstd")
    direction.to_parquet(direction_p, index=False, compression="zstd")

    fig_consistency = figures_dir / "figure_consistency_vs_fulfillment.png"
    fig_direction = figures_dir / "figure_parliament_direction_over_time.png"
    fig_benchmark = figures_dir / "figure_consistency_fulfillment_vs_benchmark_party_year.png"

    plot_consistency_vs_fulfillment(consistency, fig_consistency)
    plot_parliament_direction(direction, fig_direction)
    metric_party_year = build_party_year_consistency_fulfillment(consistency, topic_year)
    external_party_year = _load_external_party_year_benchmark_with_source(
        analysis_dir,
        metric_party_year,
        benchmark_source=args.benchmark_source,
        ches_url=args.ches_url,
        ches_country_code=args.ches_country_code,
    )
    metric_vs_benchmark = metric_party_year.merge(external_party_year, on=["party", "year"], how="inner")
    plot_metric_vs_benchmark_party_year(metric_vs_benchmark, fig_benchmark)

    metric_party_year_p = analysis_dir / "consistency_fulfillment_party_year.parquet"
    metric_vs_benchmark_p = analysis_dir / "consistency_fulfillment_vs_benchmark_party_year.parquet"
    metric_party_year.to_parquet(metric_party_year_p, index=False, compression="zstd")
    metric_vs_benchmark.to_parquet(metric_vs_benchmark_p, index=False, compression="zstd")

    sd = lead_lag[lead_lag["party"] == "SD"].copy() if not lead_lag.empty else pd.DataFrame()

    summary = {
        "inputs": {
            "ideological_gap_party": str(analysis_dir / "ideological_gap_party.parquet"),
            "promise_fulfillment_party_summary": str(analysis_dir / "promise_fulfillment_party_summary.parquet"),
            "party_ideology_drift_by_modality_year": str(analysis_dir / "party_ideology_drift_by_modality_year.parquet"),
            "promise_fulfillment_party_topic_year": str(analysis_dir / "promise_fulfillment_party_topic_year.parquet"),
        },
        "outputs": {
            "consistency_score_party": str(consistency_p),
            "lead_lag_speech_to_action_party_year": str(lead_lag_p),
            "parliament_direction_over_time": str(direction_p),
            "figure_consistency_vs_fulfillment": str(fig_consistency),
            "figure_parliament_direction_over_time": str(fig_direction),
            "figure_consistency_fulfillment_vs_benchmark_party_year": str(fig_benchmark),
            "consistency_fulfillment_party_year": str(metric_party_year_p),
            "consistency_fulfillment_vs_benchmark_party_year": str(metric_vs_benchmark_p),
        },
        "summary_metrics": {
            "consistency_mean": float(consistency["consistency_score"].mean()) if not consistency.empty else None,
            "consistency_top_party": None if consistency.empty else str(consistency.iloc[0]["party"]),
            "parliament_action_slope_per_year": _slope(direction["year"], direction["parliament_median_action"]) if not direction.empty else None,
            "parliament_speech_slope_per_year": _slope(direction["year"], direction["parliament_median_speech"]) if not direction.empty else None,
            "sd_mean_action_vs_parliament_median_t_plus_1": None if sd.empty else float(sd["action_vs_parliament_median_t_plus_1"].mean()),
            "sd_mean_lead_shift_t_to_t1": None if sd.empty else float(sd["lead_shift_t_to_t1"].mean()),
        },
    }

    summary_p = analysis_dir / "consistency_trends_summary.json"
    summary_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run.log_params(
        {
            "analysis_dir": str(analysis_dir),
            "figures_dir": str(figures_dir),
            "cpu_fraction": args.cpu_fraction,
            "max_threads": throttle.get("max_threads"),
            "speech_motion_weight": args.speech_motion_weight,
            "vote_alignment_weight": args.vote_alignment_weight,
            "fulfillment_fill": args.fulfillment_fill,
            "expected_contradiction_fill": args.expected_contradiction_fill,
            "expected_uphold_fill": args.expected_uphold_fill,
            "contradiction_penalty_power": args.contradiction_penalty_power,
            **{f"input_rows_{k}": v for k, v in input_counts.items()},
        }
    )
    run.log_metrics(
        {
            "consistency_mean": summary["summary_metrics"]["consistency_mean"],
            "parliament_action_slope_per_year": summary["summary_metrics"]["parliament_action_slope_per_year"],
            "parliament_speech_slope_per_year": summary["summary_metrics"]["parliament_speech_slope_per_year"],
            "sd_mean_action_vs_parliament_median_t_plus_1": summary["summary_metrics"]["sd_mean_action_vs_parliament_median_t_plus_1"],
            "sd_mean_lead_shift_t_to_t1": summary["summary_metrics"]["sd_mean_lead_shift_t_to_t1"],
        }
    )
    run.log_artifact(str(summary_p))
    run.log_artifact(str(fig_consistency))
    run.log_artifact(str(fig_direction))
    run.end(status="FINISHED")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
