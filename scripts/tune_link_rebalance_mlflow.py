#!/usr/bin/env python3
"""Tune graph rebalance hyperparameters for speech-action linking with MLflow logging.

The script optimizes threshold-oriented parameters for the constrained graph
rebalance stage using an existing no-balance linkage table containing graph
candidate columns.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from swedish_parliament_policy_classifier.runtime.experiment import ExperimentRun


def _read_total_votes(votering_parquet_dir: Path, motion_votes_path: Path) -> int:
    total_votes = 0
    if votering_parquet_dir.exists():
        pieces = []
        for p in sorted(votering_parquet_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(p)
            except Exception:
                continue
            if "votering_id" in df.columns:
                pieces.append(df[["votering_id"]].copy())
            elif "votering" in df.columns:
                pieces.append(df[["votering"]].rename(columns={"votering": "votering_id"}))
        if pieces:
            all_votes = pd.concat(pieces, ignore_index=True)
            total_votes = int(all_votes["votering_id"].astype(str).nunique())

    if total_votes <= 0 and motion_votes_path.exists():
        mv = pd.read_parquet(motion_votes_path, columns=["votering_count"]).copy()
        mv["votering_count"] = pd.to_numeric(mv["votering_count"], errors="coerce").fillna(0.0)
        total_votes = int(mv["votering_count"].sum())

    return max(1, total_votes)


def _compute_target_ratio(normalized_motions_path: Path, votering_parquet_dir: Path, motion_votes_path: Path) -> tuple[float, int, int]:
    nm = pd.read_parquet(normalized_motions_path, columns=["id"])
    total_motions = int(nm["id"].astype(str).nunique())
    total_votes = _read_total_votes(votering_parquet_dir, motion_votes_path)
    ratio = float(total_motions / max(1, total_votes))
    return ratio, total_motions, total_votes


def _ratio_to_motion_share(ratio: float) -> float:
    ratio = max(1e-9, float(ratio))
    return float(ratio / (1.0 + ratio))


def _score_ratio_error(observed_ratio: float, target_ratio: float) -> float:
    # Symmetric relative error in log-space.
    eps = 1e-9
    return float(abs(math.log((observed_ratio + eps) / (target_ratio + eps))))


def _prepare_rebalance_simulation(df_base: pd.DataFrame) -> dict[str, object]:
    n_total = int(len(df_base))
    n_motion_base = int((df_base["action_type"] == "motion").sum())
    n_vote_base = int((df_base["action_type"] == "vote").sum())

    eligible_mask = (
        (df_base["action_type"] == "vote")
        & (~df_base["link_source"].astype(str).str.startswith("existing:"))
        & (df_base["graph_motion_candidate_motion_id"].astype(str) != "")
    )
    eligible_df = df_base.loc[eligible_mask, ["graph_margin_motion_minus_vote", "graph_motion_score"]].copy()
    eligible_df["graph_margin_motion_minus_vote"] = pd.to_numeric(
        eligible_df["graph_margin_motion_minus_vote"], errors="coerce"
    ).fillna(-1.0)
    eligible_df["graph_motion_score"] = pd.to_numeric(eligible_df["graph_motion_score"], errors="coerce").fillna(-1.0)
    eligible_df = eligible_df.sort_values(
        ["graph_margin_motion_minus_vote", "graph_motion_score"], ascending=[False, False]
    ).reset_index(drop=True)

    margins_sorted = eligible_df["graph_margin_motion_minus_vote"].to_numpy(dtype=float)
    return {
        "n_total": n_total,
        "n_motion_base": n_motion_base,
        "n_vote_base": n_vote_base,
        "margins_sorted_desc": margins_sorted,
        "n_eligible_all": int(len(margins_sorted)),
    }


def _eligible_count_for_margin(margins_sorted_desc: np.ndarray, min_margin: float) -> int:
    # Margins sorted descending; count values >= threshold.
    # Convert to ascending for searchsorted equivalently.
    asc = margins_sorted_desc[::-1]
    left_idx = int(np.searchsorted(asc, float(min_margin), side="left"))
    return int(len(asc) - left_idx)


def _run_trial(sim: dict[str, object], target_share: float, min_margin: float, target_ratio: float) -> dict[str, float | int]:
    n_total = int(sim["n_total"])
    n_motion_base = int(sim["n_motion_base"])
    n_vote_base = int(sim["n_vote_base"])
    margins = sim["margins_sorted_desc"]

    desired_motion = int(math.ceil(max(0.0, min(1.0, float(target_share))) * n_total))
    needed = max(0, desired_motion - n_motion_base)
    eligible = _eligible_count_for_margin(margins, float(min_margin))
    rebalanced = int(min(needed, eligible))

    n_motion = int(n_motion_base + rebalanced)
    n_vote = int(n_vote_base - rebalanced)
    observed_ratio = float(n_motion / max(1, n_vote))
    observed_share = float(n_motion / max(1, n_total))
    ratio_error = _score_ratio_error(observed_ratio, target_ratio)

    return {
        "target_motion_share": float(target_share),
        "min_motion_margin": float(min_margin),
        "n_motion": n_motion,
        "n_vote": n_vote,
        "motion_share": observed_share,
        "observed_ratio_motion_to_vote": observed_ratio,
        "target_ratio_motion_to_vote": float(target_ratio),
        "ratio_error_log": float(ratio_error),
        "rebalanced_rows": rebalanced,
        "needed": int(needed),
        "eligible": int(eligible),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Tune graph rebalance thresholds with MLflow logging")
    p.add_argument("--base-links", default="data/parquet/speech_action_links_nobalance.parquet")
    p.add_argument("--normalized-motions", default="data/parquet/normalized_motions.parquet")
    p.add_argument("--motion-votes", default="data/parquet/motion_votes.parquet")
    p.add_argument("--votering-parquet-dir", default="data/votering/parquet")
    p.add_argument("--trials-out", default="output/analysis/link_rebalance_hparam_trials.parquet")
    p.add_argument("--summary-out", default="output/analysis/link_rebalance_hparam_best.json")
    p.add_argument("--target-share-center", type=float, default=None)
    p.add_argument("--target-share-span", type=float, default=0.35)
    p.add_argument("--target-share-steps", type=int, default=12)
    p.add_argument("--margin-min", type=float, default=0.00)
    p.add_argument("--margin-max", type=float, default=0.20)
    p.add_argument("--margin-steps", type=int, default=21)
    p.add_argument("--mlflow", action="store_true")
    p.add_argument("--mlflow-experiment", default="speech-action-link-rebalance")
    p.add_argument("--mlflow-tracking-uri", default=None)
    args = p.parse_args()

    base_path = Path(args.base_links)
    if not base_path.exists():
        raise SystemExit(
            f"Missing base links file: {base_path}. Run link_all_speeches_to_action.py with graph balance disabled first."
        )

    df_base = pd.read_parquet(base_path)
    required_cols = {
        "action_type",
        "link_source",
        "graph_motion_candidate_motion_id",
        "graph_margin_motion_minus_vote",
        "graph_motion_score",
    }
    missing = required_cols.difference(df_base.columns)
    if missing:
        raise SystemExit(f"Base links file missing graph columns: {sorted(missing)}")

    target_ratio, total_motions, total_votes = _compute_target_ratio(
        Path(args.normalized_motions),
        Path(args.votering_parquet_dir),
        Path(args.motion_votes),
    )

    target_center = args.target_share_center
    if target_center is None:
        target_center = _ratio_to_motion_share(target_ratio)

    target_lo = max(0.01, float(target_center - args.target_share_span / 2.0))
    target_hi = min(0.99, float(target_center + args.target_share_span / 2.0))
    target_grid = np.linspace(target_lo, target_hi, max(2, int(args.target_share_steps)))
    margin_grid = np.linspace(float(args.margin_min), float(args.margin_max), max(2, int(args.margin_steps)))

    sim = _prepare_rebalance_simulation(df_base)

    exp = ExperimentRun.start(
        enabled=args.mlflow,
        experiment_name=args.mlflow_experiment,
        run_name="link-rebalance-hparam-search",
        tracking_uri=args.mlflow_tracking_uri,
    )
    exp.log_params(
        {
            "base_links": str(base_path),
            "target_ratio_motion_to_vote": target_ratio,
            "total_motions": total_motions,
            "total_votes": total_votes,
            "target_share_center": target_center,
            "target_share_steps": len(target_grid),
            "margin_steps": len(margin_grid),
            "search_space_size": int(len(target_grid) * len(margin_grid)),
            "n_total_links": int(sim["n_total"]),
            "n_motion_base": int(sim["n_motion_base"]),
            "n_vote_base": int(sim["n_vote_base"]),
            "n_eligible_all": int(sim["n_eligible_all"]),
        }
    )

    trials: list[dict[str, float | int]] = []
    step = 0
    for target_share in target_grid:
        for min_margin in margin_grid:
            trial = _run_trial(
                sim,
                target_share=float(target_share),
                min_margin=float(min_margin),
                target_ratio=target_ratio,
            )
            trial["trial_index"] = int(step)
            trials.append(trial)
            exp.log_metrics(
                {
                    "ratio_error_log": float(trial["ratio_error_log"]),
                    "motion_share": float(trial["motion_share"]),
                    "observed_ratio_motion_to_vote": float(trial["observed_ratio_motion_to_vote"]),
                    "rebalanced_rows": float(trial["rebalanced_rows"]),
                },
                step=step,
            )
            step += 1

    trials_df = pd.DataFrame(trials)
    trials_df = trials_df.sort_values(["ratio_error_log", "rebalanced_rows"], ascending=[True, False]).reset_index(drop=True)
    best = trials_df.iloc[0].to_dict()

    trials_out = Path(args.trials_out)
    trials_out.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_parquet(trials_out, index=False, compression="zstd")

    summary = {
        "target_ratio_motion_to_vote": float(target_ratio),
        "total_motions": int(total_motions),
        "total_votes": int(total_votes),
        "base_links": str(base_path),
        "trials_path": str(trials_out),
        "best": best,
    }

    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    exp.log_artifact(str(trials_out))
    exp.log_artifact(str(summary_out))
    exp.log_metrics(
        {
            "best_ratio_error_log": float(best["ratio_error_log"]),
            "best_motion_share": float(best["motion_share"]),
            "best_observed_ratio_motion_to_vote": float(best["observed_ratio_motion_to_vote"]),
            "best_rebalanced_rows": float(best["rebalanced_rows"]),
        }
    )
    exp.end(status="FINISHED")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
