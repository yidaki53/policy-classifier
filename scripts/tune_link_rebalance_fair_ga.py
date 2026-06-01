#!/usr/bin/env python3
"""Fairness-aware GA tuning for speech-action link rebalance.

Goal: find rebalance hyperparameters that increase motion linkage while
minimizing party-level disturbance. Uses a lightweight genetic algorithm
(DEAP-style evolutionary loop, no extra dependency required).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from swedish_parliament_policy_classifier.runtime.experiment import ExperimentRun


@dataclass
class SimState:
    n_total: int
    n_motion_base: int
    n_vote_base: int
    baseline_ratio: float
    target_ratio: float
    party_names: np.ndarray
    party_total: np.ndarray
    party_motion_base: np.ndarray
    party_share_base: np.ndarray
    eligible_margins_desc: np.ndarray
    eligible_party_idx_desc: np.ndarray


def _read_total_votes(votering_parquet_dir: Path, motion_votes_path: Path) -> int:
    total_votes = 0
    if votering_parquet_dir.exists():
        vote_ids: list[pd.Series] = []
        for p in sorted(votering_parquet_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(p)
            except Exception:
                continue
            if "votering_id" in df.columns:
                vote_ids.append(df["votering_id"].astype(str))
            elif "votering" in df.columns:
                vote_ids.append(df["votering"].astype(str))
        if vote_ids:
            total_votes = int(pd.concat(vote_ids, ignore_index=True).nunique())

    if total_votes <= 0 and motion_votes_path.exists():
        mv = pd.read_parquet(motion_votes_path, columns=["votering_count"]).copy()
        mv["votering_count"] = pd.to_numeric(mv["votering_count"], errors="coerce").fillna(0.0)
        total_votes = int(mv["votering_count"].sum())

    return max(1, total_votes)


def _compute_target_ratio(normalized_motions_path: Path, votering_parquet_dir: Path, motion_votes_path: Path) -> tuple[float, int, int]:
    nm = pd.read_parquet(normalized_motions_path, columns=["id"])
    total_motions = int(nm["id"].astype(str).nunique())
    total_votes = _read_total_votes(votering_parquet_dir, motion_votes_path)
    return float(total_motions / max(1, total_votes)), total_motions, total_votes


def _prepare_state(df: pd.DataFrame, target_ratio: float) -> SimState:
    required = {
        "speech_party",
        "action_type",
        "link_source",
        "graph_motion_candidate_motion_id",
        "graph_margin_motion_minus_vote",
        "graph_motion_score",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in base links for GA tuning: {sorted(missing)}")

    parties = df["speech_party"].astype(str).fillna("Unknown")
    party_codes, party_names = pd.factorize(parties)
    n_parties = int(len(party_names))

    party_total = np.bincount(party_codes, minlength=n_parties).astype(float)
    motion_mask = (df["action_type"].astype(str) == "motion").to_numpy()
    party_motion_base = np.bincount(party_codes[motion_mask], minlength=n_parties).astype(float)
    party_share_base = np.divide(party_motion_base, party_total, out=np.zeros_like(party_motion_base), where=party_total > 0)

    n_total = int(len(df))
    n_motion_base = int(motion_mask.sum())
    n_vote_base = int((df["action_type"].astype(str) == "vote").sum())
    baseline_ratio = float(n_motion_base / max(1, n_vote_base))

    eligible_mask = (
        (df["action_type"].astype(str) == "vote")
        & (~df["link_source"].astype(str).str.startswith("existing:"))
        & (df["graph_motion_candidate_motion_id"].astype(str) != "")
    )

    elig = df.loc[eligible_mask, ["graph_margin_motion_minus_vote", "graph_motion_score"]].copy()
    elig["graph_margin_motion_minus_vote"] = pd.to_numeric(elig["graph_margin_motion_minus_vote"], errors="coerce").fillna(-1.0)
    elig["graph_motion_score"] = pd.to_numeric(elig["graph_motion_score"], errors="coerce").fillna(-1.0)

    elig_party_idx = party_codes[eligible_mask.to_numpy()]

    order = np.lexsort((-elig["graph_motion_score"].to_numpy(), -elig["graph_margin_motion_minus_vote"].to_numpy()))
    # lexsort sorts ascending by last key; we negate for descending and then reverse order behavior is already descending.
    # Ensure true descending by margin then score.
    order = order
    margins = elig["graph_margin_motion_minus_vote"].to_numpy()[order]
    pidx = elig_party_idx[order]

    return SimState(
        n_total=n_total,
        n_motion_base=n_motion_base,
        n_vote_base=n_vote_base,
        baseline_ratio=baseline_ratio,
        target_ratio=float(target_ratio),
        party_names=np.asarray(party_names, dtype=object),
        party_total=party_total,
        party_motion_base=party_motion_base,
        party_share_base=party_share_base,
        eligible_margins_desc=margins,
        eligible_party_idx_desc=pidx,
    )


def _eligible_count_for_margin(margins_desc: np.ndarray, min_margin: float) -> int:
    asc = margins_desc[::-1]
    left = int(np.searchsorted(asc, float(min_margin), side="left"))
    return int(len(asc) - left)


def evaluate_candidate(
    state: SimState,
    target_share: float,
    min_margin: float,
    min_ratio_gain: float,
    w_max: float,
    w_mean: float,
    w_ratio: float,
) -> dict[str, float | int]:
    target_share = float(np.clip(target_share, 0.01, 0.99))
    min_margin = float(np.clip(min_margin, 0.0, 1.0))

    desired_motion = int(math.ceil(target_share * state.n_total))
    needed = max(0, desired_motion - state.n_motion_base)

    eligible = _eligible_count_for_margin(state.eligible_margins_desc, min_margin)
    k = int(min(needed, eligible))

    n_motion = int(state.n_motion_base + k)
    n_vote = int(state.n_vote_base - k)
    observed_ratio = float(n_motion / max(1, n_vote))
    motion_share = float(n_motion / max(1, state.n_total))

    add_counts = np.zeros_like(state.party_motion_base)
    if k > 0:
        add_counts += np.bincount(state.eligible_party_idx_desc[:k], minlength=len(state.party_names)).astype(float)

    party_share = np.divide(
        state.party_motion_base + add_counts,
        state.party_total,
        out=np.zeros_like(state.party_motion_base),
        where=state.party_total > 0,
    )
    deltas = np.abs(party_share - state.party_share_base)

    max_delta = float(deltas.max()) if deltas.size else 0.0
    mean_delta = float(deltas.mean()) if deltas.size else 0.0
    ratio_error = float(abs(math.log((observed_ratio + 1e-9) / (state.target_ratio + 1e-9))))

    min_ratio = float(state.baseline_ratio + max(0.0, min_ratio_gain))
    ratio_penalty = float(max(0.0, min_ratio - observed_ratio) * 1000.0)

    score = float(w_max * max_delta + w_mean * mean_delta + w_ratio * ratio_error + ratio_penalty)

    return {
        "target_motion_share": target_share,
        "min_motion_margin": min_margin,
        "n_motion": n_motion,
        "n_vote": n_vote,
        "motion_share": motion_share,
        "observed_ratio_motion_to_vote": observed_ratio,
        "baseline_ratio_motion_to_vote": float(state.baseline_ratio),
        "target_ratio_motion_to_vote": float(state.target_ratio),
        "rebalanced_rows": k,
        "needed": int(needed),
        "eligible": int(eligible),
        "max_party_motion_share_shift": max_delta,
        "mean_party_motion_share_shift": mean_delta,
        "ratio_error_log": ratio_error,
        "score": score,
    }


def _tournament_select(pop: list[np.ndarray], fitness: list[float], k: int, rng: np.random.Generator) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    n = len(pop)
    for _ in range(k):
        a, b, c = rng.integers(0, n, size=3)
        best = min([a, b, c], key=lambda i: fitness[i])
        out.append(pop[best].copy())
    return out


def _crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    alpha = rng.uniform(0.2, 0.8)
    c1 = alpha * a + (1.0 - alpha) * b
    c2 = alpha * b + (1.0 - alpha) * a
    return c1, c2


def _mutate(x: np.ndarray, rng: np.random.Generator, sigma_share: float, sigma_margin: float) -> np.ndarray:
    y = x.copy()
    if rng.random() < 0.7:
        y[0] += rng.normal(0.0, sigma_share)
    if rng.random() < 0.7:
        y[1] += rng.normal(0.0, sigma_margin)
    y[0] = float(np.clip(y[0], 0.01, 0.99))
    y[1] = float(np.clip(y[1], 0.0, 1.0))
    return y


def run_ga(state: SimState, args: argparse.Namespace, exp: ExperimentRun) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rng = np.random.default_rng(args.seed)

    pop_size = int(args.population)
    generations = int(args.generations)
    elite = max(1, int(args.elite))

    share_lo = float(np.clip(args.share_min, 0.01, 0.99))
    share_hi = float(np.clip(args.share_max, 0.01, 0.99))
    margin_lo = float(np.clip(args.margin_min, 0.0, 1.0))
    margin_hi = float(np.clip(args.margin_max, 0.0, 1.0))

    pop = [
        np.array([
            rng.uniform(share_lo, share_hi),
            rng.uniform(margin_lo, margin_hi),
        ], dtype=float)
        for _ in range(pop_size)
    ]

    records: list[dict[str, float | int]] = []

    for gen in range(generations):
        evals = [
            evaluate_candidate(
                state,
                target_share=float(ind[0]),
                min_margin=float(ind[1]),
                min_ratio_gain=float(args.min_ratio_gain),
                w_max=float(args.weight_max_party_shift),
                w_mean=float(args.weight_mean_party_shift),
                w_ratio=float(args.weight_ratio_error),
            )
            for ind in pop
        ]
        for r in evals:
            r["generation"] = int(gen)
        records.extend(evals)

        fitness = [float(r["score"]) for r in evals]
        order = np.argsort(fitness)

        best = evals[int(order[0])]
        exp.log_metrics(
            {
                "best_score": float(best["score"]),
                "best_max_party_shift": float(best["max_party_motion_share_shift"]),
                "best_mean_party_shift": float(best["mean_party_motion_share_shift"]),
                "best_ratio": float(best["observed_ratio_motion_to_vote"]),
                "best_rebalanced_rows": float(best["rebalanced_rows"]),
            },
            step=gen,
        )

        elites = [pop[int(i)].copy() for i in order[:elite]]

        selected = _tournament_select(pop, fitness, pop_size - elite, rng)

        children: list[np.ndarray] = []
        i = 0
        while i < len(selected):
            p1 = selected[i]
            p2 = selected[(i + 1) % len(selected)]
            if rng.random() < float(args.crossover_prob):
                c1, c2 = _crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if rng.random() < float(args.mutation_prob):
                c1 = _mutate(c1, rng, float(args.sigma_share), float(args.sigma_margin))
            if rng.random() < float(args.mutation_prob):
                c2 = _mutate(c2, rng, float(args.sigma_share), float(args.sigma_margin))

            children.append(c1)
            if len(children) < (pop_size - elite):
                children.append(c2)
            i += 2

        pop = elites + children[: pop_size - elite]

    trials = pd.DataFrame(records)
    trials = trials.sort_values(["score", "max_party_motion_share_shift", "ratio_error_log"], ascending=[True, True, True]).reset_index(drop=True)
    best_row = trials.iloc[0].to_dict()
    return trials, best_row


def main() -> None:
    p = argparse.ArgumentParser(description="Fairness-aware GA tuning for speech-action rebalance")
    p.add_argument("--base-links", default="data/parquet/speech_action_links_nobalance.parquet")
    p.add_argument("--normalized-motions", default="data/parquet/normalized_motions.parquet")
    p.add_argument("--motion-votes", default="data/parquet/motion_votes.parquet")
    p.add_argument("--votering-parquet-dir", default="data/votering/parquet")
    p.add_argument("--trials-out", default="output/analysis/link_rebalance_fair_ga_trials.parquet")
    p.add_argument("--summary-out", default="output/analysis/link_rebalance_fair_ga_best.json")

    p.add_argument("--population", type=int, default=80)
    p.add_argument("--generations", type=int, default=45)
    p.add_argument("--elite", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--crossover-prob", type=float, default=0.8)
    p.add_argument("--mutation-prob", type=float, default=0.7)
    p.add_argument("--sigma-share", type=float, default=0.06)
    p.add_argument("--sigma-margin", type=float, default=0.04)

    p.add_argument("--share-min", type=float, default=0.08)
    p.add_argument("--share-max", type=float, default=0.85)
    p.add_argument("--margin-min", type=float, default=0.0)
    p.add_argument("--margin-max", type=float, default=0.45)

    p.add_argument("--min-ratio-gain", type=float, default=0.20)
    p.add_argument("--weight-max-party-shift", type=float, default=1.00)
    p.add_argument("--weight-mean-party-shift", type=float, default=0.60)
    p.add_argument("--weight-ratio-error", type=float, default=0.15)

    p.add_argument("--mlflow", action="store_true")
    p.add_argument("--mlflow-experiment", default="speech-action-link-fairness-ga")
    p.add_argument("--mlflow-tracking-uri", default=None)
    args = p.parse_args()

    df = pd.read_parquet(args.base_links)
    target_ratio, total_motions, total_votes = _compute_target_ratio(
        Path(args.normalized_motions),
        Path(args.votering_parquet_dir),
        Path(args.motion_votes),
    )
    state = _prepare_state(df, target_ratio=target_ratio)

    exp = ExperimentRun.start(
        enabled=args.mlflow,
        experiment_name=args.mlflow_experiment,
        run_name="fairness-ga-search",
        tracking_uri=args.mlflow_tracking_uri,
    )
    exp.log_params(
        {
            "base_links": args.base_links,
            "target_ratio_motion_to_vote": target_ratio,
            "total_motions": total_motions,
            "total_votes": total_votes,
            "population": args.population,
            "generations": args.generations,
            "elite": args.elite,
            "seed": args.seed,
            "min_ratio_gain": args.min_ratio_gain,
            "w_max_party_shift": args.weight_max_party_shift,
            "w_mean_party_shift": args.weight_mean_party_shift,
            "w_ratio_error": args.weight_ratio_error,
        }
    )

    trials, best = run_ga(state, args, exp)

    trials_out = Path(args.trials_out)
    trials_out.parent.mkdir(parents=True, exist_ok=True)
    trials.to_parquet(trials_out, index=False, compression="zstd")

    summary = {
        "target_ratio_motion_to_vote": float(target_ratio),
        "baseline_ratio_motion_to_vote": float(state.baseline_ratio),
        "total_motions": int(total_motions),
        "total_votes": int(total_votes),
        "base_links": str(args.base_links),
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
            "best_score": float(best["score"]),
            "best_ratio": float(best["observed_ratio_motion_to_vote"]),
            "best_max_party_shift": float(best["max_party_motion_share_shift"]),
            "best_mean_party_shift": float(best["mean_party_motion_share_shift"]),
            "best_rebalanced_rows": float(best["rebalanced_rows"]),
        }
    )
    exp.end(status="FINISHED")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
