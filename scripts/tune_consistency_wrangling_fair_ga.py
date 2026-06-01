#!/usr/bin/env python3
"""Fairness-aware GA tuning for consistency-score wrangling.

Optimizes wrangling parameters to minimize party-level movement across multiple
linkage scenarios (via scenario-specific expected_contradiction tables).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from swedish_parliament_policy_classifier.runtime.experiment import ExperimentRun


IDEOLOGY_ORDER = [
    "far_left",
    "left",
    "centre_left",
    "centre",
    "centre_right",
    "right",
    "far_right",
]


@dataclass
class BaseFrames:
    pivot: pd.DataFrame
    pf: pd.DataFrame


def _prepare_base(gap_path: Path, pf_path: Path) -> BaseFrames:
    gap = pd.read_parquet(gap_path)
    pf = pd.read_parquet(pf_path)

    gap = gap[~gap["party"].astype(str).str.lower().isin({"unknown", "nyd", ""})].copy()
    pf = pf[~pf["party"].astype(str).str.lower().isin({"unknown", "nyd", ""})].copy()

    pivot = gap.pivot_table(index="party", columns="comparison", values="js_distance", aggfunc="mean").reset_index()
    for col in ["speech_vs_motion", "speech_vs_vote", "motion_vs_vote"]:
        if col not in pivot.columns:
            pivot[col] = np.nan

    return BaseFrames(pivot=pivot, pf=pf)


def _party_expected(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    g = (
        df.groupby("party", as_index=False)[["expected_contradiction", "expected_uphold"]]
        .mean()
        .fillna({"expected_contradiction": 0.0, "expected_uphold": 1.0})
    )
    return g


def _party_vote_alignment(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    votes = df[df["action_type"].astype(str) == "vote"].copy()
    votes = votes[~votes["speech_party"].astype(str).str.lower().isin({"unknown", "nyd", ""})].copy()
    if votes.empty:
        return pd.DataFrame(columns=["party", "vote_alignment_fulfillment"])

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
        .apply(lambda g: pd.Series({"vote_alignment_fulfillment": _weighted_mean(g)}))
        .reset_index(drop=True)
        .rename(columns={"speech_party": "party"})
    )
    return out


def _score_one_scenario(
    base: BaseFrames,
    exp_party: pd.DataFrame,
    align_party: pd.DataFrame,
    speech_motion_weight: float,
    vote_alignment_weight: float,
    fulfillment_fill: float,
    expected_contradiction_fill: float,
    contradiction_penalty_power: float,
) -> pd.DataFrame:
    w_sm = float(np.clip(speech_motion_weight, 0.0, 1.0))
    w_sv = float(1.0 - w_sm)

    out = base.pivot.copy()
    out["consistency_score"] = 1.0 - (w_sm * out["speech_vs_motion"] + w_sv * out["speech_vs_vote"])
    out["action_cohesion_score"] = 1.0 - out["motion_vs_vote"]

    out = out.merge(base.pf, on="party", how="left")
    out = out.merge(exp_party, on="party", how="left")
    out = out.merge(align_party, on="party", how="left")

    out["expected_contradiction"] = pd.to_numeric(out["expected_contradiction"], errors="coerce").fillna(
        float(expected_contradiction_fill)
    )
    out["expected_uphold"] = pd.to_numeric(out["expected_uphold"], errors="coerce").fillna(1.0)

    vote = pd.to_numeric(out.get("pct_speech_motion_vote"), errors="coerce").fillna(0.0)
    motion_no_vote = pd.to_numeric(out.get("pct_speech_motion_no_vote"), errors="coerce").fillna(0.0)
    denom = vote + motion_no_vote
    out["motion_pathway_fulfillment"] = np.where(denom > 0, vote / denom, np.nan)

    out["vote_alignment_fulfillment"] = pd.to_numeric(
        out.get("vote_alignment_fulfillment"), errors="coerce"
    ).fillna(float(fulfillment_fill))

    w_vote = float(np.clip(vote_alignment_weight, 0.0, 1.0))
    w_motion = float(1.0 - w_vote)
    out["fulfillment_signal"] = (
        w_vote * out["vote_alignment_fulfillment"]
        + w_motion * pd.to_numeric(out["motion_pathway_fulfillment"], errors="coerce").fillna(float(fulfillment_fill))
    )

    penalty = np.power(1.0 - out["expected_contradiction"], float(max(0.1, contradiction_penalty_power)))
    out["consistency_x_fulfillment"] = out["consistency_score"] * out["fulfillment_signal"]
    out["contradiction_adjusted_consistency"] = out["consistency_score"] * penalty
    out["ideology_uphold_v2"] = out["consistency_score"] * out["fulfillment_signal"] * penalty

    return out[["party", "consistency_score", "contradiction_adjusted_consistency", "ideology_uphold_v2"]].copy()


def evaluate_candidate(
    base: BaseFrames,
    scenarios: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    params: np.ndarray,
    weight_max: float,
    weight_mean: float,
    weight_std: float,
) -> dict[str, float]:
    speech_motion_weight = float(np.clip(params[0], 0.0, 1.0))
    vote_alignment_weight = float(np.clip(params[1], 0.0, 1.0))
    fulfillment_fill = float(np.clip(params[2], 0.0, 1.0))
    expected_contradiction_fill = float(np.clip(params[3], 0.0, 1.0))
    contradiction_penalty_power = float(np.clip(params[4], 0.1, 2.5))

    per = []
    for name, pair in scenarios.items():
        exp_party, align_party = pair
        sc = _score_one_scenario(
            base,
            exp_party,
            align_party,
            speech_motion_weight=speech_motion_weight,
            vote_alignment_weight=vote_alignment_weight,
            fulfillment_fill=fulfillment_fill,
            expected_contradiction_fill=expected_contradiction_fill,
            contradiction_penalty_power=contradiction_penalty_power,
        )
        sc = sc.rename(columns={
            "consistency_score": f"consistency_score__{name}",
            "contradiction_adjusted_consistency": f"contradiction_adjusted_consistency__{name}",
            "ideology_uphold_v2": f"ideology_uphold_v2__{name}",
        })
        per.append(sc)

    merged = per[0]
    for nxt in per[1:]:
        merged = merged.merge(nxt, on="party", how="outer")

    cols = [c for c in merged.columns if c.startswith("ideology_uphold_v2__")]
    vals = merged[cols].to_numpy(dtype=float)
    ranges = np.nanmax(vals, axis=1) - np.nanmin(vals, axis=1)

    max_shift = float(np.nanmax(ranges)) if len(ranges) else 0.0
    mean_shift = float(np.nanmean(ranges)) if len(ranges) else 0.0

    # avoid degenerate flattening by keeping some spread in base scenario
    base_cols = [c for c in merged.columns if c.endswith("__fair") and c.startswith("ideology_uphold_v2")]
    if not base_cols:
        base_cols = [c for c in merged.columns if c.startswith("ideology_uphold_v2__")][:1]
    base_std = float(np.nanstd(merged[base_cols[0]].to_numpy(dtype=float))) if base_cols else 0.0
    std_penalty = float(max(0.0, 0.05 - base_std))

    score = float(weight_max * max_shift + weight_mean * mean_shift + weight_std * std_penalty)

    return {
        "speech_motion_weight": speech_motion_weight,
        "vote_alignment_weight": vote_alignment_weight,
        "fulfillment_fill": fulfillment_fill,
        "expected_contradiction_fill": expected_contradiction_fill,
        "contradiction_penalty_power": contradiction_penalty_power,
        "max_party_shift": max_shift,
        "mean_party_shift": mean_shift,
        "base_std": base_std,
        "std_penalty": std_penalty,
        "score": score,
    }


def _tournament_select(pop: list[np.ndarray], fit: list[float], k: int, rng: np.random.Generator) -> list[np.ndarray]:
    out = []
    n = len(pop)
    for _ in range(k):
        ids = rng.integers(0, n, size=3)
        best = min(ids, key=lambda i: fit[int(i)])
        out.append(pop[int(best)].copy())
    return out


def _crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    alpha = rng.uniform(0.2, 0.8)
    return alpha * a + (1 - alpha) * b, alpha * b + (1 - alpha) * a


def _mutate(x: np.ndarray, rng: np.random.Generator, sigmas: np.ndarray) -> np.ndarray:
    y = x.copy()
    for i in range(len(y)):
        if rng.random() < 0.7:
            y[i] += rng.normal(0.0, sigmas[i])
    y[0] = float(np.clip(y[0], 0.0, 1.0))
    y[1] = float(np.clip(y[1], 0.0, 1.0))
    y[2] = float(np.clip(y[2], 0.0, 1.0))
    y[3] = float(np.clip(y[3], 0.0, 1.0))
    y[4] = float(np.clip(y[4], 0.1, 2.5))
    return y


def run_ga(base: BaseFrames, scenarios: dict[str, tuple[pd.DataFrame, pd.DataFrame]], args: argparse.Namespace, run: ExperimentRun) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(args.seed)

    pop = [
        np.array(
            [
                rng.uniform(0.2, 0.8),
                rng.uniform(0.2, 0.8),
                rng.uniform(0.0, 0.3),
                rng.uniform(0.0, 0.1),
                rng.uniform(0.7, 1.6),
            ],
            dtype=float,
        )
        for _ in range(args.population)
    ]
    # include current default policy
    pop[0] = np.array([0.5, 0.5, 0.0, 0.0, 1.0], dtype=float)

    records: list[dict[str, float]] = []
    sigmas = np.array([0.05, 0.05, 0.04, 0.03, 0.10], dtype=float)

    for gen in range(args.generations):
        evals = [
            evaluate_candidate(
                base,
                scenarios,
                ind,
                weight_max=args.weight_max_shift,
                weight_mean=args.weight_mean_shift,
                weight_std=args.weight_std_penalty,
            )
            for ind in pop
        ]
        for r in evals:
            r["generation"] = float(gen)
        records.extend(evals)

        fit = [float(r["score"]) for r in evals]
        order = np.argsort(fit)
        best = evals[int(order[0])]
        run.log_metrics(
            {
                "best_score": best["score"],
                "best_max_party_shift": best["max_party_shift"],
                "best_mean_party_shift": best["mean_party_shift"],
                "best_base_std": best["base_std"],
            },
            step=gen,
        )

        elite = [pop[int(i)].copy() for i in order[: args.elite]]
        selected = _tournament_select(pop, fit, args.population - args.elite, rng)

        children: list[np.ndarray] = []
        i = 0
        while i < len(selected):
            p1 = selected[i]
            p2 = selected[(i + 1) % len(selected)]
            if rng.random() < args.crossover_prob:
                c1, c2 = _crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            if rng.random() < args.mutation_prob:
                c1 = _mutate(c1, rng, sigmas)
            if rng.random() < args.mutation_prob:
                c2 = _mutate(c2, rng, sigmas)
            children.append(c1)
            if len(children) < (args.population - args.elite):
                children.append(c2)
            i += 2

        pop = elite + children[: args.population - args.elite]

    trials = pd.DataFrame(records).sort_values(["score", "max_party_shift", "mean_party_shift"], ascending=[True, True, True]).reset_index(drop=True)
    best = trials.iloc[0].to_dict()
    return trials, best


def main() -> None:
    p = argparse.ArgumentParser(description="GA tune consistency wrangling for fairness")
    p.add_argument("--gap", default="output/analysis/ideological_gap_party.parquet")
    p.add_argument("--pf", default="output/analysis/promise_fulfillment_party_summary.parquet")
    p.add_argument(
        "--expected-scenarios",
        default="fair=output/analysis/speech_action_expected_contradiction_party_topic_year.parquet,nobalance=output/analysis/speech_action_expected_contradiction_party_topic_year_nobalance.parquet,highmotion=output/analysis/speech_action_expected_contradiction_party_topic_year_highmotion.parquet",
    )
    p.add_argument(
        "--edge-scenarios",
        default="fair=output/analysis/speech_action_contradiction_edges.parquet,nobalance=output/analysis/speech_action_contradiction_edges_nobalance.parquet,highmotion=output/analysis/speech_action_contradiction_edges_highmotion.parquet",
    )
    p.add_argument("--trials-out", default="output/analysis/consistency_wrangling_fair_ga_trials.parquet")
    p.add_argument("--summary-out", default="output/analysis/consistency_wrangling_fair_ga_best.json")
    p.add_argument("--population", type=int, default=72)
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--elite", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--crossover-prob", type=float, default=0.8)
    p.add_argument("--mutation-prob", type=float, default=0.7)
    p.add_argument("--weight-max-shift", type=float, default=1.0)
    p.add_argument("--weight-mean-shift", type=float, default=0.7)
    p.add_argument("--weight-std-penalty", type=float, default=0.4)
    p.add_argument("--mlflow", action="store_true")
    p.add_argument("--mlflow-experiment", default="consistency-wrangling-fairness-ga")
    p.add_argument("--mlflow-tracking-uri", default=None)
    args = p.parse_args()

    base = _prepare_base(Path(args.gap), Path(args.pf))

    expected_scenarios: dict[str, pd.DataFrame] = {}
    for part in args.expected_scenarios.split(","):
        part = part.strip()
        if not part:
            continue
        name, path = part.split("=", 1)
        pth = Path(path.strip())
        if pth.exists():
            expected_scenarios[name.strip()] = _party_expected(pth)

    edge_scenarios: dict[str, pd.DataFrame] = {}
    for part in args.edge_scenarios.split(","):
        part = part.strip()
        if not part:
            continue
        name, path = part.split("=", 1)
        pth = Path(path.strip())
        if pth.exists():
            edge_scenarios[name.strip()] = _party_vote_alignment(pth)

    scenario_names = sorted(set(expected_scenarios.keys()) & set(edge_scenarios.keys()))
    scenarios: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        n: (expected_scenarios[n], edge_scenarios[n]) for n in scenario_names
    }

    if len(scenarios) < 2:
        raise SystemExit("Need at least 2 expected_contradiction scenarios for fairness tuning.")

    run = ExperimentRun.start(
        enabled=args.mlflow,
        experiment_name=args.mlflow_experiment,
        run_name="consistency-wrangling-fairness-ga",
        tracking_uri=args.mlflow_tracking_uri,
    )
    run.log_params(
        {
            "gap": args.gap,
            "pf": args.pf,
            "scenarios": ",".join(sorted(scenarios.keys())),
            "population": args.population,
            "generations": args.generations,
            "elite": args.elite,
            "seed": args.seed,
            "weight_max_shift": args.weight_max_shift,
            "weight_mean_shift": args.weight_mean_shift,
            "weight_std_penalty": args.weight_std_penalty,
        }
    )

    trials, best = run_ga(base, scenarios, args, run)

    trials_out = Path(args.trials_out)
    trials_out.parent.mkdir(parents=True, exist_ok=True)
    trials.to_parquet(trials_out, index=False, compression="zstd")

    summary = {
        "scenarios": sorted(scenarios.keys()),
        "trials_path": str(trials_out),
        "best": best,
    }
    summary_out = Path(args.summary_out)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    run.log_artifact(str(trials_out))
    run.log_artifact(str(summary_out))
    run.log_metrics(
        {
            "best_score": float(best["score"]),
            "best_max_party_shift": float(best["max_party_shift"]),
            "best_mean_party_shift": float(best["mean_party_shift"]),
            "best_base_std": float(best["base_std"]),
        }
    )
    run.end(status="FINISHED")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
