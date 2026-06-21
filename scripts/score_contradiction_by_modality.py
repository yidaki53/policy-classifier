#!/usr/bin/env python3
"""Extend contradiction scoring with explicit modality-aware expected contradiction.

Reads classifications.parquet and speech_action_links.parquet, computes expected
contradiction separately for each action type (vote, proposition, betankande, motion).
Also outputs a combined summary that integrates all modalities into the say-vs-do framework.

Usage:
    uv run python scripts/score_contradiction_by_modality.py
    uv run python scripts/score_contradiction_by_modality.py --links data/parquet/speech_action_links.parquet --edges output/analysis/speech_action_contradiction_edges.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_edge_scores(edges_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(edges_path).copy()
    df["speech_date"] = pd.to_datetime(df["speech_date"], errors="coerce", utc=True)
    df["year"] = df["speech_date"].dt.year
    df["party"] = df["speech_party"].astype(str)
    df["topic"] = df["category"].astype(str)
    df["weighted_contradiction"] = df["edge_confidence_raw"] * df["contradiction_score_raw"]
    return df


def compute_modality_contradiction(edges: pd.DataFrame, action_type: str | None = None) -> pd.DataFrame:
    """Compute expected contradiction for a specific action type, or all if None."""
    df = edges.copy()
    if action_type is not None:
        df = df[df["action_type"] == action_type]
    if df.empty:
        return pd.DataFrame(columns=[
            "party", "topic", "year", "action_type",
            "n_candidate_edges", "n_speeches",
            "mean_edge_confidence", "expected_contradiction", "expected_uphold",
        ])

    g = (
        df.groupby(["party", "topic", "year"], as_index=False)
        .agg(
            n_candidate_edges=("speech_id", "size"),
            n_speeches=("speech_id", "nunique"),
            mean_edge_confidence=("edge_confidence_raw", "mean"),
            weighted_contradiction_sum=("weighted_contradiction", "sum"),
            edge_conf_sum=("edge_confidence_raw", "sum"),
        )
    )
    g["expected_contradiction"] = 0.0
    mask = g["edge_conf_sum"] > 0
    g.loc[mask, "expected_contradiction"] = g.loc[mask, "weighted_contradiction_sum"] / g.loc[mask, "edge_conf_sum"]
    g["expected_contradiction"] = g["expected_contradiction"].clip(0.0, 1.0)
    g["expected_uphold"] = 1.0 - g["expected_contradiction"]
    g["action_type"] = action_type or "all"
    return g


def main() -> None:
    p = argparse.ArgumentParser(description="Modality-aware expected contradiction scoring")
    p.add_argument("--edges", default="output/analysis/speech_action_contradiction_edges.parquet")
    p.add_argument("--out", default="output/analysis/speech_action_expected_contradiction_by_modality.parquet")
    p.add_argument("--summary-out", default="output/analysis/speech_action_expected_contradiction_by_modality_summary.json")
    args = p.parse_args()

    edges = load_edge_scores(args.edges)

    modalities = ["vote", "motion", "proposition", "betankande"]
    parts = [compute_modality_contradiction(edges, mt) for mt in modalities + [None]]
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, compression="zstd")

    summary = {
        "output": str(out_path),
        "rows": int(len(out)),
        "action_types": sorted(out["action_type"].unique().tolist()) if not out.empty else [],
        "mean_expected_contradiction_by_modality": {
            at: float(out[out["action_type"] == at]["expected_contradiction"].mean()) if not out.empty else None
            for at in (modalities + ["all"])
        },
    }

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

if False:
    # Graphify hint: extends contradiction scoring with modality-aware aggregation
    from swedish_parliament_policy_classifier.analysis.contradiction_scoring import score_contradiction_edges, aggregate_expected_contradiction
