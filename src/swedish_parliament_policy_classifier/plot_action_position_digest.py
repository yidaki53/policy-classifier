#!/usr/bin/env python3
"""Create a compact figure for supported-action party positions and say/do transitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot supported-action positions and say/do transitions")
    parser.add_argument("--positions", default="output/analysis/party_supported_action_positions.parquet")
    parser.add_argument("--transitions", default="output/analysis/say_do_transitions.parquet")
    parser.add_argument("--out", default="output/manuscript/figures/figure_action_position_digest.png")
    args = parser.parse_args()

    positions_path = Path(args.positions)
    transitions_path = Path(args.transitions)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    positions = pd.read_parquet(positions_path) if positions_path.exists() else pd.DataFrame(columns=["party", "score_0_100"])
    transitions = pd.read_parquet(transitions_path) if transitions_path.exists() else pd.DataFrame(columns=["say_do_transition"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    if positions.empty:
        axes[0].text(0.5, 0.5, "No action-position data available", ha="center", va="center")
    else:
        positions = positions.sort_values("score_0_100", ascending=False).head(12).copy()
        axes[0].barh(positions["party"].astype(str), positions["score_0_100"].astype(float))
        axes[0].invert_yaxis()
        axes[0].set_title("Supported-action party positions")
        axes[0].set_xlabel("Support score (0–100)")

    if transitions.empty:
        axes[1].text(0.5, 0.5, "No say/do transition data available", ha="center", va="center")
    else:
        counts = transitions["say_do_transition"].fillna("unknown").astype(str).value_counts().sort_index()
        labels = [str(v) for v in counts.index]
        values = counts.to_numpy(dtype=float)
        axes[1].bar(labels, values)
        axes[1].set_title("Say/do transition counts")
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Action-side evidence digest")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
