#!/usr/bin/env python3
"""Materialize party-position and say/do outputs from action evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from swedish_parliament_policy_classifier.analysis.contracts import StudySpecification
from swedish_parliament_policy_classifier.analysis.party_position import (
    estimate_supported_action_positions,
)
from swedish_parliament_policy_classifier.analysis.say_do import (
    SayDoTransition,
    classify_say_do_transition,
)


def build_action_position_outputs(
    party_choices_path: str | Path,
    policy_probabilities_path: str | Path,
    speech_stances_path: str | Path,
    out_dir: str | Path,
    *,
    specification: StudySpecification | None = None,
) -> dict[str, Any]:
    party_choices = pd.read_parquet(party_choices_path)
    policy_probabilities = pd.read_parquet(policy_probabilities_path)
    speech_stances = pd.read_parquet(speech_stances_path)

    if specification is None:
        decision_dates = pd.to_datetime(party_choices.get("decision_date"), errors="coerce", utc=True)
        latest_decision_date = decision_dates.max()
        if pd.isna(latest_decision_date):
            raise ValueError("party choices must contain at least one valid decision_date")
        specification = StudySpecification(data_cutoff=latest_decision_date)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    positions = estimate_supported_action_positions(
        party_choices,
        policy_probabilities,
        specification=specification,
    )
    positions_path = out_dir_path / "party_supported_action_positions.parquet"
    positions.to_parquet(positions_path, index=False, compression="zstd")

    merged = speech_stances.merge(
        party_choices[["decision_id", "party", "party_choice"]],
        left_on=["decision_id", "speech_party"],
        right_on=["decision_id", "party"],
        how="left",
    )
    merged["say_do_transition"] = merged.apply(
        lambda row: classify_say_do_transition(
            row.get("speech_stance", "unclear"),
            row.get("party_choice", "no_present"),
        ).value,
        axis=1,
    )
    transitions = merged[["speech_id", "motion_id", "decision_id", "speech_party", "say_do_transition"]].copy()
    transitions_path = out_dir_path / "say_do_transitions.parquet"
    transitions.to_parquet(transitions_path, index=False, compression="zstd")

    summary = {
        "study_specification": specification.to_dict(),
        "party_positions": str(positions_path),
        "say_do_transitions": str(transitions_path),
        "party_position_n": int(len(positions)),
        "transition_n": int(len(transitions)),
        "transition_counts": {
            str(key): int(value)
            for key, value in transitions["say_do_transition"].value_counts().sort_index().items()
        },
    }
    summary_path = out_dir_path / "action_position_outputs_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build action-position analysis outputs")
    parser.add_argument("--party-choices", default="output/analysis/party_decision_choices.parquet")
    parser.add_argument("--policy-probabilities", default="data/parquet/policy_probabilities.parquet")
    parser.add_argument("--speech-stances", default="data/parquet/speech_action_links_with_prop_bet.parquet")
    parser.add_argument("--out-dir", default="output/analysis")
    args = parser.parse_args()

    summary = build_action_position_outputs(
        args.party_choices,
        args.policy_probabilities,
        args.speech_stances,
        args.out_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
