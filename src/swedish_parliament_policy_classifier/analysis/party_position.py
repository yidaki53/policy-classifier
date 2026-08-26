"""Estimate party placement from policy alternatives selected in roll calls."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from swedish_parliament_policy_classifier.analysis.ideology_axes import (
    canonical_axis_order,
)
from swedish_parliament_policy_classifier.analysis.contracts import StudySpecification


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _decision_scores(
    policy_probabilities: pd.DataFrame,
    category_order: Sequence[str],
) -> pd.DataFrame:
    required = ("decision_id", "category", "probability")
    _require_columns(policy_probabilities, required, "Policy probabilities")

    probabilities = policy_probabilities.loc[:, required].copy()
    probabilities["decision_id"] = probabilities["decision_id"].astype(str)
    probabilities["category"] = probabilities["category"].astype(str)
    probabilities["probability"] = pd.to_numeric(
        probabilities["probability"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    probabilities = probabilities[probabilities["category"].isin(category_order)].copy()

    matrix = probabilities.pivot_table(
        index="decision_id",
        columns="category",
        values="probability",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(columns=list(category_order), fill_value=0.0)
    row_sums = matrix.sum(axis=1)
    matrix = matrix.loc[row_sums > 0].div(row_sums[row_sums > 0], axis=0)

    anchors = np.linspace(0.0, 100.0, num=len(category_order), dtype=float)
    scores = matrix.to_numpy(dtype=float) @ anchors
    return pd.DataFrame(
        {"decision_id": matrix.index.astype(str), "decision_score_0_100": scores}
    )


def estimate_supported_action_positions(
    party_choices: pd.DataFrame,
    policy_probabilities: pd.DataFrame,
    *,
    category_order: Sequence[str] | None = None,
    specification: StudySpecification | None = None,
    start_date: object | None = None,
    end_date: object | None = None,
) -> pd.DataFrame:
    """Estimate an equal-decision party score from affirmatively selected content.

    ``0`` and ``100`` are anchors of the declared category ontology.  They are
    not universal ideological endpoints.  No votes are excluded from this
    primary placement rather than converted into an assumed opposite position.
    """

    required = ("decision_id", "party", "decision_date", "party_choice")
    _require_columns(party_choices, required, "Party choices")
    choices = party_choices.loc[:, required].copy()
    choices["decision_id"] = choices["decision_id"].astype(str)
    choices["party"] = choices["party"].astype(str)
    choices["decision_date"] = pd.to_datetime(
        choices["decision_date"], errors="coerce", utc=True
    )
    choices["party_choice"] = choices["party_choice"].astype(str).str.casefold()

    duplicate = choices.duplicated(["decision_id", "party"], keep=False)
    if duplicate.any():
        raise ValueError("Party choices must contain one row per decision and party")

    if specification is not None:
        if start_date is not None or end_date is not None:
            raise ValueError("Use either specification or explicit date bounds, not both")
        start_date = specification.primary_start_date
        end_date = specification.primary_end_date

    if start_date is not None:
        start = pd.to_datetime(start_date, utc=True).normalize()
        choices = choices[choices["decision_date"] >= start]
    if end_date is not None:
        end_exclusive = pd.to_datetime(end_date, utc=True).normalize() + pd.Timedelta(days=1)
        choices = choices[choices["decision_date"] < end_exclusive]

    choices = choices[choices["party_choice"] == "yes"].copy()
    axis = tuple(category_order or canonical_axis_order())
    if len(axis) < 2 or len(set(axis)) != len(axis):
        raise ValueError("Category order must contain at least two unique categories")

    scored = choices.merge(
        _decision_scores(policy_probabilities, axis), on="decision_id", how="inner"
    )
    if scored.empty:
        return pd.DataFrame(
            columns=[
                "party",
                "supported_decision_n",
                "score_0_100",
                "score_median_0_100",
                "score_q25_0_100",
                "score_q75_0_100",
            ]
        )

    return (
        scored.groupby("party", as_index=False)["decision_score_0_100"]
        .agg(
            supported_decision_n="size",
            score_0_100="mean",
            score_median_0_100="median",
            score_q25_0_100=lambda values: values.quantile(0.25),
            score_q75_0_100=lambda values: values.quantile(0.75),
        )
        .sort_values("score_0_100")
        .reset_index(drop=True)
    )


__all__ = ["estimate_supported_action_positions"]