"""Canonical construction of party-level roll-call choices.

Raw Riksdag vote files contain one row per member.  Publication analyses need
one auditable observation per party and decision, while retaining the complete
Yes/No/abstain/absence composition rather than reducing every non-Yes record to
No.  This module owns that conversion.
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd


DECISION_KEY_COLUMNS = ("rm", "beteckning", "punkt", "votering_id")
VOTE_STATES = ("yes", "no", "abstain", "absent", "other")


def _ascii_casefold(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value).strip().casefold())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _vote_state(value: object) -> str:
    normalized = _ascii_casefold(value)
    if normalized in {"ja", "j", "yes", "y", "for", "1"}:
        return "yes"
    if normalized in {"nej", "n", "no", "0"}:
        return "no"
    if normalized in {"avstar", "avstod", "abstain", "abstained"}:
        return "abstain"
    if normalized in {"franvarande", "absent", "franvaro"}:
        return "absent"
    return "other"


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Vote records missing required columns: {missing}")


def aggregate_party_choices(vote_records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate member votes into one row per decision and party.

    The resulting ``party_choice`` is the unique plurality among present
    members' Yes, No, and abstention records.  Equal leading counts are marked
    ``split``; parties with no present members are marked ``no_present``.
    Absence is never interpreted as opposition.
    """

    required = (*DECISION_KEY_COLUMNS, "datum", "parti", "rost")
    _require_columns(vote_records, required)

    work = vote_records.loc[:, required].copy()
    for column in DECISION_KEY_COLUMNS:
        work[column] = work[column].fillna("").astype(str).str.strip()
    work["party"] = work["parti"].fillna("").astype(str).str.strip()
    work = work[work["party"] != ""].copy()
    work["decision_date"] = pd.to_datetime(work["datum"], errors="coerce", utc=True)
    work["vote_state"] = work["rost"].map(_vote_state)
    work["decision_id"] = work.loc[:, DECISION_KEY_COLUMNS].agg("|".join, axis=1)

    group_columns = ["decision_id", *DECISION_KEY_COLUMNS, "decision_date", "party"]
    counts = (
        work.groupby(group_columns + ["vote_state"], dropna=False)
        .size()
        .unstack("vote_state", fill_value=0)
        .reindex(columns=VOTE_STATES, fill_value=0)
        .reset_index()
        .rename(columns={state: f"{state}_n" for state in VOTE_STATES})
    )

    count_columns = [f"{state}_n" for state in VOTE_STATES]
    counts[count_columns] = counts[count_columns].astype(int)
    counts["present_n"] = counts["yes_n"] + counts["no_n"] + counts["abstain_n"]
    counts["member_records_n"] = counts[count_columns].sum(axis=1)

    present_columns = ["yes_n", "no_n", "abstain_n"]
    present_counts = counts[present_columns].to_numpy(dtype=int)
    leading_count = present_counts.max(axis=1)
    leading_ties = (present_counts == leading_count[:, None]).sum(axis=1)
    choice_names = np.asarray(["yes", "no", "abstain"], dtype=object)
    plurality_choice = choice_names[present_counts.argmax(axis=1)]

    counts["is_tie"] = (counts["present_n"] > 0) & (leading_ties > 1)
    counts["party_choice"] = np.where(
        counts["present_n"] == 0,
        "no_present",
        np.where(counts["is_tie"], "split", plurality_choice),
    )
    counts["has_dissent"] = (counts["present_n"] > 0) & (
        counts["present_n"] > leading_count
    )
    counts["cohesion"] = np.where(
        counts["present_n"] > 0,
        leading_count / counts["present_n"],
        np.nan,
    )

    return counts.sort_values(["decision_date", "decision_id", "party"]).reset_index(drop=True)


__all__ = ["DECISION_KEY_COLUMNS", "VOTE_STATES", "aggregate_party_choices"]