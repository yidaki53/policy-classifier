import numpy as np
import pandas as pd

from swedish_parliament_policy_classifier.analysis.action_evidence import (
    aggregate_party_choices,
)
from swedish_parliament_policy_classifier.analysis.contracts import StudySpecification


def test_aggregate_party_choices_preserves_vote_states_and_equal_decision_unit() -> None:
    votes = pd.DataFrame(
        {
            "rm": ["202526"] * 5,
            "beteckning": ["AU1"] * 5,
            "punkt": ["3"] * 5,
            "votering_id": ["vote-1"] * 5,
            "datum": pd.to_datetime(["2026-02-12"] * 5, utc=True),
            "parti": ["S"] * 5,
            "rost": ["Ja", "ja", "Ja", "Nej", "Frånvarande"],
        }
    )

    out = aggregate_party_choices(votes)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["decision_id"] == "202526|AU1|3|vote-1"
    assert row["party"] == "S"
    assert row["yes_n"] == 3
    assert row["no_n"] == 1
    assert row["abstain_n"] == 0
    assert row["absent_n"] == 1
    assert row["present_n"] == 4
    assert row["member_records_n"] == 5
    assert row["party_choice"] == "yes"
    assert row["has_dissent"]
    assert not row["is_tie"]
    assert row["cohesion"] == 0.75


def test_study_specification_uses_latest_complete_24_calendar_months() -> None:
    specification = StudySpecification(data_cutoff="2026-07-30T18:00:00Z")

    assert specification.primary_start_date.isoformat() == "2024-07-01"
    assert specification.primary_end_date.isoformat() == "2026-06-30"


def test_aggregate_party_choices_keeps_split_and_no_present_non_binary() -> None:
    votes = pd.DataFrame(
        {
            "rm": ["202526"] * 6,
            "beteckning": ["AU1"] * 6,
            "punkt": ["3"] * 6,
            "votering_id": ["vote-1"] * 6,
            "datum": pd.to_datetime(["2026-02-12"] * 6, utc=True),
            "parti": ["M", "M", "M", "M", "KD", "KD"],
            "rost": ["Ja", "Ja", "Nej", "Nej", "Frånvarande", "Frånvarande"],
        }
    )

    out = aggregate_party_choices(votes).set_index("party")

    assert out.loc["M", "party_choice"] == "split"
    assert out.loc["M", "is_tie"]
    assert out.loc["M", "cohesion"] == 0.5
    assert out.loc["KD", "party_choice"] == "no_present"
    assert not out.loc["KD", "is_tie"]
    assert np.isnan(out.loc["KD", "cohesion"])


def test_aggregate_party_choices_excludes_missing_party_labels() -> None:
    votes = pd.DataFrame(
        {
            "rm": ["202526", "202526"],
            "beteckning": ["AU1", "AU1"],
            "punkt": ["3", "3"],
            "votering_id": ["vote-1", "vote-1"],
            "datum": pd.to_datetime(["2026-02-12", "2026-02-12"], utc=True),
            "parti": ["S", None],
            "rost": ["Ja", "Frånvarande"],
        }
    )

    out = aggregate_party_choices(votes)

    assert out["party"].tolist() == ["S"]