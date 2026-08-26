import pandas as pd

from swedish_parliament_policy_classifier.analysis.contracts import StudySpecification
from swedish_parliament_policy_classifier.analysis.party_position import (
    estimate_supported_action_positions,
)


def test_estimate_supported_action_positions_does_not_invert_no_votes() -> None:
    choices = pd.DataFrame(
        {
            "decision_id": ["d1", "d1", "d2", "d2"],
            "party": ["S", "M", "S", "M"],
            "decision_date": pd.to_datetime(
                ["2026-01-10", "2026-01-10", "2026-02-10", "2026-02-10"],
                utc=True,
            ),
            "party_choice": ["yes", "no", "yes", "yes"],
        }
    )
    probabilities = pd.DataFrame(
        {
            "decision_id": ["d1", "d1", "d2"],
            "category": ["far_left", "left", "right"],
            "probability": [0.5, 0.5, 1.0],
        }
    )

    out = estimate_supported_action_positions(choices, probabilities)

    s = out.set_index("party").loc["S"]
    m = out.set_index("party").loc["M"]
    assert s["supported_decision_n"] == 2
    assert s["score_0_100"] == 45.833333333333336
    assert m["supported_decision_n"] == 1
    assert m["score_0_100"] == 83.33333333333334


def test_estimate_supported_action_positions_applies_complete_month_window() -> None:
    choices = pd.DataFrame(
        {
            "decision_id": ["old", "first", "last", "partial"],
            "party": ["S"] * 4,
            "decision_date": pd.to_datetime(
                ["2024-06-30", "2024-07-01", "2026-06-30", "2026-07-01"],
                utc=True,
            ),
            "party_choice": ["yes"] * 4,
        }
    )
    probabilities = pd.DataFrame(
        {
            "decision_id": ["old", "first", "last", "partial"],
            "category": ["far_left", "centre", "far_right", "far_left"],
            "probability": [1.0] * 4,
        }
    )
    specification = StudySpecification(data_cutoff="2026-07-30T18:00:00Z")

    out = estimate_supported_action_positions(
        choices,
        probabilities,
        specification=specification,
    )

    assert out.iloc[0]["supported_decision_n"] == 2
    assert out.iloc[0]["score_0_100"] == 75.0