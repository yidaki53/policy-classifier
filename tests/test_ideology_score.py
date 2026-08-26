from math import isclose

from swedish_parliament_policy_classifier.visualization.style_config import (
    compute_ideology_score_from_proportions,
)


def test_center_is_treated_as_neutral_without_diluting_left_right_balance() -> None:
    proportions = {
        "far_left": 0.0,
        "left": 0.0,
        "centre_left": 0.3,
        "centre": 0.6,
        "centre_right": 0.0,
        "right": 0.1,
        "far_right": 0.0,
    }

    score = compute_ideology_score_from_proportions(proportions)

    assert isclose(score, -0.5, rel_tol=0.0, abs_tol=1e-9)
