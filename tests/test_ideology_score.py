from math import isclose

from swedish_parliament_policy_classifier.visualization.style_config import (
    compute_ideology_score_from_proportions,
)


def test_center_is_neutral_but_dilutes_extremity() -> None:
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

    assert isclose(score, -0.2, rel_tol=0.0, abs_tol=1e-9)


def test_left_right_inversion_is_symmetric() -> None:
    left = {"left": 0.6, "centre": 0.2, "right": 0.2}
    right = {"left": 0.2, "centre": 0.2, "right": 0.6}

    assert isclose(
        compute_ideology_score_from_proportions(left),
        -compute_ideology_score_from_proportions(right),
        rel_tol=0.0,
        abs_tol=1e-9,
    )
