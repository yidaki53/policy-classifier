from swedish_parliament_policy_classifier.analysis.ideology_axes import canonical_axis_order


def test_canonical_axis_order_is_seven_categories():
    axis = canonical_axis_order()
    assert len(axis) == 7
    assert axis == [
        "far_left",
        "left",
        "centre_left",
        "centre",
        "centre_right",
        "right",
        "far_right",
    ]
