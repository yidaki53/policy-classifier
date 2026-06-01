import pandas as pd

from swedish_parliament_policy_classifier.analysis.manuscript_exports import _filter_overlay_profiles


def test_filter_overlay_profiles_removes_requested_parties():
    profiles = pd.DataFrame(
        {
            "party": ["M", "Moderaterna", "V", "Vänsterpartiet", "Unknown", "X"],
            "modality": ["speech"] * 6,
            "category": ["right"] * 6,
            "proportion": [1.0] * 6,
        }
    )

    out = _filter_overlay_profiles(profiles)

    assert sorted(out["party"].unique().tolist()) == ["M", "V"]