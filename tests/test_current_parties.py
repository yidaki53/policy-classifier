import swedish_parliament_policy_classifier.visualization.style_config as sc


def test_current_parties_non_empty():
    assert sc.CURRENT_PARTIES, "CURRENT_PARTIES is empty"


def test_current_parties_disjoint_from_bad_values():
    bad = {"", "NYD", "Unknown", "None", "nan"}
    assert not sc.CURRENT_PARTIES & bad, f"CURRENT_PARTIES contains bad values: {sc.CURRENT_PARTIES & bad}"


def test_party_labels_cover_current_parties():
    missing = {p for p in sc.CURRENT_PARTIES if p not in sc.PARTY_LABELS}
    assert not missing, f"Missing party labels for: {missing}"


def test_party_colors_cover_current_parties():
    missing = {p for p in sc.CURRENT_PARTIES if p not in sc.PARTY_COLORS_PLOT}
    assert not missing, f"Missing party colors for: {missing}"