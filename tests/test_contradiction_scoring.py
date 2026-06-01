from swedish_parliament_policy_classifier.analysis.contradiction_scoring import _source_confidence


def test_source_confidence_bounds_and_order():
    a = _source_confidence("existing:betankande_ref_dok_id")
    b = _source_confidence("fallback:party_category_vote")
    c = _source_confidence("fallback:vote_any")
    d = _source_confidence("unknown")

    assert 0.0 <= d <= 1.0
    assert a > b > c > d
