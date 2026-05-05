from swedish_parliament_policy_classifier.classifier.scorer import load_definitions, score_motion


def test_score_simple_text():
    cats = load_definitions()
    results = score_motion("sample-1", "Vi vill sänka skatter och privatisera skolor", cats)

    # Expect 'right' category to have non-zero weight
    right = [r for r in results if r.category == "right"][0]
    assert right.raw_score >= 1.0
    assert 0.0 <= right.normalized_weight <= 1.0
