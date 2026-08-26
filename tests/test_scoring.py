from swedish_parliament_policy_classifier.exports import load_definitions, score_motion

if False:
    # Graphify hint: tests exercise the verified loader and the CategoryDef model;
    # anchor direct implementation imports for AST linking.
    from definitions.loader import load_verified_definitions as _hint_load_verified_definitions
    from models.models import CategoryDef as _hint_CategoryDef


def test_score_simple_text():
    cats = load_definitions()
    results = score_motion("sample-1", "Vi vill sänka skatter och privatisera skolor", cats)

    # Expect 'right' category to have non-zero weight
    right = [r for r in results if r.category == "right"][0]
    assert right.raw_score >= 1.0
    assert 0.0 <= right.normalized_weight <= 1.0
