from swedish_parliament_policy_classifier.exports import load_definitions, score_motion
from swedish_parliament_policy_classifier.models import CategoryDef
from swedish_parliament_policy_classifier.classifier import scorer

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

    assert results[0].provenance is not None
    assert "keyword" in results[0].provenance.signals_requested
    assert results[0].provenance.pipeline_version == results[0].classifier_version


def test_empty_text_returns_uniform_degraded_distribution() -> None:
    results = score_motion("empty", "", load_definitions(), use_zero_shot=False)

    assert sum(result.normalized_weight for result in results) == 1.0
    assert results[0].provenance is not None
    assert results[0].provenance.degraded is True
    assert "no_signal" in results[0].provenance.signals_failed


def test_lemma_cache_changes_when_category_keywords_change(monkeypatch) -> None:
    monkeypatch.setattr(scorer, "_get_spacy", lambda: None)
    scorer._LEMMA_KW_INDEX = None
    scorer._LEMMA_KW_INDEX_KEY = None
    first = {"left": CategoryDef(name="left", keywords=["unique-term"])}
    second = {"right": CategoryDef(name="right", keywords=["other-term"])}

    first_results = scorer.score_motion(
        "cache-1", "unique-term", first, use_zero_shot=False, skip_policy_extraction=True
    )
    second_results = scorer.score_motion(
        "cache-2", "unique-term", second, use_zero_shot=False, skip_policy_extraction=True
    )

    assert first_results[0].raw_score == 1.0
    assert second_results[0].raw_score == 0.0
    assert second_results[0].normalized_weight == 1.0


def test_classification_is_party_independent() -> None:
    categories = load_definitions()
    text = "Vi vill stärka välfärden och införa tydliga regler för skolan och vården."
    outputs = []
    for party in ("V", "S", "M", "SD", None):
        results = score_motion(
            "party-independent",
            text,
            categories,
            party=party,
            use_zero_shot=False,
        )
        outputs.append([(result.category, result.raw_score, result.normalized_weight) for result in results])

    assert all(output == outputs[0] for output in outputs[1:])
