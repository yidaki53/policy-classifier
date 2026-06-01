from swedish_parliament_policy_classifier.exports import load_definitions, score_motion
from swedish_parliament_policy_classifier.classifier.deep_scoring_service import DeepScoringService
import swedish_parliament_policy_classifier.classifier.deep_scoring_service as deep_scoring_service

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


def test_speech_service_forwards_meta_learner(monkeypatch):
    sentinel_meta = {"clf": object(), "label_encoder": object()}
    captured = {}

    def fake_score_motion(*, motion_id, text, categories, embedding_matcher, topic_distributions, meta_clf, use_zero_shot, skip_policy_extraction, use_speech_preprocessing, use_ollama):
        captured.update(
            motion_id=motion_id,
            text=text,
            categories=categories,
            embedding_matcher=embedding_matcher,
            topic_distributions=topic_distributions,
            meta_clf=meta_clf,
            use_zero_shot=use_zero_shot,
            skip_policy_extraction=skip_policy_extraction,
            use_speech_preprocessing=use_speech_preprocessing,
            use_ollama=use_ollama,
        )
        return ["sentinel-result"]

    monkeypatch.setattr(deep_scoring_service, "score_motion", fake_score_motion)

    service = DeepScoringService(
        categories={},
        embedding_matcher="matcher",
        topic_distributions={"topic": [0.1]},
        meta_clf=sentinel_meta,
    )

    result = service.classify("speech-1", "Vi talar om arbeten och välfärd")

    assert result == ["sentinel-result"]
    assert captured["motion_id"] == "speech-1"
    assert captured["text"] == "Vi talar om arbeten och välfärd"
    assert captured["meta_clf"] is sentinel_meta
    assert captured["skip_policy_extraction"] is True
    assert captured["use_speech_preprocessing"] is True
    assert captured["use_zero_shot"] is True
    assert captured["use_ollama"] is False
