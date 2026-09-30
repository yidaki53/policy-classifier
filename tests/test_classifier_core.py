from swedish_parliament_policy_classifier.classifier.core import ClassifierCore


def test_classifier_core_dispatches_by_mode() -> None:
    calls = []

    def motion_scorer(**kwargs):
        calls.append(("motion", kwargs))
        return []

    def speech_scorer(**kwargs):
        calls.append(("speech", kwargs))
        return []

    core = ClassifierCore(motion_scorer=motion_scorer, speech_scorer=speech_scorer)

    core.classify("m1", "motion text", {}, mode="motion")
    core.classify("s1", "speech text", {}, mode="speech")

    assert [kind for kind, _ in calls] == ["motion", "speech"]
    assert calls[1][1]["speech_id"] == "s1"