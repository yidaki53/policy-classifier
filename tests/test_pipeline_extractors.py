from swedish_parliament_policy_classifier.classifier.scorer import (
    _extract_speech_argumentative_text,
    _sentence_stance,
)


def test_sentence_stance_basic():
    s1 = "Vi vill sänka skatterna och stärka utbildningen."
    assert _sentence_stance(s1) == "own_position"

    s2 = "Ni säger att skatterna måste höjas för att balansera budgeten."
    assert _sentence_stance(s2) in ("opponent_report", "neutral")


def test_extract_speech_argumentative_text_returns_own_position():
    text = (
        "Moderaterna säger att skatterna bör höjas. "
        "Vi anser att skatterna ska sänkas och att skolorna bör privatiseras. "
        "Detta är en viktig reform."
    )
    out = _extract_speech_argumentative_text(text, max_chars=500)
    assert isinstance(out, str)
    assert "Vi anser" in out or "Vi" in out