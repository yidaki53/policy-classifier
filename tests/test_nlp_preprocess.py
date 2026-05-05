import pytest

from swedish_parliament_policy_classifier.nlp.preprocess import preprocess_text


def test_preprocess_empty():
    res = preprocess_text("")
    assert res["text"] == ""
    assert res["tokens"] == []
    assert res["lemmas"] == []


def test_preprocess_basic():
    text = "Det här är ett TEST."
    res = preprocess_text(text, normalize=True)
    assert isinstance(res, dict)
    assert "tokens" in res and "lemmas" in res
    assert isinstance(res["tokens"], list)
    assert isinstance(res["lemmas"], list)
    assert len(res["tokens"]) == len(res["lemmas"]) 
    # tokens should be lowercased in the output
    for t in res["tokens"]:
        assert t == t.lower()
    for l in res["lemmas"]:
        assert isinstance(l, str)
