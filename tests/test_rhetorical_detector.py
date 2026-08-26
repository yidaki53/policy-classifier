"""Tests for the rhetorical pattern detector module."""

import pytest
from swedish_parliament_policy_classifier.nlp.rhetorical_detector import (
    detect_rhetorical_patterns,
    detect_rhetorical_patterns_with_metadata,
    load_rhetorical_weights,
)


class TestLoadRhetoricalWeights:
    def test_returns_defaults_when_no_file(self, tmp_path, monkeypatch):
        # Ensure weights file doesn't exist
        import swedish_parliament_policy_classifier.nlp.rhetorical_detector as mod

        mod._RHETORICAL_WEIGHTS = None
        monkeypatch.chdir(tmp_path)
        weights = load_rhetorical_weights()
        assert "base_far_left" in weights
        assert "inc_far_left" in weights
        assert weights["base_far_left"] == 1.20

    def test_loads_from_custom_path(self, tmp_path, monkeypatch):
        import json
        from swedish_parliament_policy_classifier.nlp.rhetorical_detector import _RHETORICAL_WEIGHTS
        
        custom_weights = {
            "base_far_left": 2.0,
            "inc_far_left": 0.5,
            "base_left": 1.5,
        }
        weights_file = tmp_path / "custom_weights.json"
        weights_file.write_text(json.dumps({"params": custom_weights}))
        
        # Reset cache to force reload
        import swedish_parliament_policy_classifier.nlp.rhetorical_detector as mod
        mod._RHETORICAL_WEIGHTS = None
        
        monkeypatch.chdir(tmp_path)
        weights = load_rhetorical_weights(weights_path=weights_file)
        assert weights["base_far_left"] == 2.0
        assert weights["inc_far_left"] == 0.5


class TestDetectRhetoricalPatterns:
    def test_empty_text_returns_empty(self):
        result = detect_rhetorical_patterns("")
        assert result == {}

    def test_none_text_returns_empty(self):
        result = detect_rhetorical_patterns(None)
        assert result == {}

    def test_far_left_signals(self):
        text = "Vi behöver en revolution för att omstörta kapitalismen och bygga kollektivt ägande."
        result = detect_rhetorical_patterns(text)
        assert result["far_left"] > 0

    def test_left_signals(self):
        text = "Vi måste höja skatten och stärka välfärden för vanliga människor."
        result = detect_rhetorical_patterns(text)
        assert result["left"] > 0

    def test_centre_left_signals(self):
        text = "Vi behöver investera i utbildning, sjukvård och hållbarhet."
        result = detect_rhetorical_patterns(text)
        assert result["centre_left"] > 0

    def test_right_signals(self):
        text = "Vi måste sänka skatter och stärka försvaret för svensk säkerhet."
        result = detect_rhetorical_patterns(text)
        assert result["right"] > 0

    def test_far_right_signals(self):
        text = "Vi måste stoppa massinvandring och försvara svenska värden."
        result = detect_rhetorical_patterns(text)
        assert result["far_right"] > 0

    def test_multiple_signals_accumulate(self):
        text = "kapitalism kapitalism kapitalism borgarklass klasskamp"
        result = detect_rhetorical_patterns(text)
        # Multiple signals should give higher score
        assert result["far_left"] > 1.0

    def test_compound_pattern_mining_plus_environment(self):
        text = "Vi behöver utveckla miljö och biologisk mångfald genom gruvdrift som är positivt för näringslivet."
        result = detect_rhetorical_patterns(text)
        assert result["right"] > 0

    def test_all_categories_present_in_output(self):
        text = "Some neutral text about politics."
        result = detect_rhetorical_patterns(text)
        expected_cats = ["far_left", "left", "centre_left", "centre", "centre_right", "right", "far_right"]
        for cat in expected_cats:
            assert cat in result


class TestDetectRhetoricalPatternsWithMetadata:
    def test_metadata_for_neutral_text(self):
        text = "This is a neutral statement about parliamentary procedure."
        result = detect_rhetorical_patterns_with_metadata(text)
        assert "adjustments" in result
        assert "total_signals" in result
        assert "dominant_category" in result
        assert "is_rhetorical" in result
        assert result["is_rhetorical"] is False

    def test_metadata_for_rhetorical_text(self):
        text = "Vi behöver en revolution för att omstörta kapitalismen."
        result = detect_rhetorical_patterns_with_metadata(text)
        assert result["is_rhetorical"] is True
        assert result["total_signals"] > 0
        assert result["dominant_category"] == "far_left"