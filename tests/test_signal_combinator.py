"""Tests for the signal combinator module."""

import pytest
from fractions import Fraction
from swedish_parliament_policy_classifier.classifier.signal_combinator import (
    normalize_signal_scores,
    compute_weighted_combination,
    apply_rhetorical_adjustments,
    SignalCombinator,
)


class TestNormalizeSignalScores:
    def test_empty_input_returns_empty(self):
        result = normalize_signal_scores({})
        assert result == {}

    def test_single_category(self):
        result = normalize_signal_scores({"cat_a": 10.0})
        assert result["cat_a"] == Fraction(1, 1)

    def test_multiple_categories_normalize_to_one(self):
        scores = {"cat_a": 3.0, "cat_b": 7.0}
        result = normalize_signal_scores(scores)
        assert result["cat_a"] == Fraction(3, 10)
        assert result["cat_b"] == Fraction(7, 10)
        total = sum(result.values())
        assert total == Fraction(1, 1)

    def test_zero_total_returns_zeros(self):
        result = normalize_signal_scores({"cat_a": 0.0, "cat_b": 0.0})
        assert result["cat_a"] == Fraction(0, 1)
        assert result["cat_b"] == Fraction(0, 1)


class TestComputeWeightedCombination:
    def test_basic_combination(self):
        kw_norm = {"a": Fraction(1, 2), "b": Fraction(1, 2)}
        emb_norm = {"a": Fraction(1, 4), "b": Fraction(3, 4)}
        zs_norm = {"a": Fraction(3, 4), "b": Fraction(1, 4)}
        
        result = compute_weighted_combination(
            kw_norm, emb_norm, zs_norm,
            kw_weight=0.5, emb_weight=0.25, zs_weight=0.25
        )
        
        # Should produce valid probabilities
        total = sum(result.values())
        assert total == Fraction(1, 1)
        for v in result.values():
            assert v >= 0

    def test_with_optional_signals(self):
        kw_norm = {"a": Fraction(1, 1)}
        emb_norm = {}
        zs_norm = {"a": Fraction(1, 2), "b": Fraction(1, 2)}
        oll_norm = {"a": Fraction(1, 3), "b": Fraction(2, 3)}
        
        result = compute_weighted_combination(
            kw_norm, emb_norm, zs_norm,
            ollama_norm=oll_norm,
            kw_weight=0.4, emb_weight=0.0, zs_weight=0.3, oll_weight=0.3
        )
        
        assert "b" in result
        total = sum(result.values())
        assert total == Fraction(1, 1)

    def test_equal_weights_produce_equal_probabilities(self):
        kw_norm = {"a": Fraction(1, 2), "b": Fraction(1, 2)}
        emb_norm = {"a": Fraction(1, 2), "b": Fraction(1, 2)}
        zs_norm = {"a": Fraction(1, 2), "b": Fraction(1, 2)}
        
        result = compute_weighted_combination(
            kw_norm, emb_norm, zs_norm,
            kw_weight=1/3, emb_weight=1/3, zs_weight=1/3
        )
        
        # With equal signals and equal weights, should be close to uniform
        assert abs(float(result["a"]) - 0.5) < 0.01
        assert abs(float(result["b"]) - 0.5) < 0.01


class TestApplyRhetoricalAdjustments:
    def test_no_adjustments_returns_original(self):
        dist = {"a": Fraction(1, 2), "b": Fraction(1, 2)}
        result = apply_rhetorical_adjustments(dist, {})
        assert result == dist

    def test_positive_adjustments_boost_categories(self):
        dist = {"a": Fraction(1, 2), "b": Fraction(1, 2)}
        adjustments = {"a": 0.5}
        result = apply_rhetorical_adjustments(dist, adjustments, boost_factor=2.0)
        
        # Category 'a' should be boosted
        assert result["a"] > Fraction(1, 2)
        total = sum(result.values())
        assert total == Fraction(1, 1)

    def test_zero_adjustments_unchanged(self):
        dist = {"a": Fraction(1, 2), "b": Fraction(1, 2)}
        adjustments = {"a": 0.0, "b": 0.0}
        result = apply_rhetorical_adjustments(dist, adjustments)
        assert result == dist


class TestSignalCombinator:
    def test_empty_combinator_returns_empty(self):
        combinator = SignalCombinator()
        result = combinator.combine()
        assert result == {}

    def test_incremental_signal_addition(self):
        combinator = SignalCombinator()
        combinator.add_signal("keyword", {"a": 10.0, "b": 5.0}, 0.5)
        combinator.add_signal("embedding", {"a": 7.0, "b": 3.0}, 0.5)
        
        result = combinator.combine()
        total = sum(result.values())
        assert total == Fraction(1, 1)

    def test_get_intermediate_distribution(self):
        combinator = SignalCombinator()
        combinator.add_signal("keyword", {"a": 10.0, "b": 5.0}, 0.5)
        
        intermediate = combinator.get_intermediate_distribution("keyword")
        assert intermediate is not None
        assert intermediate["a"] == Fraction(2, 3)
        assert intermediate["b"] == Fraction(1, 3)

    def test_get_missing_distribution_returns_none(self):
        combinator = SignalCombinator()
        assert combinator.get_intermediate_distribution("nonexistent") is None