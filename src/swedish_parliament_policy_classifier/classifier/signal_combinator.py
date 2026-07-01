"""Signal combination and weight normalization for multi-source classification.

This module extracts the Fraction-based exact arithmetic weight combination
from the legacy scorer to enable:
- Independent testing of signal fusion logic
- Transparent weight normalization audit trail
- Swappable combination strategies (linear, learned, ensemble)

The combinator normalizes keyword, embedding, zero-shot, Ollama, and BERT
signals into a single probability distribution using exact rational arithmetic
to avoid floating-point drift.
"""

import logging
from fractions import Fraction
from typing import Dict, List, Optional, Union

LOG = logging.getLogger(__name__)


def normalize_signal_scores(scores: Dict[str, float]) -> Dict[str, Fraction]:
    """Normalize raw signal scores to fractions summing to 1.
    
    Args:
        scores: Dictionary mapping category names to raw float scores.
    
    Returns:
        Dictionary mapping category names to Fraction weights normalized to sum=1.
    """
    if not scores:
        return {}
    
    total = sum(scores.values())
    if total <= 0:
        return {k: Fraction(0, 1) for k in scores.keys()}
    
    return {k: Fraction(v) / Fraction(total) for k, v in scores.items()}


def compute_weighted_combination(
    keyword_norm: Dict[str, Fraction],
    embedding_norm: Dict[str, Fraction],
    zero_shot_norm: Dict[str, Fraction],
    ollama_norm: Optional[Dict[str, Fraction]] = None,
    bert_norm: Optional[Dict[str, Fraction]] = None,
    kw_weight: float = 0.3,
    emb_weight: float = 0.4,
    zs_weight: float = 0.3,
    oll_weight: float = 0.0,
    bert_weight: float = 0.0,
) -> Dict[str, Fraction]:
    """Combine multiple normalized signal distributions into a single distribution.
    
    All arithmetic uses Fraction for exactness. Output distribution sums to 1.
    
    Args:
        keyword_norm: Normalized keyword scores (Fraction).
        embedding_norm: Normalized embedding scores (Fraction).
        zero_shot_norm: Normalized zero-shot scores (Fraction).
        ollama_norm: Optional normalized Ollama scores (Fraction).
        bert_norm: Optional normalized BERT scores (Fraction).
        kw_weight: Weight for keyword signal (0-1).
        emb_weight: Weight for embedding signal (0-1).
        zs_weight: Weight for zero-shot signal (0-1).
        oll_weight: Weight for Ollama signal (0-1).
        bert_weight: Weight for BERT signal (0-1).
    
    Returns:
        Combined normalized distribution as Dict[str, Fraction].
    """
    # Convert weights to Fractions with limited denominators for efficiency
    _kw_w = Fraction(kw_weight).limit_denominator(100)
    _emb_w = Fraction(emb_weight).limit_denominator(100)
    _zs_w = Fraction(zs_weight).limit_denominator(100)
    _oll_w = Fraction(oll_weight).limit_denominator(100)
    _bert_w = Fraction(bert_weight).limit_denominator(100)
    
    # Get all category names from all inputs
    all_categories = set(keyword_norm.keys()) | set(embedding_norm.keys()) | set(zero_shot_norm.keys())
    if ollama_norm:
        all_categories |= set(ollama_norm.keys())
    if bert_norm:
        all_categories |= set(bert_norm.keys())
    
    # Weighted combination
    combined = {}
    for cat in all_categories:
        val = (
            _kw_w * keyword_norm.get(cat, Fraction(0, 1))
            + _emb_w * embedding_norm.get(cat, Fraction(0, 1))
            + _zs_w * zero_shot_norm.get(cat, Fraction(0, 1))
        )
        if ollama_norm:
            val += _oll_w * ollama_norm.get(cat, Fraction(0, 1))
        if bert_norm:
            val += _bert_w * bert_norm.get(cat, Fraction(0, 1))
        combined[cat] = val
    
    # Final normalization to sum=1
    total = sum(combined.values())
    if total > 0:
        combined = {k: v / total for k, v in combined.items()}
    
    return combined


def apply_rhetorical_adjustments(
    distribution: Dict[str, Fraction],
    adjustments: Dict[str, float],
    boost_factor: float = 2.0,
) -> Dict[str, Fraction]:
    """Apply rhetorical pattern adjustments to a probability distribution.
    
    Rhetorical signals get a multiplicative boost to override motion-trained
    biases when processing speeches.
    
    Args:
        distribution: Current probability distribution (Fraction).
        adjustments: Rhetorical adjustment floats per category.
        boost_factor: Multiplier for positive adjustments (default 2.0).
    
    Returns:
        Adjusted and re-normalized distribution.
    """
    if not adjustments or all(v == 0.0 for v in adjustments.values()):
        return distribution
    
    adjusted = {}
    for cat, prob in distribution.items():
        adj = adjustments.get(cat, 0.0)
        if adj > 0:
            # Boost categories with rhetorical signals
            boost = Fraction(boost_factor).limit_denominator(100) + Fraction(adj).limit_denominator(100)
            adjusted[cat] = prob * boost
        else:
            adjusted[cat] = prob
    
    # Re-normalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
    
    return adjusted


class SignalCombinator:
    """Stateful combinator for incremental signal combination.
    
    Useful when signals arrive asynchronously or when you need to inspect
    intermediate distributions.
    """
    
    def __init__(self):
        self.signals: Dict[str, Dict[str, float]] = {}
        self.weights: Dict[str, float] = {}
    
    def add_signal(self, name: str, scores: Dict[str, float], weight: float):
        """Add a named signal distribution.
        
        Args:
            name: Signal identifier (e.g., 'keyword', 'embedding', 'zero_shot').
            scores: Raw category scores.
            weight: Importance weight for this signal.
        """
        self.signals[name] = scores
        self.weights[name] = weight
    
    def combine(self) -> Dict[str, Fraction]:
        """Combine all added signals using stored weights.
        
        Returns:
            Final normalized distribution. Returns empty dict if no signals added.
        """
        if not self.signals:
            return {}
        
        # Normalize each signal
        normalized = {}
        for name, scores in self.signals.items():
            normalized[name] = normalize_signal_scores(scores)
        
        # Extract individual distributions with defaults
        kw_norm = normalized.get('keyword', {})
        emb_norm = normalized.get('embedding', {})
        zs_norm = normalized.get('zero_shot', {})
        oll_norm = normalized.get('ollama')
        bert_norm = normalized.get('bert')
        
        # Get weights with defaults
        kw_w = self.weights.get('keyword', 0.0)
        emb_w = self.weights.get('embedding', 0.0)
        zs_w = self.weights.get('zero_shot', 0.0)
        oll_w = self.weights.get('ollama', 0.0)
        bert_w = self.weights.get('bert', 0.0)
        
        return compute_weighted_combination(
            kw_norm, emb_norm, zs_norm,
            ollama_norm=oll_norm,
            bert_norm=bert_norm,
            kw_weight=kw_w,
            emb_weight=emb_w,
            zs_weight=zs_w,
            oll_weight=oll_w,
            bert_weight=bert_w,
        )
    
    def get_intermediate_distribution(self, signal_name: str) -> Optional[Dict[str, Fraction]]:
        """Get normalized distribution for a specific signal.
        
        Args:
            signal_name: Name of the signal to inspect.
        
        Returns:
            Normalized distribution or None if signal not found.
        """
        if signal_name not in self.signals:
            return None
        return normalize_signal_scores(self.signals[signal_name])