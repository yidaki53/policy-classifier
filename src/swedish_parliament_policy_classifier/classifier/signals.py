"""Typed signal outputs shared by classifier strategies and score assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping


@dataclass(frozen=True)
class SignalOutput:
    """Scores and evidence emitted by one classifier signal."""

    name: str
    scores: Mapping[str, float] = field(default_factory=dict)
    evidence: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def score_map(self) -> dict[str, float]:
        """Return a mutable score map for model adapters and feature builders."""
        return {str(category): float(score) for category, score in self.scores.items()}


def extract_signal_output(
    results: Iterable[object],
    *,
    name: str,
    evidence_prefix: str,
) -> SignalOutput:
    """Extract one signal from the legacy classification-result evidence."""
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    prefix = f"{evidence_prefix}:"

    for result in results:
        category = str(getattr(result, "category", ""))
        for rule in getattr(result, "matched_rules", []) or []:
            if not rule.startswith(prefix):
                continue
            try:
                score = float(rule[len(prefix):])
            except (TypeError, ValueError):
                continue
            scores[category] = score
            evidence.setdefault(category, []).append(rule)

    return SignalOutput(
        name=name,
        scores=scores,
        evidence={category: tuple(rules) for category, rules in evidence.items()},
    )


__all__ = ["SignalOutput", "extract_signal_output"]