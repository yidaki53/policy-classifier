"""Feature registry to compose scoring signals behind a small interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable


FeatureFn = Callable[[str], Dict[str, float]]


@dataclass
class FeatureRegistry:
    features: Dict[str, FeatureFn]

    def run(self, text: str) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for name, fn in self.features.items():
            out[name] = fn(text)
        return out


def merge_feature_outputs(outputs: Iterable[Dict[str, float]]) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    for o in outputs:
        for k, v in o.items():
            merged[k] = merged.get(k, 0.0) + float(v)
    return merged
