"""Feature specification utilities for the ensemble meta-classifier.

Provides a stable programmatic way to generate feature names used by
`build_feature_vector` so callers can validate and inspect feature order
without duplicating string logic across the codebase.
"""
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class FeatureSpec:
    category_names: Sequence[str]
    max_topics: int = 100
    schema_version: str = "features-v1"

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for cat in self.category_names:
            names.append(f"kw_{cat}")
        for cat in self.category_names:
            names.append(f"emb_{cat}")
        for cat in self.category_names:
            names.append(f"zs_{cat}")
        for cat in self.category_names:
            names.append(f"bert_cls_{cat}")
        for i in range(self.max_topics):
            names.append(f"topic_{i}")
        names.extend(["rhet_irony", "rhet_sarcasm", "rhet_posturing", "rhet_none"])
        names.extend(["text_len_log", "recency_years", "doc_mot", "doc_prop", "doc_votering"])
        return names

    def validate_feature_names(self, names: Sequence[str]) -> None:
        expected = self.feature_names()
        if list(names) != expected:
            raise ValueError(
                f"Feature schema mismatch for {self.schema_version}: "
                f"expected {len(expected)} ordered features, received {len(names)}"
            )


def get_feature_names(category_names: Sequence[str], max_topics: int = 100) -> list[str]:
    return FeatureSpec(category_names=category_names, max_topics=max_topics).feature_names()
