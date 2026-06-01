"""Feature specification utilities for the ensemble meta-classifier.

Provides a stable programmatic way to generate feature names used by
`build_feature_vector` so callers can validate and inspect feature order
without duplicating string logic across the codebase.
"""
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FeatureSpec:
    category_names: List[str]
    max_topics: int = 100

    def feature_names(self) -> List[str]:
        names: List[str] = []
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


def get_feature_names(category_names: List[str], max_topics: int = 100) -> List[str]:
    return FeatureSpec(category_names=category_names, max_topics=max_topics).feature_names()
