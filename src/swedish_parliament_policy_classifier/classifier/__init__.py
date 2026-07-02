"""Re-exports for the `classifier` package to aid static analysis.

Expose persistence helpers so tools operating on the `src/` package
layout can resolve cross-module references more reliably.
"""

from swedish_parliament_policy_classifier.classifier.persist_parquet import (
    persist_classification,
    persist_classifications_batch,
    save_annotation,
    get_next_unlabeled_motion,
)
from swedish_parliament_policy_classifier.classifier.persist_parquet import (
    record_lineage_parquet as record_lineage,
)
from swedish_parliament_policy_classifier.classifier.scorer import score_motion
from swedish_parliament_policy_classifier.definitions.loader import load_verified_definitions as load_definitions

__all__ = [
    "record_lineage",
    "persist_classification",
    "persist_classifications_batch",
    "save_annotation",
    "get_next_unlabeled_motion",
    "load_definitions",
    "score_motion",
]
