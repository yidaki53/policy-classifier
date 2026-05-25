"""Re-exports for the `classifier` package to aid static analysis.

Expose persistence helpers so tools operating on the `src/` package
layout can resolve cross-module references more reliably.
"""

# Prefer the top-level `classifier` implementation modules (repo-root layout)
# during local development. This avoids importing the src-level shim modules
# which intentionally reference the canonical `exports` module (causing
# circular imports at runtime). Fall back to the src-packaged modules only
# if the top-level packages are not present.
try:
    from classifier.persist import (
        record_lineage,
        persist_classification,
        persist_classifications_batch,
        save_annotation,
        get_annotation_by_motion,
        get_next_unlabeled_motion,
    )
except Exception:
    from swedish_parliament_policy_classifier.classifier.persist import (
        record_lineage,
        persist_classification,
        persist_classifications_batch,
        save_annotation,
        get_annotation_by_motion,
        get_next_unlabeled_motion,
    )

try:
    # Prefer top-level implementation for runtime
    from classifier.scorer import score_motion
    from definitions.loader import load_verified_definitions as load_definitions
except Exception:
    from swedish_parliament_policy_classifier.classifier.scorer import score_motion
    from definitions.loader import load_verified_definitions as load_definitions

__all__ = [
    "record_lineage",
    "persist_classification",
    "persist_classifications_batch",
    "save_annotation",
    "get_annotation_by_motion",
    "get_next_unlabeled_motion",
    "load_definitions",
    "score_motion",
]
