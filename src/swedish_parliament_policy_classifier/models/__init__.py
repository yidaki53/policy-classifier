"""Re-exports for `models` to expose core Pydantic types for static tools.

This module first tries to import the implementation from the packaged
location `swedish_parliament_policy_classifier.models.models`. In local
development (src/ layout) the project's top-level `models` package may live
at the repository root rather than under `src/`. In that case, fall back to
importing from the top-level `models` package so imports succeed both for
editable installs and for direct local test runs.
"""

from .models import (
    CategoryDef,
    ClassificationResult,
    NormalizedMotion,
    RawMotion,
    PartyProfile,
)

__all__ = [
    "CategoryDef",
    "ClassificationResult",
    "NormalizedMotion",
    "RawMotion",
    "PartyProfile",
]
