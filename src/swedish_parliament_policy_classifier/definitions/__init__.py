"""Re-exports for the `definitions` package to aid static analysis.

These lightweight re-exports expose the primary loader/verification API so
tools that operate on the `src/` package layout can resolve cross-module
references more reliably.
"""

from swedish_parliament_policy_classifier.definitions.loader import (
    load_verified_definitions,
    verify,
    _compute_checksum,
    _read_stored_checksum,
)

__all__ = ["load_verified_definitions", "verify", "_compute_checksum", "_read_stored_checksum"]

# Graphify import hint to link package-level re-exports to the top-level
# implementation file. Inert at runtime; helps semantic extractors.
if False:
    from definitions.loader import load_verified_definitions as _impl_load_verified_definitions
