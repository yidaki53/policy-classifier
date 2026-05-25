"""Re-exports for the `db` package to help static extractors resolve helpers.

Expose `get_connection` and `init_db` from the repository's `db.schema`
module so cross-module references appear under the `src.` package namespace.
"""

from swedish_parliament_policy_classifier.db.schema import get_connection, init_db

__all__ = ["get_connection", "init_db"]
