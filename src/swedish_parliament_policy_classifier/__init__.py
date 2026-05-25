"""Package initializer for ``swedish_parliament_policy_classifier``.

This file makes the package behave as a namespace during development while
also allowing top-level repository packages (e.g. ``classifier``, ``db``)
to be importable as subpackages of
``swedish_parliament_policy_classifier``. This keeps the ``src/`` layout for
packaging while making tests and local imports work.
"""
from __future__ import annotations

from pkgutil import extend_path

# Keep namespace capability for packaging tools. Do NOT append the repo
# root to `__path__` — that created ambiguous imports where top-level
# packages in the repo could be resolved as
# `swedish_parliament_policy_classifier.<submodule>` which breaks tests.
__path__ = extend_path(__path__, __name__)

__all__ = []
