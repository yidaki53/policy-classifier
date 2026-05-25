# Ensure src/ is on sys.path for src-layout import resolution
import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(REPO_ROOT, 'src')
# Always prefer the `src/` layout during tests so the packaged `src/`
# implementation (which contains `exports.py`) is used instead of the
# repository-root top-level package. Insert `src/` at the front of
# `sys.path` so imports resolve consistently during pytest runs.
if REPO_ROOT not in sys.path:
    # Insert repo root so top-level packages (e.g., `classifier`, `models`, `definitions`)
    # are importable during tests.
    sys.path.insert(0, REPO_ROOT)
if SRC_PATH not in sys.path:
    # Prefer src/ layout for the namespaced package import
    sys.path.insert(0, SRC_PATH)
# NOTE: keep this file minimal to avoid side effects during test collection.
