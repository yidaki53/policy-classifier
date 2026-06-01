#!/usr/bin/env bash
set -euo pipefail

# Run pytest with coverage enforcement (minimum 70%).
#
# This helper intentionally separates coverage invocation from the default
# `pytest` command because some system-managed Pythons disallow installing
# test plugins globally. Use this script from a virtualenv where
# `pytest-cov` is available (or in CI) to get an enforced coverage check.
PYTEST_CMD=("python" "-m" "pytest" "--cov=src" "--cov-report=term-missing" "--cov-fail-under=70")

"${PYTEST_CMD[@]}" "$@"
