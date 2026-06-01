"""Utilities to limit parallelism for heavy ML exports and training.

Set environment variables for BLAS/OpenMP and optionally call
`threadpoolctl.threadpool_limits` to restrict native thread pools.

Usage:
    from scripts.parallel_utils import limit_threads
    limit_threads(1)

This is intentionally small and dependency-light; it will quietly no-op
if `threadpoolctl` is not available.
"""
from __future__ import annotations

import os
from typing import Optional


def limit_threads(n_jobs: Optional[int] = 1) -> None:
    """Limit native thread pools and set common environment variables.

    Args:
        n_jobs: desired maximum number of threads per process. If None,
            function returns without changes.
    """
    if n_jobs is None:
        return

    try:
        n = int(n_jobs)
    except Exception:
        return

    if n < 1:
        n = 1

    env_vars = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OMP_THREAD_LIMIT",
    ]
    for v in env_vars:
        os.environ[v] = str(n)

    # Try to apply threadpoolctl limits if available
    try:
        from threadpoolctl import threadpool_limits

        try:
            # preferred API: threadpool_limits(n)
            threadpool_limits(n)
        except TypeError:
            try:
                # older API: threadpool_limits(limits=n)
                threadpool_limits(limits=n)
            except Exception:
                # best-effort: ignore failures
                pass
    except Exception:
        # threadpoolctl not installed — that's fine
        pass


__all__ = ["limit_threads"]
