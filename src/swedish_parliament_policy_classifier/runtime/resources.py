"""Resource throttling helpers for CPU-intensive runs."""

from __future__ import annotations

import os
from typing import Dict


def apply_cpu_throttle(cpu_fraction: float = 0.25, min_threads: int = 1) -> Dict[str, int]:
    """Throttle common CPU thread pools to a fraction of available cores.

    Returns a dict with configured `max_threads`.
    """
    cpu_fraction = max(0.05, min(1.0, float(cpu_fraction)))
    ncpu = os.cpu_count() or 1
    max_threads = max(min_threads, int(round(ncpu * cpu_fraction)))

    thread_vars = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    for key in thread_vars:
        os.environ[key] = str(max_threads)

    try:
        import torch

        torch.set_num_threads(max_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, max_threads // 2))
    except Exception:
        pass

    return {"max_threads": max_threads, "cpu_count": ncpu}


def thermal_safe_defaults(mode: str = "safe") -> Dict[str, float]:
    """Return conservative runtime defaults for laptop thermals.

    Modes:
    - safe: balanced thermal control
    - cool: stronger cooling bias
    """
    mode = (mode or "safe").lower()
    if mode == "cool":
        return {"cpu_fraction": 0.15, "sleep_every": 20, "sleep_seconds": 0.3}
    return {"cpu_fraction": 0.25, "sleep_every": 50, "sleep_seconds": 0.2}
