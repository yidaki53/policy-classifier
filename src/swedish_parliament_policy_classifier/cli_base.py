"""Shared CLI entry-point plumbing for repository scripts.

Provides a single place to own the repeating cross-cutting concerns that
appear in almost every script:

* argparse scaffolding for resource/experiment flags,
* CPU throttling / thermal-safe defaults,
* MLflow experiment run lifecycle
"""
from __future__ import annotations

import argparse
import os
from typing import Any

from swedish_parliament_policy_classifier.runtime.experiment import ExperimentRun
from swedish_parliament_policy_classifier.runtime.resources import (
    apply_cpu_throttle,
    thermal_safe_defaults,
)


def build_common_parser(description: str) -> argparse.ArgumentParser:
    """Return an ``argparse.ArgumentParser`` pre-loaded with the flags shared
    across every repository script.

    Callers add script-specific flags on top and call ``parse_args()``
    themselves; this keeps the CLI surface under caller control while
    avoiding repeated boilerplate.
    """
    safe = thermal_safe_defaults("safe")

    p = argparse.ArgumentParser(description=description)

    # Resource controls
    p.add_argument(
        "--cpu-fraction",
        type=float,
        default=float(
            os.environ.get("CLASSIFIER_CPU_FRACTION", str(safe["cpu_fraction"]))
        ),
        help="CPU throttle fraction (0..1). Default from CLASSIFIER_CPU_FRACTION env.",
    )
    p.add_argument(
        "--sleep-every",
        type=int,
        default=int(safe["sleep_every"]),
        help="Sleep every N items (0 = disabled).",
    )
    p.add_argument(
        "--sleep-seconds",
        type=float,
        default=float(safe["sleep_seconds"]),
        help="Seconds to sleep when --sleep-every triggers.",
    )

    # Experiment tracking (MLflow)
    p.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow experiment tracking.",
    )
    p.add_argument(
        "--mlflow-experiment",
        default=None,
        help="MLflow experiment name (default: script-specific).",
    )
    p.add_argument(
        "--mlflow-tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI env).",
    )

    return p


def apply_resource_controls(args: argparse.Namespace) -> dict[str, Any]:
    """Apply CPU throttling using caller-supplied args.

    Returns a small context dict; today it exposes ``throttle``, but the
    dict gives us room to grow without reshaping every caller.
    """
    throttle = apply_cpu_throttle(cpu_fraction=args.cpu_fraction)
    return {"throttle": throttle}


class NullExperiment:
    """No-op stand-in for ``ExperimentRun`` when MLflow is disabled.

    Allows ``with``/``try..finally`` caller code to remain identical
    whether experiment tracking is on or off.
    """

    def __enter__(self) -> "NullExperiment":
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None

    def start(self, **_kwargs: Any) -> "NullExperiment":
        return self

    def log_params(self, **_kwargs: Any) -> None:
        return None

    def log_metrics(self, **_kwargs: Any) -> None:
        return None

    def log_artifact(self, **_kwargs: Any) -> None:
        return None

    def end(self, **_kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


def start_experiment(
    args: argparse.Namespace,
    run_name: str,
    *,
    experiment_name: str | None = None,
) -> ExperimentRun | NullExperiment:
    """Start an experiment run when MLflow is enabled, otherwise return a
    ``NullExperiment`` so caller code can stay uniform.
    """
    if not args.mlflow:
        return NullExperiment()

    return ExperimentRun.start(
        enabled=True,
        experiment_name=experiment_name or args.mlflow_experiment or run_name,
        run_name=run_name,
        tracking_uri=args.mlflow_tracking_uri,
    )