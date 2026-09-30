#!/usr/bin/env python3
"""Run the structured workflow adapter around the existing update phases."""

from __future__ import annotations

import argparse
import json

try:
    from scripts import update_pipeline
except ModuleNotFoundError:
    import update_pipeline
from swedish_parliament_policy_classifier.orchestration.workflow import (
    WorkflowConfig,
    WorkflowOrchestrator,
    build_update_pipeline_handlers,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the structured reproducible workflow")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu-fraction", type=float, default=1.0)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--phase",
        dest="phases",
        action="append",
        choices=("update", "classify", "analyze", "visualize", "manuscript"),
        help="Run only the selected phase(s); repeat to choose multiple phases",
    )
    args = parser.parse_args()

    handlers = build_update_pipeline_handlers(
        update_pipeline,
        dry_run=args.dry_run,
        cpu_fraction=args.cpu_fraction,
        allow_partial_failure=args.allow_partial,
    )
    phases = tuple(args.phases) if args.phases else WorkflowConfig().phases
    result = WorkflowOrchestrator(
        handlers,
        WorkflowConfig(phases=phases, allow_partial_failure=args.allow_partial),
    ).run()
    print(
        json.dumps(
            {
                "status": result.status,
                "failed_phases": list(result.failed_phases),
                "phases": [
                    {"name": phase.name, "ok": phase.ok, "error": phase.error}
                    for phase in result.phases
                ],
            },
            indent=2,
        )
    )
    return 0 if result.status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
