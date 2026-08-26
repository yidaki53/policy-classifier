#!/usr/bin/env python3
"""Audit an update-pipeline manifest for hidden nested failures.

The update orchestrator records subprocess outcomes at several nesting levels.
This command walks the complete manifest so a successful outer command cannot
mask a failed classification, analysis, figure, or rendering subprocess.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _failed_steps(value: Any, path: str = "") -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if not isinstance(value, Mapping):
        return failures

    current_path = path or "manifest"
    if value.get("ok") is False:
        failures.append(
            {
                "path": current_path,
                "step": str(value.get("step") or current_path.rsplit(".", 1)[-1]),
                "returncode": value.get("returncode"),
            }
        )

    for key, child in value.items():
        child_path = f"{path}.{key}" if path else str(key)
        failures.extend(_failed_steps(child, child_path))
    return failures


def audit_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic pass/fail report for one pipeline manifest."""
    failures = _failed_steps(manifest)
    manifest_errors = [str(manifest["error"])] if manifest.get("error") else []
    return {
        "status": "failed" if failures or manifest_errors else "passed",
        "run_ts": manifest.get("run_ts"),
        "completed_at": manifest.get("completed_at"),
        "failed_steps": failures,
        "manifest_errors": manifest_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an update pipeline manifest")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = audit_manifest(manifest)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())