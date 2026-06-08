"""Utilities for writing reproducibility provenance records."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_run_provenance(
    *,
    script: str,
    inputs: dict[str, Any],
    outputs: list[str],
    output_dir: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a JSON provenance record for one script run.

    The record is written inside ``<output_dir>/provenance/`` with a
    timestamped filename and returns the path to the JSON file.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir)
    prov_dir = out_dir / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "script": script,
        "timestamp_utc": ts,
        "inputs": inputs,
        "outputs": outputs,
        "metadata": metadata or {},
    }

    out_path = prov_dir / f"{Path(script).stem}_{ts}.json"
    out_path.write_text(json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path
