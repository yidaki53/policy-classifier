"""Small optional MLflow wrapper with graceful fallback when MLflow is missing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ExperimentRun:
    enabled: bool
    name: str
    output_dir: Path = Path("logs")
    _mlflow: Any = None
    _run: Any = None
    _fallback: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        enabled: bool,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        output_dir: str = "logs",
    ) -> "ExperimentRun":
        inst = cls(enabled=bool(enabled), name=run_name or experiment_name, output_dir=Path(output_dir))
        inst.output_dir.mkdir(parents=True, exist_ok=True)

        if not inst.enabled:
            return inst

        try:
            import mlflow

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            inst._run = mlflow.start_run(run_name=run_name)
            inst._mlflow = mlflow
        except Exception:
            inst._mlflow = None
            inst._run = None

        return inst

    def log_params(self, params: Dict[str, Any]) -> None:
        clean = {str(k): str(v) for k, v in (params or {}).items() if v is not None}
        if self._mlflow is not None:
            self._mlflow.log_params(clean)
            return
        self._fallback.setdefault("params", {}).update(clean)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        clean = {}
        for k, v in (metrics or {}).items():
            try:
                clean[str(k)] = float(v)
            except Exception:
                continue

        if self._mlflow is not None:
            self._mlflow.log_metrics(clean, step=step)
            return
        self._fallback.setdefault("metrics", []).append({"step": step, **clean})

    def log_artifact(self, local_path: str) -> None:
        if self._mlflow is not None:
            try:
                self._mlflow.log_artifact(local_path)
            except Exception:
                pass
            return
        self._fallback.setdefault("artifacts", []).append(local_path)

    def end(self, status: str = "FINISHED") -> None:
        if self._mlflow is not None:
            try:
                self._mlflow.end_run(status=status)
            except Exception:
                pass
            return

        if self.enabled:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out = self.output_dir / f"experiment_fallback_{ts}.json"
            out.write_text(json.dumps(self._fallback, indent=2), encoding="utf-8")
