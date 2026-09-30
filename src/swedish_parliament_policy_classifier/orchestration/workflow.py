"""Small, testable orchestration boundary for reproducible workflow phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping


PhaseHandler = Callable[[], Mapping[str, Any]]


@dataclass(frozen=True)
class WorkflowConfig:
    phases: tuple[str, ...] = ("update", "classify", "analyze", "visualize", "manuscript")
    allow_partial_failure: bool = False


@dataclass(frozen=True)
class PhaseOutcome:
    name: str
    ok: bool
    result: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class WorkflowResult:
    status: str
    started_at: datetime
    completed_at: datetime
    phases: tuple[PhaseOutcome, ...]

    @property
    def failed_phases(self) -> tuple[str, ...]:
        return tuple(phase.name for phase in self.phases if not phase.ok)


class WorkflowOrchestrator:
    """Execute injected workflow phases with explicit failure semantics."""

    def __init__(self, handlers: Mapping[str, PhaseHandler], config: WorkflowConfig | None = None):
        self.handlers = dict(handlers)
        self.config = config or WorkflowConfig()

    def run(self, phases: tuple[str, ...] | None = None) -> WorkflowResult:
        started_at = datetime.now(timezone.utc)
        outcomes: list[PhaseOutcome] = []
        selected = phases or self.config.phases

        for name in selected:
            handler = self.handlers.get(name)
            if handler is None:
                outcome = PhaseOutcome(name=name, ok=False, error="phase handler is not configured")
            else:
                try:
                    outcome = PhaseOutcome(name=name, ok=True, result=dict(handler()))
                except Exception as exc:
                    outcome = PhaseOutcome(name=name, ok=False, error=str(exc))
            outcomes.append(outcome)
            if not outcome.ok and not self.config.allow_partial_failure:
                break

        completed_at = datetime.now(timezone.utc)
        if any(not phase.ok for phase in outcomes):
            status = "partial" if self.config.allow_partial_failure else "failed"
        else:
            status = "success"
        return WorkflowResult(status, started_at, completed_at, tuple(outcomes))


def build_update_pipeline_handlers(
    pipeline: Any,
    *,
    dry_run: bool = False,
    cpu_fraction: float = 1.0,
    allow_partial_failure: bool = False,
) -> dict[str, PhaseHandler]:
    """Adapt the existing update_pipeline module to the workflow boundary."""
    return {
        "update": lambda: {
            "download": pipeline.download_data(dry_run),
            "extract": pipeline.extract_data(dry_run),
        },
        "classify": lambda: pipeline.classify_and_adjust(
            dry_run,
            cpu_fraction,
            allow_partial_failure,
        ),
        "analyze": lambda: pipeline.rebuild_analysis(
            dry_run,
            cpu_fraction,
            allow_partial_failure,
        ),
        "visualize": lambda: pipeline.regenerate_figures(
            dry_run,
            cpu_fraction,
            allow_partial_failure,
        ),
        "manuscript": lambda: pipeline.render_manuscript(
            dry_run,
            allow_partial_failure,
        ),
    }


__all__ = [
    "PhaseOutcome",
    "WorkflowConfig",
    "WorkflowOrchestrator",
    "WorkflowResult",
    "build_update_pipeline_handlers",
]