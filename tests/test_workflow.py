from swedish_parliament_policy_classifier.orchestration.workflow import (
    build_update_pipeline_handlers,
    WorkflowConfig,
    WorkflowOrchestrator,
)


def test_workflow_orchestrator_runs_phases_in_order() -> None:
    calls: list[str] = []
    orchestrator = WorkflowOrchestrator(
        {
            "update": lambda: calls.append("update") or {"new": 1},
            "classify": lambda: calls.append("classify") or {"rows": 2},
        },
        WorkflowConfig(phases=("update", "classify")),
    )

    result = orchestrator.run()

    assert result.status == "success"
    assert calls == ["update", "classify"]
    assert result.phases[1].result["rows"] == 2


def test_workflow_orchestrator_can_record_partial_failure() -> None:
    orchestrator = WorkflowOrchestrator(
        {
            "update": lambda: {"new": 1},
            "classify": lambda: (_ for _ in ()).throw(RuntimeError("model unavailable")),
            "analyze": lambda: {"rows": 3},
        },
        WorkflowConfig(phases=("update", "classify", "analyze"), allow_partial_failure=True),
    )

    result = orchestrator.run()

    assert result.status == "partial"
    assert result.failed_phases == ("classify",)
    assert result.phases[-1].name == "analyze"


def test_update_pipeline_adapter_preserves_phase_arguments() -> None:
    calls = []

    class Pipeline:
        def download_data(self, dry_run):
            calls.append(("download", dry_run))
            return {"downloaded": 1}

        def extract_data(self, dry_run):
            calls.append(("extract", dry_run))
            return {"extracted": 1}

        def classify_and_adjust(self, dry_run, cpu_fraction, allow_partial):
            calls.append(("classify", dry_run, cpu_fraction, allow_partial))
            return {"classified": 1}

        def rebuild_analysis(self, dry_run, cpu_fraction, allow_partial):
            return {"analysis": 1}

        def regenerate_figures(self, dry_run, cpu_fraction, allow_partial):
            return {"figures": 1}

        def render_manuscript(self, dry_run, allow_partial):
            return {"manuscript": 1}

    handlers = build_update_pipeline_handlers(
        Pipeline(), dry_run=True, cpu_fraction=0.25, allow_partial_failure=True
    )
    result = WorkflowOrchestrator(handlers, WorkflowConfig(phases=("update", "classify"))).run()

    assert result.status == "success"
    assert calls == [
        ("download", True),
        ("extract", True),
        ("classify", True, 0.25, True),
    ]