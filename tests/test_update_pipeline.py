from datetime import datetime
from typing import Any
import signal
import subprocess
import sys
import json

import pandas as pd
import pytest

from scripts import update_pipeline


def _write_analysis_stub_outputs(root: Any) -> None:
    for rel in [
        "output/analysis/party_decision_choices.parquet",
        "output/analysis/party_decision_choices_summary.json",
        "output/analysis/party_supported_action_positions.parquet",
        "output/analysis/say_do_transitions.parquet",
        "output/analysis/speech_action_axis_scores.parquet",
        "data/parquet/party_profiles_recency.parquet",
    ]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")


def test_classification_steps_fail_fast_by_default(monkeypatch) -> None:
    allow_fail_values: list[bool] = []

    def fake_run_step(
        _cmd: list[str],
        _step_name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        allow_fail_values.append(bool(kwargs.get("allow_fail", False)))
        return {"ok": True, "returncode": 0}

    monkeypatch.setattr(update_pipeline, "_run_step", fake_run_step)

    update_pipeline.classify_and_adjust(dry_run=False, cpu_fraction=0.25)

    assert allow_fail_values == [False, False, False]


def test_analysis_begins_with_action_evidence_and_fails_fast(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_run_step(
        _cmd: list[str],
        step_name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append((step_name, bool(kwargs.get("allow_fail", False))))
        return {"ok": True, "returncode": 0}

    monkeypatch.setattr(update_pipeline, "_run_step", fake_run_step)
    monkeypatch.setattr(update_pipeline, "REPO_ROOT", tmp_path)
    _write_analysis_stub_outputs(tmp_path)

    update_pipeline.rebuild_analysis(dry_run=False, cpu_fraction=0.25)

    assert calls[0] == ("build_party_action_evidence", False)
    assert all(not allow_fail for _, allow_fail in calls)


def test_analysis_includes_action_position_outputs_step(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    def fake_run_step(
        _cmd: list[str],
        step_name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append(step_name)
        return {"ok": True, "returncode": 0}

    monkeypatch.setattr(update_pipeline, "_run_step", fake_run_step)
    monkeypatch.setattr(update_pipeline, "REPO_ROOT", tmp_path)
    _write_analysis_stub_outputs(tmp_path)

    update_pipeline.rebuild_analysis(dry_run=False, cpu_fraction=0.25)

    assert "action_position_outputs" in calls
    assert calls.index("build_party_action_evidence") < calls.index("action_position_outputs")
    assert calls.index("action_position_outputs") < calls.index("build_profiles")


def test_classification_partial_mode_is_explicit(monkeypatch) -> None:
    allow_fail_values: list[bool] = []

    def fake_run_step(
        _cmd: list[str],
        _step_name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        allow_fail_values.append(bool(kwargs.get("allow_fail", False)))
        return {"ok": True, "returncode": 0}

    monkeypatch.setattr(update_pipeline, "_run_step", fake_run_step)

    update_pipeline.classify_and_adjust(
        dry_run=False,
        cpu_fraction=0.25,
        allow_partial=True,
    )

    assert allow_fail_values == [True, True, True]


def test_figure_steps_fail_fast_by_default(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_run_step(
        _cmd: list[str],
        step_name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append((step_name, bool(kwargs.get("allow_fail", False))))
        return {"ok": True, "returncode": 0}

    monkeypatch.setattr(update_pipeline, "_run_step", fake_run_step)
    monkeypatch.setattr(update_pipeline, "REPO_ROOT", tmp_path)

    update_pipeline.regenerate_figures(dry_run=False, cpu_fraction=0.25)

    assert calls
    assert all(not allow_fail for _, allow_fail in calls)


def test_regenerate_figures_includes_party_trend_step(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    def fake_run_step(
        _cmd: list[str],
        step_name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append(step_name)
        return {"ok": True, "returncode": 0}

    monkeypatch.setattr(update_pipeline, "_run_step", fake_run_step)
    monkeypatch.setattr(update_pipeline, "REPO_ROOT", tmp_path)

    update_pipeline.regenerate_figures(dry_run=False, cpu_fraction=0.25)

    assert "party_trends" in calls


def test_latest_dates_in_parquet_uses_recent_api_fetch_files(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "parquet" / "api_motions").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "parquet" / "api_speeches").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "parquet" / "normalized_motions.parquet").parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"date": ["2026-05-07"]}).to_parquet(tmp_path / "data" / "parquet" / "normalized_motions.parquet")
    pd.DataFrame({"date": ["2026-07-15"]}).to_parquet(tmp_path / "data" / "parquet" / "api_motions" / "motions.parquet")
    pd.DataFrame({"datum": ["2026-08-01"]}).to_parquet(tmp_path / "data" / "parquet" / "api_speeches" / "speeches.parquet")

    latest_dates = update_pipeline._latest_dates_in_parquet()

    assert latest_dates["motion"] == pd.Timestamp("2026-07-15")
    assert latest_dates["speech"] == pd.Timestamp("2026-08-01")


def test_fetch_new_items_from_api_handles_datetime_cutoff(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(update_pipeline, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(update_pipeline, "_latest_dates_in_parquet", lambda: {"motion": datetime(2026, 7, 15), "speech": None, "vote": None})
    monkeypatch.setattr(update_pipeline, "_load_last_fetch_cache", lambda: {})
    monkeypatch.setattr(update_pipeline, "_save_last_fetch_cache", lambda cache: None)
    monkeypatch.setattr(update_pipeline, "_get_existing_ids", lambda _doktyp: set())

    import swedish_parliament_policy_classifier.fetch.riksdag_client as riksdag_client

    def fake_fetch_page(*args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], bool]:
        return [], False

    monkeypatch.setattr(riksdag_client, "fetch_page", fake_fetch_page)

    result = update_pipeline.fetch_new_items_from_api(dry_run=False)

    assert result["motions"]["fetched"] == 0


def test_get_existing_speech_ids_supports_api_and_canonical_schemas(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(update_pipeline, "REPO_ROOT", tmp_path)
    canonical_dir = tmp_path / "data" / "speeches" / "parquet"
    api_dir = tmp_path / "data" / "parquet" / "api_speeches"
    canonical_dir.mkdir(parents=True)
    api_dir.mkdir(parents=True)
    pd.DataFrame({"anforande_id": ["canonical-speech"]}).to_parquet(canonical_dir / "speeches.parquet", index=False)
    pd.DataFrame({"id": ["api-id-speech"]}).to_parquet(api_dir / "api-id.parquet", index=False)
    pd.DataFrame({"anforande_id": ["api-anforande-speech"]}).to_parquet(api_dir / "api-anforande.parquet", index=False)

    existing_ids = update_pipeline._get_existing_ids("anf")

    assert existing_ids == {"canonical-speech", "api-id-speech", "api-anforande-speech"}


def test_run_step_streams_directly_when_tty_available(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = None
            self.stderr = None

        def wait(self) -> int:
            return 0

    def fake_popen(*args: Any, **kwargs: Any) -> FakeProcess:
        calls.append(kwargs)
        return FakeProcess()

    monkeypatch.setattr(update_pipeline.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(update_pipeline.sys.stderr, "isatty", lambda: True)
    monkeypatch.setattr(update_pipeline.sys.stdout, "isatty", lambda: True)

    update_pipeline._run_step(["echo", "ok"], "demo")

    assert calls and calls[0]["stdout"] is None
    assert calls[0]["stderr"] is None


def test_run_step_terminates_interactive_process_group_on_interrupt(monkeypatch) -> None:
    terminated: list[object] = []

    class FakeProcess:
        pid = 12345
        returncode = None

        def wait(self, timeout=None):
            if timeout is None:
                raise KeyboardInterrupt()
            self.returncode = -15
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr(update_pipeline.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(update_pipeline.sys.stderr, "isatty", lambda: True)
    monkeypatch.setattr(update_pipeline.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(update_pipeline, "_terminate_process_group", lambda proc: terminated.append(proc))

    with pytest.raises(KeyboardInterrupt):
        update_pipeline._run_step(["echo", "ok"], "demo")

    assert len(terminated) == 1


def test_manuscript_steps_fail_fast_by_default(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_run_step(
        _cmd: list[str],
        step_name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append((step_name, bool(kwargs.get("allow_fail", False))))
        return {"ok": True, "returncode": 0}

    monkeypatch.setattr(update_pipeline, "_run_step", fake_run_step)
    monkeypatch.setattr(update_pipeline, "REPO_ROOT", tmp_path)
    report_path = tmp_path / "manuscript" / "build" / "journal_requirements_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"status": "ready"}), encoding="utf-8")

    update_pipeline.render_manuscript(dry_run=False)

    assert calls == [
        ("manuscript_render", False),
        ("manuscript_combined", False),
        ("journal_requirements_check", False),
        ("publication_bundle", False),
    ]


def test_rebuild_analysis_requires_expected_outputs(monkeypatch, tmp_path) -> None:
    def fake_run_step(
        _cmd: list[str],
        step_name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if step_name == "build_party_action_evidence":
            output = tmp_path / "output/analysis/party_decision_choices.parquet"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"data")
        elif step_name == "action_position_outputs":
            output = tmp_path / "output/analysis/party_supported_action_positions.parquet"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"data")
        elif step_name == "build_profiles":
            output = tmp_path / "data/parquet/party_profiles_recency.parquet"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"data")
        return {"ok": True, "returncode": 0}

    monkeypatch.setattr(update_pipeline, "_run_step", fake_run_step)
    monkeypatch.setattr(update_pipeline, "REPO_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="party_decision_choices_summary.json"):
        update_pipeline.rebuild_analysis(dry_run=False, cpu_fraction=0.25)


def test_terminate_process_group_escalates_when_child_ignores_term(monkeypatch) -> None:
    signals: list[signal.Signals] = []

    class FakeProcess:
        pid = 12345

        def poll(self):
            return None

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("child", timeout)
            return -9

    monkeypatch.setattr(
        update_pipeline.os,
        "killpg",
        lambda _pid, sent_signal: signals.append(sent_signal),
    )

    update_pipeline._terminate_process_group(FakeProcess(), grace_seconds=0.01)

    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_main_returns_failure_when_critical_step_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(update_pipeline, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        update_pipeline,
        "check_api_new_periods",
        lambda dry_run: (_ for _ in ()).throw(RuntimeError("API audit failed")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "update_pipeline.py",
            "--skip-download",
            "--skip-extract",
            "--skip-api-fetch",
            "--skip-classify",
            "--skip-analysis",
            "--skip-figures",
            "--skip-manuscript",
            "--skip-questions",
            "--skip-betankande",
            "--skip-ip",
            "--skip-prop-classify",
        ],
    )

    returncode = update_pipeline.main()

    assert returncode == 1
    manifests = list((tmp_path / "logs").glob("update_pipeline_*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["error"] == "API audit failed"


def test_main_records_user_interruption_and_returns_130(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(update_pipeline, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        update_pipeline,
        "check_api_new_periods",
        lambda dry_run: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "update_pipeline.py",
            "--skip-download",
            "--skip-extract",
            "--skip-api-fetch",
            "--skip-classify",
            "--skip-analysis",
            "--skip-figures",
            "--skip-manuscript",
            "--skip-questions",
            "--skip-betankande",
            "--skip-ip",
            "--skip-prop-classify",
        ],
    )

    returncode = update_pipeline.main()

    assert returncode == 130
    manifest_path = next((tmp_path / "logs").glob("update_pipeline_*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["interrupted"] is True
    assert manifest["error"] == "Interrupted by user"