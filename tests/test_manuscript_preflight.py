from pathlib import Path

import pytest

from swedish_parliament_policy_classifier.analysis.manuscript_preflight import (
    assert_manuscript_ready,
    validate_manuscript_sections,
)


def test_manuscript_preflight_reports_missing_declared_artifacts(tmp_path: Path) -> None:
    sections = tmp_path / "sections"
    sections.mkdir()
    (sections / "results.md").write_text(
        "---\nsection_id: results\nrequired_inputs:\n  - data/results.parquet\nrequired_figures_tables:\n  - figures/results.png\n---\n# Results\n",
        encoding="utf-8",
    )

    result = validate_manuscript_sections(tmp_path)[0]

    assert not result.ok
    assert result.missing_inputs == ("data/results.parquet",)
    assert result.missing_figures == ("figures/results.png",)


def test_manuscript_preflight_accepts_existing_artifacts(tmp_path: Path) -> None:
    sections = tmp_path / "sections"
    sections.mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "results.parquet").write_bytes(b"artifact")
    (sections / "results.md").write_text(
        "---\nsection_id: results\nrequired_inputs:\n  - data/results.parquet\n---\n# Results\n",
        encoding="utf-8",
    )

    assert assert_manuscript_ready(tmp_path)[0].ok