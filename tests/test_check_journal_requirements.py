from pathlib import Path

from scripts import check_journal_requirements


def test_bibliography_doi_validation_passes_when_metadata_matches(tmp_path, monkeypatch) -> None:
    manuscript_dir = tmp_path / "manuscript"
    sections_dir = manuscript_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    (sections_dir / "01_title.md").write_text("# Title\n\nAbstract.\n", encoding="utf-8")
    (sections_dir / "02_question.md").write_text("## Question\n\nTest.\n", encoding="utf-8")

    bib_dir = manuscript_dir / "bibliography"
    bib_dir.mkdir(parents=True, exist_ok=True)
    (bib_dir / "references.bib").write_text(
        """
@article{demo2024,
  title={A Valid Paper},
  author={Jane Doe and John Smith},
  doi={10.1000/demo},
  year={2024}
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        check_journal_requirements,
        "_fetch_bibliography_metadata",
        lambda doi: {
            "title": "A Valid Paper",
            "authors": ["Jane Doe", "John Smith"],
        },
    )

    report = check_journal_requirements._run_checks(
        repo_root=tmp_path,
        manuscript_dir=manuscript_dir,
        profile={"name": "Test Journal"},
    )

    checks = {item["id"]: item for item in report["checks"]}
    assert checks["bibliography_doi_metadata"]["status"] == "pass"


def test_bibliography_doi_validation_normalizes_title_dash_variants(monkeypatch) -> None:
    entry = {
        "raw": """
@article{demo2008,
  title={A Scaling Model for Estimating Time-Series Party Positions from Texts},
  author={Jane Doe},
  doi={10.1000/demo}
}
"""
    }
    monkeypatch.setattr(
        check_journal_requirements,
        "_fetch_bibliography_metadata",
        lambda doi: {
            "title": "A Scaling Model for Estimating Time‐Series Party Positions from Texts",
            "authors": ["Jane Doe"],
        },
    )

    valid, detail = check_journal_requirements._validate_bibliography_entry(entry)

    assert valid
    assert detail == ""


def test_unavailable_doi_metadata_does_not_block_journal_readiness(tmp_path, monkeypatch) -> None:
    manuscript_dir = tmp_path / "manuscript"
    sections_dir = manuscript_dir / "sections"
    sections_dir.mkdir(parents=True)
    (sections_dir / "01_title.md").write_text("# Title\n\nAbstract.\n\nData Availability.", encoding="utf-8")

    bib_dir = manuscript_dir / "bibliography"
    bib_dir.mkdir()
    (bib_dir / "references.bib").write_text(
        "\n".join(
            f"@article{{entry{index},\n  title={{Paper {index}}},\n  doi={{10.1000/demo{index}}}\n}}"
            for index in range(10)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(check_journal_requirements, "_fetch_bibliography_metadata", lambda doi: None)

    report = check_journal_requirements._run_checks(
        repo_root=tmp_path,
        manuscript_dir=manuscript_dir,
        profile={"name": "Test Journal"},
    )

    checks = {item["id"]: item for item in report["checks"]}
    assert checks["bibliography_doi_metadata"]["status"] == "warn"
    assert checks["bibliography_doi_metadata"]["blocking"] is False
    assert report["status"] == "ready"
