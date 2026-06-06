import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "sync_zenodo_doi.py"
    spec = importlib.util.spec_from_file_location("sync_zenodo_doi_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_update_data_availability_replaces_existing_line():
    module = _load_module()
    text = (
        "The full reproducible project is publicly accessible at "
        "`https://github.com/yidaki53/policy-classifier`. "
        "Submission and production versions should cite the exact release tag and commit hash used for manuscript generation.\n\n"
        "Archival DOI for the submission snapshot (`submission-2026-06-06-r2`): `https://doi.org/10.5281/zenodo.11111111`.\n"
    )

    updated = module.update_data_availability_text(
        text=text,
        tag="submission-2026-06-06-r3",
        doi_url="https://doi.org/10.5281/zenodo.20572644",
    )

    assert "submission-2026-06-06-r3" in updated
    assert "10.5281/zenodo.20572644" in updated
    assert "10.5281/zenodo.11111111" not in updated


def test_update_data_availability_inserts_line_when_missing():
    module = _load_module()
    text = (
        "The full reproducible project is publicly accessible at "
        "`https://github.com/yidaki53/policy-classifier`. "
        "Submission and production versions should cite the exact release tag and commit hash used for manuscript generation.\n"
    )

    updated = module.update_data_availability_text(
        text=text,
        tag="submission-2026-06-06-r3",
        doi_url="https://doi.org/10.5281/zenodo.20572644",
    )

    assert "Archival DOI for the submission snapshot" in updated
    assert "10.5281/zenodo.20572644" in updated


def test_update_checklist_text_marks_doi_complete():
    module = _load_module()
    text = (
        "- [x] Add CRediT author contribution statement in manuscript source and submission metadata.\n"
        "- [ ] Mint persistent archival DOI for the exact submission snapshot (for example via Zenodo-linked release) and add DOI citation to Data Availability text.\n"
        "- [x] Create and push a superseding final submission tag on latest commit, then update Stage 7 metadata (tag, SHA, UTC, DOI link).\n"
    )

    updated = module.update_checklist_text(text=text, doi_url="https://doi.org/10.5281/zenodo.20572644")

    assert "- [x] Mint persistent archival DOI for the exact submission snapshot" in updated
    assert "10.5281/zenodo.20572644" in updated
    assert "- [ ] Mint persistent archival DOI" not in updated


def test_find_matching_doi_prefers_tag_or_repo_indicators():
    module = _load_module()

    records = [
        {
            "doi": "10.5281/zenodo.00000001",
            "metadata": {"title": "Unrelated artifact", "related_identifiers": []},
        },
        {
            "doi": "10.5281/zenodo.20572644",
            "metadata": {
                "title": "yidaki53/policy-classifier: Submission snapshot 2026-06-06-r3",
                "related_identifiers": [
                    {
                        "identifier": "https://github.com/yidaki53/policy-classifier/releases/tag/submission-2026-06-06-r3"
                    }
                ],
            },
        },
    ]

    doi = module._find_matching_doi(
        records=records,
        tag="submission-2026-06-06-r3",
        repo="yidaki53/policy-classifier",
    )

    assert doi == "https://doi.org/10.5281/zenodo.20572644"
