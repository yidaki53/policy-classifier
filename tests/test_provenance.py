import json
from pathlib import Path

from swedish_parliament_policy_classifier.provenance import write_run_provenance


def test_write_run_provenance_creates_timestamped_record(tmp_path):
    out_path = write_run_provenance(
        script="scripts/generate_figures.py",
        inputs={"db": "data/swedish_parliament.db"},
        outputs=["figures/manuscript/pie_chart_categories.png"],
        output_dir=tmp_path,
        metadata={"n_classified_motions": 123},
    )

    assert out_path.exists()
    assert out_path.parent == tmp_path / "provenance"
    assert out_path.name.startswith("generate_figures_")
    assert out_path.suffix == ".json"

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["script"] == "scripts/generate_figures.py"
    assert payload["inputs"]["db"] == "data/swedish_parliament.db"
    assert payload["outputs"] == ["figures/manuscript/pie_chart_categories.png"]
    assert payload["metadata"]["n_classified_motions"] == 123
    assert "timestamp_utc" in payload


def test_write_run_provenance_handles_empty_metadata(tmp_path):
    out_path = write_run_provenance(
        script="scripts/visualize_voting.py",
        inputs={"votering_parquet": "data/votering/parquet"},
        outputs=[],
        output_dir=tmp_path,
    )

    payload = json.loads(Path(out_path).read_text(encoding="utf-8"))
    assert payload["metadata"] == {}
