import importlib.util
from pathlib import Path

import pandas as pd


def _load_visualize_voting_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "visualize_voting.py"
    spec = importlib.util.spec_from_file_location("visualize_voting_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_all_writes_provenance(tmp_path, monkeypatch):
    module = _load_visualize_voting_module()

    df = pd.DataFrame(
        {
            "rm": ["202001"],
            "votering_id": ["v1"],
            "parti": ["S"],
            "rost": ["ja"],
            "year": [2020],
            "beteckning": ["FiU1"],
        }
    )

    monkeypatch.setattr(module, "_load_all_votering", lambda _path: df.copy())
    monkeypatch.setattr(module, "plot_cohesion", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "plot_agreement_matrix", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "plot_vote_distribution", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "plot_motions_vs_votes_timeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "plot_committee_distribution", lambda *args, **kwargs: None)

    calls = {}

    def _fake_write_run_provenance(**kwargs):
        calls.update(kwargs)
        return tmp_path / "figures" / "provenance" / "visualize_voting_20260101T000000Z.json"

    monkeypatch.setattr(module, "write_run_provenance", _fake_write_run_provenance)

    out_dir = tmp_path / "figures"
    module.generate_all("data/votering/parquet", str(out_dir), "data/swedish_parliament.db")

    assert calls["script"] == "scripts/visualize_voting.py"
    assert calls["inputs"]["votering_parquet"] == "data/votering/parquet"
    assert calls["inputs"]["db"] == "data/swedish_parliament.db"
    assert calls["output_dir"] == out_dir
    assert any(name.endswith("cross_party_agreement.pdf") for name in calls["outputs"])
