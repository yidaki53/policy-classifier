import importlib.util
from pathlib import Path

import pandas as pd


def _load_generate_figures_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_figures.py"
    spec = importlib.util.spec_from_file_location("generate_figures_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_classifications_is_deterministic_on_ties(tmp_path):
    module = _load_generate_figures_module()

    cls_path = tmp_path / "classifications.parquet"
    nm_path = tmp_path / "normalized_motions.parquet"

    cls = pd.DataFrame(
        [
            ("m1", "right", 0.8),
            ("m1", "left", 0.8),
            ("m2", "centre", 0.3),
            ("m2", "far_left", 0.9),
        ],
        columns=["motion_id", "category", "normalized_weight"],
    )
    nm = pd.DataFrame(
        [
            ("m1", "2020-01-01", "V", "mot"),
            ("m2", "2021-01-01", "M", "mot"),
        ],
        columns=["id", "date", "party", "doc_type"],
    )
    cls.to_parquet(cls_path, index=False)
    nm.to_parquet(nm_path, index=False)

    rows = module.load_classifications_parquet(str(cls_path), str(nm_path))
    by_motion = {r[0]: r for r in rows}

    # For tied normalized_weight, category ASC tie-break should make this deterministic.
    assert by_motion["m1"][1] == "left"
    assert by_motion["m1"][2] == 0.8
    assert by_motion["m2"][1] == "far_left"


def test_generate_all_figures_writes_provenance(tmp_path, monkeypatch):
    module = _load_generate_figures_module()

    cls_path = tmp_path / "classifications.parquet"
    nm_path = tmp_path / "normalized_motions.parquet"

    cls = pd.DataFrame(
        [("m1", "left", 0.9)],
        columns=["motion_id", "category", "normalized_weight"],
    )
    nm = pd.DataFrame(
        [("m1", "2020-01-01", "V", "mot")],
        columns=["id", "date", "party", "doc_type"],
    )
    cls.to_parquet(cls_path, index=False)
    nm.to_parquet(nm_path, index=False)

    monkeypatch.setattr(module, "plot_pie_chart", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "plot_party_motions", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "plot_ideology_timeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "plot_party_ideology_heatmap", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "query_summary_stats_parquet",
        lambda _c, _n: {"n_parties": 1, "n_motions": 1, "date_range": "2020-2020"},
    )

    calls = {}

    def _fake_write_run_provenance(**kwargs):
        calls.update(kwargs)
        return tmp_path / "figures" / "provenance" / "generate_figures_20260101T000000Z.json"

    monkeypatch.setattr(module, "write_run_provenance", _fake_write_run_provenance)

    out_dir = tmp_path / "figures"
    module.generate_all_figures(str(cls_path), str(nm_path), str(out_dir))

    assert calls["script"] == "scripts/generate_figures.py"
    assert calls["inputs"]["classifications"] == str(cls_path)
    assert calls["inputs"]["normalized_motions"] == str(nm_path)
    assert calls["output_dir"] == out_dir
    assert any(name.endswith("pie_chart_categories.png") for name in calls["outputs"])
