import importlib.util
import sqlite3
from pathlib import Path


def _load_generate_figures_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_figures.py"
    spec = importlib.util.spec_from_file_location("generate_figures_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_classifications_is_deterministic_on_ties(tmp_path):
    module = _load_generate_figures_module()

    db_path = tmp_path / "figures.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE classifications (
            motion_id TEXT,
            category TEXT,
            normalized_weight REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE normalized_motions (
            id TEXT PRIMARY KEY,
            date TEXT,
            party TEXT,
            doc_type TEXT
        )
        """
    )

    cur.executemany(
        "INSERT INTO classifications (motion_id, category, normalized_weight) VALUES (?, ?, ?)",
        [
            ("m1", "right", 0.8),
            ("m1", "left", 0.8),
            ("m2", "centre", 0.3),
            ("m2", "far_left", 0.9),
        ],
    )
    cur.executemany(
        "INSERT INTO normalized_motions (id, date, party, doc_type) VALUES (?, ?, ?, ?)",
        [
            ("m1", "2020-01-01", "V", "mot"),
            ("m2", "2021-01-01", "M", "mot"),
        ],
    )
    conn.commit()

    rows = module.load_classifications(conn)
    by_motion = {r[0]: r for r in rows}

    # For tied normalized_weight, category ASC tie-break should make this deterministic.
    assert by_motion["m1"][1] == "left"
    assert by_motion["m1"][2] == 0.8
    assert by_motion["m2"][1] == "far_left"

    conn.close()


def test_generate_all_figures_writes_provenance(tmp_path, monkeypatch):
    module = _load_generate_figures_module()

    db_path = tmp_path / "figures.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE classifications (
            motion_id TEXT,
            category TEXT,
            normalized_weight REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE normalized_motions (
            id TEXT PRIMARY KEY,
            date TEXT,
            party TEXT,
            doc_type TEXT
        )
        """
    )
    cur.executemany(
        "INSERT INTO classifications (motion_id, category, normalized_weight) VALUES (?, ?, ?)",
        [
            ("m1", "left", 0.9),
        ],
    )
    cur.executemany(
        "INSERT INTO normalized_motions (id, date, party, doc_type) VALUES (?, ?, ?, ?)",
        [
            ("m1", "2020-01-01", "V", "mot"),
        ],
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(module, "plot_pie_chart", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "plot_party_motions", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "plot_ideology_timeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "plot_party_ideology_heatmap", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "query_summary_stats",
        lambda _conn: {"n_parties": 1, "n_motions": 1, "date_range": "2020-2020"},
    )

    calls = {}

    def _fake_write_run_provenance(**kwargs):
        calls.update(kwargs)
        return tmp_path / "figures" / "provenance" / "generate_figures_20260101T000000Z.json"

    monkeypatch.setattr(module, "write_run_provenance", _fake_write_run_provenance)

    out_dir = tmp_path / "figures"
    module.generate_all_figures(str(db_path), str(out_dir))

    assert calls["script"] == "scripts/generate_figures.py"
    assert calls["inputs"]["db"] == str(db_path)
    assert calls["output_dir"] == out_dir
    assert any(name.endswith("pie_chart_categories.png") for name in calls["outputs"])
