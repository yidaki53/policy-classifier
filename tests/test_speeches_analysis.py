import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "speeches_analysis.py"
    spec = importlib.util.spec_from_file_location("speeches_analysis_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vote_signal_treats_only_yes_and_no_as_binary():
    module = _load_module()

    assert module.vote_signal("ja") == 1.0
    assert module.vote_signal("NEJ") == 0.0
    assert np.isnan(module.vote_signal("avstår"))
    assert np.isnan(module.vote_signal("frånvarande"))


def test_filter_excluded_parties_removes_non_substantive_labels():
    module = _load_module()
    df = pd.DataFrame({"party": ["M", "Moderaterna", "V", "Vänsterpartiet", "Unknown", "X"], "value": [1, 2, 3, 4, 5, 6]})

    out = module._filter_excluded_parties(df, module.DEFAULT_EXCLUDED_PARTIES)

    assert sorted(out["party"].unique().tolist()) == ["M", "V"]


def test_run_tests_on_profiles_returns_category_level_rows():
    module = _load_module()

    rows = []
    for party, speech_base, motion_base, vote_base in [("M", 0.1, 0.2, 0.3), ("V", 0.3, 0.2, 0.1)]:
        for modality, base in [("speech", speech_base), ("motion", motion_base), ("vote", vote_base)]:
            row = {"party": party, "modality": modality}
            for i, cat in enumerate(module.IDEOLOGY_ORDER):
                row[cat] = base + 0.01 * i
            rows.append(row)

    pivot = pd.DataFrame(rows)
    out = module.run_tests_on_profiles(pivot)

    assert set(out["party"].unique().tolist()) == {"ALL_PARTIES"}
    assert set(out["category"].unique().tolist()) == set(module.IDEOLOGY_ORDER)
    assert set(out["comparison"].unique().tolist()) == {"speech_vs_motion", "speech_vs_vote", "speech_vs_combined"}
    assert len(out) == len(module.IDEOLOGY_ORDER) * 3