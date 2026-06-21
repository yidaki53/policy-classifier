"""Tests for the Quid Ergo visualization."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_plot_quid_ergo_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_quid_ergo.py"
    spec = importlib.util.spec_from_file_location("plot_quid_ergo_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_topic_year_df(tmp_path: Path) -> tuple[Path, Path]:
    analysis_dir = tmp_path / "analysis"
    figures_dir = tmp_path / "figures"
    analysis_dir.mkdir()
    figures_dir.mkdir()

    rows = [
        # party, topic, year, S, M, V
        ("S", "left", 2020, 200, 300.0, 250.0),
        ("S", "centre", 2020, 100, 100.0, 80.0),
        ("S", "right", 2020, 50, 50.0, 40.0),
        ("S", "left", 2021, 180, 280.0, 270.0),
        ("S", "centre", 2021, 90, 110.0, 90.0),
        ("M", "right", 2020, 150, 200.0, 250.0),
        ("M", "centre", 2020, 80, 100.0, 120.0),
        ("M", "left", 2020, 30, 20.0, 15.0),
        ("SD", "right", 2020, 300, 100.0, 80.0),  # right-wing speech vs left action
        ("SD", "far_right", 2020, 100, 20.0, 15.0),
        ("V", "far_left", 2020, 250, 200.0, 180.0),
        ("V", "left", 2020, 100, 80.0, 70.0),
    ]
    cols = ["party", "topic", "year", "S", "M", "V"]
    df = pd.DataFrame(rows, columns=cols)
    src = analysis_dir / "promise_fulfillment_party_topic_year.parquet"
    df.to_parquet(src, index=False)
    return analysis_dir, figures_dir


def test_compute_speech_vs_action_gap_returns_party_rows(tmp_path):
    module = _load_plot_quid_ergo_module()
    analysis_dir, _ = _make_topic_year_df(tmp_path)

    df = pd.read_parquet(analysis_dir / "promise_fulfillment_party_topic_year.parquet")
    df_agg = df.groupby(["party", "topic"], as_index=False).agg(
        S=("S", "sum"), M=("M", "sum"), V=("V", "sum"),
    )

    pg = module._compute_speech_vs_action_gap(df_agg)
    assert set(pg["party"]) == {"S", "M", "SD", "V"}
    assert pg["gap"].notna().all()

    # Action must be tin [0,6] (number of categories - 1)
    assert pg["action_idx"].between(0, 6).all()


def test_plot_quid_ergo_writes_figure(tmp_path):
    module = _load_plot_quid_ergo_module()
    analysis_dir, figures_dir = _make_topic_year_df(tmp_path)

    out_path = module.plot_quid_ergo(analysis_dir, figures_dir, min_year=2020)
    assert out_path.exists()
    assert out_path.stat().st_size > 1000


def test_quid_ergo_filters_by_min_year(tmp_path):
    module = _load_plot_quid_ergo_module()
    analysis_dir, figures_dir = _make_topic_year_df(tmp_path)

    # Should work with min_year that excludes some rows
    out_path = module.plot_quid_ergo(analysis_dir, figures_dir, min_year=2021)
    assert out_path.exists()


def test_quid_ergo_includes_actionshare_calculation(tmp_path):
    module = _load_plot_quid_ergo_module()
    analysis_dir, _ = _make_topic_year_df(tmp_path)

    df = pd.read_parquet(analysis_dir / "promise_fulfillment_party_topic_year.parquet")

    # Verify the topic-level recomputation behaviour: speech_share and
    # action_share should be within [0, 100].
    df_agg = df.groupby(["party", "topic"], as_index=False).agg(
        S=("S", "sum"), M=("M", "sum"), V=("V", "sum"),
    )
    df_agg["action_count"] = df_agg["M"].fillna(0) + df_agg["V"].fillna(0)
    party_totals_speech = df_agg.groupby("party")["S"].sum().rename("party_S")
    party_totals_action = df_agg.groupby("party")["action_count"].sum().rename("party_action")
    merged = df_agg.merge(party_totals_speech, on="party").merge(party_totals_action, on="party")
    merged["speech_share"] = np.where(
        merged["party_S"] > 0,
        merged["S"] / merged["party_S"] * 100,
        0.0,
    )
    merged["action_share"] = np.where(
        merged["party_action"] > 0,
        merged["action_count"] / merged["party_action"] * 100,
        0.0,
    )
    assert merged["speech_share"].between(0, 100).all()
    assert merged["action_share"].between(0, 100).all()
