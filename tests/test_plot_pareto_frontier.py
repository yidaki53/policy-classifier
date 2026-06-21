"""Tests for the Pareto frontier visualization."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_plot_pareto_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_pareto_frontier.py"
    spec = importlib.util.spec_from_file_location("plot_pareto_frontier_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_party_year_df(tmp_path: Path) -> tuple[Path, Path]:
    analysis_dir = tmp_path / "analysis"
    figures_dir = tmp_path / "figures"
    analysis_dir.mkdir()
    figures_dir.mkdir()

    party_year = pd.DataFrame(
        [
            ("S", 2020, 0.400, 0.60),
            ("S", 2021, 0.420, 0.65),
            ("S", 2022, 0.380, 0.55),
            ("M", 2020, 0.350, 0.40),
            ("M", 2021, 0.330, 0.38),
            ("SD", 2020, 0.500, 0.30),  # frontier: high consistency, lower fidelity
            ("SD", 2021, 0.520, 0.35),
            ("V", 2020, 0.300, 0.20),  # dominated on both axes
            ("V", 2021, 0.310, 0.18),
            ("MP", 2020, 0.450, 0.50),  # frontier candidate
        ],
        columns=["party", "year", "pct_speech_motion_vote", "consistency_score"],
    )
    src = analysis_dir / "consistency_fulfillment_party_year.parquet"
    party_year.to_parquet(src, index=False)
    return analysis_dir, figures_dir


def test_compute_pareto_frontier_returns_nondominated_points(tmp_path):
    module = _load_plot_pareto_module()
    analysis_dir, _ = _make_party_year_df(tmp_path)

    df = pd.read_parquet(analysis_dir / "consistency_fulfillment_party_year.parquet")
    df = df[df["pct_speech_motion_vote"] > 0].copy()

    frontier = module.compute_pareto_frontier(df)
    frontier_parties = set(frontier["party"].tolist())

    # V should be dominated (low consistency AND low fidelity)
    assert "V" not in frontier_parties
    # SD and MP should remain on frontier
    assert "SD" in frontier_parties
    assert "MP" in frontier_parties


def test_plot_pareto_frontier_writes_figure(tmp_path):
    module = _load_plot_pareto_module()
    analysis_dir, figures_dir = _make_party_year_df(tmp_path)

    out_path = module.plot_pareto_frontier(analysis_dir, figures_dir, min_year=2020)
    assert out_path.exists()
    assert out_path.parent == figures_dir
    # File should be non-empty
    assert out_path.stat().st_size > 1000


def test_plot_pareto_frontier_handles_min_year_filter(tmp_path):
    module = _load_plot_pareto_module()
    analysis_dir, figures_dir = _make_party_year_df(tmp_path)

    # Use min_year that excludes some rows
    out_path = module.plot_pareto_frontier(analysis_dir, figures_dir, min_year=2021)
    assert out_path.exists()

    # Recency weighting is included
    df = pd.read_parquet(analysis_dir / "consistency_fulfillment_party_year.parquet")
    df = df[df["year"] >= 2021].copy()
    df = df[df["pct_speech_motion_vote"] > 0].copy()
    assert len(df) > 0
