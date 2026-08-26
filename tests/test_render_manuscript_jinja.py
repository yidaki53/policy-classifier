import pandas as pd

from scripts import render_manuscript_jinja


def test_build_action_position_table_renders_markdown_table(tmp_path) -> None:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    positions = pd.DataFrame(
        [
            {"party": "S", "score_0_100": 72.5, "supported_decision_n": 12},
            {"party": "M", "score_0_100": 48.0, "supported_decision_n": 8},
        ]
    )
    positions.to_parquet(analysis_dir / "party_supported_action_positions.parquet", index=False)

    table = render_manuscript_jinja._build_action_position_table(analysis_dir)

    assert "| party | score_0_100 | supported_decision_n |" in table
    assert "| S | 72.5000 | 12 |" in table
    assert "| M | 48.0000 | 8 |" in table
