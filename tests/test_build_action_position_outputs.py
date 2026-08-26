import pandas as pd

from scripts.build_action_position_outputs import build_action_position_outputs


def test_build_action_position_outputs_writes_expected_artifacts(tmp_path) -> None:
    party_choices = pd.DataFrame(
        [
            {
                "decision_id": "d1",
                "party": "S",
                "decision_date": pd.Timestamp("2023-12-01", tz="UTC"),
                "party_choice": "yes",
            },
            {
                "decision_id": "d2",
                "party": "M",
                "decision_date": pd.Timestamp("2024-01-02", tz="UTC"),
                "party_choice": "yes",
            },
        ]
    )
    policy_probabilities = pd.DataFrame(
        [
            {"decision_id": "d1", "category": "left", "probability": 0.2},
            {"decision_id": "d1", "category": "right", "probability": 0.8},
            {"decision_id": "d2", "category": "left", "probability": 0.6},
            {"decision_id": "d2", "category": "right", "probability": 0.4},
        ]
    )
    speech_stances = pd.DataFrame(
        [
            {
                "speech_id": "s1",
                "motion_id": "m1",
                "decision_id": "d1",
                "speech_party": "S",
                "speech_stance": "support",
            },
            {
                "speech_id": "s2",
                "motion_id": "m2",
                "decision_id": "d2",
                "speech_party": "M",
                "speech_stance": "oppose",
            },
        ]
    )

    out_dir = tmp_path / "analysis"
    party_choices.to_parquet(tmp_path / "party_choices.parquet", index=False)
    policy_probabilities.to_parquet(tmp_path / "policy_probabilities.parquet", index=False)
    speech_stances.to_parquet(tmp_path / "speech_stances.parquet", index=False)

    summary = build_action_position_outputs(
        party_choices_path=tmp_path / "party_choices.parquet",
        policy_probabilities_path=tmp_path / "policy_probabilities.parquet",
        speech_stances_path=tmp_path / "speech_stances.parquet",
        out_dir=out_dir,
    )

    assert (out_dir / "party_supported_action_positions.parquet").exists()
    assert (out_dir / "say_do_transitions.parquet").exists()
    assert (out_dir / "action_position_outputs_summary.json").exists()
    assert summary["party_position_n"] == 1
    assert summary["transition_n"] == 2
    assert summary["study_specification"]["data_cutoff_utc"] == "2024-01-02T00:00:00Z"
