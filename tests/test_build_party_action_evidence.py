from pathlib import Path

import pandas as pd

from scripts.build_party_action_evidence import build_party_action_evidence


def test_build_party_action_evidence_materializes_all_decisions(tmp_path: Path) -> None:
    vote_dir = tmp_path / "votes"
    vote_dir.mkdir()
    pd.DataFrame(
        {
            "rm": ["202526", "202526", "202526", "202526", "202526"],
            "beteckning": ["AU1", "AU1", "AU1", "FiU2", "FiU2"],
            "punkt": ["3", "3", "3", "1", "1"],
            "votering_id": ["v1", "v1", "v1", "v2", "v2"],
            "datum": pd.to_datetime(
                [
                    "2026-02-12",
                    "2026-02-12",
                    "2026-02-12",
                    "2026-03-01",
                    "2026-03-01",
                ],
                utc=True,
            ),
            "parti": ["S", "S", "M", "S", None],
            "rost": ["Ja", "Ja", "Nej", "Frånvarande", "Ja"],
        }
    ).to_parquet(vote_dir / "votering-202526.parquet", index=False)
    out_path = tmp_path / "party_decisions.parquet"

    summary = build_party_action_evidence(vote_dir, out_path)

    out = pd.read_parquet(out_path)
    assert summary["member_records_n"] == 5
    assert summary["member_records_with_party_n"] == 4
    assert summary["excluded_missing_party_n"] == 1
    assert summary["decision_n"] == 2
    assert summary["party_decision_n"] == 3
    assert len(out) == 3
    assert set(out["party_choice"]) == {"yes", "no", "no_present"}