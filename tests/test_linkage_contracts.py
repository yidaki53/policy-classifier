import pandas as pd
import pytest

from swedish_parliament_policy_classifier.analysis.linkage_contracts import summarize_linkage


def test_summarize_linkage_reports_coverage_and_confidence() -> None:
    summary = summarize_linkage(
        pd.DataFrame(
            {
                "speech_id": ["s1", "s2"],
                "motion_id": ["m1", "m1"],
                "confidence": [0.8, 0.6],
            }
        )
    )

    assert summary.linked_rate == 1.0
    assert summary.unique_speeches == 2
    assert summary.unique_motions == 1
    assert summary.mean_confidence == 0.7


def test_summarize_linkage_rejects_missing_identifiers() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        summarize_linkage(pd.DataFrame({"speech_id": ["s1"]}))