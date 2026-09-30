from dataclasses import dataclass

from swedish_parliament_policy_classifier.classifier.signals import extract_signal_output


@dataclass
class Result:
    category: str
    matched_rules: list[str]


def test_extract_signal_output_parses_only_requested_signal() -> None:
    output = extract_signal_output(
        [
            Result("left", ["embedding:0.25", "zero_shot:0.75"]),
            Result("right", ["embedding:not-a-score", "embedding:0.9"]),
        ],
        name="embedding",
        evidence_prefix="embedding",
    )

    assert output.name == "embedding"
    assert output.score_map() == {"left": 0.25, "right": 0.9}
    assert output.evidence["left"] == ("embedding:0.25",)