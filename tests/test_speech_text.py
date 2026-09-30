import pandas as pd

from swedish_parliament_policy_classifier.io.speech_text import (
    InMemorySpeechTextRepository,
    ParquetSpeechTextRepository,
)


def test_in_memory_speech_text_repository_handles_missing_ids() -> None:
    repository = InMemorySpeechTextRepository({"s1": "Hej"})

    assert repository.get("s1") == "Hej"
    assert repository.get("missing") == ""
    assert repository.get_many(["s1", "missing"]) == {"s1": "Hej", "missing": ""}


def test_parquet_speech_text_repository_loads_once_and_preserves_first_duplicate(tmp_path) -> None:
    pd.DataFrame(
        {
            "anforande_id": ["s1", "s2"],
            "anforandetext": ["Första", "Andra"],
        }
    ).to_parquet(tmp_path / "speeches.parquet", index=False)

    repository = ParquetSpeechTextRepository(tmp_path)

    assert repository.get("s1") == "Första"
    assert repository.get_many(["s1", "s3"]) == {"s1": "Första", "s3": ""}
    assert repository._texts is not None