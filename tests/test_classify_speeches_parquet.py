import pandas as pd

from scripts.classify_speeches_parquet import _fallback_speech_ids, _flush_rows, _speech_inventory


def test_speech_inventory_separates_source_rows_from_unique_speeches(tmp_path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pd.DataFrame({"anforande_id": ["s1", "s2", "s2"]}).to_parquet(first, index=False)
    pd.DataFrame({"anforande_id": ["s2", "s3"]}).to_parquet(second, index=False)

    source_rows, speech_ids = _speech_inventory([first, second])

    assert source_rows == 5
    assert speech_ids == {"s1", "s2", "s3"}


def test_flush_rows_deduplicates_a_speech_category_pair(tmp_path) -> None:
    output = tmp_path / "classifications.parquet"

    row_count = _flush_rows(
        output,
        [
            {"speech_id": "s1", "category": "left", "normalized_weight": 0.4},
            {"speech_id": "s1", "category": "left", "normalized_weight": 0.8},
            {"speech_id": "s1", "category": "right", "normalized_weight": 0.2},
        ],
    )

    result = pd.read_parquet(output).sort_values("category").reset_index(drop=True)
    assert row_count == 2
    assert result["normalized_weight"].tolist() == [0.8, 0.2]


def test_fallback_speech_ids_only_selects_retired_scorer_rows() -> None:
    existing = pd.DataFrame(
        {
            "speech_id": ["s1", "s1", "s2", "s3"],
            "classifier_version": ["deterministic-fallback", "deterministic-fallback", "hybrid_ensemble+0.8.0", None],
        }
    )

    assert _fallback_speech_ids(existing) == {"s1"}