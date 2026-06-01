import pandas as pd

from swedish_parliament_policy_classifier.classifier.persistence_port import ParquetClassificationWriter


def test_parquet_writer_dedupes_by_speech_id_and_category(tmp_path):
    out_path = tmp_path / "speech_classifications.parquet"
    writer = ParquetClassificationWriter(output_path=out_path)

    first = pd.DataFrame(
        [
            {"speech_id": "s1", "category": "left", "score": 0.6},
            {"speech_id": "s1", "category": "right", "score": 0.4},
            {"speech_id": "s2", "category": "right", "score": 0.7},
        ]
    )
    second = pd.DataFrame(
        [
            {"speech_id": "s1", "category": "left", "score": 0.9},
        ]
    )

    n1 = writer.write(first)
    n2 = writer.write(second)

    assert n1 == 3
    assert n2 == 3

    out = pd.read_parquet(out_path)
    # Multi-category distributions for a speech should be preserved.
    s1 = out[out["speech_id"] == "s1"].sort_values("category").reset_index(drop=True)
    assert len(s1) == 2
    assert set(s1["category"].tolist()) == {"left", "right"}
    # Exact duplicate key (speech_id, category) should keep the most recent row.
    left_row = s1[s1["category"] == "left"].iloc[0]
    assert left_row["score"] == 0.9
