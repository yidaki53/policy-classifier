import pandas as pd

from scripts.prepare_speech_gold_review import prepare_review_sample


def test_prepare_review_sample_is_stratified_and_blinded(tmp_path) -> None:
    labels_path = tmp_path / "labels.parquet"
    speech_dir = tmp_path / "speeches"
    speech_dir.mkdir()
    pd.DataFrame(
        {
            "speech_id": ["s1", "s2", "s3", "s4"],
            "category": ["left", "left", "right", "right"],
        }
    ).to_parquet(labels_path, index=False)
    pd.DataFrame(
        {"anforande_id": ["s1", "s2", "s3", "s4"], "anforandetext": ["A", "B", "C", "D"]}
    ).to_parquet(speech_dir / "speeches.parquet", index=False)

    output = prepare_review_sample(labels_path, speech_dir, tmp_path / "review.parquet", per_category=1)
    review = pd.read_parquet(output)

    assert len(review) == 2
    assert "category" not in review.columns
    assert set(review["adjudicated_category"]) == {""}
    key = pd.read_parquet(tmp_path / "review_key.parquet")
    assert set(key["category"]) == {"left", "right"}