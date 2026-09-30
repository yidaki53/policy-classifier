import numpy as np
import pandas as pd
import pytest

from scripts.train_speech_meta_clf_parquet import _split_data, load_data


def test_grouped_split_has_disjoint_groups() -> None:
    X = pd.DataFrame({"feature": np.arange(20)})
    y = pd.Series([0, 1] * 10)
    groups = np.repeat(np.arange(10), 2)

    _, _, _, _, train_groups, test_groups = _split_data(
        X, y, groups, None, "grouped"
    )

    assert set(train_groups).isdisjoint(set(test_groups))


def test_temporal_split_uses_later_dates_for_test() -> None:
    X = pd.DataFrame({"feature": np.arange(20)})
    y = pd.Series([0, 1] * 10)
    groups = np.arange(20)
    dates = pd.Series(pd.date_range("2020-01-01", periods=20, freq="D"))

    X_train, X_test, *_ = _split_data(X, y, groups, dates, "temporal")

    assert X_train["feature"].max() < X_test["feature"].min()


def test_training_data_rejects_mixed_teacher_sources(tmp_path) -> None:
    pd.DataFrame(
        {
            "speech_id": ["s1", "s2"],
            "category": ["left", "right"],
            "raw_response": ['{"left": 1.0}', '{"right": 1.0}'],
            "model": ["teacher-a", "teacher-b"],
            "prompt_version": ["v1", "v1"],
            "temperature": [0.0, 0.0],
        }
    ).to_parquet(tmp_path / "speech_gold_labels.parquet", index=False)
    pd.DataFrame(
        {"speech_id": ["s1", "s2"], "irony": [0.0, 0.0], "sarcasm": [0.0, 0.0], "posturing": [0.0, 0.0], "none": [1.0, 1.0], "top_label": ["none", "none"]}
    ).to_parquet(tmp_path / "speech_rhetoric_labels.parquet", index=False)

    with pytest.raises(ValueError, match="mixed teacher sources"):
        load_data(tmp_path, allow_teacher_labels=True)
