import pandas as pd
import pytest

from swedish_parliament_policy_classifier.analysis import summarize_classification_results
from swedish_parliament_policy_classifier.analysis.evaluation import (
    summarize_classification_results,
)


def test_summarize_classification_results_reports_balanced_accuracy_and_macro_f1() -> None:
    frame = pd.DataFrame(
        {
            "label": ["left", "left", "right", "right", "centre"],
            "prediction": ["left", "right", "right", "right", "centre"],
        }
    )

    report = summarize_classification_results(frame, label_col="label", prediction_col="prediction")

    assert report["accuracy"] == pytest.approx(0.8)
    assert report["balanced_accuracy"] == pytest.approx(0.833333)
    assert report["macro_f1"] == pytest.approx(0.822222)
    assert report["n_samples"] == 5
    assert report["confusion_matrix"]["left"]["right"] == 1


def test_summarize_classification_results_includes_probabilistic_metrics() -> None:
    frame = pd.DataFrame(
        {
            "label": ["left", "right"],
            "prediction": ["left", "right"],
            "prob_left": [1.0, 0.0],
            "prob_right": [0.0, 1.0],
        }
    )

    report = summarize_classification_results(
        frame,
        label_col="label",
        prediction_col="prediction",
        probability_columns=["prob_left", "prob_right"],
    )

    assert report["brier_score"] == pytest.approx(0.0)
    assert report["log_loss"] == pytest.approx(0.0)
