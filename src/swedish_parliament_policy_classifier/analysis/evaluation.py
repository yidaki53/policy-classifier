from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss


def cohen_kappa(labels: list[int] | list[str], predictions: list[int] | list[str]) -> float:
    """Compute Cohen's kappa for a pair of label sequences."""
    if len(labels) != len(predictions):
        raise ValueError("Labels and predictions must have the same length")
    if not labels:
        return float("nan")

    values = sorted({*labels, *predictions})
    if len(values) < 2:
        return 1.0

    observed = sum(1 for left, right in zip(labels, predictions) if left == right) / len(labels)
    expected = 0.0
    for value in values:
        left_prob = sum(1 for label in labels if label == value) / len(labels)
        right_prob = sum(1 for label in predictions if label == value) / len(predictions)
        expected += left_prob * right_prob
    if expected == 1.0:
        return 1.0
    return round((observed - expected) / (1.0 - expected), 6)


def bootstrap_confidence_interval(values: list[float], *, n_boot: int = 100, seed: int | None = None) -> tuple[float, float]:
    """Bootstrap a simple confidence interval for a mean statistic."""
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = np.asarray(values, dtype=float)
    means = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        resampled = rng.choice(samples, size=len(samples), replace=True)
        means[idx] = float(np.mean(resampled))
    lower = float(np.quantile(means, 0.05))
    upper = float(np.quantile(means, 0.95))
    return (round(lower, 6), round(upper, 6))


def run_sensitivity_analysis(frame: pd.DataFrame, *, scenario_col: str, weight_col: str) -> dict[str, Any]:
    """Summarize a lightweight sensitivity sweep over scenarios."""
    if frame.empty:
        return {"n_scenarios": 0, "results": []}
    results = []
    for _, row in frame.iterrows():
        results.append({
            "scenario": str(row[scenario_col]),
            "weight": float(row[weight_col]),
        })
    return {"n_scenarios": int(len(results)), "results": results}


def _class_names_from_probability_columns(probability_columns: list[str]) -> list[str]:
    names: list[str] = []
    for column in probability_columns:
        if column.startswith("prob_"):
            names.append(column[5:])
        else:
            names.append(column)
    return names


def summarize_classification_results(
    frame: pd.DataFrame,
    *,
    label_col: str,
    prediction_col: str,
    probability_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize classification performance with standard and uncertainty-aware metrics."""
    if frame.empty:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "n_samples": 0,
            "confusion_matrix": {},
            "brier_score": float("nan"),
            "log_loss": float("nan"),
        }

    labels = sorted({*frame[label_col].tolist(), *frame[prediction_col].tolist()})
    cm = confusion_matrix(frame[label_col], frame[prediction_col], labels=labels)
    confusion = {
        label: {pred: int(cm[idx, jdx]) for jdx, pred in enumerate(labels)}
        for idx, label in enumerate(labels)
    }

    report = {
        "accuracy": round(float((frame[label_col] == frame[prediction_col]).mean()), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(frame[label_col], frame[prediction_col])), 6),
        "macro_f1": round(float(f1_score(frame[label_col], frame[prediction_col], average="macro")), 6),
        "n_samples": int(len(frame)),
        "confusion_matrix": confusion,
    }

    if probability_columns:
        probs = frame[probability_columns].to_numpy(dtype=float)
        class_names = _class_names_from_probability_columns(probability_columns)
        if len(class_names) != probs.shape[1]:
            raise ValueError("Probability columns must align to the number of probability columns")
        if len(class_names) == 0:
            report["brier_score"] = float("nan")
            report["log_loss"] = float("nan")
        else:
            true_labels = pd.DataFrame(
                np.eye(len(class_names), dtype=float),
                columns=class_names,
                index=pd.Index(range(len(class_names))),
            )
            label_lookup = {label: idx for idx, label in enumerate(class_names)}
            encoded_targets = pd.DataFrame(
                {
                    label: (frame[label_col] == label).astype(float)
                    for label in class_names
                }
            ).to_numpy(dtype=float)
            if probs.shape[0] != encoded_targets.shape[0]:
                raise ValueError("Probability columns must align with the number of rows in the frame")
            report["brier_score"] = round(float(np.mean(np.sum((probs - encoded_targets) ** 2, axis=1))), 6)
            try:
                report["log_loss"] = round(float(log_loss(frame[label_col], probs, labels=class_names)), 6)
            except ValueError:
                report["log_loss"] = float("nan")
    else:
        report["brier_score"] = float("nan")
        report["log_loss"] = float("nan")

    return report
