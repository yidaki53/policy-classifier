#!/usr/bin/env python3
"""Train a speech meta-classifier directly from Parquet exports.

Features used:
- probabilities parsed from `speech_gold_labels.raw_response` (one column per
  ideological class)
- numeric rhetoric scores from `speech_rhetoric_labels` (irony, sarcasm, posturing, none)

This avoids requiring the SQLite DB and works purely from `data/parquet/`.

Example:
    uv run python3 scripts/train_speech_meta_clf_parquet.py --parquet-dir data/parquet --out models/speech_meta_clf_parquet.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

import os
import sys

try:
    from scripts.parallel_utils import limit_threads
except Exception:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from scripts.parallel_utils import limit_threads


LOG = logging.getLogger("train_speech_meta_clf_parquet")


def load_data(
    parquet_dir: Path,
    *,
    speech_parquet_dir: Path | None = None,
    return_metadata: bool = False,
    allow_mixed_teacher_sources: bool = False,
    allow_teacher_labels: bool = False,
):
    p = parquet_dir
    gold = pd.read_parquet(p / "speech_gold_labels.parquet")
    rhetoric = pd.read_parquet(p / "speech_rhetoric_labels.parquet")

    label_source = (
        gold["label_source"].fillna("unknown").astype(str).str.casefold()
        if "label_source" in gold.columns
        else pd.Series("teacher", index=gold.index)
    )
    if not allow_teacher_labels and not label_source.isin({"human", "adjudicated"}).all():
        raise ValueError(
            "speech_gold_labels contains non-human labels; pass "
            "allow_teacher_labels=True only for teacher-agreement experiments"
        )

    teacher_columns = [column for column in ("model", "prompt_version", "temperature") if column in gold.columns]
    teacher_sources = gold[teacher_columns].drop_duplicates() if teacher_columns else pd.DataFrame()
    if len(teacher_sources) > 1 and not allow_mixed_teacher_sources:
        raise ValueError(
            "Speech gold labels contain mixed teacher sources; pass "
            "allow_mixed_teacher_sources=True only after explicit review"
        )

    # parse raw_response JSON into numeric columns
    probs = []
    for r in gold["raw_response"].fillna("{}").astype(str):
        try:
            probs.append(json.loads(r))
        except Exception:
            # fallback: try eval-ish parsing
            try:
                probs.append(json.loads(r.replace("'", '"')))
            except Exception:
                probs.append({})

    probs_df = pd.DataFrame(probs).fillna(0.0)

    # Ensure consistent ordering of columns
    probs_df = probs_df.reindex(sorted(probs_df.columns), axis=1).fillna(0.0)

    # merge by speech_id
    gold_idx = gold[["speech_id", "category"]].reset_index(drop=True)
    df = gold_idx.join(probs_df)
    # attach rhetoric features
    rhet = rhetoric[["speech_id", "irony", "sarcasm", "posturing", "none", "top_label"]]
    df = df.merge(rhet, on="speech_id", how="left")

    # fill missing rhetoric numeric values with 0
    for c in ["irony", "sarcasm", "posturing", "none"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
        else:
            df[c] = 0.0

    # one-hot encode top_label if present
    if "top_label" in df.columns:
        ohe = pd.get_dummies(df["top_label"].fillna("<none>"), prefix="rhet_top")
        df = pd.concat([df.drop(columns=["top_label"]), ohe], axis=1)

    # final feature set: all columns except speech_id and category
    feature_cols = [c for c in df.columns if c not in ("speech_id", "category")]

    X = df[feature_cols].astype(float).fillna(0.0)
    y = df["category"].astype(str).fillna("unknown")

    metadata = {
        "groups": df["speech_id"].astype(str).to_numpy(),
        "label_sources": sorted(label_source.unique().tolist()),
    }
    if not teacher_sources.empty:
        metadata["teacher_sources"] = teacher_sources.to_dict("records")
    if speech_parquet_dir is not None:
        speech_dates: dict[str, pd.Timestamp] = {}
        for path in sorted(speech_parquet_dir.glob("*.parquet")):
            columns = pd.read_parquet(path).columns
            date_column = next((column for column in ("datum", "date") if column in columns), None)
            if date_column is None or "anforande_id" not in columns:
                continue
            speech_frame = pd.read_parquet(path, columns=["anforande_id", date_column])
            for speech_id, value in speech_frame.itertuples(index=False, name=None):
                if pd.notna(speech_id) and str(speech_id) not in speech_dates:
                    speech_dates[str(speech_id)] = pd.to_datetime(value, errors="coerce", utc=True)
        metadata["dates"] = pd.Series(df["speech_id"].astype(str).map(speech_dates), index=df.index)

    if return_metadata:
        return X, y, feature_cols, metadata
    return X, y, feature_cols


def _split_data(X, y, groups, dates, strategy: str, random_state: int = 42):
    if strategy == "temporal" and dates is not None and dates.notna().sum() >= len(dates) * 0.8:
        cutoff = dates.dropna().quantile(0.8)
        train_mask = dates <= cutoff
        test_mask = dates > cutoff
        if train_mask.any() and test_mask.any() and y[train_mask].nunique() == y.nunique():
            return X.loc[train_mask], X.loc[test_mask], y[train_mask], y[test_mask], groups[train_mask], groups[test_mask]
        LOG.warning("Temporal split lacks class coverage; falling back to grouped split")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], groups[train_idx], groups[test_idx]


def train(
    X,
    y,
    out_path: Path,
    tune: bool = False,
    n_iter: int = 12,
    n_jobs: int = 1,
    *,
    groups=None,
    dates=None,
    split_strategy: str = "grouped",
    teacher_sources=None,
):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if LGBMClassifier is None:
        raise RuntimeError("lightgbm not available in the environment")

    if groups is None:
        groups = np.arange(len(y_enc))
    groups = np.asarray(groups)
    y_encoded = pd.Series(y_enc, index=X.index)
    X_train, X_test, y_train, y_test, train_groups, test_groups = _split_data(
        X, y_encoded, groups, dates, split_strategy
    )
    if set(train_groups) & set(test_groups):
        raise RuntimeError("Grouped training/test split contains overlapping groups")

    if tune:
        param_dist = {
            "num_leaves": [31, 63, 127],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "n_estimators": [100, 200, 500],
            "min_child_samples": [5, 10, 20, 50],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }

        base = LGBMClassifier(
            objective="multiclass",
            random_state=42,
            class_weight="balanced",
            n_jobs=n_jobs,
        )
        search = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
            n_jobs=n_jobs,
            verbose=1,
            scoring="f1_macro",
        )
        search.fit(X_train, y_train, groups=train_groups)
        model = search.best_estimator_
        LOG.info("Best params: %s", search.best_params_)
    else:
        model = LGBMClassifier(
            objective="multiclass",
            random_state=42,
            class_weight="balanced",
            n_estimators=200,
            n_jobs=n_jobs,
        )
        model.fit(X_train, y_train)

    # evaluate
    preds = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    report = classification_report(y_test, preds, target_names=le.classes_, zero_division=0)
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "log_loss": float(log_loss(y_test, probabilities, labels=np.arange(len(le.classes_)))),
        "n_test": int(len(y_test)),
        "n_train_groups": int(len(set(train_groups))),
        "n_test_groups": int(len(set(test_groups))),
        "split_strategy": split_strategy,
        "teacher_sources": teacher_sources or [],
    }
    print("Evaluation report:\n", report)
    print("Evaluation metrics:\n", json.dumps(metrics, indent=2))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "label_encoder": le,
            "feature_columns": list(X.columns),
            "training_metadata": metrics,
        },
        out_path,
    )
    print("Saved model to:", out_path)

    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", default="data/parquet", help="Parquet export directory")
    p.add_argument("--speech-parquet-dir", default="data/speeches/parquet", help="Optional speech source directory for dates")
    p.add_argument("--out", default="models/speech_meta_clf_parquet.pkl", help="Output model path")
    p.add_argument("--tune", action="store_true", help="Run randomized hyperparameter search")
    p.add_argument("--n-iter", type=int, default=12)
    p.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs to use (passed to RandomizedSearchCV and LightGBM).")
    p.add_argument("--split-strategy", choices=("grouped", "temporal"), default="grouped")
    p.add_argument("--allow-mixed-teacher-sources", action="store_true")
    p.add_argument("--allow-teacher-labels", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)

    # Limit native threadpools early to avoid oversubscription
    try:
        limit_threads(args.n_jobs)
    except Exception:
        pass

    X, y, cols, metadata = load_data(
        Path(args.parquet_dir),
        speech_parquet_dir=Path(args.speech_parquet_dir),
        return_metadata=True,
        allow_mixed_teacher_sources=args.allow_mixed_teacher_sources,
        allow_teacher_labels=args.allow_teacher_labels,
    )
    LOG.info("Loaded %d samples and %d features", X.shape[0], X.shape[1])
    train(
        X,
        y,
        Path(args.out),
        tune=args.tune,
        n_iter=args.n_iter,
        n_jobs=args.n_jobs,
        groups=metadata["groups"],
        dates=metadata.get("dates"),
        split_strategy=args.split_strategy,
        teacher_sources=metadata.get("teacher_sources"),
    )


if __name__ == "__main__":
    main()
