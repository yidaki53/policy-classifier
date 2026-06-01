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
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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


def load_data(parquet_dir: Path):
    p = parquet_dir
    gold = pd.read_parquet(p / "speech_gold_labels.parquet")
    rhetoric = pd.read_parquet(p / "speech_rhetoric_labels.parquet")

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

    return X, y, feature_cols


def train(X, y, out_path: Path, tune: bool = False, n_iter: int = 12, n_jobs: int = 1):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if LGBMClassifier is None:
        raise RuntimeError("lightgbm not available in the environment")

    # quick train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    if tune:
        param_dist = {
            "num_leaves": [31, 63, 127],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "n_estimators": [100, 200, 500],
            "min_child_samples": [5, 10, 20, 50],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }

        base = LGBMClassifier(objective="multiclass", random_state=42, n_jobs=n_jobs)
        search = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=3,
            n_jobs=n_jobs,
            verbose=1,
            scoring="accuracy",
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        LOG.info("Best params: %s", search.best_params_)
    else:
        model = LGBMClassifier(objective="multiclass", random_state=42, n_estimators=200, n_jobs=n_jobs)
        model.fit(X_train, y_train)

    # evaluate
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, target_names=le.classes_, zero_division=0)
    print("Evaluation report:\n", report)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "label_encoder": le, "feature_columns": list(X.columns)}, out_path)
    print("Saved model to:", out_path)

    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", default="data/parquet", help="Parquet export directory")
    p.add_argument("--out", default="models/speech_meta_clf_parquet.pkl", help="Output model path")
    p.add_argument("--tune", action="store_true", help="Run randomized hyperparameter search")
    p.add_argument("--n-iter", type=int, default=12)
    p.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs to use (passed to RandomizedSearchCV and LightGBM).")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)

    # Limit native threadpools early to avoid oversubscription
    try:
        limit_threads(args.n_jobs)
    except Exception:
        pass

    X, y, cols = load_data(Path(args.parquet_dir))
    LOG.info("Loaded %d samples and %d features", X.shape[0], X.shape[1])
    train(X, y, Path(args.out), tune=args.tune, n_iter=args.n_iter, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
