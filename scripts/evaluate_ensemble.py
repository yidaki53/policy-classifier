#!/usr/bin/env python3
"""Evaluate trained ensemble model on test split of gold labels.

Usage:
    uv run python3 scripts/evaluate_ensemble.py --db data/swedish_parliament.db
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

if os.getenv("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.getenv("HF_TOKEN"))

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions
from swedish_parliament_policy_classifier.nlp.topic_modeler import load_topic_distributions
from swedish_parliament_policy_classifier.classifier.ensemble import (
    prepare_training_data_from_gold_labels,
    load_meta_classifier,
)
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher
from swedish_parliament_policy_classifier.classifier.pipeline import score_motion


def evaluate(db_path: str = "data/swedish_parliament.db"):
    conn = init_db(db_path)
    defs = load_definitions()
    topic_dists = load_topic_distributions()

    matcher = None
    try:
        matcher = EmbeddingMatcher()
        if matcher.model is None:
            matcher = None
    except Exception as e:
        print(f"Embedding matcher unavailable: {e}", file=sys.stderr)
        matcher = None

    model = load_meta_classifier()
    if model is None:
        print("No trained ensemble model found. Run train_models.py first.", file=sys.stderr)
        sys.exit(1)

    print("Building features for test split...", file=sys.stderr)
    X_test, y_test, category_names = prepare_training_data_from_gold_labels(
        conn,
        topic_distributions=topic_dists,
        categories=defs,
        scorer_func=score_motion,
        embedding_matcher=matcher,
        split="test",
    )

    print(f"Test set: {len(X_test)} samples", file=sys.stderr)

    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    # Use the same label encoder that was used during training
    le = model["label_encoder"]

    y_test_enc = le.transform(y_test)
    clf = model["clf"]
    expected_names = model.get("_feature_names")
    if expected_names is None:
        from swedish_parliament_policy_classifier.classifier.ensemble import _build_feature_names
        expected_names = _build_feature_names(category_names)

    import pandas as pd
    if not hasattr(X_test, "columns"):
        X_test = pd.DataFrame(np.asarray(X_test, dtype=np.float32), columns=expected_names)
    else:
        X_test = X_test.copy()
        missing = [c for c in expected_names if c not in X_test.columns]
        if missing:
            for c in missing:
                X_test[c] = 0.0
        # Drop any unexpected extra columns and enforce training-time order.
        X_test = X_test[expected_names]

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test_enc, y_pred)
    print(
        classification_report(
            y_test_enc,
            y_pred,
            labels=range(len(le.classes_)),
            target_names=le.classes_,
        ),
        file=sys.stderr,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test_enc, y_pred, labels=range(len(le.classes_)))
    print("\nConfusion matrix (rows=true, cols=pred):", file=sys.stderr)
    print("  " + "  ".join(le.classes_), file=sys.stderr)
    for i, row in enumerate(cm):
        print(f"{le.classes_[i]:12s} " + "  ".join(f"{c:3d}" for c in row), file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/swedish_parliament.db")
    args = parser.parse_args()
    evaluate(db_path=args.db)


if __name__ == "__main__":
    main()
