#!/usr/bin/env python3
"""Train BERTopic and ensemble meta-classifier on gold-label data.

Usage:
    uv run python3 scripts/train_models.py --db data/swedish_parliament.db
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
from swedish_parliament_policy_classifier.nlp.topic_modeler import fit_topic_model, load_topic_distributions
from swedish_parliament_policy_classifier.classifier.ensemble import (
    prepare_training_data_from_gold_labels,
    train_meta_classifier,
    load_meta_classifier,
)
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher


def train_all(
    db_path: str = "data/swedish_parliament.db",
    force_retrain_topic: bool = False,
    force_retrain_ensemble: bool = False,
):
    conn = init_db(db_path)
    cur = conn.cursor()
    defs = load_definitions()

    # Check if topic model already exists
    existing_topics = load_topic_distributions()
    if not existing_topics or force_retrain_topic:
        print("Training BERTopic model...", file=sys.stderr)
        # Fetch all normalized motions with text
        cur.execute("SELECT id, text FROM normalized_motions WHERE text IS NOT NULL AND text != ''")
        rows = cur.fetchall()

        motion_ids = []
        texts = []
        for row in rows:
            motion_ids.append(row[0])
            texts.append(row[1] or "")

        if len(texts) < 100:
            print(f"Not enough motions ({len(texts)}) to train BERTopic. Skipping.", file=sys.stderr)
        else:
            _, topic_dists = fit_topic_model(texts, motion_ids)
            print(f"Trained BERTopic on {len(texts)} motions.", file=sys.stderr)
    else:
        print(f"Using existing BERTopic model with {len(existing_topics)} motion topic distributions.", file=sys.stderr)
        topic_dists = existing_topics

    # Check if ensemble exists
    meta_clf = load_meta_classifier()
    if not meta_clf or force_retrain_ensemble:
        print("Training ensemble meta-classifier from gold labels...", file=sys.stderr)

        matcher = None
        try:
            matcher = EmbeddingMatcher()
            if matcher.model is None:
                matcher = None
        except Exception as e:
            print(f"Embedding matcher unavailable: {e}", file=sys.stderr)
            matcher = None

        # Import score_motion from the refactored pipeline
        from swedish_parliament_policy_classifier.classifier.pipeline import score_motion

        # Load zero-shot and transformer prediction functions for feature extraction
        zs_func = None
        try:
            from swedish_parliament_policy_classifier.nlp.zero_shot_values import zero_shot_score
            zs_func = zero_shot_score
            print("Zero-shot NLI model will be used for training features.", file=sys.stderr)
        except Exception as e:
            print(f"Zero-shot unavailable: {e}", file=sys.stderr)

        bert_func = None
        try:
            from swedish_parliament_policy_classifier.classifier.transformer_predict import predict_proba
            bert_func = predict_proba
            print("Transformer ideology classifier will be used for training features.", file=sys.stderr)
        except Exception as e:
            print(f"Transformer classifier unavailable: {e}", file=sys.stderr)

        # Train on gold-label training split
        X_train, y_train, category_names = prepare_training_data_from_gold_labels(
            conn,
            topic_distributions=topic_dists,
            categories=defs,
            scorer_func=score_motion,
            embedding_matcher=matcher,
            split="train",
            zero_shot_func=zs_func,
            bert_cls_func=bert_func,
        )

        if len(X_train) < 100:
            print(f"Not enough training data ({len(X_train)} samples). Skipping ensemble training.", file=sys.stderr)
        else:
            print(f"Training on {len(X_train)} samples, {len(category_names)} classes...", file=sys.stderr)
            train_meta_classifier(X_train, y_train, category_names)
            print(f"Trained ensemble on {len(X_train)} samples.", file=sys.stderr)

            # Evaluate on test set
            from sklearn.metrics import classification_report, accuracy_score
            import numpy as np
            import pandas as pd

            X_test, y_test, _ = prepare_training_data_from_gold_labels(
                conn,
                topic_distributions=topic_dists,
                categories=defs,
                scorer_func=score_motion,
                embedding_matcher=matcher,
                split="test",
                zero_shot_func=zs_func,
                bert_cls_func=bert_func,
            )
            if len(X_test) > 0:
                model = load_meta_classifier()
                clf = model["clf"]
                le = model["label_encoder"]
                expected_names = model.get("_feature_names")

                # Align test features to the model's expected column order
                if expected_names is not None and isinstance(X_test, pd.DataFrame):
                    missing = [c for c in expected_names if c not in X_test.columns]
                    for c in missing:
                        X_test[c] = 0.0
                    X_test = X_test[expected_names]

                # Drop test samples whose category the model has never seen
                known_cats = set(le.classes_)
                mask = np.array([cat in known_cats for cat in y_test])
                if not mask.all():
                    n_dropped = int((~mask).sum())
                    print(f"Dropping {n_dropped} test samples with categories unseen during training.", file=sys.stderr)
                    X_test = X_test.loc[mask].reset_index(drop=True)
                    y_test = y_test[mask]

                y_test_enc = le.transform(y_test)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test_enc, y_pred)
                print(f"\nTest accuracy: {acc:.3f}", file=sys.stderr)
                print(
                    classification_report(
                        y_test_enc,
                        y_pred,
                        labels=range(len(le.classes_)),
                        target_names=list(le.classes_),
                    ),
                    file=sys.stderr,
                )
    else:
        print("Using existing ensemble meta-classifier.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Train BERTopic and ensemble meta-classifier")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--force-retrain-topic", action="store_true", help="Force retraining BERTopic")
    parser.add_argument("--force-retrain-ensemble", action="store_true", help="Force retraining ensemble")
    args = parser.parse_args()

    train_all(
        db_path=args.db,
        force_retrain_topic=args.force_retrain_topic,
        force_retrain_ensemble=args.force_retrain_ensemble,
    )


if __name__ == "__main__":
    main()
