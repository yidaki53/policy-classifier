#!/usr/bin/env python3
"""Debug script: trace types, shapes, and column names through the training pipeline."""

import warnings
import sys

# Elevate the sklearn feature-name warning to an error so we can catch it
warnings.filterwarnings("error", message=".*does not have valid feature names.*", category=UserWarning)

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions, score_motion
from swedish_parliament_policy_classifier.nlp.topic_modeler import load_topic_distributions
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher
from swedish_parliament_policy_classifier.classifier.ensemble import (
    build_feature_vector,
    _build_feature_names,
    prepare_training_data_from_gold_labels,
    train_meta_classifier,
    load_meta_classifier,
    predict_with_meta_classifier,
)

def trace_pipeline():
    conn = init_db("data/swedish_parliament.db")
    defs = load_definitions()
    topic_dists = load_topic_distributions()

    matcher = None
    try:
        matcher = EmbeddingMatcher()
        if matcher.model is None:
            matcher = None
    except Exception as e:
        print(f"Matcher unavailable: {e}", file=sys.stderr)

    # --- Step 1: Raw SQL query ---
    import sqlite3
    cur = conn.cursor()
    cur.execute("""
        SELECT a.motion_id, a.category, COALESCE(nm.text, a.text) AS text,
               nm.date, nm.doc_type, nm.party
        FROM augmented_gold_labels a
        LEFT JOIN normalized_motions nm ON a.motion_id = nm.id
        WHERE a.split = 'train'
          AND COALESCE(nm.text, a.text) IS NOT NULL
          AND LENGTH(COALESCE(nm.text, a.text)) > 0
    """)
    rows = cur.fetchall()
    print(f"[SQL] Returned {len(rows)} rows for split='train'")

    # --- Step 2: score_motion per row ---
    category_names = sorted(defs.keys())
    success = 0
    fail = 0
    fail_reasons = {}
    vecs = []
    for i, row in enumerate(rows[:10]):  # just first 10 for speed
        motion_id, cat, text, date, doc_type, party = row
        try:
            results = score_motion(
                motion_id, text[:2500], defs,
                party=party,
                embedding_matcher=matcher,
                use_zero_shot=False,
                use_meta_classifier=False,
            )
            keyword_scores = {r.category: r.raw_score for r in results}
            embedding_scores = {r.category: r.normalized_weight for r in results}
            topic_vec = topic_dists.get(motion_id)
            vec = build_feature_vector(
                keyword_scores=keyword_scores,
                embedding_scores=embedding_scores,
                topic_features=topic_vec,
                text_length=len(text),
                category_names=category_names,
                date_days_ago=None,
                doc_type=doc_type,
            )
            vecs.append(vec)
            success += 1
        except Exception as e:
            fail += 1
            fail_reasons[type(e).__name__] = fail_reasons.get(type(e).__name__, 0) + 1
            if i < 3:
                print(f"[FAIL] row {i} ({motion_id}): {e}")

    print(f"[score_motion] success={success} fail={fail} (first 10 rows)")
    if fail_reasons:
        print(f"[score_motion] failure types: {fail_reasons}")

    # --- Step 3: build_feature_vector output ---
    print(f"[build_feature_vector] type={type(vecs[0])} shape={vecs[0].shape} dtype={vecs[0].dtype}")
    print(f"[build_feature_vector] first 5 values: {vecs[0][:5]}")

    # --- Step 4: prepare_training_data_from_gold_labels ---
    print("\n[prepare_training_data_from_gold_labels] calling...")
    X, y, cats = prepare_training_data_from_gold_labels(
        conn, topic_dists, defs, score_motion, matcher, split="train"
    )
    print(f"[prepare_training_data_from_gold_labels] X type={type(X)}")
    if hasattr(X, "shape"):
        print(f"[prepare_training_data_from_gold_labels] X shape={X.shape}")
    if hasattr(X, "columns"):
        print(f"[prepare_training_data_from_gold_labels] X columns (first 5)={list(X.columns[:5])}")
        print(f"[prepare_training_data_from_gold_labels] X columns (last 5)={list(X.columns[-5:])}")
        print(f"[prepare_training_data_from_gold_labels] X is DataFrame: {isinstance(X, __import__('pandas').DataFrame)}")
    else:
        print(f"[prepare_training_data_from_gold_labels] X has NO columns attribute!")
    print(f"[prepare_training_data_from_gold_labels] y type={type(y)} shape={y.shape}")
    print(f"[prepare_training_data_from_gold_labels] classes={sorted(set(y))}")

    # --- Step 5: train_meta_classifier ---
    print("\n[train_meta_classifier] calling...")
    # Use tiny params for speed
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight
    from lightgbm import LGBMClassifier

    feature_names = _build_feature_names(category_names, max_topics=100)
    X_df = __import__('pandas').DataFrame(np.asarray(X, dtype=np.float32), columns=feature_names)
    print(f"[train_meta_classifier] X_df type={type(X_df)} shape={X_df.shape}")
    print(f"[train_meta_classifier] X_df columns (first 5)={list(X_df.columns[:5])}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_enc), y=y_enc)
    weight_dict = {i: w for i, w in enumerate(class_weights)}
    sample_weights = np.array([weight_dict[i] for i in y_enc])

    clf = LGBMClassifier(n_estimators=10, max_depth=3, learning_rate=0.1, num_leaves=7, verbose=-1)
    clf.fit(X_df, y_enc, sample_weight=sample_weights)
    print(f"[train_meta_classifier] clf.feature_names_in_ (first 5)={list(clf.feature_names_in_[:5])}")

    # Save and reload
    import pickle
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp_path = f.name
    with open(tmp_path, "wb") as f:
        pickle.dump({"clf": clf, "label_encoder": le, "category_names": category_names, "_feature_names": feature_names}, f)
    with open(tmp_path, "rb") as f:
        model = pickle.load(f)
    print(f"[pickle] reloaded model has _feature_names={model['_feature_names'] is not None}")

    # --- Step 6: predict_with_meta_classifier ---
    print("\n[predict_with_meta_classifier] calling...")
    try:
        probs = predict_with_meta_classifier(vecs[0], model, defs)
        print(f"[predict_with_meta_classifier] OK, probs keys={list(probs.keys())[:3]}")
    except UserWarning as e:
        print(f"[predict_with_meta_classifier] WARNING RAISED: {e}")
    except Exception as e:
        print(f"[predict_with_meta_classifier] ERROR: {type(e).__name__}: {e}")

    # --- Step 7: direct clf.predict on DataFrame ---
    print("\n[direct predict] calling clf.predict on X_df.head(3)...")
    try:
        preds = clf.predict(X_df.head(3))
        print(f"[direct predict] OK, preds={preds}")
    except UserWarning as e:
        print(f"[direct predict] WARNING RAISED: {e}")

    # --- Step 8: direct clf.predict on numpy array (should warn) ---
    print("\n[direct predict numpy] calling clf.predict on numpy array...")
    try:
        preds = clf.predict(X_df.head(3).to_numpy())
        print(f"[direct predict numpy] OK (no warning raised?)")
    except UserWarning as e:
        print(f"[direct predict numpy] WARNING RAISED (expected): {e}")

    print("\n=== TRACE COMPLETE ===")

if __name__ == "__main__":
    trace_pipeline()
