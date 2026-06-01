#!/usr/bin/env python3
"""Self-training: score unlabeled motions in bulk, keep high-confidence rare-class candidates.

Usage:
    poetry run python3 scripts/self_train.py --db data/swedish_parliament.db --confidence 0.75 --max-per-class 500
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

if os.getenv("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.getenv("HF_TOKEN"))

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions
from swedish_parliament_policy_classifier.classifier.pipeline import score_motion
from swedish_parliament_policy_classifier.nlp.topic_modeler import load_topic_distributions
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher
from swedish_parliament_policy_classifier.classifier.ensemble import (
    build_feature_vector,
    load_meta_classifier,
    train_meta_classifier,
    prepare_training_data_from_gold_labels,
)


def _build_lemma_kw_index(categories):
    """Fast lemma keyword index."""
    import spacy
    index: Dict[str, List[Tuple[str, str]]] = {}
    nlp = spacy.load("sv_core_news_sm", disable=["parser", "ner"])
    for name, cat in categories.items():
        for kw in cat.keywords or []:
            if not kw:
                continue
            doc = nlp(kw.lower())
            lemmas = [t.lemma_.lower() for t in doc if not t.is_space and not t.is_punct]
            key = " ".join(lemmas)
            index.setdefault(key, []).append((name, kw))
    return index


def _keyword_scores_for_text(text: str, kw_index: Dict) -> Dict[str, float]:
    """Return keyword match counts per category."""
    import spacy
    scores: Dict[str, float] = {}
    if not text:
        return scores
    text_l = text.lower()
    # Simple substring matching for speed (lemmatization already in kw_index keys)
    for lemma_key, cat_kw_pairs in kw_index.items():
        if lemma_key in text_l:
            for cat_name, _ in cat_kw_pairs:
                scores[cat_name] = scores.get(cat_name, 0.0) + 1.0
    return scores


def _embedding_scores_for_texts(
    texts: List[str],
    categories,
    matcher: EmbeddingMatcher,
) -> List[Dict[str, float]]:
    """Batch embedding similarity for a list of texts."""
    if matcher is None or not texts:
        return [{} for _ in texts]
    # Build category embeddings once
    if not hasattr(matcher, "_cached_cat_embs"):
        matcher._cached_cat_embs = matcher.build_category_embeddings(categories)
    cat_embs = matcher._cached_cat_embs
    results = []
    # Batch encode texts
    text_embeddings = matcher.model.encode(texts, show_progress_bar=False, batch_size=32)
    # Normalize
    from sklearn.preprocessing import normalize
    text_embeddings = normalize(text_embeddings)
    cat_names = list(cat_embs.keys())
    cat_vectors = normalize(np.stack([cat_embs[n] for n in cat_names]))
    # Cosine similarities = dot product of normalized vectors
    sims = text_embeddings @ cat_vectors.T  # shape: (n_texts, n_cats)
    for i in range(len(texts)):
        row = {cat_names[j]: float(sims[i, j]) for j in range(len(cat_names))}
        results.append(row)
    return results


def fetch_unlabeled_motions(conn, labeled_ids: set, batch_size: int = 5000):
    """Yield batches of unlabeled motions."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id, text, date, doc_type, party
        FROM normalized_motions
        WHERE text IS NOT NULL AND LENGTH(text) > 0
        ORDER BY id
    """)
    batch = []
    while True:
        row = cur.fetchone()
        if row is None:
            if batch:
                yield batch
            break
        motion_id = row[0]
        if motion_id in labeled_ids:
            continue
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []


def predict_batch(
    rows: List[Tuple],
    categories,
    kw_index: Dict,
    matcher: Optional[EmbeddingMatcher],
    topic_dists: Dict[str, List[float]],
    clf,
) -> List[Tuple[str, str, float]]:
    """Predict categories for a batch of motion rows. Returns (motion_id, pred_cat, confidence)."""
    texts = [r[1] or "" for r in rows]
    dates = [r[2] for r in rows]
    doc_types = [r[3] for r in rows]

    # Keyword scores
    kw_scores = [_keyword_scores_for_text(t, kw_index) for t in texts]

    # Embedding scores (batched)
    emb_scores = _embedding_scores_for_texts(texts, categories, matcher)

    # Build feature vectors
    category_names = sorted(categories.keys())
    feature_vecs = []
    for i, row in enumerate(rows):
        motion_id = row[0]
        text = texts[i]
        date = dates[i]
        doc_type = doc_types[i]

        topic_vec = topic_dists.get(motion_id)

        date_days_ago = None
        if date:
            try:
                dt = datetime.fromisoformat(date)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                date_days_ago = (datetime.now(timezone.utc) - dt).days
            except Exception:
                pass

        vec = build_feature_vector(
            keyword_scores=kw_scores[i],
            embedding_scores=emb_scores[i],
            topic_features=topic_vec,
            text_length=len(text),
            category_names=category_names,
            date_days_ago=date_days_ago,
            doc_type=doc_type,
        )
        feature_vecs.append(vec)

    X = pd.concat(feature_vecs, ignore_index=True) if feature_vecs else pd.DataFrame()
    probs = clf.predict_proba(X)

    results = []
    for i, row in enumerate(rows):
        motion_id = row[0]
        best_idx = int(np.argmax(probs[i]))
        confidence = float(probs[i][best_idx])
        pred_cat = category_names[best_idx]
        results.append((motion_id, pred_cat, confidence))
    return results


def self_train(
    db_path: str = "data/swedish_parliament.db",
    confidence_threshold: float = 0.75,
    max_per_class: int = 500,
    rare_classes: Optional[List[str]] = None,
):
    conn = init_db(db_path)
    cur = conn.cursor()
    defs = load_definitions()
    topic_dists = load_topic_distributions()
    model = load_meta_classifier()
    if model is None:
        print("No trained ensemble model found. Run train_models.py first.", file=sys.stderr)
        sys.exit(1)

    clf = model["clf"]
    category_names = sorted(defs.keys())

    if rare_classes is None:
        # Automatically detect rare classes from training data
        cur.execute("""
            SELECT category, COUNT(*) as cnt
            FROM augmented_gold_labels
            WHERE split = 'train'
            GROUP BY category
            ORDER BY cnt ASC
        """)
        rows = cur.fetchall()
        median_cnt = np.median([r[1] for r in rows])
        rare_classes = [r[0] for r in rows if r[1] < median_cnt]
        print(f"Auto-detected rare classes (< median {median_cnt:.0f}): {rare_classes}", file=sys.stderr)

    # Load already labeled motion IDs
    cur.execute("SELECT DISTINCT motion_id FROM augmented_gold_labels")
    labeled_ids = {r[0] for r in cur.fetchall()}
    print(f"Already labeled motions: {len(labeled_ids)}", file=sys.stderr)

    # Build keyword index
    print("Building keyword index...", file=sys.stderr)
    kw_index = _build_lemma_kw_index(defs)

    # Load embedding matcher
    matcher = None
    try:
        matcher = EmbeddingMatcher()
        if matcher.model is None:
            matcher = None
    except Exception as e:
        print(f"Embedding matcher unavailable: {e}", file=sys.stderr)

    # Score unlabeled motions in batches
    collected: Dict[str, List[Tuple[str, float]]] = {cat: [] for cat in rare_classes}
    total_scored = 0
    batch_size = 5000

    print(f"Scoring unlabeled motions (batch size {batch_size})...", file=sys.stderr)
    for batch_idx, batch in enumerate(fetch_unlabeled_motions(conn, labeled_ids, batch_size=batch_size)):
        try:
            preds = predict_batch(batch, defs, kw_index, matcher, topic_dists, clf)
        except Exception as e:
            print(f"Batch {batch_idx} failed: {e}", file=sys.stderr)
            continue

        for motion_id, pred_cat, confidence in preds:
            if pred_cat in collected and confidence >= confidence_threshold:
                collected[pred_cat].append((motion_id, confidence))

        total_scored += len(batch)
        if batch_idx % 10 == 0:
            counts = {cat: len(v) for cat, v in collected.items()}
            print(f"  Scored {total_scored} motions, collected so far: {counts}", file=sys.stderr)

        # Early stop if all rare classes are full
        if all(len(v) >= max_per_class for v in collected.values()):
            print("All rare classes filled. Stopping early.", file=sys.stderr)
            break

    # Sort by confidence and keep top max_per_class per rare category
    pseudo_labels = []
    for cat, items in collected.items():
        items.sort(key=lambda x: -x[1])
        kept = items[:max_per_class]
        print(f"Class {cat}: collected {len(items)}, keeping {len(kept)} (threshold {confidence_threshold})", file=sys.stderr)
        for motion_id, conf in kept:
            pseudo_labels.append((motion_id, cat, conf))

    if not pseudo_labels:
        print("No high-confidence pseudo-labels found. Exiting.", file=sys.stderr)
        return

    # Insert pseudo-labels into DB as a new table/flag
    print(f"Inserting {len(pseudo_labels)} pseudo-labels into self_training_labels...", file=sys.stderr)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS self_training_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            motion_id TEXT NOT NULL,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("DELETE FROM self_training_labels")
    now = datetime.now(timezone.utc).isoformat()
    for motion_id, cat, conf in pseudo_labels:
        cur.execute(
            "INSERT INTO self_training_labels (motion_id, category, confidence, created_at) VALUES (?, ?, ?, ?)",
            (motion_id, cat, conf, now),
        )
    conn.commit()
    print("Done.", file=sys.stderr)


def retrain_with_pseudo_labels(
    db_path: str = "data/swedish_parliament.db",
):
    """Retrain ensemble mixing gold labels + pseudo labels."""
    conn = init_db(db_path)
    defs = load_definitions()
    topic_dists = load_topic_distributions()
    model = load_meta_classifier()
    if model is None:
        print("No existing model. Run train_models.py first.", file=sys.stderr)
        sys.exit(1)

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM self_training_labels")
    n_pseudo = cur.fetchone()[0]
    print(f"Pseudo-labels available: {n_pseudo}", file=sys.stderr)

    if n_pseudo == 0:
        print("No pseudo-labels. Skipping retrain.", file=sys.stderr)
        return

    matcher = None
    try:
        matcher = EmbeddingMatcher()
        if matcher.model is None:
            matcher = None
    except Exception as e:
        print(f"Embedding matcher unavailable: {e}", file=sys.stderr)

    # Re-use pipeline scorer import
    from swedish_parliament_policy_classifier.classifier.pipeline import score_motion

    # Prepare gold training data
    X_gold, y_gold, category_names = prepare_training_data_from_gold_labels(
        conn,
        topic_distributions=topic_dists,
        categories=defs,
        scorer_func=score_motion,
        embedding_matcher=matcher,
        split="train",
    )

    # Prepare pseudo training data
    cur.execute("""
        SELECT s.motion_id, s.category, nm.text, nm.date, nm.doc_type, nm.party
        FROM self_training_labels s
        JOIN normalized_motions nm ON s.motion_id = nm.id
        WHERE nm.text IS NOT NULL AND LENGTH(nm.text) > 0
    """)
    rows = cur.fetchall()
    print(f"Building features for {len(rows)} pseudo-labeled motions...", file=sys.stderr)

    X_pseudo_list = []
    y_pseudo_list = []
    for row in rows:
        motion_id = row[0]
        true_category = row[1]
        text = row[2] or ""
        date = row[3]
        doc_type = row[4]
        party = row[5]

        try:
            results = score_motion(
                motion_id, text[:2500], defs,
                party=party,
                embedding_matcher=matcher,
                use_zero_shot=False,
                use_meta_classifier=False,
            )
        except Exception as e:
            continue

        keyword_scores = {}
        embedding_scores = {}
        for r in results:
            keyword_scores[r.category] = r.raw_score
            embedding_scores[r.category] = r.normalized_weight

        topic_vec = topic_dists.get(motion_id)
        date_days_ago = None
        if date:
            try:
                dt = datetime.fromisoformat(date)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                date_days_ago = (datetime.now(timezone.utc) - dt).days
            except Exception:
                pass

        vec = build_feature_vector(
            keyword_scores=keyword_scores,
            embedding_scores=embedding_scores,
            topic_features=topic_vec,
            text_length=len(text),
            category_names=category_names,
            date_days_ago=date_days_ago,
            doc_type=doc_type,
        )
        X_pseudo_list.append(vec)
        y_pseudo_list.append(true_category)

    import pandas as pd
    X_pseudo = pd.concat(X_pseudo_list, ignore_index=True) if X_pseudo_list else pd.DataFrame()
    y_pseudo = np.array(y_pseudo_list)

    # Combine
    X_combined = pd.concat([X_gold, X_pseudo], ignore_index=True)
    y_combined = np.concatenate([y_gold, y_pseudo])

    print(f"Combined training set: gold={len(X_gold)}, pseudo={len(X_pseudo)}, total={len(X_combined)}", file=sys.stderr)

    # Train
    train_meta_classifier(X_combined, y_combined, category_names)
    print("Retrained ensemble with pseudo-labels.", file=sys.stderr)

    # Evaluate on test
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import LabelEncoder

    X_test, y_test, _ = prepare_training_data_from_gold_labels(
        conn,
        topic_distributions=topic_dists,
        categories=defs,
        scorer_func=score_motion,
        embedding_matcher=matcher,
        split="test",
    )
    if len(X_test) > 0:
        le = LabelEncoder()
        le.fit(np.concatenate([y_combined, y_test]))
        y_test_enc = le.transform(y_test)
        model = load_meta_classifier()
        clf = model["clf"]
        expected_names = model.get("_feature_names")
        if expected_names is None:
            expected_names = _build_feature_names(category_names, max_topics=100)

        import pandas as pd
        X_test = pd.DataFrame(np.asarray(X_test, dtype=np.float32), columns=expected_names) if not hasattr(X_test, "columns") else X_test
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test_enc, y_pred)
        print(f"\nTest accuracy after self-training: {acc:.3f}", file=sys.stderr)
        print(
            classification_report(
                y_test_enc,
                y_pred,
                labels=range(len(le.classes_)),
                target_names=le.classes_,
            ),
            file=sys.stderr,
        )


def main():
    parser = argparse.ArgumentParser(description="Self-training for rare classes")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--retrain", action="store_true", help="Also retrain after collecting pseudo-labels")
    args = parser.parse_args()

    self_train(
        db_path=args.db,
        confidence_threshold=args.confidence,
        max_per_class=args.max_per_class,
    )
    if args.retrain:
        retrain_with_pseudo_labels(db_path=args.db)


if __name__ == "__main__":
    main()
