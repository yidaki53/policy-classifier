#!/usr/bin/env python3
"""Self-training loop for speeches: collect high-confidence pseudo-labels and optionally retrain.

Scans `data/speeches/parquet/*.parquet` for unlabeled speeches, scores them
with a speech meta-classifier, and inserts high-confidence predictions into
`speech_self_training_labels` for later retraining.

Usage:
    uv run python3 scripts/self_train_speech.py --db data/swedish_parliament.db --confidence 0.90 --max-per-class 500 --retrain
"""

import argparse
import os
import sys
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

if os.getenv("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.getenv("HF_TOKEN"))

import numpy as np
import pandas as pd

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.definitions.loader import load_verified_definitions
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher
from swedish_parliament_policy_classifier.classifier.ensemble import build_feature_vector
from swedish_parliament_policy_classifier.orchestration.speech_pipeline import train_and_save_speech_meta_classifier


def _build_lemma_kw_index(categories):
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
    scores: Dict[str, float] = {}
    if not text:
        return scores
    text_l = text.lower()
    for lemma_key, cat_kw_pairs in kw_index.items():
        if lemma_key in text_l:
            for cat_name, _ in cat_kw_pairs:
                scores[cat_name] = scores.get(cat_name, 0.0) + 1.0
    return scores


def _list_speech_parquets() -> List[Path]:
    d = Path("data") / "speeches" / "parquet"
    return sorted(d.glob("*.parquet"))


def fetch_unlabeled_speeches(conn, labeled_ids: set):
    parquet_files = _list_speech_parquets()
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf, columns=["anforande_id", "anforandetext", "parti"]) if "parti" in pd.read_parquet(pf, nrows=0).columns else pd.read_parquet(pf, columns=["anforande_id", "anforandetext"])
        except Exception:
            try:
                df = pd.read_parquet(pf)
            except Exception:
                continue
        for _, r in df.iterrows():
            sid = r.get("anforande_id")
            if sid is None:
                continue
            sid_key = str(sid)
            if sid_key in labeled_ids:
                continue
            text = r.get("anforandetext") or ""
            party = r.get("parti") if "parti" in r.index else None
            yield (sid_key, text, None, None, party)


def predict_batch(
    rows: List[Tuple],
    categories,
    kw_index: Dict,
    matcher: Optional[EmbeddingMatcher],
    clf,
) -> List[Tuple[str, str, float]]:
    texts = [r[1] or "" for r in rows]

    kw_scores = [_keyword_scores_for_text(t, kw_index) for t in texts]

    emb_scores = [{} for _ in texts]
    if matcher is not None and matcher.model is not None:
        try:
            if not hasattr(matcher, "_cached_cat_embs"):
                matcher._cached_cat_embs = matcher.build_category_embeddings(categories)
            # batch encode
            text_embs = matcher.encode(texts)
            from sklearn.preprocessing import normalize
            import numpy as _np
            text_embs = normalize(text_embs)
            cat_names = list(matcher._cached_cat_embs.keys())
            cat_vectors = normalize(np.stack([matcher._cached_cat_embs[n] for n in cat_names]))
            sims = text_embs @ cat_vectors.T
            for i in range(len(texts)):
                emb_scores[i] = {cat_names[j]: float(sims[i, j]) for j in range(len(cat_names))}
        except Exception:
            emb_scores = [{} for _ in texts]

    feature_vecs = []
    category_names = sorted(categories.keys())
    for i, row in enumerate(rows):
        motion_id = row[0]
        text = texts[i]
        topic_vec = None
        date_days_ago = None
        doc_type = None
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

    import pandas as pd
    X = pd.concat(feature_vecs, ignore_index=True) if feature_vecs else pd.DataFrame()

    # Ensure model feature alignment is handled by predict_with_meta_classifier; but here we call clf.predict_proba directly
    # clf may be a dict (loader.save_pickle output) or sklearn estimator
    probs = None
    if isinstance(clf, dict):
        # speech model saved via train_and_save_speech_meta_classifier returns a Path; use loader.load_pickle externally
        raise RuntimeError("Pass an sklearn estimator object (clf) to predict_batch")
    else:
        probs = clf.predict_proba(X)

    results = []
    names = sorted(categories.keys())
    for i, row in enumerate(rows):
        best_idx = int(np.argmax(probs[i]))
        confidence = float(probs[i][best_idx])
        pred_cat = names[best_idx]
        results.append((row[0], pred_cat, confidence))
    return results


def self_train_speech(
    db_path: str = "data/swedish_parliament.db",
    confidence_threshold: float = 0.90,
    max_per_class: int = 500,
    retrain: bool = False,
    model_path: Optional[str] = None,
):
    conn = init_db(db_path)
    cur = conn.cursor()
    categories = load_verified_definitions()

    # Load or train seed speech meta-classifier
    if model_path is None:
        model_path = "models/speech_meta_clf.pkl"
    if not Path(model_path).exists():
        print("Seed speech meta-classifier not found; training a seed model...", file=sys.stderr)
        train_and_save_speech_meta_classifier(db_path=db_path, out_path=model_path, tune=False)

    from swedish_parliament_policy_classifier.io.loader import load_pickle
    model_obj = load_pickle(Path(model_path))
    clf = model_obj["clf"]
    category_names = sorted(categories.keys())

    # Build labeled id set from speech_gold_labels + augmented + self-training table (if any)
    cur.execute("SELECT speech_id FROM speech_gold_labels")
    labeled = {str(r[0]) for r in cur.fetchall()}
    try:
        cur.execute("SELECT speech_id FROM augmented_speech_gold_labels")
        labeled.update({str(r[0]) for r in cur.fetchall()})
    except Exception:
        pass
    try:
        cur.execute("SELECT speech_id FROM speech_self_training_labels")
        labeled.update({str(r[0]) for r in cur.fetchall()})
    except Exception:
        pass

    # Build keyword index
    print("Building keyword index...", file=sys.stderr)
    kw_index = _build_lemma_kw_index(categories)

    # Embedding matcher
    matcher = None
    try:
        matcher = EmbeddingMatcher()
        if matcher.model is None:
            matcher = None
    except Exception as e:
        print(f"Embedding matcher unavailable: {e}", file=sys.stderr)

    # Score unlabeled speeches
    collected: Dict[str, List[Tuple[str, float]]] = {cat: [] for cat in category_names}
    total_scored = 0
    batch_size = 2000
    batch = []

    print("Scanning unlabeled speeches and scoring...", file=sys.stderr)
    for row in fetch_unlabeled_speeches(conn, labeled):
        batch.append(row)
        if len(batch) >= batch_size:
            preds = predict_batch(batch, categories, kw_index, matcher, clf)
            for sid, pred_cat, conf in preds:
                if conf >= confidence_threshold:
                    collected[pred_cat].append((sid, conf))
            total_scored += len(batch)
            batch = []
            print(f"  Scored {total_scored} speeches so far...", file=sys.stderr)
    # final batch
    if batch:
        preds = predict_batch(batch, categories, kw_index, matcher, clf)
        for sid, pred_cat, conf in preds:
            if conf >= confidence_threshold:
                collected[pred_cat].append((sid, conf))
        total_scored += len(batch)

    print(f"Scored total {total_scored} speeches. Preparing pseudo-labels...", file=sys.stderr)

    pseudo_labels = []
    for cat, items in collected.items():
        items.sort(key=lambda x: -x[1])
        kept = items[:max_per_class]
        print(f"Class {cat}: collected {len(items)}, keeping {len(kept)} (thr={confidence_threshold})", file=sys.stderr)
        for sid, conf in kept:
            pseudo_labels.append((sid, cat, conf))

    if not pseudo_labels:
        print("No high-confidence pseudo-labels found. Exiting.", file=sys.stderr)
        return

    # Insert into speech_self_training_labels
    cur.execute("""
        CREATE TABLE IF NOT EXISTS speech_self_training_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            speech_id TEXT NOT NULL,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("DELETE FROM speech_self_training_labels")
    now = datetime.now(timezone.utc).isoformat()
    for sid, cat, conf in pseudo_labels:
        cur.execute(
            "INSERT INTO speech_self_training_labels (speech_id, category, confidence, created_at) VALUES (?, ?, ?, ?)",
            (sid, cat, float(conf), now),
        )
    conn.commit()

    print(f"Inserted {len(pseudo_labels)} pseudo-labels into speech_self_training_labels.", file=sys.stderr)

    if retrain:
        print("Retraining speech meta-classifier including pseudo-labels...", file=sys.stderr)
        train_and_save_speech_meta_classifier(db_path=db_path, out_path=model_path, tune=False)
        print("Retrain complete.", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description="Self-train speech classifier")
    p.add_argument("--db", default="data/swedish_parliament.db")
    p.add_argument("--confidence", type=float, default=0.90)
    p.add_argument("--max-per-class", type=int, default=500)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--model", default=None, help="Path to seed speech meta-classifier pickle")
    args = p.parse_args()

    self_train_speech(db_path=args.db, confidence_threshold=args.confidence, max_per_class=args.max_per_class, retrain=args.retrain, model_path=args.model)


if __name__ == "__main__":
    main()
