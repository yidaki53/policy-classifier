"""Train a simple supervised multi-label classifier using deterministic labels.

This script bootstraps a reproducible supervised fallback by using the
deterministic rule labels stored in the `classifications` table as noisy
training targets. It trains a `TfidfVectorizer` + `OneVsRestClassifier(LogisticRegression)`
and saves the fitted model and label encoder into `models/`.

Usage:
    python -m swedish_parliament_policy_classifier.scripts.train_supervised
"""

from pathlib import Path
import argparse
import logging
import sqlite3
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline

from swedish_parliament_policy_classifier.db.schema import init_db, get_connection

LOG = logging.getLogger(__name__)


def load_training_data(conn: sqlite3.Connection, weight_threshold: float = 0.05):
    cur = conn.cursor()
    cur.execute("SELECT id, text, title FROM normalized_motions")
    rows = cur.fetchall()

    X = []
    y = []
    for r in rows:
        mid = r[0]
        text = (r[2] or "") + "\n" + (r[1] or "")
        # fetch categories for this motion that meet threshold
        cur.execute(
            "SELECT category, normalized_weight FROM classifications WHERE motion_id = ?",
            (mid,),
        )
        cats = [cr[0] for cr in cur.fetchall() if cr[1] and cr[1] >= weight_threshold]
        if not cats:
            continue
        X.append(text)
        y.append(cats)

    return X, y


def train_and_save(conn, out_dir: Path, weight_threshold: float = 0.05):
    X, y = load_training_data(conn, weight_threshold=weight_threshold)
    if not X:
        LOG.error("No training data found (increase weight_threshold or run deterministic classifier)")
        return

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=2000))),
    ])

    LOG.info("Training classifier on %d samples and %d labels", len(X), Y.shape[1])
    pipeline.fit(X, Y)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / "supervised_clf.joblib")
    joblib.dump(mlb, out_dir / "supervised_mlb.joblib")
    print(f"Wrote supervised classifier and label binarizer to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--out", default="models", help="Output directory for saved models")
    parser.add_argument("--threshold", type=float, default=0.05, help="Minimum normalized_weight to treat as positive")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    conn = init_db(args.db)
    train_and_save(conn, Path(args.out), weight_threshold=args.threshold)


if __name__ == "__main__":
    main()
