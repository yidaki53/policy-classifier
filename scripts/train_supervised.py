"""Train a simple supervised multi-label classifier using deterministic labels (Parquet-first).

Reads `data/parquet/normalized_motions.parquet` and `data/parquet/classifications.parquet`
to assemble noisy training targets and saves the resulting model into `models/`.
"""

from pathlib import Path
import argparse
import logging
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline

import pandas as pd

LOG = logging.getLogger(__name__)


def load_training_data_parquet(normalized_parquet: str | Path = "data/parquet/normalized_motions.parquet", classifications_parquet: str | Path = "data/parquet/classifications.parquet", weight_threshold: float = 0.05):
    nm_p = Path(normalized_parquet)
    cls_p = Path(classifications_parquet)
    if not nm_p.exists() or not cls_p.exists():
        LOG.error("Missing normalized motions or classifications parquet files")
        return [], []

    nm = pd.read_parquet(nm_p)
    cls = pd.read_parquet(cls_p)

    X = []
    y = []
    # group classifications by motion_id
    cls_grouped = cls.groupby("motion_id")
    for _, r in nm.iterrows():
        mid = str(r.get("id"))
        text = (r.get("title") or "") + "\n" + (r.get("text") or "")
        if mid not in cls_grouped.groups:
            continue
        group = cls_grouped.get_group(mid)
        cats = [str(c) for c in group.loc[group["normalized_weight"] >= weight_threshold, "category"].tolist()]
        if not cats:
            continue
        X.append(text)
        y.append(cats)

    return X, y


def train_and_save_parquet(out_dir: Path, normalized_parquet: str | Path = "data/parquet/normalized_motions.parquet", classifications_parquet: str | Path = "data/parquet/classifications.parquet", weight_threshold: float = 0.05):
    X, y = load_training_data_parquet(normalized_parquet, classifications_parquet, weight_threshold=weight_threshold)
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
    parser.add_argument("--normalized", default="data/parquet/normalized_motions.parquet")
    parser.add_argument("--classifications", default="data/parquet/classifications.parquet")
    parser.add_argument("--out", default="models", help="Output directory for saved models")
    parser.add_argument("--threshold", type=float, default=0.05, help="Minimum normalized_weight to treat as positive")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train_and_save_parquet(Path(args.out), normalized_parquet=args.normalized, classifications_parquet=args.classifications, weight_threshold=args.threshold)


if __name__ == "__main__":
    main()
