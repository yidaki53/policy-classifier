#!/usr/bin/env python3
"""Hybrid ensemble: LightGBM meta-classifier augmented with BERT [CLS] embeddings.

Extracts [CLS] token embeddings from the fine-tuned KBLab Swedish BERT classifier
and concatenates them with the existing keyword/embedding/topic/metadata features.
Retrains LightGBM on the combined feature vector.

Usage:
    uv run python scripts/train_hybrid_ensemble.py --db data/swedish_parliament.db --bert-model models/transformer_ideology_classifier/final
"""

import argparse
import json
import os
import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except Exception as e:
    print(f"Missing transformers/torch: {e}", file=sys.stderr)
    HAS_TRANSFORMERS = False
    sys.exit(1)

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions, score_motion
from swedish_parliament_policy_classifier.nlp.topic_modeler import load_topic_distributions
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher
from swedish_parliament_policy_classifier.classifier.ensemble import (
    prepare_training_data_from_gold_labels,
    train_meta_classifier,
    load_meta_classifier,
    _build_feature_names,
)

BATCH_SIZE = 32
MAX_LEN = 512


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}


def extract_cls_embeddings(texts: List[str], model_path: str, batch_size: int = BATCH_SIZE, device: str = "cuda") -> np.ndarray:
    """Extract [CLS] embeddings from a fine-tuned BERT classifier."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # Access base BERT encoder
    bert_encoder = model.bert if hasattr(model, "bert") else model.base_model
    bert_encoder.eval()
    bert_encoder.to(device)

    dataset = TextDataset(texts, tokenizer, max_len=MAX_LEN)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_cls = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
            # outputs.last_hidden_state shape: (batch, seq_len, hidden)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            all_cls.append(cls_emb.cpu().numpy())

    return np.concatenate(all_cls, axis=0)


def prepare_hybrid_data(
    conn,
    topic_distributions,
    categories,
    scorer_func,
    embedding_matcher,
    split: str,
    bert_cls: Optional[np.ndarray] = None,
    bert_dim: int = 768,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Prepare feature matrix with optional BERT [CLS] concatenation."""
    X_base, y, category_names = prepare_training_data_from_gold_labels(
        conn,
        topic_distributions=topic_distributions,
        categories=categories,
        scorer_func=scorer_func,
        embedding_matcher=embedding_matcher,
        split=split,
    )

    if bert_cls is not None:
        # Concatenate BERT CLS features
        cls_df = pd.DataFrame(
            bert_cls,
            columns=[f"bert_cls_{i}" for i in range(bert_cls.shape[1])],
            index=X_base.index if hasattr(X_base, "index") else None,
        )
        X_combined = pd.concat([X_base.reset_index(drop=True), cls_df.reset_index(drop=True)], axis=1)
    else:
        X_combined = X_base

    return X_combined, y, category_names


def train_hybrid_ensemble(
    db_path: str = "data/swedish_parliament.db",
    bert_model_path: str = "models/transformer_ideology_classifier/final",
    output_model_path: Optional[str] = None,
):
    if not HAS_TRANSFORMERS:
        sys.exit(1)

    conn = init_db(db_path)
    defs = load_definitions()
    topic_dists = load_topic_distributions()
    category_names = sorted(defs.keys())

    matcher = None
    try:
        matcher = EmbeddingMatcher()
        if matcher.model is None:
            matcher = None
    except Exception as e:
        print(f"Embedding matcher unavailable: {e}", file=sys.stderr)

    # Load texts for BERT CLS extraction
    def _load_texts_for_split(split: str) -> List[str]:
        cur = conn.cursor()
        cur.execute("""
            SELECT COALESCE(nm.text, a.text) AS text
            FROM augmented_gold_labels a
            LEFT JOIN normalized_motions nm ON a.motion_id = nm.id
            WHERE a.split = ? AND COALESCE(nm.text, a.text) IS NOT NULL
              AND LENGTH(COALESCE(nm.text, a.text)) > 50
        """, (split,))
        rows = cur.fetchall()
        texts = []
        for row in rows:
            t = row[0] or ""
            if len(t) > 4000:
                t = t[:4000]
            texts.append(t)
        return texts

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Extracting BERT [CLS] embeddings on {device}...", file=sys.stderr)

    train_texts = _load_texts_for_split("train")
    val_texts = _load_texts_for_split("val")
    test_texts = _load_texts_for_split("test")

    print(f"Train texts: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}", file=sys.stderr)

    train_cls = extract_cls_embeddings(train_texts, bert_model_path, device=device)
    val_cls = extract_cls_embeddings(val_texts, bert_model_path, device=device)
    test_cls = extract_cls_embeddings(test_texts, bert_model_path, device=device)

    print(f"BERT CLS shapes: train={train_cls.shape}, val={val_cls.shape}, test={test_cls.shape}", file=sys.stderr)

    # Prepare base + BERT features
    print("Preparing hybrid feature matrices...", file=sys.stderr)
    X_train, y_train, _ = prepare_hybrid_data(
        conn, topic_dists, defs, score_motion, matcher, "train", bert_cls=train_cls
    )
    X_val, y_val, _ = prepare_hybrid_data(
        conn, topic_dists, defs, score_motion, matcher, "val", bert_cls=val_cls
    )
    X_test, y_test, _ = prepare_hybrid_data(
        conn, topic_dists, defs, score_motion, matcher, "test", bert_cls=test_cls
    )

    print(f"Hybrid feature dims: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}", file=sys.stderr)

    # Train LightGBM on combined features
    print("Training hybrid LightGBM meta-classifier...", file=sys.stderr)
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    if output_model_path is None:
        output_model_path = Path("models") / "hybrid_ensemble_meta_clf.pkl"
    else:
        output_model_path = Path(output_model_path)

    train_meta_classifier(X_train, y_train, category_names, model_path=output_model_path)

    # Evaluate
    model = load_meta_classifier(model_path=output_model_path)
    clf = model["clf"]

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test_enc, y_pred)
    print(f"\nHybrid ensemble test accuracy: {acc:.3f}", file=sys.stderr)
    print(
        classification_report(
            y_test_enc,
            y_pred,
            labels=range(len(le.classes_)),
            target_names=le.classes_,
        ),
        file=sys.stderr,
    )

    # Compare with baseline LightGBM
    baseline = load_meta_classifier()
    if baseline is not None:
        baseline_clf = baseline["clf"]
        baseline_le = baseline["label_encoder"]
        expected_names = baseline.get("_feature_names")
        if expected_names is None:
            expected_names = _build_feature_names(category_names)

        # Re-prepare base-only test features for fair comparison
        X_test_base, _, _ = prepare_training_data_from_gold_labels(
            conn, topic_distributions=topic_dists, categories=defs,
            scorer_func=score_motion, embedding_matcher=matcher, split="test",
        )
        y_test_baseline_enc = baseline_le.transform(y_test)
        y_pred_base = baseline_clf.predict(X_test_base)
        base_acc = accuracy_score(y_test_baseline_enc, y_pred_base)
        print(f"\nBaseline LightGBM test accuracy: {base_acc:.3f}", file=sys.stderr)
        print(f"Hybrid improvement: {acc - base_acc:+.3f}", file=sys.stderr)

    return acc


def main():
    parser = argparse.ArgumentParser(description="Train hybrid BERT+LightGBM ensemble")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--bert-model", default="models/transformer_ideology_classifier/final")
    parser.add_argument("--output", default=None, help="Path to save hybrid model")
    args = parser.parse_args()

    train_hybrid_ensemble(
        db_path=args.db,
        bert_model_path=args.bert_model,
        output_model_path=args.output,
    )


if __name__ == "__main__":
    main()
