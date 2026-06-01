#!/usr/bin/env python3
"""Deep meta-learner: small MLP replacing LightGBM for the ensemble meta-classifier.

Trains a feed-forward neural network on the same feature vectors used by the
LightGBM ensemble (keyword scores, embedding similarities, topic distributions,
metadata). Compares test accuracy to the LightGBM baseline.

Usage:
    uv run python scripts/train_mlp_meta_learner.py --db data/swedish_parliament.db --epochs 50 --hidden 256 128
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

load_dotenv()
if os.getenv("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.getenv("HF_TOKEN"))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight
    HAS_TORCH = True
except Exception as e:
    print(f"Missing torch/sklearn: {e}", file=sys.stderr)
    HAS_TORCH = False
    sys.exit(1)

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions
from swedish_parliament_policy_classifier.nlp.topic_modeler import load_topic_distributions
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher
from swedish_parliament_policy_classifier.classifier.ensemble import (
    prepare_training_data_from_gold_labels,
    load_meta_classifier,
    _build_feature_names,
)

DEFAULT_MODEL_PATH = "models/mlp_meta_learner.pt"


class FeatureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = [256, 128], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(
    db_path: str = "data/swedish_parliament.db",
    hidden_dims: List[int] = [256, 128],
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    model_path: str = DEFAULT_MODEL_PATH,
    seed: int = 42,
):
    if not HAS_TORCH:
        sys.exit(1)

    torch.manual_seed(seed)
    np.random.seed(seed)

    conn = init_db(db_path)
    defs = load_definitions()
    topic_dists = load_topic_distributions()
    category_names = sorted(defs.keys())

    # Check for existing LightGBM baseline
    lgbm_model = load_meta_classifier()
    if lgbm_model is None:
        print("WARNING: No LightGBM baseline found. Train ensemble first for comparison.", file=sys.stderr)

    matcher = None
    try:
        matcher = EmbeddingMatcher()
        if matcher.model is None:
            matcher = None
    except Exception as e:
        print(f"Embedding matcher unavailable: {e}", file=sys.stderr)

    from swedish_parliament_policy_classifier.classifier.pipeline import score_motion

    print("Preparing training data...", file=sys.stderr)
    X_train, y_train, _ = prepare_training_data_from_gold_labels(
        conn, topic_distributions=topic_dists, categories=defs,
        scorer_func=score_motion, embedding_matcher=matcher, split="train",
    )
    X_val, y_val, _ = prepare_training_data_from_gold_labels(
        conn, topic_distributions=topic_dists, categories=defs,
        scorer_func=score_motion, embedding_matcher=matcher, split="val",
    )
    X_test, y_test, _ = prepare_training_data_from_gold_labels(
        conn, topic_distributions=topic_dists, categories=defs,
        scorer_func=score_motion, embedding_matcher=matcher, split="test",
    )

    if len(X_train) < 100:
        print("Not enough training data.", file=sys.stderr)
        sys.exit(1)

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    # Convert to numpy arrays
    X_train_np = np.asarray(X_train, dtype=np.float32)
    X_val_np = np.asarray(X_val, dtype=np.float32)
    X_test_np = np.asarray(X_test, dtype=np.float32)

    input_dim = X_train_np.shape[1]
    num_classes = len(le.classes_)

    print(f"Feature dim: {input_dim}, Classes: {num_classes}", file=sys.stderr)

    # Class weights for imbalance
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_train_enc)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = class_weights_tensor.to(device)

    # Datasets
    train_ds = FeatureDataset(X_train_np, y_train_enc)
    val_ds = FeatureDataset(X_val_np, y_val_enc)
    test_ds = FeatureDataset(X_test_np, y_test_enc)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2)

    # Model
    model = MLPClassifier(input_dim, num_classes, hidden_dims=hidden_dims).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print("Training MLP meta-learner...", file=sys.stderr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_preds = []
        val_truths = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_truths.extend(yb.cpu().numpy())

        val_acc = accuracy_score(val_truths, val_preds)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "le_classes": le.classes_.tolist(),
                "label2id": {cls: i for i, cls in enumerate(le.classes_)},
                "input_dim": input_dim,
                "hidden_dims": hidden_dims,
                "best_val_acc": best_val_acc,
            }, model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: train_loss={np.mean(train_losses):.4f} val_acc={val_acc:.3f} (best={best_val_acc:.3f} @ epoch {best_epoch+1})", file=sys.stderr)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}", file=sys.stderr)
            break

    # Load best and evaluate on test
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_preds = []
    test_truths = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_truths.extend(yb.cpu().numpy())

    test_acc = accuracy_score(test_truths, test_preds)
    print(f"\nTest accuracy (MLP): {test_acc:.3f}", file=sys.stderr)
    print(classification_report(test_truths, test_preds, target_names=le.classes_, zero_division=0), file=sys.stderr)

    # Compare with LightGBM baseline if available
    if lgbm_model is not None:
        lgbm_clf = lgbm_model["clf"]
        le_lgbm = lgbm_model["label_encoder"]
        expected_names = lgbm_model.get("_feature_names")
        if expected_names is None:
            expected_names = _build_feature_names(category_names)

        X_test_df = pd.DataFrame(X_test_np, columns=expected_names)
        y_test_lgbm_enc = le_lgbm.transform(y_test)
        y_pred_lgbm = lgbm_clf.predict(X_test_df)
        lgbm_acc = accuracy_score(y_test_lgbm_enc, y_pred_lgbm)
        print(f"\nLightGBM baseline test accuracy: {lgbm_acc:.3f}", file=sys.stderr)
        print(f"MLP improvement vs LightGBM: {test_acc - lgbm_acc:+.3f}", file=sys.stderr)

    return test_acc


def main():
    parser = argparse.ArgumentParser(description="Train MLP meta-learner replacing LightGBM")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--hidden", nargs="+", type=int, default=[256, 128], help="Hidden layer sizes")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_mlp(
        db_path=args.db,
        hidden_dims=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_path=args.model_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
