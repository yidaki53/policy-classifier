#!/usr/bin/env python3
"""Quick evaluation of saved MLP meta-learner on test set."""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

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

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128], dropout=0.3):
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

def main():
    model_path = Path(DEFAULT_MODEL_PATH)
    if not model_path.exists():
        print(f"No saved model at {model_path}", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(str(model_path), weights_only=False)
    hidden_dims = ckpt["hidden_dims"]
    input_dim = ckpt["input_dim"]
    classes = ckpt["le_classes"]
    num_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(input_dim, num_classes, hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    conn = init_db("data/swedish_parliament.db")
    defs = load_definitions()
    topic_dists = load_topic_distributions()
    matcher = None
    try:
        matcher = EmbeddingMatcher()
        if matcher.model is None:
            matcher = None
    except Exception:
        pass

    from swedish_parliament_policy_classifier.classifier.pipeline import score_motion
    print("Generating test features...", file=sys.stderr)
    X_test, y_test, _ = prepare_training_data_from_gold_labels(
        conn, topic_distributions=topic_dists, categories=defs,
        scorer_func=score_motion, embedding_matcher=matcher, split="test",
    )
    X_test_np = np.asarray(X_test, dtype=np.float32)
    le = LabelEncoder()
    le.fit(classes)
    y_test_enc = le.transform(y_test)

    test_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_test_np, dtype=torch.float32),
        torch.tensor(y_test_enc, dtype=torch.long),
    )
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128)

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
    print(classification_report(test_truths, test_preds, target_names=classes, zero_division=0), file=sys.stderr)

    # LightGBM baseline
    lgbm = load_meta_classifier()
    if lgbm is not None:
        lgbm_clf = lgbm["clf"]
        le_lgbm = lgbm["label_encoder"]
        expected_names = lgbm.get("_feature_names")
        if expected_names is None:
            expected_names = _build_feature_names(sorted(defs.keys()))
        X_test_df = pd.DataFrame(X_test_np, columns=expected_names)
        y_test_lgbm_enc = le_lgbm.transform(y_test)
        y_pred_lgbm = lgbm_clf.predict(X_test_df)
        lgbm_acc = accuracy_score(y_test_lgbm_enc, y_pred_lgbm)
        print(f"\nLightGBM baseline test accuracy: {lgbm_acc:.3f}", file=sys.stderr)
        print(f"MLP improvement vs LightGBM: {test_acc - lgbm_acc:+.3f}", file=sys.stderr)

if __name__ == "__main__":
    main()
