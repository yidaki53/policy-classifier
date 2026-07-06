#!/usr/bin/env python3
"""Hyperparameter tuning for the ensemble meta-classifier.

Uses Optuna to search for optimal LightGBM hyperparameters, maximizing
validation accuracy while avoiding overfitting.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from swedish_parliament_policy_classifier.exports import load_definitions
from swedish_parliament_policy_classifier.classifier.scorer import score_motion
from swedish_parliament_policy_classifier.classifier.ensemble import (
    build_feature_vector,
    prepare_training_data_from_gold_labels,
)
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher
from swedish_parliament_policy_classifier.nlp.topic_modeler import load_topic_distributions

try:
    import optuna
    from lightgbm import LGBMClassifier

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("WARNING: optuna or lightgbm not available. Install with: uv pip add optuna lightgbm")

import sqlite3
from swedish_parliament_policy_classifier.exports import get_connection


def _build_feature_vector_for_motion(
    motion_id: str,
    text: str,
    categories: Dict,
    embedding_matcher,
    topic_dists: Dict,
    category_names: list,
) -> Optional[pd.DataFrame]:
    """Build feature vector for a single motion text."""
    from swedish_parliament_policy_classifier.nlp.zero_shot_values import zero_shot_score
    from swedish_parliament_policy_classifier.classifier.transformer_predict import predict_proba as bert_predict

    # Get keyword scores
    results = score_motion(
        motion_id, text[:2500], categories,
        embedding_matcher=None, use_zero_shot=False, meta_clf=None
    )
    keyword_scores = {r.category: r.raw_score for r in results}
    
    # Compute embedding scores
    embedding_scores = {}
    if embedding_matcher is not None:
        try:
            if not hasattr(embedding_matcher, "_cached_cat_embs") or embedding_matcher._cached_cat_embs is None:
                embedding_matcher._cached_cat_embs = embedding_matcher.build_category_embeddings(categories)
            emb_matches = embedding_matcher.match(text[:2500], embedding_matcher._cached_cat_embs, top_k=len(categories))
            embedding_scores = {name: float(score) for name, score in emb_matches}
        except Exception:
            pass
    
    # Zero-shot scores
    zs_scores = {}
    try:
        zs_scores = zero_shot_score(text[:1500])
    except Exception:
        pass
    
    # BERT scores
    bert_scores = {}
    try:
        bert_scores = bert_predict(text[:2500])
    except Exception:
        pass
    
    # Topic features
    topic_vec = topic_dists.get(motion_id)
    
    # Build feature vector
    vec = build_feature_vector(
        keyword_scores=keyword_scores,
        embedding_scores=embedding_scores,
        topic_features=topic_vec,
        text_length=len(text),
        category_names=category_names,
        zero_shot_scores=zs_scores,
        bert_cls_scores=bert_scores,
    )
    
    return vec


def load_training_data(
    db_path: Path,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load gold labels and compute features for training using the existing pipeline."""
    from swedish_parliament_policy_classifier.classifier.ensemble import prepare_training_data_from_gold_labels
    
    conn = get_connection(db_path)
    categories = load_definitions()
    category_names = sorted(categories.keys())
    
    matcher = None
    try:
        matcher = EmbeddingMatcher()
        if matcher.model is None:
            matcher = None
    except Exception:
        pass
    
    topic_dists = load_topic_distributions() or {}
    
    # Import scoring and transformer functions
    from swedish_parliament_policy_classifier.classifier.scorer import score_motion
    zs_func = None
    try:
        from swedish_parliament_policy_classifier.nlp.zero_shot_values import zero_shot_score
        zs_func = zero_shot_score
    except Exception:
        pass
    
    bert_func = None
    try:
        from swedish_parliament_policy_classifier.classifier.transformer_predict import predict_proba
        bert_func = predict_proba
    except Exception:
        pass
    
    # Use the existing gold-label pipeline
    X, y, category_names = prepare_training_data_from_gold_labels(
        conn=conn,
        topic_distributions=topic_dists,
        categories=categories,
        scorer_func=score_motion,
        embedding_matcher=matcher,
        split=split,
        zero_shot_func=zs_func,
        bert_cls_func=bert_func,
    )
    
    # Sample if needed
    if max_samples and len(y) > max_samples:
        idx = np.random.choice(len(y), size=max_samples, replace=False)
        X = X.iloc[idx].reset_index(drop=True) if isinstance(X, pd.DataFrame) else X[idx]
        y = y[idx]
    
    return X, y, category_names


def objective(trial, X: np.ndarray, y: np.ndarray, category_names: list):
    """Optuna objective function."""
    from sklearn.model_selection import cross_val_score
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_enc), y=y_enc)
    weight_dict = {i: w for i, w in enumerate(class_weights)}
    sample_weights = np.array([weight_dict[i] for i in y_enc])
    
    # Hyperparameter search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "random_state": 42,
        "verbose": -1,
    }
    
    clf = LGBMClassifier(**params)
    
    # Cross-validation
    try:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
        for train_idx, val_idx in cv.split(X, y_enc):
            clf_fold = LGBMClassifier(**params)
            clf_fold.fit(
                X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx],
                y_enc[train_idx],
                sample_weight=sample_weights[train_idx],
            )
            score = clf_fold.score(
                X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx],
                y_enc[val_idx],
            )
            fold_scores.append(score)
        cv_score = np.mean(fold_scores)
    except Exception as e:
        print(f"CV failed: {e}")
        return 0.0
    
    return cv_score


def main(
    db_path: str = "data/swedish_parliament.db",
    output_path: Optional[str] = None,
    n_trials: int = 50,
    max_samples: Optional[int] = None,
):
    if not OPTUNA_AVAILABLE:
        print("ERROR: optuna and lightgbm required. Install with: uv pip add optuna lightgbm")
        return 1
    
    db_path = Path(db_path)
    if output_path is None:
        output_path = "models/ensemble_meta_clf_tuned.pkl.zst"
    output_path = Path(output_path)
    
    print(f"Loading training data from {db_path} (split=train)...")
    X, y, category_names = load_training_data(db_path, split="train", max_samples=max_samples)
    print(f"Loaded {len(y)} training samples with {len(category_names)} categories")
    print(f"Feature matrix shape: {X.shape}")
    
    # Optuna study
    study = optuna.create_study(direction="maximize", study_name="meta_clf_tuning")
    study.optimize(
        lambda trial: objective(trial, X, y, category_names),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    
    print("\n=== Best Trial ===")
    print(f"Best CV accuracy: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, val in study.best_params.items():
        print(f"  {key}: {val}")
    
    # Train final model with best params
    print("\nTraining final model with best hyperparameters...")
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_enc), y=y_enc)
    weight_dict = {i: w for i, w in enumerate(class_weights)}
    sample_weights = np.array([weight_dict[i] for i in y_enc])
    
    best_params = study.best_params
    best_params.update({"random_state": 42, "verbose": -1})
    
    clf = LGBMClassifier(**best_params)
    clf.fit(X, y_enc, sample_weight=sample_weights)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    X_val, y_val, _ = load_training_data(db_path, split="val", max_samples=max_samples)
    y_val_enc = le.transform(y_val)
    val_score = clf.score(X_val, y_val_enc)
    print(f"Validation accuracy: {val_score:.4f}")
    
    # Save model
    import pickle
    import zstandard as zstd
    
    model_state = {
        "clf": clf,
        "label_encoder": le,
        "category_names": category_names,
        "_feature_names": list(X.columns),
        "best_params": study.best_params,
        "cv_accuracy": study.best_value,
        "val_accuracy": val_score,
        "n_training_samples": len(y),
        "n_features": X.shape[1],
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        with zstd.ZstdCompressor().stream_writer(f) as compressor:
            pickle.dump(model_state, compressor)
    
    print(f"\nSaved tuned model to {output_path}")
    
    # Save tuning history
    history_path = output_path.with_suffix('.tuning_history.json')
    history = {
        "best_params": study.best_params,
        "best_cv_accuracy": study.best_value,
        "val_accuracy": val_score,
        "n_trials": len(study.trials),
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ],
    }
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    print(f"Saved tuning history to {history_path}")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune ensemble meta-classifier hyperparameters")
    parser.add_argument("--db", default="data/swedish_parliament.db", help="Database path")
    parser.add_argument("--output", default=None, help="Output model path")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--max-samples", type=int, default=None, help="Max training samples")
    args = parser.parse_args()
    
    exit(main(
        db_path=args.db,
        output_path=args.output,
        n_trials=args.trials,
        max_samples=args.max_samples,
    ))