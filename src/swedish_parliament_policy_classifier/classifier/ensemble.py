"""Ensemble stacking meta-classifier using LightGBM.

Replaces hardcoded fixed signal weights with a learned meta-learner that
combines keyword matches, embedding similarities, topic distributions, and
metadata into adaptive per-motion category weights.
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from swedish_parliament_policy_classifier.io import loader

import numpy as np

LOG = logging.getLogger(__name__)


def _notna(val) -> bool:
    """Return True if *val* is a usable (non-null) value.

    Handles Python None, numpy NaN, and pandas NA without importing
    pandas at module level.
    """
    if val is None:
        return False
    try:
        # numpy NaN / pandas NA both make math.isnan or != work oddly;
        # the safest generic check is pandas.notna which handles all types.
        import pandas as _pd
        return bool(_pd.notna(val))
    except Exception:
        return True


def _get_default_model_path() -> Path:
    try:
        # ensemble.py is in src/<package>/classifier/ → repo root is parents[3]
        return Path(__file__).resolve().parents[3] / "models" / "ensemble_meta_clf.pkl"
    except Exception:
        return Path("models/ensemble_meta_clf.pkl")


def _build_feature_names(category_names: List[str], max_topics: int = 100) -> List[str]:
    """Generate human-readable feature names matching build_feature_vector ordering.

    Delegates to `classifier.features.get_feature_names` to keep the canonical
    feature-name generation logic in one place for easier testing and evolution.
    """
    try:
        from swedish_parliament_policy_classifier.classifier.features import get_feature_names

        return get_feature_names(category_names, max_topics=max_topics)
    except Exception:
        # Fallback to internal logic if the helper is unavailable
        names = []
        for cat in category_names:
            names.append(f"kw_{cat}")
        for cat in category_names:
            names.append(f"emb_{cat}")
        for cat in category_names:
            names.append(f"zs_{cat}")
        for cat in category_names:
            names.append(f"bert_cls_{cat}")
        for i in range(max_topics):
            names.append(f"topic_{i}")
        names.extend(["rhet_irony", "rhet_sarcasm", "rhet_posturing", "rhet_none"])
        names.extend(["text_len_log", "recency_years", "doc_mot", "doc_prop", "doc_votering"])
        return names


def build_feature_vector(
    keyword_scores: Dict[str, float],
    embedding_scores: Dict[str, float],
    topic_features: Optional[List[float]],
    text_length: int,
    category_names: List[str],
    date_days_ago: Optional[float] = None,
    doc_type: Optional[str] = None,
    max_topics: int = 100,
    zero_shot_scores: Optional[Dict[str, float]] = None,
    bert_cls_scores: Optional[Dict[str, float]] = None,
    rhetoric_scores: Optional[Dict[str, float]] = None,
) -> "pd.DataFrame":
    """Build a named 1-row feature DataFrame for the meta-classifier.

    Features are concatenated in a fixed order:
    1. keyword match counts (one per category)
    2. embedding cosine similarities (one per category)
    3. zero-shot NLI entailment scores (one per category)
    4. transformer classifier probabilities (one per category)
    5. topic distribution (variable length, 0-padded)
    6. text length (normalized log)
    7. recency (days ago, normalized)
    8. doc_type one-hot (mot, prop, votering)

    Returns a 1-row pandas DataFrame with explicit column names.
    """
    import pandas as pd

    features = []
    names = _build_feature_names(category_names, max_topics=max_topics)

    # 1. Keyword scores per category
    for cat in category_names:
        features.append(keyword_scores.get(cat, 0.0))

    # 2. Embedding scores per category
    for cat in category_names:
        features.append(embedding_scores.get(cat, 0.0))

    # 3. Zero-shot NLI scores per category
    zs = zero_shot_scores or {}
    for cat in category_names:
        features.append(zs.get(cat, 0.0))

    # 4. Transformer classifier probabilities per category
    bc = bert_cls_scores or {}
    for cat in category_names:
        features.append(bc.get(cat, 0.0))

    # 5. Topic distribution (pad/truncate to fixed length)
    if topic_features is not None:
        if isinstance(topic_features, (float, int)):
            vec = [float(topic_features)]
        else:
            vec = list(topic_features)[:max_topics]
        vec = vec + [0.0] * (max_topics - len(vec))
    else:
        vec = [0.0] * max_topics
    features.extend(vec)

    # 5b. Rhetoric signals (irony, sarcasm, posturing, none)
    rs = rhetoric_scores or {}
    features.append(float(rs.get("irony", 0.0)))
    features.append(float(rs.get("sarcasm", 0.0)))
    features.append(float(rs.get("posturing", 0.0)))
    features.append(float(rs.get("none", 0.0)))

    # 4. Text length (log-normalized)
    features.append(np.log1p(text_length))

    # 5. Recency (normalized)
    if date_days_ago is not None:
        features.append(date_days_ago / 365.25)
    else:
        features.append(0.0)

    # 6. Doc type one-hot
    dt_map = {"mot": [1, 0, 0], "prop": [0, 1, 0], "votering": [0, 0, 1]}
    features.extend(dt_map.get(doc_type, [0, 0, 0]))

    return pd.DataFrame([features], columns=names, dtype=np.float32)


def prepare_training_data_from_gold_labels(
    conn,
    topic_distributions: Dict[str, List[float]],
    categories: Dict[str, object],
    scorer_func,
    embedding_matcher,
    split: str = "train",
    zero_shot_func=None,
    bert_cls_func=None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare feature matrix X and label vector y from augmented gold labels.

    Uses the gold-label split (train/test/val) and computes base signals
    (keyword, embedding, topic, metadata) for each labeled motion.
    """
    import sqlite3
    cur = conn.cursor()
    from swedish_parliament_policy_classifier.db.readers import fetch_augmented_gold_label_rows

    # Use DB-backed reader which prefers parquet exports for normalized_motions
    rows = fetch_augmented_gold_label_rows(conn, split=split)

    category_names = sorted(categories.keys())

    X_list = []
    y_list = []

    LOG.info("Preparing training data from %d gold-label rows (split=%s)...", len(rows), split)

    try:
        from tqdm import tqdm
        row_iter = tqdm(rows, desc=f"Features {split}", file=sys.stderr)
    except Exception:
        row_iter = rows

    for row in row_iter:
        motion_id = row[0]
        true_category = row[1]
        # Sanitize pd.NA / np.nan values that leak from parquet-backed readers
        text = str(row[2]) if _notna(row[2]) else ""
        date = str(row[3]) if _notna(row[3]) else None
        doc_type = str(row[4]) if _notna(row[4]) else None
        party = str(row[5]) if _notna(row[5]) else None

        # Run scorer for keyword/regex signals only (disable meta-clf to
        # avoid circularity, skip embeddings and zero-shot — we compute
        # embeddings separately below to get the raw cosine similarities).
        try:
            results = scorer_func(
                motion_id, text[:2500], categories,
                party=party,
                embedding_matcher=None,
                use_zero_shot=False,
                meta_clf=None,
            )
        except Exception as e:
            LOG.warning("Scorer failed for %s: %s", motion_id, e)
            continue

        # Extract keyword/regex raw counts from scorer results
        keyword_scores = {}
        for r in results:
            keyword_scores[r.category] = r.raw_score

        # Compute raw embedding cosine similarities directly so training
        # features match the inference path in score_motion (Stage 2).
        embedding_scores = {}
        if embedding_matcher is not None:
            try:
                if not hasattr(embedding_matcher, "_cached_cat_embs"):
                    embedding_matcher._cached_cat_embs = (
                        embedding_matcher.build_category_embeddings(categories)
                    )
                emb_matches = embedding_matcher.match(
                    text[:2500],
                    embedding_matcher._cached_cat_embs,
                    top_k=len(categories),
                )
                embedding_scores = {
                    name: float(score) for name, score in emb_matches
                }
            except Exception as e:
                LOG.warning("Embedding matching failed for %s: %s", motion_id, e)

        # Compute zero-shot NLI scores
        zs_scores: Dict[str, float] = {}
        if zero_shot_func is not None:
            try:
                zs_scores = zero_shot_func(text[:1500])
            except Exception as e:
                LOG.warning("Zero-shot failed for %s: %s", motion_id, e)

        # Compute transformer classifier probabilities
        bert_scores: Dict[str, float] = {}
        if bert_cls_func is not None:
            try:
                bert_scores = bert_cls_func(text[:2500])
            except Exception as e:
                LOG.warning("Transformer predict failed for %s: %s", motion_id, e)

        # Get topic features
        topic_vec = topic_distributions.get(motion_id)

        # Compute recency
        date_days_ago = None
        if date:
            from datetime import datetime, timezone
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
            date_days_ago=date_days_ago,
            doc_type=doc_type,
            category_names=category_names,
            zero_shot_scores=zs_scores,
            bert_cls_scores=bert_scores,
        )

        X_list.append(vec)
        y_list.append(true_category)

    import pandas as pd
    X = pd.concat(X_list, ignore_index=True) if X_list else pd.DataFrame()
    y = np.array(y_list)

    LOG.info("Training data shape: X=%s, classes=%s", X.shape, len(set(y)))
    return X, y, category_names


def train_meta_classifier(
    X,
    y: np.ndarray,
    category_names: List[str],
    model_path: Optional[Path] = None,
) -> object:
    """Train a LightGBM classifier with class balancing and save it.

    Converts input to numpy internally to avoid sklearn/LightGBM feature-name
    mismatch bugs, while preserving feature names for debugging.
    """
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight

    LOG.info("Training LightGBM meta-classifier on %d samples, %d classes...", len(X), len(category_names))

    import pandas as pd

    # Ensure X is a DataFrame with explicit column names.
    # If it already is one, reuse its names; otherwise reconstruct.
    if isinstance(X, pd.DataFrame) and hasattr(X, "columns"):
        X_df = X.astype(np.float32)
        feature_names = list(X_df.columns)
    else:
        feature_names = _build_feature_names(category_names, max_topics=100)
        X_df = pd.DataFrame(np.asarray(X, dtype=np.float32), columns=feature_names)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Compute balanced class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_encoded),
        y=y_encoded,
    )
    weight_dict = {i: w for i, w in enumerate(class_weights)}
    sample_weights = np.array([weight_dict[i] for i in y_encoded])

    clf = LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    clf.fit(X_df, y_encoded, sample_weight=sample_weights)

    # Save model + label encoder + feature names
    if model_path is None:
        model_path = _get_default_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    loader.save_pickle(model_path, {
        "clf": clf,
        "label_encoder": le,
        "category_names": category_names,
        "_feature_names": feature_names,
    })

    LOG.info("Saved ensemble meta-classifier to %s", model_path)
    return {"clf": clf, "label_encoder": le, "category_names": category_names, "_feature_names": feature_names}


def load_meta_classifier(model_path: Optional[Path] = None) -> Optional[Dict]:
    """Load a saved ensemble meta-classifier."""
    if model_path is None:
        model_path = _get_default_model_path()
    if not model_path.exists() and not Path(str(model_path) + ".zst").exists():
        return None
    return loader.load_pickle(model_path)


def predict_with_meta_classifier(
    feature_vector,
    meta_clf: Dict,
    categories: Dict[str, object],
) -> Dict[str, float]:
    """Predict category probabilities using the ensemble meta-classifier.

    Accepts a 1-D numpy array (or list) and ALWAYS reconstructs a named
    DataFrame before calling LightGBM. This guarantees sklearn never sees a
    bare ndarray and the feature-name validation passes.
    Returns a dict mapping category -> probability.
    """
    import pandas as pd

    clf = meta_clf["clf"]
    le = meta_clf["label_encoder"]
    category_names = meta_clf["category_names"]
    expected_names = meta_clf.get("_feature_names")

    # Build the exact feature names if the pickle is from an older model
    if expected_names is None:
        expected_names = _build_feature_names(category_names, max_topics=100)

    # Ensure input is a DataFrame with the exact column names used during training.
    # If expected columns (e.g. bert_cls_*) are missing, pad with zeros.
    if isinstance(feature_vector, pd.DataFrame):
        X = feature_vector.copy()
        missing = [c for c in expected_names if c not in X.columns]
        if missing:
            for c in missing:
                X[c] = 0.0
        X = X[expected_names]
    else:
        arr = np.asarray(feature_vector, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != len(expected_names):
            # Pad with zeros if dimensions mismatch (e.g. missing BERT features)
            padded = np.zeros((arr.shape[0], len(expected_names)), dtype=np.float32)
            padded[:, :arr.shape[1]] = arr
            arr = padded
        X = pd.DataFrame(arr, columns=expected_names)

    # Get probabilities
    probs = clf.predict_proba(X)[0]

    # Map back to category names
    result = {}
    for idx, prob in enumerate(probs):
        cat_name = le.inverse_transform([idx])[0]
        result[cat_name] = float(prob)

    # Ensure all categories from definitions have a value
    for cat in categories.keys():
        if cat not in result:
            result[cat] = 0.0

    # Normalize to sum to 1
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}

    return result
