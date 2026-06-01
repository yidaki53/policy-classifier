"""Orchestration helpers for speech-focused pipelines.

Provides functions to export active-learning candidates from prediction CSVs
and to prepare / train a speech-only LightGBM meta-classifier using the
project's existing feature builder. The helpers are thin wrappers so the
steps remain reproducible and scriptable via the CLI scripts in `scripts/`.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tempfile
import subprocess
import sys

from swedish_parliament_policy_classifier.db import schema
from swedish_parliament_policy_classifier.definitions.loader import load_verified_definitions
from swedish_parliament_policy_classifier.classifier.ensemble import build_feature_vector, train_meta_classifier
from swedish_parliament_policy_classifier.io import loader

LOG = logging.getLogger(__name__)


def _find_latest_preds(logs_dir: Path = Path("logs"), prefix: str = "speech_eval_preds") -> Optional[Path]:
    # Prefer parquet predictions, fall back to CSV for compatibility
    candidates = list(logs_dir.glob(f"{prefix}_*.parquet")) + list(logs_dir.glob(f"{prefix}_*.csv"))
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _entropy(probs: List[float]) -> float:
    # numerical stability
    eps = 1e-12
    ent = 0.0
    for p in probs:
        p = max(min(float(p), 1.0 - eps), eps)
        ent -= p * math.log(p)
    return ent


def export_active_learning_candidates(
    db_path: str = "data/swedish_parliament.db",
    preds_csv: Optional[str] = None,
    top_n: int = 500,
    out_path: Optional[str] = None,
    include_preview_chars: int = 600,
) -> Path:
    """Export top-`top_n` high-entropy speech examples for annotation.

    If `preds_csv` is not provided the function picks the latest
    `logs/speech_eval_preds_*.csv` file. The output CSV contains
    `speech_id`, `truth`, `pred`, per-category `prob_*` columns, computed
    `entropy`, `top_prob`, `second_prob` and a short `text_preview` fetched
    from the database when available.
    """
    logs_dir = Path("logs")
    if preds_csv is None:
        latest = _find_latest_preds(logs_dir)
        if latest is None:
            raise FileNotFoundError("No prediction CSV found in logs/")
        preds_csv = str(latest)

    pred_path = Path(preds_csv)
    if pred_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(pred_path)
    else:
        df = pd.read_csv(pred_path)

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        raise ValueError("No probability columns found in predictions CSV")

    # compute entropy and top/second probs
    ent_list = []
    top1 = []
    top2 = []
    topcat = []
    for _, row in df.iterrows():
        probs = [row[c] for c in prob_cols]
        ent = _entropy(probs)
        ent_list.append(ent)
        sorted_idx = np.argsort(probs)[::-1]
        top1.append(float(probs[sorted_idx[0]]))
        top2.append(float(probs[sorted_idx[1]] if len(probs) > 1 else 0.0))
        topcat.append(prob_cols[sorted_idx[0]].replace("prob_", ""))

    df["entropy"] = ent_list
    df["top_prob"] = top1
    df["second_prob"] = top2
    df["top_category_from_probs"] = topcat

    # fetch short text preview: prefer speech parquet exports (load once),
    # otherwise fall back to normalized_motions via DB reader
    conn = schema.get_connection(db_path)
    previews = []

    speech_parquet_dir = Path("data") / "speeches" / "parquet"
    speech_text_map: dict[str, str] = {}
    if speech_parquet_dir.exists() and pd is not None:
        for pf in sorted(speech_parquet_dir.glob("*.parquet")):
            try:
                # try to read only the ID/text columns where supported
                try:
                    sdf = pd.read_parquet(pf, columns=["anforande_id", "anforandetext"])
                except Exception:
                    sdf = pd.read_parquet(pf)

                if "anforande_id" in sdf.columns and "anforandetext" in sdf.columns:
                    for _, r in sdf.iterrows():
                        sid_val = r["anforande_id"]
                        if sid_val is None:
                            continue
                        sid_key = str(sid_val)
                        if sid_key not in speech_text_map:
                            speech_text_map[sid_key] = r["anforandetext"] or ""
            except Exception as e:
                LOG.warning("Failed to read speech parquet %s: %s", pf, e)

    for sid in df["speech_id"]:
        sid_key = str(sid)
        txt = speech_text_map.get(sid_key, "")
        previews.append((txt or "")[:include_preview_chars])
    df["text_preview"] = previews

    df_sorted = df.sort_values("entropy", ascending=False).head(top_n)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    if out_path is None:
        out_path = out_dir / f"active_learning_candidates_{ts}.parquet"
    else:
        out_path = Path(out_path)
        if out_path.suffix.lower() == '.csv':
            out_path = out_path.with_suffix('.parquet')
    df_sorted.to_parquet(out_path, index=False, compression='zstd')
    LOG.info("Wrote active-learning candidates to %s", out_path)
    return out_path


def prepare_speech_training_data(
    db_path: str = "data/swedish_parliament.db",
    zero_shot_func=None,
    bert_cls_func=None,
    embedding_matcher=None,
    topic_distributions: Optional[Dict[str, List[float]]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Prepare X, y for speech gold labels using the same feature layout.

    This mirrors `ensemble.prepare_training_data_from_gold_labels` but reads
    labels from `speech_gold_labels` and is speech-aware (applies
    speech preprocessing when calling the lightweight scorer for keyword
    counts).
    """
    conn = schema.get_connection(db_path)
    cur = conn.cursor()
    # Primary gold labels
    cur.execute("SELECT speech_id, category FROM speech_gold_labels")
    rows = list(cur.fetchall())
    if not rows:
        raise RuntimeError("No rows found in speech_gold_labels; nothing to train on")

    # Optionally include augmented speech gold (synthetic / back-translated)
    augmented_map = {}
    try:
        cur.execute("SELECT speech_id, category, text FROM augmented_speech_gold_labels")
        aug_rows = cur.fetchall()
        for r in aug_rows:
            # r: (speech_id, category, text)
            sid = r[0]
            cat = r[1]
            txt = r[2]
            # prefer augmented text when building features later
            if sid is not None and txt is not None:
                augmented_map[str(sid)] = txt
            # append to training rows (as (speech_id, category)) so it's included
            rows.append((sid, cat))
    except Exception:
        augmented_map = {}

    # Optionally include pseudo / self-training labels
    try:
        cur.execute("SELECT speech_id, category FROM speech_self_training_labels")
        pseudo_rows = cur.fetchall()
        for r in pseudo_rows:
            rows.append((r[0], r[1]))
    except Exception:
        # table may not exist yet; ignore
        pass

    categories = load_verified_definitions()
    category_names = sorted(categories.keys())

    from swedish_parliament_policy_classifier.classifier.pipeline import score_motion

    X_list = []
    y_list = []

    # Load speech texts from parquet exports (map anforande_id -> anforandetext).
    # This avoids reading motions/parquet files which are unrelated to speech analysis.
    speech_parquet_dir = Path("data") / "speeches" / "parquet"
    speech_text_map: dict[str, str] = {}
    if speech_parquet_dir.exists() and pd is not None:
        for pf in sorted(speech_parquet_dir.glob("*.parquet")):
            try:
                try:
                    sdf = pd.read_parquet(pf, columns=["anforande_id", "anforandetext"])
                except Exception:
                    sdf = pd.read_parquet(pf)

                if "anforande_id" in sdf.columns and "anforandetext" in sdf.columns:
                    for _, r in sdf.iterrows():
                        sid_val = r["anforande_id"]
                        if sid_val is None:
                            continue
                        sid_key = str(sid_val)
                        if sid_key not in speech_text_map:
                            speech_text_map[sid_key] = r["anforandetext"] or ""
            except Exception as e:
                LOG.warning("Failed to read speech parquet %s: %s", pf, e)

    for sid, category in rows:
        # fetch speech text from parquet exports (do not use motions)
        sid_key = str(sid)
        # prefer augmented text when present
        if sid_key in augmented_map:
            text = augmented_map.get(sid_key, "")
        else:
            text = speech_text_map.get(sid_key, "")
        # load rhetoric signals if available
        try:
            cur.execute("SELECT irony, sarcasm, posturing, none, top_label FROM speech_rhetoric_labels WHERE speech_id = ?", (sid_key,))
            rr = cur.fetchone()
            if rr:
                rhetoric_scores = {"irony": float(rr[0] or 0.0), "sarcasm": float(rr[1] or 0.0), "posturing": float(rr[2] or 0.0), "none": float(rr[3] or 0.0)}
            else:
                rhetoric_scores = {}
        except Exception:
            rhetoric_scores = {}
        # Use scorer for keyword counts (speech-preprocessing enabled)
        try:
            results = score_motion(
                sid,
                text,
                categories,
                party=None,
                embedding_matcher=None,
                use_zero_shot=False,
                meta_clf=None,
                skip_policy_extraction=True,
                use_speech_preprocessing=True,
            )
        except Exception as e:
            LOG.warning("Scorer failed for %s: %s", sid, e)
            continue

        keyword_scores = {r.category: float(r.raw_score) for r in results}

        embedding_scores = {}
        if embedding_matcher is not None:
            try:
                if not hasattr(embedding_matcher, "_cached_cat_embs"):
                    embedding_matcher._cached_cat_embs = embedding_matcher.build_category_embeddings(categories)
                emb_matches = embedding_matcher.match(text[:2500], embedding_matcher._cached_cat_embs, top_k=len(categories))
                embedding_scores = {name: float(score) for name, score in emb_matches}
            except Exception as e:
                LOG.warning("Embedding match failed for %s: %s", sid, e)

        zs_scores = {}
        if zero_shot_func is not None:
            try:
                zs_scores = zero_shot_func(text)
            except Exception as e:
                LOG.warning("Zero-shot failed for %s: %s", sid, e)

        bert_scores = {}
        if bert_cls_func is not None:
            try:
                bert_scores = bert_cls_func(text)
            except Exception as e:
                LOG.warning("BERT predict failed for %s: %s", sid, e)

        # topic features (likely None for speeches but kept for API parity)
        topic_vec = None if topic_distributions is None else topic_distributions.get(sid)

        # date -> days ago (speech parquet exports typically don't include motion-level dates)
        date_days_ago = None

        vec = build_feature_vector(
            keyword_scores=keyword_scores,
            embedding_scores=embedding_scores,
            topic_features=topic_vec,
            text_length=len(text or ""),
            category_names=category_names,
            date_days_ago=date_days_ago,
            doc_type=None,
            zero_shot_scores=zs_scores,
            bert_cls_scores=bert_scores,
            rhetoric_scores=rhetoric_scores,
        )

        X_list.append(vec)
        y_list.append(category)

    X = pd.concat(X_list, ignore_index=True) if X_list else pd.DataFrame()
    y = np.array(y_list)
    return X, y, category_names


def _auto_label_missing_speeches(
    db_path: str,
    speech_ids: List[str],
    model: str = "llama3.1:8b",
    temp_val: float = 0.2,
    sleep: float = 0.05,
    include_preview_chars: int = 2000,
) -> None:
    """Run the zero-shot labeler on missing speeches and ingest the results.

    Writes a temporary CSV with `speech_id` and `text_preview`, invokes
    `scripts/label_rhetoric_zero_shot.py`, and ingests the produced CSV
    via `scripts/ingest_rhetoric_labels.py`.
    """
    if not speech_ids:
        LOG.info("No missing speeches to auto-label")
        return

    # Map speech_id -> short text preview using exported speech parquet files
    speech_parquet_dir = Path("data") / "speeches" / "parquet"
    speech_text_map: dict[str, str] = {}
    if speech_parquet_dir.exists() and pd is not None:
        for pf in sorted(speech_parquet_dir.glob("*.parquet")):
            try:
                try:
                    sdf = pd.read_parquet(pf, columns=["anforande_id", "anforandetext"])
                except Exception:
                    sdf = pd.read_parquet(pf)
                if "anforande_id" in sdf.columns and "anforandetext" in sdf.columns:
                    for _, r in sdf.iterrows():
                        sid_val = r["anforande_id"]
                        if sid_val is None:
                            continue
                        sid_key = str(sid_val)
                        if sid_key not in speech_text_map:
                            speech_text_map[sid_key] = (r["anforandetext"] or "")[:include_preview_chars]
            except Exception as e:
                LOG.debug("Failed reading speech parquet %s: %s", pf, e)

    rows = []
    for sid in speech_ids:
        txt = speech_text_map.get(str(sid), "")
        if not txt:
            LOG.debug("No text found for speech_id %s; skipping auto-label", sid)
            continue
        rows.append({"speech_id": sid, "text_preview": txt})

    if not rows:
        LOG.info("No speech texts available to auto-label; skipping")
        return

    tmp_in = tempfile.NamedTemporaryFile(prefix="rhet_input_", suffix=".parquet", delete=False)
    try:
        pd.DataFrame(rows).to_parquet(tmp_in.name, index=False, compression='zstd')
        tmp_in.flush()
        tmp_in.close()

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path("logs")
        out_dir.mkdir(exist_ok=True)
        out_parquet = out_dir / f"rhetoric_llm_labels_auto_{ts}.parquet"

        cmd = [sys.executable, "scripts/label_rhetoric_zero_shot.py", "--in", tmp_in.name, "--out", str(out_parquet), "--model", model, "--temp", str(temp_val), "--sleep", str(sleep)]
        LOG.info("Running zero-shot rhetoric labeler for %d speeches", len(rows))
        subprocess.run(cmd, check=True)

        # Ingest results (ingest script accepts parquet or csv)
        cmd2 = [sys.executable, "scripts/ingest_rhetoric_labels.py", "--csv", str(out_parquet), "--db", db_path]
        LOG.info("Ingesting rhetoric labels from %s", out_parquet)
        subprocess.run(cmd2, check=True)

        LOG.info("Auto-labeled and ingested %d speeches", len(rows))
    finally:
        try:
            tmp_in.close()
        except Exception:
            pass


def train_and_save_speech_meta_classifier(
    db_path: str = "data/swedish_parliament.db",
    out_path: str = "models/speech_meta_clf.pkl",
    tune: bool = False,
    n_iter: int = 12,
    zero_shot_func=None,
    bert_cls_func=None,
    embedding_matcher=None,
    auto_label_missing: bool = True,
    label_model: str = "llama3.1:8b",
    label_temp: float = 0.2,
    label_sleep: float = 0.05,
) -> Path:
    """Prepare training data from `speech_gold_labels`, train (optionally tune), and save model.

    Returns the path to the saved model file.
    """

    # Auto-label missing rhetoric rows by default (opt-out via CLI)
    if auto_label_missing:
        try:
            conn = schema.get_connection(db_path)
            cur = conn.cursor()
            speech_ids = set()
            try:
                cur.execute("SELECT speech_id FROM speech_gold_labels")
                speech_ids.update([str(r[0]) for r in cur.fetchall() if r[0] is not None])
            except Exception:
                pass
            try:
                cur.execute("SELECT speech_id FROM augmented_speech_gold_labels")
                speech_ids.update([str(r[0]) for r in cur.fetchall() if r[0] is not None])
            except Exception:
                pass
            try:
                cur.execute("SELECT speech_id FROM speech_self_training_labels")
                speech_ids.update([str(r[0]) for r in cur.fetchall() if r[0] is not None])
            except Exception:
                pass

            if speech_ids:
                placeholders = ",".join("?" for _ in speech_ids)
                try:
                    cur.execute(f"SELECT speech_id FROM speech_rhetoric_labels WHERE speech_id IN ({placeholders})", tuple(speech_ids))
                    existing = {str(r[0]) for r in cur.fetchall()}
                except Exception:
                    existing = set()
                missing = [sid for sid in speech_ids if sid not in existing]
                if missing:
                    LOG.info("Found %d speeches missing rhetoric labels; auto-labeling...", len(missing))
                    try:
                        _auto_label_missing_speeches(db_path, missing, model=label_model, temp_val=label_temp, sleep=label_sleep)
                    except Exception as e:
                        LOG.warning("Auto-labeling failed: %s", e)
        except Exception as e:
            LOG.debug("Auto-label pre-check failed: %s", e)

    X, y, category_names = prepare_speech_training_data(
        db_path=db_path,
        zero_shot_func=zero_shot_func,
        bert_cls_func=bert_cls_func,
        embedding_matcher=embedding_matcher,
    )

    if X.empty:
        raise RuntimeError("Prepared feature matrix X is empty; aborting training")

    # Delegate to ensemble training when not tuning; that function saves the model.
    if not tune:
        model_path = Path(out_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # ensemble.train_meta_classifier will save the model for us
        train_meta_classifier(X, y, category_names, model_path=model_path)
        LOG.info("Saved speech meta-classifier to %s", model_path)
        return model_path

    # Tuning path: randomized search over LightGBM params
    try:
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.class_weight import compute_class_weight
        from sklearn.model_selection import RandomizedSearchCV
        from lightgbm import LGBMClassifier
    except Exception as e:
        raise RuntimeError("scikit-learn and lightgbm are required for tuning: %s" % e)

    X_df = X.astype("float32")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_enc), y=y_enc)
    weight_dict = {i: w for i, w in enumerate(class_weights)}
    sample_weights = np.array([weight_dict[i] for i in y_enc])

    base = LGBMClassifier(random_state=42, verbose=-1)
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 8, 12, -1],
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    rs = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    LOG.info("Starting randomized hyperparameter search (n_iter=%d)...", n_iter)
    rs.fit(X_df, y_enc, sample_weight=sample_weights)
    best = rs.best_estimator_
    LOG.info("Best params: %s", rs.best_params_)

    # Save artifact: classifier + label encoder + feature names
    model_path = Path(out_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    loader.save_pickle(model_path, {"clf": best, "label_encoder": le, "category_names": category_names, "_feature_names": list(X_df.columns)})
    LOG.info("Saved tuned speech meta-classifier to %s", model_path)
    return model_path
