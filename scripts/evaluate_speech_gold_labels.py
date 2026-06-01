#!/usr/bin/env python3
"""Evaluate speech classifier against parquet speech gold labels.

Saves confusion matrix and reliability plot to `figures/` and prints a short report.

Usage:
    uv run python3 scripts/evaluate_speech_gold_labels.py
"""

import argparse
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from swedish_parliament_policy_classifier.classifier.scorer import (
    load_definitions,
)
from swedish_parliament_policy_classifier.classifier.deep_scoring_service import DeepScoringService
from swedish_parliament_policy_classifier.nlp.topic_modeler import load_topic_distributions
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher
from swedish_parliament_policy_classifier.classifier.ensemble import load_meta_classifier
from swedish_parliament_policy_classifier.runtime.resources import apply_cpu_throttle, thermal_safe_defaults
from swedish_parliament_policy_classifier.runtime.experiment import ExperimentRun

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.calibration import calibration_curve


def main(
    gold_parquet: str = "data/parquet/speech_gold_labels.parquet",
    speech_parquet_dir: str = "data/speeches/parquet",
    max_samples: int | None = None,
    cpu_fraction: float = 0.5,
    min_speech_rows: int = 100000,
    mlflow_enabled: bool = False,
    mlflow_experiment: str = "speech-eval",
    mlflow_tracking_uri: str | None = None,
):
    throttle = apply_cpu_throttle(cpu_fraction=cpu_fraction)
    run = ExperimentRun.start(
        enabled=mlflow_enabled,
        experiment_name=mlflow_experiment,
        run_name=f"evaluate-speech-gold-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        tracking_uri=mlflow_tracking_uri,
    )
    defs = load_definitions()
    category_names = list(defs.keys())

    topic_dists = load_topic_distributions()

    try:
        matcher = EmbeddingMatcher()
        if matcher.model is None:
            matcher = None
    except Exception as e:
        print(f"Embedding matcher unavailable: {e}")
        matcher = None

    try:
        meta_clf = load_meta_classifier()
    except Exception:
        meta_clf = None

    scoring = DeepScoringService(
        categories=defs,
        embedding_matcher=matcher,
        topic_distributions=topic_dists,
        meta_clf=meta_clf,
    )

    # Load speech gold labels from parquet
    gold_path = Path(gold_parquet)
    if not gold_path.exists():
        print(f"Gold-label parquet not found: {gold_path}")
        return
    gold_df = pd.read_parquet(gold_path)
    if not {"speech_id", "category"}.issubset(set(gold_df.columns)):
        print("Gold-label parquet must contain columns: speech_id, category")
        return
    if gold_df.empty:
        print("No speech_gold_labels rows found in parquet. Nothing to evaluate.")
        return

    if max_samples is not None and max_samples > 0:
        gold_df = gold_df.head(max_samples).copy()

    rows = list(gold_df[["speech_id", "category"]].itertuples(index=False, name=None))

    speech_ids = [r[0] for r in rows]
    y_true = [r[1] for r in rows]

    # Load speeches from parquet files
    parquet_dir = Path(speech_parquet_dir)
    if not parquet_dir.exists():
        print(f"Speech parquet directory not found: {parquet_dir}")
        return

    speeches = []
    for p in parquet_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(p)
            speeches.append(df)
        except Exception:
            continue
    if not speeches:
        print("No parquet speech files found.")
        return

    all_speeches = pd.concat(speeches, ignore_index=True)
    if len(all_speeches) < min_speech_rows:
        raise ValueError(
            f"Speech parquet corpus is too small ({len(all_speeches)} rows). "
            f"Expected at least {min_speech_rows} rows; ensure full parquet shards are used."
        )
    speech_map = {r['anforande_id']: r for _, r in all_speeches.iterrows()}

    probs = []
    y_pred = []
    missing = 0

    for sid in speech_ids:
        row = speech_map.get(sid)
        if row is None:
            missing += 1
            # append uniform probability
            probs.append(np.ones(len(category_names)) / len(category_names))
            y_pred.append(None)
            continue

        text = row.get('anforandetext') or ''
        # Score the speech
        result = scoring.classify(item_id=sid, text=text)
        # Build prob vector in category_names order
        res_map = {r.category: r.normalized_weight for r in result}
        vec = np.array([res_map.get(c, 0.0) for c in category_names], dtype=float)
        # Renormalize guard
        s = vec.sum()
        if s <= 0:
            vec = np.ones(len(category_names)) / len(category_names)
        else:
            vec = vec / s
        probs.append(vec)
        pred_idx = int(np.argmax(vec))
        y_pred.append(category_names[pred_idx])

    if missing:
        print(f"Warning: {missing} gold-labeled speeches not found in parquet files; padded with uniform probabilities.")

    probs = np.vstack(probs)

    # Filter out rows where y_pred is None? Keep them but they are uniform.

    le = LabelEncoder()
    le.fit(category_names)
    y_true_enc = le.transform(y_true)
    y_pred_enc = le.transform([p if p in le.classes_ else le.classes_[0] for p in y_pred])

    # Metrics
    acc = accuracy_score(y_true_enc, y_pred_enc)
    report = classification_report(y_true_enc, y_pred_enc, target_names=le.classes_, zero_division=0)
    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=range(len(le.classes_)))

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = Path('figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ylabel='True label',
        xlabel='Predicted label',
        title=f'Confusion matrix (accuracy={acc:.3f})',
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Annotate cells
    fmt = 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')

    conf_path = out_dir / f'eval_speech_confusion_{timestamp}.png'
    fig.tight_layout()
    fig.savefig(conf_path, dpi=150)
    plt.close(fig)

    # Calibration / reliability plot (overlay per-class curves)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(le.classes_):
        true_bin = (y_true_enc == i).astype(int)
        prob_pos = probs[:, i]
        # calibration_curve requires at least one positive and negative sample; wrap in try
        try:
            frac_pos, mean_pred = calibration_curve(true_bin, prob_pos, n_bins=10, strategy='uniform')
            ax.plot(mean_pred, frac_pos, marker='o', label=cls)
        except Exception:
            continue

    ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Reliability diagram (per-class)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    calib_path = out_dir / f'eval_speech_reliability_{timestamp}.png'
    fig.tight_layout()
    fig.savefig(calib_path, dpi=150)
    plt.close(fig)

    # Save predictions
    pred_df = pd.DataFrame({
        'speech_id': speech_ids,
        'truth': y_true,
        'pred': y_pred,
    })
    for i, c in enumerate(category_names):
        pred_df[f'prob_{c}'] = probs[:, i]
    pred_out = Path('logs')
    pred_out.mkdir(parents=True, exist_ok=True)
    pred_parquet = pred_out / f'speech_eval_preds_{timestamp}.parquet'
    pred_df.to_parquet(pred_parquet, index=False, compression='zstd')

    run.log_params(
        {
            "gold_parquet": gold_parquet,
            "speech_parquet_dir": speech_parquet_dir,
            "max_samples": max_samples,
            "cpu_fraction": cpu_fraction,
            "max_threads": throttle.get("max_threads"),
            "min_speech_rows": min_speech_rows,
        }
    )
    run.log_metrics(
        {
            "accuracy": acc,
            "n_gold": len(speech_ids),
            "n_speech_rows": len(all_speeches),
            "n_missing": missing,
        }
    )
    run.log_artifact(str(conf_path))
    run.log_artifact(str(calib_path))
    run.log_artifact(str(pred_parquet))
    run.end(status="FINISHED")

    # Print short report
    print(f"Evaluated {len(speech_ids)} speech gold labels")
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:\n")
    print(report)
    print(f"Confusion matrix plot saved to: {conf_path}")
    print(f"Reliability plot saved to: {calib_path}")
    print(f"Predictions saved to: {pred_parquet}")


if __name__ == '__main__':
    safe = thermal_safe_defaults("safe")
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold-parquet', default='data/parquet/speech_gold_labels.parquet')
    parser.add_argument('--speech-parquet-dir', default='data/speeches/parquet')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--cpu-fraction', type=float, default=float(os.environ.get('CLASSIFIER_CPU_FRACTION', str(safe['cpu_fraction']))))
    parser.add_argument('--min-speech-rows', type=int, default=100000)
    parser.add_argument('--mlflow', action='store_true')
    parser.add_argument('--mlflow-experiment', default='speech-eval')
    parser.add_argument('--mlflow-tracking-uri', default=os.environ.get('MLFLOW_TRACKING_URI'))
    args = parser.parse_args()
    main(
        gold_parquet=args.gold_parquet,
        speech_parquet_dir=args.speech_parquet_dir,
        max_samples=args.max_samples,
        cpu_fraction=args.cpu_fraction,
        min_speech_rows=args.min_speech_rows,
        mlflow_enabled=args.mlflow,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )
