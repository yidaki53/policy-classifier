#!/usr/bin/env python3
"""Run calibration checks (temperature scaling + isotonic) on the latest
speech predictions CSV saved in `logs/` and produce updated reliability
and confusion matrix plots.

Usage:
    uv run python3 scripts/run_calibration_checks.py
"""

import argparse
import os
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, log_loss
from sklearn.calibration import calibration_curve


def softmax_from_logp(logp, T, eps=1e-12):
    # logp: (n_samples, n_classes) = log(probabilities)
    scaled = logp / float(T)
    # subtract max for stability
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exps = np.exp(scaled)
    denom = np.sum(exps, axis=1, keepdims=True)
    return exps / (denom + eps)


def find_best_temperature(logp, y_true_idx, grid=None, eps=1e-12):
    if grid is None:
        grid = np.logspace(-2, 1, 120)  # 0.01 .. 10
    best_T = None
    best_nll = float('inf')
    for T in grid:
        p_temp = softmax_from_logp(logp, T, eps=eps)
        # negative log-likelihood
        nll = -np.mean(np.log(p_temp[np.arange(len(y_true_idx)), y_true_idx] + eps))
        if nll < best_nll:
            best_nll = nll
            best_T = T
    return best_T, best_nll


def plot_confusion(cm, classes, out_path, title=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label',
        title=title or 'Confusion matrix',
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    fmt = 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reliability(probs, y_true_idx, classes, out_path, n_bins=10, title=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, cls in enumerate(classes):
        true_bin = (y_true_idx == i).astype(int)
        prob_pos = probs[:, i]
        try:
            frac_pos, mean_pred = calibration_curve(true_bin, prob_pos, n_bins=n_bins, strategy='uniform')
            ax.plot(mean_pred, frac_pos, marker='o', label=cls)
        except Exception:
            continue
    ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(title or 'Reliability diagram (per-class)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(predictions_path: str | None = None):
    logs_dir = Path('logs')
    if not logs_dir.exists():
        print('No logs/ directory found. Run scripts/evaluate_speech_gold_labels.py first.')
        return

    # Prefer parquet prediction files, fall back to CSV for compatibility
    if predictions_path:
        latest = Path(predictions_path)
        if not latest.exists():
            print(f'Predictions file not found: {latest}')
            return
    else:
        candidates = list(logs_dir.glob('speech_eval_preds_*.parquet')) + list(logs_dir.glob('speech_eval_preds_*.csv'))
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print('No speech_eval_preds_* found in logs/. Run evaluation first.')
            return
        latest = candidates[0]
    print(f'Using predictions file: {latest}')
    if latest.suffix.lower() == '.parquet':
        df = pd.read_parquet(latest)
    else:
        df = pd.read_csv(latest)

    # Detect prob columns
    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    if not prob_cols:
        print('No prob_ columns found in predictions CSV')
        return
    classes = [c[len('prob_'):] for c in prob_cols]
    probs = df[prob_cols].to_numpy(dtype=float)
    probs = np.clip(probs, 0.0, None)
    # Normalize guard
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = probs / row_sums

    # ground truth
    if 'truth' not in df.columns:
        print('truth column missing in predictions CSV')
        return
    y_true = df['truth'].astype(str).to_numpy()
    # map classes
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true_idx = np.array([class_to_idx.get(t, 0) for t in y_true], dtype=int)

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = Path('figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline metrics
    y_pred_idx = np.argmax(probs, axis=1)
    acc_base = accuracy_score(y_true_idx, y_pred_idx)
    ll_base = log_loss(y_true_idx, probs, labels=list(range(len(classes))))
    print('Baseline accuracy:', acc_base)
    print('Baseline NLL:', ll_base)
    print('Baseline classification report:\n')
    print(classification_report(y_true_idx, y_pred_idx, target_names=classes, zero_division=0))

    # Save baseline plots
    cm_base = confusion_matrix(y_true_idx, y_pred_idx, labels=range(len(classes)))
    conf_path_base = out_dir / f'calibration_confusion_baseline_{timestamp}.png'
    plot_confusion(cm_base, classes, conf_path_base, title=f'Baseline confusion (acc={acc_base:.3f})')
    rel_path_base = out_dir / f'calibration_reliability_baseline_{timestamp}.png'
    plot_reliability(probs, y_true_idx, classes, rel_path_base, title='Baseline reliability')

    # Temperature scaling
    eps = 1e-12
    logp = np.log(np.clip(probs, eps, 1.0))
    T_best, best_nll = find_best_temperature(logp, y_true_idx)
    print(f'Best temperature: {T_best:.4f}, NLL: {best_nll:.4f}')
    probs_temp = softmax_from_logp(logp, T_best, eps=eps)
    y_pred_temp = np.argmax(probs_temp, axis=1)
    acc_temp = accuracy_score(y_true_idx, y_pred_temp)
    ll_temp = log_loss(y_true_idx, probs_temp, labels=list(range(len(classes))))
    print('Temp-scaled accuracy:', acc_temp)
    print('Temp-scaled NLL:', ll_temp)
    print('Temp-scaled classification report:\n')
    print(classification_report(y_true_idx, y_pred_temp, target_names=classes, zero_division=0))

    # Save temp plots
    cm_temp = confusion_matrix(y_true_idx, y_pred_temp, labels=range(len(classes)))
    conf_path_temp = out_dir / f'calibration_confusion_temp_{timestamp}.png'
    plot_confusion(cm_temp, classes, conf_path_temp, title=f'Temperature-scaled confusion (acc={acc_temp:.3f})')
    rel_path_temp = out_dir / f'calibration_reliability_temp_{timestamp}.png'
    plot_reliability(probs_temp, y_true_idx, classes, rel_path_temp, title='Temp-scaled reliability')

    # Save temp-calibrated predictions
    pred_out_dir = Path('logs')
    pred_temp_csv = pred_out_dir / f'speech_eval_preds_tempcal_{timestamp}.parquet'
    df_temp = df.copy()
    for i, c in enumerate(classes):
        df_temp[f'prob_{c}'] = probs_temp[:, i]
    df_temp['pred_temp'] = [classes[i] for i in np.argmax(probs_temp, axis=1)]
    df_temp.to_parquet(pred_temp_csv, index=False, compression='zstd')

    # Isotonic calibration (per-class) - one-vs-rest
    probs_iso = np.zeros_like(probs)
    for i, cls in enumerate(classes):
        y_bin = (y_true_idx == i).astype(int)
        x = probs[:, i]
        unique_vals = np.unique(x)
        if len(unique_vals) < 2:
            # Can't fit isotonic on constant column
            probs_iso[:, i] = x
            continue
        try:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(x, y_bin)
            probs_iso[:, i] = iso.transform(x)
        except Exception:
            probs_iso[:, i] = x

    # Renormalize rows
    row_sums = probs_iso.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs_iso = probs_iso / row_sums

    y_pred_iso = np.argmax(probs_iso, axis=1)
    acc_iso = accuracy_score(y_true_idx, y_pred_iso)
    ll_iso = log_loss(y_true_idx, probs_iso, labels=list(range(len(classes))))
    print('Isotonic accuracy:', acc_iso)
    print('Isotonic NLL:', ll_iso)
    print('Isotonic classification report:\n')
    print(classification_report(y_true_idx, y_pred_iso, target_names=classes, zero_division=0))

    # Save isotonic plots
    cm_iso = confusion_matrix(y_true_idx, y_pred_iso, labels=range(len(classes)))
    conf_path_iso = out_dir / f'calibration_confusion_iso_{timestamp}.png'
    plot_confusion(cm_iso, classes, conf_path_iso, title=f'Isotonic confusion (acc={acc_iso:.3f})')
    rel_path_iso = out_dir / f'calibration_reliability_iso_{timestamp}.png'
    plot_reliability(probs_iso, y_true_idx, classes, rel_path_iso, title='Isotonic reliability')

    # Save isotonic-calibrated predictions
    pred_iso_csv = pred_out_dir / f'speech_eval_preds_isotonic_{timestamp}.parquet'
    df_iso = df.copy()
    for i, c in enumerate(classes):
        df_iso[f'prob_{c}'] = probs_iso[:, i]
    df_iso['pred_iso'] = [classes[i] for i in np.argmax(probs_iso, axis=1)]
    df_iso.to_parquet(pred_iso_csv, index=False, compression='zstd')

    print('\nOutputs saved:')
    print('Baseline confusion:', conf_path_base)
    print('Temp confusion:', conf_path_temp)
    print('Iso confusion:', conf_path_iso)
    print('Baseline reliability:', rel_path_base)
    print('Temp reliability:', rel_path_temp)
    print('Iso reliability:', rel_path_iso)
    print('Temp calibrated preds:', pred_temp_csv)
    print('Iso calibrated preds:', pred_iso_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', default=None, help='Optional explicit predictions parquet/csv path')
    args = parser.parse_args()
    main(predictions_path=args.predictions)
