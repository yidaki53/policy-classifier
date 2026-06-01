#!/usr/bin/env python3
"""Run Ollama classifier on speeches listed in `stratified_classification_report.md`.

Extracts IDs and top-category entries from the markdown report, finds the
corresponding speech texts in `data/speeches/parquet/*.parquet`, runs the local
Ollama classifier (via `swedish_parliament_policy_classifier.nlp.ollama_classifier`),
and writes a review CSV under `logs/` plus a confusion plot under `figures/`.

Usage:
    uv run python3 scripts/ollama_review_from_stratified_report.py --report stratified_classification_report.md --model llama3.1:8b
"""

import argparse
import re
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from swedish_parliament_policy_classifier.nlp.ollama_classifier import (
    classify_speech_with_cache,
    _CATEGORIES_ORDER,
)
from swedish_parliament_policy_classifier.classifier.scorer import _extract_speech_argumentative_text
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def parse_report_ids(report_path: Path):
    text = report_path.read_text(encoding='utf-8')
    segments = re.split(r'\n##\s+', text)
    rows = []
    for seg in segments:
        if not seg.strip():
            continue
        # header is first line up to newline
        header_line = seg.splitlines()[0].strip()
        # try to find ID
        m = re.search(r'\*\*ID:\*\*\s*([0-9a-f\-]+)', seg)
        if not m:
            continue
        sid = m.group(1).strip()
        # find top category
        m2 = re.search(r'\*\*Top category:\*\*\s*([a-zA-Z_\-]+)\s*(?:\(([0-9.]+)\))?', seg)
        top_cat = m2.group(1) if m2 else None
        top_score = float(m2.group(2)) if m2 and m2.group(2) else None
        rows.append({'speech_id': sid, 'header': header_line, 'stratified_top': top_cat, 'stratified_score': top_score, 'segment': seg})
    return rows


def find_speeches_parquet():
    pdir = Path('data/speeches/parquet')
    if not pdir.exists():
        raise FileNotFoundError(f'speech parquet dir not found: {pdir}')
    parts = []
    for p in sorted(pdir.glob('*.parquet')):
        try:
            df = pd.read_parquet(p)
            parts.append(df)
        except Exception:
            continue
    if not parts:
        raise FileNotFoundError('no parquet speech files found')
    return pd.concat(parts, ignore_index=True)


def main(report: str, model: str = 'llama3.1:8b', timeout: int = 80):
    report_path = Path(report)
    if not report_path.exists():
        print('Report file not found:', report_path)
        return

    rows = parse_report_ids(report_path)
    print(f'Found {len(rows)} entries in report')
    if not rows:
        return

    all_speeches = find_speeches_parquet()
    speech_map = {r['anforande_id']: r for _, r in all_speeches.iterrows()}

    cache = {}
    results = []
    missing = 0
    for r in rows:
        sid = r['speech_id']
        seg = r['segment']
        header = r['header']
        strat_top = r['stratified_top']
        strat_score = r['stratified_score']

        row = speech_map.get(sid)
        if row is None:
            missing += 1
            results.append({'speech_id': sid, 'header': header, 'talare': None, 'party': None, 'stratified_top': strat_top, 'stratified_score': strat_score, 'ollama_top': None, 'ollama_top_score': None, 'ollama_scores': None, 'text_preview': ''})
            continue

        text = row.get('anforandetext') or ''
        text_proc = _extract_speech_argumentative_text(text, max_chars=3500)

        try:
            scores = classify_speech_with_cache(text_proc, speech_id=sid, cache=cache, model=model, max_chars=3500, timeout=timeout)
        except Exception as e:
            print('Ollama call failed for', sid, e)
            scores = None

        if scores is None:
            results.append({'speech_id': sid, 'header': header, 'talare': row.get('talare'), 'party': row.get('parti'), 'stratified_top': strat_top, 'stratified_score': strat_score, 'ollama_top': None, 'ollama_top_score': None, 'ollama_scores': None, 'text_preview': text_proc[:400]})
            continue

        vec = [scores.get(c, 0.0) for c in _CATEGORIES_ORDER]
        total = sum(vec)
        if total > 0:
            vec = [v/total for v in vec]
        else:
            vec = [0.0]*len(_CATEGORIES_ORDER)
        top_idx = int(np.argmax(vec)) if any(v>0 for v in vec) else None
        top_cat = _CATEGORIES_ORDER[top_idx] if top_idx is not None else None
        top_score = vec[top_idx] if top_idx is not None else None

        results.append({
            'speech_id': sid,
            'header': header,
            'talare': row.get('talare'),
            'party': row.get('parti'),
            'stratified_top': strat_top,
            'stratified_score': strat_score,
            'ollama_top': top_cat,
            'ollama_top_score': top_score,
            'ollama_scores': json.dumps({c: float(v) for c, v in zip(_CATEGORIES_ORDER, vec)}),
            'text_preview': text_proc[:400],
        })

    df = pd.DataFrame(results)
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = Path('logs')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'ollama_stratified_review_{ts}.parquet'

    # Expand probs
    for c in _CATEGORIES_ORDER:
        df[f'ollama_prob_{c}'] = df['ollama_scores'].apply(lambda s: (json.loads(s).get(c) if isinstance(s, str) and s else (s.get(c) if isinstance(s, dict) else None)))

    df.to_parquet(out_csv, index=False, compression='zstd')
    print('Wrote:', out_csv)
    if missing:
        print('Missing from parquet:', missing)

    # Compare stratified_top vs ollama_top for non-null
    mask = df['stratified_top'].notnull() & df['ollama_top'].notnull()
    if mask.sum() == 0:
        print('No overlapping stratified/ollama predictions to compare')
        return
    y_true = df.loc[mask, 'stratified_top'].tolist()
    y_pred = df.loc[mask, 'ollama_top'].tolist()

    acc = accuracy_score(y_true, y_pred)
    print(f'Ollama vs stratified accuracy: {acc:.4f} on {mask.sum()} samples')
    print('Classification report:')
    print(classification_report(y_true, y_pred, zero_division=0))

    labels = _CATEGORIES_ORDER
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='Stratified top',
        xlabel='Ollama top',
        title=f'Ollama vs stratified confusion (acc={acc:.3f})',
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig_dir = Path('figures')
    fig_dir.mkdir(exist_ok=True)
    fig_path = fig_dir / f'ollama_stratified_confusion_{ts}.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print('Saved confusion plot to', fig_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', default='stratified_classification_report.md')
    parser.add_argument('--model', default='llama3.1:8b')
    parser.add_argument('--timeout', type=int, default=80)
    args = parser.parse_args()
    main(report=args.report, model=args.model, timeout=args.timeout)
