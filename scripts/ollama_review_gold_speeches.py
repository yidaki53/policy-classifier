#!/usr/bin/env python3
"""Run local Ollama model to classify gold-labeled speeches and produce a review CSV.

Usage:
    uv run python3 scripts/ollama_review_gold_speeches.py --db data/swedish_parliament.db --model llama3.1:8b
"""

import argparse
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import _extract_speech_argumentative_text
from swedish_parliament_policy_classifier.nlp.ollama_classifier import (
    classify_speech_with_cache,
    _CATEGORIES_ORDER,
)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def main(db_path: str = 'data/swedish_parliament.db', model: str = 'llama3.1:8b'):
    conn = init_db(db_path)
    cur = conn.cursor()
    cur.execute('SELECT speech_id, category FROM speech_gold_labels')
    rows = cur.fetchall()
    if not rows:
        print('No speech_gold_labels found in DB. Aborting.')
        return

    speech_ids = [r[0] for r in rows]
    truths = [r[1] for r in rows]

    # Load parquets
    parquet_dir = Path('data/speeches/parquet')
    if not parquet_dir.exists():
        print('Speech parquet dir not found:', parquet_dir)
        return
    parts = []
    for p in sorted(parquet_dir.glob('*.parquet')):
        try:
            parts.append(pd.read_parquet(p))
        except Exception as e:
            print('Failed to read', p, e)
    if not parts:
        print('No speech parquet files loaded.')
        return
    all_speeches = pd.concat(parts, ignore_index=True)
    speech_map = {r['anforande_id']: r for _, r in all_speeches.iterrows()}

    cache = {}
    results = []
    missing = 0
    for sid, truth in zip(speech_ids, truths):
        row = speech_map.get(sid)
        if row is None:
            missing += 1
            results.append({'speech_id': sid, 'talare': None, 'party': None, 'truth': truth, 'ollama_top': None, 'ollama_scores': None, 'text_preview': ''})
            continue
        text = row.get('anforandetext') or ''
        # Preprocess argumentative text
        text_proc = _extract_speech_argumentative_text(text, max_chars=3500)
        # Call Ollama
        try:
            scores = classify_speech_with_cache(text_proc, speech_id=sid, cache=cache, model=model, max_chars=3500, timeout=120)
        except Exception as e:
            print('Ollama call failed for', sid, e)
            scores = None
        if scores is None:
            results.append({'speech_id': sid, 'talare': row.get('talare'), 'party': row.get('parti'), 'truth': truth, 'ollama_top': None, 'ollama_scores': None, 'text_preview': text_proc[:400]})
            continue
        # Ensure vector in _CATEGORIES_ORDER
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
            'talare': row.get('talare'),
            'party': row.get('parti'),
            'truth': truth,
            'ollama_top': top_cat,
            'ollama_top_score': top_score,
            'ollama_scores': json.dumps({c: float(v) for c, v in zip(_CATEGORIES_ORDER, vec)}),
            'text_preview': text_proc[:400],
        })

    df = pd.DataFrame(results)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = Path('logs')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'ollama_gold_review_{timestamp}.parquet'

    # Expand probs into columns
    for c in _CATEGORIES_ORDER:
        df[f'ollama_prob_{c}'] = df['ollama_scores'].apply(lambda s: (json.loads(s).get(c) if isinstance(s, str) and s else (s.get(c) if isinstance(s, dict) else None)))

    df.to_parquet(out_csv, index=False, compression='zstd')

    print('Wrote review parquet to', out_csv)
    if missing:
        print(f'Warning: {missing} gold-labeled speeches missing from parquet files')

    # Metrics where Ollama provided a top choice
    mask = df['ollama_top'].notnull()
    if mask.sum() == 0:
        print('No Ollama predictions produced.')
        return
    y_true = df.loc[mask, 'truth'].to_list()
    y_pred = df.loc[mask, 'ollama_top'].to_list()

    acc = accuracy_score(y_true, y_pred)
    print(f'Ollama vs gold accuracy: {acc:.4f} on {mask.sum()} speeches')
    print('Classification report:')
    print(classification_report(y_true, y_pred, zero_division=0))

    # confusion matrix
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
        ylabel='True label',
        xlabel='Ollama top prediction',
        title=f'Ollama vs gold confusion (acc={acc:.3f})',
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig_path = Path('figures') / f'ollama_gold_confusion_{timestamp}.png'
    Path('figures').mkdir(exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print('Saved confusion plot to', fig_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='data/swedish_parliament.db')
    parser.add_argument('--model', default='llama3.1:8b')
    args = parser.parse_args()
    main(db_path=args.db, model=args.model)
