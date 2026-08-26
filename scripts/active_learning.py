#!/usr/bin/env python3
"""
Active learning helper for speeches.

Commands:
  export  - export top-N uncertain speech examples for annotation
  ingest  - ingest an annotated CSV (speech_id,corrected_category,annotator,notes)

Ingested rows are upserted into `speech_gold_labels` and a lineage entry is written.
Optionally retrain the speech meta-classifier (with or without tuning) and/or run
self-training to harvest pseudo-labels.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import logging

import pandas as pd

from swedish_parliament_policy_classifier.orchestration.speech_pipeline import (
    export_active_learning_candidates,
    train_and_save_speech_meta_classifier,
)

LOG = logging.getLogger(__name__)


def upsert_manual_label(conn: sqlite3.Connection, speech_id: str, category: str, annotator: str, notes: str):
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    # Delete existing label for id (unique index on speech_id enforced)
    cur.execute("DELETE FROM speech_gold_labels WHERE speech_id = ?", (speech_id,))
    cur.execute(
        "INSERT INTO speech_gold_labels (speech_id, category, reasoning, prompt_version, model, temperature, raw_response, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (speech_id, category, notes or None, 'manual-correction', None, None, None, now),
    )
    cur.execute(
        "INSERT INTO lineage (source_table, source_id, operation, timestamp, notes) VALUES (?, ?, ?, ?, ?)",
        ("speech_gold_labels", speech_id, "manual_upsert", now, f"annotator={annotator}"),
    )
    conn.commit()


def ingest_csv(db_path: str, csv_path: str, retrain: bool = False, tune: bool = False, n_iter: int = 12, self_train: bool = False, st_conf: float = 0.85, st_max: int = 200):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    df = pd.read_csv(csv_path)
    applied = 0
    for _, r in df.iterrows():
        sid = str(r.get('speech_id') or r.get('id') or '')
        cat = r.get('corrected_category') or r.get('corrected_label') or r.get('category')
        annotator = r.get('annotator') or 'manual'
        notes = r.get('notes') or ''
        if not sid or not cat:
            LOG.warning('Skipping invalid row: %s', dict(r))
            continue
        upsert_manual_label(conn, sid, cat, annotator, notes)
        applied += 1

    print(f"Ingested {applied} manual corrections into {db_path}")

    if retrain:
        print("Retraining speech meta-classifier...")
        train_and_save_speech_meta_classifier(db_path=db_path, out_path='models/speech_meta_clf_manual_retrain.pkl', tune=tune, n_iter=n_iter)

    if self_train:
        print(f"Running self-training (confidence={st_conf}, max_per_class={st_max})...")
        # Run self-train script as subprocess to avoid import-time side effects
        cmd = [sys.executable, 'scripts/self_train_speech.py', '--db', db_path, '--confidence', str(st_conf), '--max-per-class', str(st_max), '--retrain']
        subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description='Active learning helpers for speeches')
    sub = p.add_subparsers(dest='cmd')

    ex = sub.add_parser('export', help='Export top-N uncertain examples for annotation')
    ex.add_argument('--db', default='data/swedish_parliament.db')
    ex.add_argument('--preds-csv', default=None, help='Predictions CSV to use (defaults to latest in logs/)')
    ex.add_argument('--top-n', type=int, default=500)
    ex.add_argument('--out', default=None, help='Output CSV path')

    ig = sub.add_parser('ingest', help='Ingest annotated CSV (speech_id,corrected_category,annotator,notes)')
    ig.add_argument('--db', default='data/swedish_parliament.db')
    ig.add_argument('--csv', required=True, help='Annotated CSV to ingest')
    ig.add_argument('--retrain', action='store_true', help='Retrain speech meta-classifier after ingest')
    ig.add_argument('--tune', action='store_true', help='Run hyperparameter tuning during retrain')
    ig.add_argument('--n-iter', type=int, default=12, help='RandomizedSearch n_iter when tuning')
    ig.add_argument('--self-train', action='store_true', help='Run self-training after retrain')
    ig.add_argument('--st-confidence', type=float, default=0.85, help='Self-training confidence threshold')
    ig.add_argument('--st-max-per-class', type=int, default=200, help='Self-training max per class to keep')

    args = p.parse_args()

    if args.cmd == 'export':
        out = export_active_learning_candidates(db_path=args.db, preds_csv=args.preds_csv, top_n=args.top_n, out_path=args.out)
        print('Wrote candidates to:', out)
        return

    if args.cmd == 'ingest':
        ingest_csv(db_path=args.db, csv_path=args.csv, retrain=args.retrain, tune=args.tune, n_iter=args.n_iter, self_train=args.self_train, st_conf=args.st_confidence, st_max=args.st_max_per_class)
        return

    p.print_help()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
#!/usr/bin/env python3
"""Active learning pipeline for parliamentary motion classification.

Scores unlabeled motions using current ensemble entropy, selects top-N
most uncertain motions, and labels them via local LLM (Ollama) and
Hugging Face zero-shot classifier. Stores high-confidence agreed labels
into augmented_gold_labels for iterative improvement.

Usage:
    uv run python scripts/active_learning.py --db data/swedish_parliament.db --select 500 --label-model qwen2.5-coder-14b-32k:latest
"""

import argparse
import json
import os
import sqlite3
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
if os.getenv("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.getenv("HF_TOKEN"))

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions, score_motion
from swedish_parliament_policy_classifier.nlp.topic_modeler import load_topic_distributions
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher
from swedish_parliament_policy_classifier.classifier.ensemble import (
    build_feature_vector,
    load_meta_classifier,
    _build_feature_names,
)

LOG = print


def _keyword_scores_for_text(text: str, kw_index: Dict) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if not text:
        return scores
    text_l = text.lower()
    for lemma_key, cat_kw_pairs in kw_index.items():
        if lemma_key in text_l:
            for cat_name, _ in cat_kw_pairs:
                scores[cat_name] = scores.get(cat_name, 0.0) + 1.0
    return scores


def _build_kw_index(categories):
    import spacy
    index: Dict[str, List[Tuple[str, str]]] = {}
    try:
        nlp = spacy.load("sv_core_news_sm", disable=["parser", "ner"])
    except Exception:
        return {}
    for name, cat in categories.items():
        for kw in cat.keywords or []:
            if not kw:
                continue
            doc = nlp(kw.lower())
            lemmas = [t.lemma_.lower() for t in doc if not t.is_space and not t.is_punct]
            key = " ".join(lemmas)
            index.setdefault(key, []).append((name, kw))
    return index


def _embedding_scores_for_text(text: str, categories, matcher: EmbeddingMatcher) -> Dict[str, float]:
    if matcher is None or not text:
        return {}
    if not hasattr(matcher, "_cached_cat_embs"):
        matcher._cached_cat_embs = matcher.build_category_embeddings(categories)
    cat_embs = matcher._cached_cat_embs
    q = matcher.encode([text])[0]
    scores = {}
    for name, emb in cat_embs.items():
        scores[name] = matcher._cosine(q, emb)
    return scores


def _predict_proba_for_text(text: str, motion_id: str, categories, kw_index, matcher, topic_dists, clf, category_names) -> np.ndarray:
    """Return probability vector for a single text."""
    kw_scores = _keyword_scores_for_text(text, kw_index)
    emb_scores = _embedding_scores_for_text(text, categories, matcher)
    topic_vec = topic_dists.get(motion_id)

    import pandas as pd
    vec = build_feature_vector(
        keyword_scores=kw_scores,
        embedding_scores=emb_scores,
        topic_features=topic_vec,
        text_length=len(text),
        category_names=category_names,
        date_days_ago=None,
        doc_type=None,
    )
    probs = clf.predict_proba(vec)[0]
    return probs


def compute_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability distribution."""
    probs = np.array(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def label_with_ollama(text: str, categories: List[str], model: str = "qwen2.5-coder-14b-32k:latest", temperature: float = 0.1) -> Optional[Dict]:
    """Label a single motion with Ollama."""
    import ollama
    cat_list = ", ".join(f'"{c}"' for c in categories)
    prompt = f"""Du är en expert på svensk politik och ideologisk analys.

Uppgift: Läs följande motion från Sveriges riksdag och placera den på en politisk skala.

Kategorier att välja mellan: {cat_list}

Instruktioner:
- Välj ENDAST en kategori.
- Motivet ska vara baserat på motionens POLICY-INNEHÅLL, inte på vilket parti som författat den.
- Förklara KORT varför du valde den kategorin.

Motionstext:
---
{text[:2500]}
---

Svara ENDAST med ett JSON-objekt i exakt detta format:
{{"category": "<kategori>", "reasoning": "<motivering>", "confidence": <0-10>}}
"""
    start = time.time()
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        content = response["message"]["content"].strip()
    except Exception as e:
        return {"model": model, "error": str(e), "latency": time.time() - start}

    latency = time.time() - start

    # Extract JSON
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        result = json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            try:
                result = json.loads(content[start:end+1])
            except Exception:
                return {"model": model, "raw": content[:200], "latency": latency}
        else:
            return {"model": model, "raw": content[:200], "latency": latency}

    result["model"] = model
    result["latency"] = latency
    result["category"] = result.get("category", "").strip().lower()
    return result


def label_with_hf_zero_shot(text: str, categories: List[str]) -> Optional[Dict]:
    """Label a single motion with the existing zero-shot HF pipeline (mDeBERTa NLI)."""
    start = time.time()
    try:
        from swedish_parliament_policy_classifier.nlp.zero_shot_values import zero_shot_score
        scores = zero_shot_score(text[:2500])
        best_cat = max(scores, key=scores.get)
        latency = time.time() - start
        return {
            "model": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            "category": best_cat,
            "confidence": round(scores[best_cat] * 10, 2),
            "latency": latency,
            "scores": scores,
        }
    except Exception as e:
        return {"model": "mdeberta-zero-shot", "error": str(e), "latency": time.time() - start}


def active_learning(
    db_path: str = "data/swedish_parliament.db",
    n_select: int = 500,
    n_pool: int = 5000,
    ollama_model: str = "qwen2.5-coder-14b-32k:latest",
    compare_hf: bool = True,
    confidence_threshold: float = 7.0,  # Ollama confidence (0-10)
    agree_insert: bool = True,
    retrain_after: bool = False,
):
    """Main active learning loop."""
    conn = init_db(db_path)
    cur = conn.cursor()
    defs = load_definitions()
    topic_dists = load_topic_distributions()
    model = load_meta_classifier()
    if model is None:
        print("No trained ensemble model found. Run train_models.py first.", file=sys.stderr)
        sys.exit(1)

    clf = model["clf"]
    category_names = sorted(defs.keys())
    kw_index = _build_kw_index(defs)

    matcher = None
    try:
        matcher = EmbeddingMatcher()
        if matcher.model is None:
            matcher = None
    except Exception as e:
        print(f"Embedding matcher unavailable: {e}", file=sys.stderr)

    # Load already labeled motion IDs
    cur.execute("SELECT DISTINCT motion_id FROM augmented_gold_labels")
    labeled_ids = {r[0] for r in cur.fetchall()}
    print(f"Already labeled motions: {len(labeled_ids)}", file=sys.stderr)

    # Sample unlabeled motions
    cur.execute(f"""
        SELECT id, text FROM normalized_motions
        WHERE text IS NOT NULL AND LENGTH(text) > 100
          AND id NOT IN (SELECT motion_id FROM augmented_gold_labels)
        ORDER BY RANDOM()
        LIMIT {n_pool}
    """)
    rows = cur.fetchall()
    print(f"Sampled {len(rows)} unlabeled motions for entropy scoring.", file=sys.stderr)

    # Batch scoring for efficiency
    print("Computing ensemble entropy (batched)...", file=sys.stderr)
    batch_size = 128
    scored = []
    total = len(rows)
    for i in range(0, total, batch_size):
        batch = rows[i:i+batch_size]
        try:
            import pandas as pd
            batch_texts = [r[1] or "" for r in batch]
            batch_ids = [r[0] for r in batch]
            batch_kw = [_keyword_scores_for_text(t, kw_index) for t in batch_texts]
            batch_emb = [_embedding_scores_for_text(t, defs, matcher) for t in batch_texts]
            batch_topic = [topic_dists.get(mid) for mid in batch_ids]

            vecs = []
            for j in range(len(batch)):
                vec = build_feature_vector(
                    keyword_scores=batch_kw[j],
                    embedding_scores=batch_emb[j],
                    topic_features=batch_topic[j],
                    text_length=len(batch_texts[j]),
                    category_names=category_names,
                    date_days_ago=None,
                    doc_type=None,
                )
                vecs.append(vec)

            X_batch = pd.concat(vecs, ignore_index=True)
            probs_batch = clf.predict_proba(X_batch)

            for j, (motion_id, text) in enumerate(batch):
                ent = compute_entropy(probs_batch[j])
                scored.append((motion_id, text, ent, probs_batch[j]))
        except Exception as e:
            if (i // batch_size) % 5 == 0:
                print(f"  Batch {i//batch_size} error: {e}", file=sys.stderr)
            continue

        if (i + batch_size) % 500 == 0 or i == 0:
            print(f"  Scored {min(i+batch_size, total)}/{total} motions...", file=sys.stderr)

    if not scored:
        print("No motions successfully scored. Exiting.", file=sys.stderr)
        return 0

    # Sort by descending entropy (most uncertain first)
    scored.sort(key=lambda x: -x[2])
    top_n = scored[:n_select]
    print(f"Selected top-{len(top_n)} highest-entropy motions (entropy range {top_n[-1][2]:.3f} - {top_n[0][2]:.3f})", file=sys.stderr)

    # Label with Ollama
    print(f"Labeling with Ollama ({ollama_model})...", file=sys.stderr)
    ollama_results = {}
    ollama_times = []
    for i, (motion_id, text, ent, probs) in enumerate(top_n):
        result = label_with_ollama(text, category_names, model=ollama_model)
        ollama_results[motion_id] = result
        if "latency" in result:
            ollama_times.append(result["latency"])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(top_n)} labeled, avg latency={np.mean(ollama_times):.1f}s" if ollama_times else f"  {i+1}/{len(top_n)} labeled", file=sys.stderr)

    # Optionally compare with HF zero-shot
    hf_results = {}
    if compare_hf:
        print("Labeling with HF zero-shot classifier...", file=sys.stderr)
        hf_times = []
        for i, (motion_id, text, ent, probs) in enumerate(top_n):
            result = label_with_hf_zero_shot(text, category_names)
            hf_results[motion_id] = result
            if "latency" in result:
                hf_times.append(result["latency"])
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(top_n)} labeled, avg latency={np.mean(hf_times):.3f}s" if hf_times else f"  {i+1}/{len(top_n)} labeled", file=sys.stderr)

    # Evaluate agreement and insert agreed labels
    agreed = 0
    inserted = 0
    ollama_valid = 0
    hf_valid = 0

    for motion_id, text, ent, probs in top_n:
        ores = ollama_results.get(motion_id, {})
        hres = hf_results.get(motion_id, {})

        ocat = ores.get("category", "").lower()
        hcat = hres.get("category", "").lower()
        oconf = ores.get("confidence", 0)

        if ocat in category_names:
            ollama_valid += 1
        if hcat in category_names:
            hf_valid += 1

        # Insert if Ollama confident and optionally agrees with HF
        if ocat in category_names and oconf >= confidence_threshold:
            if compare_hf and hcat in category_names:
                if ocat == hcat:
                    agreed += 1
                    source = "active_learning_agreed"
                else:
                    source = "active_learning_ollama_disputed"
            else:
                agreed += 1
                source = "active_learning_ollama"

            if agree_insert:
                reason = ores.get("reasoning", "Active learning Ollama label")
                cur.execute(
                    """INSERT OR IGNORE INTO augmented_gold_labels
                    (motion_id, category, reasoning, source, parent_id, text, created_at, split)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        motion_id,
                        ocat,
                        f"[entropy={ent:.3f}] {reason[:500]}",
                        source,
                        None,
                        text,
                        datetime.now(timezone.utc).isoformat(),
                        "train",
                    ),
                )
                if cur.rowcount > 0:
                    inserted += 1

    conn.commit()

    # Summary
    print("\n=== Active Learning Summary ===", file=sys.stderr)
    print(f"Pool size: {n_pool}", file=sys.stderr)
    print(f"Selected: {n_select} (highest entropy)", file=sys.stderr)
    print(f"Ollama valid labels: {ollama_valid}/{n_select}", file=sys.stderr)
    if compare_hf:
        print(f"HF valid labels: {hf_valid}/{n_select}", file=sys.stderr)
    print(f"Agreement (ollama>=conf + hf match): {agreed}/{n_select}", file=sys.stderr)
    print(f"Inserted into gold set: {inserted}", file=sys.stderr)
    if ollama_times:
        print(f"Ollama avg latency: {np.mean(ollama_times):.2f}s (total {np.sum(ollama_times):.0f}s)", file=sys.stderr)
    if compare_hf and hf_times:
        print(f"HF avg latency: {np.mean(hf_times):.3f}s (total {np.sum(hf_times):.0f}s)", file=sys.stderr)

    # Per-category distribution of inserted labels
    cur.execute("SELECT category, COUNT(*) FROM augmented_gold_labels WHERE source LIKE 'active_learning%' GROUP BY category")
    print("\nActive learning label distribution:", file=sys.stderr)
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}", file=sys.stderr)

    # Optionally retrain
    if retrain_after and inserted > 0:
        print("\nRetraining ensemble with new labels...", file=sys.stderr)
        from swedish_parliament_policy_classifier.scripts.train_models import train_all
        train_all(db_path=db_path, force_retrain_ensemble=True)

    return inserted


def main():
    parser = argparse.ArgumentParser(description="Active learning pipeline")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--select", type=int, default=500, help="Number of high-entropy motions to label")
    parser.add_argument("--pool", type=int, default=5000, help="Pool size to sample from")
    parser.add_argument("--ollama-model", default="qwen2.5-coder-14b-32k:latest", help="Ollama model for labeling")
    parser.add_argument("--compare-hf", action="store_true", help="Compare with HF zero-shot classifier")
    parser.add_argument("--confidence-threshold", type=float, default=7.0, help="Ollama confidence threshold (0-10)")
    parser.add_argument("--no-insert", action="store_true", help="Do not insert labels into DB (dry run)")
    parser.add_argument("--retrain", action="store_true", help="Retrain ensemble after inserting labels")
    args = parser.parse_args()

    active_learning(
        db_path=args.db,
        n_select=args.select,
        n_pool=args.pool,
        ollama_model=args.ollama_model,
        compare_hf=args.compare_hf,
        confidence_threshold=args.confidence_threshold,
        agree_insert=not args.no_insert,
        retrain_after=args.retrain,
    )


if __name__ == "__main__":
    main()
