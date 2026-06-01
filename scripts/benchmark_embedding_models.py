#!/usr/bin/env python3
"""Benchmark multiple sentence-transformer base models on validation split.

Evaluates each model's ability to match motions to their correct category
definition via cosine similarity. Reports validation accuracy and avg cosine
score for ranking.

Usage:
    uv run python3 scripts/benchmark_embedding_models.py --db data/swedish_parliament.db
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dotenv import load_dotenv
load_dotenv()

if os.getenv("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.getenv("HF_TOKEN"))

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Models to benchmark
# ---------------------------------------------------------------------------
MODELS = [
    ("KBLab/sentence-bert-swedish-cased", "KB-SBERT (current baseline)"),
    ("intfloat/multilingual-e5-base", "multilingual-e5-base"),
    ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "paraphrase-mpnet-multilingual"),
    ("KBLab/bert-base-swedish-cased-new", "KB-BERT-new (swedish-cased-new)"),
]


def load_split_motions(conn, split: str):
    """Return list of (motion_id, text, category) for a given split."""
    cur = conn.cursor()
    cur.execute("""
        SELECT a.motion_id, COALESCE(nm.text, a.text), a.category
        FROM augmented_gold_labels a
        LEFT JOIN normalized_motions nm ON a.motion_id = nm.id
        WHERE a.split = ?
          AND COALESCE(nm.text, a.text) IS NOT NULL
          AND LENGTH(COALESCE(nm.text, a.text)) > 0
    """, (split,))
    return cur.fetchall()


def evaluate_model(model, val_rows: List[Tuple], categories) -> Tuple[float, float]:
    """Compute validation accuracy and average cosine similarity.

    Returns (accuracy, avg_cosine_score).
    """
    cat_defs = {cat: categories[cat].definition or "" for cat in categories}
    cat_embs = {
        cat: model.encode([d], show_progress_bar=False, convert_to_numpy=True)[0]
        for cat, d in cat_defs.items()
    }

    total = 0
    correct = 0
    scores = []

    for _, text, true_cat in val_rows:
        q = model.encode([text[:1500]], show_progress_bar=False, convert_to_numpy=True)
        best_cat = None
        best_score = -2.0
        for cat, emb in cat_embs.items():
            s = cosine_similarity(q, emb.reshape(1, -1))[0, 0]
            if s > best_score:
                best_score = s
                best_cat = cat
        if best_cat == true_cat:
            correct += 1
        total += 1
        scores.append(best_score)

    acc = correct / total if total > 0 else 0.0
    avg_score = float(np.mean(scores)) if scores else 0.0
    return acc, avg_score


def benchmark(db_path: str = "data/swedish_parliament.db"):
    conn = init_db(db_path)
    defs = load_definitions()
    val_rows = load_split_motions(conn, "val")
    print(f"Loaded {len(val_rows)} validation motions.", file=sys.stderr)

    results = []
    for model_name, display_name in MODELS:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Evaluating: {display_name} ({model_name})", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(model_name, device=device)
            acc, avg_cos = evaluate_model(model, val_rows, defs)
            print(f"  -> val_acc={acc:.3f}, avg_cosine={avg_cos:.3f}", file=sys.stderr)
            results.append((display_name, model_name, acc, avg_cos))

            # Move to CPU and clear GPU memory before next model
            model = model.to("cpu")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  -> FAILED: {e}", file=sys.stderr)
            results.append((display_name, model_name, 0.0, 0.0))

    # Rank by validation accuracy
    print(f"\n{'='*60}", file=sys.stderr)
    print("BENCHMARK RESULTS (ranked by val accuracy)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for rank, (display, name, acc, avg_cos) in enumerate(
        sorted(results, key=lambda x: x[2], reverse=True), start=1
    ):
        print(f"  {rank}. {display} ({name})", file=sys.stderr)
        print(f"      val_acc={acc:.3f} | avg_cosine={avg_cos:.3f}", file=sys.stderr)

    best = max(results, key=lambda x: x[2])
    print(f"\nBest base model: {best[0]} ({best[1]}) — val_acc={best[2]:.3f}", file=sys.stderr)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding base models")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    args = parser.parse_args()
    benchmark(db_path=args.db)


if __name__ == "__main__":
    main()
