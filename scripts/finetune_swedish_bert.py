#!/usr/bin/env python3
"""Comprehensive fine-tuning pipeline for Swedish parliamentary motion embeddings.

Phases
------
1. Benchmark multiple base embedding models on validation split (with e5 prefixes).
2. MLM domain adaptation on ~200 k unlabeled motions.
3. Fine-tune best model with party-neutral objectives:
   - CosineSimilarityLoss (motion vs its category definition -> 1.0)
   - ContrastiveLoss (positive motion-definition, negative motion-other_definitions)
   - MultipleNegativesRankingLoss (motion-definition with in-batch negatives)
4. Early stopping (patience=3, max 10 epochs).
5. Save best checkpoint to models/finetuned_swedish_bert/.
6. Optionally retrain the LightGBM ensemble.

Usage:
    poetry run python3 scripts/finetune_swedish_bert.py --db data/swedish_parliament.db
    poetry run python3 scripts/finetune_swedish_bert.py --db data/swedish_parliament.db --retrain-ensemble
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()
if os.getenv("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.getenv("HF_TOKEN"))

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import (
    CosineSimilarityLoss,
    ContrastiveLoss,
    MultipleNegativesRankingLoss,
)
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_CANDIDATES: List[Tuple[str, str]] = [
    ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "mpnet-multilingual"),
    ("intfloat/multilingual-e5-base", "e5-base"),
    ("KBLab/sentence-bert-swedish-cased", "KB-SBERT"),
    ("KBLab/bert-base-swedish-cased-new", "KB-BERT-new"),
    ("KBLab/bert-base-swedish-cased", "KB-BERT"),
]

MAX_FT_EPOCHS = 10
FT_PATIENCE = 3
FT_BATCH_SIZE = 16
MLM_MAX_STEPS = 5000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def apply_prefix(text: str, model_name: str, role: str = "query") -> str:
    """Apply required prefixes for instruction-tuned embedding models."""
    if "e5" in model_name.lower():
        prefix = "query: " if role == "query" else "passage: "
        return prefix + text
    return text


def load_split_motions(conn, split: str):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT a.motion_id, COALESCE(nm.text, a.text), a.category
        FROM augmented_gold_labels a
        LEFT JOIN normalized_motions nm ON a.motion_id = nm.id
        WHERE a.split = ?
          AND COALESCE(nm.text, a.text) IS NOT NULL
          AND LENGTH(COALESCE(nm.text, a.text)) > 0
        """,
        (split,),
    )
    return cur.fetchall()


def load_unlabeled_motions(conn, min_length: int = 50, max_samples: int = 200_000):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT text FROM normalized_motions
        WHERE text IS NOT NULL AND LENGTH(text) >= ?
        LIMIT ?
        """,
        (min_length, max_samples),
    )
    rows = [r[0] for r in cur.fetchall()]
    _log(f"Loaded {len(rows)} unlabeled motions for MLM")
    return rows


def evaluate_model(model, model_name: str, val_rows, categories) -> Tuple[float, float]:
    cat_defs = {
        cat: apply_prefix(categories[cat].definition or "", model_name, "passage")
        for cat in categories
    }
    cat_embs = {
        cat: model.encode([d], show_progress_bar=False, convert_to_numpy=True)[0]
        for cat, d in cat_defs.items()
    }

    total = correct = 0
    scores = []
    for _, text, true_cat in val_rows:
        q_text = apply_prefix(text[:1500], model_name, "query")
        q = model.encode([q_text], show_progress_bar=False, convert_to_numpy=True)
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

    acc = correct / total if total else 0.0
    avg_score = float(np.mean(scores)) if scores else 0.0
    return acc, avg_score


# ---------------------------------------------------------------------------
# Example builders (party-neutral: only motion text + category definitions)
# ---------------------------------------------------------------------------
def build_cosine_examples(train_rows, categories, model_name: str) -> List[InputExample]:
    examples = []
    for _, text, cat in train_rows:
        motion = apply_prefix(text[:1500], model_name, "query")
        definition = apply_prefix(categories[cat].definition or "", model_name, "passage")
        examples.append(InputExample(texts=[motion, definition], label=1.0))
    return examples


def build_contrastive_examples(train_rows, categories, model_name: str) -> List[InputExample]:
    examples = []
    cat_names = list(categories.keys())
    for _, text, cat in train_rows:
        motion = apply_prefix(text[:1500], model_name, "query")
        pos_def = apply_prefix(categories[cat].definition or "", model_name, "passage")
        examples.append(InputExample(texts=[motion, pos_def], label=1.0))
        for neg_cat in cat_names:
            if neg_cat == cat:
                continue
            neg_def = apply_prefix(categories[neg_cat].definition or "", model_name, "passage")
            examples.append(InputExample(texts=[motion, neg_def], label=0.0))
    return examples


def build_mnrl_examples(train_rows, categories, model_name: str) -> List[InputExample]:
    examples = []
    for _, text, cat in train_rows:
        motion = apply_prefix(text[:1500], model_name, "query")
        definition = apply_prefix(categories[cat].definition or "", model_name, "passage")
        examples.append(InputExample(texts=[motion, definition]))
    return examples


# ---------------------------------------------------------------------------
# Generic early-stopping trainer
# ---------------------------------------------------------------------------
def train_variant(
    train_rows,
    val_rows,
    categories,
    base_model_name: str,
    build_examples_fn,
    loss_cls,
    epochs: int = MAX_FT_EPOCHS,
    patience: int = FT_PATIENCE,
    batch_size: int = FT_BATCH_SIZE,
):
    model = SentenceTransformer(base_model_name)
    examples = build_examples_fn(train_rows, categories, base_model_name)
    if not examples:
        _log("  No training examples generated.")
        return None, 0.0

    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = loss_cls(model)

    best_acc = -1.0
    best_model_dir = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        _log(f"  Epoch {epoch}/{epochs}...")
        try:
            model.fit(
                train_objectives=[(loader, loss)],
                epochs=1,
                warmup_steps=100 if epoch == 1 else 0,
                show_progress_bar=False,
                use_amp=True,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                _log(f"  OOM with bs={batch_size}, halving...")
                batch_size = max(4, batch_size // 2)
                loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
                model.fit(
                    train_objectives=[(loader, loss)],
                    epochs=1,
                    warmup_steps=100 if epoch == 1 else 0,
                    show_progress_bar=False,
                    use_amp=True,
                )
            else:
                raise

        acc, _ = evaluate_model(model, base_model_name, val_rows, categories)
        _log(f"  Epoch {epoch} -> val_acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            if best_model_dir and os.path.isdir(best_model_dir):
                shutil.rmtree(best_model_dir, ignore_errors=True)
            best_model_dir = tempfile.mkdtemp(prefix=f"ft_{loss_cls.__name__}_")
            model.save(best_model_dir)
            _log(f"  -> New best checkpoint (acc={acc:.3f})")
        else:
            no_improve += 1
            _log(f"  -> No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                _log(f"  Early stopping at epoch {epoch}")
                break

    if best_model_dir:
        best_model = SentenceTransformer(best_model_dir)
        shutil.rmtree(best_model_dir, ignore_errors=True)
        return best_model, best_acc
    return model, best_acc


# ---------------------------------------------------------------------------
# MLM Domain Adaptation
# ---------------------------------------------------------------------------
def run_mlm_domain_adaptation(model_name: str, texts: List[str], output_dir: Path):
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForMaskedLM,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
        from datasets import Dataset
    except ImportError as e:
        _log(f"  Skipping MLM: missing dependency ({e})")
        return None

    _log(f"Starting MLM domain adaptation for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    dataset = Dataset.from_dict({"text": texts})

    max_length = 256

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    mlm_dir = output_dir / "mlm_adapted"
    mlm_dir.mkdir(parents=True, exist_ok=True)

    # Try gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=str(mlm_dir),
        max_steps=MLM_MAX_STEPS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        save_steps=2500,
        save_total_limit=2,
        logging_steps=500,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    try:
        trainer.train()
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            _log("  OOM during MLM, skipping domain adaptation and proceeding with base model...")
            torch.cuda.empty_cache()
            return None
        else:
            raise

    final_dir = mlm_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    _log(f"MLM adapted model saved to {final_dir}")
    return str(final_dir)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark_models(db_path: str) -> List[Tuple[str, str, float, float]]:
    conn = init_db(db_path)
    defs = load_definitions()
    val_rows = load_split_motions(conn, "val")
    _log(f"Benchmarking on {len(val_rows)} validation motions")

    results = []
    for model_name, display_name in MODEL_CANDIDATES:
        _log(f"Evaluating {display_name} ({model_name})")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(model_name, device=device)
            acc, avg_cos = evaluate_model(model, model_name, val_rows, defs)
            _log(f"  -> val_acc={acc:.3f}, avg_cosine={avg_cos:.3f}")
            results.append((display_name, model_name, acc, avg_cos))

            model = model.to("cpu")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            _log(f"  -> FAILED: {e}")
            results.append((display_name, model_name, 0.0, 0.0))

    results.sort(key=lambda x: x[2], reverse=True)
    _log("Benchmark ranking:")
    for rank, (display, name, acc, avg_cos) in enumerate(results, 1):
        _log(f"  {rank}. {display} ({name}) — val_acc={acc:.3f}, avg_cos={avg_cos:.3f}")
    return results


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def fine_tune_and_select(db_path: str, retrain_ensemble: bool = False):
    start_time = time.time()
    out_dir = Path(__file__).resolve().parents[1] / "models" / "finetuned_swedish_bert"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Benchmark ------------------------------------------------
    _log("=" * 60)
    _log("PHASE 1: Benchmarking base models")
    _log("=" * 60)
    bench_results = benchmark_models(db_path)
    best_base_name = bench_results[0][1]
    best_base_display = bench_results[0][0]
    _log(f"Best base model: {best_base_display} ({best_base_name})")

    # --- Phase 2: MLM Domain Adaptation ----------------------------------
    _log("=" * 60)
    _log("PHASE 2: MLM Domain Adaptation")
    _log("=" * 60)
    conn = init_db(db_path)
    unlabeled_texts = load_unlabeled_motions(conn)
    mlm_path = run_mlm_domain_adaptation(best_base_name, unlabeled_texts, out_dir.parent)

    base_for_ft = mlm_path if mlm_path else best_base_name
    _log(f"Base for fine-tuning: {base_for_ft}")

    # --- Phase 3: Task-specific fine-tuning -------------------------------
    _log("=" * 60)
    _log("PHASE 3: Task-specific fine-tuning")
    _log("=" * 60)
    train_rows = load_split_motions(conn, "train")
    val_rows = load_split_motions(conn, "val")
    defs = load_definitions()
    _log(f"Train={len(train_rows)}, Val={len(val_rows)}")

    variants = [
        ("baseline", None, None),
        ("cosine", build_cosine_examples, CosineSimilarityLoss),
        ("contrastive", build_contrastive_examples, ContrastiveLoss),
        ("mnrl", build_mnrl_examples, MultipleNegativesRankingLoss),
    ]

    results = []
    for variant_name, build_fn, loss_cls in variants:
        if build_fn is None:
            _log(f"\n[Baseline – {base_for_ft}]")
            model = SentenceTransformer(base_for_ft)
            acc, _ = evaluate_model(model, base_for_ft, val_rows, defs)
            _log(f"  -> val_acc={acc:.3f}")
            model_cpu = model.to("cpu")
            del model
            torch.cuda.empty_cache()
            results.append((variant_name, acc, model_cpu))
            continue

        _log(f"\n[{variant_name.upper()} fine-tuning]")
        ft_model, ft_acc = train_variant(
            train_rows,
            val_rows,
            defs,
            base_for_ft,
            build_fn,
            loss_cls,
            epochs=MAX_FT_EPOCHS,
            patience=FT_PATIENCE,
        )
        if ft_model is not None:
            ft_model_cpu = ft_model.to("cpu")
            del ft_model
            torch.cuda.empty_cache()
            results.append((variant_name, ft_acc, ft_model_cpu))
        else:
            results.append((variant_name, 0.0, None))

    # --- Phase 4: Pick best and save --------------------------------------
    valid_results = [(n, a, m) for n, a, m in results if m is not None]
    best_name, best_acc, best_model = max(valid_results, key=lambda x: x[1])
    _log(f"\n{'='*60}")
    _log(f"BEST MODEL: {best_name} (val_acc={best_acc:.3f})")
    _log(f"{'='*60}")

    best_model.save(str(out_dir))
    _log(f"Saved to {out_dir}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": best_base_name,
        "mlm_adapted": mlm_path is not None,
        "fine_tuning_base": base_for_ft,
        "best_variant": best_name,
        "best_val_acc": best_acc,
        "benchmark": [
            {"name": n, "model": m, "val_acc": a, "avg_cos": c}
            for n, m, a, c in bench_results
        ],
        "variants": [
            {"name": n, "val_acc": a}
            for n, a, _ in results
        ],
    }
    summary_path = out_dir.parent / "finetuning_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"Summary saved to {summary_path}")

    elapsed = time.time() - start_time
    _log(f"Total elapsed: {timedelta(seconds=int(elapsed))}")

    # --- Phase 5: Retrain ensemble (optional) -----------------------------
    if retrain_ensemble:
        _log("=" * 60)
        _log("PHASE 5: Retraining ensemble")
        _log("=" * 60)
        repo_root = Path(__file__).resolve().parents[1]
        train_script = repo_root / "scripts" / "train_models.py"
        try:
            result = subprocess.run(
                [sys.executable, str(train_script)],
                capture_output=True,
                text=True,
                cwd=str(repo_root),
            )
            _log(result.stdout)
            if result.returncode != 0:
                _log(f"Ensemble stderr: {result.stderr}")
        except Exception as e:
            _log(f"  Ensemble retraining failed: {e}")

    return best_name, best_acc, str(out_dir)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive embedding fine-tuning")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument(
        "--retrain-ensemble",
        action="store_true",
        help="Retrain LightGBM ensemble after fine-tuning",
    )
    args = parser.parse_args()
    fine_tune_and_select(db_path=args.db, retrain_ensemble=args.retrain_ensemble)


if __name__ == "__main__":
    main()
