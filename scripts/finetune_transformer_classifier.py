#!/usr/bin/env python3
"""Fine-tune KBLab/bert-base-swedish-cased-new as a 7-class ideology classifier.

End-to-end transformer approach that classifies parliamentary motions directly
from their text, bypassing the feature-engineered ensemble pipeline.

Usage:
    uv run python scripts/finetune_transformer_classifier.py --db data/swedish_parliament.db --epochs 3 --batch-size 8
"""

import argparse
import json
import os
import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()
if os.getenv("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.getenv("HF_TOKEN"))

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
        DataCollatorWithPadding,
    )
    from datasets import Dataset as HFDataset
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    HAS_TRANSFORMERS = True
except Exception as e:
    print(f"Missing required libraries: {e}", file=sys.stderr)
    HAS_TRANSFORMERS = False
    sys.exit(1)

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions

MODEL_NAME = "KBLab/bert-base-swedish-cased-new"
MAX_LEN = 512
DEFAULT_OUTPUT_DIR = "models/transformer_ideology_classifier"


def load_gold_data(conn: sqlite3.Connection, split: str, max_text_len: int = 4000) -> Tuple[List[str], List[str]]:
    """Load (text, category) pairs from augmented_gold_labels for a given split."""
    cur = conn.cursor()
    cur.execute("""
        SELECT COALESCE(nm.text, a.text) AS text, a.category
        FROM augmented_gold_labels a
        LEFT JOIN normalized_motions nm ON a.motion_id = nm.id
        WHERE a.split = ? AND COALESCE(nm.text, a.text) IS NOT NULL
          AND LENGTH(COALESCE(nm.text, a.text)) > 50
    """, (split,))
    rows = cur.fetchall()
    texts = []
    labels = []
    for text, cat in rows:
        # Truncate very long texts to avoid excessive tokenization time
        if len(text) > max_text_len:
            text = text[:max_text_len]
        texts.append(text)
        labels.append(cat)
    return texts, labels


class MotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN, label2id=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id or {cat: i for i, cat in enumerate(sorted(set(labels)))}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.label2id[self.labels[idx]], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


def finetune_transformer(
    db_path: str = "data/swedish_parliament.db",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_text_len: int = 4000,
    seed: int = 42,
):
    if not HAS_TRANSFORMERS:
        sys.exit(1)

    conn = init_db(db_path)
    defs = load_definitions()
    categories = sorted(defs.keys())
    label2id = {cat: i for i, cat in enumerate(categories)}
    id2label = {i: cat for cat, i in label2id.items()}

    print(f"Loading data for {len(categories)} categories...", file=sys.stderr)
    train_texts, train_labels = load_gold_data(conn, "train", max_text_len=max_text_len)
    val_texts, val_labels = load_gold_data(conn, "val", max_text_len=max_text_len)
    test_texts, test_labels = load_gold_data(conn, "test", max_text_len=max_text_len)

    print(f"Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}", file=sys.stderr)
    if len(train_texts) < 100:
        print("Not enough training data. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Compute class weights for imbalance
    from sklearn.utils.class_weight import compute_class_weight
    y_train_idx = [label2id[l] for l in train_labels]
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(categories)), y=y_train_idx)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    print("Loading tokenizer and model...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(categories),
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    class_weights_tensor = class_weights_tensor.to(device)

    # Custom loss with class weights
    original_forward = model.forward

    def weighted_forward(*args, **kwargs):
        labels = kwargs.get("labels")
        outputs = original_forward(*args, **kwargs)
        if labels is not None:
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
            loss = loss_fct(logits, labels)
            outputs.loss = loss
        return outputs

    model.forward = weighted_forward

    # Create datasets
    train_dataset = MotionDataset(train_texts, train_labels, tokenizer, label2id=label2id)
    val_dataset = MotionDataset(val_texts, val_labels, tokenizer, label2id=label2id)
    test_dataset = MotionDataset(test_texts, test_labels, tokenizer, label2id=label2id)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training...", file=sys.stderr)
    trainer.train()

    # Evaluate on test set
    print("\nEvaluating on test set...", file=sys.stderr)
    test_results = trainer.evaluate(test_dataset)
    print(f"\nTest results: {json.dumps(test_results, indent=2)}", file=sys.stderr)

    # Save final model and tokenizer
    trainer.save_model(str(output_path / "final"))
    tokenizer.save_pretrained(str(output_path / "final"))

    # Save config
    config = {
        "model_name": MODEL_NAME,
        "categories": categories,
        "label2id": label2id,
        "id2label": id2label,
        "max_len": MAX_LEN,
        "test_results": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in test_results.items()},
    }
    with open(output_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\nModel saved to {output_path / 'final'}", file=sys.stderr)
    return test_results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Swedish BERT as ideology classifier")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-text-len", type=int, default=4000, help="Max characters per motion text")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    finetune_transformer(
        db_path=args.db,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_text_len=args.max_text_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
