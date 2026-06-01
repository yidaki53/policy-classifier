#!/usr/bin/env python3
"""Create stratified train/test/validate split from augmented gold labels.

Stores split assignments in augmented_gold_labels.split column (or separate table).
70/15/15 split, stratified by category.

Usage:
    uv run python3 scripts/create_train_test_split.py --db data/swedish_parliament.db
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def create_split_column(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(augmented_gold_labels)")
    cols = [r[1] for r in cur.fetchall()]
    if "split" not in cols:
        cur.execute("ALTER TABLE augmented_gold_labels ADD COLUMN split TEXT")
        conn.commit()


def load_augmented_labels(conn: sqlite3.Connection) -> List[Tuple[str, str]]:
    cur = conn.cursor()
    cur.execute("SELECT motion_id, category FROM augmented_gold_labels")
    return cur.fetchall()


def create_split(
    db_path: str = "data/swedish_parliament.db",
    train_ratio: float = 0.50,
    test_ratio: float = 0.25,
    val_ratio: float = 0.25,
    random_state: int = 42,
):
    assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 0.001

    conn = sqlite3.connect(db_path)
    create_split_column(conn)

    rows = load_augmented_labels(conn)
    ids = [r[0] for r in rows]
    labels = [r[1] for r in rows]

    # First split: train vs temp (test+val)
    temp_ratio = test_ratio + val_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=random_state)
    train_idx, temp_idx = next(sss1.split(ids, labels))

    # Second split: test vs val from temp
    temp_ids = [ids[i] for i in temp_idx]
    temp_labels = [labels[i] for i in temp_idx]
    val_frac = val_ratio / temp_ratio
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
    test_idx, val_idx = next(sss2.split(temp_ids, temp_labels))

    train_ids = {ids[i] for i in train_idx}
    test_ids = {temp_ids[i] for i in test_idx}
    val_ids = {temp_ids[i] for i in val_idx}

    print(f"Split: train={len(train_ids)} test={len(test_ids)} val={len(val_ids)}", file=sys.stderr)

    # Check category distribution in each split
    from collections import Counter
    print("\nPer-split distribution:", file=sys.stderr)
    for split_name, split_ids in [("train", train_ids), ("test", test_ids), ("val", val_ids)]:
        split_labels = [labels[ids.index(i)] for i in split_ids]
        counts = Counter(split_labels)
        print(f"  {split_name}: {dict(counts)}", file=sys.stderr)

    # Assign splits in DB
    cur = conn.cursor()
    cur.execute("UPDATE augmented_gold_labels SET split = NULL")
    for split_name, split_ids in [("train", train_ids), ("test", test_ids), ("val", val_ids)]:
        cur.executemany(
            "UPDATE augmented_gold_labels SET split = ? WHERE motion_id = ?",
            [(split_name, mid) for mid in split_ids],
        )
    conn.commit()

    # Verify
    cur.execute("SELECT split, COUNT(*), COUNT(DISTINCT category) FROM augmented_gold_labels WHERE split IS NOT NULL GROUP BY split")
    for row in cur.fetchall():
        print(f"  DB: {row[0]}={row[1]} rows, {row[2]} classes", file=sys.stderr)

    conn.close()
    return len(train_ids), len(test_ids), len(val_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--train-ratio", type=float, default=0.50)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    args = parser.parse_args()
    train, test, val = create_split(
        db_path=args.db,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"Split created: train={train} test={test} val={val}")


if __name__ == "__main__":
    main()
