#!/usr/bin/env python3
"""Prepare a blinded, stratified speech sample for independent adjudication."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from swedish_parliament_policy_classifier.io.speech_text import ParquetSpeechTextRepository


def prepare_review_sample(
    labels_path: str | Path,
    speech_dir: str | Path,
    output_path: str | Path,
    *,
    per_category: int = 25,
    random_state: int = 42,
) -> Path:
    labels = pd.read_parquet(labels_path)
    required = {"speech_id", "category"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Gold labels missing required columns: {missing}")

    repository = ParquetSpeechTextRepository(Path(speech_dir))
    labels = labels.copy()
    labels["speech_id"] = labels["speech_id"].astype(str)
    labels["review_text"] = [repository.get(speech_id) for speech_id in labels["speech_id"]]
    labels = labels[labels["review_text"].str.strip().ne("")].copy()
    sampled = []
    for _, category_frame in labels.groupby("category", sort=True):
        sampled.append(
            category_frame.sample(
                min(len(category_frame), per_category), random_state=random_state
            )
        )
    labels = pd.concat(sampled, ignore_index=True) if sampled else labels.iloc[0:0].copy()
    labels.insert(0, "review_id", [f"speech-review-{index:04d}" for index in range(1, len(labels) + 1)])
    labels["adjudicated_category"] = ""
    labels["adjudicator"] = ""
    labels["adjudication_notes"] = ""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels[["review_id", "speech_id", "category"]].to_parquet(
        output.with_name(f"{output.stem}_key{output.suffix}"), index=False
    )
    labels[["review_id", "speech_id", "review_text", "adjudicated_category", "adjudicator", "adjudication_notes"]].to_parquet(output, index=False)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a blinded speech gold-label review sample")
    parser.add_argument("--labels", default="data/parquet/speech_gold_labels.parquet")
    parser.add_argument("--speech-dir", default="data/speeches/parquet")
    parser.add_argument("--out", default="output/review/speech_gold_review.parquet")
    parser.add_argument("--per-category", type=int, default=25)
    args = parser.parse_args()
    path = prepare_review_sample(args.labels, args.speech_dir, args.out, per_category=args.per_category)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())