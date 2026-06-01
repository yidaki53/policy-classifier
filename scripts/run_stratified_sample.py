#!/usr/bin/env python3
"""Run the speech classifier on a stratified sample and write a reviewable report."""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
import pandas as pd
import re

from swedish_parliament_policy_classifier.classifier.scorer import load_definitions, score_motion
from swedish_parliament_policy_classifier.classifier.ensemble import load_meta_classifier
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher
from swedish_parliament_policy_classifier.nlp.topic_modeler import load_topic_distributions
from swedish_parliament_policy_classifier.nlp.preprocess import extract_plain_text_from_html


def main():
    defs = load_definitions()

    try:
        matcher = EmbeddingMatcher()
        if matcher.model is None:
            matcher = None
    except Exception as e:
        print(f"Embedding matcher unavailable: {e}")
        matcher = None

    topic_dists = load_topic_distributions()
    try:
        meta_clf = load_meta_classifier()
    except Exception as e:
        print(f"Meta-classifier not loaded: {e}")
        meta_clf = None

    with open("stratified_sample_ids.txt") as f:
        sample_ids = [line.strip() for line in f]

    parquet_dir = Path("data/speeches/parquet")
    speeches = [pd.read_parquet(f) for f in parquet_dir.glob("*.parquet")]
    all_speeches = pd.concat(speeches, ignore_index=True)
    sample_df = all_speeches[all_speeches["anforande_id"].isin(sample_ids)]

    with open("stratified_classification_report.md", "w", encoding="utf-8") as out:
        out.write("# Stratified Speech Classification Report\n\n")

        for _, row in sample_df.iterrows():
            text = row["anforandetext"] or ""
            if "<" in text and ">" in text:
                text = extract_plain_text_from_html(text)

            result = score_motion(
                motion_id=row["anforande_id"],
                text=text,
                categories=defs,
                party=None,
                embedding_matcher=matcher,
                use_zero_shot=True,
                topic_distributions=topic_dists,
                meta_clf=meta_clf,
                skip_policy_extraction=True,
                use_speech_preprocessing=True,
                use_ollama=True,
                ollama_weight=0.55,
            )

            # Sort by normalized_weight descending
            result = sorted(result, key=lambda r: r.normalized_weight, reverse=True)
            top = result[0]

            out.write(f"\n## {row['talare']} ({row['parti']})\n")
            out.write(f"**Title:** {row['avsnittsrubrik']}\n")
            out.write(f"**Date:** {row['datum']}\n")
            out.write(f"**Top category:** {top.category} ({top.normalized_weight:.4f})\n")
            out.write(f"**ID:** {row['anforande_id']}\n\n")

            out.write("### All category scores\n")
            for r in result:
                out.write(f"- {r.category}: {r.normalized_weight:.4f}\n")

            # Show first 800 chars of preprocessed text
            out.write("\n### Text preview\n")
            text_clean = re.sub(r'<[^>]+>', '', row["anforandetext"] or "")[:800]
            out.write(text_clean)
            if len(text_clean) >= 800:
                out.write("...")
            out.write("\n\n---\n")

            # Also print to stderr for progress
            print(
                f"[{row['parti']:>3}] {row['talare']:30s} -> {top.category} ({top.normalized_weight:.4f})",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
