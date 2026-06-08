#!/usr/bin/env python3
"""Run the speech classifier on a stratified sample and write a reviewable report."""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
import pandas as pd
import re

from swedish_parliament_policy_classifier.exports import load_definitions
from swedish_parliament_policy_classifier.classifier.scorer import score_motion
from swedish_parliament_policy_classifier.classifier.ensemble import load_meta_classifier
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher
from swedish_parliament_policy_classifier.nlp.topic_modeler import load_topic_distributions
from swedish_parliament_policy_classifier.nlp.preprocess import extract_plain_text_from_html
from swedish_parliament_policy_classifier.io.markdown_frontmatter import ensure_frontmatter


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
    meta_clf = None
    try:
        speech_model = Path("models/speech_meta_clf.pkl")
        if speech_model.exists():
            meta_clf = load_meta_classifier(model_path=speech_model)
            print(f"Using speech-specific meta-classifier: {speech_model}")
        else:
            print("No speech-specific meta-classifier found; skipping meta-classifier for speech validation")
    except Exception as e:
        print(f"Meta-classifier not loaded: {e}")
        meta_clf = None

    with open("stratified_sample_ids.txt") as f:
        sample_ids = [line.strip() for line in f]

    parquet_dir = Path("data/speeches/parquet")
    speeches = [pd.read_parquet(f) for f in parquet_dir.glob("*.parquet")]
    all_speeches = pd.concat(speeches, ignore_index=True)
    sample_df = all_speeches[all_speeches["anforande_id"].isin(sample_ids)]

    report_header = ensure_frontmatter(
        "# Stratified Speech Classification Report\n\n",
        {
            "_agent_frontmatter": {
                "id": "reports.stratified_classification",
                "purpose": "Generated review report for stratified speech classification sample.",
                "steward": "classifier",
                "edit_policy": "generated_do_not_edit",
                "generator": "scripts/run_stratified_sample.py",
            },
            "source_ids": "stratified_sample_ids.txt",
        },
    )

    with open("stratified_classification_report.md", "w", encoding="utf-8") as out:
        out.write(report_header)

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
                ollama_weight=0.60,
            )

            # Sort by normalized_weight descending
            result = sorted(result, key=lambda r: r.normalized_weight, reverse=True)
            top = result[0]

            out.write(f"\n## {row['talare']} ({row['parti']})\n")
            out.write(f"**Title:** {row['avsnittsrubrik']}\n")
            out.write(f"**Date:** {row['datum']}\n")
            out.write(f"**Top category:** {top.category} ({top.normalized_weight:.4f})\n")
            out.write(f"**Classifier version:** {top.classifier_version}\n")
            out.write(f"**ID:** {row['anforande_id']}\n\n")
            out.write("### Matched rules\n")
            for r in result:
                if r.matched_rules:
                    out.write(f"- {r.category}: {', '.join(r.matched_rules[:5])}\n")
            out.write("\n")

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
