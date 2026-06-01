#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

from swedish_parliament_policy_classifier.analysis.contradiction_scoring import (
    aggregate_expected_contradiction,
    score_contradiction_edges,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Score say-vs-do contradiction edges and aggregate expected contradiction")
    p.add_argument("--speech-action-links", default="data/parquet/speech_action_links.parquet")
    p.add_argument("--axis-scores", default="output/analysis/speech_action_axis_scores.parquet")
    p.add_argument("--edge-out", default="output/analysis/speech_action_contradiction_edges.parquet")
    p.add_argument(
        "--expected-out",
        default="output/analysis/speech_action_expected_contradiction_party_topic_year.parquet",
    )
    args = p.parse_args()

    edge_summary = score_contradiction_edges(
        speech_action_links_path=args.speech_action_links,
        axis_scores_path=args.axis_scores,
        out_path=args.edge_out,
    )
    expected_summary = aggregate_expected_contradiction(
        edge_scores_path=args.edge_out,
        out_path=args.expected_out,
    )

    print(json.dumps({"edge_summary": edge_summary, "expected_summary": expected_summary}, indent=2))


if __name__ == "__main__":
    main()
