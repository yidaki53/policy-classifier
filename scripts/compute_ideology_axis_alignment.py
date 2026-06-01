#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

from swedish_parliament_policy_classifier.analysis.ideology_axes import compute_axis_alignment


def main() -> None:
    p = argparse.ArgumentParser(description="Compute canonical 7-axis alignment for linked speech-action pairs")
    p.add_argument("--speech-classifications", default="data/parquet/speech_classifications_with_rhetoric_full.parquet")
    p.add_argument("--motion-classifications", default="data/parquet/classifications.parquet")
    p.add_argument("--speech-action-links", default="data/parquet/speech_action_links.parquet")
    p.add_argument("--out", default="output/analysis/speech_action_axis_scores.parquet")
    args = p.parse_args()

    summary = compute_axis_alignment(
        speech_classifications_path=args.speech_classifications,
        motion_classifications_path=args.motion_classifications,
        speech_action_links_path=args.speech_action_links,
        out_path=args.out,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
