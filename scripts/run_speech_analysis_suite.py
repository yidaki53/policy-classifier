#!/usr/bin/env python3
"""Run all new speech-side analysis modules as a bundle."""

import argparse
import json

from swedish_parliament_policy_classifier.analysis.ideological_gap import run_ideological_gap_analysis
from swedish_parliament_policy_classifier.analysis.promise_fulfillment import run_promise_fulfillment_analysis
from swedish_parliament_policy_classifier.analysis.speech_visualizations import plot_speech_profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Run speech analysis suite")
    parser.add_argument("--db", default="data/swedish_parliament.db", help="Path to SQLite database")
    parser.add_argument(
        "--speech-classifications",
        default="data/parquet/speech_classifications_with_rhetoric_full.parquet",
        help="Path to speech classification parquet",
    )
    parser.add_argument(
        "--speech-parquet-dir",
        default="data/speeches/parquet",
        help="Directory with speech parquet files",
    )
    parser.add_argument("--fig-out", default="figures/speeches", help="Speech figure output directory")
    parser.add_argument("--analysis-out", default="output/analysis", help="Analysis output directory")
    args = parser.parse_args()

    speech = plot_speech_profiles(
        speech_classifications_path=args.speech_classifications,
        speech_parquet_dir=args.speech_parquet_dir,
        out_dir=args.fig_out,
    )
    gap = run_ideological_gap_analysis(
        db_path=args.db,
        speech_classifications_path=args.speech_classifications,
        speech_parquet_dir=args.speech_parquet_dir,
        out_dir=args.analysis_out,
    )
    promise = run_promise_fulfillment_analysis(
        db_path=args.db,
        speech_classifications_path=args.speech_classifications,
        speech_parquet_dir=args.speech_parquet_dir,
        out_dir=args.analysis_out,
    )

    print(json.dumps({"speech": speech, "ideological_gap": gap, "promise_fulfillment": promise}, indent=2))


if __name__ == "__main__":
    main()
