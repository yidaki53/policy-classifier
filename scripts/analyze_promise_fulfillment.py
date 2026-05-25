#!/usr/bin/env python3
"""Run promise-fulfillment analysis with time-aware controls."""

import argparse
import json

from swedish_parliament_policy_classifier.analysis.promise_fulfillment import run_promise_fulfillment_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run promise-fulfillment analysis")
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
    parser.add_argument("--out", default="output/analysis", help="Output directory")
    args = parser.parse_args()

    out = run_promise_fulfillment_analysis(
        db_path=args.db,
        speech_classifications_path=args.speech_classifications,
        speech_parquet_dir=args.speech_parquet_dir,
        out_dir=args.out,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
