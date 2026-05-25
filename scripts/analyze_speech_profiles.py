#!/usr/bin/env python3
"""Generate speech-side ideology visualizations from speech classifications."""

import argparse
import json

from swedish_parliament_policy_classifier.analysis.speech_visualizations import plot_speech_profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate speech ideology profile visualizations")
    parser.add_argument(
        "--speech-classifications",
        default="data/parquet/speech_classifications_with_rhetoric_full.parquet",
        help="Path to speech classification parquet",
    )
    parser.add_argument(
        "--speech-parquet-dir",
        default="data/speeches/parquet",
        help="Directory with raw speech parquet files",
    )
    parser.add_argument("--out", default="figures/speeches", help="Output directory")
    parser.add_argument("--basename", default="speech_profiles", help="Output file basename")
    args = parser.parse_args()

    out = plot_speech_profiles(
        speech_classifications_path=args.speech_classifications,
        speech_parquet_dir=args.speech_parquet_dir,
        out_dir=args.out,
        basename=args.basename,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
