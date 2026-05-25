#!/usr/bin/env python3
"""Generate manuscript-ready tables and publication figure for five analyses."""

import argparse
import json

from swedish_parliament_policy_classifier.analysis.manuscript_exports import generate_manuscript_tables_and_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript-ready tables and figures")
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
    parser.add_argument("--tables-out", default="output/manuscript/tables", help="Output directory for tables")
    parser.add_argument("--figures-out", default="output/manuscript/figures", help="Output directory for figures")
    parser.add_argument("--analysis-out", default="output/analysis", help="Working analysis output directory")
    args = parser.parse_args()

    out = generate_manuscript_tables_and_figure(
        db_path=args.db,
        speech_classifications_path=args.speech_classifications,
        speech_parquet_dir=args.speech_parquet_dir,
        tables_out_dir=args.tables_out,
        figures_out_dir=args.figures_out,
        analysis_cache_dir=args.analysis_out,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
