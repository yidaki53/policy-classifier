#!/usr/bin/env python3
"""Small CLI to generate the final combined visualization.

This script now prefers Parquet-based party profiles instead of the SQLite DB.
Provide `--profiles` to point at a `party_profiles` Parquet table (default
`data/parquet/party_profiles_recency.parquet`).
"""

import argparse

from swedish_parliament_policy_classifier.visualization.plot_final_profiles import plot_final_profiles


def main():
    parser = argparse.ArgumentParser(description="Generate final party visualizations")
    parser.add_argument("--profiles", default="data/parquet/party_profiles_recency.parquet", help="Parquet table with party profiles")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    out = plot_final_profiles(args.profiles, out_dir=args.out)
    print(f"Saved final visualization: {out}")


if __name__ == "__main__":
    main()
