#!/usr/bin/env python3
"""Small CLI to generate the final combined visualization."""

import argparse

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.visualization.plot_final_profiles import plot_final_profiles


def main():
    parser = argparse.ArgumentParser(description="Generate final party visualizations")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    conn = init_db(args.db)
    out = plot_final_profiles(conn, out_dir=args.out)
    print(f"Saved final visualization: {out}")


if __name__ == "__main__":
    main()
