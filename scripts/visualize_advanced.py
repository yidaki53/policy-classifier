#!/usr/bin/env python3
"""Generate advanced visualisations (clustered heatmap + PCA biplot).

This script prefers Parquet-based party profiles. Use `--profiles` to point
at a `party_profiles` Parquet table (default
`data/parquet/party_profiles_recency.parquet`).
"""

import argparse

from swedish_parliament_policy_classifier.visualization.advanced import plot_clustered_heatmap, plot_pca_biplot


def main():
    parser = argparse.ArgumentParser(description="Generate advanced party visualizations")
    parser.add_argument("--profiles", default="data/parquet/party_profiles_recency.parquet", help="Parquet table with party profiles")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    a = plot_clustered_heatmap(args.profiles, out_dir=args.out)
    b = plot_pca_biplot(args.profiles, out_dir=args.out)
    print(f"Saved clustered heatmap: {a}")
    print(f"Saved PCA biplot: {b}")


if __name__ == "__main__":
    main()
