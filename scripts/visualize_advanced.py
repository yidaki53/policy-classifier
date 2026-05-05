#!/usr/bin/env python3
"""Generate advanced visualisations (clustered heatmap + PCA biplot)."""

import argparse

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.visualization.advanced import plot_clustered_heatmap, plot_pca_biplot


def main():
    parser = argparse.ArgumentParser(description="Generate advanced party visualizations")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    conn = init_db(args.db)
    a = plot_clustered_heatmap(conn, out_dir=args.out)
    b = plot_pca_biplot(conn, out_dir=args.out)
    print(f"Saved clustered heatmap: {a}")
    print(f"Saved PCA biplot: {b}")


if __name__ == "__main__":
    main()
