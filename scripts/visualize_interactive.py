#!/usr/bin/env python3
"""Interactive Plotly visualisation: stacked bars with hover and ordering."""

import argparse
import os

import plotly.graph_objects as go

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.analysis.aggregate import compute_party_profiles


def main():
    parser = argparse.ArgumentParser(description="Generate interactive party stacked-bar chart (HTML)")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--file", default="party_profiles_interactive.html")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    conn = init_db(args.db)
    profiles = compute_party_profiles(conn)
    if not profiles:
        print("No profiles available")
        return

    categories = list(next(iter(profiles.values())).keys())
    parties = sorted(profiles.keys())

    fig = go.Figure()
    for cat in categories:
        fig.add_trace(
            go.Bar(name=cat, x=parties, y=[profiles[p].get(cat, 0.0) for p in parties], hoverinfo="x+y+name")
        )

    fig.update_layout(barmode='stack', title='Party policy profile (interactive stacked bars)', xaxis_title='Party', yaxis_title='Proportion')
    out_path = os.path.join(args.out, args.file)
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f"Wrote interactive HTML to {out_path}")


if __name__ == "__main__":
    main()
