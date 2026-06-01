#!/usr/bin/env python3
"""Interactive Plotly visualisation: stacked bars with hover and ordering (Parquet input)."""

import argparse
import os
from datetime import datetime, timezone

import plotly.graph_objects as go
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Generate interactive party stacked-bar chart (HTML)")
    parser.add_argument("--profiles", default="data/parquet/party_profiles_recency.parquet", help="Party profiles parquet (from scripts/build_profiles.py)")
    parser.add_argument("--modality", default="motion", help="Modality to visualise: motion, vote, or speech")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--file", default="party_profiles_interactive.html")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    p = args.profiles
    if not os.path.exists(p):
        print("Profiles parquet not found:", p)
        return

    df = pd.read_parquet(p)
    df = df[df["modality"] == args.modality].copy()
    if df.empty:
        print("No profiles for modality", args.modality)
        return

    pivot = df.pivot(index="party", columns="category", values="proportion").fillna(0.0)
    parties = sorted(pivot.index.tolist())
    categories = list(pivot.columns)

    year_min = None
    year_max = None
    if "year" in df.columns:
        years = pd.to_numeric(df["year"], errors="coerce").dropna()
        if not years.empty:
            year_min = int(years.min())
            year_max = int(years.max())

    fig = go.Figure()
    for cat in categories:
        fig.add_trace(
            go.Bar(name=cat, x=parties, y=[float(pivot.loc[p, cat]) if p in pivot.index and cat in pivot.columns else 0.0 for p in parties], hoverinfo="x+y+name")
        )

    period = f"{year_min}-{year_max}" if year_min is not None and year_max is not None else "n/a"
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fig.update_layout(
        barmode='stack',
        title=f'Party policy profile (interactive stacked bars, modality={args.modality}, period={period})',
        xaxis_title='Party',
        yaxis_title='Proportion (share, 0-1)',
        margin=dict(b=120),
        annotations=[
            dict(
                text=(
                    f"Author: Robin Oberg ({datetime.now(timezone.utc).year}) | "
                    f"Source: {args.profiles} | Period: {period} | Generated: {generated_utc}"
                ),
                xref="paper",
                yref="paper",
                x=1,
                y=-0.25,
                xanchor="right",
                yanchor="top",
                showarrow=False,
                font=dict(size=10, color="#555555"),
            )
        ],
    )
    out_path = os.path.join(args.out, args.file)
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f"Wrote interactive HTML to {out_path}")


if __name__ == "__main__":
    main()
