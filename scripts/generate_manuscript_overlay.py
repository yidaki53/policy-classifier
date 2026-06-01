#!/usr/bin/env python3
"""Generate only the manuscript modality-overlay figure from profiles parquet.

This is a lightweight alternative to full manuscript asset generation when the
goal is visualization refresh only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from swedish_parliament_policy_classifier.analysis.manuscript_exports import plot_modality_overlay_figure


def main() -> None:
    p = argparse.ArgumentParser(description="Generate manuscript modality overlay figure")
    p.add_argument("--profiles", default="data/parquet/party_profiles_recency.parquet")
    p.add_argument("--out", default="output/manuscript/figures/figure_modality_overlay_by_party.png")
    args = p.parse_args()

    profiles = pd.read_parquet(args.profiles)
    out_path = plot_modality_overlay_figure(profiles, Path(args.out))
    print(json.dumps({"overlay_figure": out_path}, indent=2))


if __name__ == "__main__":
    main()
