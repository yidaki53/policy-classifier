#!/usr/bin/env python3
"""Create minimal stub PNG files for CI/test environments lacking real figures."""

import base64
from pathlib import Path

# Minimal 1x1 transparent PNG
PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
PNG_DATA = base64.b64decode(PNG_B64)

FIGURES = [
    "output/manuscript/figures/figure_consistency_vs_fulfillment.png",
    "output/manuscript/figures/figure_parliament_direction_over_time.png",
    "output/manuscript/figures/figure_consistency_fulfillment_vs_benchmark_party_year.png",
    "output/manuscript/figures/figure_modality_overlay_by_party.png",
    "figures/manuscript/pie_chart_categories.png",
    "figures/manuscript/party_motions_stacked.png",
    "figures/voting/party_cohesion_timeseries.png",
    "figures/three_way/divergence_speech_vs_combined_significance.png",
    "figures/speeches/speech_profiles_heatmap.png",
]


def main() -> None:
    repo = Path(".").resolve()
    for rel in FIGURES:
        out = repo / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(PNG_DATA)
        print(f"Created stub: {out}")


if __name__ == "__main__":
    main()