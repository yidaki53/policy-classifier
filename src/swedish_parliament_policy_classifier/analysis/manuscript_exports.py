from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swedish_parliament_policy_classifier.analysis.ideological_gap import build_modality_profiles, run_ideological_gap_analysis
from swedish_parliament_policy_classifier.analysis.promise_fulfillment import run_promise_fulfillment_analysis
from swedish_parliament_policy_classifier.analysis.speech_visualizations import IDEOLOGY_ORDER


def _markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _write_table(df: pd.DataFrame, out_dir: Path, stem: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pq_p = out_dir / f"{stem}.parquet"
    md_p = out_dir / f"{stem}.md"
    tex_p = out_dir / f"{stem}.tex"

    df.to_parquet(pq_p, index=False, compression="zstd")
    md_p.write_text(_markdown_table(df), encoding="utf-8")
    tex_p.write_text(df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}"), encoding="utf-8")

    return {"parquet": str(pq_p), "md": str(md_p), "tex": str(tex_p)}


def _pivot_modality_table(profiles: pd.DataFrame, modality: str) -> pd.DataFrame:
    df = profiles[profiles["modality"] == modality].copy()
    pivot = (
        df.pivot_table(index="party", columns="category", values="proportion", aggfunc="mean", fill_value=0.0)
        .reindex(columns=IDEOLOGY_ORDER, fill_value=0.0)
        .reset_index()
    )
    pivot.columns = [str(c) for c in pivot.columns]
    return pivot


def plot_modality_overlay_figure(
    profiles: pd.DataFrame,
    out_path: str | Path,
    max_cols: int = 3,
) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parties = sorted(profiles["party"].unique().tolist())
    if not parties:
        raise ValueError("No party profiles available for modality overlay figure.")

    n = len(parties)
    n_cols = min(max_cols, max(1, n))
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.0 * n_rows), squeeze=False, sharey=True)
    modalities = ["speech", "motion", "vote"]
    colors = {"speech": "#1f77b4", "motion": "#ff7f0e", "vote": "#2ca02c"}

    x = np.arange(len(IDEOLOGY_ORDER))
    for i, party in enumerate(parties):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        sub = profiles[profiles["party"] == party]
        for m in modalities:
            s = (
                sub[sub["modality"] == m]
                .set_index("category")["proportion"]
                .reindex(IDEOLOGY_ORDER, fill_value=0.0)
                .to_numpy(dtype=float)
            )
            ax.plot(x, s, marker="o", linewidth=1.8, markersize=4, label=m, color=colors[m], alpha=0.9)

        ax.set_title(party)
        ax.set_xticks(x)
        ax.set_xticklabels(IDEOLOGY_ORDER, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)

    for j in range(n, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r][c].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Party ideology profiles by modality (speech vs motion vs vote)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def generate_manuscript_tables_and_figure(
    db_path: str | Path,
    speech_classifications_path: str | Path,
    speech_parquet_dir: str | Path,
    tables_out_dir: str | Path = "output/manuscript/tables",
    figures_out_dir: str | Path = "output/manuscript/figures",
    analysis_cache_dir: str | Path = "output/analysis",
) -> dict:
    tables_out = Path(tables_out_dir)
    figures_out = Path(figures_out_dir)
    analysis_out = Path(analysis_cache_dir)

    profiles = build_modality_profiles(db_path, speech_classifications_path, speech_parquet_dir)

    motion_tbl = _pivot_modality_table(profiles, "motion")
    vote_tbl = _pivot_modality_table(profiles, "vote")
    speech_tbl = _pivot_modality_table(profiles, "speech")

    gap_payload = run_ideological_gap_analysis(db_path, speech_classifications_path, speech_parquet_dir, out_dir=analysis_out)
    gap_path = Path(gap_payload["output_party_parquet"])
    gap_df = pd.read_parquet(gap_path) if gap_path.exists() else pd.DataFrame()

    pf_payload = run_promise_fulfillment_analysis(db_path, speech_classifications_path, speech_parquet_dir, out_dir=analysis_out)
    pf_party_path = Path(pf_payload["outputs"]["party_summary"])
    pf_df = pd.read_parquet(pf_party_path) if pf_party_path.exists() else pd.DataFrame()

    outputs = {
        "motion_profile_table": _write_table(motion_tbl, tables_out, "table_motion_party_profiles"),
        "vote_profile_table": _write_table(vote_tbl, tables_out, "table_vote_party_profiles"),
        "speech_profile_table": _write_table(speech_tbl, tables_out, "table_speech_party_profiles"),
        "ideological_gap_table": _write_table(gap_df, tables_out, "table_ideological_gap_distances"),
        "promise_fulfillment_table": _write_table(
            pf_df,
            tables_out,
            "table_promise_fulfillment_party_summary",
        ),
    }

    overlay_path = figures_out / "figure_modality_overlay_by_party.png"
    outputs["combined_overlay_figure"] = plot_modality_overlay_figure(profiles, overlay_path)

    summary = {
        "tables": outputs,
        "analysis_cache_dir": str(analysis_out),
    }
    summary_path = tables_out / "manuscript_assets_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary
