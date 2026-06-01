#!/usr/bin/env python3
"""Derive link-confidence strata from speech-action linkage provenance.

This script maps `link_source` values in `speech_action_links.parquet` to a
small set of auditable confidence strata and writes both row-level and summary
outputs for downstream robustness reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _classify_link_source(source: str) -> tuple[str, float, bool]:
    s = str(source or "").strip()

    if s in {"existing:rel_dok_id", "existing:betankande_ref_dok_id"}:
        return "structural_high", 0.95, True

    if s.startswith("existing:"):
        return "existing_unknown", 0.70, False

    if s in {"graph_direct_signatory_motion", "graph_direct_signatory_vote"}:
        return "graph_signatory", 0.85, True

    if s in {"graph_direct_motion", "graph_direct_vote"}:
        return "graph_direct", 0.65, False

    if s == "graph_rebalanced_motion":
        return "heuristic_graph", 0.50, False

    if s.startswith("fallback_after_stale_existing:"):
        return "heuristic_fallback", 0.40, False

    if s.startswith("fallback:"):
        return "heuristic_fallback", 0.35, False

    return "unknown", 0.20, False


def _share_table(df: pd.DataFrame, group_col: str) -> list[dict]:
    counts = df[group_col].value_counts(dropna=False).rename_axis(group_col).reset_index(name="n")
    total = int(len(df))
    counts["share"] = counts["n"] / total if total > 0 else 0.0
    return counts.to_dict(orient="records")


def main() -> None:
    p = argparse.ArgumentParser(description="Compute link-confidence strata from speech_action_links")
    p.add_argument("--links", default="data/parquet/speech_action_links.parquet")
    p.add_argument("--out", default="output/analysis/speech_action_link_confidence_strata.parquet")
    p.add_argument("--summary-out", default="output/analysis/speech_action_link_confidence_summary.json")
    args = p.parse_args()

    links_path = Path(args.links)
    if not links_path.exists():
        raise FileNotFoundError(f"Missing links parquet: {links_path}")

    df = pd.read_parquet(links_path)
    required = {"speech_id", "action_id", "action_type", "speech_party", "link_source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in links parquet: {sorted(missing)}")

    cls = df["link_source"].apply(_classify_link_source)
    df = df.copy()
    df["link_confidence_stratum"] = cls.apply(lambda x: x[0])
    df["link_confidence_score"] = cls.apply(lambda x: x[1]).astype(float)
    df["has_structural_evidence"] = cls.apply(lambda x: x[2]).astype(bool)

    keep_cols = [
        "speech_id",
        "action_id",
        "action_type",
        "speech_party",
        "link_source",
        "link_confidence_stratum",
        "link_confidence_score",
        "has_structural_evidence",
    ]
    if "motion_id" in df.columns:
        keep_cols.insert(1, "motion_id")

    out_df = df[keep_cols].copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False, compression="zstd")

    party_stratum = (
        out_df.groupby(["speech_party", "link_confidence_stratum"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    party_totals = out_df.groupby("speech_party", dropna=False).size().rename("party_n").reset_index()
    party_stratum = party_stratum.merge(party_totals, on="speech_party", how="left")
    party_stratum["share_within_party"] = party_stratum["n"] / party_stratum["party_n"]

    summary = {
        "input": str(links_path),
        "output": str(out_path),
        "n_rows": int(len(out_df)),
        "stratum_counts": _share_table(out_df, "link_confidence_stratum"),
        "source_counts": _share_table(out_df, "link_source"),
        "action_type_counts": _share_table(out_df, "action_type"),
        "party_stratum_counts": party_stratum.sort_values(["speech_party", "n"], ascending=[True, False]).to_dict(
            orient="records"
        ),
    }

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({"rows": len(out_df), "out": str(out_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
