from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _source_confidence(link_source: str) -> float:
    s = str(link_source or "")
    if s.startswith("existing:betankande_ref_dok_id"):
        return 0.98
    if s.startswith("existing:rel_dok_id"):
        return 0.95
    if s.startswith("existing:"):
        return 0.9
    if s.startswith("fallback:party_category_vote"):
        return 0.8
    if s.startswith("fallback:party_category"):
        return 0.75
    if s.startswith("fallback:party_vote"):
        return 0.7
    if s.startswith("fallback:party"):
        return 0.65
    if s.startswith("fallback:category_vote"):
        return 0.6
    if s.startswith("fallback:category"):
        return 0.55
    if s.startswith("fallback:vote_any"):
        return 0.45
    return 0.35


def score_contradiction_edges(
    speech_action_links_path: str | Path,
    axis_scores_path: str | Path,
    out_path: str | Path,
) -> dict:
    links = pd.read_parquet(speech_action_links_path)
    axis = pd.read_parquet(axis_scores_path)

    merged = links.merge(
        axis[["speech_id", "motion_id", "axis_js_distance", "axis_peak_flip"]],
        on=["speech_id", "motion_id"],
        how="left",
    )

    merged["axis_js_distance"] = pd.to_numeric(merged["axis_js_distance"], errors="coerce").fillna(0.0)
    merged["axis_peak_flip"] = merged["axis_peak_flip"].fillna(False).astype(bool)
    merged["edge_confidence_raw"] = merged["link_source"].map(_source_confidence).fillna(0.35)

    # Placeholder NLI-style probabilities for v1 until a dedicated Swedish
    # parliamentary NLI model is integrated.
    merged["nli_entail_prob"] = (1.0 - merged["axis_js_distance"]).clip(0.0, 1.0)
    merged["nli_contradict_prob"] = (
        0.7 * merged["axis_js_distance"] + 0.3 * merged["axis_peak_flip"].astype(float)
    ).clip(0.0, 1.0)
    merged["nli_neutral_prob"] = (1.0 - merged["nli_entail_prob"] - merged["nli_contradict_prob"]).clip(0.0, 1.0)
    merged["rule_conflict_flag"] = merged["axis_peak_flip"].astype(bool)

    merged["contradiction_score_raw"] = (
        0.6 * merged["nli_contradict_prob"]
        + 0.3 * merged["axis_js_distance"]
        + 0.1 * merged["rule_conflict_flag"].astype(float)
    ).clip(0.0, 1.0)

    out_cols = [
        "speech_id",
        "motion_id",
        "action_id",
        "action_type",
        "speech_party",
        "category",
        "speech_date",
        "link_source",
        "nli_entail_prob",
        "nli_neutral_prob",
        "nli_contradict_prob",
        "rule_conflict_flag",
        "axis_js_distance",
        "axis_peak_flip",
        "contradiction_score_raw",
        "edge_confidence_raw",
    ]
    out = merged[out_cols].copy()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, compression="zstd")

    summary = {
        "output": str(out_path),
        "rows": int(len(out)),
        "mean_contradiction_score_raw": float(out["contradiction_score_raw"].mean()) if len(out) else None,
        "mean_edge_confidence_raw": float(out["edge_confidence_raw"].mean()) if len(out) else None,
    }
    return summary


def aggregate_expected_contradiction(
    edge_scores_path: str | Path,
    out_path: str | Path,
) -> dict:
    df = pd.read_parquet(edge_scores_path).copy()
    df["speech_date"] = pd.to_datetime(df["speech_date"], errors="coerce", utc=True)
    df["year"] = df["speech_date"].dt.year
    df["party"] = df["speech_party"].astype(str)
    df["topic"] = df["category"].astype(str)

    df["weighted_contradiction"] = df["edge_confidence_raw"] * df["contradiction_score_raw"]

    g = (
        df.groupby(["party", "topic", "year"], as_index=False)
        .agg(
            n_candidate_edges=("speech_id", "size"),
            n_speeches=("speech_id", "nunique"),
            mean_edge_confidence=("edge_confidence_raw", "mean"),
            weighted_contradiction_sum=("weighted_contradiction", "sum"),
            edge_conf_sum=("edge_confidence_raw", "sum"),
        )
    )

    g["expected_contradiction"] = 0.0
    mask = g["edge_conf_sum"] > 0
    g.loc[mask, "expected_contradiction"] = g.loc[mask, "weighted_contradiction_sum"] / g.loc[mask, "edge_conf_sum"]
    g["expected_contradiction"] = g["expected_contradiction"].clip(0.0, 1.0)
    g["expected_uphold"] = 1.0 - g["expected_contradiction"]
    g["unlinked_rate_post_retrieval"] = 0.0

    out = g[
        [
            "party",
            "topic",
            "year",
            "n_speeches",
            "n_candidate_edges",
            "expected_contradiction",
            "expected_uphold",
            "mean_edge_confidence",
            "unlinked_rate_post_retrieval",
        ]
    ].copy()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, compression="zstd")

    return {
        "output": str(out_path),
        "rows": int(len(out)),
        "mean_expected_contradiction": float(out["expected_contradiction"].mean()) if len(out) else None,
    }
