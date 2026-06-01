#!/usr/bin/env python3
"""Fit a simple latent party-ideology factor from multimodal indicators.

Outputs party-level latent scores with bootstrap uncertainty intervals.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _zscore(col: pd.Series) -> pd.Series:
    x = pd.to_numeric(col, errors="coerce")
    mu = x.mean()
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - mu) / sd


def _axis_to_index(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip()
    if not text:
        return np.nan

    parsed = None
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None

    if isinstance(parsed, list) and len(parsed) > 0:
        arr = np.array(parsed, dtype=float)
        den = float(arr.sum())
        if den <= 0:
            return np.nan
        idx = np.arange(len(arr), dtype=float)
        return float((idx * arr).sum() / den)

    return np.nan


def _factor_scores(ind: pd.DataFrame, anchor_col: str) -> pd.Series:
    numeric = ind.apply(pd.to_numeric, errors="coerce")
    row_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    if row_mask.sum() < 2:
        return pd.Series(np.nan, index=ind.index, dtype=float)

    ind_ok = numeric.loc[row_mask].copy()
    x = ind_ok.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    loading = vt[0, :]
    scores = x @ loading

    # Orientation: higher latent should align with higher anchor value.
    anchor = ind_ok[anchor_col].to_numpy(dtype=float)
    if np.corrcoef(scores, anchor)[0, 1] < 0:
        scores = -scores

    s_ok = _zscore(pd.Series(scores, index=ind_ok.index, dtype=float))
    s = pd.Series(np.nan, index=ind.index, dtype=float)
    s.loc[s_ok.index] = s_ok.to_numpy(dtype=float)
    return s


def _bootstrap_scores(
    merged: pd.DataFrame,
    parties: list[str],
    base_scores: pd.Series,
    ideology_map: dict[str, float],
    n_boot: int,
    seed: int,
    min_party_n: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[float]] = {p: [] for p in parties}

    groups = {p: g for p, g in merged.groupby("speech_party", dropna=False)}

    for _ in range(n_boot):
        rows = []
        for p in parties:
            g = groups.get(p)
            if g is None or len(g) < min_party_n:
                continue
            idx = rng.integers(0, len(g), size=len(g))
            samp = g.iloc[idx]
            rows.append(
                {
                    "party": p,
                    "speech_axis_mean": float(pd.to_numeric(samp["speech_axis"], errors="coerce").mean()),
                    "action_axis_mean": float(pd.to_numeric(samp["action_axis"], errors="coerce").mean()),
                }
            )

        b = pd.DataFrame(rows)
        if b.empty:
            continue

        b["speech_axis_mean_z"] = _zscore(b["speech_axis_mean"]).to_numpy(dtype=float)
        b["action_axis_mean_z"] = _zscore(b["action_axis_mean"]).to_numpy(dtype=float)
        b["ideology_uphold_v2"] = b["party"].astype(str).map(ideology_map)
        b["ideology_uphold_v2_z"] = _zscore(b["ideology_uphold_v2"]).to_numpy(dtype=float)

        ind_cols = ["speech_axis_mean_z", "action_axis_mean_z", "ideology_uphold_v2_z"]
        ind = b.set_index(b["party"].astype(str))[ind_cols]
        keep_cols = [c for c in ind.columns if float(np.nanstd(ind[c].to_numpy(dtype=float))) > 0]
        ind = ind[keep_cols]
        if len(keep_cols) < 2:
            continue

        anchor = "action_axis_mean_z" if "action_axis_mean_z" in ind.columns else keep_cols[0]
        scores = _factor_scores(ind, anchor_col=anchor)

        common = sorted(set(scores.index).intersection(set(base_scores.index)))
        if len(common) >= 2:
            s_boot = pd.to_numeric(scores.loc[common], errors="coerce")
            s_base = pd.to_numeric(base_scores.loc[common], errors="coerce")
            m = s_boot.notna() & s_base.notna()
            if int(m.sum()) >= 2:
                corr = np.corrcoef(s_boot[m].to_numpy(dtype=float), s_base[m].to_numpy(dtype=float))[0, 1]
                if np.isfinite(corr) and corr < 0:
                    scores = -scores

        for p, v in scores.items():
            out[str(p)].append(float(v))

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fit latent party ideology factor with bootstrap intervals")
    p.add_argument("--axis-scores", default="output/analysis/speech_action_axis_scores.parquet")
    p.add_argument("--links", default="data/parquet/speech_action_links.parquet")
    p.add_argument("--consistency", default="output/analysis/consistency_score_party.parquet")
    p.add_argument("--out", default="output/analysis/party_latent_ideology_estimates.parquet")
    p.add_argument("--summary-out", default="output/analysis/party_latent_ideology_summary.json")
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-party-n", type=int, default=200)
    args = p.parse_args()

    axis = pd.read_parquet(Path(args.axis_scores), columns=["speech_id", "motion_id", "speech_axis", "action_axis"])
    links = pd.read_parquet(Path(args.links), columns=["speech_id", "motion_id", "speech_party"])
    consistency = pd.read_parquet(Path(args.consistency), columns=["party", "ideology_uphold_v2"])

    merged = axis.merge(links, on=["speech_id", "motion_id"], how="inner")
    merged["speech_party"] = merged["speech_party"].fillna("Unknown").astype(str)
    merged["speech_axis"] = merged["speech_axis"].map(_axis_to_index)
    merged["action_axis"] = merged["action_axis"].map(_axis_to_index)
    merged = merged.dropna(subset=["speech_axis", "action_axis"]).copy()

    party = (
        merged.groupby("speech_party", as_index=False)
        .agg(
            n=("speech_id", "size"),
            speech_axis_mean=("speech_axis", "mean"),
            action_axis_mean=("action_axis", "mean"),
        )
        .rename(columns={"speech_party": "party"})
    )
    party = party[party["n"] >= int(args.min_party_n)].copy()
    party = party.merge(consistency, on="party", how="left")

    party["speech_axis_mean_z"] = _zscore(party["speech_axis_mean"]).to_numpy(dtype=float)
    party["action_axis_mean_z"] = _zscore(party["action_axis_mean"]).to_numpy(dtype=float)
    party["ideology_uphold_v2_z"] = _zscore(party["ideology_uphold_v2"]).to_numpy(dtype=float)
    ind = party.set_index(party["party"].astype(str))[["speech_axis_mean_z", "action_axis_mean_z", "ideology_uphold_v2_z"]]

    # Keep indicators with variance after z-scoring.
    keep_cols = [c for c in ind.columns if float(np.nanstd(ind[c].to_numpy(dtype=float))) > 0]
    if len(keep_cols) < 2:
        raise RuntimeError("Need at least two informative indicators to fit latent ideology factor")
    ind = ind[keep_cols]

    scores = _factor_scores(ind, anchor_col="action_axis_mean_z" if "action_axis_mean_z" in ind.columns else keep_cols[0])
    out = party.copy()
    out["latent_ideology_score"] = out["party"].map(scores).astype(float)

    boot = _bootstrap_scores(
        merged=merged,
        parties=out["party"].astype(str).tolist(),
        base_scores=scores,
        ideology_map={str(r.party): float(r.ideology_uphold_v2) if pd.notna(r.ideology_uphold_v2) else np.nan for r in party.itertuples(index=False)},
        n_boot=int(args.n_boot),
        seed=int(args.seed),
        min_party_n=int(args.min_party_n),
    )

    lo_q = float(args.alpha / 2.0)
    hi_q = float(1.0 - args.alpha / 2.0)
    ci_lo = []
    ci_hi = []
    n_boot_eff = []
    for pty in out["party"].astype(str):
        vals = np.array(boot.get(pty, []), dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            ci_lo.append(np.nan)
            ci_hi.append(np.nan)
            n_boot_eff.append(0)
        else:
            ci_lo.append(float(np.quantile(vals, lo_q)))
            ci_hi.append(float(np.quantile(vals, hi_q)))
            n_boot_eff.append(int(vals.size))

    out["latent_ideology_ci_low"] = ci_lo
    out["latent_ideology_ci_high"] = ci_hi
    out["latent_ideology_bootstrap_n"] = n_boot_eff
    out = out.sort_values("latent_ideology_score", ascending=False).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, compression="zstd")

    summary = {
        "axis_scores": str(Path(args.axis_scores)),
        "links": str(Path(args.links)),
        "consistency": str(Path(args.consistency)),
        "n_parties": int(len(out)),
        "indicator_columns": keep_cols,
        "n_boot": int(args.n_boot),
        "alpha": float(args.alpha),
        "output": str(out_path),
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
