#!/usr/bin/env python3
"""Protocol-grade validation of internal ideology estimates against external benchmarks.

Supports:
- Party-only or party-time matching
- Explicit orientation handling (fixed or auto-selected)
- Rank-robustness diagnostics via bootstrap resampling
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _rankdata(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy(dtype=float)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    x0 = x - x.mean()
    y0 = y - y.mean()
    den = np.sqrt((x0**2).sum() * (y0**2).sum())
    if den == 0:
        return np.nan
    return float((x0 * y0).sum() / den)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported benchmark file extension: {path.suffix}")


def _normalize_time_key(series: pd.Series) -> pd.Series:
    # Canonicalize mixed year encodings (e.g., 2014, 2014.0, "2014") to stable string keys.
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(index=series.index, dtype=object)
    has_numeric = numeric.notna()
    out.loc[has_numeric] = numeric.loc[has_numeric].round().astype("int64").astype(str)
    out.loc[~has_numeric] = series.loc[~has_numeric].astype(str).str.strip()
    return out.astype(str)


def _bootstrap_stat(
    x: np.ndarray,
    y: np.ndarray,
    fn,
    n_boot: int,
    seed: int,
) -> tuple[float | None, float | None, int]:
    if x.size < 2 or n_boot <= 0:
        return None, None, 0
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    n = x.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = fn(x[idx], y[idx])
        if not np.isnan(v):
            vals.append(float(v))
    if not vals:
        return None, None, 0
    arr = np.asarray(vals, dtype=float)
    lo = float(np.quantile(arr, 0.025))
    hi = float(np.quantile(arr, 0.975))
    return lo, hi, int(arr.size)


def main() -> None:
    p = argparse.ArgumentParser(description="Validate internal party ideology estimates against external benchmarks")
    p.add_argument("--internal", default="output/analysis/consistency_score_party.parquet")
    p.add_argument("--internal-party-col", default="party")
    p.add_argument("--internal-score-col", default="ideology_uphold_v2")
    p.add_argument("--internal-time-col", default=None, help="Optional internal time column for party-time matching")
    p.add_argument("--external", default="output/analysis/external_party_benchmarks.csv")
    p.add_argument("--external-party-col", default="party")
    p.add_argument("--external-score-col", default="benchmark_ideology_score")
    p.add_argument("--external-time-col", default=None, help="Optional external time column for party-time matching")
    p.add_argument(
        "--orientation",
        choices=["auto", "positive", "negative"],
        default="auto",
        help="Orientation protocol: keep internal sign, flip internal sign, or auto-select by |Spearman|",
    )
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations for rank robustness CI")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-merged", default="output/analysis/party_ideology_benchmark_merged.parquet")
    p.add_argument("--summary-out", default="output/analysis/party_ideology_benchmark_validation.json")
    p.add_argument("--write-template", action="store_true")
    p.add_argument("--template-out", default="docs/external_party_benchmarks_template.csv")
    args = p.parse_args()

    internal_path = Path(args.internal)
    if not internal_path.exists():
        raise FileNotFoundError(f"Missing internal table: {internal_path}")

    internal = _read_table(internal_path)
    if args.internal_party_col not in internal.columns or args.internal_score_col not in internal.columns:
        raise ValueError(
            f"Internal columns missing. Expected `{args.internal_party_col}` and `{args.internal_score_col}` in {internal_path}"
        )

    if args.internal_time_col and args.internal_time_col not in internal.columns:
        raise ValueError(f"Internal time column missing: {args.internal_time_col}")

    internal_cols = [args.internal_party_col, args.internal_score_col]
    if args.internal_time_col:
        internal_cols.append(args.internal_time_col)
    internal_sub = internal[internal_cols].copy()
    rename_map = {
        args.internal_party_col: "party",
        args.internal_score_col: "internal_score",
    }
    if args.internal_time_col:
        rename_map[args.internal_time_col] = "time_key"
    internal_sub = internal_sub.rename(columns=rename_map)
    internal_sub["party"] = internal_sub["party"].astype(str).str.strip()
    if "time_key" in internal_sub.columns:
        internal_sub["time_key"] = _normalize_time_key(internal_sub["time_key"])

    external_path = Path(args.external)
    if not external_path.exists():
        if args.write_template:
            tmpl_cols = ["party"] + (["time_key"] if "time_key" in internal_sub.columns else [])
            tmpl = internal_sub[tmpl_cols].drop_duplicates().sort_values(tmpl_cols).copy()
            tmpl["benchmark_ideology_score"] = np.nan
            if "time_key" in tmpl.columns:
                tmpl = tmpl.rename(columns={"time_key": args.external_time_col or "year"})
            out_tmpl = Path(args.template_out)
            out_tmpl.parent.mkdir(parents=True, exist_ok=True)
            tmpl.to_csv(out_tmpl, index=False)
            print(
                json.dumps(
                    {
                        "status": "template_written",
                        "message": "External benchmark file missing; template created.",
                        "template": str(out_tmpl),
                    },
                    indent=2,
                )
            )
            return
        raise FileNotFoundError(
            f"Missing external benchmark file: {external_path}. Use --write-template to scaffold one."
        )

    external = _read_table(external_path)
    if args.external_party_col not in external.columns or args.external_score_col not in external.columns:
        raise ValueError(
            f"External columns missing. Expected `{args.external_party_col}` and `{args.external_score_col}` in {external_path}"
        )
    if args.external_time_col and args.external_time_col not in external.columns:
        raise ValueError(f"External time column missing: {args.external_time_col}")

    if bool(args.internal_time_col) != bool(args.external_time_col):
        raise ValueError("Use both --internal-time-col and --external-time-col for party-time matching, or neither.")

    external_cols = [args.external_party_col, args.external_score_col]
    if args.external_time_col:
        external_cols.append(args.external_time_col)
    external_sub = external[external_cols].copy()
    rename_map = {
        args.external_party_col: "party",
        args.external_score_col: "external_score",
    }
    if args.external_time_col:
        rename_map[args.external_time_col] = "time_key"
    external_sub = external_sub.rename(columns=rename_map)
    external_sub["party"] = external_sub["party"].astype(str).str.strip()
    if "time_key" in external_sub.columns:
        external_sub["time_key"] = _normalize_time_key(external_sub["time_key"])

    merge_keys = ["party", "time_key"] if args.internal_time_col and args.external_time_col else ["party"]
    merged = internal_sub.merge(external_sub, on=merge_keys, how="inner")
    merged = merged.dropna(subset=["internal_score", "external_score"]).copy()

    x_raw = merged["internal_score"].to_numpy(dtype=float)
    y = merged["external_score"].to_numpy(dtype=float)

    x_pos = x_raw.copy()
    x_neg = -x_raw
    spearman_pos = _spearman(x_pos, y)
    spearman_neg = _spearman(x_neg, y)

    if args.orientation == "positive":
        selected_orientation = "positive"
        x = x_pos
    elif args.orientation == "negative":
        selected_orientation = "negative"
        x = x_neg
    else:
        if np.isnan(spearman_pos) and np.isnan(spearman_neg):
            selected_orientation = "positive"
            x = x_pos
        elif np.isnan(spearman_neg) or (not np.isnan(spearman_pos) and abs(spearman_pos) >= abs(spearman_neg)):
            selected_orientation = "positive"
            x = x_pos
        else:
            selected_orientation = "negative"
            x = x_neg

    pearson = _pearson(x, y)
    spearman = _spearman(x, y)
    pearson_ci = _bootstrap_stat(x, y, _pearson, n_boot=args.bootstrap, seed=args.seed)
    spearman_ci = _bootstrap_stat(x, y, _spearman, n_boot=args.bootstrap, seed=args.seed + 1)

    merged["internal_score_oriented"] = x
    merged["orientation"] = selected_orientation

    out_merged = Path(args.out_merged)
    out_merged.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_merged, index=False, compression="zstd")

    summary = {
        "internal": str(internal_path),
        "external": str(external_path),
        "matching_keys": merge_keys,
        "internal_score_col": args.internal_score_col,
        "external_score_col": args.external_score_col,
        "requested_orientation": args.orientation,
        "selected_orientation": selected_orientation,
        "spearman_if_positive": float(spearman_pos) if not np.isnan(spearman_pos) else None,
        "spearman_if_negative": float(spearman_neg) if not np.isnan(spearman_neg) else None,
        "n_parties_overlap": int(len(merged)),
        "pearson": float(pearson) if not np.isnan(pearson) else None,
        "spearman": float(spearman) if not np.isnan(spearman) else None,
        "pearson_bootstrap_ci95": {
            "low": pearson_ci[0],
            "high": pearson_ci[1],
            "n_valid_bootstrap": pearson_ci[2],
        },
        "spearman_bootstrap_ci95": {
            "low": spearman_ci[0],
            "high": spearman_ci[1],
            "n_valid_bootstrap": spearman_ci[2],
        },
        "bootstrap_iterations_requested": int(args.bootstrap),
        "seed": int(args.seed),
        "merged_output": str(out_merged),
    }

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
