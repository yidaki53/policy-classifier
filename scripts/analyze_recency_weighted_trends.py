#!/usr/bin/env python3
"""Recency-weighted party and parliament ideology trends.

Inputs:
- output/analysis/promise_fulfillment_party_topic_year.parquet

Outputs:
- output/analysis/recency_weighted_party_scores.parquet
- output/analysis/recency_weighted_parliament_timeseries.parquet
- output/analysis/recency_weighted_summary.json

Optional SARIMAX outputs:
- output/analysis/sarimax_monthly_series.parquet
- output/analysis/sarimax_hyperparam_trials.parquet
- output/analysis/sarimax_best_models.parquet
- output/analysis/sarimax_fitted_series.parquet
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from swedish_parliament_policy_classifier.analysis.speech_visualizations import load_speech_metadata
from swedish_parliament_policy_classifier.runtime.experiment import ExperimentRun
from swedish_parliament_policy_classifier.runtime.resources import apply_cpu_throttle, thermal_safe_defaults


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    m = (~np.isnan(x)) & (~np.isnan(w))
    if m.sum() == 0:
        return float("nan")
    xs = x[m]
    ws = w[m]
    s = ws.sum()
    if s <= 0:
        return float("nan")
    return float(np.sum(xs * ws) / s)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _parse_int_grid(spec: str) -> list[int]:
    vals = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return sorted(set(vals))


def _parse_modes(spec: str) -> list[str]:
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out


IDEOLOGY_ORDER = [
    "far_left",
    "left",
    "centre_left",
    "centre",
    "centre_right",
    "right",
    "far_right",
]


def _election_years_between(min_year: int, max_year: int, cadence_years: int = 4, anchor_year: int = 2010) -> list[int]:
    years: list[int] = []
    if cadence_years <= 0:
        return years

    y = anchor_year
    while y > min_year:
        y -= cadence_years
    while y <= max_year:
        if y >= min_year:
            years.append(int(y))
        y += cadence_years
    return years


def _is_election_runup_year(year: object, election_years: list[int], runup_years: int = 1) -> bool:
    try:
        y = int(year)
    except Exception:
        return False
    return any((election_year - runup_years) <= y <= election_year for election_year in election_years)


def _election_dates_between(
    min_month: pd.Timestamp,
    max_month: pd.Timestamp,
    cadence_years: int = 4,
    anchor_year: int = 2010,
    anchor_month: int = 9,
) -> list[pd.Timestamp]:
    years = _election_years_between(
        min_year=int(min_month.year),
        max_year=int(max_month.year),
        cadence_years=int(cadence_years),
        anchor_year=int(anchor_year),
    )
    return [pd.Timestamp(year=y, month=int(anchor_month), day=1) for y in years]


def _month_distance(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return abs((a.year - b.year) * 12 + (a.month - b.month))


def _runup_indicator(months: pd.Series, election_dates: list[pd.Timestamp], runup_months: int) -> pd.Series:
    flags = []
    for m in months:
        if pd.isna(m):
            flags.append(False)
            continue
        hit = False
        for e in election_dates:
            if m <= e and _month_distance(m, e) <= int(runup_months):
                hit = True
                break
        flags.append(hit)
    return pd.Series(flags, index=months.index)


def _motion_dates_from_normalized(normalized_motions_path: str | Path) -> pd.DataFrame:
    nm = pd.read_parquet(normalized_motions_path).copy()
    if "id" not in nm.columns:
        raise ValueError("normalized_motions parquet missing 'id' column")

    nm["motion_id"] = nm["id"].astype(str)
    if "date" in nm.columns:
        nm["motion_date"] = pd.to_datetime(nm["date"], errors="coerce")
    else:
        nm["motion_date"] = pd.NaT

    if nm["motion_date"].isna().all() and "metadata" in nm.columns:
        def _extract_md_date(s: object) -> str | None:
            if s is None or (isinstance(s, float) and pd.isna(s)):
                return None
            try:
                md = json.loads(s) if isinstance(s, str) else s
            except Exception:
                return None
            if isinstance(md, dict):
                for k in ("systemdatum", "datum", "date"):
                    v = md.get(k)
                    if v:
                        return str(v)
            return None

        nm["motion_date"] = pd.to_datetime(nm["metadata"].map(_extract_md_date), errors="coerce")

    return nm[["motion_id", "motion_date"]].dropna(subset=["motion_date"]).copy()


def _build_monthly_series(
    speech_classifications_path: str | Path,
    speech_parquet_dir: str | Path,
    motion_classifications_path: str | Path,
    normalized_motions_path: str | Path,
    election_cadence_years: int,
    election_anchor_year: int,
    election_anchor_month: int,
) -> pd.DataFrame:
    idx_map = {k: i for i, k in enumerate(IDEOLOGY_ORDER)}

    # Speech monthly ideology index from top category per speech
    scls = pd.read_parquet(speech_classifications_path, columns=["speech_id", "category", "normalized_weight"]).copy()
    scls["normalized_weight"] = pd.to_numeric(scls["normalized_weight"], errors="coerce").fillna(0.0)
    scls = scls.sort_values(["speech_id", "normalized_weight"], ascending=[True, False])
    stop = scls.groupby("speech_id", sort=False).first().reset_index()[["speech_id", "category"]]
    stop["speech_idx"] = stop["category"].map(idx_map)

    smeta = load_speech_metadata(speech_parquet_dir)[["speech_id", "date"]].copy()
    smeta["speech_id"] = smeta["speech_id"].astype(str)
    smeta["date"] = pd.to_datetime(smeta["date"], errors="coerce")
    s = stop.merge(smeta, on="speech_id", how="left").dropna(subset=["speech_idx", "date"])
    s["month"] = s["date"].dt.to_period("M").dt.to_timestamp()
    s_monthly = s.groupby("month", as_index=False)["speech_idx"].mean()

    # Motion monthly ideology index from top category per motion
    mcls = pd.read_parquet(motion_classifications_path, columns=["motion_id", "category", "normalized_weight"]).copy()
    mcls["normalized_weight"] = pd.to_numeric(mcls["normalized_weight"], errors="coerce").fillna(0.0)
    mcls = mcls.sort_values(["motion_id", "normalized_weight"], ascending=[True, False])
    mtop = mcls.groupby("motion_id", sort=False).first().reset_index()[["motion_id", "category"]]
    mtop["motion_idx"] = mtop["category"].map(idx_map)

    mdate = _motion_dates_from_normalized(normalized_motions_path)
    m = mtop.merge(mdate, on="motion_id", how="left").dropna(subset=["motion_idx", "motion_date"])
    m["month"] = m["motion_date"].dt.to_period("M").dt.to_timestamp()
    m_monthly = m.groupby("month", as_index=False)["motion_idx"].mean()

    min_month = min(s_monthly["month"].min(), m_monthly["month"].min())
    max_month = max(s_monthly["month"].max(), m_monthly["month"].max())
    monthly = pd.DataFrame({"month": pd.date_range(min_month, max_month, freq="MS")})
    monthly = monthly.merge(s_monthly, on="month", how="left").merge(m_monthly, on="month", how="left")
    monthly["action_idx"] = monthly[["speech_idx", "motion_idx"]].mean(axis=1)

    election_dates = _election_dates_between(
        min_month=min_month,
        max_month=max_month,
        cadence_years=election_cadence_years,
        anchor_year=election_anchor_year,
        anchor_month=election_anchor_month,
    )
    monthly["runup_12m"] = _runup_indicator(monthly["month"], election_dates, runup_months=12).astype(int)
    monthly["runup_6m"] = _runup_indicator(monthly["month"], election_dates, runup_months=6).astype(int)
    monthly["election_month"] = monthly["month"].isin(election_dates).astype(int)
    return monthly


def _select_exog(df: pd.DataFrame, mode: str) -> pd.DataFrame | None:
    if mode == "none":
        return None
    cols = [c for c in mode.split("+") if c]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return None
    return df[cols].astype(float)


def run_sarimax_search(monthly: pd.DataFrame, out_dir: Path, run: ExperimentRun, args: argparse.Namespace) -> dict:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as e:
        return {"status": "unavailable", "reason": f"statsmodels not available: {e}"}

    p_grid = _parse_int_grid(args.sarimax_p_grid)
    d_grid = _parse_int_grid(args.sarimax_d_grid)
    q_grid = _parse_int_grid(args.sarimax_q_grid)
    P_grid = _parse_int_grid(args.sarimax_P_grid)
    D_grid = _parse_int_grid(args.sarimax_D_grid)
    Q_grid = _parse_int_grid(args.sarimax_Q_grid)
    exog_modes = _parse_modes(args.sarimax_exog_modes)
    seasonal_period = int(args.sarimax_seasonal_period)
    holdout = int(args.sarimax_holdout_months)
    min_points = int(args.sarimax_min_points)

    targets = ["speech_idx", "motion_idx", "action_idx"]
    monthly_idx = monthly.copy().sort_values("month").set_index("month")
    trials = []
    best_rows = []
    fitted_rows = []
    step = 0

    for target in targets:
        y = monthly_idx[target].astype(float).copy()
        y = y.interpolate(limit_direction="both").ffill().bfill()
        if y.notna().sum() < min_points:
            continue

        split_ix = max(min_points, len(y) - holdout)
        if split_ix >= len(y):
            split_ix = int(len(y) * 0.8)
        y_train = y.iloc[:split_ix]
        y_test = y.iloc[split_ix:]
        if len(y_test) == 0:
            continue

        for p, d, q, P, D, Q, mode in itertools.product(p_grid, d_grid, q_grid, P_grid, D_grid, Q_grid, exog_modes):
            ex = _select_exog(monthly_idx, mode)
            ex_train = ex.iloc[:split_ix] if ex is not None else None
            ex_test = ex.iloc[split_ix:] if ex is not None else None
            trial = {
                "target": target,
                "p": p,
                "d": d,
                "q": q,
                "P": P,
                "D": D,
                "Q": Q,
                "s": seasonal_period,
                "exog_mode": mode,
                "ok": False,
            }
            try:
                model = SARIMAX(
                    y_train,
                    exog=ex_train,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False)
                pred = res.get_forecast(steps=len(y_test), exog=ex_test).predicted_mean
                trial["aic"] = float(res.aic)
                trial["bic"] = float(res.bic)
                trial["rmse"] = _rmse(y_test.to_numpy(dtype=float), pred.to_numpy(dtype=float))
                trial["mae"] = _mae(y_test.to_numpy(dtype=float), pred.to_numpy(dtype=float))
                trial["ok"] = True
            except Exception as e:
                trial["error"] = str(e)[:220]

            trials.append(trial)
            run.log_metrics(
                {
                    f"trial_{target}_rmse": trial.get("rmse", np.nan),
                    f"trial_{target}_aic": trial.get("aic", np.nan),
                    f"trial_{target}_mae": trial.get("mae", np.nan),
                },
                step=step,
            )
            step += 1

        tdf = pd.DataFrame([t for t in trials if t["target"] == target and t.get("ok")])
        if tdf.empty:
            continue

        tdf = tdf.sort_values(["rmse", "aic"], ascending=[True, True]).reset_index(drop=True)
        best = tdf.iloc[0].to_dict()
        best_rows.append(best)

        ex_full = _select_exog(monthly_idx, str(best["exog_mode"]))
        model_full = SARIMAX(
            y,
            exog=ex_full,
            order=(int(best["p"]), int(best["d"]), int(best["q"])),
            seasonal_order=(int(best["P"]), int(best["D"]), int(best["Q"]), seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res_full = model_full.fit(disp=False)
        fitted = res_full.fittedvalues
        for dt, yv, fv in zip(y.index, y.values, fitted.values):
            fitted_rows.append(
                {
                    "target": target,
                    "month": dt,
                    "observed": float(yv),
                    "fitted": float(fv),
                }
            )

        run.log_metrics({f"best_{target}_rmse": float(best["rmse"]), f"best_{target}_aic": float(best["aic"])})

    trials_df = pd.DataFrame(trials)
    best_df = pd.DataFrame(best_rows)
    fitted_df = pd.DataFrame(fitted_rows)

    trials_p = out_dir / "sarimax_hyperparam_trials.parquet"
    best_p = out_dir / "sarimax_best_models.parquet"
    fitted_p = out_dir / "sarimax_fitted_series.parquet"
    trials_df.to_parquet(trials_p, index=False, compression="zstd")
    best_df.to_parquet(best_p, index=False, compression="zstd")
    fitted_df.to_parquet(fitted_p, index=False, compression="zstd")

    run.log_artifact(str(trials_p))
    run.log_artifact(str(best_p))
    run.log_artifact(str(fitted_p))

    return {
        "status": "ok",
        "n_trials": int(len(trials_df)),
        "n_successful_trials": int(trials_df.get("ok", pd.Series(dtype=bool)).sum()) if not trials_df.empty else 0,
        "trials_path": str(trials_p),
        "best_models_path": str(best_p),
        "fitted_series_path": str(fitted_p),
        "best_by_target": best_df.to_dict(orient="records") if not best_df.empty else [],
    }


def _build_party_year_modalities(topic_year: pd.DataFrame) -> pd.DataFrame:
    idx_map = {k: i for i, k in enumerate(IDEOLOGY_ORDER)}
    df = topic_year[topic_year["topic"].isin(idx_map)].copy()
    df["idx"] = df["topic"].map(idx_map).astype(float)

    rows = []
    for party, year in df[["party", "year"]].drop_duplicates().itertuples(index=False):
        sub = df[(df["party"] == party) & (df["year"] == year)]

        def wmean(col: str) -> float:
            w = sub[col].to_numpy(dtype=float)
            x = sub["idx"].to_numpy(dtype=float)
            s = w.sum()
            if s <= 0:
                return float("nan")
            return float((x * w).sum() / s)

        rows.append({"party": party, "year": int(year), "speech": wmean("S"), "motion": wmean("M"), "vote": wmean("V")})

    out = pd.DataFrame(rows)
    out = out[~out["party"].astype(str).str.lower().isin({"unknown", "nyd", ""})].copy()
    out["action_idx"] = out[["motion", "vote"]].mean(axis=1)
    return out


def build(
    topic_year: pd.DataFrame,
    half_life_years: float,
    election_cadence_years: int = 4,
    election_anchor_year: int = 2010,
    runup_years: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    piv = _build_party_year_modalities(topic_year)
    for col in ["speech", "motion", "vote"]:
        if col not in piv.columns:
            piv[col] = np.nan
    piv["action_idx"] = piv[["motion", "vote"]].mean(axis=1)

    max_year = int(piv["year"].max())
    piv["age_years"] = max_year - piv["year"]
    piv["recency_weight"] = np.power(0.5, piv["age_years"] / float(half_life_years))

    party_rows = []
    for party, g in piv.groupby("party"):
        w = g["recency_weight"].to_numpy(dtype=float)
        party_rows.append(
            {
                "party": party,
                "n_years": int(len(g)),
                "latest_year": int(g["year"].max()),
                "weighted_speech_idx": _weighted_mean(g["speech"].to_numpy(dtype=float), w),
                "weighted_action_idx": _weighted_mean(g["action_idx"].to_numpy(dtype=float), w),
                "weighted_motion_idx": _weighted_mean(g["motion"].to_numpy(dtype=float), w),
                "weighted_vote_idx": _weighted_mean(g["vote"].to_numpy(dtype=float), w),
            }
        )
    party_df = pd.DataFrame(party_rows).sort_values("weighted_action_idx", ascending=False).reset_index(drop=True)

    year_df = (
        piv.groupby("year", as_index=False)
        .agg(
            parliament_speech_mean=("speech", "mean"),
            parliament_action_mean=("action_idx", "mean"),
            parliament_motion_mean=("motion", "mean"),
            parliament_vote_mean=("vote", "mean"),
            n_parties=("party", "nunique"),
            recency_weight=("recency_weight", "first"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

    election_years = _election_years_between(
        min_year=int(year_df["year"].min()),
        max_year=int(year_df["year"].max()),
        cadence_years=int(election_cadence_years),
        anchor_year=int(election_anchor_year),
    )
    year_df["election_runup"] = year_df["year"].map(
        lambda y: _is_election_runup_year(y, election_years=election_years, runup_years=int(runup_years))
    )

    ws = year_df["recency_weight"].to_numpy(dtype=float)
    runup_mask = year_df["election_runup"].to_numpy(dtype=bool)
    nonrunup_mask = ~runup_mask
    summary = {
        "half_life_years": float(half_life_years),
        "max_year": max_year,
        "election_cadence_years": int(election_cadence_years),
        "election_anchor_year": int(election_anchor_year),
        "runup_years": int(runup_years),
        "election_years_used": election_years,
        "parliament_weighted_speech_idx": _weighted_mean(year_df["parliament_speech_mean"].to_numpy(dtype=float), ws),
        "parliament_weighted_action_idx": _weighted_mean(year_df["parliament_action_mean"].to_numpy(dtype=float), ws),
        "parliament_runup_weighted_speech_idx": _weighted_mean(year_df.loc[runup_mask, "parliament_speech_mean"].to_numpy(dtype=float), ws[runup_mask]),
        "parliament_runup_weighted_action_idx": _weighted_mean(year_df.loc[runup_mask, "parliament_action_mean"].to_numpy(dtype=float), ws[runup_mask]),
        "parliament_nonrunup_weighted_speech_idx": _weighted_mean(year_df.loc[nonrunup_mask, "parliament_speech_mean"].to_numpy(dtype=float), ws[nonrunup_mask]),
        "parliament_nonrunup_weighted_action_idx": _weighted_mean(year_df.loc[nonrunup_mask, "parliament_action_mean"].to_numpy(dtype=float), ws[nonrunup_mask]),
        "top_party_weighted_action": None if party_df.empty else str(party_df.iloc[0]["party"]),
        "sd_weighted_action_idx": None if "SD" not in set(party_df["party"]) else float(party_df.loc[party_df["party"] == "SD", "weighted_action_idx"].iloc[0]),
        "m_weighted_action_idx": None if "M" not in set(party_df["party"]) else float(party_df.loc[party_df["party"] == "M", "weighted_action_idx"].iloc[0]),
        "runup_minus_nonrunup_action_idx": None,
    }
    if pd.notna(summary["parliament_runup_weighted_action_idx"]) and pd.notna(summary["parliament_nonrunup_weighted_action_idx"]):
        summary["runup_minus_nonrunup_action_idx"] = float(
            summary["parliament_runup_weighted_action_idx"] - summary["parliament_nonrunup_weighted_action_idx"]
        )
    return party_df, year_df, summary


def main() -> None:
    safe = thermal_safe_defaults("safe")
    ap = argparse.ArgumentParser(description="Recency-weighted party/parliament trend analysis")
    ap.add_argument("--topic-year", default="output/analysis/promise_fulfillment_party_topic_year.parquet")
    ap.add_argument("--speech-classifications", default="data/parquet/speech_classifications_with_rhetoric_full.parquet")
    ap.add_argument("--speech-parquet-dir", default="data/speeches/parquet")
    ap.add_argument("--motion-classifications", default="data/parquet/classifications.parquet")
    ap.add_argument("--normalized-motions", default="data/parquet/normalized_motions.parquet")
    ap.add_argument("--out-dir", default="output/analysis")
    ap.add_argument("--half-life-years", type=float, default=3.0)
    ap.add_argument("--election-cadence-years", type=int, default=4)
    ap.add_argument("--election-anchor-year", type=int, default=2010)
    ap.add_argument("--election-anchor-month", type=int, default=9)
    ap.add_argument("--runup-years", type=int, default=1)
    ap.add_argument("--cpu-fraction", type=float, default=float(os.environ.get("CLASSIFIER_CPU_FRACTION", str(safe["cpu_fraction"]))))

    ap.add_argument("--sarimax", action="store_true", help="Run SARIMAX hyperparameter search on monthly speech/motion series")
    ap.add_argument("--sarimax-p-grid", default="0,1")
    ap.add_argument("--sarimax-d-grid", default="0,1")
    ap.add_argument("--sarimax-q-grid", default="0,1")
    ap.add_argument("--sarimax-P-grid", default="0,1")
    ap.add_argument("--sarimax-D-grid", default="0")
    ap.add_argument("--sarimax-Q-grid", default="0,1")
    ap.add_argument("--sarimax-seasonal-period", type=int, default=12)
    ap.add_argument("--sarimax-holdout-months", type=int, default=24)
    ap.add_argument("--sarimax-min-points", type=int, default=60)
    ap.add_argument(
        "--sarimax-exog-modes",
        default="none,runup_6m,runup_12m,election_month,runup_6m+election_month",
    )
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--mlflow-experiment", default="recency-sarimax")
    ap.add_argument("--mlflow-tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))

    args = ap.parse_args()

    throttle = apply_cpu_throttle(cpu_fraction=args.cpu_fraction)
    run = ExperimentRun.start(
        enabled=args.mlflow,
        experiment_name=args.mlflow_experiment,
        run_name="analyze-recency-weighted-trends",
        tracking_uri=args.mlflow_tracking_uri,
    )
    run.log_params(
        {
            "topic_year": args.topic_year,
            "half_life_years": args.half_life_years,
            "election_cadence_years": args.election_cadence_years,
            "election_anchor_year": args.election_anchor_year,
            "election_anchor_month": args.election_anchor_month,
            "runup_years": args.runup_years,
            "cpu_fraction": args.cpu_fraction,
            "max_threads": throttle.get("max_threads"),
            "sarimax": args.sarimax,
        }
    )

    topic_year_path = Path(args.topic_year)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    topic_year = pd.read_parquet(topic_year_path)
    party_df, year_df, summary = build(
        topic_year,
        half_life_years=args.half_life_years,
        election_cadence_years=args.election_cadence_years,
        election_anchor_year=args.election_anchor_year,
        runup_years=args.runup_years,
    )

    party_p = out_dir / "recency_weighted_party_scores.parquet"
    year_p = out_dir / "recency_weighted_parliament_timeseries.parquet"
    sum_p = out_dir / "recency_weighted_summary.json"

    party_df.to_parquet(party_p, index=False, compression="zstd")
    year_df.to_parquet(year_p, index=False, compression="zstd")

    summary["outputs"] = {
        "party_scores": str(party_p),
        "parliament_timeseries": str(year_p),
    }

    if args.sarimax:
        monthly = _build_monthly_series(
            speech_classifications_path=args.speech_classifications,
            speech_parquet_dir=args.speech_parquet_dir,
            motion_classifications_path=args.motion_classifications,
            normalized_motions_path=args.normalized_motions,
            election_cadence_years=args.election_cadence_years,
            election_anchor_year=args.election_anchor_year,
            election_anchor_month=args.election_anchor_month,
        )
        monthly_p = out_dir / "sarimax_monthly_series.parquet"
        monthly.to_parquet(monthly_p, index=False, compression="zstd")
        run.log_artifact(str(monthly_p))
        summary["outputs"]["sarimax_monthly_series"] = str(monthly_p)
        summary["sarimax"] = run_sarimax_search(monthly=monthly, out_dir=out_dir, run=run, args=args)

    sum_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run.log_artifact(str(sum_p))
    run.log_metrics(
        {
            "parliament_weighted_speech_idx": summary.get("parliament_weighted_speech_idx"),
            "parliament_weighted_action_idx": summary.get("parliament_weighted_action_idx"),
            "runup_minus_nonrunup_action_idx": summary.get("runup_minus_nonrunup_action_idx"),
        }
    )
    run.end(status="FINISHED")

    print(
        json.dumps(
            {
                "inputs": {"topic_year": str(topic_year_path)},
                "outputs": {
                    "party_scores": str(party_p),
                    "parliament_timeseries": str(year_p),
                    "summary": str(sum_p),
                },
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
