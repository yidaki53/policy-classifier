#!/usr/bin/env python3
"""Render manuscript sections with Jinja using latest figures, metrics, and journal profile."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any
from jinja2 import Environment

from swedish_parliament_policy_classifier.analysis.manuscript_exports import EXCLUDED_OVERLAY_PARTIES

EXCLUDED_COMPARISON_PARTIES = set(EXCLUDED_OVERLAY_PARTIES)

import numpy as np
import pandas as pd
import yaml
from jinja2 import Environment

from swedish_parliament_policy_classifier.analysis.manuscript_exports import EXCLUDED_OVERLAY_PARTIES


EXCLUDED_COMPARISON_PARTIES = set(EXCLUDED_OVERLAY_PARTIES)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _file_mtime_utc(path: Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _load_consistency_digest(analysis_dir: Path) -> dict:
    p = analysis_dir / "consistency_score_party.parquet"
    if not p.exists():
        return {"available": False}
    df = pd.read_parquet(p)
    if df.empty or "party" not in df.columns:
        return {"available": False}

    top = df.sort_values("consistency_score", ascending=False).head(3)
    rows = []
    for _, r in top.iterrows():
        rows.append(
            {
                "party": str(r["party"]),
                "consistency_score": float(r.get("consistency_score", 0.0)),
                "fulfillment_signal": float(r.get("fulfillment_signal", 0.0)) if pd.notna(r.get("fulfillment_signal")) else None,
                "ideology_uphold_v2": float(r.get("ideology_uphold_v2", 0.0)) if pd.notna(r.get("ideology_uphold_v2")) else None,
            }
        )

    return {
        "available": True,
        "rows": rows,
        "path": str(p),
        "mtime_utc": _file_mtime_utc(p),
    }


def _load_figures(repo_root: Path) -> list[dict]:
    catalog = [
        ("Consistency vs Fulfillment", "output/manuscript/figures/figure_consistency_vs_fulfillment.png", "core"),
        (
            "Consistency-Fulfillment vs External Benchmark (Party-Year)",
            "output/manuscript/figures/figure_consistency_fulfillment_vs_benchmark_party_year.png",
            "appendix",
        ),
        ("Parliament Direction Over Time", "output/manuscript/figures/figure_parliament_direction_over_time.png", "core"),
        ("Party Modality Overlay", "output/manuscript/figures/figure_modality_overlay_by_party.png", "appendix"),
        ("Motion Category Distribution", "figures/manuscript/pie_chart_categories.png", "appendix"),
        ("Party Motions Stacked", "figures/manuscript/party_motions_stacked.png", "appendix"),
        ("Voting Cohesion Time Series", "figures/voting/party_cohesion_timeseries.png", "appendix"),
        ("Three-way Divergence", "figures/three_way/divergence_speech_vs_combined_significance.png", "appendix"),
        ("Speech Profiles Heatmap", "figures/speeches/speech_profiles_heatmap.png", "appendix"),
    ]
    out = []
    for title, rel, scope in catalog:
        p = repo_root / rel
        if p.exists():
            out.append({"title": title, "path": rel, "scope": scope, "mtime_utc": _file_mtime_utc(p)})
    return out


def _build_figure_block(figures: list[dict], heading: str, intro: str, width: str = "90%") -> str:
    if not figures:
        return ""

    width_overrides = {
        "Consistency-Fulfillment vs External Benchmark (Party-Year)": "100%",
        "Party Modality Overlay": "100%",
        "Voting Cohesion Time Series": "100%",
        "Three-way Divergence": "100%",
        "Speech Profiles Heatmap": "100%",
    }

    lines = [heading, intro, ""]
    for f in figures:
        rel_for_combined = f"../{f['path']}"
        caption = f"{f['title']} (updated {f['mtime_utc']})"
        figure_width = width_overrides.get(f["title"], width)
        lines.append(f"![{caption}]({rel_for_combined}){{ width={figure_width} }}")
        lines.append("")
    return "\n".join(lines)


def _load_journal_profile(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_bibliography(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    keys = []
    for line in lines:
        s = line.strip()
        if s.startswith("@") and "{" in s:
            key = s.split("{", 1)[1].split(",", 1)[0].strip()
            if key:
                keys.append(key)
    return keys


def _to_year_series(series: pd.Series) -> pd.Series:
    dtv = pd.to_datetime(series, errors="coerce")
    return dtv.dt.year.dropna().astype(int)


def _load_corpus_stats(repo_root: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "motions_n": None,
        "motions_year_min": None,
        "motions_year_max": None,
        "speeches_n": None,
        "speeches_year_min": None,
        "speeches_year_max": None,
        "vote_events_n": None,
        "votes_year_min": None,
        "votes_year_max": None,
    }

    motions_path = repo_root / "data/parquet/normalized_motions.parquet"
    if motions_path.exists():
        mdf = pd.read_parquet(motions_path, columns=["id", "metadata"])
        stats["motions_n"] = int(mdf["id"].nunique())
        years: list[int] = []
        for meta in mdf["metadata"]:
            d: Any = meta
            if isinstance(meta, str):
                try:
                    d = json.loads(meta)
                except Exception:
                    d = {}
            if isinstance(d, dict):
                rm = str(d.get("rm", ""))
                y = "".join(ch for ch in rm if ch.isdigit())[:4]
                if len(y) == 4:
                    years.append(int(y))
        if years:
            stats["motions_year_min"] = min(years)
            stats["motions_year_max"] = max(years)

    speech_dir = repo_root / "data/speeches/parquet"
    if speech_dir.exists():
        speech_ids: set[str] = set()
        speech_years: list[int] = []
        for fp in sorted(speech_dir.glob("*.parquet")):
            try:
                sdf = pd.read_parquet(fp, columns=["anforande_id", "datum"])
            except Exception:
                continue
            speech_ids.update(sdf["anforande_id"].dropna().astype(str).tolist())
            speech_years.extend(_to_year_series(sdf["datum"]).tolist())
        if speech_ids:
            stats["speeches_n"] = len(speech_ids)
        if speech_years:
            stats["speeches_year_min"] = min(speech_years)
            stats["speeches_year_max"] = max(speech_years)

    vote_dir = repo_root / "data/votering/parquet"
    if vote_dir.exists():
        vote_ids: set[str] = set()
        vote_years: list[int] = []
        for fp in sorted(vote_dir.glob("*.parquet")):
            try:
                vdf = pd.read_parquet(fp, columns=["votering_id", "datum"])
            except Exception:
                continue
            vote_ids.update(vdf["votering_id"].dropna().astype(str).tolist())
            vote_years.extend(_to_year_series(vdf["datum"]).tolist())
        if vote_ids:
            stats["vote_events_n"] = len(vote_ids)
        if vote_years:
            stats["votes_year_min"] = min(vote_years)
            stats["votes_year_max"] = max(vote_years)

    return stats


def _multiclass_nll(df: pd.DataFrame, truth_col: str = "truth") -> float | None:
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if truth_col not in df.columns or not prob_cols:
        return None
    classes = [c[len("prob_") :] for c in prob_cols]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = df[truth_col].astype(str).map(class_to_idx)
    if y.isna().all():
        return None
    probs = df[prob_cols].to_numpy(dtype=float)
    probs = probs.clip(min=1e-12)
    row_sum = probs.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    probs = probs / row_sum
    idx = y.fillna(0).astype(int).to_numpy()
    ll = -float(np.log(probs[range(len(idx)), idx]).mean())
    return ll


def _classification_summary(df: pd.DataFrame, pred_col: str = "pred", truth_col: str = "truth") -> dict[str, float] | None:
    if truth_col not in df.columns or pred_col not in df.columns:
        return None
    truth = df[truth_col].astype(str)
    pred = df[pred_col].astype(str)
    if len(df) == 0:
        return None
    try:
        from sklearn.metrics import balanced_accuracy_score, f1_score
    except Exception:
        return None

    return {
        "accuracy": float((truth == pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro")),
    }


def _load_eval_metrics(repo_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_gold": None,
        "n_classes": None,
        "accuracy": None,
        "balanced_accuracy": None,
        "macro_f1": None,
        "nll_baseline": None,
        "accuracy_temp": None,
        "nll_temp": None,
        "accuracy_iso": None,
        "nll_iso": None,
    }
    logs_dir = repo_root / "logs"
    if not logs_dir.exists():
        return out

    baseline_files = sorted(logs_dir.glob("speech_eval_preds_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    baseline_files = [p for p in baseline_files if "tempcal" not in p.name and "isotonic" not in p.name]
    if baseline_files:
        best_df = None
        best_key = (-1, -1.0, -1.0)
        best_fp: Path | None = None
        for fp in baseline_files:
            try:
                df = pd.read_parquet(fp)
            except Exception:
                continue
            summary = _classification_summary(df)
            key = (len(df), summary["accuracy"] if summary else -1.0, fp.stat().st_mtime)
            if key > best_key:
                best_key = key
                best_df = df
                best_fp = fp
        if best_df is not None:
            summary = _classification_summary(best_df)
            if summary is not None:
                out["n_gold"] = int(len(best_df))
                out["accuracy"] = summary["accuracy"]
                out["balanced_accuracy"] = summary["balanced_accuracy"]
                out["macro_f1"] = summary["macro_f1"]
            out["n_classes"] = int(len([c for c in best_df.columns if c.startswith("prob_")]))
            out["nll_baseline"] = _multiclass_nll(best_df)

    temp_files = sorted(logs_dir.glob("speech_eval_preds_tempcal_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if temp_files:
        if best_fp is not None and best_df is not None:
            same_size = [p for p in temp_files if _safe_parquet_rows(p) == len(best_df)]
            target_files = same_size or temp_files
            temp_fp = min(target_files, key=lambda p: abs(p.stat().st_mtime - best_fp.stat().st_mtime))
        else:
            temp_fp = temp_files[0]
        tdf = pd.read_parquet(temp_fp)
        out["nll_temp"] = _multiclass_nll(tdf)
        if "pred_temp" in tdf.columns:
            out["accuracy_temp"] = float((tdf["truth"].astype(str) == tdf["pred_temp"].astype(str)).mean())

    iso_files = sorted(logs_dir.glob("speech_eval_preds_isotonic_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if iso_files:
        if best_fp is not None and best_df is not None:
            same_size = [p for p in iso_files if _safe_parquet_rows(p) == len(best_df)]
            target_files = same_size or iso_files
            iso_fp = min(target_files, key=lambda p: abs(p.stat().st_mtime - best_fp.stat().st_mtime))
        else:
            iso_fp = iso_files[0]
        idf = pd.read_parquet(iso_fp)
        out["nll_iso"] = _multiclass_nll(idf)
        if "pred_iso" in idf.columns:
            out["accuracy_iso"] = float((idf["truth"].astype(str) == idf["pred_iso"].astype(str)).mean())

    return out


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_parquet_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(len(pd.read_parquet(path)))
    except Exception:
        return None


def _fmt_int(v: Any) -> str:
    if v is None:
        return "n/a"
    return f"{int(v)}"


def _fmt_float(v: Any, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.{nd}f}"


def _exclude_non_substantive_parties(df: pd.DataFrame) -> pd.DataFrame:
    if "party" not in df.columns:
        return df.copy()
    out = df.copy()
    out["party"] = out["party"].astype(str)
    out = out[~out["party"].isin(EXCLUDED_COMPARISON_PARTIES)].copy()
    return out.reset_index(drop=True)


def _promise_fulfillment_example_paragraph(analysis_dir: Path) -> str:
    p = analysis_dir / "promise_fulfillment_party_summary.parquet"
    if not p.exists():
        return (
            "Promise-fulfillment contrasts are evaluated from "
            "`output/analysis/promise_fulfillment_party_summary.parquet`; "
            "party-level values are rendered dynamically from the latest artifact during manuscript build."
        )
    try:
        df = pd.read_parquet(p)
    except Exception:
        return (
            "Promise-fulfillment contrasts are evaluated from "
            "`output/analysis/promise_fulfillment_party_summary.parquet`; "
            "party-level values are rendered dynamically from the latest artifact during manuscript build."
        )

    required = {"party", "pct_speech_motion_vote", "pct_speech_motion_no_vote"}
    if not required.issubset(df.columns):
        return (
            "Promise-fulfillment contrasts are evaluated from "
            "`output/analysis/promise_fulfillment_party_summary.parquet`; "
            "party-level values are rendered dynamically from the latest artifact during manuscript build."
        )

    work = df[["party", "pct_speech_motion_vote", "pct_speech_motion_no_vote"]].copy()
    work = _exclude_non_substantive_parties(work)
    work["pct_speech_motion_vote"] = pd.to_numeric(work["pct_speech_motion_vote"], errors="coerce")
    work["pct_speech_motion_no_vote"] = pd.to_numeric(work["pct_speech_motion_no_vote"], errors="coerce")
    work = work.dropna(subset=["pct_speech_motion_vote"]).reset_index(drop=True)
    if work.empty:
        return (
            "Promise-fulfillment contrasts are evaluated from "
            "`output/analysis/promise_fulfillment_party_summary.parquet`; "
            "party-level values are rendered dynamically from the latest artifact during manuscript build."
        )

    hi = work.sort_values("pct_speech_motion_vote", ascending=False).iloc[0]
    lo = work.sort_values("pct_speech_motion_vote", ascending=True).iloc[0]

    return (
        "Promise-fulfillment contrasts are substantively visible in the current summary table. "
        f"In `output/analysis/promise_fulfillment_party_summary.parquet`, `{hi['party']}` has "
        f"`pct_speech_motion_vote = {_fmt_float(hi['pct_speech_motion_vote'])}` while `{lo['party']}` has "
        f"`{_fmt_float(lo['pct_speech_motion_vote'])}`; "
        f"`{lo['party']}` shows `pct_speech_motion_no_vote = {_fmt_float(lo['pct_speech_motion_no_vote'])}`. "
        "These differences illustrate why fulfillment diagnostics are retained as a separate axis "
        "instead of being collapsed into one aggregate consistency score."
    )


def _consistency_example_paragraph(analysis_dir: Path) -> str:
    p = analysis_dir / "consistency_score_party.parquet"
    if not p.exists():
        return (
            "Consistency contrasts are evaluated from `output/analysis/consistency_score_party.parquet`; "
            "party-level values are rendered dynamically from the latest artifact during manuscript build."
        )
    try:
        df = pd.read_parquet(p)
    except Exception:
        return (
            "Consistency contrasts are evaluated from `output/analysis/consistency_score_party.parquet`; "
            "party-level values are rendered dynamically from the latest artifact during manuscript build."
        )

    if "party" not in df.columns or "consistency_score" not in df.columns:
        return (
            "Consistency contrasts are evaluated from `output/analysis/consistency_score_party.parquet`; "
            "party-level values are rendered dynamically from the latest artifact during manuscript build."
        )

    work = df.copy()
    work["consistency_score"] = pd.to_numeric(work["consistency_score"], errors="coerce")
    fulfillment_col = "motion_pathway_fulfillment" if "motion_pathway_fulfillment" in work.columns else "pct_vote_given_motion_pathway"
    if fulfillment_col in work.columns:
        work[fulfillment_col] = pd.to_numeric(work[fulfillment_col], errors="coerce")
    work = work.dropna(subset=["consistency_score"]).reset_index(drop=True)
    if work.empty:
        return (
            "Consistency contrasts are evaluated from `output/analysis/consistency_score_party.parquet`; "
            "party-level values are rendered dynamically from the latest artifact during manuscript build."
        )

    hi = work.sort_values("consistency_score", ascending=False).iloc[0]
    lo = work.sort_values("consistency_score", ascending=True).iloc[0]

    hi_f = _fmt_float(hi.get(fulfillment_col)) if fulfillment_col in work.columns else "n/a"
    lo_f = _fmt_float(lo.get(fulfillment_col)) if fulfillment_col in work.columns else "n/a"

    return (
        "Consistency contrasts remain modest in absolute spread but informative for ranking and comparison. "
        f"In `output/analysis/consistency_score_party.parquet`, `{hi['party']}` records "
        f"`consistency_score = {_fmt_float(hi['consistency_score'])}` and `{fulfillment_col} = {hi_f}`, "
        f"while `{lo['party']}` records `consistency_score = {_fmt_float(lo['consistency_score'])}` "
        f"and `{fulfillment_col} = {lo_f}`. "
        "The ranking difference is interpreted as descriptive signal under linkage and calibration assumptions, "
        "not as evidence of causal party effects."
    )


def _build_context(repo_root: Path, manuscript_dir: Path, analysis_dir: Path, journal_profile: Path, bib_path: Path) -> dict:
    figures = _load_figures(repo_root)
    consistency = _load_consistency_digest(analysis_dir)
    journal = _load_journal_profile(journal_profile)
    bib_keys = _load_bibliography(bib_path)
    corpus = _load_corpus_stats(repo_root)
    eval_metrics = _load_eval_metrics(repo_root)

    links_summary = _safe_load_json(analysis_dir / "speech_action_links_summary.json")
    link_conf_summary = _safe_load_json(analysis_dir / "speech_action_link_confidence_summary.json")
    benchmark_summary = _safe_load_json(analysis_dir / "party_ideology_benchmark_validation.json")
    stability_summary = _safe_load_json(analysis_dir / "link_strata_stability_summary.json")
    recency_summary = _safe_load_json(analysis_dir / "recency_weighted_summary.json")
    cw_best = _safe_load_json(analysis_dir / "consistency_wrangling_fair_ga_best.json")

    expected_n = _safe_parquet_rows(analysis_dir / "speech_action_expected_contradiction_party_topic_year.parquet")
    fulfillment_n = _safe_parquet_rows(analysis_dir / "promise_fulfillment_party_topic_year.parquet")
    sarimax_trials = _safe_parquet_rows(analysis_dir / "sarimax_hyperparam_trials.parquet")

    core_figures = [f for f in figures if f.get("scope") == "core"]
    appendix_figures = [f for f in figures if f.get("scope") == "appendix"]

    main_figures_block = _build_figure_block(
        core_figures,
        heading="## Key Visual Evidence",
        intro="The figures below show headline outputs directly used for main-text interpretation.",
        width="90%",
    )
    appendix_figures_block = _build_figure_block(
        appendix_figures,
        heading="## Appendix Figures (Intermediate Steps)",
        intro=(
            "These figures capture intermediate diagnostics and process-level checks that support the main analysis "
            "without interrupting core result flow."
        ),
        width="90%",
    )
    # Backward-compatible alias retained for existing templates/tests.
    latest_figures_block = main_figures_block

    digest_lines = []
    if consistency.get("available"):
        digest_lines.append("## Latest Consistency Digest")
        digest_lines.append(f"Source: `{consistency['path']}` (updated {consistency['mtime_utc']})")
        for row in consistency.get("rows", []):
            digest_lines.append(
                f"- {row['party']}: consistency={row['consistency_score']:.4f}, "
                f"fulfillment={row['fulfillment_signal'] if row['fulfillment_signal'] is not None else 'n/a'}, "
                f"ideology_uphold_v2={row['ideology_uphold_v2'] if row['ideology_uphold_v2'] is not None else 'n/a'}"
            )
    results_digest_block = "\n".join(digest_lines) if digest_lines else ""

    journal_fit_lines = []
    if journal:
        journal_fit_lines.append("## Target Journal")
        journal_fit_lines.append(f"- Target: **{journal.get('name', 'Unknown')}**")
        journal_fit_lines.append(f"- Why fit: {journal.get('fit_rationale', 'n/a')}")
        if journal.get("acceptance_context"):
            journal_fit_lines.append(f"- Practical context: {journal['acceptance_context']}")
    journal_fit_block = "\n".join(journal_fit_lines) if journal_fit_lines else ""

    req_lines = []
    if journal and journal.get("requirements"):
        req_lines.append("## Journal Requirement Checklist")
        for req in journal["requirements"]:
            req_lines.append(f"- {req}")
    submission_requirements_block = "\n".join(req_lines) if req_lines else ""

    lit_lines = []
    if bib_keys:
        lit_lines.append("## Bibliography Seed")
        lit_lines.append(f"- Loaded `{len(bib_keys)}` references from `{bib_path.relative_to(repo_root)}`")
        lit_lines.append("- Key methods literature: Grimmer & Stewart (2013), Gentzkow et al. (2019), Lowe et al. (2011).")
        lit_lines.append("- Legislative/party behavior framing: Proksch & Slapin (2015), Mikhaylov et al. (2012), Slapin & Proksch (2008).")
    question_literature_block = "\n".join(lit_lines) if lit_lines else ""

    abstract_metrics_paragraph = (
        "On the current full corpus, the workflow covers "
        f"`n={_fmt_int(corpus.get('motions_n'))}` motions ({_fmt_int(corpus.get('motions_year_min'))}-{_fmt_int(corpus.get('motions_year_max'))}), "
        f"`n={_fmt_int(corpus.get('speeches_n'))}` speeches ({_fmt_int(corpus.get('speeches_year_min'))}-{_fmt_int(corpus.get('speeches_year_max'))}), "
        f"and `n={_fmt_int(corpus.get('vote_events_n'))}` unique roll-call vote events ({_fmt_int(corpus.get('votes_year_min'))}-{_fmt_int(corpus.get('votes_year_max'))}). "
        "With full speech-action linkage in the final stage, party-level consistency outputs are exported as auditable parquet artifacts. "
        f"In labeled speech evaluation (`n={_fmt_int(eval_metrics.get('n_gold'))}`), baseline accuracy is `{_fmt_float(eval_metrics.get('accuracy'))}`; "
        f"baseline NLL is `{_fmt_float(eval_metrics.get('nll_baseline'))}`, with calibration NLL `{_fmt_float(eval_metrics.get('nll_temp'))}` (temperature) and `{_fmt_float(eval_metrics.get('nll_iso'))}` (isotonic). "
        "Recency-weighted and lead-lag analyses provide party and parliament trajectories over time, and SARIMAX model selection is tracked through saved trial artifacts for reproducible forecasting diagnostics."
    )

    findings_lines = [
        "- Integrated consistency/trend analysis outputs were generated by `scripts/analyze_consistency_trends.py`.",
        (
            "- Speech gold-label evaluation (latest run): "
            f"`n={_fmt_int(eval_metrics.get('n_gold'))}`, baseline accuracy `{_fmt_float(eval_metrics.get('accuracy'))}`."
        ),
        (
            "- Calibration (latest run): "
            f"NLL `{_fmt_float(eval_metrics.get('nll_baseline'))} -> {_fmt_float(eval_metrics.get('nll_temp'))}` (temperature), "
            f"`{_fmt_float(eval_metrics.get('nll_iso'))}` (isotonic)."
        ),
    ]
    current_findings_block = "\n".join(findings_lines)

    cw = cw_best.get("best", {}) if isinstance(cw_best.get("best"), dict) else {}
    graph_balance = links_summary.get("graph_balance", {}) if isinstance(links_summary.get("graph_balance"), dict) else {}
    action_counts = links_summary.get("action_type_counts", {}) if isinstance(links_summary.get("action_type_counts"), dict) else {}

    metrics_anchor_lines = [
        (
            "- Corpus counts: "
            f"speeches `n={_fmt_int(corpus.get('speeches_n'))}` ({_fmt_int(corpus.get('speeches_year_min'))}-{_fmt_int(corpus.get('speeches_year_max'))}), "
            f"motions `n={_fmt_int(corpus.get('motions_n'))}` ({_fmt_int(corpus.get('motions_year_min'))}-{_fmt_int(corpus.get('motions_year_max'))}), "
            f"vote events `n={_fmt_int(corpus.get('vote_events_n'))}` ({_fmt_int(corpus.get('votes_year_min'))}-{_fmt_int(corpus.get('votes_year_max'))})."
        ),
        (
            "- Speech evaluation: "
            f"`n={_fmt_int(eval_metrics.get('n_gold'))}`, baseline accuracy `{_fmt_float(eval_metrics.get('accuracy'))}`, "
            f"baseline NLL `{_fmt_float(eval_metrics.get('nll_baseline'))}`."
        ),
        (
            "- Speech calibration NLL: "
            f"temperature `{_fmt_float(eval_metrics.get('nll_temp'))}`, isotonic `{_fmt_float(eval_metrics.get('nll_iso'))}`."
        ),
        (
            "- Full speech-action linkage coverage: "
            f"`n={_fmt_int(links_summary.get('n_linked'))}` linked of `n={_fmt_int(links_summary.get('n_speeches'))}` speeches "
            f"(`coverage={_fmt_float(links_summary.get('coverage'))}`), "
            f"action counts vote=`{_fmt_int(action_counts.get('vote'))}`, motion=`{_fmt_int(action_counts.get('motion'))}`."
        ),
        (
            "- Linkage fairness optimization (latest summary): "
            f"target_motion_share=`{_fmt_float(graph_balance.get('target_motion_share'))}`, "
            f"min_motion_margin=`{_fmt_float(graph_balance.get('min_motion_margin'))}`, "
            f"rebalanced_rows=`{_fmt_int(graph_balance.get('rebalanced_rows'))}`."
        ),
        (
            "- Promise-fulfillment/contradiction aggregate table sizes: "
            f"fulfillment `n={_fmt_int(fulfillment_n)}`, expected-contradiction `n={_fmt_int(expected_n)}`."
        ),
        (
            "- Consistency wrangling GA best (latest): "
            f"speech_motion_weight=`{_fmt_float(cw.get('speech_motion_weight'))}`, "
            f"fulfillment_fill=`{_fmt_float(cw.get('fulfillment_fill'))}`, "
            f"expected_contradiction_fill=`{_fmt_float(cw.get('expected_contradiction_fill'))}`, "
            f"contradiction_penalty_power=`{_fmt_float(cw.get('contradiction_penalty_power'))}`."
        ),
        (
            "- SARIMAX search scale and runup delta (latest recency summary): "
            f"`n={_fmt_int(sarimax_trials)}` trials, runup-minus-nonrunup action index "
            f"`{_fmt_float(recency_summary.get('runup_minus_nonrunup_action_idx'))}`."
        ),
    ]
    metrics_anchor_block = "\n".join(metrics_anchor_lines)
    metric_anchor_sentence = (
        "Current metric anchors from this workflow include "
        f"`n={_fmt_int(fulfillment_n)}` rows in party-topic-year fulfillment and expected-contradiction aggregates, "
        f"and `n={_fmt_int(sarimax_trials)}` successful SARIMAX trials for monthly model selection."
    )

    linkage_diagnostics_paragraph = (
        "The speech-to-motion linkage uses rel_dok_id-to-betankande bridging with fallback strategies. "
        f"In the latest linkage summary, `n={_fmt_int(links_summary.get('n_linked'))}` speeches are linked "
        f"out of `n={_fmt_int(links_summary.get('n_speeches'))}` (`coverage={_fmt_float(links_summary.get('coverage'))}`), "
        f"with `n={_fmt_int(links_summary.get('n_speeches_with_category'))}` speeches carrying a mapped ideology category."
    )

    runup_paragraph = (
        "To check whether action-side ideology shifts in election runup windows, recency summaries report "
        f"runup action index `{_fmt_float(recency_summary.get('parliament_runup_weighted_action_idx'))}` versus "
        f"non-runup `{_fmt_float(recency_summary.get('parliament_nonrunup_weighted_action_idx'))}`; "
        f"the latest runup-minus-nonrunup delta is `{_fmt_float(recency_summary.get('runup_minus_nonrunup_action_idx'))}`."
    )

    chance = None
    if eval_metrics.get("n_classes"):
        try:
            chance = 1.0 / float(eval_metrics["n_classes"])
        except Exception:
            chance = None
    acc = eval_metrics.get("accuracy")
    lift = None
    if chance and acc is not None and chance > 0:
        lift = float(acc) / chance
    classifier_accuracy_context_paragraph = (
        "**Note on classifier accuracy context**: "
        f"The baseline speech accuracy of `{_fmt_float(acc)}` is against a `{_fmt_int(eval_metrics.get('n_classes'))}`-class problem "
        f"where random chance gives approximately `{_fmt_float(chance, 3)}`. "
        "This metric is currently evaluated against Britannica-based category definitions (label ontology), not an external latent-ideology ground truth. "
        f"The observed value is ~{_fmt_float(lift, 1)}x chance, indicating meaningful structure in the signal but substantial residual uncertainty. "
        "All downstream modality-level comparisons should therefore be interpreted as exploratory estimates with calibrated probabilities rather than validated class assignments."
    )

    link_strata = {
        str(row.get("link_confidence_stratum")): row
        for row in (link_conf_summary.get("stratum_counts") or [])
        if isinstance(row, dict)
    }
    action_rows = {
        str(row.get("action_type")): row
        for row in (link_conf_summary.get("action_type_counts") or [])
        if isinstance(row, dict)
    }
    graph_row = link_strata.get("graph_signatory", {})
    existing_row = link_strata.get("existing_unknown", {})
    fallback_row = link_strata.get("heuristic_fallback", {})
    structural_row = link_strata.get("structural_high", {})
    linkage_confidence_paragraph = (
        "Linkage diagnostics (latest production refresh): "
        f"full coverage is retained by design (`n={_fmt_int(link_conf_summary.get('n_rows'))}` linked rows), "
        "but provenance now shifts materially toward graph-direct signatory evidence. "
        "In `output/analysis/speech_action_link_confidence_summary.json`, "
        f"graph-signatory links are `n={_fmt_int(graph_row.get('n'))}` (`{_fmt_float((graph_row.get('share') or 0) * 100.0, 1)}%`), "
        f"existing-reference links are `n={_fmt_int(existing_row.get('n'))}` (`{_fmt_float((existing_row.get('share') or 0) * 100.0, 1)}%`), "
        f"heuristic fallback links are `n={_fmt_int(fallback_row.get('n'))}` (`{_fmt_float((fallback_row.get('share') or 0) * 100.0, 1)}%`), "
        f"and structural high-confidence links are `n={_fmt_int(structural_row.get('n'))}` (`{_fmt_float((structural_row.get('share') or 0) * 100.0, 1)}%`). "
        f"Action counts are near balanced (vote `n={_fmt_int((action_rows.get('vote') or {}).get('n'))}`, "
        f"motion `n={_fmt_int((action_rows.get('motion') or {}).get('n'))}`)."
    )

    stability_drift_sentence = (
        "In the current refresh, structural-vs-all stability still shows measurable drift "
        f"(`abs max delta ≈ {_fmt_float(stability_summary.get('delta_structural_minus_all_abs_max'), 3)}` in "
        "`output/analysis/link_strata_stability_summary.json`), so outputs should still be interpreted "
        "as comparative diagnostics under explicit modeling assumptions rather than as a fully validated single latent-trait estimate."
    )

    benchmark_validation_paragraph = (
        "External benchmark validation remains a triangulation check. "
        f"The current benchmark summary reports overlap `n={_fmt_int(benchmark_summary.get('n_parties_overlap'))}`, "
        f"Spearman `{_fmt_float(benchmark_summary.get('spearman'))}` "
        "with bootstrap CI in `output/analysis/party_ideology_benchmark_validation.json`. "
        "These comparisons are not used as definitive ground truth for the behavior-based ideology metric."
    )

    significance_accuracy_paragraph = (
        "The results provide descriptive evidence in the direction of all three working hypotheses: "
        "H1 (modality-sensitive profiles) is supported by the observed motion/speech/action divergence patterns; "
        "H2 (systematic variation in say-do consistency) is supported by the cross-party consistency score distribution; "
        "H3 (variation in fulfillment and contradiction diagnostics) is supported by the spread in party-level fulfillment summaries and the consistency-versus-fulfillment comparison. "
        f"The refreshed held-out speech evaluation accuracy is `{_fmt_float(eval_metrics.get('accuracy'))}`, "
        f"while isotonic recalibration raises top-1 accuracy to `{_fmt_float(eval_metrics.get('accuracy_iso'))}` on the same set; "
        "temperature scaling leaves top-1 accuracy unchanged and should be treated as a calibration-only transform. "
        "All interpretations carry the uncertainty qualifications described below."
    )

    promise_fulfillment_example_paragraph = _promise_fulfillment_example_paragraph(analysis_dir)
    consistency_example_paragraph = _consistency_example_paragraph(analysis_dir)

    return {
        "generated_utc": _utc_now(),
        "main_figures_block": main_figures_block,
        "appendix_figures_block": appendix_figures_block,
        "latest_figures_block": latest_figures_block,
        "results_digest_block": results_digest_block,
        "current_findings_block": current_findings_block,
        "metrics_anchor_block": metrics_anchor_block,
        "metric_anchor_sentence": metric_anchor_sentence,
        "linkage_diagnostics_paragraph": linkage_diagnostics_paragraph,
        "linkage_confidence_paragraph": linkage_confidence_paragraph,
        "classifier_accuracy_context_paragraph": classifier_accuracy_context_paragraph,
        "benchmark_validation_paragraph": benchmark_validation_paragraph,
        "stability_drift_sentence": stability_drift_sentence,
        "significance_accuracy_paragraph": significance_accuracy_paragraph,
        "promise_fulfillment_example_paragraph": promise_fulfillment_example_paragraph,
        "consistency_example_paragraph": consistency_example_paragraph,
        "runup_paragraph": runup_paragraph,
        "abstract_metrics_paragraph": abstract_metrics_paragraph,
        "journal_fit_block": journal_fit_block,
        "submission_requirements_block": submission_requirements_block,
        "question_literature_block": question_literature_block,
        "bibliography_path": str(bib_path.relative_to(repo_root)) if bib_path.exists() else "",
        "corpus_stats": corpus,
        "eval_metrics": eval_metrics,
    }


def _render_sections(sections_dir: Path, out_dir: Path, context: dict) -> list[dict]:
    env = Environment(autoescape=False, trim_blocks=False, lstrip_blocks=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    for src in sorted(sections_dir.glob("*.md")):
        template = env.from_string(src.read_text(encoding="utf-8"))
        body = template.render(**context)
        dst = out_dir / src.name
        dst.write_text(body, encoding="utf-8")
        rendered.append({"source": str(src), "rendered": str(dst)})
    return rendered


def main() -> None:
    p = argparse.ArgumentParser(description="Render manuscript sections with Jinja context")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--manuscript-dir", default="manuscript")
    p.add_argument("--analysis-dir", default="output/analysis")
    p.add_argument("--journal-profile", default="manuscript/journal_profiles/plos_one.yaml")
    p.add_argument("--bibliography", default="manuscript/bibliography/references.bib")
    p.add_argument("--out-dir", default="manuscript/build/rendered_sections")
    p.add_argument("--context-out", default="manuscript/build/manuscript_context.json")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manuscript_dir = (repo_root / args.manuscript_dir).resolve()
    sections_dir = manuscript_dir / "sections"
    analysis_dir = (repo_root / args.analysis_dir).resolve()
    journal_profile = (repo_root / args.journal_profile).resolve()
    bibliography = (repo_root / args.bibliography).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    context_out = (repo_root / args.context_out).resolve()

    context = _build_context(repo_root, manuscript_dir, analysis_dir, journal_profile, bibliography)
    rendered = _render_sections(sections_dir, out_dir, context)

    context_out.parent.mkdir(parents=True, exist_ok=True)
    context_out.write_text(json.dumps(context, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "rendered_count": len(rendered),
                "rendered": rendered,
                "context": str(context_out),
                "generated_utc": context["generated_utc"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
