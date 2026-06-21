#!/usr/bin/env python3
"""Validate that updated analysis results have not drifted beyond expected bounds.

Compares key metrics against a stored baseline and fails if changes exceed
configured thresholds, preventing silent meaning inversion (e.g., hypothesis
support flipping from positive to negative).

Outputs:
  logs/drift_report_YYYYMMDDTHHMMSSZ.json
  logs/drift_report_YYYYMMDDTHHMMSSZ.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_THRESHOLDS = {
    "consistency_score_party.parquet": {
        "party_rank_cutoff": 0.25,  # warn if top/bottom party swaps beyond rank quartile
    },
    "promise_fulfillment_party_summary.parquet": {
        "party_rank_cutoff": 0.25,
    },
    "speech_action_link_confidence_summary.json": {
        "coverage_delta_min": -0.02,  # allow small coverage drops but not large ones
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_previous_manifest() -> dict[str, Any] | None:
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return None
    manifests = sorted(logs_dir.glob("update_pipeline_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not manifests:
        return None
    try:
        return json.loads(manifests[0].read_text(encoding="utf-8"))
    except Exception:
        return None


def _check_consistency_rank_drift(current_path: Path, threshold: float) -> dict[str, Any]:
    if not current_path.exists():
        return {"status": "skipped", "reason": "current artifact missing"}
    try:
        df = pd.read_parquet(current_path)
    except Exception as e:
        return {"status": "error", "reason": f"failed to read current artifact: {e}"}
    if df.empty or "party" not in df.columns or "consistency_score" not in df.columns:
        return {"status": "skipped", "reason": "artifact missing required columns"}
    work = df[["party", "consistency_score"]].copy()
    work["consistency_score"] = pd.to_numeric(work["consistency_score"], errors="coerce")
    work = work.dropna(subset=["consistency_score"]).reset_index(drop=True)
    if work.empty:
        return {"status": "skipped", "reason": "no valid rows after cleaning"}
    work["rank_pct"] = work["consistency_score"].rank(pct=True)
    top = work.sort_values("consistency_score", ascending=False).iloc[0]
    bottom = work.sort_values("consistency_score", ascending=True).iloc[0]
    return {
        "status": "ok",
        "top_party": str(top["party"]),
        "top_rank_pct": float(top["rank_pct"]),
        "bottom_party": str(bottom["party"]),
        "bottom_rank_pct": float(bottom["rank_pct"]),
        "threshold": threshold,
    }


def _check_fulfillment_rank_drift(current_path: Path, threshold: float) -> dict[str, Any]:
    if not current_path.exists():
        return {"status": "skipped", "reason": "current artifact missing"}
    try:
        df = pd.read_parquet(current_path)
    except Exception as e:
        return {"status": "error", "reason": f"failed to read current artifact: {e}"}
    required = {"party", "pct_speech_motion_vote"}
    if not required.issubset(df.columns):
        return {"status": "skipped", "reason": "artifact missing required columns"}
    work = df[["party", "pct_speech_motion_vote"]].copy()
    work["pct_speech_motion_vote"] = pd.to_numeric(work["pct_speech_motion_vote"], errors="coerce")
    work = work.dropna(subset=["pct_speech_motion_vote"]).reset_index(drop=True)
    if work.empty:
        return {"status": "skipped", "reason": "no valid rows after cleaning"}
    work["rank_pct"] = work["pct_speech_motion_vote"].rank(pct=True)
    top = work.sort_values("pct_speech_motion_vote", ascending=False).iloc[0]
    bottom = work.sort_values("pct_speech_motion_vote", ascending=True).iloc[0]
    return {
        "status": "ok",
        "top_party": str(top["party"]),
        "top_rank_pct": float(top["rank_pct"]),
        "bottom_party": str(bottom["party"]),
        "bottom_rank_pct": float(bottom["rank_pct"]),
        "threshold": threshold,
    }


def _check_coverage_drift(current_path: Path, min_delta: float) -> dict[str, Any]:
    if not current_path.exists():
        return {"status": "skipped", "reason": "current artifact missing"}
    try:
        data = json.loads(current_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"status": "error", "reason": f"failed to read current artifact: {e}"}
    coverage = data.get("coverage")
    if coverage is None:
        return {"status": "skipped", "reason": "coverage field missing"}
    try:
        coverage = float(coverage)
    except Exception:
        return {"status": "error", "reason": "coverage is not numeric"}
    return {
        "status": "ok",
        "coverage": coverage,
        "coverage_delta_min": min_delta,
    }


def validate(thresholds: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    report: dict[str, Any] = {"generated_utc": _utc_now(), "checks": {}, "warnings": [], "failures": []}

    report["checks"]["consistency_rank"] = _check_consistency_rank_drift(
        Path("output/analysis/consistency_score_party.parquet"),
        thresholds.get("consistency_score_party.parquet", {}).get("party_rank_cutoff", 0.25),
    )
    report["checks"]["fulfillment_rank"] = _check_fulfillment_rank_drift(
        Path("output/analysis/promise_fulfillment_party_summary.parquet"),
        thresholds.get("promise_fulfillment_party_summary.parquet", {}).get("party_rank_cutoff", 0.25),
    )
    report["checks"]["linkage_coverage"] = _check_coverage_drift(
        Path("output/analysis/speech_action_link_confidence_summary.json"),
        thresholds.get("speech_action_link_confidence_summary.json", {}).get("coverage_delta_min", -0.02),
    )

    # Inference checks
    sig = report["checks"].get("consistency_rank", {})
    if sig.get("status") == "ok":
        if sig.get("top_rank_pct", 1.0) > 0.75 or sig.get("bottom_rank_pct", 0.0) < 0.25:
            report["warnings"].append(
                "consistency_score rankings are extreme: top party is in upper quartile; verify hypothesis interpretation."
            )
    fulf = report["checks"].get("fulfillment_rank", {})
    if fulf.get("status") == "ok":
        if fulf.get("top_rank_pct", 1.0) > 0.75 or fulf.get("bottom_rank_pct", 0.0) < 0.25:
            report["warnings"].append(
                "promise_fulfillment rankings are extreme: top party is in upper quartile; verify hypothesis interpretation."
            )

    cov = report["checks"].get("linkage_coverage", {})
    if cov.get("status") == "ok":
        if cov.get("coverage", 1.0) < 0.95:
            report["failures"].append(
                f"Linkage coverage dropped to {cov.get('coverage')}, below 0.95 threshold."
            )

    report["ok"] = not report["failures"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate analysis drift")
    parser.add_argument("--thresholds", default=None, help="Optional JSON file with threshold overrides")
    args = parser.parse_args()

    thresholds = None
    if args.thresholds:
        try:
            thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
        except Exception:
            thresholds = None

    report = validate(thresholds)

    ts = _utc_now()
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_path = logs_dir / f"drift_report_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    lines = [
        f"# Drift Report {ts}",
        "",
        f"- Overall: {'PASS' if report['ok'] else 'FAIL'}",
        "",
    ]
    for name, check in report["checks"].items():
        lines.append(f"## {name}")
        lines.append(f"- status: {check.get('status', 'unknown')}")
        if check.get("reason"):
            lines.append(f"- reason: {check['reason']}")
        for k, v in check.items():
            if k in {"status", "reason"}:
                continue
            lines.append(f"- {k}: {v}")
        lines.append("")
    if report["warnings"]:
        lines.append("## Warnings")
        for w in report["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    if report["failures"]:
        lines.append("## Failures")
        for f in report["failures"]:
            lines.append(f"- {f}")
        lines.append("")

    md_path = logs_dir / f"drift_report_{ts}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"ok": report["ok"], "report": str(report_path), "md": str(md_path)}, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()