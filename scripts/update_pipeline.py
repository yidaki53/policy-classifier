#!/usr/bin/env python3
"""Check Riksdagen data API for new motions/votes/speeches, download,
extract, classify, and regenerate all downstream analysis and manuscript artifacts.

Parquet-only pipeline. No SQLite. Uses loguru for structured logging.

Usage:
    uv run python scripts/update_pipeline.py
    uv run python scripts/update_pipeline.py --dry-run
    uv run python scripts/update_pipeline.py --cpu-fraction 0.5

Output manifest:
    logs/update_pipeline_YYYYMMDDTHHMMSSZ.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from loguru import logger

BASE_URLS = {
    "anforande": "https://data.riksdagen.se/dataset/anforande",
    "votering": "https://data.riksdagen.se/dataset/votering",
    "dokument": "https://data.riksdagen.se/dataset/dokument",
}

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _check_url_exists(url: str, timeout: int = 10) -> bool:
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        return resp.status_code == 200
    except Exception as e:
        logger.warning("HEAD check failed for {}: {}", url, e)
        return False


def _check_archive_freshness(url: str, local_path: str | Path, timeout: int = 10) -> bool:
    """Return True if the server archive is newer or larger than the local file."""
    local = Path(local_path)
    if not local.exists():
        return True
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        if resp.status_code != 200:
            return True  # check failed, assume stale to be safe
        server_len = resp.headers.get("Content-Length")
        if server_len is not None:
            try:
                if int(server_len) != local.stat().st_size:
                    return True
            except ValueError:
                pass
        last_modified = resp.headers.get("Last-Modified")
        if not last_modified:
            return True
        server_mtime = parsedate_to_datetime(last_modified)
        local_mtime = datetime.fromtimestamp(local.stat().st_mtime, tz=timezone.utc)
        return server_mtime > local_mtime + timedelta(hours=1)
    except Exception as e:
        logger.warning("Freshness check failed for {}: {}", url, e)
        return True


def _list_existing_periods(pattern: str, directory: str) -> set[str]:
    """Return set of period strings extracted from existing filenames.

    Handles both single-period (anforande-202526, votering-202526) and
    two-period bulk (mot-2022-2025, prop-2018-2021) naming conventions.
    """
    d = Path(directory)
    if not d.exists():
        return set()
    periods = set()
    for f in d.glob(pattern):
        stem = f.stem  # e.g. anforande-202526.json -> anforande-202526
        if "-" not in stem:
            continue
        clean = stem.replace(".json", "").replace(".csv", "").replace(".zip", "").replace(".parquet", "")
        parts = clean.split("-")
        if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
            period = f"{parts[-2]}-{parts[-1]}"
        else:
            period = parts[-1]
        periods.add(period)
    return periods


def _next_periods(current: set[str]) -> list[str]:
    """Generate next 3 speech/votering periods to check (YYMM format).

    Falls back to the current year when no local data is found.
    """
    from datetime import datetime

    now_year = datetime.now().year
    if current:
        max_year = max(
            (int(p[:4]) for p in current if p.isdigit() and len(p) >= 4),
            default=now_year - 1,
        )
    else:
        max_year = now_year - 1
    next_years = [max_year + i for i in range(1, 4)]
    periods = []
    for y in next_years:
        end = (y + 1) % 100
        periods.append(f"{y}{end:02d}")
    return periods


def _next_bulk_periods(current: set[str]) -> list[str]:
    """Generate next 2 bulk dataset periods (YYYY-YYYY format).

    Falls back to the current year when no local data is found.
    """
    from datetime import datetime

    now_year = datetime.now().year
    if current:
        max_end = max(
            (int(p.split("-")[1]) for p in current if "-" in p and p.split("-")[1].isdigit()),
            default=now_year - 1,
        )
    else:
        max_end = now_year - 1
    # Round up to next 4-year boundary
    next_start = max_end + 1
    if next_start % 4 != 0:
        next_start += (4 - next_start % 4)
    next_starts = [next_start + i * 4 for i in range(2)]
    periods = []
    for start in next_starts:
        end = start + 3
        periods.append(f"{start}-{end}")
    return periods


def _run_step(
    cmd: list[str],
    step_name: str,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    allow_fail: bool = False,
) -> dict[str, Any]:
    """Run a subprocess step and return structured result."""
    logger.info("STEP: {}", step_name)
    logger.debug("CMD: {}", " ".join(cmd))
    merged_env = {**os.environ, **(env or {})}
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        env=merged_env,
    )
    ok = proc.returncode == 0
    if not ok and not allow_fail:
        logger.error("FAILED: {} (exit {})", step_name, proc.returncode)
        logger.error("STDERR:\n{}", proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr)
        raise RuntimeError(f"Step {step_name} failed with exit code {proc.returncode}")
    if not ok:
        logger.warning("ALLOWED FAILURE: {} (exit {})", step_name, proc.returncode)
    else:
        logger.info("OK: {}", step_name)
    return {
        "step": step_name,
        "ok": ok,
        "returncode": proc.returncode,
        "stdout_preview": proc.stdout[:500] if proc.stdout else "",
        "stderr_preview": proc.stderr[:500] if proc.stderr else "",
    }


def _find_latest_period_file(pattern: str, directory: str) -> tuple[Path | None, str | None]:
    """Return the most recently modified local file matching the pattern, and its period.

    Handles both single-period and two-period bulk naming conventions.
    """
    d = Path(directory)
    if not d.exists():
        return None, None
    matches = list(d.glob(pattern))
    if not matches:
        return None, None
    latest = max(matches, key=lambda p: p.stat().st_mtime)
    stem = latest.stem
    if "-" not in stem:
        return None, None
    clean = stem.replace(".json", "").replace(".csv", "").replace(".zip", "").replace(".parquet", "")
    parts = clean.split("-")
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        period = f"{parts[-2]}-{parts[-1]}"
    else:
        period = parts[-1]
    return latest, period


def _latest_dates_in_parquet() -> dict[str, datetime | None]:
    """Return latest dates found in speech, motion, and vote parquet data."""
    from pathlib import Path
    import glob

    latest_speech = None
    latest_vote = None
    latest_motion = None

    for f in sorted(glob.glob("data/speeches/parquet/*.parquet")):
        try:
            df = pd.read_parquet(f, columns=["datum"])
            max_d = df["datum"].max()
            if pd.notna(max_d) and (latest_speech is None or max_d > latest_speech):
                latest_speech = max_d
        except Exception:
            pass

    for f in sorted(glob.glob("data/votering/parquet/*.parquet")):
        try:
            df = pd.read_parquet(f, columns=["datum"])
            max_d = df["datum"].max()
            if pd.notna(max_d) and (latest_vote is None or max_d > latest_vote):
                latest_vote = max_d
        except Exception:
            pass

    try:
        mot_df = pd.read_parquet("data/parquet/normalized_motions.parquet", columns=["date"])
        latest_motion = mot_df["date"].max()
    except Exception:
        pass

    return {
        "speech": latest_speech,
        "vote": latest_vote,
        "motion": latest_motion,
    }


def check_api_new_periods(dry_run: bool) -> dict[str, Any]:
    """Check Riksdagen data API for new periods and stale current-period archives."""
    logger.info("Checking API for new periods and stale archives...")
    results = {
        "anforande": {"new": [], "checked": [], "stale": []},
        "votering": {"new": [], "checked": [], "stale": []},
        "mot": {"new": [], "checked": [], "stale": []},
        "prop": {"new": [], "checked": [], "stale": []},
        "bet": {"new": [], "checked": [], "stale": []},
    }

    # --- New periods check ---
    existing_speech = _list_existing_periods("anforande-*.json.zip", "data/bulk_datasets")
    speech_candidates = _next_periods(existing_speech)
    for period in speech_candidates:
        url = f"{BASE_URLS['anforande']}/anforande-{period}.json.zip"
        exists = _check_url_exists(url)
        results["anforande"]["checked"].append({"period": period, "url": url, "exists": exists})
        if exists:
            results["anforande"]["new"].append(period)
            logger.info("New speech period available: {} at {}", period, url)
        else:
            logger.debug("Speech period not available: {}", period)

    existing_vot = _list_existing_periods("votering-*.csv.zip", "data/votering")
    vot_candidates = _next_periods(existing_vot)
    for period in vot_candidates:
        url = f"{BASE_URLS['votering']}/votering-{period}.csv.zip"
        exists = _check_url_exists(url)
        results["votering"]["checked"].append({"period": period, "url": url, "exists": exists})
        if exists:
            results["votering"]["new"].append(period)
            logger.info("New votering period available: {} at {}", period, url)
        else:
            logger.debug("Votering period not available: {}", period)

    for doktyp in ("mot", "prop", "bet"):
        existing = _list_existing_periods(f"{doktyp}-*.json.zip", "data/bulk_datasets")
        candidates = _next_bulk_periods(existing)
        for period in candidates:
            url = f"{BASE_URLS['dokument']}/{doktyp}-{period}.json.zip"
            exists = _check_url_exists(url)
            results[doktyp]["checked"].append({"period": period, "url": url, "exists": exists})
            if exists:
                results[doktyp]["new"].append(period)
                logger.info("New {} period available: {} at {}", doktyp, period, url)
            else:
                logger.debug("{} period not available: {}", doktyp, period)

    # --- Stale current-period check ---
    latest_speech_file, latest_speech_period = _find_latest_period_file("anforande-*.json.zip", "data/bulk_datasets")
    if latest_speech_period and latest_speech_file:
        url = f"{BASE_URLS['anforande']}/anforande-{latest_speech_period}.json.zip"
        if _check_archive_freshness(url, latest_speech_file):
            results["anforande"]["stale"].append({"period": latest_speech_period, "url": url})
            logger.info("Stale speech archive detected: {} at {}", latest_speech_period, url)

    latest_vot_file, latest_vot_period = _find_latest_period_file("votering-*.csv.zip", "data/votering")
    if latest_vot_period and latest_vot_file:
        url = f"{BASE_URLS['votering']}/votering-{latest_vot_period}.csv.zip"
        if _check_archive_freshness(url, latest_vot_file):
            results["votering"]["stale"].append({"period": latest_vot_period, "url": url})
            logger.info("Stale votering archive detected: {} at {}", latest_vot_period, url)

    for doktyp in ("mot", "prop", "bet"):
        latest_file, latest_period = _find_latest_period_file(f"{doktyp}-*.json.zip", "data/bulk_datasets")
        if latest_period and latest_file:
            url = f"{BASE_URLS['dokument']}/{doktyp}-{latest_period}.json.zip"
            if _check_archive_freshness(url, latest_file):
                results[doktyp]["stale"].append({"period": latest_period, "url": url})
                logger.info("Stale {} archive detected: {} at {}", doktyp, latest_period, url)

    # --- Latest dates in data ---
    latest_dates = _latest_dates_in_parquet()
    results["latest_dates"] = {k: str(v) if v is not None else None for k, v in latest_dates.items()}
    logger.info("Latest dates in local parquet: speech={}, vote={}, motion={}",
                latest_dates.get("speech"), latest_dates.get("vote"), latest_dates.get("motion"))

    return results


def download_data(dry_run: bool) -> dict[str, Any]:
    """Run all download scripts. They are incremental (skip existing)."""
    logger.info("Downloading data (incremental, skipping existing)...")
    steps = {}

    if dry_run:
        logger.info("DRY-RUN: previewing download commands")
        steps["speeches"] = _run_step(
            [sys.executable, str(SCRIPT_DIR / "download_speeches.py"), "--dry-run"],
            "download_speeches_dry_run",
            allow_fail=True,
        )
        steps["votering"] = _run_step(
            [sys.executable, str(SCRIPT_DIR / "download_votering.py"), "--dry-run"],
            "download_votering_dry_run",
            allow_fail=True,
        )
        steps["betankande"] = _run_step(
            [sys.executable, str(SCRIPT_DIR / "download_betankande.py"), "--dry-run"],
            "download_betankande_dry_run",
            allow_fail=True,
        )
        steps["bulk"] = _run_step(
            [sys.executable, str(SCRIPT_DIR / "download_bulk_datasets.py"), "--dry-run"],
            "download_bulk_dry_run",
            allow_fail=True,
        )
    else:
        steps["speeches"] = _run_step(
            [sys.executable, str(SCRIPT_DIR / "download_speeches.py")],
            "download_speeches",
            allow_fail=True,
        )
        steps["votering"] = _run_step(
            [sys.executable, str(SCRIPT_DIR / "download_votering.py")],
            "download_votering",
            allow_fail=True,
        )
        steps["betankande"] = _run_step(
            [sys.executable, str(SCRIPT_DIR / "download_betankande.py")],
            "download_betankande",
            allow_fail=True,
        )
        steps["bulk"] = _run_step(
            [sys.executable, str(SCRIPT_DIR / "download_bulk_datasets.py")],
            "download_bulk",
            allow_fail=True,
        )

    return {"dry_run": dry_run, **steps}


def extract_data(dry_run: bool) -> dict[str, Any]:
    """Extract new ZIPs to Parquet. Uses --force only on new periods implicitly."""
    logger.info("Extracting data to Parquet...")
    if dry_run:
        return {"dry_run": True, "note": "Extraction skipped in dry-run"}

    steps = {}
    steps["speeches"] = _run_step(
        [sys.executable, str(SCRIPT_DIR / "extract_speeches.py"), "--force"],
        "extract_speeches",
        allow_fail=True,
    )
    steps["votering"] = _run_step(
        [sys.executable, str(SCRIPT_DIR / "extract_votering.py"), "--force"],
        "extract_votering",
        allow_fail=True,
    )
    steps["betankande"] = _run_step(
        [sys.executable, str(SCRIPT_DIR / "extract_betankande.py"), "--force"],
        "extract_betankande",
        allow_fail=True,
    )
    return steps


def classify_and_adjust(dry_run: bool, cpu_fraction: float) -> dict[str, Any]:
    """Run classification pipeline."""
    logger.info("Classifying data...")
    if dry_run:
        return {"dry_run": True, "note": "Classification skipped in dry-run"}

    env = {"CLASSIFIER_CPU_FRACTION": str(cpu_fraction)}
    steps = {}

    # Classify speeches (resumes by default)
    steps["classify_speeches"] = _run_step(
        [sys.executable, str(SCRIPT_DIR / "classify_speeches_parquet.py")],
        "classify_speeches_parquet",
        env=env,
        allow_fail=True,
    )

    # Classify motions (from normalized motions parquet)
    steps["classify_motions"] = _run_step(
        [sys.executable, str(SCRIPT_DIR / "classify.py"), "--raw", "data/parquet/raw_motions.parquet"],
        "classify_motions",
        env=env,
        allow_fail=True,
    )

    # Rhetorical adjustment
    steps["rhetorical_adjustment"] = _run_step(
        [
            sys.executable,
            str(SCRIPT_DIR / "apply_rhetorical_adjustments.py"),
            "--classifications", "data/parquet/speech_classifications.parquet",
            "--speeches", "data/speeches/parquet",
            "--out", "data/parquet/speech_classifications_rhetorical_adjusted.parquet",
        ],
        "rhetorical_adjustment",
        env=env,
        allow_fail=True,
    )

    return steps


def rebuild_analysis(dry_run: bool, cpu_fraction: float) -> dict[str, Any]:
    """Rebuild all downstream analysis artifacts."""
    logger.info("Rebuilding analysis artifacts...")
    if dry_run:
        return {"dry_run": True, "note": "Analysis skipped in dry-run"}

    env = {"CLASSIFIER_CPU_FRACTION": str(cpu_fraction)}
    steps = {}

    # Speech-motion linkage
    steps["linkage"] = _run_step(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_speech_motion_linkage.py"),
            "--speech-classifications", "data/parquet/speech_classifications_rhetorical_adjusted.parquet",
            "--force",
        ],
        "build_speech_motion_linkage",
        env=env,
        allow_fail=True,
    )

    # Party profiles
    steps["profiles"] = _run_step(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_profiles.py"),
            "--speech-classifications", "data/parquet/speech_classifications_rhetorical_adjusted.parquet",
            "--force",
        ],
        "build_profiles",
        env=env,
        allow_fail=True,
    )

    # Speech analysis suite (individual steps to avoid OOM)
    analysis_steps = [
        (
            "link_all_speeches",
            [
                sys.executable,
                str(SCRIPT_DIR / "link_all_speeches_to_action.py"),
                "--speech-classifications", "data/parquet/speech_classifications_rhetorical_adjusted.parquet",
                "--force",
            ],
        ),
        (
            "axis_alignment",
            [
                sys.executable,
                str(SCRIPT_DIR / "compute_ideology_axis_alignment.py"),
                "--speech-classifications", "data/parquet/speech_classifications_rhetorical_adjusted.parquet",
            ],
        ),
        (
            "contradiction",
            [
                sys.executable,
                str(SCRIPT_DIR / "score_say_vs_do_contradiction.py"),
                "--axis-scores", "output/analysis/speech_action_axis_scores.parquet",
                "--edge-out", "output/analysis/speech_action_contradiction_edges.parquet",
                "--expected-out", "output/analysis/speech_action_expected_contradiction_party_topic_year.parquet",
            ],
        ),
        (
            "link_confidence",
            [
                sys.executable,
                str(SCRIPT_DIR / "compute_link_confidence_strata.py"),
                "--links", "data/parquet/speech_action_links.parquet",
                "--out", "output/analysis/speech_action_link_confidence_strata.parquet",
                "--summary-out", "output/analysis/speech_action_link_confidence_summary.json",
            ],
        ),
        (
            "uncertainty",
            [
                sys.executable,
                str(SCRIPT_DIR / "bootstrap_say_do_uncertainty.py"),
                "--axis-scores", "output/analysis/speech_action_axis_scores.parquet",
                "--links", "data/parquet/speech_action_links.parquet",
                "--out", "output/analysis/say_do_uncertainty_intervals_party.parquet",
                "--summary-out", "output/analysis/say_do_uncertainty_summary.json",
            ],
        ),
        (
            "consistency",
            [
                sys.executable,
                str(SCRIPT_DIR / "analyze_consistency_trends.py"),
                "--analysis-dir", "output/analysis",
                "--figures-dir", "output/manuscript/figures",
            ],
        ),
        (
            "link_stability",
            [
                sys.executable,
                str(SCRIPT_DIR / "analyze_link_strata_stability.py"),
                "--axis-scores", "output/analysis/speech_action_axis_scores.parquet",
                "--link-strata", "output/analysis/speech_action_link_confidence_strata.parquet",
                "--out", "output/analysis/link_strata_stability_party.parquet",
                "--summary-out", "output/analysis/link_strata_stability_summary.json",
            ],
        ),
        (
            "latent",
            [
                sys.executable,
                str(SCRIPT_DIR / "fit_latent_party_ideology.py"),
                "--axis-scores", "output/analysis/speech_action_axis_scores.parquet",
                "--links", "data/parquet/speech_action_links.parquet",
                "--consistency", "output/analysis/consistency_score_party.parquet",
                "--out", "output/analysis/party_latent_ideology_estimates.parquet",
                "--summary-out", "output/analysis/party_latent_ideology_summary.json",
            ],
        ),
        (
            "recency",
            [
                sys.executable,
                str(SCRIPT_DIR / "analyze_recency_weighted_trends.py"),
                "--topic-year", "output/analysis/promise_fulfillment_party_topic_year.parquet",
                "--out-dir", "output/analysis",
                "--election-cadence-years", "4",
                "--runup-years", "1",
            ],
        ),
    ]

    for name, cmd in analysis_steps:
        steps[name] = _run_step(cmd, name, env=env, allow_fail=True)

    return steps


def regenerate_figures(dry_run: bool, cpu_fraction: float) -> dict[str, Any]:
    """Regenerate all visualization artifacts."""
    logger.info("Regenerating figures...")
    if dry_run:
        return {"dry_run": True, "note": "Figures skipped in dry-run"}

    env = {"CLASSIFIER_CPU_FRACTION": str(cpu_fraction)}
    steps = {}

    figure_steps = [
        (
            "manuscript_motion_figures",
            [sys.executable, str(SCRIPT_DIR / "generate_figures.py"), "--db", "data/swedish_parliament.db", "--out-dir", "figures/manuscript"],
        ),
        (
            "party_profiles",
            [sys.executable, str(SCRIPT_DIR / "visualize.py"), "--profiles", "data/parquet/party_profiles_recency.parquet", "--out", "figures"],
        ),
        (
            "party_profiles_advanced",
            [sys.executable, str(SCRIPT_DIR / "visualize_advanced.py"), "--profiles", "data/parquet/party_profiles_recency.parquet", "--out", "figures"],
        ),
        (
            "interactive",
            [
                sys.executable,
                str(SCRIPT_DIR / "visualize_interactive.py"),
                "--profiles", "data/parquet/party_profiles_recency.parquet",
                "--modality", "motion",
                "--out", "figures",
                "--file", "party_profiles_interactive.html",
            ],
        ),
        (
            "voting",
            [
                sys.executable,
                str(SCRIPT_DIR / "visualize_voting.py"),
                "--votering-parquet", "data/votering/parquet",
                "--out", "figures/voting",
                "--db", "data/swedish_parliament.db",
            ],
        ),
        (
            "speech_profiles",
            [
                sys.executable,
                str(SCRIPT_DIR / "analyze_speech_profiles.py"),
                "--speech-classifications", "data/parquet/speech_classifications_rhetorical_adjusted.parquet",
                "--speech-parquet-dir", "data/speeches/parquet",
                "--out", "figures/speeches",
            ],
        ),
        (
            "overlay",
            [
                sys.executable,
                str(SCRIPT_DIR / "generate_manuscript_overlay.py"),
                "--profiles", "data/parquet/party_profiles_recency.parquet",
                "--out", "output/manuscript/figures/figure_modality_overlay_by_party.png",
            ],
        ),
        (
            "three_way",
            [
                sys.executable,
                str(SCRIPT_DIR / "speeches_analysis.py"),
                "--profiles", "data/parquet/party_profiles_recency.parquet",
                "--out-dir", "figures/three_way",
                "--speech-motions", "data/parquet/speech_motions.parquet",
                "--motion-votes", "data/parquet/motion_votes.parquet",
                "--votering-dir", "data/votering/parquet",
            ],
        ),
    ]

    for name, cmd in figure_steps:
        steps[name] = _run_step(cmd, name, env=env, allow_fail=True)

    # Legacy alias copies
    alias_steps = [
        ("figures/manuscript/pie_chart_categories.png", "figures/combined/combined_pie_chart_categories.png"),
        ("figures/manuscript/party_motions_stacked.png", "figures/combined/combined_party_motions_stacked.png"),
        ("figures/manuscript/party_motions_stacked_normalized.png", "figures/combined/combined_party_motions_stacked_normalized.png"),
        ("figures/manuscript/ideology_timeline.png", "figures/combined/combined_ideology_timeline.png"),
        ("figures/manuscript/party_ideology_heatmap.png", "figures/combined/combined_party_ideology_heatmap.png"),
        ("figures/party_profiles_final.png", "figures/combined/combined_heatmap.png"),
        ("figures/party_profiles_final.pdf", "figures/combined/combined_heatmap.pdf"),
        ("figures/three_way/divergence_speech_vs_combined_significance.png", "figures/combined/three_way_comparison.png"),
    ]

    for src_rel, dst_rel in alias_steps:
        src = REPO_ROOT / src_rel
        dst = REPO_ROOT / dst_rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            logger.info("Copied alias {} -> {}", src_rel, dst_rel)

    return steps


def render_manuscript(dry_run: bool) -> dict[str, Any]:
    """Render manuscript sections and combine."""
    logger.info("Rendering manuscript...")
    if dry_run:
        return {"dry_run": True, "note": "Manuscript render skipped in dry-run"}

    steps = {}
    steps["render"] = _run_step(
        ["make", "render"],
        "manuscript_render",
        cwd=REPO_ROOT / "manuscript",
        allow_fail=True,
    )
    steps["combined"] = _run_step(
        ["make", "combined"],
        "manuscript_combined",
        cwd=REPO_ROOT / "manuscript",
        allow_fail=True,
    )
    return steps


def main():
    parser = argparse.ArgumentParser(
        description="Check Riksdagen API for new data, download, and run full downstream pipeline."
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview what would run without downloading or processing")
    parser.add_argument("--cpu-fraction", type=float, default=0.25, help="CPU fraction for thermal-safe execution")
    parser.add_argument("--skip-download", action="store_true", help="Skip download step (use existing data)")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction step")
    parser.add_argument("--skip-classify", action="store_true", help="Skip classification step")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis rebuild step")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure regeneration step")
    parser.add_argument("--skip-manuscript", action="store_true", help="Skip manuscript render step")
    args = parser.parse_args()

    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")
    ts = _utc_now()
    logger.info("=== Pipeline update started at {} ===", ts)
    logger.info("Dry-run: {}", args.dry_run)

    manifest: dict[str, Any] = {
        "run_ts": ts,
        "dry_run": args.dry_run,
        "cpu_fraction": args.cpu_fraction,
        "steps": {},
    }

    try:
        # Step 1: Check API for new periods
        manifest["steps"]["api_check"] = check_api_new_periods(dry_run=args.dry_run)

        # Step 2: Download
        if not args.skip_download:
            manifest["steps"]["download"] = download_data(dry_run=args.dry_run)
        else:
            manifest["steps"]["download"] = {"skipped": True}

        # Step 3: Extract
        if not args.skip_extract:
            manifest["steps"]["extract"] = extract_data(dry_run=args.dry_run)
        else:
            manifest["steps"]["extract"] = {"skipped": True}

        # Step 4: Classify + rhetorical adjustment
        if not args.skip_classify:
            manifest["steps"]["classify"] = classify_and_adjust(dry_run=args.dry_run, cpu_fraction=args.cpu_fraction)
        else:
            manifest["steps"]["classify"] = {"skipped": True}

        # Step 5: Rebuild analysis
        if not args.skip_analysis:
            manifest["steps"]["analysis"] = rebuild_analysis(dry_run=args.dry_run, cpu_fraction=args.cpu_fraction)
        else:
            manifest["steps"]["analysis"] = {"skipped": True}

        # Step 6: Regenerate figures
        if not args.skip_figures:
            manifest["steps"]["figures"] = regenerate_figures(dry_run=args.dry_run, cpu_fraction=args.cpu_fraction)
        else:
            manifest["steps"]["figures"] = {"skipped": True}

        # Step 7: Render manuscript
        if not args.skip_manuscript:
            manifest["steps"]["manuscript"] = render_manuscript(dry_run=args.dry_run)
        else:
            manifest["steps"]["manuscript"] = {"skipped": True}

    except Exception as e:
        logger.exception("Pipeline failed: {}", e)
        manifest["error"] = str(e)

    manifest["completed_at"] = _utc_now()

    # Write manifest
    out_dir = REPO_ROOT / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"update_pipeline_{ts}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    logger.info("Manifest written to {}", manifest_path)

    # Also update version/frontmatter in a summary file
    summary_path = out_dir / f"update_pipeline_{ts}.md"
    summary_path.write_text(
        f"---\n_agent_frontmatter:\n  id: pipeline-update-{ts}\n  purpose: Pipeline update run manifest and summary\n  steward: analysis\n  edit_policy: generated_do_not_edit\n  generator: scripts/update_pipeline.py\n  version: 1.0.0\n  run_ts: {ts}\n  dry_run: {args.dry_run}\n---\n\n# Pipeline Update {ts}\n\nSee manifest: `{manifest_path}`\n",
        encoding="utf-8",
    )
    logger.info("Summary written to {}", summary_path)

    logger.info("=== Pipeline update completed at {} ===", manifest["completed_at"])


if __name__ == "__main__":
    main()
