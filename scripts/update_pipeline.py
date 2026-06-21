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
import time as _time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from loguru import logger

_FRESHNESS_CACHE_PATH = Path("data/.api_freshness_cache.json")
_FRESHNESS_CACHE_TTL = timedelta(hours=6)
_LAST_FETCH_CACHE_PATH = Path("data/.last_fetch_cache.json")

def _load_freshness_cache() -> dict[str, Any]:
    if _FRESHNESS_CACHE_PATH.exists():
        try:
            raw = json.loads(_FRESHNESS_CACHE_PATH.read_text())
            # Migrate legacy float entries (checked timestamp only) to dict format
            migrated: dict[str, Any] = {}
            for k, v in raw.items():
                if isinstance(v, (int, float)):
                    migrated[k] = {"checked": v}
                else:
                    migrated[k] = v
            return migrated
        except Exception:
            return {}
    return {}

def _save_freshness_cache(cache: dict[str, Any]):
    _FRESHNESS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _FRESHNESS_CACHE_PATH.write_text(json.dumps(cache, indent=2, default=str))


def _load_last_fetch_cache() -> dict[str, str]:
    """Load cache of last successful fetch dates per document type."""
    if _LAST_FETCH_CACHE_PATH.exists():
        try:
            return json.loads(_LAST_FETCH_CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_last_fetch_cache(cache: dict[str, str]):
    """Save cache of last successful fetch dates per document type."""
    _LAST_FETCH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LAST_FETCH_CACHE_PATH.write_text(json.dumps(cache, indent=2, default=str))

def _url_is_stale(url: str, local_path: str | Path) -> bool:
    """Return True if the server archive might be newer than the local file.

    Uses GET+stream (more widely accepted than HEAD) with a local
    freshness cache to avoid hammering the API.
    """
    local = Path(local_path)
    if not local.exists():
        return True

    cache = _load_freshness_cache()
    now = _time.time()
    cache_key = f"stale:{url}"
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        checked_at = cached.get("checked", 0)
        if (now - checked_at) < _FRESHNESS_CACHE_TTL.total_seconds():
            return cached.get("stale", False)
    elif isinstance(cached, (int, float)):
        # Legacy format: just a timestamp, assume not stale if recent
        if (now - cached) < _FRESHNESS_CACHE_TTL.total_seconds():
            return False

    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "riksdagen-pipeline/1.0"})
        resp = session.get(url, timeout=10, stream=True)
        resp.close()
        if resp.status_code != 200:
            return False  # can't reach it, assume not stale
        server_len = resp.headers.get("Content-Length")
        if server_len is not None:
            try:
                if int(server_len) != local.stat().st_size:
                    cache[cache_key] = {"stale": True, "checked": now}
                    _save_freshness_cache(cache)
                    return True
            except ValueError:
                pass
        last_modified = resp.headers.get("Last-Modified")
        if last_modified:
            try:
                server_mtime = parsedate_to_datetime(last_modified)
                local_mtime = datetime.fromtimestamp(local.stat().st_mtime, tz=timezone.utc)
                stale = server_mtime > local_mtime + timedelta(hours=1)
                cache[cache_key] = {"stale": stale, "checked": now}
                _save_freshness_cache(cache)
                return stale
            except Exception:
                pass
        cache[cache_key] = {"stale": False, "checked": now}
        _save_freshness_cache(cache)
        return False
    except Exception:
        # network failure: return False (assume not stale) to avoid blocking pipeline
        return False

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
    """Check if a URL exists. Uses cached results to avoid repeated API calls."""
    cache = _load_freshness_cache()
    now = _time.time()
    cache_key = f"exists:{url}"
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        checked_at = cached.get("checked", 0)
        if (now - checked_at) < 86400:
            return cached.get("exists", False)

    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "riksdagen-pipeline/1.0"})
        resp = session.get(url, timeout=timeout, stream=True)
        exists = resp.status_code == 200
        resp.close()
        cache[cache_key] = {"exists": exists, "checked": now}
        _save_freshness_cache(cache)
        return exists
    except Exception:
        return False


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
    """Generate next plausible speech/votering period to check (YYMM format).

    The Riksdagen server only publishes data for riksmöten that have
    started or completed. Riksmöte runs Sep--Aug. We derive the
    current riksmöte from the calendar date and only check at most
    one period beyond the latest one we already have.
    """
    from datetime import datetime

    now = datetime.now()
    now_year = now.year
    now_month = now.month

    # Current riksmöte: if we're in Jan-Aug, the current rm started in (year-1).
    # If we're in Sep-Dec, the current rm started in year.
    current_riksmote_start = now_year - 1 if now_month <= 8 else now_year

    if current:
        max_year = max(
            (int(p[:4]) for p in current if p.isdigit() and len(p) >= 4),
            default=now_year - 1,
        )
    else:
        max_year = now_year - 1

    # Only check one period beyond what we already have, and never beyond
    # the current riksmöte + 1.
    next_period_start = max_year + 1
    if next_period_start > current_riksmote_start + 1:
        return []

    end = (next_period_start + 1) % 100
    period = f"{next_period_start}{end:02d}"
    return [period]


def _next_bulk_periods(current: set[str]) -> list[str]:
    """Generate next bulk dataset period (YYYY-YYYY format).

    Bulk periods are 4-year blocks aligned to election cycles.
    Only generate the next block after the latest one we have,
    and never more than one block ahead of the current year.
    """
    from datetime import datetime

    now_year = datetime.now().year
    if current:
        max_end = max(
            (int(p.split("-")[1]) for p in current if "-" in p and p.split("-")[1].isdigit()),
            default=now_year - 4,
        )
    else:
        max_end = now_year - 4

    # Next block starts the year after the latest end year,
    # rounded to the next 4-year boundary.
    next_start = max_end + 1
    if next_start % 4 != 0:
        next_start += (4 - next_start % 4)

    # Don't generate periods that are more than one block ahead.
    if next_start > now_year + 1:
        return []

    end = next_start + 3
    return [f"{next_start}-{end}"]


def _run_step(
    cmd: list[str],
    step_name: str,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    allow_fail: bool = False,
) -> dict[str, Any]:
    """Run a subprocess step and return structured result.

    Uses Popen with real-time stderr forwarding so progress messages
    appear in the terminal immediately (not buffered until completion).
    """
    logger.info(">>> STEP: {}", step_name)
    logger.debug("CMD: {}", " ".join(cmd))
    merged_env = {**os.environ, **(env or {})}
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=merged_env,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    # Read output line by line, forwarding stderr to parent real-time
    import selectors
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    sel.register(proc.stderr, selectors.EVENT_READ)
    while True:
        for key, _ in sel.select(timeout=0.5):
            line = key.fileobj.readline()
            if not line:
                continue
            if key.fileobj is proc.stdout:
                stdout_lines.append(line)
            else:
                stderr_lines.append(line)
                sys.stderr.write(line)
                sys.stderr.flush()
        if proc.poll() is not None:
            # Drain remaining output
            for line in proc.stdout.readlines():
                stdout_lines.append(line)
            for line in proc.stderr.readlines():
                stderr_lines.append(line)
                sys.stderr.write(line)
                sys.stderr.flush()
            break
    sel.close()
    proc.wait()
    elapsed = proc.returncode  # placeholder, will compute below
    ok = proc.returncode == 0
    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)
    status = "OK" if ok else "FAILED"
    elapsed_msg = ""
    if "started_at" in locals() or 'start' in dir():
        pass
    if not ok and not allow_fail:
        logger.error("{}: {} (exit {})", status, step_name, proc.returncode)
        logger.error("STDERR:\n{}", stderr[-2000:] if len(stderr) > 2000 else stderr)
        raise RuntimeError(f"Step {step_name} failed with exit code {proc.returncode}")
    if not ok:
        logger.warning("{}: {} (exit {})", status, step_name, proc.returncode)
    else:
        logger.info("{}: {}", status, step_name)
    return {
        "step": step_name,
        "ok": ok,
        "returncode": proc.returncode,
        "stdout_preview": stdout[:500] if stdout else "",
        "stderr_preview": stderr[:500] if stderr else "",
    }


def _find_latest_period_file(pattern: str, directory: str) -> tuple[Path | None, str | None]:
    """Return the chronologically latest-period local file matching the pattern, and its period.

    Handles both single-period (e.g. 202526) and two-period bulk (e.g. 2022-2025) naming.
    Sorts by the period itself (numeric), not by file modification time.
    """
    d = Path(directory)
    if not d.exists():
        return None, None
    matches = list(d.glob(pattern))
    if not matches:
        return None, None

    def _period_key(path: Path):
        stem = path.stem
        if "-" not in stem:
            return 0
        clean = stem.replace(".json", "").replace(".csv", "").replace(".zip", "").replace(".parquet", "")
        parts = clean.split("-")
        # two-period bulk: mot-2022-2025 -> period = "2022-2025", key = 2022
        if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
            return int(parts[-2])
        # single-period: anforande-202526 -> period = "202526", key = 202526
        if parts[-1].isdigit():
            return int(parts[-1])
        return 0

    latest = max(matches, key=_period_key)
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
        if latest_motion is None or (isinstance(latest_motion, float) and pd.isna(latest_motion)):
            latest_motion = None
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

    # --- Stale current-period check (uses local freshness cache) ---
    latest_speech_file, latest_speech_period = _find_latest_period_file("anforande-*.json.zip", "data/bulk_datasets")
    if latest_speech_period and latest_speech_file:
        url = f"{BASE_URLS['anforande']}/anforande-{latest_speech_period}.json.zip"
        if _url_is_stale(url, latest_speech_file):
            results["anforande"]["stale"].append({"period": latest_speech_period, "url": url})
            logger.info("Stale speech archive detected: {} at {}", latest_speech_period, url)

    latest_vot_file, latest_vot_period = _find_latest_period_file("votering-*.csv.zip", "data/votering")
    if latest_vot_period and latest_vot_file:
        url = f"{BASE_URLS['votering']}/votering-{latest_vot_period}.csv.zip"
        if _url_is_stale(url, latest_vot_file):
            results["votering"]["stale"].append({"period": latest_vot_period, "url": url})
            logger.info("Stale votering archive detected: {} at {}", latest_vot_period, url)

    for doktyp in ("mot", "prop", "bet"):
        latest_file, latest_period = _find_latest_period_file(f"{doktyp}-*.json.zip", "data/bulk_datasets")
        if latest_period and latest_file:
            url = f"{BASE_URLS['dokument']}/{doktyp}-{latest_period}.json.zip"
            if _url_is_stale(url, latest_file):
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
        [sys.executable, str(SCRIPT_DIR / "extract_speeches.py")],
        "extract_speeches",
        allow_fail=True,
    )
    steps["votering"] = _run_step(
        [sys.executable, str(SCRIPT_DIR / "extract_votering.py")],
        "extract_votering",
        allow_fail=True,
    )
    steps["betankande"] = _run_step(
        [sys.executable, str(SCRIPT_DIR / "extract_betankande.py")],
        "extract_betankande",
        allow_fail=True,
    )
    return steps


def extract_new_data_sources(dry_run: bool, args: argparse.Namespace) -> dict[str, Any]:
    """Extract questions, betankande normalization, and interpellations."""
    logger.info("Extracting new data sources...")
    if dry_run:
        return {"dry_run": True, "note": "New source extraction skipped in dry-run"}

    steps = {}

    if not args.skip_questions:
        steps["extract_questions"] = _run_step(
            [sys.executable, str(SCRIPT_DIR / "extract_questions.py"), "--force"],
            "extract_questions",
            allow_fail=True,
        )

    if not args.skip_betankande:
        steps["normalize_betankande"] = _run_step(
            [sys.executable, str(SCRIPT_DIR / "extract_betankande_to_normalized.py"), "--force"],
            "extract_betankande_to_normalized",
            allow_fail=True,
        )

    if not args.skip_ip:
        steps["extract_interpellations"] = _run_step(
            [sys.executable, str(SCRIPT_DIR / "extract_interpellations.py"), "--force"],
            "extract_interpellations",
            allow_fail=True,
        )

    return steps


def classify_new_data_sources(dry_run: bool, cpu_fraction: float, args: argparse.Namespace) -> dict[str, Any]:
    """Classify questions, betankande, and interpellations."""
    logger.info("Classifying new data sources...")
    if dry_run:
        return {"dry_run": True, "note": "New source classification skipped in dry-run"}

    env = {"CLASSIFIER_CPU_FRACTION": str(cpu_fraction)}
    steps = {}

    if not args.skip_questions:
        steps["classify_questions"] = _run_step(
            [sys.executable, str(SCRIPT_DIR / "classify_questions.py")],
            "classify_questions",
            env=env,
            allow_fail=True,
        )

    if not args.skip_betankande:
        steps["classify_betankande"] = _run_step(
            [sys.executable, str(SCRIPT_DIR / "classify_betankande.py")],
            "classify_betankande",
            env=env,
            allow_fail=True,
        )

    if not args.skip_ip:
        steps["classify_interpellations"] = _run_step(
            [sys.executable, str(SCRIPT_DIR / "classify_interpellations.py")],
            "classify_interpellations",
            env=env,
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
                "--speech-action-links", "data/parquet/speech_action_links_with_prop_bet.parquet",
                "--speech-classifications", "data/parquet/speech_classifications_rhetorical_adjusted.parquet",
            ],
        ),
        (
            "contradiction",
            [
                sys.executable,
                str(SCRIPT_DIR / "score_say_vs_do_contradiction.py"),
                "--speech-action-links", "data/parquet/speech_action_links_with_prop_bet.parquet",
                "--axis-scores", "output/analysis/speech_action_axis_scores.parquet",
                "--edge-out", "output/analysis/speech_action_contradiction_edges.parquet",
                "--expected-out", "output/analysis/speech_action_expected_contradiction_party_topic_year.parquet",
            ],
        ),
        (
            "contradiction_by_modality",
            [
                sys.executable,
                str(SCRIPT_DIR / "score_contradiction_by_modality.py"),
                "--edges", "output/analysis/speech_action_contradiction_edges.parquet",
            ],
        ),
        (
            "link_confidence",
            [
                sys.executable,
                str(SCRIPT_DIR / "compute_link_confidence_strata.py"),
                "--links", "data/parquet/speech_action_links_with_prop_bet.parquet",
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
                "--links", "data/parquet/speech_action_links_with_prop_bet.parquet",
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
                "--links", "data/parquet/speech_action_links_with_prop_bet.parquet",
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
            [sys.executable, str(SCRIPT_DIR / "generate_figures.py"), "--classifications", "data/parquet/classifications.parquet", "--normalized-motions", "data/parquet/normalized_motions.parquet", "--out-dir", "figures/manuscript"],
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
                "--normalized-motions", "data/parquet/normalized_motions.parquet",
                "--motion-votes", "data/parquet/motion_votes.parquet",
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


def link_prop_bet(dry_run: bool) -> dict[str, Any]:
    """Link speeches to propositions and betankande for modality-aware contradiction scoring."""
    logger.info("Linking speeches to propositions and betankande...")
    if dry_run:
        return {"dry_run": True, "note": "Prop/bet linking skipped in dry-run"}
    return _run_step(
        [
            sys.executable,
            str(SCRIPT_DIR / "link_prop_bet_to_speech.py"),
            "--speech-classifications", "data/parquet/speech_classifications_rhetorical_adjusted.parquet",
            "--force",
        ],
        "link_prop_bet_to_speech",
        allow_fail=True,
    )


def merge_prop_bet(dry_run: bool) -> dict[str, Any]:
    """Merge prop/bet links into the main speech_action_links table."""
    logger.info("Merging prop/bet links into speech_action_links...")
    if dry_run:
        return {"dry_run": True, "note": "Prop/bet merge skipped in dry-run"}
    return _run_step(
        [
            sys.executable,
            str(SCRIPT_DIR / "merge_prop_bet_into_action_links.py"),
            "--force",
        ],
        "merge_prop_bet_into_action_links",
        allow_fail=True,
    )


def _get_existing_ids(doktyp: str) -> set[str]:
    """Get set of existing document IDs from local parquet data.
    
    Args:
        doktyp: Document type ('mot' for motions, 'anf' for speeches)
    
    Returns:
        Set of document IDs already in local data
    """
    existing_ids = set()
    
    if doktyp == "mot":
        # Check normalized_motions.parquet
        try:
            mot_df = pd.read_parquet("data/parquet/normalized_motions.parquet", columns=["id"])
            existing_ids.update(mot_df["id"].astype(str).tolist())
        except Exception as e:
            logger.warning("Could not read normalized_motions.parquet: {}", e)
        
        # Check api_motions directory for previously fetched items
        api_dir = Path("data/parquet/api_motions")
        if api_dir.exists():
            for f in api_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(f, columns=["id"])
                    existing_ids.update(df["id"].astype(str).tolist())
                except Exception as e:
                    logger.warning("Could not read {}: {}", f, e)
    
    elif doktyp == "anf":
        # Check speech parquet files
        import glob
        for f in glob.glob("data/speeches/parquet/*.parquet"):
            try:
                df = pd.read_parquet(f, columns=["anforande_id"])
                existing_ids.update(df["anforande_id"].astype(str).tolist())
            except Exception as e:
                logger.warning("Could not read {}: {}", f, e)
        
        # Check api_speeches directory for previously fetched items
        api_dir = Path("data/parquet/api_speeches")
        if api_dir.exists():
            for f in api_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(f, columns=["id"])
                    existing_ids.update(df["id"].astype(str).tolist())
                except Exception as e:
                    logger.warning("Could not read {}: {}", f, e)
    
    return existing_ids


def fetch_new_items_from_api(dry_run: bool) -> dict[str, Any]:
    """Query live Riksdagen API for new items since our latest dataset date and append to parquet."""
    logger.info("Fetching new items from live API...")
    if dry_run:
        return {"dry_run": True, "note": "API fetch skipped in dry-run"}

    from swedish_parliament_policy_classifier.fetch.riksdag_client import fetch_page

    latest_dates = _latest_dates_in_parquet()
    last_fetch_cache = _load_last_fetch_cache()
    steps: dict[str, Any] = {}

    for doktyp, label, date_key in [
        ("mot", "motions", "motion"),
        ("anf", "speeches", "speech"),
    ]:
        since_date = latest_dates.get(date_key)
        if since_date is None:
            logger.warning("No latest date for {}; skipping live API fetch", label)
            continue
        if isinstance(since_date, str):
            since_date = pd.to_datetime(since_date)
        if pd.isna(since_date):
            logger.warning("Latest date is NaN for {}; skipping live API fetch", label)
            continue
        
        # Get existing IDs to avoid re-fetching
        existing_ids = _get_existing_ids(doktyp)
        logger.info("Found {} existing {} IDs in local data", len(existing_ids), label)
        
        # Check when we last fetched this document type
        last_fetch_date_str = last_fetch_cache.get(doktyp)
        if last_fetch_date_str:
            last_fetch_date = pd.to_datetime(last_fetch_date_str)
            # Only fetch from last fetch date forward (not 7 days back)
            from_date = last_fetch_date.strftime("%Y-%m-%d")
            logger.info("Last fetched {} on {}, fetching from {} forward", label, last_fetch_date_str, from_date)
        else:
            # First time fetching, use a small window to catch recent data
            from_date = (since_date - timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info("No previous fetch record for {}, fetching from {} forward", label, from_date)
        
        to_date = datetime.now().strftime("%Y-%m-%d")
        logger.info("Fetching {} from API: {} to {}", label, from_date, to_date)

        # Cross-page rate-limit guard: sleep progressively longer between pages
        def _page_sleep(p: int):
            import time as _t
            wait = min(0.5 + (p - 1) * 0.5, 10.0)  # 0.5s, 1.0s, 1.5s, ... up to 10s
            _t.sleep(wait)

        all_docs: list[dict[str, Any]] = []
        consecutive_failures = 0
        page = 1
        has_more = True
        max_consecutive_failures = 8
        while has_more and page <= 200:
            _page_sleep(page)
            try:
                docs, has_more = fetch_page(
                    doktyp=doktyp,
                    page=page,
                    fr=from_date,
                    till=to_date,
                    sort="datum",
                    sortorder="desc",
                    retries=6,
                    retry_delay=4.0,
                    timeout=20,
                )
                if not docs:
                    break
                all_docs.extend(docs)
                consecutive_failures = 0
                page += 1
                if len(all_docs) >= 5000:
                    logger.info("Reached 5000 docs for {}, stopping pagination", label)
                    break
            except Exception as e:
                consecutive_failures += 1
                logger.warning(
                    "API fetch failed for {} page {}: {} (consecutive failures: {}/{})",
                    label, page, e, consecutive_failures, max_consecutive_failures,
                )
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Too many consecutive API fetch failures for {}, aborting pagination", label)
                    break
                page += 1
                continue

        if not all_docs:
            logger.info("No new {} items from API", label)
            steps[label] = {"fetched": 0, "output": None}
            continue

        # Deduplicate against existing IDs
        df = pd.DataFrame(all_docs)
        id_col = "id" if "id" in df.columns else ("anforande_id" if doktyp == "anf" else "dok_id")
        if id_col in df.columns:
            df[id_col] = df[id_col].astype(str)
            new_docs = df[~df[id_col].isin(existing_ids)]
            logger.info("Deduplicated {} -> {} new {} items ({} already exist)", 
                       len(df), len(new_docs), label, len(df) - len(new_docs))
            df = new_docs
        
        if df.empty:
            logger.info("No new {} items after deduplication", label)
            steps[label] = {"fetched": 0, "output": None}
            continue

        out_path = REPO_ROOT / "data" / "parquet" / f"api_{label}"
        out_path.mkdir(parents=True, exist_ok=True)
        ts = _utc_now()
        parquet_file = out_path / f"{doktyp}_{ts}.parquet"

        df.to_parquet(parquet_file, index=False, compression="zstd")
        logger.info("Wrote {} new {} items to {}", len(df), label, parquet_file)
        steps[label] = {"fetched": len(df), "output": str(parquet_file)}
        
        # Update last fetch cache
        last_fetch_cache[doktyp] = to_date
        _save_last_fetch_cache(last_fetch_cache)

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
    parser.add_argument("--skip-api-fetch", action="store_true", help="Skip live API fetch step")
    parser.add_argument("--skip-classify", action="store_true", help="Skip classification step")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis rebuild step")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure regeneration step")
    parser.add_argument("--skip-manuscript", action="store_true", help="Skip manuscript render step")
    parser.add_argument("--skip-questions", action="store_true", help="Skip question download/extract/classify steps")
    parser.add_argument("--skip-betankande", action="store_true", help="Skip betankande normalization/classify steps")
    parser.add_argument("--skip-ip", action="store_true", help="Skip interpellation extraction/classify steps")
    parser.add_argument("--skip-prop-classify", action="store_true", help="Skip proposition classification (prop already classified in normalized_motions)")
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

    # Build ordered step plan for progress reporting
    step_plan: list[tuple[str, Any]] = []
    step_plan.append(("api_check", lambda: check_api_new_periods(dry_run=args.dry_run)))
    if not args.skip_download:
        step_plan.append(("download", lambda: download_data(dry_run=args.dry_run)))
    else:
        manifest["steps"]["download"] = {"skipped": True}
    if not args.skip_extract:
        step_plan.append(("extract", lambda: extract_data(dry_run=args.dry_run)))
    else:
        manifest["steps"]["extract"] = {"skipped": True}
    if not args.skip_api_fetch:
        step_plan.append(("api_fetch", lambda: fetch_new_items_from_api(dry_run=args.dry_run)))
    else:
        manifest["steps"]["api_fetch"] = {"skipped": True}
    if not args.skip_questions or not args.skip_betankande or not args.skip_ip:
        step_plan.append(("extract_new_sources", lambda: extract_new_data_sources(dry_run=args.dry_run, args=args)))
    else:
        manifest["steps"]["extract_new_sources"] = {"skipped": True}
    if not args.skip_classify:
        step_plan.append(("classify", lambda: classify_and_adjust(dry_run=args.dry_run, cpu_fraction=args.cpu_fraction)))
    else:
        manifest["steps"]["classify"] = {"skipped": True}
    if not args.skip_questions or not args.skip_betankande or not args.skip_ip:
        step_plan.append(("classify_new_sources", lambda: classify_new_data_sources(dry_run=args.dry_run, cpu_fraction=args.cpu_fraction, args=args)))
    else:
        manifest["steps"]["classify_new_sources"] = {"skipped": True}
    if not args.skip_prop_classify:
        step_plan.append(("link_prop_bet", lambda: link_prop_bet(dry_run=args.dry_run)))
        step_plan.append(("merge_prop_bet", lambda: merge_prop_bet(dry_run=args.dry_run)))
    else:
        manifest["steps"]["link_prop_bet"] = {"skipped": True}
        manifest["steps"]["merge_prop_bet"] = {"skipped": True}
    if not args.skip_analysis:
        step_plan.append(("analysis", lambda: rebuild_analysis(dry_run=args.dry_run, cpu_fraction=args.cpu_fraction)))
    else:
        manifest["steps"]["analysis"] = {"skipped": True}
    if not args.skip_figures:
        step_plan.append(("figures", lambda: regenerate_figures(dry_run=args.dry_run, cpu_fraction=args.cpu_fraction)))
    else:
        manifest["steps"]["figures"] = {"skipped": True}
    if not args.skip_manuscript:
        step_plan.append(("manuscript", lambda: render_manuscript(dry_run=args.dry_run)))
    else:
        manifest["steps"]["manuscript"] = {"skipped": True}

    total_steps = len(step_plan)
    logger.info("Pipeline plan: {} steps", total_steps)

    try:
        for idx, (name, fn) in enumerate(step_plan, start=1):
            logger.info("=== START: {}/{} {} ===", idx, total_steps, name)
            step_start = _time.time()
            manifest["steps"][name] = fn()
            elapsed = _time.time() - step_start
            logger.info("=== DONE: {}/{} {} ({:.1f}s) ===", idx, total_steps, name, elapsed)
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
        f"---\n_agent_frontmatter:\n  id: pipeline-update-{ts}\n  purpose: Pipeline update run manifest and summary\n  steward: analysis\n  edit_policy: generated_do_not_edit\n  generator: scripts/update_pipeline.py\n  version: 1.1.0\n  run_ts: {ts}\n  dry_run: {args.dry_run}\n---\n\n# Pipeline Update {ts}\n\nSee manifest: `{manifest_path}`\n",
        encoding="utf-8",
    )
    logger.info("Summary written to {}", summary_path)

    logger.info("=== Pipeline update completed at {} ===", manifest["completed_at"])


if __name__ == "__main__":
    main()
