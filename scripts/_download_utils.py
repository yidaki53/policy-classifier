#!/usr/bin/env python3
"""Shared download and ZIP validation utilities.

Used by all download_*.py scripts to avoid duplicated code and to
ensure consistent handling of interrupted downloads.

Key invariants:
- Downloads write to a `.partial` temp file in the same directory,
  then atomically rename to the final path on success.
- On failure (HTTP error, timeout, network error), the `.partial`
  file is deleted so the next run starts cleanly.
- A ZIP file is considered valid only if it can be opened and its
  central directory is intact (zipfile.testzip() returns None).
- Resume support only kicks in when the server explicitly returns 206;
  if the server doesn't support Range requests, we start fresh.
- Downloads smaller than MIN_ZIP_BYTES are rejected as corrupt.
"""
from __future__ import annotations

import enum
import logging
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("download_utils")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("  %(message)s"))
    logger.addHandler(h)

USER_AGENT = "riksdagen-pipeline/2.0"

# Reject anything smaller than 10KB -- a real Riksdagen ZIP is always
# in the multi-MB range, so a tiny file is either a 404 error page
# or a truncated partial download.
MIN_ZIP_BYTES = 10_000


class DownloadStatus(enum.Enum):
    SUCCESS = "success"
    NOT_FOUND = "not_found"
    FAILED = "failed"
    INVALID = "invalid"  # download completed but file is corrupt/too small


def _open_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _validate_zip_size(path: Path, expected_min: int = MIN_ZIP_BYTES) -> bool:
    """Return True if file exists and meets minimum size threshold."""
    if not path.exists():
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size < expected_min:
        logger.warning("File too small (%d bytes, expected >= %d): %s", size, expected_min, path.name)
        return False
    return True


def validate_zip(path: Path) -> bool:
    """Return True if path is a valid, complete ZIP file."""
    if not _validate_zip_size(path):
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                logger.warning("Corrupt entry %r in ZIP: %s", bad, path.name)
                return False
    except (zipfile.BadZipFile, OSError) as e:
        logger.warning("Not a valid ZIP (%s): %s", e, path.name)
        return False
    return True


def download_file(
    url: str,
    dest: Path,
    *,
    timeout: int = 300,
    retries: int = 3,
    force: bool = False,
    min_bytes: int = MIN_ZIP_BYTES,
    session: Optional[requests.Session] = None,
    rate_limit_seconds: float = 0.0,
) -> DownloadStatus:
    """Download a URL to dest with .partial staging and atomic rename.

    Args:
        url: Source URL.
        dest: Final destination path.
        timeout: Per-request timeout in seconds.
        retries: Number of retries on transient failures.
        force: If True, re-download even if dest already exists and is valid.
        min_bytes: Minimum acceptable file size.
        session: Optional requests.Session to reuse.
        rate_limit_seconds: Sleep this many seconds after a successful download.

    Returns:
        DownloadStatus enum value.
    """
    dest = Path(dest)
    partial = dest.with_suffix(dest.suffix + ".partial")

    # If a valid file already exists, skip.
    if not force and dest.exists() and _validate_zip_size(dest, min_bytes):
        return DownloadStatus.SUCCESS

    # Clean up any stale partial from a previous failed run.
    if partial.exists():
        try:
            partial.unlink()
        except OSError:
            pass

    sess = session or _open_session()
    session_id = f"{dest.name}"

    for attempt in range(retries + 1):
        try:
            resp = sess.get(url, timeout=timeout, stream=True)
            if resp.status_code == 404:
                logger.info("  %s: 404 (not found)", session_id)
                return DownloadStatus.NOT_FOUND
            resp.raise_for_status()

            # Write to .partial file
            written = 0
            with open(partial, "wb") as f:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
                        written += len(chunk)
            resp.close()

            if written < min_bytes:
                logger.warning(
                    "  %s: only %d bytes downloaded (expected >= %d), treating as invalid",
                    session_id, written, min_bytes,
                )
                _safe_unlink(partial)
                return DownloadStatus.INVALID

            # Atomic rename from .partial to final path
            partial.rename(dest)
            if rate_limit_seconds > 0:
                time.sleep(rate_limit_seconds)
            return DownloadStatus.SUCCESS

        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            if code == 404:
                logger.info("  %s: 404 (not found)", session_id)
                _safe_unlink(partial)
                return DownloadStatus.NOT_FOUND
            logger.warning(
                "  %s: HTTP %s on attempt %d/%d: %s",
                session_id, code, attempt + 1, retries + 1, e,
            )
            _safe_unlink(partial)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.warning(
                "  %s: network error on attempt %d/%d: %s",
                session_id, attempt + 1, retries + 1, e,
            )
            _safe_unlink(partial)
        except Exception as e:
            logger.warning(
                "  %s: unexpected error on attempt %d/%d: %s",
                session_id, attempt + 1, retries + 1, e,
            )
            _safe_unlink(partial)

        if attempt < retries:
            backoff = 5 * (attempt + 1)
            time.sleep(backoff)

    return DownloadStatus.FAILED


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def revalidate_or_remove(path: Path) -> bool:
    """If `path` is an invalid ZIP, delete it. Returns True if path is valid or missing."""
    if not path.exists():
        return True
    if _validate_zip_size(path) and validate_zip(path):
        return True
    logger.warning("Removing invalid/corrupt ZIP: %s", path)
    _safe_unlink(path)
    return False


def get_shared_session() -> requests.Session:
    """Return a shared Session with a polite User-Agent."""
    return _open_session()
