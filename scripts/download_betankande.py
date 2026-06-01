#!/usr/bin/env python3
"""Download Riksdag betänkande (committee report) bulk dataset ZIP archives.

Uses the same pattern as download_bulk_datasets.py but for 'bet' doktyp.

Usage:
    uv run python scripts/download_betankande.py
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import requests

BASE_URL = "https://data.riksdagen.se/dataset/dokument"

PERIODS = [
    "2022-2025",
    "2018-2021",
    "2014-2017",
    "2010-2013",
    "2006-2009",
    "2002-2005",
    "1998-2001",
    "1990-1997",
    "1980-1989",
    "1971-1979",
]

DOKTYPS = ["bet"]

def download_file(url: str, dest: Path, timeout: int = 120, retries: int = 3):
    """Download a single file with resume support and retries."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        local_size = dest.stat().st_size
        headers = {"Range": f"bytes={local_size}-"}
    else:
        local_size = 0
        headers = {}

    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
            resp.raise_for_status()

            mode = "ab" if local_size > 0 and resp.status_code == 206 else "wb"
            with open(dest, mode) as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True

        except Exception as e:
            print(f"  Download attempt {attempt + 1}/{retries + 1} failed: {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(5 * (attempt + 1))
            else:
                return False

    return False


def download_all(
    out_dir: str = "data/bulk_datasets",
    formats: List[str] = None,
    dry_run: bool = False,
):
    if formats is None:
        formats = ["json"]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for doktyp in DOKTYPS:
        for period in PERIODS:
            for fmt in formats:
                filename = f"{doktyp}-{period}.{fmt}.zip"
                url = f"{BASE_URL}/{filename}"
                dest = out_path / filename

                if dest.exists() and dest.stat().st_size > 1000:
                    print(f"SKIP {filename} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
                    total_skipped += 1
                    continue

                if dry_run:
                    print(f"DRY-RUN {url} -> {dest}")
                    continue

                print(f"DOWNLOAD {url} -> {dest}")
                ok = download_file(url, dest)
                if ok:
                    size_mb = dest.stat().st_size / 1024 / 1024
                    print(f"  OK ({size_mb:.1f} MB)")
                    total_downloaded += 1
                else:
                    print(f"  FAILED")
                    total_failed += 1
                    if dest.exists():
                        dest.unlink()
                time.sleep(5)

    print(f"\nSummary: downloaded={total_downloaded}, skipped={total_skipped}, failed={total_failed}")
    return total_failed == 0


def main():
    parser = argparse.ArgumentParser(description="Download Riksdag betänkande bulk datasets")
    parser.add_argument("--out", default="data/bulk_datasets", help="Download directory")
    parser.add_argument("--formats", nargs="+", default=["json"], help="File formats to download")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ok = download_all(out_dir=args.out, formats=args.formats, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
