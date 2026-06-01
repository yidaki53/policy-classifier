#!/usr/bin/env python3
"""Download votering (voting record) CSV ZIPs from data.riksdagen.se.

Available from riksmöte 1993/94 onward.
URL pattern: https://data.riksdagen.se/votering/YYYY-YY.csv.zip
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import requests

# Riksmöte years from 1993/94 to 2025/26
RISMOTE_YEARS = [
    ("1993", "94"), ("1994", "95"), ("1995", "96"), ("1996", "97"), ("1997", "98"),
    ("1998", "99"), ("1999", "00"), ("2000", "01"), ("2001", "02"), ("2002", "03"),
    ("2003", "04"), ("2004", "05"), ("2005", "06"), ("2006", "07"), ("2007", "08"),
    ("2008", "09"), ("2009", "10"), ("2010", "11"), ("2011", "12"), ("2012", "13"),
    ("2013", "14"), ("2014", "15"), ("2015", "16"), ("2016", "17"), ("2017", "18"),
    ("2018", "19"), ("2019", "20"), ("2020", "21"), ("2021", "22"), ("2022", "23"),
    ("2023", "24"), ("2024", "25"), ("2025", "26"),
]

BASE_URL = "https://data.riksdagen.se/dataset/votering"

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
    out_dir: str = "data/votering",
    dry_run: bool = False,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for y1, y2 in RISMOTE_YEARS:
        filename = f"votering-{y1}{y2}.csv.zip"
        url = f"{BASE_URL}/{filename}"
        dest = out_path / filename

        if dest.exists() and dest.stat().st_size > 1000:
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"SKIP {filename} ({size_mb:.1f} MB)")
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
        time.sleep(2)  # rate limit

    print(f"\nSummary: downloaded={total_downloaded}, skipped={total_skipped}, failed={total_failed}")
    return total_failed == 0


def main():
    parser = argparse.ArgumentParser(description="Download Riksdag votering bulk datasets")
    parser.add_argument("--out", default="data/votering", help="Download directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ok = download_all(out_dir=args.out, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
