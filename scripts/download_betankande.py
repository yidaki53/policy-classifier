#!/usr/bin/env python3
"""Download Riksdag betänkande (committee report) bulk dataset ZIP archives.

Uses the same pattern as download_bulk_datasets.py but for 'bet' doktyp.

Usage:
    uv run python scripts/download_betankande.py
"""

import argparse
import sys
from pathlib import Path
from typing import List

from _download_utils import DownloadStatus, download_file, get_shared_session

BASE_URL = "https://data.riksdagen.se/dataset/dokument"

# Public dataset periods. The server only publishes periods
# 1971-1979 and onward; 1971-1979 is the earliest available.
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


def download_all(
    out_dir: str = "data/bulk_datasets",
    formats: List[str] = None,
    dry_run: bool = False,
):
    if formats is None:
        formats = ["json"]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if dry_run:
        for doktyp in DOKTYPS:
            for period in PERIODS:
                for fmt in formats:
                    filename = f"{doktyp}-{period}.{fmt}.zip"
                    print(f"DRY-RUN {BASE_URL}/{filename} -> {out_path / filename}")
        return True

    session = get_shared_session()
    counts = {s: 0 for s in DownloadStatus}

    for doktyp in DOKTYPS:
        for period in PERIODS:
            for fmt in formats:
                filename = f"{doktyp}-{period}.{fmt}.zip"
                url = f"{BASE_URL}/{filename}"
                dest = out_path / filename

                if dest.exists() and dest.stat().st_size > 1_000_000:
                    print(f"SKIP {filename} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
                    counts[DownloadStatus.SUCCESS] += 1
                    continue

                print(f"DOWNLOAD {url}")
                status = download_file(url, dest, session=session, rate_limit_seconds=2.0)
                counts[status] += 1
                if status == DownloadStatus.SUCCESS:
                    print(f"  OK ({dest.stat().st_size / 1024 / 1024:.1f} MB)")

    print(
        f"\nSummary: downloaded={counts[DownloadStatus.SUCCESS]}, "
        f"not_found={counts[DownloadStatus.NOT_FOUND]}, "
        f"invalid={counts[DownloadStatus.INVALID]}, "
        f"failed={counts[DownloadStatus.FAILED]}"
    )
    return counts[DownloadStatus.FAILED] == 0


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
