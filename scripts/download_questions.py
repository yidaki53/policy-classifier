#!/usr/bin/env python3
"""Download Riksdag skriftliga frågor (written questions) as JSON ZIP archives.

Written questions (doktyp=fr) are NOT available as bulk dataset ZIPs
from the Riksdagen server. They are only available via:
1. The "Sagt och gjort" CSV dataset (10.65 MB, 2010-present)
2. The live REST API with paginated doktyp=fr requests

This script attempts the bulk URL pattern for completeness; if the
server returns 404, that is expected and not an error.

Usage:
    uv run python scripts/download_questions.py
"""

import argparse
import sys
from pathlib import Path

from _download_utils import DownloadStatus, download_file, get_shared_session

BASE_URL = "https://data.riksdagen.se/dataset/dokument"

# Historical attempt: fr isn't published as bulk, but the script
# documents the periods that *would* be relevant. Each one will
# return NOT_FOUND.
PERIODS = [
    "2022-2025",
    "2018-2021",
    "2014-2017",
    "2010-2013",
]


def download_all(out_dir: str = "data/bulk_datasets", dry_run: bool = False):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if dry_run:
        for period in PERIODS:
            filename = f"fr-{period}.json.zip"
            print(f"DRY-RUN {BASE_URL}/{filename} -> {out_path / filename}")
        return True

    session = get_shared_session()
    counts = {s: 0 for s in DownloadStatus}

    for period in PERIODS:
        filename = f"fr-{period}.json.zip"
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
        elif status == DownloadStatus.NOT_FOUND:
            print(f"  NOT FOUND (fr not published as bulk - use Sagt och gjort or API)")

    print(
        f"\nSummary: downloaded={counts[DownloadStatus.SUCCESS]}, "
        f"not_found={counts[DownloadStatus.NOT_FOUND]} (expected for fr doktyp), "
        f"invalid={counts[DownloadStatus.INVALID]}, "
        f"failed={counts[DownloadStatus.FAILED]}"
    )
    return counts[DownloadStatus.FAILED] == 0


def main():
    parser = argparse.ArgumentParser(description="Download Riksdag skriftliga frågor bulk datasets")
    parser.add_argument("--out", default="data/bulk_datasets")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ok = download_all(out_dir=args.out, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
