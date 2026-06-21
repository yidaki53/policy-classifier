#!/usr/bin/env python3
"""Download votering (voting record) CSV ZIPs from data.riksdagen.se.

Available from riksmöte 1993/94 onward.
URL pattern: https://data.riksdagen.se/dataset/votering/votering-YYYYMM.csv.zip

Uses the shared download utility for atomic writes and validation.
"""
import argparse
import sys
from pathlib import Path

from _download_utils import DownloadStatus, download_file, get_shared_session

# Riksmöte years from 1993/94 to 2025/26
# The 1999/2000 period uses the full 4-digit year (19992000) on the server.
RISMOTE_YEARS = [
    ("1993", "94"), ("1994", "95"), ("1995", "96"), ("1996", "97"), ("1997", "98"),
    ("1998", "99"), ("1999", "2000"), ("2000", "01"), ("2001", "02"), ("2002", "03"),
    ("2003", "04"), ("2004", "05"), ("2005", "06"), ("2006", "07"), ("2007", "08"),
    ("2008", "09"), ("2009", "10"), ("2010", "11"), ("2011", "12"), ("2012", "13"),
    ("2013", "14"), ("2014", "15"), ("2015", "16"), ("2016", "17"), ("2017", "18"),
    ("2018", "19"), ("2019", "20"), ("2020", "21"), ("2021", "22"), ("2022", "23"),
    ("2023", "24"), ("2024", "25"), ("2025", "26"),
]

BASE_URL = "https://data.riksdagen.se/dataset/votering"


def download_all(
    out_dir: str = "data/votering",
    dry_run: bool = False,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if dry_run:
        for y1, y2 in RISMOTE_YEARS:
            filename = f"votering-{y1}{y2}.csv.zip"
            print(f"DRY-RUN {BASE_URL}/{filename} -> {out_path / filename}")
        return True

    session = get_shared_session()
    counts = {s: 0 for s in DownloadStatus}

    for y1, y2 in RISMOTE_YEARS:
        filename = f"votering-{y1}{y2}.csv.zip"
        url = f"{BASE_URL}/{filename}"
        dest = out_path / filename

        if dest.exists() and dest.stat().st_size > 100_000:
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
    parser = argparse.ArgumentParser(description="Download Riksdag votering bulk datasets")
    parser.add_argument("--out", default="data/votering", help="Download directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ok = download_all(out_dir=args.out, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
