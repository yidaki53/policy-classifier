#!/usr/bin/env python3
"""Extract and clean votering CSV ZIPs into compressed Parquet files.

Handles two CSV formats:
- 1993/94–2001/02: header row present, column order rm,beteckning,punkt,votering_id,...
- 2002/03–present: no header row, column order rm,beteckning,votering_id,punkt,...

Output: one snappy-compressed Parquet per riksmöte in data/votering/parquet/.

Usage:
    uv run python scripts/extract_votering.py --src data/votering --out data/votering/parquet
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

# Same party normalisation as clean_party_labels.py
VALID_PARTIES = {"S", "M", "C", "L", "KD", "V", "MP", "SD"}
PARTY_MAP = {
    "s": "S", "m": "M", "c": "C", "v": "V", "mp": "MP", "kd": "KD",
    "fp": "L", "vpk": "V", "apk": "S", "-": None,
}


def _normalize_party(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    p = p.strip()
    if not p:
        return None
    if p in PARTY_MAP:
        return PARTY_MAP[p]
    up = p.upper()
    if up in VALID_PARTIES:
        return up
    return None


# Pre-2002/03 header names (with BOM stripping)
OLD_HEADER = [
    "rm", "beteckning", "punkt", "votering_id", "namn",
    "intressent_id", "parti", "valkrets", "rost", "avser",
    "banknummer", "kon", "fodd", "datum",
]

# Post-2002/03 positional columns (no header)
NEW_HEADER = [
    "rm", "beteckning", "votering_id", "punkt", "namn",
    "intressent_id", "parti", "valkrets", "rost", "avser",
    "banknummer", "kon", "fodd", "datum",
]


def _detect_format(first_line: str) -> tuple[bool, list[str]]:
    """Return (has_header, column_names) based on first CSV line."""
    first = first_line.lstrip("\ufeff").strip().lower()
    if "votering_id" in first or first.startswith("rm,") and "beteckning" in first:
        return True, OLD_HEADER
    return False, NEW_HEADER


def _read_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    """Read a single ZIP into a DataFrame, handling both formats."""
    dfs: list[pd.DataFrame] = []

    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        for csv_name in csv_names:
            with zf.open(csv_name) as raw:
                bytes_data = raw.read()

            # Peek at first line to detect header
            first_line = bytes_data.split(b"\n")[0].decode("utf-8-sig", errors="replace")
            has_header, columns = _detect_format(first_line)

            if has_header:
                df = pd.read_csv(
                    io.BytesIO(bytes_data),
                    encoding="utf-8-sig",
                    header=0,
                    low_memory=False,
                )
            else:
                df = pd.read_csv(
                    io.BytesIO(bytes_data),
                    encoding="utf-8-sig",
                    names=columns,
                    header=None,
                    low_memory=False,
                )

            # Standardize column names
            for col in NEW_HEADER:
                if col not in df.columns:
                    match = [c for c in df.columns if c.lower() == col.lower()]
                    if match:
                        df.rename(columns={match[0]: col}, inplace=True)

            dfs.append(df)

    if not dfs:
        raise ValueError(f"No CSV found in {zip_path}")

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise strings, clean parties, and drop empty rows."""
    # Keep only relevant columns
    keep = ["rm", "beteckning", "punkt", "votering_id", "namn",
            "intressent_id", "parti", "valkrets", "rost", "avser", "datum"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Strip whitespace and drop fully empty rows
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": None, "": None, "?": None, "null": None})

    df.dropna(subset=["votering_id", "rost"], how="any", inplace=True)

    # Clean party labels
    if "parti" in df.columns:
        df["parti"] = df["parti"].apply(_normalize_party)

    # Parse dates where possible
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce", utc=True)

    # Normalise rm: remove slash, keep 6 digits e.g. 199394, 200203
    if "rm" in df.columns:
        df["rm"] = df["rm"].astype(str).str.replace("/", "", regex=False)

    return df


def extract_all(
    src_dir: str,
    out_dir: str,
    force: bool = False,
):
    src = Path(src_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    zips = sorted(src.glob("*.csv.zip"))
    print(f"Found {len(zips)} ZIP files", file=sys.stderr)

    for z in zips:
        stem = z.stem  # e.g. votering-199394
        dest = out / f"{stem}.parquet"

        if dest.exists() and not force:
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"SKIP {stem} ({size_mb:.1f} MB)", file=sys.stderr)
            continue

        print(f"EXTRACT {stem} ...", file=sys.stderr)
        try:
            df = _read_csv_from_zip(z)
            df = _clean_df(df)
            df.to_parquet(dest, index=False, compression="zstd")
            size_mb = dest.stat().st_size / 1024 / 1024
            rows = len(df)
            print(f"  OK {rows:,} rows -> {dest} ({size_mb:.1f} MB)", file=sys.stderr)
        except Exception as e:
            print(f"  FAILED {stem}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Extract votering CSV ZIPs to Parquet")
    parser.add_argument("--src", default="data/votering", help="Directory with CSV ZIPs")
    parser.add_argument("--out", default="data/votering/parquet", help="Output Parquet directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing Parquet files")
    args = parser.parse_args()

    extract_all(args.src, args.out, force=args.force)


if __name__ == "__main__":
    main()
