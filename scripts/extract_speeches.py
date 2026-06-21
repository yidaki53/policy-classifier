#!/usr/bin/env python3
"""Extract anföranden (speeches) JSON ZIPs into compressed Parquet files.

Reads the Riksdagen JSON structure from bulk dataset ZIPs and extracts key
fields.  Outputs one snappy-compressed Parquet per riksmöte under
data/speeches/parquet/.

Usage:
    uv run python scripts/extract_speeches.py --src data/bulk_datasets --out data/speeches/parquet
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""
    return str(value).strip()


def _read_json_from_zip(zip_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        json_files = [n for n in zf.namelist() if n.lower().endswith(".json")]
        for json_name in json_files:
            raw = zf.read(json_name)
            if len(raw) == 0:
                continue
            try:
                # utf-8-sig handles UTF-8 BOM present in pre-2014 Riksdag archives
                raw_s = raw.decode("utf-8-sig", errors="replace")
                data = json.loads(raw_s)
            except json.JSONDecodeError:
                print(f"    SKIP corrupt JSON: {json_name}", file=sys.stderr)
                continue
            af = data.get("anforande")
            if not isinstance(af, dict):
                continue

            rows.append({
                "dok_id": _safe_str(af.get("dok_id")),
                "anforande_id": _safe_str(af.get("anforande_id")),
                "rm": _safe_str(af.get("dok_rm")),
                "datum": _safe_str(af.get("dok_datum")),
                "talare": _safe_str(af.get("talare")),
                "parti": _safe_str(af.get("parti")) or None,
                "anforandetext": _safe_str(af.get("anforandetext")),
                "avsnittsrubrik": _safe_str(af.get("avsnittsrubrik")) or None,
                "kammaraktivitet": _safe_str(af.get("kammaraktivitet")) or None,
                "rel_dok_id": _safe_str(af.get("rel_dok_id")) or None,
                "intressent_id": _safe_str(af.get("intressent_id")) or None,
            })
    return pd.DataFrame(rows)


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Parse dates
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce", utc=True)

    # Normalise rm: 2013/14 -> 201314, 1972 -> 1972
    if "rm" in df.columns:
        df["rm"] = df["rm"].astype(str).str.replace("/", "", regex=False)

    # Clean party labels
    VALID_PARTIES = {"S", "M", "C", "L", "KD", "V", "MP", "SD"}
    PARTY_MAP = {"s": "S", "m": "M", "c": "C", "v": "V", "mp": "MP", "kd": "KD", "fp": "L"}

    def _norm_party(p):
        if p is None or (isinstance(p, float) and pd.isna(p)):
            return None
        p = _safe_str(p).lower()
        if not p:
            return None
        return PARTY_MAP.get(p, p.upper() if p.upper() in VALID_PARTIES else None)

    if "parti" in df.columns:
        df["parti"] = df["parti"].apply(_norm_party)

    # Drop empty text rows
    if "anforandetext" in df.columns:
        df = df[df["anforandetext"].str.len() > 0].copy()

    return df


def extract_all(
    src_dir: str,
    out_dir: str,
    force: bool = False,
):
    src = Path(src_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    zips = sorted(src.glob("anforande-*.json.zip"))
    print(f"Found {len(zips)} anförande ZIP files", file=sys.stderr)

    for z in zips:
        stem = z.stem  # e.g. anforande-202223
        dest = out / f"{stem}.parquet"

        if dest.exists() and not force:
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"SKIP {stem} ({size_mb:.1f} MB)", file=sys.stderr)
            continue

        print(f"EXTRACT {stem} ...", file=sys.stderr)
        try:
            df = _read_json_from_zip(z)
            df = _clean_df(df)
            df.to_parquet(dest, index=False, compression="zstd")
            size_mb = dest.stat().st_size / 1024 / 1024
            rows = len(df)
            print(f"  OK {rows:,} rows -> {dest} ({size_mb:.1f} MB)", file=sys.stderr)
        except Exception as e:
            print(f"  FAILED {stem}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Extract anförande JSON ZIPs to Parquet")
    parser.add_argument("--src", default="data/bulk_datasets", help="Directory with anforande-*.json.zip files")
    parser.add_argument("--out", default="data/speeches/parquet", help="Output Parquet directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing Parquet files")
    args = parser.parse_args()

    extract_all(args.src, args.out, force=args.force)


if __name__ == "__main__":
    main()