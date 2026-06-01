#!/usr/bin/env python3
"""Extract and clean betänkande JSON ZIPs into compressed Parquet files.

Reads the original Riksdagen JSON structure from bulk dataset ZIPs and
extracts key fields including the motion linkage (relaterat_id).  Outputs
one snappy-compressed Parquet per time period under data/betankande/parquet/.

Usage:
    uv run python scripts/extract_betankande.py --src data/bulk_datasets --out data/betankande/parquet
"""

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd


def _read_json_from_zip(zip_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        json_files = [n for n in zf.namelist() if n.lower().endswith(".json")]
        for json_name in json_files:
            raw = zf.read(json_name)
            if len(raw) == 0:
                continue
            try:
                data = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                print(f"    SKIP corrupt JSON: {json_name}", file=sys.stderr)
                continue
            docs = data if isinstance(data, list) else [data]
            for d in docs:
                if not isinstance(d, dict):
                    continue
                ds = d.get("dokumentstatus")
                if not isinstance(ds, dict):
                    continue
                doc = ds.get("dokument")
                if not isinstance(doc, dict):
                    continue

                rm = str(doc.get("rm", "")).strip()
                beteckning = str(doc.get("beteckning", "")).strip()
                dok_id = str(doc.get("dok_id", "")).strip()
                organ = str(doc.get("organ", "")).strip()
                datum = str(doc.get("datum", "")).strip()
                titel = str(doc.get("titel", "")).strip()

                # Related motions: dokreferens (committee report -> motion refs)
                ref_docs: list[str] = []
                dokreferens = ds.get("dokreferens")
                if isinstance(dokreferens, dict):
                    refs = dokreferens.get("referens")
                    if isinstance(refs, dict):
                        refs = [refs]
                    if isinstance(refs, list):
                        for r in refs:
                            if isinstance(r, dict):
                                rid = r.get("ref_dok_id")
                                if rid:
                                    ref_docs.append(str(rid).strip())

                rows.append({
                    "dok_id": dok_id,
                    "rm": rm,
                    "beteckning": beteckning,
                    "organ": organ,
                    "datum": datum,
                    "titel": titel,
                    "ref_dok_ids": json.dumps(ref_docs) if ref_docs else None,
                    "ref_dok_count": len(ref_docs),
                })
    return pd.DataFrame(rows)


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace and replace empty strings with None
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().replace({"nan": None, "": None, "null": None})

    # Parse dates
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce", utc=True)

    # Normalise rm: 2013/14 -> 201314, 1972 -> 1972
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

    zips = sorted(src.glob("bet-*.json.zip"))
    print(f"Found {len(zips)} betänkande ZIP files", file=sys.stderr)

    for z in zips:
        stem = z.stem  # e.g. bet-2022-2025.json
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
    parser = argparse.ArgumentParser(description="Extract betänkande JSON ZIPs to Parquet")
    parser.add_argument("--src", default="data/bulk_datasets", help="Directory with bet-*.json.zip files")
    parser.add_argument("--out", default="data/betankande/parquet", help="Output Parquet directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing Parquet files")
    args = parser.parse_args()

    extract_all(args.src, args.out, force=args.force)


if __name__ == "__main__":
    main()
