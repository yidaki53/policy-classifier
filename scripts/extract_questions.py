#!/usr/bin/env python3
"""Extract skriftliga frågor (written questions) JSON ZIPs into compressed Parquet files.

Reads the Riksdagen JSON structure from bulk dataset ZIPs and extracts key
fields: dok_id, datum, titel, parti (from intressent), doktext/sammanfattning.
Outputs to data/parquet/questions.parquet.

Usage:
    uv run python scripts/extract_questions.py --src data/bulk_datasets --out data/parquet/questions.parquet
"""

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd


_VALID_PARTIES = {"S", "M", "C", "L", "KD", "V", "MP", "SD", "fi"}
_PARTY_MAP = {"s": "S", "m": "M", "c": "C", "v": "V", "mp": "MP", "kd": "KD", "fp": "L", "fi": "FI"}


def _norm_party(p: str | None) -> str | None:
    if p is None:
        return None
    if isinstance(p, float) and pd.isna(p):
        return None
    p = str(p).strip().lower()
    if not p:
        return None
    return _PARTY_MAP.get(p, p.upper() if p.upper() in _VALID_PARTIES else None)


def _parse_party_from_intressent(d: dict) -> str | None:
    """Extract party from the intressent (speaker/author) list."""
    intressent = d.get("intressent") or d.get("intressenter") or d.get("dokumentstatus", {}).get("intressent")
    if isinstance(intressent, list):
        for i in intressent:
            parti = i.get("parti") or i.get("part")
            if parti:
                return _norm_party(parti)
    elif isinstance(intressent, dict):
        parti = intressent.get("parti") or intressent.get("part")
        if parti:
            return _norm_party(parti)
    return None


def _parse_party_from_undertitel(undertitel: str) -> str | None:
    """Fallback: parse party from subtitle like 'av Rebecka Le Moine m.fl. (MP)'."""
    if not undertitel:
        return None
    m = re.search(r'\(([A-Z]+)\)\s*$', undertitel)
    if m:
        return _norm_party(m.group(1))
    return None


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

            # Unwrap dokumentstatus wrapper if present
            inner = data
            if isinstance(data, dict) and "dokumentstatus" in data:
                ds = data["dokumentstatus"]
                if isinstance(ds, dict):
                    inner = ds.get("dokument") or ds

            if not isinstance(inner, dict):
                docs = data if isinstance(data, list) else [data]
                inner = docs[0] if docs else {}

            def _safe(v):
                if v is None:
                    return ""
                if isinstance(v, str):
                    return v.strip()
                return str(v).strip()

            dok_id = inner.get("dok_id") or inner.get("id")
            if not dok_id:
                continue

            titel = inner.get("titel") or ""
            undertitel = inner.get("undertitel") or ""
            text = inner.get("doktext") or inner.get("text") or inner.get("sammanfattning") or undertitel or ""
            datum = inner.get("datum") or inner.get("dok_datum") or None

            rows.append({
                "id": _safe(dok_id),
                "title": _safe(titel),
                "text": _safe(text),
                "date": _safe(datum) if datum else None,
                "party": party,
                "doc_type": "fr",
            })
    return pd.DataFrame(rows)


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().replace({"nan": None, "": None, "null": None})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    return df


def extract_all(src_dir: str, out_path: str, force: bool = False):
    src = Path(src_dir)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    zips = sorted(src.glob("fr-*.json.zip"))
    print(f"Found {len(zips)} fr ZIP files", file=sys.stderr)

    if not zips:
        print("No fr ZIP files found, creating empty output", file=sys.stderr)
        pd.DataFrame(columns=["id", "title", "text", "date", "party", "doc_type"]).to_parquet(out, index=False, compression="zstd")
        return

    all_rows = []
    for z in zips:
        print(f"READ {z.name} ...", file=sys.stderr)
        try:
            df = _read_json_from_zip(z)
            if not df.empty:
                all_rows.append(df)
                print(f"  {len(df)} rows from {z.name}", file=sys.stderr)
        except Exception as e:
            print(f"  FAILED {z.name}: {e}", file=sys.stderr)

    if not all_rows:
        print("No rows extracted, creating empty output", file=sys.stderr)
        pd.DataFrame(columns=["id", "title", "text", "date", "party", "doc_type"]).to_parquet(out, index=False, compression="zstd")
        return

    combined = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=["id"], keep="first")
    combined = _clean_df(combined)

    existing_rows = 0
    if out.exists() and not force:
        try:
            existing = pd.read_parquet(out)
            existing_rows = len(existing)
            combined = pd.concat([existing, combined], ignore_index=True).drop_duplicates(subset=["id"], keep="first")
        except Exception:
            pass

    combined.to_parquet(out, index=False, compression="zstd")
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"Wrote {len(combined)} total rows ({size_mb:.1f} MB) -> {out}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Extract skriftliga frågor JSON ZIPs to Parquet")
    parser.add_argument("--src", default="data/bulk_datasets", help="Directory with fr-*.json.zip files")
    parser.add_argument("--out", default="data/parquet/questions.parquet", help="Output Parquet file")
    parser.add_argument("--force", action="store_true", help="Overwrite existing Parquet file")
    args = parser.parse_args()

    extract_all(args.src, args.out, force=args.force)


if __name__ == "__main__":
    main()