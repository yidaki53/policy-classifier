#!/usr/bin/env python3
"""Extract motion signatories from raw bulk motion ZIPs into parquet.

Reads `mot-*.json.zip` under `data/bulk_datasets` and writes a flattened
signatory table to `data/parquet/motion_signatories.parquet`.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd


def _iter_intressent_rows(raw: object) -> list[dict]:
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def _extract_rows_from_doc(doc: dict) -> list[dict]:
    ds = doc.get("dokumentstatus") if isinstance(doc, dict) else None
    if not isinstance(ds, dict):
        return []

    d = ds.get("dokument")
    motion_id = None
    if isinstance(d, dict):
        motion_id = d.get("dok_id") or d.get("id")
    if not motion_id:
        return []

    rm = d.get("rm") if isinstance(d, dict) else None
    di = ds.get("dokintressent")
    if not isinstance(di, dict):
        return []

    raw_intressent = di.get("intressent")
    rows = []
    for row in _iter_intressent_rows(raw_intressent):
        rows.append(
            {
                "motion_id": str(motion_id),
                "rm": str(rm) if rm is not None else "",
                "intressent_id": str(row.get("intressent_id") or "").strip(),
                "signatory_name": str(row.get("namn") or "").strip(),
                "signatory_party": str(row.get("partibet") or "").strip(),
                "signatory_role": str(row.get("roll") or "").strip(),
                "signatory_order": pd.to_numeric(row.get("ordning"), errors="coerce"),
            }
        )

    return rows


def extract_motion_signatories(src_dir: Path, out_path: Path, force: bool = False) -> dict[str, int]:
    if out_path.exists() and not force:
        raise SystemExit(f"Output exists: {out_path}. Use --force to overwrite.")

    zips = sorted(src_dir.glob("mot-*.json.zip"))
    all_rows: list[dict] = []
    files_scanned = 0
    docs_scanned = 0

    for zip_path in zips:
        files_scanned += 1
        with zipfile.ZipFile(zip_path) as zf:
            json_files = [n for n in zf.namelist() if n.lower().endswith(".json")]
            for name in json_files:
                raw = zf.read(name)
                if not raw:
                    continue
                try:
                    data = json.loads(raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    print(f"SKIP corrupt JSON: {zip_path.name}:{name}", file=sys.stderr)
                    continue

                docs = data if isinstance(data, list) else [data]
                for doc in docs:
                    docs_scanned += 1
                    all_rows.extend(_extract_rows_from_doc(doc))

    df = pd.DataFrame(all_rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "motion_id",
                "rm",
                "intressent_id",
                "signatory_name",
                "signatory_party",
                "signatory_role",
                "signatory_order",
            ]
        )
    else:
        df = df.drop_duplicates(
            subset=["motion_id", "intressent_id", "signatory_name", "signatory_role", "signatory_order"],
            keep="first",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="zstd")

    summary = {
        "files_scanned": files_scanned,
        "docs_scanned": docs_scanned,
        "rows": int(len(df)),
        "motions": int(df["motion_id"].nunique()) if len(df) else 0,
        "unique_signatories": int(df["intressent_id"].astype(str).replace("", pd.NA).dropna().nunique()) if len(df) else 0,
        "output": str(out_path),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract motion signatories from raw motion ZIPs")
    parser.add_argument("--src", default="data/bulk_datasets", help="Directory with mot-*.json.zip files")
    parser.add_argument("--out", default="data/parquet/motion_signatories.parquet", help="Output parquet path")
    parser.add_argument("--force", action="store_true", help="Overwrite output if it exists")
    args = parser.parse_args()

    summary = extract_motion_signatories(Path(args.src), Path(args.out), force=args.force)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
