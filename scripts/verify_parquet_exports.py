#!/usr/bin/env python3
"""Verify Parquet exports match SQLite tables.

This script computes a deterministic, order-independent checksum for each
table in the given SQLite database by hashing a canonical JSON representation
of each row. It does the same for the corresponding Parquet file(s) and
compares the results. If all tables match, the script moves the DB file to a
timestamped backup (does not permanently delete).

Why this exists
--------------
Byte-for-byte parity between the canonical row serializations of the SQLite
dump and Parquet exports is required when we want to safely remove the large
SQLite file from disk (to recover space) while retaining a machine-readable
archive of the same data in Parquet form. Because different IO layers (SQLite
cursor -> Python types vs Parquet -> pandas) can serialize identical logical
rows differently (column order, pandas NA vs Python None, index-columns),
this script canonicalizes rows deterministically and checksums them in an
order-independent way.

Usage:
    python3 scripts/verify_parquet_exports.py --db data/swedish_parliament.db \
            --parquet-dir data/parquet --temp-dir /tmp/parquet_verify

Notes:
- Requires Python packages: pyarrow (or pandas+pyarrow) and sqlite3 (stdlib).
- Uses external `sort` and `sha256sum` for scalable, disk-backed comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from glob import glob
from typing import Iterable, List


def canonical_row_hash(row: List, enc: str = "utf-8") -> str:
    # Represent the row as a JSON list with stable separators and str-fallback
    s = json.dumps(list(row), ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(s.encode(enc)).hexdigest()


def iter_db_row_hashes(conn: sqlite3.Connection, table: str, out_fpath: str):
    cur = conn.cursor()
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info('{table}')")]
    q = f"SELECT {', '.join([f'"{c}"' for c in cols])} FROM \"{table}\""
    cur.execute(q)
    batch = cur.fetchmany(10000)
    written = 0
    with open(out_fpath, "w", encoding="utf-8") as fh:
        while batch:
            for row in batch:
                fh.write(canonical_row_hash(row) + "\n")
                written += 1
            batch = cur.fetchmany(10000)
    return written


def iter_parquet_row_hashes(parquet_paths: List[str], cols: List[str], out_fpath: str):
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyarrow is required to read Parquet files") from exc

    written = 0
    with open(out_fpath, "w", encoding="utf-8") as fh:
        for p in parquet_paths:
            pf = pq.ParquetFile(p)
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg, columns=cols)

                # Convert to pandas DataFrame to leverage consistent iteration
                # semantics. Parquet files contain metadata that pandas may use
                # to reconstruct an index column (or to reorder columns). That
                # behavior leads to two canonicalization pitfalls:
                #  1. Index columns can be omitted from row iteration (if not
                #     reset) or included twice (if the index is also a column).
                #  2. pandas represents missing values as `pd.NA` / `np.nan`
                #     which JSON-serializes differently from Python `None`.
                # To produce hashes comparable to SQLite's `cursor.fetchall()`
                # (which uses Python `None` for NULLs and preserves PRAGMA
                # column order), we:
                #  - reset the pandas index so any index columns become regular
                #    columns,
                #  - reindex columns to match the DB `cols` ordering,
                #  - fill missing columns with `None`,
                #  - normalize pandas NA / numpy.nan to Python `None`.
                df = table.to_pandas()

                try:
                    import pandas as _pd
                except Exception:
                    _pd = None

                # If Parquet wrote index metadata, reset it so index values
                # are included as normal columns rather than hidden.
                try:
                    if hasattr(df, "index") and df.index is not None and df.index.name is not None:
                        df = df.reset_index()
                except Exception:
                    # If resetting fails, continue - we'll still try to match
                    # as best-effort below.
                    pass

                # Ensure all DB columns exist in the DataFrame; add missing
                # columns as None so canonical JSON lines have the same shape.
                missing_cols = [c for c in cols if c not in df.columns]
                for c in missing_cols:
                    df[c] = None

                # Reorder DataFrame columns to match DB PRAGMA column order.
                try:
                    df = df.reindex(columns=cols)
                except Exception:
                    # If reindexing fails for any reason, fall back to the
                    # intersection order to avoid crashing the verification.
                    available = [c for c in cols if c in df.columns]
                    df = df[available]

                # Normalize missing/pandas-na values to Python None so the JSON
                # canonicalization matches SQLite's Python-typed rows.
                if _pd is not None:
                    df = df.where(_pd.notna(df), None)
                else:
                    # Best-effort: replace common sentinel values
                    df = df.where(df.notna(), None)

                for tup in df.itertuples(index=False, name=None):
                    fh.write(canonical_row_hash(tup) + "\n")
                    written += 1
    return written


def sorted_sha256_of_file(fpath: str) -> str:
    # sort file and pipe to sha256sum (uses disk-based sort)
    sorted_path = fpath + ".sorted"
    with open(sorted_path, "wb") as out:
        # use system sort for robustness on large files
        p1 = subprocess.Popen(["sort", fpath], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["sha256sum"], stdin=p1.stdout, stdout=out)
        p1.stdout.close()
        p2.communicate()
    # read the sha256 hex (first token)
    with open(sorted_path, "rb") as fh:
        first = fh.read().decode("utf-8", errors="ignore").strip().split()[0]
    os.remove(sorted_path)
    return first


def find_parquet_for_table(parquet_dir: str, table: str) -> List[str]:
    candidates = glob(os.path.join(parquet_dir, f"{table}*.parquet*"))
    return sorted(candidates)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--parquet-dir", required=True)
    p.add_argument("--temp-dir", default=tempfile.gettempdir())
    args = p.parse_args(argv)

    db_path = args.db
    parquet_dir = args.parquet_dir
    temp_dir = args.temp_dir

    os.makedirs(temp_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)

    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")]
    if not tables:
        print("No tables found in DB.")
        return 2

    all_ok = True
    for table in tables:
        print(f"\nChecking table: {table}")
        parquet_paths = find_parquet_for_table(parquet_dir, table)
        if not parquet_paths:
            print(f"  MISSING: no parquet found for table '{table}' in {parquet_dir}")
            all_ok = False
            continue

        # columns in DB
        cur = conn.cursor()
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info('{table}')")]
        if not cols:
            print(f"  WARNING: table '{table}' has no columns")

        db_hashes = os.path.join(temp_dir, f"{table}.db.hashes.txt")
        pq_hashes = os.path.join(temp_dir, f"{table}.pq.hashes.txt")

        print(f"  Exporting DB row hashes to {db_hashes}...")
        db_count = iter_db_row_hashes(conn, table, db_hashes)
        print(f"  DB rows: {db_count}")

        print(f"  Exporting Parquet row hashes to {pq_hashes} (files: {parquet_paths})...")
        try:
            pq_count = iter_parquet_row_hashes(parquet_paths, cols, pq_hashes)
        except Exception as exc:
            print(f"  ERROR reading parquet for {table}: {exc}")
            all_ok = False
            continue
        print(f"  Parquet rows: {pq_count}")

        if db_count != pq_count:
            print(f"  COUNT MISMATCH for {table}: DB {db_count} != Parquet {pq_count}")
            all_ok = False
            continue

        print("  Computing sorted SHA256 of row-hash lists (order-independent)...")
        db_sha = sorted_sha256_of_file(db_hashes)
        pq_sha = sorted_sha256_of_file(pq_hashes)
        if db_sha != pq_sha:
            print(f"  CONTENT MISMATCH for {table}: {db_sha} != {pq_sha}")
            all_ok = False
        else:
            print(f"  OK: {table} (rows={db_count})")

        # Clean up per-table hash files
        try:
            os.remove(db_hashes)
            os.remove(pq_hashes)
        except OSError:
            pass

    conn.close()

    if all_ok:
        # move DB to backup (do not delete permanently)
        ts = time.strftime("%Y%m%dT%H%M%S")
        backup_path = db_path + f".removed_after_verify.{ts}.bak"
        print(f"\nAll tables match. Moving DB to backup: {backup_path}")
        shutil.move(db_path, backup_path)
        print("DB moved. You can remove the backup when satisfied.")
        return 0
    else:
        print("\nSome tables failed verification. DB will not be removed.")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
