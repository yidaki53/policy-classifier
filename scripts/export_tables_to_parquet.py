"""Export SQLite tables to Parquet files (chunked) under `data/parquet/`.

Usage:
    python scripts/export_tables_to_parquet.py --db data/swedish_parliament.db --out data/parquet --tables normalized_motions motions

If `--tables` is omitted, the script will list available tables and export all.
"""
import argparse
import sqlite3
from pathlib import Path
import sys
import logging

try:
    import pandas as pd
except ImportError as _err:
    raise RuntimeError("pandas is required for parquet export") from _err

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as _err:
    raise RuntimeError("pyarrow is required for parquet export") from _err

LOG = logging.getLogger(__name__)

# Map table -> timestamp column to use as index
_TABLE_TIMESTAMP_INDEX = {
    "raw_motions": "retrieved_at",
    "normalized_motions": "date",
    "classifications": "created_at",
    "party_profiles": "last_updated",
}


def list_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [r[0] for r in cur.fetchall()]


def _coerce_timestamp(df: "pd.DataFrame", table: str) -> "pd.DataFrame":
    """Convert the table's timestamp column to datetime and set as index."""
    ts_col = _TABLE_TIMESTAMP_INDEX.get(table)
    if ts_col and ts_col in df.columns:
        # Coerce to datetime using mixed format so future ISO dates and timestamps parse
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", format="mixed", utc=True)
        nulls = df[ts_col].isna().sum()
        if nulls:
            LOG.warning("Table %s has %d rows with unparseable %s; dropping them", table, nulls, ts_col)
            df = df.dropna(subset=[ts_col])
        df = df.set_index(ts_col).sort_index()
    return df


def export_table(conn, table: str, outdir: Path, chunksize: int = 100000, compression: str = "snappy"):
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{table}.parquet"

    sql = f'SELECT * FROM "{table}"'

    # Read a sample to build a stable schema so chunked reads produce
    # compatible parquet schema. If the sample is empty, create an
    # empty parquet and return.
    sample_size = min(chunksize, 10000)
    sample_df = pd.read_sql_query(sql + f' LIMIT {sample_size}', conn)
    if sample_df.shape[0] == 0:
        df = pd.read_sql_query(sql + ' LIMIT 0', conn)
        df = _coerce_timestamp(df, table)
        df.to_parquet(out_path, compression=compression)
        LOG.info("Finished export of empty table %s to %s", table, out_path)
        return out_path

    # Determine numeric vs other columns and coerce other columns to pandas
    # 'string' dtype to avoid schema inference differences across chunks.
    num_cols = sample_df.select_dtypes(include=['number']).columns.tolist()
    other_cols = [c for c in sample_df.columns if c not in num_cols]
    for c in other_cols:
        sample_df[c] = sample_df[c].astype('string')

    # Apply timestamp coercion to the sample so the base schema reflects it
    sample_df = _coerce_timestamp(sample_df, table)
    base_schema = pa.Table.from_pandas(sample_df).schema

    writer = None
    written = 0
    for chunk in pd.read_sql_query(sql, conn, chunksize=chunksize):
        # Coerce chunk dtypes to match the sample-derived schema
        for c in other_cols:
            if c in chunk.columns:
                chunk[c] = chunk[c].astype('string')
        for c in num_cols:
            if c in chunk.columns:
                chunk[c] = pd.to_numeric(chunk[c], errors='coerce')

        chunk = _coerce_timestamp(chunk, table)
        table_pa = pa.Table.from_pandas(chunk, schema=base_schema)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), base_schema, compression=compression)
        writer.write_table(table_pa)
        written += len(chunk)
        LOG.info("Wrote %d rows for table %s...", written, table)

    if writer is not None:
        writer.close()
    else:
        # empty table: create an empty parquet file
        df = pd.read_sql_query(sql + ' LIMIT 0', conn)
        df = _coerce_timestamp(df, table)
        df.to_parquet(out_path, compression=compression)

    LOG.info("Finished export of table %s to %s (rows approx: %d)", table, out_path, written)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("data/swedish_parliament.db"))
    parser.add_argument("--out", type=Path, default=Path("data/parquet"))
    parser.add_argument("--tables", nargs="*", help="Tables to export (default: all)")
    parser.add_argument("--chunksize", type=int, default=100000)
    parser.add_argument("--compression", default="zstd")

    args = parser.parse_args()

    if not args.db.exists():
        LOG.error("Database %s not found", args.db)
        sys.exit(2)

    conn = sqlite3.connect(str(args.db))

    tables = args.tables or list_tables(conn)
    if not tables:
        LOG.error("No tables found in database")
        sys.exit(1)

    LOG.info("Exporting tables: %s", ", ".join(tables))
    for t in tables:
        try:
            export_table(conn, t, args.out, chunksize=args.chunksize, compression=args.compression)
        except Exception as e:
            LOG.exception("Failed to export table %s: %s", t, e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
