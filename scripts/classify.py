#!/usr/bin/env python3
"""Classify normalized motions and persist results to Parquet.

This script is a Parquet-first replacement for the legacy DB-backed `classify.py`.
It reads `data/parquet/raw_motions.parquet`, builds or updates
`data/parquet/normalized_motions.parquet` and appends classification rows to
`data/parquet/classifications.parquet`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator, Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from swedish_parliament_policy_classifier.exports import load_definitions, classify_motion

# Rows per streaming batch when normalizing raw motions. Bounds peak memory:
# the raw motions file expands to roughly 7 GB in pandas, which does not fit
# alongside the normalized file on a 32 GB machine.
NORMALIZE_BATCH_ROWS = 20_000

# Columns produced by _normalize_row, in output order.
NORMALIZED_COLUMNS = [
    "id",
    "title",
    "text",
    "date",
    "party",
    "doc_type",
    "metadata",
]


def _iter_normalized_batches(
    raw_parquet: str | Path,
    batch_rows: int = NORMALIZE_BATCH_ROWS,
    limit: Optional[int] = None,
) -> Iterator[list[dict]]:
    """Yield normalized records batch by batch, never holding the whole file.

    Uses a Polars lazy scan with the streaming engine so peak memory stays
    proportional to one batch rather than to the full corpus.
    """
    raw_p = Path(raw_parquet)
    if not raw_p.exists():
        return

    cursor = pl.scan_parquet(raw_p).select(["id", "json"])
    if limit is not None:
        cursor = cursor.head(limit)

    seen = 0
    batch: list[dict] = []
    for record in cursor.collect(engine="streaming").iter_rows(named=True):
        mid = str(record.get("id")) if record.get("id") is not None else None
        if mid:
            raw_json = record.get("json")
            try:
                data = (
                    json.loads(raw_json)
                    if isinstance(raw_json, str)
                    else (raw_json or {})
                )
            except Exception:
                data = raw_json or {}
            title = data.get("title") or data.get("rubrik") or ""
            text = data.get("text") or data.get("body") or title or ""
            date = data.get("date") or data.get("datum") or None
            party = data.get("party") or data.get("parti") or None
            doc_type = data.get("doc_type") or data.get("dokumenttyp") or None
            metadata = json.dumps(
                {
                    k: v
                    for k, v in data.items()
                    if k not in ("title", "text", "date", "party")
                },
                ensure_ascii=False,
            )
            batch.append(
                {
                    "id": mid,
                    "title": title,
                    "text": text,
                    "date": date,
                    "party": party,
                    "doc_type": doc_type,
                    "metadata": metadata,
                }
            )
            seen += 1

        if len(batch) >= batch_rows:
            yield batch
            batch = []

        if limit is not None and seen >= limit:
            break

    if batch:
        yield batch


def _row_count(path: Path) -> int:
    """Count rows without loading the file."""
    if not path.exists():
        return 0
    try:
        return (
            pl.scan_parquet(path)
            .select(pl.len())
            .collect(engine="streaming")
            .item()
        )
    except Exception:  # noqa: BLE001 - diagnostics only
        return 0


def _normalize_raw_to_parquet(
    raw_parquet: str | Path,
    normalized_out: str | Path,
    limit: Optional[int] = None,
) -> int:
    """Merge raw motions into the normalized parquet without loading it whole.

    Reads only the existing id column rather than the full 2 GB normalized
    table, and appends batch by batch. Peak memory is proportional to one
    batch instead of the whole corpus.
    """
    raw_p = Path(raw_parquet)
    out_p = Path(normalized_out)
    if not raw_p.exists():
        print("No raw motions parquet found at", raw_p)
        return 0

    known_ids: set[str] = set()
    if out_p.exists():
        try:
            known_ids = set(
                pl.scan_parquet(out_p)
                .select("id")
                .collect(engine="streaming")["id"]
                .cast(pl.Utf8)
                .to_list()
            )
        except Exception as exc:  # noqa: BLE001 - fall back to a full rebuild
            print(f"Could not read existing normalized ids ({exc}); rebuilding")
            known_ids = set()

    out_p.parent.mkdir(parents=True, exist_ok=True)
    staging = out_p.with_suffix(".partial.parquet")
    # New rows go to their own file first. A ParquetWriter truncates the path it
    # opens, so it must never share a path with the reader over existing rows.
    new_rows_path = out_p.with_suffix(".newrows.parquet")
    new_rows_path.unlink(missing_ok=True)
    writer: Optional[pq.ParquetWriter] = None

    try:
        for chunk in _iter_normalized_batches(raw_p, limit=limit):
            fresh = [r for r in chunk if r["id"] not in known_ids]
            if not fresh:
                continue
            known_ids.update(r["id"] for r in fresh)
            table = pa.Table.from_pylist(
                fresh,
                schema=pa.schema([(c, pa.string()) for c in NORMALIZED_COLUMNS]),
            )
            if writer is None:
                writer = pq.ParquetWriter(new_rows_path, table.schema)
            writer.write_table(table)

        if writer is None:
            # Nothing new to add; leave the existing file untouched.
            staging.unlink(missing_ok=True)
            new_rows_path.unlink(missing_ok=True)
            return _row_count(out_p)
        writer.close()
        writer = None

        # Stream existing rows then new rows into the staging file. Both are
        # read from paths the writer does not own, so memory stays flat.
        # Cast to the writer's schema: a Polars-written file reports
        # large_string, while new rows are plain string.
        new_table = pq.read_table(new_rows_path)
        if out_p.exists():
            existing = pq.ParquetFile(out_p)
            target_schema = existing.schema_arrow
            out_writer = pq.ParquetWriter(staging, target_schema)
            for prior in existing.iter_batches(batch_size=5_000):
                prior = pa.Table.from_batches([prior])
                out_writer.write_table(prior)
            new_table = new_table.cast(target_schema)
        else:
            out_writer = pq.ParquetWriter(staging, new_table.schema)
        out_writer.write_table(new_table)
        out_writer.close()
    except Exception as exc:  # noqa: BLE001 - never leave partial output
        print(f"Normalization failed ({exc}); discarding staged output")
        if writer is not None:
            writer.close()
        staging.unlink(missing_ok=True)
        new_rows_path.unlink(missing_ok=True)
        raise

    new_rows_path.unlink(missing_ok=True)
    staging.replace(out_p)
    return _row_count(out_p)


def classify_parquet(
    normalized_parquet: str | Path = "data/parquet/normalized_motions.parquet",
    classifications_out: str | Path = "data/parquet/classifications.parquet",
    limit: Optional[int] = None,
    reclassify: bool = False,
    pause_every: int = 0,
    pause_seconds: float = 0.0,
    flush_every: int = 10_000,
) -> int:
    defs = load_definitions()
    nm_p = Path(normalized_parquet)
    out_p = Path(classifications_out)
    if not nm_p.exists():
        print("No normalized motions found; run ingest first to create normalized_motions.parquet")
        return 0

    # Determine already classified motions (id column only, never the whole file)
    classified_ids = set()
    if out_p.exists() and not reclassify:
        try:
            classified_ids = set(
                pl.scan_parquet(out_p)
                .select("motion_id")
                .collect(engine="streaming")["motion_id"]
                .cast(pl.Utf8)
                .to_list()
            )
        except Exception:  # noqa: BLE001 - fall back to classifying everything
            classified_ids = set()

    # Stream the normalized table batch by batch. Reading it whole costs
    # several GB; iterating batches keeps peak memory near one batch.
    reader = pq.ParquetFile(nm_p)
    if "id" not in reader.schema_arrow.names:
        print("normalized_motions.parquet missing 'id' column")
        return 0

    columns = [
        name for name in ("id", "title", "text") if name in reader.schema_arrow.names
    ]
    out_p.parent.mkdir(parents=True, exist_ok=True)
    staging = out_p.with_suffix(".partial.parquet")
    rows: list[dict] = []
    processed_index = 0
    total_written = 0

    # New rows are buffered to their own file, then merged with existing
    # history at the end. A ParquetWriter truncates the path it opens, so it
    # must never read and write the same file.
    new_rows_path = out_p.with_suffix(".newrows.parquet")
    new_rows_path.unlink(missing_ok=True)
    writer: Optional[pq.ParquetWriter] = None

    def _flush() -> None:
        nonlocal writer, total_written
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(new_rows_path, table.schema)
        writer.write_table(table)
        total_written += table.num_rows
        rows.clear()

    for batch in reader.iter_batches(batch_size=2_000, columns=columns):
        for record in batch.to_pylist():
            mid = str(record.get("id"))
            if not reclassify and mid in classified_ids:
                continue

            processed_index += 1
            title = record.get("title") or ""
            body = record.get("text") or ""
            text = f"{title}\n{body}" if title else body
            try:
                results = classify_motion(motion_id=mid, text=text, categories=defs)
            except Exception as e:  # noqa: BLE001 - keep going on one bad row
                print(f"Failed to classify {mid}: {e}")
                continue

            for rr in results:
                rows.append({
                    "motion_id": rr.motion_id,
                    "category": rr.category,
                    "raw_score": float(rr.raw_score),
                    "normalized_weight": float(rr.normalized_weight),
                    "matched_rules": json.dumps(rr.matched_rules, ensure_ascii=False),
                    "classifier_version": rr.classifier_version,
                    "created_at": rr.created_at.isoformat(),
                })

            if pause_every > 0 and pause_seconds > 0 and processed_index % pause_every == 0:
                time.sleep(pause_seconds)

            if flush_every > 0 and len(rows) >= flush_every:
                _flush()

            # Honour --limit exactly rather than at batch granularity.
            if limit is not None and processed_index >= limit:
                break

        if limit is not None and processed_index >= limit:
            break

    _flush()
    if writer is not None:
        writer.close()
        writer = None

    if total_written == 0:
        staging.unlink(missing_ok=True)
        new_rows_path.unlink(missing_ok=True)
        print("No motions required classification")
        return 0

    # Merge existing history with the new rows, streaming both. Cast to the
    # writer's schema: a Polars-written file reports large_string, while new
    # rows are plain string.
    new_table = pq.read_table(new_rows_path)
    if out_p.exists():
        existing = pq.ParquetFile(out_p)
        target_schema = existing.schema_arrow
        out_writer = pq.ParquetWriter(staging, target_schema)
        for prior in existing.iter_batches(batch_size=5_000):
            prior = pa.Table.from_batches([prior])
            out_writer.write_table(prior)
        new_table = new_table.cast(target_schema)
    else:
        out_writer = pq.ParquetWriter(staging, new_table.schema)
    out_writer.write_table(new_table)
    out_writer.close()

    new_rows_path.unlink(missing_ok=True)
    staging.replace(out_p)
    print(f"Processed {processed_index} motions; wrote {total_written} rows")
    return total_written


def main():
    parser = argparse.ArgumentParser(description="Classify motions and persist to Parquet")
    parser.add_argument("--raw", default="data/parquet/raw_motions.parquet", help="Raw motions parquet (input)")
    parser.add_argument("--normalized-out", default="data/parquet/normalized_motions.parquet", help="Normalized motions parquet (output)")
    parser.add_argument("--classifications-out", default="data/parquet/classifications.parquet", help="Classifications parquet (output)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--reclassify",
        action="store_true",
        help="Reclassify all normalized motions and preserve model-version history",
    )
    parser.add_argument("--pause-every", type=int, default=0, help="Pause after every N motions")
    parser.add_argument("--pause-seconds", type=float, default=0.0, help="Seconds to pause after each pause interval")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10_000,
        help="Flush buffered classifications to parquet every N rows (0 disables)",
    )
    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help=(
            "Classify from the existing normalized parquet without re-reading "
            "raw motions. raw_motions.parquet expands to roughly 7 GB, so "
            "skipping it keeps a re-classification run inside memory."
        ),
    )
    args = parser.parse_args()

    if args.skip_normalize:
        print(
            "Skipping normalization (--skip-normalize); "
            f"classifying from {args.normalized_out}"
        )
    else:
        written_nm = _normalize_raw_to_parquet(args.raw, args.normalized_out)
        print(f"Normalized motions (rows) now: {written_nm}")

    classified = classify_parquet(
        args.normalized_out,
        args.classifications_out,
        limit=args.limit,
        reclassify=args.reclassify,
        pause_every=args.pause_every,
        pause_seconds=args.pause_seconds,
        flush_every=args.flush_every,
    )
    print(f"Appended {classified} classification rows to {args.classifications_out}")


if __name__ == "__main__":
    main()

if False:
    # Graphify hint: classify() calls load_definitions() and score_motion()
    # Anchor the verified loader implementation directly (avoid classifier->scorer indirection)
    from definitions.loader import load_verified_definitions as _hint_load_verified_definitions
    from swedish_parliament_policy_classifier.models import CategoryDef as _hint_CategoryDef
