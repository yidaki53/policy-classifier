"""Tests for the streaming, memory-bounded classification path."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "classify.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("classify_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_raw(path: Path, n: int) -> None:
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"m{i}",
                "json": json.dumps(
                    {
                        "title": f"Title {i}",
                        "text": "Body text " * 20,
                        "date": "2020-01-01",
                        "party": "S",
                    },
                    ensure_ascii=False,
                ),
            }
        )
    pl.DataFrame(rows).write_parquet(path)


def _write_norm(path: Path, ids) -> None:
    pl.DataFrame(
        [{"id": i, "title": "t", "text": "body " * 40} for i in ids]
    ).write_parquet(path)


def _row(motion_id: str, version: str) -> dict:
    return {
        "motion_id": motion_id,
        "category": "left",
        "raw_score": 0.1,
        "normalized_weight": 0.5,
        "matched_rules": "[]",
        "classifier_version": version,
        "created_at": "2020-01-01T00:00:00+00:00",
    }


def test_iter_normalized_batches_respects_limit(tmp_path):
    module = _load_module()
    raw = tmp_path / "raw.parquet"
    _write_raw(raw, 50)

    batches = list(module._iter_normalized_batches(raw, limit=7))
    assert sum(len(b) for b in batches) == 7


def test_iter_normalized_batches_yields_all_rows(tmp_path):
    module = _load_module()
    raw = tmp_path / "raw.parquet"
    _write_raw(raw, 25)

    assert sum(len(b) for b in module._iter_normalized_batches(raw)) == 25


def test_normalize_writes_expected_columns(tmp_path):
    module = _load_module()
    raw = tmp_path / "raw.parquet"
    out = tmp_path / "norm.parquet"
    _write_raw(raw, 10)

    assert module._normalize_raw_to_parquet(raw, out) == 10
    frame = pl.read_parquet(out)
    assert frame.columns == module.NORMALIZED_COLUMNS
    assert frame.height == 10


def test_normalize_is_idempotent(tmp_path):
    """Re-running must not duplicate rows."""
    module = _load_module()
    raw = tmp_path / "raw.parquet"
    out = tmp_path / "norm.parquet"
    _write_raw(raw, 12)

    module._normalize_raw_to_parquet(raw, out)
    assert module._normalize_raw_to_parquet(raw, out) == 12
    assert pl.read_parquet(out).height == 12


def test_normalize_preserves_existing_rows(tmp_path):
    """Existing normalized rows must survive a later normalize run."""
    module = _load_module()
    out = tmp_path / "norm.parquet"
    pl.DataFrame(
        [
            {
                "id": "existing-1",
                "title": "kept",
                "text": "kept body",
                "date": "2019-01-01",
                "party": "M",
                "doc_type": "motion",
                "metadata": "{}",
            }
        ],
        schema={c: pl.Utf8 for c in module.NORMALIZED_COLUMNS},
    ).write_parquet(out)

    raw = tmp_path / "raw.parquet"
    _write_raw(raw, 5)

    module._normalize_raw_to_parquet(raw, out)
    assert "existing-1" in set(pl.read_parquet(out)["id"].to_list())
def test_classify_appends_to_existing_output(tmp_path):
    """Prior classifier versions must be preserved when re-classifying."""
    module = _load_module()

    normalized = tmp_path / "norm.parquet"
    _write_norm(normalized, ["m1", "m2"])

    out = tmp_path / "classifications.parquet"
    pq.write_table(pa.Table.from_pylist([_row("m1", "old-version")]), out)

    module.classify_parquet(
        normalized, out, reclassify=True, flush_every=1, limit=2
    )

    assert "old-version" in set(pl.read_parquet(out)["classifier_version"].to_list())


def test_classify_honours_limit_exactly(tmp_path):
    """--limit must not overshoot to the enclosing batch size."""
    module = _load_module()

    normalized = tmp_path / "norm.parquet"
    _write_norm(normalized, [f"m{i}" for i in range(500)])

    out = tmp_path / "classifications.parquet"
    assert module.classify_parquet(
        normalized, out, reclassify=True, limit=5, flush_every=1
    ) > 0
    assert pl.read_parquet(out)["motion_id"].n_unique() == 5


def test_classify_no_op_leaves_output_untouched(tmp_path):
    """When everything is already classified, the file must not change."""
    module = _load_module()

    normalized = tmp_path / "norm.parquet"
    _write_norm(normalized, ["m1"])

    out = tmp_path / "classifications.parquet"
    pq.write_table(pa.Table.from_pylist([_row("m1", "old")]), out)
    before = pq.ParquetFile(out).metadata.num_rows

    module.classify_parquet(normalized, out, reclassify=False)

    assert pq.ParquetFile(out).metadata.num_rows == before


def test_output_is_readable_after_run(tmp_path):
    """The staged file must be renamed into place, not left partial."""
    module = _load_module()

    normalized = tmp_path / "norm.parquet"
    _write_norm(normalized, ["m1"])
    out = tmp_path / "classifications.parquet"

    module.classify_parquet(normalized, out, reclassify=True, flush_every=1)

    assert out.exists()
    assert not out.with_suffix(".partial.parquet").exists()
    assert pq.ParquetFile(out).metadata.num_rows > 0
