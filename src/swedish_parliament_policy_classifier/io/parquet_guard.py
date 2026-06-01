"""Guards for validating parquet size expectations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


def ensure_min_rows(path: str | Path, min_rows: int, label: str | None = None) -> int:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing parquet file: {p}")
    n = len(pd.read_parquet(p))
    if n < int(min_rows):
        name = label or str(p)
        raise ValueError(f"{name} has {n} rows, expected at least {min_rows}.")
    return n


def ensure_min_rows_many(specs: Iterable[Tuple[str | Path, int, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for path, min_rows, label in specs:
        out[str(label)] = ensure_min_rows(path, min_rows, label)
    return out
