"""Shared Parquet I/O utilities for the analysis layer.

Single source of truth for:
* Reading Parquet tables with fallback extensions (``.parquet``, ``.parquet.zst``, ``.pq``)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_parquet_table(parquet_dir: Path, stem: str, columns: list[str]) -> pd.DataFrame:
    """Return ``columns`` from the first existing table matching ``stem`` under ``parquet_dir``.

    Tries, in order:
      1. ``{stem}.parquet``
      2. ``{stem}.parquet.zst``
      3. ``{stem}.pq``

    Raises ``FileNotFoundError`` if none are found.
    """
    candidates = [
        parquet_dir / f"{stem}.parquet",
        parquet_dir / f"{stem}.parquet.zst",
        parquet_dir / f"{stem}.pq",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p, columns=columns)
    raise FileNotFoundError(f"Missing parquet table for '{stem}' under {parquet_dir}")