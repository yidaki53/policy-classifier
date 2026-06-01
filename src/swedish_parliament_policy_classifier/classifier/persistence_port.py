"""Persistence ports and adapters for parquet-first classification outputs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Protocol
import uuid

import pandas as pd


class ClassificationWriter(Protocol):
    def write(self, rows: pd.DataFrame) -> int:
        ...


@dataclass
class ParquetClassificationWriter:
    output_path: Path
    lock_timeout_seconds: float = 120.0
    stale_lock_seconds: float = 6 * 60 * 60
    dedupe_keys: tuple[str, ...] = ("speech_id", "category")

    def _acquire_lock(self, lock_path: Path) -> None:
        start = time.monotonic()
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return
            except FileExistsError:
                try:
                    age_seconds = time.time() - lock_path.stat().st_mtime
                    if age_seconds > self.stale_lock_seconds:
                        lock_path.unlink(missing_ok=True)
                        continue
                except Exception:
                    pass

                if (time.monotonic() - start) > self.lock_timeout_seconds:
                    raise TimeoutError(
                        f"Timed out waiting for parquet write lock: {lock_path}"
                    )
                time.sleep(0.1)

    def _atomic_write(self, df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.output_path.with_suffix(
            f"{self.output_path.suffix}.{uuid.uuid4().hex}.tmp"
        )
        try:
            df.to_parquet(temp_path, index=False, compression="zstd")
            os.replace(temp_path, self.output_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def write(self, rows: pd.DataFrame) -> int:
        lock_path = self.output_path.with_suffix(f"{self.output_path.suffix}.lock")
        self._acquire_lock(lock_path)
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.output_path.exists():
                prev = pd.read_parquet(self.output_path)
                merged = pd.concat([prev, rows], ignore_index=True)
            else:
                merged = rows.copy()

            dedupe_cols = [c for c in self.dedupe_keys if c in merged.columns]
            if dedupe_cols:
                merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")

            self._atomic_write(merged)
            return len(merged)
        finally:
            lock_path.unlink(missing_ok=True)
