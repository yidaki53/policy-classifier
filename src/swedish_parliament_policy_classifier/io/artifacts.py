"""Validated local artifact storage for analysis and publication tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence
import uuid

import pandas as pd


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    relative_path: Path
    required_columns: tuple[str, ...]
    primary_keys: tuple[str, ...] = ()


class ArtifactRepository(Protocol):
    def read(self, spec: ArtifactSpec) -> pd.DataFrame:
        ...

    def write(self, spec: ArtifactSpec, frame: pd.DataFrame) -> Path:
        ...

    def validate(self, spec: ArtifactSpec, frame: pd.DataFrame) -> None:
        ...


@dataclass
class LocalParquetArtifactRepository:
    root: Path

    def path_for(self, spec: ArtifactSpec) -> Path:
        return self.root / spec.relative_path

    def validate(self, spec: ArtifactSpec, frame: pd.DataFrame) -> None:
        missing = [column for column in spec.required_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Artifact {spec.name} is missing required columns: {missing}")
        if spec.primary_keys:
            duplicated = frame.duplicated(list(spec.primary_keys), keep=False)
            if duplicated.any():
                raise ValueError(
                    f"Artifact {spec.name} contains duplicate primary keys: "
                    f"{spec.primary_keys}"
                )

    def read(self, spec: ArtifactSpec) -> pd.DataFrame:
        path = self.path_for(spec)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        self.validate(spec, frame)
        return frame

    def write(self, spec: ArtifactSpec, frame: pd.DataFrame) -> Path:
        self.validate(spec, frame)
        path = self.path_for(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path


__all__ = ["ArtifactRepository", "ArtifactSpec", "LocalParquetArtifactRepository"]