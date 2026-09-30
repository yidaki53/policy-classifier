"""Contracts and diagnostics for speech/action linkage artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


LINKAGE_REQUIRED_COLUMNS = ("speech_id", "motion_id")


@dataclass(frozen=True)
class LinkageCoverage:
    candidate_rows: int
    linked_rows: int
    unique_speeches: int
    unique_motions: int
    mean_confidence: float | None

    @property
    def linked_rate(self) -> float:
        if self.candidate_rows == 0:
            return 0.0
        return self.linked_rows / self.candidate_rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_rows": self.candidate_rows,
            "linked_rows": self.linked_rows,
            "linked_rate": self.linked_rate,
            "unique_speeches": self.unique_speeches,
            "unique_motions": self.unique_motions,
            "mean_confidence": self.mean_confidence,
        }


def validate_linkage_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in LINKAGE_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Linkage artifact missing required columns: {missing}")

    for column in LINKAGE_REQUIRED_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"Linkage artifact contains missing identifiers in {column}")


def summarize_linkage(frame: pd.DataFrame, confidence_column: str = "confidence") -> LinkageCoverage:
    validate_linkage_frame(frame)
    confidence: float | None = None
    if confidence_column in frame.columns:
        values = pd.to_numeric(frame[confidence_column], errors="coerce").dropna()
        confidence = float(values.mean()) if not values.empty else None
    linked = frame["speech_id"].astype(str).ne("") & frame["motion_id"].astype(str).ne("")
    return LinkageCoverage(
        candidate_rows=int(len(frame)),
        linked_rows=int(linked.sum()),
        unique_speeches=int(frame.loc[linked, "speech_id"].astype(str).nunique()),
        unique_motions=int(frame.loc[linked, "motion_id"].astype(str).nunique()),
        mean_confidence=confidence,
    )


__all__ = ["LINKAGE_REQUIRED_COLUMNS", "LinkageCoverage", "summarize_linkage", "validate_linkage_frame"]