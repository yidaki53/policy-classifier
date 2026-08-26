"""Contracts for analysis outputs used by visualization/manuscript layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd


@dataclass(frozen=True)
class PublicationContractBundle:
    """Canonical publication bundle for action-first analysis outputs."""

    study_specification: StudySpecification
    action_evidence: pd.DataFrame
    party_position: pd.DataFrame
    say_do: pd.DataFrame
    evaluation: pd.DataFrame

    def validate(self) -> None:
        required = {
            "action_evidence": ["party", "decision"],
            "party_position": ["party", "position"],
            "say_do": ["party", "transition"],
            "evaluation": ["metric", "value"],
        }
        for name, columns in required.items():
            frame = getattr(self, name)
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise ValueError(f"{name} missing required columns: {missing}")


@dataclass(frozen=True)
class StudySpecification:
    """Frozen publication choices shared by analysis and presentation layers."""

    data_cutoff: str | date | datetime
    schema_version: str = "1.0.0"
    primary_window_months: int = 24
    party_codes: tuple[str, ...] = ("C", "KD", "L", "M", "MP", "S", "SD", "V")
    category_order: tuple[str, ...] = (
        "far_left",
        "left",
        "centre_left",
        "centre",
        "centre_right",
        "right",
        "far_right",
    )

    def __post_init__(self) -> None:
        cutoff = pd.Timestamp(self.data_cutoff)
        if pd.isna(cutoff):
            raise ValueError("data_cutoff must be a valid date or timestamp")
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        if self.primary_window_months < 1:
            raise ValueError("primary_window_months must be positive")
        if len(set(self.party_codes)) != len(self.party_codes):
            raise ValueError("party_codes must be unique")
        if len(self.category_order) < 2 or len(set(self.category_order)) != len(self.category_order):
            raise ValueError("category_order must contain at least two unique categories")
        object.__setattr__(self, "data_cutoff", cutoff.to_pydatetime())

    @property
    def primary_start_date(self) -> date:
        cutoff = pd.Timestamp(self.data_cutoff)
        current_month_start = cutoff.normalize().replace(day=1)
        start = current_month_start - pd.DateOffset(months=self.primary_window_months)
        return start.date()

    @property
    def primary_end_date(self) -> date:
        cutoff = pd.Timestamp(self.data_cutoff)
        current_month_start = cutoff.normalize().replace(day=1)
        return (current_month_start - pd.Timedelta(days=1)).date()

    def to_dict(self) -> dict[str, object]:
        cutoff = pd.Timestamp(self.data_cutoff)
        return {
            "schema_version": self.schema_version,
            "data_cutoff_utc": cutoff.isoformat().replace("+00:00", "Z"),
            "primary_window_months": self.primary_window_months,
            "primary_start_date": self.primary_start_date.isoformat(),
            "primary_end_date": self.primary_end_date.isoformat(),
            "party_codes": list(self.party_codes),
            "category_order": list(self.category_order),
        }


@dataclass
class AnalysisResultBundle:
    party_profiles: pd.DataFrame
    ideological_gap: pd.DataFrame
    fulfillment_summary: pd.DataFrame

    def validate(self) -> None:
        required = {
            "party_profiles": ["party", "category", "weight"],
            "ideological_gap": ["party", "comparison", "js_distance"],
            "fulfillment_summary": ["party"],
        }
        frames = {
            "party_profiles": self.party_profiles,
            "ideological_gap": self.ideological_gap,
            "fulfillment_summary": self.fulfillment_summary,
        }
        for name, cols in required.items():
            missing = [c for c in cols if c not in frames[name].columns]
            if missing:
                raise ValueError(f"{name} missing required columns: {missing}")
