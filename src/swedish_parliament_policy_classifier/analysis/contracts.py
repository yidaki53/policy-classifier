"""Contracts for analysis outputs used by visualization/manuscript layers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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
