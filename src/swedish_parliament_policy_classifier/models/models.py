"""Pydantic models used across the scaffold (package-local implementation)."""

from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from datetime import datetime, date


class RawMotion(BaseModel):
    id: str
    raw: Dict[str, Any]
    retrieved_at: Optional[datetime] = None


class NormalizedMotion(BaseModel):
    id: str
    title: Optional[str] = None
    text: str
    date: Optional[date] = None
    party: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CategoryDef(BaseModel):
    name: str
    definition: Optional[str] = None
    keywords: List[str] = []
    regexes: List[str] = []


class ClassificationResult(BaseModel):
    motion_id: str
    category: str
    raw_score: float
    normalized_weight: float
    matched_rules: List[str]
    classifier_version: str
    created_at: datetime


class PartyProfile(BaseModel):
    party: str
    totals: Dict[str, float]
    updated_at: datetime


__all__ = ["RawMotion", "NormalizedMotion", "CategoryDef", "ClassificationResult", "PartyProfile"]
