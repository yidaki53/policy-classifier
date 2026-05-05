"""Validated DB read helpers.

Every public function returns Pydantic-validated model instances rather than
raw sqlite3.Row tuples.  This enforces the type contract at the DB boundary
and surfaces schema drift as validation errors early rather than as downstream
AttributeError or silent wrong-type values.
"""

import json
import sqlite3
from typing import List, Optional

from swedish_parliament_policy_classifier.models.models import (
    ClassificationResult,
    NormalizedMotion,
    RawMotion,
    PartyProfile,
)
from datetime import datetime


# ---------------------------------------------------------------------------
# RawMotion
# ---------------------------------------------------------------------------

def fetch_raw_motion(conn: sqlite3.Connection, motion_id: str) -> Optional[RawMotion]:
    cur = conn.cursor()
    cur.execute("SELECT id, json, retrieved_at FROM raw_motions WHERE id = ?", (motion_id,))
    row = cur.fetchone()
    if row is None:
        return None
    return _row_to_raw_motion(row)


def fetch_all_raw_motions(conn: sqlite3.Connection) -> List[RawMotion]:
    cur = conn.cursor()
    cur.execute("SELECT id, json, retrieved_at FROM raw_motions")
    return [_row_to_raw_motion(r) for r in cur.fetchall()]


def _row_to_raw_motion(row) -> RawMotion:
    raw_json = row["json"] if isinstance(row, sqlite3.Row) else row[1]
    retrieved_at = row["retrieved_at"] if isinstance(row, sqlite3.Row) else row[2]
    motion_id = row["id"] if isinstance(row, sqlite3.Row) else row[0]
    try:
        raw_dict = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
    except (json.JSONDecodeError, TypeError):
        raw_dict = {}
    return RawMotion.model_validate(
        {
            "id": motion_id,
            "raw": raw_dict,
            "retrieved_at": retrieved_at,
        }
    )


# ---------------------------------------------------------------------------
# NormalizedMotion
# ---------------------------------------------------------------------------

def fetch_normalized_motion(conn: sqlite3.Connection, motion_id: str) -> Optional[NormalizedMotion]:
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, text, date, party, metadata FROM normalized_motions WHERE id = ?",
        (motion_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return _row_to_normalized_motion(row)


def fetch_unclassified_motions(conn: sqlite3.Connection) -> List[NormalizedMotion]:
    """Return normalized motions that have not yet been classified."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT nm.id, nm.title, nm.text, nm.date, nm.party, nm.metadata
        FROM normalized_motions nm
        LEFT JOIN classifications c ON nm.id = c.motion_id
        WHERE c.id IS NULL
        """
    )
    return [_row_to_normalized_motion(r) for r in cur.fetchall()]


def fetch_all_normalized_motions(conn: sqlite3.Connection) -> List[NormalizedMotion]:
    cur = conn.cursor()
    cur.execute("SELECT id, title, text, date, party, metadata FROM normalized_motions")
    return [_row_to_normalized_motion(r) for r in cur.fetchall()]


def _row_to_normalized_motion(row) -> NormalizedMotion:
    def _get(r, key, idx):
        return r[key] if isinstance(r, sqlite3.Row) else r[idx]

    metadata_raw = _get(row, "metadata", 5)
    try:
        metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
    except (json.JSONDecodeError, TypeError):
        metadata = {}
    return NormalizedMotion.model_validate(
        {
            "id": _get(row, "id", 0),
            "title": _get(row, "title", 1),
            "text": _get(row, "text", 2) or "",
            "date": _get(row, "date", 3),
            "party": _get(row, "party", 4),
            "metadata": metadata,
        }
    )


# ---------------------------------------------------------------------------
# ClassificationResult
# ---------------------------------------------------------------------------

def fetch_classifications_for_motion(
    conn: sqlite3.Connection, motion_id: str
) -> List[ClassificationResult]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT motion_id, category, raw_score, normalized_weight,
               matched_rules, classifier_version, created_at
        FROM classifications WHERE motion_id = ?
        """,
        (motion_id,),
    )
    return [_row_to_classification(r) for r in cur.fetchall()]


def _row_to_classification(row) -> ClassificationResult:
    def _get(r, key, idx):
        return r[key] if isinstance(r, sqlite3.Row) else r[idx]

    matched_raw = _get(row, "matched_rules", 4)
    try:
        matched = json.loads(matched_raw) if isinstance(matched_raw, str) else (matched_raw or [])
    except (json.JSONDecodeError, TypeError):
        matched = []
    created_raw = _get(row, "created_at", 6)
    created_at = datetime.fromisoformat(created_raw) if created_raw else datetime.utcnow()
    return ClassificationResult.model_validate(
        {
            "motion_id": _get(row, "motion_id", 0),
            "category": _get(row, "category", 1),
            "raw_score": _get(row, "raw_score", 2) or 0.0,
            "normalized_weight": _get(row, "normalized_weight", 3) or 0.0,
            "matched_rules": matched,
            "classifier_version": _get(row, "classifier_version", 5) or "unknown",
            "created_at": created_at,
        }
    )


# ---------------------------------------------------------------------------
# PartyProfile
# ---------------------------------------------------------------------------

def fetch_all_party_profiles(conn: sqlite3.Connection) -> List[PartyProfile]:
    cur = conn.cursor()
    cur.execute("SELECT party, profile_json, last_updated FROM party_profiles")
    results: List[PartyProfile] = []
    for row in cur.fetchall():
        party = row["party"] if isinstance(row, sqlite3.Row) else row[0]
        profile_raw = row["profile_json"] if isinstance(row, sqlite3.Row) else row[1]
        last_updated_raw = row["last_updated"] if isinstance(row, sqlite3.Row) else row[2]
        try:
            totals = json.loads(profile_raw) if isinstance(profile_raw, str) else (profile_raw or {})
        except (json.JSONDecodeError, TypeError):
            totals = {}
        updated_at = (
            datetime.fromisoformat(last_updated_raw) if last_updated_raw else datetime.utcnow()
        )
        results.append(PartyProfile.model_validate({"party": party, "totals": totals, "updated_at": updated_at}))
    return results
