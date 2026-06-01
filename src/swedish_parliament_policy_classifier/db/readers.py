"""Validated DB read helpers.

Every public function returns Pydantic-validated model instances rather than
raw sqlite3.Row tuples.  This enforces the type contract at the DB boundary
and surfaces schema drift as validation errors early rather than as downstream
AttributeError or silent wrong-type values.
"""

import json
import sqlite3
from typing import List, Optional
from pathlib import Path

from swedish_parliament_policy_classifier.models.models import (
    ClassificationResult,
    NormalizedMotion,
    RawMotion,
    PartyProfile,
)
from swedish_parliament_policy_classifier.io import loader

try:
    import pandas as pd
except Exception:
    pd = None
from datetime import datetime, timezone


def _parquet_path(table: str) -> Path:
    return Path("data") / "parquet" / f"{table}.parquet"


def _has_parquet(table: str) -> bool:
    return _parquet_path(table).exists() and pd is not None


def _load_parquet_as_dicts(table: str) -> list[dict]:
    """Load a parquet file and return rows as dicts (with index reset to column)."""
    path = _parquet_path(table)
    df = loader.load_parquet(path)
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError(f"Expected DataFrame from {path}, got {type(df)}")
    df = df.reset_index()
    return df.to_dict(orient="records")


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
    if _has_parquet("raw_motions"):
        rows = _load_parquet_as_dicts("raw_motions")
        return [_row_to_raw_motion(r) for r in rows]
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
    if _has_parquet("normalized_motions"):
        rows = _load_parquet_as_dicts("normalized_motions")
        for r in rows:
            if r.get("id") == motion_id:
                return _row_to_normalized_motion(r)
        return None
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, text, date, party, metadata, doc_type, full_text_url FROM normalized_motions WHERE id = ?",
        (motion_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return _row_to_normalized_motion(row)


def fetch_unclassified_motions(conn: sqlite3.Connection) -> List[NormalizedMotion]:
    """Return normalized motions that have not yet been classified."""
    # Prefer a parquet export if available to avoid reading huge SQLite DB files.
    parquet_path = Path("data") / "parquet" / "normalized_motions.parquet"
    if parquet_path.exists() and pd is not None:
        # load classifications to determine which motions are already classified
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT motion_id FROM classifications")
        classified = {r[0] for r in cur.fetchall()}
        df = loader.load_parquet(parquet_path)
        if isinstance(df, pd.DataFrame):
            if "id" not in df.columns:
                raise RuntimeError(f"parquet file {parquet_path} missing 'id' column")
            df_un = df[~df["id"].isin(classified)]
            results: List[NormalizedMotion] = []
            for _, row in df_un.iterrows():
                metadata = row.get("metadata")
                try:
                    md = json.loads(metadata) if isinstance(metadata, str) else (metadata or {})
                except Exception:
                    md = {}
                results.append(
                    NormalizedMotion.model_validate(
                        {
                            "id": row.get("id"),
                            "title": row.get("title"),
                            "text": row.get("text") or "",
                            "date": row.get("date"),
                            "party": row.get("party"),
                            "metadata": md,
                            "doc_type": row.get("doc_type"),
                            "full_text_url": row.get("full_text_url"),
                        }
                    )
                )
            return results

    # Fallback to SQLite query
    cur = conn.cursor()
    cur.execute(
        """
        SELECT nm.id, nm.title, nm.text, nm.date, nm.party, nm.metadata, nm.doc_type, nm.full_text_url
        FROM normalized_motions nm
        LEFT JOIN classifications c ON nm.id = c.motion_id
        WHERE c.id IS NULL
        """
    )
    return [_row_to_normalized_motion(r) for r in cur.fetchall()]


def fetch_augmented_gold_label_rows(conn: sqlite3.Connection, split: str = "train"):
    """Return rows for augmented_gold_labels left-joined with normalized_motions.

    Each row is a tuple: (motion_id, category, text, date, doc_type, party)
    This function will prefer a parquet-backed `normalized_motions` export if present.
    """
    parquet_path = Path("data") / "parquet" / "normalized_motions.parquet"
    cur = conn.cursor()
    # Load augmented labels from DB (expected to be small)
    cur.execute(
        "SELECT motion_id, category, text FROM augmented_gold_labels WHERE split = ?",
        (split,),
    )
    aug_rows = cur.fetchall()

    if parquet_path.exists() and pd is not None:
        df_nm = loader.load_parquet(parquet_path)
        if not isinstance(df_nm, pd.DataFrame):
            df_nm = pd.DataFrame(df_nm)
        df_nm = df_nm.set_index("id")
        out = []
        for a in aug_rows:
            motion_id = a[0]
            category = a[1]
            a_text = a[2]
            if motion_id in df_nm.index:
                nm = df_nm.loc[motion_id]
                nm_text = nm.get("text") if isinstance(nm, pd.Series) else nm["text"]
                text = nm_text if nm_text else a_text
                date = nm.get("date") if isinstance(nm, pd.Series) else nm["date"]
                doc_type = nm.get("doc_type") if isinstance(nm, pd.Series) else nm["doc_type"]
                party = nm.get("party") if isinstance(nm, pd.Series) else nm["party"]
            else:
                text = a_text
                date = None
                doc_type = None
                party = None
            # apply filters similar to SQL WHERE COALESCE(nm.text, a.text) IS NOT NULL AND LENGTH(...)>0
            if text is None:
                continue
            if isinstance(text, str) and len(text) == 0:
                continue
            out.append((motion_id, category, text, date, doc_type, party))
        return out

    # Fallback to SQL join (original behavior)
    cur.execute(
        """
        SELECT a.motion_id, a.category, COALESCE(nm.text, a.text) AS text,
               nm.date, nm.doc_type, nm.party
        FROM augmented_gold_labels a
        LEFT JOIN normalized_motions nm ON a.motion_id = nm.id
        WHERE a.split = ?
          AND COALESCE(nm.text, a.text) IS NOT NULL
          AND LENGTH(COALESCE(nm.text, a.text)) > 0
    """,
        (split,),
    )
    return cur.fetchall()


def fetch_all_normalized_motions(conn: sqlite3.Connection) -> List[NormalizedMotion]:
    if _has_parquet("normalized_motions"):
        rows = _load_parquet_as_dicts("normalized_motions")
        return [_row_to_normalized_motion(r) for r in rows]
    cur = conn.cursor()
    cur.execute("SELECT id, title, text, date, party, metadata, doc_type, full_text_url FROM normalized_motions")
    return [_row_to_normalized_motion(r) for r in cur.fetchall()]


def _row_to_normalized_motion(row) -> NormalizedMotion:
    def _get(r, key, idx):
        return r[key] if isinstance(r, sqlite3.Row) else r[idx]

    metadata_raw = _get(row, "metadata", 5)
    doc_type_raw = _get(row, "doc_type", 6) if (isinstance(row, sqlite3.Row) or len(row) > 6) else None
    # full_text_url may not exist in older rows; be defensive
    full_text_url = _get(row, "full_text_url", 7) if (isinstance(row, sqlite3.Row) or len(row) > 7) else None
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
            "doc_type": doc_type_raw,
            "full_text_url": full_text_url,
        }
    )


# ---------------------------------------------------------------------------
# ClassificationResult
# ---------------------------------------------------------------------------

def fetch_classifications_for_motion(
    conn: sqlite3.Connection, motion_id: str
) -> List[ClassificationResult]:
    if _has_parquet("classifications"):
        rows = _load_parquet_as_dicts("classifications")
        results = []
        for r in rows:
            if r.get("motion_id") == motion_id:
                results.append(_row_to_classification(r))
        return results
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
    created_at = datetime.fromisoformat(created_raw) if created_raw else datetime.now(timezone.utc)
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
    if _has_parquet("party_profiles"):
        rows = _load_parquet_as_dicts("party_profiles")
        results: List[PartyProfile] = []
        for r in rows:
            party = r.get("party")
            profile_raw = r.get("profile_json")
            last_updated_raw = r.get("last_updated")
            try:
                totals = json.loads(profile_raw) if isinstance(profile_raw, str) else (profile_raw or {})
            except (json.JSONDecodeError, TypeError):
                totals = {}
            updated_at = (
                datetime.fromisoformat(last_updated_raw) if last_updated_raw else datetime.now(timezone.utc)
            )
            results.append(PartyProfile.model_validate({"party": party, "totals": totals, "updated_at": updated_at}))
        return results
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
            datetime.fromisoformat(last_updated_raw) if last_updated_raw else datetime.now(timezone.utc)
        )
        results.append(PartyProfile.model_validate({"party": party, "totals": totals, "updated_at": updated_at}))
    return results
