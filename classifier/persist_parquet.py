"""Parquet-based persistence helpers for classifications, lineage, and annotations.

Provides append-safe, atomic writers and upsert semantics to support a
Parquet-first workflow. Functions mirror the DB-based API in
`classifier.persist` so `classifier.boundary` can call them transparently.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd

from swedish_parliament_policy_classifier.models import ClassificationResult, NormalizedMotion


def _atomic_write_df(df: pd.DataFrame, out_path: Union[str, Path]):
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_p.with_suffix(out_p.suffix + ".tmp")
    # Prefer Parquet if engine is available; fall back to pickle for portability
    try:
        df.to_parquet(tmp, index=False)
    except Exception:
        # fallback: use pickle to avoid requiring pyarrow/fastparquet in tests
        df.to_pickle(tmp)
    os.replace(tmp, out_p)


def _read_table_compat(path: Union[str, Path]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except Exception:
        try:
            return pd.read_pickle(p)
        except Exception:
            # try json as last resort
            try:
                return pd.read_json(p, orient="records")
            except Exception:
                raise


def _to_row_from_classification(cl) -> dict:
    # Accept either Pydantic model, object with attrs, or dict
    if cl is None:
        return {}
    if hasattr(cl, "model_dump"):
        d = cl.model_dump()
    elif hasattr(cl, "dict"):
        d = cl.dict()
    elif isinstance(cl, dict):
        d = cl
    else:
        # try attribute access
        d = {}
        for a in ("motion_id", "category", "raw_score", "normalized_weight", "matched_rules", "classifier_version", "created_at"):
            if hasattr(cl, a):
                d[a] = getattr(cl, a)

    # ensure fields
    created = d.get("created_at")
    if created is None:
        created_iso = datetime.utcnow().isoformat()
    else:
        if isinstance(created, str):
            created_iso = created
        else:
            try:
                created_iso = created.isoformat()
            except Exception:
                created_iso = str(created)

    return {
        "motion_id": str(d.get("motion_id") or d.get("motionId") or ""),
        "category": str(d.get("category") or ""),
        "raw_score": float(d.get("raw_score") or d.get("rawScore") or 0.0),
        "normalized_weight": float(d.get("normalized_weight") or d.get("normalizedWeight") or 0.0),
        "matched_rules": json.dumps(d.get("matched_rules") or d.get("matchedRules") or [], ensure_ascii=False),
        "classifier_version": str(d.get("classifier_version") or d.get("classifierVersion") or ""),
        "created_at": created_iso,
    }


def upsert_normalized_motion_parquet(normalized_motion: Union[NormalizedMotion, dict], out_parquet: Union[str, Path] = "data/parquet/normalized_motions.parquet") -> None:
    """Upsert a normalized motion into `normalized_motions.parquet` (no-op if exists)."""
    out_p = Path(out_parquet)
    nm = normalized_motion
    if hasattr(nm, "model_dump"):
        try:
            nm_d = nm.model_dump()
        except Exception:
            nm_d = nm.dict()
    elif isinstance(nm, dict):
        nm_d = nm
    else:
        # try attribute access
        nm_d = {k: getattr(nm, k, None) for k in ("id", "title", "text", "date", "party", "metadata")}

    row = {
        "id": str(nm_d.get("id") or ""),
        "title": nm_d.get("title") or None,
        "text": nm_d.get("text") or "",
        "date": nm_d.get("date") or None,
        "party": nm_d.get("party") or None,
        "metadata": json.dumps(nm_d.get("metadata") or {}, ensure_ascii=False),
    }

    if out_p.exists():
        try:
            prev = _read_table_compat(out_p)
            if "id" in prev.columns and str(row["id"]) in prev["id"].astype(str).values:
                return
            out_df = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
        except Exception:
            out_df = pd.DataFrame([row])
    else:
        out_df = pd.DataFrame([row])

    _atomic_write_df(out_df, out_p)


def persist_classifications_batch(
    conn: Optional[object],
    classifications: Iterable[Union[ClassificationResult, dict]],
    source_table: Optional[str] = None,
    source_id: Optional[str] = None,
    out_parquet: Union[str, Path] = "data/parquet/classifications.parquet",
    lineage_parquet: Union[str, Path] = "data/parquet/lineage.parquet",
) -> List[int]:
    """Persist a batch of classification results into a Parquet file (upsert semantics).

    The `conn` argument is accepted for API compatibility but ignored.
    Returns a list of synthetic row-IDs (row numbers in the resulting file).
    """
    out_p = Path(out_parquet)
    rows = []
    for cl in classifications:
        rows.append(_to_row_from_classification(cl))

    if not rows:
        return []

    chunk = pd.DataFrame(rows)

    if out_p.exists():
        try:
            prev = _read_table_compat(out_p)
            combined = pd.concat([prev, chunk], ignore_index=True)
            # keep latest by created_at per (motion_id, category)
            if "created_at" in combined.columns:
                combined["_created_ts"] = pd.to_datetime(combined["created_at"], errors="coerce")
                combined = combined.sort_values("_created_ts").drop_duplicates(subset=["motion_id", "category"], keep="last")
                combined = combined.drop(columns=["_created_ts"])
            else:
                combined = combined.drop_duplicates(subset=["motion_id", "category"], keep="last")
            out_df = combined.reset_index(drop=True)
        except Exception:
            out_df = chunk
    else:
        out_df = chunk

    _atomic_write_df(out_df, out_p)

    # record lineage row
    try:
        record_lineage_parquet(source_table or "normalized_motions", source_id or "", "classification_batch", lineage_parquet)
    except Exception:
        pass

    # return synthetic ids (row positions)
    return list(range(len(out_df)))


def persist_classification(conn: Optional[object], classification: Union[ClassificationResult, dict], lineage_id: Optional[int] = None, out_parquet: Union[str, Path] = "data/parquet/classifications.parquet") -> int:
    ids = persist_classifications_batch(conn, [classification], out_parquet=out_parquet)
    return ids[0] if ids else -1


def record_lineage_parquet(source_table: str, source_id: str, operation: str, lineage_out: Union[str, Path] = "data/parquet/lineage.parquet", checksum: Optional[str] = None, parent_lineage_id: Optional[int] = None, notes: Optional[str] = None) -> int:
    out_p = Path(lineage_out)
    row = {
        "source_table": source_table,
        "source_id": source_id,
        "operation": operation,
        "timestamp": datetime.utcnow().isoformat(),
        "checksum": checksum,
        "parent_lineage_id": parent_lineage_id,
        "notes": notes,
    }
    if out_p.exists():
        try:
            prev = _read_table_compat(out_p)
            out_df = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
        except Exception:
            out_df = pd.DataFrame([row])
    else:
        out_df = pd.DataFrame([row])

    _atomic_write_df(out_df, out_p)
    try:
        return int(out_df.index[-1])
    except Exception:
        return 0


def save_annotation(conn: Optional[object], motion_id: str, annotator: str, labels: List[dict], notes: Optional[str] = None, status: str = "annotated", annotations_out: Union[str, Path] = "data/parquet/annotations.parquet") -> int:
    out_p = Path(annotations_out)
    now = datetime.utcnow().isoformat()
    row = {
        "motion_id": motion_id,
        "annotator": annotator,
        "labels": json.dumps(labels, ensure_ascii=False),
        "notes": notes,
        "status": status,
        "created_at": now,
        "updated_at": now,
    }
    if out_p.exists():
        try:
            prev = _read_table_compat(out_p)
            out_df = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
        except Exception:
            out_df = pd.DataFrame([row])
    else:
        out_df = pd.DataFrame([row])

    _atomic_write_df(out_df, out_p)
    try:
        return int(out_df.index[-1])
    except Exception:
        return 0


def get_next_unlabeled_motion(normalized_parquet: Union[str, Path] = "data/parquet/normalized_motions.parquet", classifications_parquet: Union[str, Path] = "data/parquet/classifications.parquet", annotations_parquet: Union[str, Path] = "data/parquet/annotations.parquet", threshold: float = 0.2) -> Optional[Tuple]:
    nm_p = Path(normalized_parquet)
    cls_p = Path(classifications_parquet)
    ann_p = Path(annotations_parquet)
    if not nm_p.exists():
        return None

    nm = _read_table_compat(nm_p)
    nm["id"] = nm["id"].astype(str)

    top_map = {}
    if cls_p.exists():
        try:
            cls = _read_table_compat(cls_p)
            cls_sorted = cls.sort_values(["motion_id", "normalized_weight"], ascending=[True, False])
            top = cls_sorted.groupby("motion_id", sort=False).first().reset_index()
            top_map = {str(r.motion_id): (r.category, float(r.normalized_weight)) for _, r in top.iterrows()}
        except Exception:
            top_map = {}

    annotated_ids = set()
    if ann_p.exists():
        try:
            ann = _read_table_compat(ann_p)
            if "motion_id" in ann.columns:
                annotated_ids = set(ann[ann["status"] == "annotated"]["motion_id"].astype(str).unique())
        except Exception:
            annotated_ids = set()

    # build candidate rows
    rows = []
    for _, r in nm.iterrows():
        mid = str(r.get("id"))
        if mid in annotated_ids:
            continue
        tm = top_map.get(mid)
        top_cat = tm[0] if tm else None
        top_w = float(tm[1]) if tm else 0.0
        if top_w < threshold:
            rows.append((mid, r.get("title"), r.get("text"), r.get("party"), r.get("date"), top_cat, top_w))

    if not rows:
        return None
    # prefer most recent by date if present
    try:
        # convert date to sortable values where possible
        rows_sorted = sorted(rows, key=lambda x: x[4] or "", reverse=True)
    except Exception:
        rows_sorted = rows

    return rows_sorted[0]


__all__ = [
    "persist_classifications_batch",
    "persist_classification",
    "record_lineage_parquet",
    "save_annotation",
    "get_next_unlabeled_motion",
    "upsert_normalized_motion_parquet",
]
