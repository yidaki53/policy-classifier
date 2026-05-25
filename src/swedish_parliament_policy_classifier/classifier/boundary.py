"""Boundary module for classifier functionality.

Provides a small, well-defined interface for classification + persistence.
This module is intentionally lightweight and performs lazy imports to avoid
import-time cycles with the canonical `exports` module.

Public API:
  - classify_motion(motion_id, text, ...)
  - classify_and_persist(normalized_motion, db_conn=None, ...)
  - get_next_unlabeled_motion(conn, threshold=0.2)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import json
import sqlite3
from importlib import import_module

# Use the packaged models directly (no top-level compatibility fallback)
from swedish_parliament_policy_classifier.models import NormalizedMotion, ClassificationResult, CategoryDef


def _import_module_candidate(name: str):
    try:
        return import_module(name)
    except Exception:
        return None


def _get_scorer_module():
    for name in ("classifier.scorer", "swedish_parliament_policy_classifier.classifier.scorer"):
        m = _import_module_candidate(name)
        if m and hasattr(m, "score_motion"):
            return m
    raise ImportError("No scorer implementation found")


def _get_persist_module():
    # Prefer parquet-backed persist module if available
    for name in ("classifier.persist_parquet", "swedish_parliament_policy_classifier.classifier.persist_parquet", "classifier.persist", "swedish_parliament_policy_classifier.classifier.persist"):
        m = _import_module_candidate(name)
        if m and hasattr(m, "persist_classifications_batch"):
            return m
    raise ImportError("No persist implementation found")


def _get_defs_loader():
    for name in ("definitions.loader", "swedish_parliament_policy_classifier.definitions.loader"):
        m = _import_module_candidate(name)
        if m and hasattr(m, "load_verified_definitions"):
            return m
    raise ImportError("No definitions.loader found")


def _get_db_module():
    for name in ("swedish_parliament_policy_classifier.db", "db"):
        m = _import_module_candidate(name)
        if m and hasattr(m, "init_db") and hasattr(m, "get_connection"):
            return m
    raise ImportError("No db module found")


def classify_motion(
    motion_id: str,
    text: str,
    categories: Optional[Dict[str, CategoryDef]] = None,
    embedding_matcher: Optional[Any] = None,
    **kwargs,
) -> List[ClassificationResult]:
    """Classify a motion text and return classification results.

    If `categories` is omitted, the canonical definitions loader is used.
    This function is a thin boundary over `classifier.scorer.score_motion`.
    """
    if categories is None:
        loader = _get_defs_loader()
        categories = loader.load_verified_definitions()

    scorer = _get_scorer_module()
    return scorer.score_motion(motion_id, text, categories, embedding_matcher=embedding_matcher, **kwargs)


def classify_and_persist(
    normalized_motion: Union[NormalizedMotion, dict],
    db_conn: Optional[sqlite3.Connection] = None,
    embedding_matcher: Optional[Any] = None,
    **kwargs,
) -> List[ClassificationResult]:
    """Classify a `NormalizedMotion` (or dict) and persist classifications.

    Ensures the normalized motion exists in `normalized_motions`, runs
    `score_motion`, and persists results transactionally using
    `persist.persist_classifications_batch`.
    """
    # validate/convert
    if isinstance(normalized_motion, dict):
        nm = NormalizedMotion.model_validate(normalized_motion)
    else:
        nm = normalized_motion

    # acquire connection
    if db_conn is None:
        dbm = _get_db_module()
        conn = dbm.init_db()
    else:
        conn = db_conn

    # ensure normalized motion present (upsert into parquet if we're using parquet persist)
    persist_mod = _get_persist_module()
    try:
        # if persist module exposes upsert_normalized_motion_parquet, use it
        if hasattr(persist_mod, "upsert_normalized_motion_parquet"):
            persist_mod.upsert_normalized_motion_parquet(nm)
        else:
            # fallback to DB insert if a sqlite conn is available
            cur = conn.cursor()
            meta_json = json.dumps(nm.metadata or {}, ensure_ascii=False)
            date_str = None
            if getattr(nm, "date", None):
                try:
                    date_str = nm.date.isoformat()
                except Exception:
                    date_str = str(nm.date)

            cur.execute(
                "INSERT OR IGNORE INTO normalized_motions (id, title, text, date, party, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (nm.id, nm.title, nm.text, date_str, nm.party, meta_json),
            )
            conn.commit()
    except Exception:
        # best-effort: try DB insert as last resort
        try:
            cur = conn.cursor()
            meta_json = json.dumps(nm.metadata or {}, ensure_ascii=False)
            date_str = None
            if getattr(nm, "date", None):
                try:
                    date_str = nm.date.isoformat()
                except Exception:
                    date_str = str(nm.date)

            cur.execute(
                "INSERT OR IGNORE INTO normalized_motions (id, title, text, date, party, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (nm.id, nm.title, nm.text, date_str, nm.party, meta_json),
            )
            conn.commit()
        except Exception:
            pass

    # classification
    loader = _get_defs_loader()
    cats = loader.load_verified_definitions()
    scorer = _get_scorer_module()
    results = scorer.score_motion(nm.id, nm.text, cats, embedding_matcher=embedding_matcher, **kwargs)

    # persist classifications (creates lineage internally)
    persist = _get_persist_module()
    # call persist implementation; DB-backed persists expect (conn, results, ...)
    # parquet-backed persist accepts (conn, results, source_table, source_id)
    try:
        # if parquet persist, its functions ignore conn but accept similar signature
        if hasattr(persist, "persist_classifications_batch"):
            try:
                persist.persist_classifications_batch(conn, results, source_table="normalized_motions", source_id=nm.id)
            except TypeError:
                # older signature: persist_classifications_batch(results, ...)
                persist.persist_classifications_batch(results, source_table="normalized_motions", source_id=nm.id)
        else:
            raise
    except Exception:
        # final fallback: try to call DB-backed persist functions if available elsewhere
        fallback = _import_module_candidate("classifier.persist") or _import_module_candidate("swedish_parliament_policy_classifier.classifier.persist")
        if fallback and hasattr(fallback, "persist_classifications_batch"):
            fallback.persist_classifications_batch(conn, results, source_table="normalized_motions", source_id=nm.id)
        else:
            raise

    return results


def get_next_unlabeled_motion(conn: sqlite3.Connection, threshold: float = 0.2):
    persist = _get_persist_module()
    return persist.get_next_unlabeled_motion(conn, threshold)


__all__ = ["classify_motion", "classify_and_persist", "get_next_unlabeled_motion"]
