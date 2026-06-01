"""Persistence helpers for classification results and lineage."""

import sqlite3
import json
from datetime import datetime, timezone
from typing import Optional, Iterable, List

from swedish_parliament_policy_classifier.models.models import ClassificationResult


def record_lineage(
    conn: sqlite3.Connection,
    source_table: str,
    source_id: str,
    operation: str,
    checksum: Optional[str] = None,
    parent_lineage_id: Optional[int] = None,
    notes: Optional[str] = None,
) -> int:
    cur = conn.cursor()
    timestamp = datetime.now(timezone.utc).isoformat()
    cur.execute(
        """INSERT INTO lineage (source_table, source_id, operation, timestamp, checksum, parent_lineage_id, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (source_table, source_id, operation, timestamp, checksum, parent_lineage_id, notes),
    )
    conn.commit()
    return cur.lastrowid


def persist_classification(
    conn: sqlite3.Connection, classification: ClassificationResult, lineage_id: Optional[int] = None
) -> int:
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO classifications (
            motion_id, category, raw_score, normalized_weight, matched_rules, classifier_version, lineage_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            classification.motion_id,
            classification.category,
            float(classification.raw_score),
            float(classification.normalized_weight),
            json.dumps(classification.matched_rules, ensure_ascii=False),
            classification.classifier_version,
            lineage_id,
            classification.created_at.isoformat(),
        ),
    )
    conn.commit()
    return cur.lastrowid


def persist_classifications_batch(
    conn: sqlite3.Connection, classifications: Iterable[ClassificationResult], source_table: Optional[str] = None, source_id: Optional[str] = None
) -> List[int]:
    parent_lineage = None
    if source_table and source_id:
        parent_lineage = record_lineage(conn, source_table, source_id, "classification_batch")

    ids: List[int] = []
    for cl in classifications:
        ids.append(persist_classification(conn, cl, parent_lineage))

    return ids


def save_annotation(
    conn: sqlite3.Connection,
    motion_id: str,
    annotator: str,
    labels: List[dict],
    notes: Optional[str] = None,
    status: str = "annotated",
) -> int:
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cur.execute(
        "INSERT INTO annotations (motion_id, annotator, labels, notes, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (motion_id, annotator, json.dumps(labels, ensure_ascii=False), notes, status, now, now),
    )
    conn.commit()
    return cur.lastrowid


def get_annotation_by_motion(conn: sqlite3.Connection, motion_id: str):
    cur = conn.cursor()
    cur.execute("SELECT * FROM annotations WHERE motion_id = ? ORDER BY created_at DESC LIMIT 1", (motion_id,))
    return cur.fetchone()


def get_next_unlabeled_motion(conn: sqlite3.Connection, threshold: float = 0.2):
    cur = conn.cursor()
    sql = """
    SELECT nm.id as motion_id, nm.title, nm.text, nm.party, nm.date,
           c.category as top_category, c.normalized_weight as top_weight
    FROM normalized_motions nm
    LEFT JOIN classifications c ON c.id = (
        SELECT id FROM classifications WHERE motion_id = nm.id ORDER BY normalized_weight DESC LIMIT 1
    )
    LEFT JOIN annotations a ON a.motion_id = nm.id AND a.status = 'annotated'
    WHERE a.id IS NULL AND COALESCE(c.normalized_weight, 0) < ?
    ORDER BY nm.date DESC
    LIMIT 1
    """
    cur.execute(sql, (threshold,))
    return cur.fetchone()
