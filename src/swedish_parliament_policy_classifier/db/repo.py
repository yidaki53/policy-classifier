"""Repository helpers for classification persistence.

Provides a small, well-tested surface for inserting and deleting
`classifications` rows so scripts and orchestration code don't need to
duplicate SQL or transaction logic.
"""
from pathlib import Path
import sqlite3
from typing import Iterable, Tuple, Optional

from swedish_parliament_policy_classifier.db import schema


class ClassificationRepo:
    def __init__(self, db_path: Optional[Path] = None, conn: Optional[sqlite3.Connection] = None):
        if conn is not None:
            self.conn = conn
        else:
            self.conn = schema.get_connection(db_path or "data/swedish_parliament.db")

    def bulk_insert(self, rows: Iterable[Tuple]) -> None:
        """Insert many classification rows in a transaction.

        Expects rows as tuples matching the `classifications` table columns:
          (motion_id, category, raw_score, normalized_weight, matched_rules, classifier_version, created_at)
        """
        cur = self.conn.cursor()
        cur.executemany(
            """INSERT INTO classifications (
                motion_id, category, raw_score, normalized_weight, matched_rules,
                classifier_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()

    def delete_for_motion(self, motion_id: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM classifications WHERE motion_id = ?", (motion_id,))
        self.conn.commit()


class SpeechClassificationRepo:
    def __init__(self, db_path: Optional[Path] = None, conn: Optional[sqlite3.Connection] = None):
        if conn is not None:
            self.conn = conn
        else:
            self.conn = schema.get_connection(db_path or "data/swedish_parliament.db")

    def bulk_insert(self, rows: Iterable[Tuple]) -> None:
        cur = self.conn.cursor()
        cur.executemany(
            """INSERT INTO speech_classifications (
                speech_id, category, raw_score, normalized_weight, matched_rules,
                classifier_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()
