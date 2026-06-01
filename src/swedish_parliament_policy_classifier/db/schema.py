"""SQLite schema and initialization utilities for the package.

This mirrors the repository's top-level `db/schema.py` so that consumers
importing `swedish_parliament_policy_classifier.db` find `schema` inside the
packaged namespace regardless of the working-directory layout.
"""

from pathlib import Path
import sqlite3
import json
from typing import Union
from datetime import datetime, timezone


def get_connection(db_path: Union[str, Path]) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Union[str, Path] = "data/swedish_parliament.db") -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.executescript(
        """
    CREATE TABLE IF NOT EXISTS raw_motions (
        id TEXT PRIMARY KEY,
        json TEXT NOT NULL,
        retrieved_at TEXT,
        checksum TEXT
    );

    CREATE TABLE IF NOT EXISTS normalized_motions (
        id TEXT PRIMARY KEY,
        title TEXT,
        text TEXT,
        date TEXT,
        party TEXT,
        metadata TEXT
    );

    CREATE TABLE IF NOT EXISTS categories (
        name TEXT PRIMARY KEY,
        definition TEXT,
        keywords TEXT,
        regexes TEXT
    );

    CREATE TABLE IF NOT EXISTS classifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        motion_id TEXT,
        category TEXT,
        raw_score REAL,
        normalized_weight REAL,
        matched_rules TEXT,
        classifier_version TEXT,
        lineage_id INTEGER,
        created_at TEXT,
        FOREIGN KEY(motion_id) REFERENCES normalized_motions(id)
    );

    CREATE TABLE IF NOT EXISTS party_profiles (
        party TEXT PRIMARY KEY,
        profile_json TEXT,
        last_updated TEXT
    );

    CREATE TABLE IF NOT EXISTS annotations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        motion_id TEXT,
        annotator TEXT,
        labels TEXT,
        notes TEXT,
        status TEXT,
        created_at TEXT,
        updated_at TEXT,
        FOREIGN KEY(motion_id) REFERENCES normalized_motions(id)
    );

    CREATE INDEX IF NOT EXISTS idx_annotations_motion ON annotations(motion_id);

    CREATE TABLE IF NOT EXISTS lineage (
        lineage_id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_table TEXT,
        source_id TEXT,
        operation TEXT,
        timestamp TEXT,
        checksum TEXT,
        parent_lineage_id INTEGER,
        notes TEXT
    );
    """
    )

    conn.commit()
    return conn


def _example_insert_raw(conn: sqlite3.Connection, motion_id: str, raw_json: dict) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO raw_motions (id, json, retrieved_at, checksum) VALUES (?, ?, ?, ?)",
        (motion_id, json.dumps(raw_json, ensure_ascii=False), datetime.now(timezone.utc).isoformat(), None),
    )
    conn.commit()
