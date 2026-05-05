#!/usr/bin/env python3
"""Classify normalized motions and produce party profiles + figures."""

import argparse
import json
from pathlib import Path

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.scorer import load_definitions, score_motion
from swedish_parliament_policy_classifier.classifier.persist import record_lineage, persist_classification
from swedish_parliament_policy_classifier.analysis.aggregate import compute_party_profiles
from swedish_parliament_policy_classifier.visualization.plot_party_profiles import plot_party_profiles


def normalize_and_insert(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, json FROM raw_motions")
    rows = cur.fetchall()
    inserted = 0
    for row in rows:
        motion_id = row[0]
        raw_json = row[1]
        cur.execute("SELECT 1 FROM normalized_motions WHERE id = ?", (motion_id,))
        if cur.fetchone():
            continue
        try:
            data = json.loads(raw_json)
        except Exception:
            continue
        title = data.get("title") or data.get("rubrik") or ""
        text = data.get("text") or data.get("body") or title or ""
        date = data.get("date") or data.get("datum") or None
        party = data.get("party") or data.get("parti") or None
        metadata = {k: v for k, v in data.items() if k not in ("title", "text", "date", "party")}
        cur.execute(
            "INSERT OR IGNORE INTO normalized_motions (id, title, text, date, party, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (motion_id, title, text, date, party, json.dumps(metadata, ensure_ascii=False)),
        )
        inserted += 1
    conn.commit()
    return inserted


def classify(db_path: str = "data/swedish_parliament.db", limit: int | None = None):
    conn = init_db(db_path)
    normalize_and_insert(conn)
    defs = load_definitions()

    cur = conn.cursor()
    cur.execute(
        """SELECT nm.id, nm.text FROM normalized_motions nm
        LEFT JOIN classifications c ON nm.id = c.motion_id
        WHERE c.id IS NULL"""
    )

    rows = cur.fetchall()
    if limit:
        rows = rows[:limit]

    total = 0
    for row in rows:
        motion_id = row[0]
        text = row[1] or ""
        results = score_motion(motion_id, text, defs)
        lineage_id = record_lineage(conn, "normalized_motions", motion_id, "classification_batch")
        for r in results:
            persist_classification(conn, r, lineage_id)
        total += 1

    compute_party_profiles(conn)
    out = plot_party_profiles(conn)
    return total, out


def main():
    parser = argparse.ArgumentParser(description="Classify motions and plot party profiles")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    total, out = classify(args.db, args.limit)
    print(f"Classified {total} motions; figure: {out}")


if __name__ == "__main__":
    main()
