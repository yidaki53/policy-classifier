#!/usr/bin/env python3
"""
Lookup utility for motions (normalized_motions + classifications).

Features:
- lookup by `--motion-id`(s)
- lookup by `--party`
- lookup by `--author` substring (searches `metadata` JSON text)
- lookup by `--category` (uses latest `classifications` table entries)

Usage:
  python3 scripts/lookup_motions.py --motion-id <id>
  python3 scripts/lookup_motions.py --party M --limit 50
  python3 scripts/lookup_motions.py --author "Andersson"
"""

from __future__ import annotations
import argparse
import sqlite3
import os
from pathlib import Path
import pandas as pd


def query_normalized_motions(db_path: str, motion_ids=None, party=None, author_substr=None, limit=100):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    q = "SELECT id, title, text, date, party, metadata FROM normalized_motions"
    clauses = []
    params = []
    if motion_ids:
        placeholders = ",".join(["?"] * len(motion_ids))
        clauses.append(f"id IN ({placeholders})")
        params.extend(motion_ids)
    if party:
        clauses.append("LOWER(party)=LOWER(?)")
        params.append(party)
    if author_substr:
        clauses.append("metadata LIKE ?")
        params.append(f"%{author_substr}%")
    if clauses:
        q = q + " WHERE " + " AND ".join(clauses)
    q = q + " LIMIT ?"
    params.append(limit)
    cur.execute(q, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def query_latest_classifications_for_motions(db_path: str, motion_ids=None, category=None, limit=100):
    conn = sqlite3.connect(db_path)
    q = """
    SELECT c.motion_id, c.category, c.raw_score, c.normalized_weight, c.created_at
    FROM classifications c
    JOIN (
      SELECT motion_id, MAX(created_at) as max_created FROM classifications GROUP BY motion_id
    ) latest ON c.motion_id = latest.motion_id AND c.created_at = latest.max_created
    """
    clauses = []
    params = []
    if motion_ids:
        placeholders = ",".join(["?"] * len(motion_ids))
        clauses.append(f"c.motion_id IN ({placeholders})")
        params.extend(motion_ids)
    if category:
        clauses.append("c.category = ?")
        params.append(category)
    if clauses:
        q = q + " WHERE " + " AND ".join(clauses)
    q = q + " LIMIT ?"
    params.append(limit)
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    return df.to_dict(orient="records")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/swedish_parliament.db")
    p.add_argument("--motion-id", action="append")
    p.add_argument("--party")
    p.add_argument("--author")
    p.add_argument("--category")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--out")
    args = p.parse_args()

    motions = query_normalized_motions(args.db, motion_ids=args.motion_id, party=args.party, author_substr=args.author, limit=args.limit)
    cls = query_latest_classifications_for_motions(args.db, motion_ids=[m["id"] for m in motions] if motions else args.motion_id, category=args.category, limit=args.limit)

    # Merge
    cls_map = {c["motion_id"]: c for c in cls}
    rows = []
    for m in motions:
        c = cls_map.get(m["id"]) if cls_map else None
        rows.append(
            {
                "motion_id": m.get("id"),
                "title": m.get("title"),
                "party": m.get("party"),
                "category": c.get("category") if c else None,
                "raw_score": c.get("raw_score") if c else None,
                "snippet": (m.get("text") or "")[:240].replace("\n", " "),
            }
        )

    df = pd.DataFrame(rows)
    if args.out:
        outp = Path(args.out)
        if outp.suffix.lower() == '.csv':
            outp = outp.with_suffix('.parquet')
        df.to_parquet(outp, index=False, compression='zstd')
        print(f"Wrote {len(df)} rows to {outp}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
