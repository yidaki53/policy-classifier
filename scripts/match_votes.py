#!/usr/bin/env python3
"""Build the motion-vote matching layer via betänkande bridge.

Links motions to votering records through committee reports (betänkande):

  Motion --ref_dok_id--> Betänkande --rm+beteckning--> Votering

Reads betänkande Parquet files to build the motion->committee report bridge,
then joins with votering summary to find which motions reached a roll-call vote.
Populates the small ``motion_votes`` lookup table in SQLite.

Usage:
    uv run python scripts/match_votes.py --db data/swedish_parliament.db
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _load_votering_summary(parquet_dir: str) -> pd.DataFrame:
    """Aggregate votering Parquet files by (rm, beteckning)."""
    path = Path(parquet_dir)
    files = sorted(path.glob("*.parquet"))
    print(f"Loading {len(files)} votering Parquet files ...", file=sys.stderr)

    chunks: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_parquet(f)
        keep = ["rm", "beteckning", "votering_id"]
        df = df[[c for c in keep if c in df.columns]].copy()
        if df.empty:
            continue
        chunks.append(df)

    if not chunks:
        raise RuntimeError("No votering data found in Parquet files")

    all_v = pd.concat(chunks, ignore_index=True)
    all_v.dropna(subset=["rm", "beteckning", "votering_id"], inplace=True)

    summary = (
        all_v.groupby(["rm", "beteckning"])
        .agg(votering_count=("votering_id", "nunique"), first_votering_id=("votering_id", "first"))
        .reset_index()
    )
    return summary


def _load_betankande_bridge(parquet_dir: str) -> pd.DataFrame:
    """Load betänkande Parquet files and explode ref_dok_ids to one row per motion->committee link."""
    path = Path(parquet_dir)
    files = sorted(path.glob("*.parquet"))
    print(f"Loading {len(files)} betänkande Parquet files ...", file=sys.stderr)

    chunks: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_parquet(f, columns=["dok_id", "rm", "beteckning", "ref_dok_ids"])
        df = df[df["ref_dok_ids"].notna()].copy()
        if df.empty:
            continue
        # Explode ref_dok_ids JSON array
        df["ref_dok_ids"] = df["ref_dok_ids"].apply(json.loads)
        df = df.explode("ref_dok_ids")
        df.rename(columns={"ref_dok_ids": "motion_id", "beteckning": "committee_beteckning"}, inplace=True)
        chunks.append(df[["motion_id", "rm", "committee_beteckning"]])

    if not chunks:
        raise RuntimeError("No betänkande bridge data found")

    bridge = pd.concat(chunks, ignore_index=True)
    bridge.dropna(subset=["motion_id", "rm", "committee_beteckning"], inplace=True)
    bridge.drop_duplicates(inplace=True)
    return bridge


def match_votes(
    db_path: str,
    votering_parquet_dir: str = "data/votering/parquet",
    betankande_parquet_dir: str = "data/betankande/parquet",
    batch_size: int = 1000,
):
    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(f"Database not found: {db}")

    # 1. Load votering summary (committee report -> vote count)
    vot_summary = _load_votering_summary(votering_parquet_dir)
    print(f"  {len(vot_summary):,} unique (rm, beteckning) vote records", file=sys.stderr)

    # 2. Load betänkande bridge (motion -> committee report)
    bridge = _load_betankande_bridge(betankande_parquet_dir)
    print(f"  {len(bridge):,} motion->committee links", file=sys.stderr)

    # 3. Join: motion + bridge + votering
    matched = bridge.merge(vot_summary, left_on=["rm", "committee_beteckning"], right_on=["rm", "beteckning"], how="inner")
    # Aggregate per motion: sum votering_count, pick first votering_id
    matched_agg = (
        matched.groupby("motion_id")
        .agg(votering_count=("votering_count", "sum"), first_votering_id=("first_votering_id", "first"), rm=("rm", "first"), beteckning=("committee_beteckning", "first"))
        .reset_index()
    )
    print(f"  {len(matched_agg):,} unique motions matched to at least one vote", file=sys.stderr)

    # 4. Write to SQLite
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS motion_votes (
            motion_id TEXT PRIMARY KEY,
            rm TEXT NOT NULL,
            beteckning TEXT NOT NULL,
            votering_count INTEGER DEFAULT 0,
            first_votering_id TEXT,
            matched_at TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_motion_votes_rm_beteckning ON motion_votes(rm, beteckning)")

    # Clear and repopulate (idempotent)
    cur.execute("DELETE FROM motion_votes")
    conn.commit()

    now = datetime.now(timezone.utc).isoformat()
    batch: list[tuple] = []
    inserted = 0

    for _, r in matched_agg.iterrows():
        batch.append(
            (
                r["motion_id"],
                r["rm"],
                r["beteckning"],
                int(r["votering_count"]),
                r["first_votering_id"],
                now,
            )
        )
        if len(batch) >= batch_size:
            cur.executemany(
                """
                INSERT INTO motion_votes (motion_id, rm, beteckning, votering_count, first_votering_id, matched_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            conn.commit()
            inserted += len(batch)
            batch.clear()

    if batch:
        cur.executemany(
            """
            INSERT INTO motion_votes (motion_id, rm, beteckning, votering_count, first_votering_id, matched_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        conn.commit()
        inserted += len(batch)

    conn.close()
    print(f"Inserted {inserted:,} rows into motion_votes", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Build motion-vote matching layer via betänkande bridge")
    parser.add_argument("--db", default="data/swedish_parliament.db", help="SQLite database path")
    parser.add_argument("--votering-parquet", default="data/votering/parquet", help="Directory with votering Parquet files")
    parser.add_argument("--betankande-parquet", default="data/betankande/parquet", help="Directory with betänkande Parquet files")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    match_votes(
        db_path=args.db,
        votering_parquet_dir=args.votering_parquet,
        betankande_parquet_dir=args.betankande_parquet,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
