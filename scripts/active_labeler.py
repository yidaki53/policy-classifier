"""Active learning exporter: select low-confidence motions for manual labeling.

This script queries the project's SQLite DB for motions whose top
classification weight is below a threshold and writes them to a CSV for
human review. It is intentionally read-only and portable so labels can be
collected externally and imported later.

Usage:
    python -m swedish_parliament_policy_classifier.scripts.active_labeler --threshold 0.2 --limit 200
"""

from pathlib import Path
import argparse
import csv
import json
import logging
from typing import Optional

from swedish_parliament_policy_classifier.db.schema import get_connection

LOG = logging.getLogger(__name__)


def fetch_low_confidence(conn, threshold: float = 0.2, limit: int = 100):
    cur = conn.cursor()
    sql = """
    SELECT nm.id as motion_id, nm.title, nm.text, nm.party, nm.date,
           c.category as top_category, c.normalized_weight as top_weight,
           c.matched_rules, c.classifier_version
    FROM normalized_motions nm
    LEFT JOIN classifications c ON c.id = (
        SELECT id FROM classifications WHERE motion_id = nm.id ORDER BY normalized_weight DESC LIMIT 1
    )
    WHERE COALESCE(c.normalized_weight, 0) < ?
    ORDER BY nm.date DESC
    LIMIT ?
    """
    cur.execute(sql, (threshold, limit))
    rows = cur.fetchall()
    return rows


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "motion_id",
            "title",
            "party",
            "date",
            "top_category",
            "top_weight",
            "matched_rules",
            "classifier_version",
            "text",
        ])
        for r in rows:
            mid = r[0]
            title = r[1] or ""
            text = r[2] or ""
            party = r[3] or ""
            date = r[4] or ""
            top_cat = r[5] or ""
            top_w = r[6] or 0.0
            matched_raw = r[7] or "[]"
            classifier_version = r[8] or ""
            try:
                matched = json.loads(matched_raw) if isinstance(matched_raw, str) else matched_raw
                matched_s = ";".join(matched) if isinstance(matched, list) else str(matched)
            except Exception:
                matched_s = str(matched_raw)

            writer.writerow([mid, title, party, date, top_cat, float(top_w), matched_s, classifier_version, text])


def main(db_path: Optional[str], threshold: float, limit: int, out_file: Optional[str], print_preview: bool):
    if db_path is None:
        db_path = Path(__file__).resolve().parents[2] / "data" / "swedish_parliament.db"
    conn = get_connection(db_path)
    rows = fetch_low_confidence(conn, threshold=threshold, limit=limit)
    if not rows:
        print("No low-confidence motions found with the given threshold.")
        return

    if out_file is None:
        out_file = Path(__file__).resolve().parents[2] / "data" / "active_learning_queue.csv"
    out_path = Path(out_file)
    write_csv(rows, out_path)
    print(f"Wrote {len(rows)} motions to {out_path}")

    if print_preview:
        for r in rows[:5]:
            print("-" * 60)
            print(r[0], r[1])
            print((r[2] or "")[:400])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None, help="Path to sqlite DB")
    parser.add_argument("--threshold", type=float, default=0.2, help="Max top-category weight threshold to select low-confidence items")
    parser.add_argument("--limit", type=int, default=200, help="Max number of motions to export")
    parser.add_argument("--out", default=None, help="CSV output file path")
    parser.add_argument("--preview", action="store_true", help="Print preview of first few rows")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args.db, args.threshold, args.limit, args.out, args.preview)
