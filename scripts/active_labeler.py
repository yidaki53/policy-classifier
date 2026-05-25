"""Export low-confidence motions for manual labeling (Parquet-first).

Reads `data/parquet/normalized_motions.parquet` and `data/parquet/classifications.parquet`
to select motions whose top classification weight is below a threshold and
writes them to CSV for annotation.
"""

from pathlib import Path
import argparse
import csv
import json
import logging
from typing import Optional

import pandas as pd

LOG = logging.getLogger(__name__)


def fetch_low_confidence_parquet(normalized_parquet: str | Path = "data/parquet/normalized_motions.parquet", classifications_parquet: str | Path = "data/parquet/classifications.parquet", threshold: float = 0.2, limit: int = 100):
    nm_p = Path(normalized_parquet)
    cls_p = Path(classifications_parquet)
    if not nm_p.exists():
        print("No normalized motions parquet found at", nm_p)
        return []

    nm = pd.read_parquet(nm_p)
    if cls_p.exists():
        cls = pd.read_parquet(cls_p)
        # top category per motion
        cls_sorted = cls.sort_values(["motion_id", "normalized_weight"], ascending=[True, False])
        top = cls_sorted.groupby("motion_id", sort=False).first().reset_index()
        top_map = {str(r.motion_id): (r.category, float(r.normalized_weight), r.matched_rules if "matched_rules" in r else None, r.classifier_version if "classifier_version" in r else None) for _, r in top.iterrows()}
    else:
        top_map = {}

    rows = []
    cnt = 0
    for _, r in nm.sort_values("date", ascending=False).iterrows():
        if limit and cnt >= limit:
            break
        mid = str(r.get("id"))
        top_entry = top_map.get(mid)
        top_weight = top_entry[1] if top_entry else 0.0
        if float(top_weight) < float(threshold):
            title = r.get("title") or ""
            text = r.get("text") or ""
            party = r.get("party") or ""
            date = r.get("date") or ""
            top_cat = top_entry[0] if top_entry else ""
            matched_raw = top_entry[2] if top_entry else "[]"
            classifier_version = top_entry[3] if top_entry else ""
            rows.append((mid, title, text, party, date, top_cat, float(top_weight), matched_raw, classifier_version))
            cnt += 1

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


def main(normalized_parquet: Optional[str], classifications_parquet: Optional[str], threshold: float, limit: int, out_file: Optional[str], print_preview: bool):
    if normalized_parquet is None:
        normalized_parquet = Path(__file__).resolve().parents[2] / "data" / "parquet" / "normalized_motions.parquet"
    rows = fetch_low_confidence_parquet(normalized_parquet, classifications_parquet or "data/parquet/classifications.parquet", threshold=threshold, limit=limit)
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
    parser.add_argument("--normalized", default=None, help="Path to normalized_motions.parquet")
    parser.add_argument("--classifications", default=None, help="Path to classifications.parquet")
    parser.add_argument("--threshold", type=float, default=0.2, help="Max top-category weight threshold to select low-confidence items")
    parser.add_argument("--limit", type=int, default=200, help="Max number of motions to export")
    parser.add_argument("--out", default=None, help="CSV output file path")
    parser.add_argument("--preview", action="store_true", help="Print preview of first few rows")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args.normalized, args.classifications, args.threshold, args.limit, args.out, args.preview)
