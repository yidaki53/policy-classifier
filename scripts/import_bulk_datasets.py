#!/usr/bin/env python3
"""Import bulk dataset ZIPs (JSON format) into the SQLite database.

Extracts JSON documents from ZIP archives, parses them, and inserts into
raw_motions with deduplication (skips already-present IDs).

Usage:
    uv run python3 scripts/import_bulk_datasets.py --db data/swedish_parliament.db --zips data/bulk_datasets
"""

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from swedish_parliament_policy_classifier.db.schema import init_db
from swedish_parliament_policy_classifier.classifier.persist import record_lineage
from swedish_parliament_policy_classifier.classifier.party_extractor import extract_party_from_intressent
from swedish_parliament_policy_classifier.nlp.preprocess import extract_plain_text_from_html


def _parse_bulk_json_doc(d: dict, doktyp: str) -> Optional[dict]:
    """Best-effort extraction of fields from bulk dataset JSON.

    Bulk datasets wrap each document under 'dokumentstatus.dokument'.
    """
    # Unwrap nested schema if present
    inner = d
    if isinstance(d, dict) and "dokumentstatus" in d:
        ds = d["dokumentstatus"]
        if isinstance(ds, dict):
            inner = ds.get("dokument") or ds

    if not isinstance(inner, dict):
        return None

    docid = inner.get("dok_id") or inner.get("id") or inner.get("dokument_id")
    if not docid:
        return None

    title = inner.get("titel") or inner.get("rubrik") or ""
    text = inner.get("doktext") or inner.get("text") or inner.get("sammanfattning") or inner.get("undertitel") or ""

    # Fallback to HTML field when primary text fields are empty
    if not text or len(text) < 50:
        html = inner.get("html") or ""
        if html:
            text = extract_plain_text_from_html(html)

    # Final fallback to title if nothing else available
    if not text:
        text = title or ""

    date = inner.get("datum") or inner.get("dok_datum") or inner.get("date") or inner.get("publicerad") or None
    # Bulk datasets store party in the intressent list, not the flat dokument.parti
    party = extract_party_from_intressent(d) or inner.get("parti") or None

    # Full text URL
    url_text = inner.get("dokument_url_text") or inner.get("url_text") or ""
    if url_text.startswith("//"):
        url_text = "https:" + url_text

    return {
        "id": str(docid),
        "title": title,
        "text": text,
        "date": date,
        "party": party,
        "full_text_url": url_text,
        "doc_type": doktyp,
        "raw": d,
    }


def import_zip(zip_path: Path, conn, doktyp: str, verbose: bool = True):
    """Import a single ZIP file into raw_motions."""
    cur = conn.cursor()
    inserted = 0
    skipped = 0
    errors = 0

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            json_files = [f for f in zf.namelist() if f.endswith(".json")]
            if not json_files:
                if verbose:
                    print(f"  No JSON files in {zip_path.name}", file=sys.stderr)
                return 0, 0, 0

            for json_file in json_files:
                with zf.open(json_file) as f:
                    try:
                        data = json.load(f)
                    except Exception as e:
                        if verbose:
                            print(f"  JSON parse error in {json_file}: {e}", file=sys.stderr)
                        errors += 1
                        continue

                    # JSON may be a single object or a list
                    docs = data if isinstance(data, list) else [data]

                    for d in docs:
                        parsed = _parse_bulk_json_doc(d, doktyp)
                        if not parsed:
                            errors += 1
                            continue

                        mid = parsed["id"]

                        # Skip if already present
                        cur.execute("SELECT 1 FROM raw_motions WHERE id = ?", (mid,))
                        if cur.fetchone():
                            skipped += 1
                            continue

                        raw_json = json.dumps(parsed["raw"], ensure_ascii=False)
                        retrieved_at = datetime.now(timezone.utc).isoformat()

                        cur.execute(
                            "INSERT OR IGNORE INTO raw_motions (id, json, retrieved_at, checksum, doc_type) VALUES (?, ?, ?, ?, ?)",
                            (mid, raw_json, retrieved_at, None, doktyp),
                        )
                        if cur.rowcount:
                            inserted += 1
                            try:
                                record_lineage(conn, "raw_motions", mid, "bulk_import", checksum=None)
                            except Exception:
                                pass

        conn.commit()
        if verbose:
            print(f"  {zip_path.name}: inserted={inserted}, skipped={skipped}, errors={errors}", file=sys.stderr)
        return inserted, skipped, errors

    except zipfile.BadZipFile as e:
        if verbose:
            print(f"  Bad ZIP {zip_path.name}: {e}", file=sys.stderr)
        return 0, 0, 1
    except Exception as e:
        if verbose:
            print(f"  Error processing {zip_path.name}: {e}", file=sys.stderr)
        return 0, 0, 1


def import_all(zips_dir: str, db_path: str, verbose: bool = True):
    """Import all ZIP files from directory into the database."""
    conn = init_db(db_path)
    zips_path = Path(zips_dir)

    if not zips_path.exists():
        print(f"Directory not found: {zips_path}", file=sys.stderr)
        return

    zip_files = sorted(zips_path.glob("*.zip"))
    print(f"Found {len(zip_files)} ZIP files to import", file=sys.stderr)

    total_inserted = 0
    total_skipped = 0
    total_errors = 0

    for zip_file in zip_files:
        # Derive doktyp from filename: mot-*, prop-*, etc.
        doktyp = "unknown"
        if zip_file.name.startswith("mot-"):
            doktyp = "mot"
        elif zip_file.name.startswith("prop-"):
            doktyp = "prop"
        elif zip_file.name.startswith("vot-"):
            doktyp = "votering"

        ni, ns, ne = import_zip(zip_file, conn, doktyp, verbose=verbose)
        total_inserted += ni
        total_skipped += ns
        total_errors += ne

    print(f"\nTotal: inserted={total_inserted}, skipped={total_skipped}, errors={total_errors}", file=sys.stderr)
    return total_inserted, total_skipped, total_errors


def main():
    parser = argparse.ArgumentParser(description="Import bulk dataset ZIPs into SQLite")
    parser.add_argument("--db", default="data/swedish_parliament.db", help="Path to SQLite database")
    parser.add_argument("--zips", default="data/bulk_datasets", help="Directory containing ZIP files")
    parser.add_argument("--quiet", action="store_true", dest="quiet")
    args = parser.parse_args()

    import_all(args.zips, args.db, verbose=not args.quiet)


if __name__ == "__main__":
    main()
