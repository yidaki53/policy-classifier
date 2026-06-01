#!/usr/bin/env python3
"""Bulk fetcher for Riksdag motion full texts.

Fetches `https://data.riksdagen.se/dokument/{id}.text` for every motion in the
DB that has a `full_text_url` but short/empty `text`. Extracts the HTML block,
strips tags, and writes the cleaned body text back into `normalized_motions.text`.
"""

import argparse
import re
import sqlite3
import sys
import time
from html import unescape
from typing import Optional

import requests

LOG_INTERVAL = 10


def _extract_plain_text(xml_content: str) -> str:
    """Extract human-readable text from the XML/HTML payload."""
    # Find the <html> block
    html_match = re.search(r'<html>(.*?)</html>', xml_content, re.DOTALL)
    if not html_match:
        # Fallback: just strip all XML tags
        html_content = xml_content
    else:
        html_content = html_match.group(1)

    # Unescape HTML entities (&lt; -> <, etc.)
    html_content = unescape(html_content)

    # Remove <style> blocks
    html_content = re.sub(r'<style[^>]*>.*?</style>', ' ', html_content, flags=re.DOTALL)
    # Remove <script> blocks
    html_content = re.sub(r'<script[^>]*>.*?</script>', ' ', html_content, flags=re.DOTALL)
    # Remove all remaining HTML tags
    plain = re.sub(r'<[^>]+>', ' ', html_content)
    # Collapse whitespace
    plain = re.sub(r'\s+', ' ', plain).strip()
    return plain


def _fetch_text(doc_id: str, url: str, timeout: int = 20, retries: int = 3, retry_delay: float = 1.0) -> Optional[str]:
    """Fetch and extract plain text for a single motion."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return _extract_plain_text(resp.text)
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(retry_delay * (attempt + 1))
    print(f"  [WARN] Failed to fetch {doc_id} ({url}): {last_err}", file=sys.stderr)
    return None


def fetch_all(db_path: str, min_text_len: int = 200, batch_size: int = 50, dry_run: bool = False):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Identify motions that need fetching:
    # - text is null, empty, or shorter than min_text_len
    #   (full_text_url may be empty for older ingested docs — we fall back to constructing it from the id)
    cur.execute("""
        SELECT id, full_text_url, text, title
        FROM normalized_motions
        WHERE (text IS NULL OR text = '' OR LENGTH(text) < ?)
    """, (min_text_len,))
    rows = cur.fetchall()
    total = len(rows)
    print(f"Found {total} motions needing full-text fetch.", file=sys.stderr)
    if total == 0:
        return 0, 0

    updated = 0
    failed = 0
    for i, row in enumerate(rows, start=1):
        motion_id = row["id"]
        url = row["full_text_url"] or ""
        if not url:
            url = f"https://data.riksdagen.se/dokument/{motion_id}.text"
        current_text = row["text"] or ""

        if dry_run:
            print(f"[dry-run] Would fetch {motion_id} (text len={len(current_text)})")
            continue

        plain = _fetch_text(motion_id, url)
        if plain and len(plain) > 50:
            cur.execute(
                "UPDATE normalized_motions SET text = ?, full_text_url = ? WHERE id = ?",
                (plain, url, motion_id),
            )
            conn.commit()
            updated += 1
        else:
            failed += 1

        if i % LOG_INTERVAL == 0 or i == total:
            print(f"  {i}/{total} processed — updated={updated}, failed={failed}", file=sys.stderr)
        # Be nice to the server
        time.sleep(0.3)

    conn.close()
    print(f"\nDone: updated={updated}, failed={failed} out of {total}", file=sys.stderr)
    return updated, failed


def main():
    parser = argparse.ArgumentParser(description="Bulk fetch motion full texts from Riksdag API")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--min-text-len", type=int, default=200, help="Motions with text shorter than this will be re-fetched")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fetch_all(
        db_path=args.db,
        min_text_len=args.min_text_len,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
