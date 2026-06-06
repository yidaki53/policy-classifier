#!/usr/bin/env python3
"""Sync Zenodo DOI metadata into manuscript files.

Typical usage:
    uv run python scripts/sync_zenodo_doi.py \
      --tag submission-2026-06-06-r3 \
      --repo yidaki53/policy-classifier

This script can either:
- Discover the DOI by polling the Zenodo API for a release tag/repository pair, or
- Use a DOI provided directly with --doi-url.

When a DOI is available, it updates:
- manuscript/sections/05_data_availability.md
- manuscript/review/final_submission_checklist.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_REPO = "yidaki53/policy-classifier"
DEFAULT_DATA_AVAILABILITY = Path("manuscript/sections/05_data_availability.md")
DEFAULT_CHECKLIST = Path("manuscript/review/final_submission_checklist.md")


def _fetch_zenodo_records(base_url: str, query: str, size: int = 50) -> list[dict[str, Any]]:
    encoded = urllib.parse.urlencode({"q": query, "size": size, "sort": "mostrecent"})
    url = f"{base_url.rstrip('/')}/api/records?{encoded}"
    with urllib.request.urlopen(url, timeout=30) as response:  # nosec B310 (trusted endpoint)
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("hits", {}).get("hits", [])


def _record_blob(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    parts: list[str] = []

    for field in ("title", "description"):
        value = metadata.get(field)
        if isinstance(value, str):
            parts.append(value)

    related = metadata.get("related_identifiers")
    if isinstance(related, list):
        for item in related:
            if isinstance(item, dict):
                identifier = item.get("identifier")
                if isinstance(identifier, str):
                    parts.append(identifier)

    return " ".join(parts).lower()


def _find_matching_doi(records: list[dict[str, Any]], tag: str, repo: str) -> str | None:
    repo_url = f"https://github.com/{repo}".lower()
    release_url = f"https://github.com/{repo}/releases/tag/{tag}".lower()

    for record in records:
        blob = _record_blob(record)
        if tag.lower() not in blob and repo.lower() not in blob and repo_url not in blob and release_url not in blob:
            continue

        doi = record.get("doi")
        concept_doi = record.get("conceptdoi")
        if isinstance(doi, str) and doi:
            return f"https://doi.org/{doi}"
        if isinstance(concept_doi, str) and concept_doi:
            return f"https://doi.org/{concept_doi}"

    return None


def discover_doi(base_url: str, tag: str, repo: str, max_wait_seconds: int, poll_seconds: int) -> str:
    query = f"{tag} {repo}"
    deadline = time.time() + max_wait_seconds

    last_error: Exception | None = None
    while time.time() <= deadline:
        try:
            records = _fetch_zenodo_records(base_url=base_url, query=query, size=50)
            doi_url = _find_matching_doi(records=records, tag=tag, repo=repo)
            if doi_url:
                return doi_url
        except Exception as exc:  # noqa: BLE001 - surface exact network/API issue
            last_error = exc

        time.sleep(max(1, poll_seconds))

    msg = f"Timed out waiting for Zenodo DOI for tag '{tag}' and repo '{repo}'."
    if last_error:
        msg += f" Last Zenodo API error: {last_error}"
    raise RuntimeError(msg)


def update_data_availability_text(text: str, tag: str, doi_url: str) -> str:
    doi_line = f"Archival DOI for the submission snapshot (`{tag}`): `{doi_url}`."

    # Replace existing archival DOI line if present, preserving surrounding spacing.
    lines = text.splitlines()
    doi_line_re = re.compile(
        r"^[ \t]*Archival DOI for the submission snapshot \(`[^`]+`\): `https://doi\.org/[^`]+`\.[ \t]*$"
    )
    for idx, line in enumerate(lines):
        if doi_line_re.match(line):
            lines[idx] = doi_line
            trailing_newline = "\n" if text.endswith("\n") else ""
            return "\n".join(lines) + trailing_newline

    # Insert after the canonical repository-access line if DOI line is missing.
    anchor = (
        "The full reproducible project is publicly accessible at "
        "`https://github.com/yidaki53/policy-classifier`. "
        "Submission and production versions should cite the exact release tag and commit hash used for manuscript generation."
    )
    if anchor in text:
        return text.replace(anchor, f"{anchor}\n\n{doi_line}", 1)

    raise ValueError("Could not find insertion anchor in Data Availability text")


def update_checklist_text(text: str, doi_url: str) -> str:
    replacement = (
        "- [x] Mint persistent archival DOI for the exact submission snapshot "
        f"and add DOI citation to Data Availability text: {doi_url}"
    )

    pattern = r"^[ \t]*- \[(?: |x|X)\][ \t]+Mint persistent archival DOI.*$"
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count > 0:
        return updated

    marker = "## Outstanding PLOS Submission Tasks"
    if marker in text:
        insertion = f"\n{replacement}\n"
        return text.replace(marker, f"{marker}{insertion}", 1)

    raise ValueError("Could not find DOI checklist item or insertion marker in checklist")


def _write_if_changed(path: Path, original: str, updated: str, dry_run: bool) -> bool:
    changed = original != updated
    if changed and not dry_run:
        path.write_text(updated, encoding="utf-8")
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Zenodo DOI into manuscript files")
    parser.add_argument("--tag", required=True, help="Release tag (for example: submission-2026-06-06-r3)")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo in owner/name form")
    parser.add_argument("--doi-url", default="", help="Use this DOI URL directly instead of Zenodo lookup")
    parser.add_argument("--zenodo-base-url", default="https://zenodo.org", help="Zenodo base URL")
    parser.add_argument("--max-wait-seconds", type=int, default=900, help="Max wait for DOI discovery")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Polling interval")
    parser.add_argument("--data-availability", type=Path, default=DEFAULT_DATA_AVAILABILITY)
    parser.add_argument("--checklist", type=Path, default=DEFAULT_CHECKLIST)
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    doi_url = args.doi_url.strip()
    if not doi_url:
        doi_url = discover_doi(
            base_url=args.zenodo_base_url,
            tag=args.tag,
            repo=args.repo,
            max_wait_seconds=args.max_wait_seconds,
            poll_seconds=args.poll_seconds,
        )

    if not doi_url.startswith("https://doi.org/"):
        raise ValueError(f"Invalid DOI URL format: {doi_url}")

    data_availability_text = args.data_availability.read_text(encoding="utf-8")
    checklist_text = args.checklist.read_text(encoding="utf-8")

    updated_data_availability = update_data_availability_text(
        text=data_availability_text,
        tag=args.tag,
        doi_url=doi_url,
    )
    updated_checklist = update_checklist_text(text=checklist_text, doi_url=doi_url)

    da_changed = _write_if_changed(
        path=args.data_availability,
        original=data_availability_text,
        updated=updated_data_availability,
        dry_run=args.dry_run,
    )
    checklist_changed = _write_if_changed(
        path=args.checklist,
        original=checklist_text,
        updated=updated_checklist,
        dry_run=args.dry_run,
    )

    mode = "DRY-RUN" if args.dry_run else "WRITE"
    print(f"{mode}: DOI synced: {doi_url}")
    print(f"{mode}: {args.data_availability} changed={da_changed}")
    print(f"{mode}: {args.checklist} changed={checklist_changed}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - return human-readable terminal failures
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
