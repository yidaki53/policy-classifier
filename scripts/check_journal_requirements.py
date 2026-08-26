#!/usr/bin/env python3
"""Validate manuscript readiness against a target journal profile."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import httpx
import yaml


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _count_heading_levels(text: str) -> int:
    max_level = 0
    for line in text.splitlines():
        s = line.lstrip()
        if s.startswith("#"):
            level = len(s) - len(s.lstrip("#"))
            max_level = max(max_level, level)
    return max_level


def _extract_bib_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    entries: list[dict[str, Any]] = []
    for match in re.finditer(r"@\w+\{([^,]+),", text):
        entry_id = match.group(1).strip()
        start = match.start()
        end = len(text)
        next_match = re.search(r"@\w+\{", text[match.end():])
        if next_match:
            end = match.end() + next_match.start()
        entries.append({"id": entry_id, "raw": text[start:end]})
    return entries


def _parse_bib_field(text: str, field: str) -> str | None:
    pattern = re.compile(rf"\b{field}\s*=\s*\{{([^}}]+)\}}", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None


def _iter_bib_fields(text: str) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    for match in re.finditer(r"([A-Za-z]+)\s*=\s*\{([^{}]+)\}", text):
        fields.append((match.group(1).lower(), match.group(2).strip()))
    return fields


def _normalise_author(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(
        str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-"})
    )
    return re.sub(r"\s+", " ", normalized.strip()).lower()


def _fetch_bibliography_metadata(doi: str) -> dict[str, Any] | None:
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(
                f"https://api.crossref.org/works/{doi}",
                headers={"User-Agent": "policy-classifier/1.0"},
            )
            response.raise_for_status()
            payload = response.json()
            message = payload.get("message", {})
            title = message.get("title", [""])[0]
            authors = [
                author.get("family", "") + (", " + author.get("given", "") if author.get("given") else "")
                for author in message.get("author", [])
            ]
            return {"title": title, "authors": [a.strip() for a in authors if a.strip()]}
    except Exception:
        return None


def _validate_bibliography_entry(entry: dict[str, Any]) -> tuple[bool, str]:
    raw = entry.get("raw", "")
    doi = _parse_bib_field(raw, "doi")
    title = _parse_bib_field(raw, "title")
    authors = _parse_bib_field(raw, "author")
    if not doi:
        return True, "No DOI to verify"

    metadata = _fetch_bibliography_metadata(doi)
    if metadata is None:
        return False, f"Could not verify DOI metadata for {doi}"

    normalized_title = _normalise_author(title or "")
    remote_title = _normalise_author(metadata.get("title", ""))
    remote_authors = {_normalise_author(author) for author in metadata.get("authors", [])}
    local_authors = {_normalise_author(author) for author in re.split(r"\s+and\s+", authors or "") if author}

    if normalized_title and remote_title and normalized_title not in remote_title and remote_title not in normalized_title:
        return False, f"Title mismatch for DOI {doi}: local={title!r} remote={metadata.get('title')!r}"

    if authors and remote_authors and not local_authors.issubset(remote_authors):
        return False, f"Author mismatch for DOI {doi}"

    return True, ""


def _run_checks(repo_root: Path, manuscript_dir: Path, profile: dict) -> dict:
    sections_dir = manuscript_dir / "sections"
    section_paths = sorted(sections_dir.glob("*.md"))
    all_text = "\n\n".join(p.read_text(encoding="utf-8") for p in section_paths)

    checks = []

    checks.append(
        {
            "id": "has_abstract",
            "status": "pass" if "abstract" in all_text.lower() else "warn",
            "detail": "Abstract section present in manuscript text" if "abstract" in all_text.lower() else "Add an explicit abstract section",
        }
    )

    checks.append(
        {
            "id": "has_data_availability",
            "status": "pass" if "data availability" in all_text.lower() else "warn",
            "detail": "Data availability statement detected" if "data availability" in all_text.lower() else "Add a Data Availability statement aligned with journal policy",
        }
    )

    bib = manuscript_dir / "bibliography" / "references.bib"
    bib_entries = 0
    if bib.exists():
        bib_entries = sum(1 for line in bib.read_text(encoding="utf-8").splitlines() if line.strip().startswith("@"))
    checks.append(
        {
            "id": "bibliography_seeded",
            "status": "pass" if bib_entries >= 10 else "warn",
            "detail": f"{bib_entries} bibliography entries in {bib.relative_to(repo_root)}",
        }
    )

    doi_metadata_status = "pass"
    doi_metadata_detail = "All bibliography DOIs resolved to matching title/author metadata"
    if bib.exists():
        for entry in _extract_bib_entries(bib):
            ok, detail = _validate_bibliography_entry(entry)
            if not ok:
                doi_metadata_status = "warn"
                doi_metadata_detail = detail
                break
    checks.append(
        {
            "id": "bibliography_doi_metadata",
            "status": doi_metadata_status,
            "detail": doi_metadata_detail,
            "blocking": False,
        }
    )

    max_h = 0
    for p in section_paths:
        max_h = max(max_h, _count_heading_levels(p.read_text(encoding="utf-8")))
    checks.append(
        {
            "id": "heading_depth",
            "status": "pass" if max_h <= 3 else "warn",
            "detail": f"Maximum heading depth in sections: {max_h}",
        }
    )

    checks.append(
        {
            "id": "journal_profile_loaded",
            "status": "pass" if profile else "warn",
            "detail": f"Journal profile loaded: {profile.get('name', 'missing')}",
        }
    )

    failed = [c for c in checks if c["status"] != "pass" and c.get("blocking", True)]
    return {
        "target_journal": profile.get("name", "unknown"),
        "status": "ready" if not failed else "needs-attention",
        "checks": checks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Check manuscript against target journal requirements")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--manuscript-dir", default="manuscript")
    ap.add_argument("--journal-profile", default="manuscript/journal_profiles/plos_one.yaml")
    ap.add_argument("--out", default="manuscript/build/journal_requirements_report.json")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manuscript_dir = (repo_root / args.manuscript_dir).resolve()
    profile = _load_yaml((repo_root / args.journal_profile).resolve())

    report = _run_checks(repo_root, manuscript_dir, profile)

    out = (repo_root / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
