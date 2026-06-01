#!/usr/bin/env python3
"""Validate manuscript readiness against a target journal profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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

    failed = [c for c in checks if c["status"] != "pass"]
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
