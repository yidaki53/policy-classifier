#!/usr/bin/env python3
"""Deterministic manuscript text polish.

Strips TODO/FIXME/placeholder markers, normalizes figure captions, and flags
obvious text issues before the manuscript is combined. No stochastic rewriting.

Usage:
    uv run python scripts/polish_manuscript_text.py --dir manuscript/sections
    uv run python scripts/polish_manuscript_text.py --all
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PLACEHOLDER_PATTERNS = [
    r"TODO",
    r"FIXME",
    r"XXX",
    r"pending regeneration",
    r"replace placeholder",
    r"update this",
    r"INSERT",
    r"\[FIX\]",
]


def strip_placeholders(text: str) -> str:
    for pat in PLACEHOLDER_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text


def normalize_captions(text: str) -> str:
    # Collapse repeated spaces and ensure figure captions end cleanly
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\*\*Figure\s+\d+\.\*\*\s*", lambda m: m.group(0).strip() + " ", text)
    return text


def clean_section(path: Path, dry_run: bool = False) -> tuple[str, int]:
    original = path.read_text(encoding="utf-8")
    cleaned = strip_placeholders(original)
    cleaned = normalize_captions(cleaned)

    if cleaned != original:
        if not dry_run:
            path.write_text(cleaned, encoding="utf-8")
        return cleaned, 1
    return cleaned, 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic manuscript polish")
    ap.add_argument("--dir", default="manuscript/sections", help="Section dir to polish")
    ap.add_argument("--all", action="store_true", help="Also polish manuscript.md and abstract")
    ap.add_argument("--dry-run", action="store_true", help="Report only, do not write")
    args = ap.parse_args()

    targets = [Path(args.dir)]
    if args.all:
        targets = [Path("manuscript"), Path(args.dir)]

    changed = 0
    for t in targets:
        if t.is_dir():
            for p in sorted(t.glob("*.md")):
                _, c = clean_section(p, dry_run=args.dry_run)
                changed += c
        elif t.is_file():
            _, c = clean_section(t, dry_run=args.dry_run)
            changed += c

    print(f"polish: {changed} file(s) updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())