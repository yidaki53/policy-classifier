#!/usr/bin/env python3
"""Create a blinded manuscript variant for peer review submission.

This script removes or replaces identifying content in a combined manuscript
markdown file while preserving the core scientific content and structure.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ANON_DATA_AVAILABILITY = """# Data Availability

All data and metadata underlying the findings are available from publicly
accessible parliamentary open-data sources and from a reproducible project
repository. To preserve blinded peer review, direct repository identifiers
and release metadata are redacted in this version and will be provided in the
camera-ready submission package.

All scripts required to reproduce ingest, classification, linkage, analysis,
and figure generation are included in the reproducible package. Build context
and journal-readiness reports are generated during manuscript build and will be
provided to editors upon request.
"""


ANON_ACKNOWLEDGMENTS = """# Acknowledgments

Acknowledgments, author-contribution metadata (CRediT), and identifying
administrative details are redacted for blinded peer review and will be
restored in the camera-ready version.
"""


def split_top_level_sections(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    sections: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("# ") and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)

    return ["".join(section) for section in sections]


def anonymize_sections(text: str) -> str:
    sections = split_top_level_sections(text)
    anonymized: list[str] = []

    for section in sections:
        heading = section.splitlines()[0].strip() if section.strip() else ""
        if heading == "# Data Availability":
            anonymized.append(ANON_DATA_AVAILABILITY.strip() + "\n\n")
            continue
        if heading == "# Acknowledgments":
            anonymized.append(ANON_ACKNOWLEDGMENTS.strip() + "\n\n")
            continue
        anonymized.append(section)

    return "".join(anonymized).rstrip() + "\n"


def redact_identifiers(text: str) -> str:
    replacements = {
        "Robin Oberg": "[author redacted for peer review]",
        "robin.oberg@klarian.com": "[email redacted for peer review]",
        "robinoberg@live.com": "[email redacted for peer review]",
        "yidaki53": "[repository-owner redacted]",
        "robin-oberg": "[repository-owner redacted]",
    }

    redacted = text
    for old, new in replacements.items():
        redacted = redacted.replace(old, new)

    redacted = re.sub(
        r"https://github\.com/[^\s`]+",
        "[repository URL redacted for peer review]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)single-author",
        "author-anonymized",
        redacted,
    )
    return redacted


def add_blinding_notice(text: str) -> str:
    notice = (
        "> **Blinded peer-review version**\n"
        "> Identifying author and repository metadata are redacted in this file.\n\n"
    )
    return notice + text


def anonymize_markdown(text: str) -> str:
    blinded = anonymize_sections(text)
    blinded = redact_identifiers(blinded)
    blinded = add_blinding_notice(blinded)
    return blinded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create anonymized manuscript markdown")
    parser.add_argument("--in", dest="input_path", required=True, help="Input markdown path")
    parser.add_argument("--out", dest="output_path", required=True, help="Output markdown path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    text = input_path.read_text(encoding="utf-8")
    output = anonymize_markdown(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")

    print(f"Wrote anonymized manuscript to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
