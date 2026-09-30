"""Preflight checks for manuscript section artifact contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SectionPreflightResult:
    section_id: str
    missing_inputs: tuple[str, ...]
    missing_figures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_inputs and not self.missing_figures

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "ok": self.ok,
            "missing_inputs": list(self.missing_inputs),
            "missing_figures": list(self.missing_figures),
        }


def _frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text:
        return {}
    block = text[4 : text.index("\n---\n", 4)]
    parsed = yaml.safe_load(block)
    return parsed if isinstance(parsed, dict) else {}


def validate_manuscript_sections(
    manuscript_root: str | Path,
    *,
    sections_dir: str | Path | None = None,
) -> list[SectionPreflightResult]:
    root = Path(manuscript_root)
    directory = Path(sections_dir) if sections_dir is not None else root / "sections"
    results: list[SectionPreflightResult] = []
    for section_path in sorted(directory.glob("*.md")):
        metadata = _frontmatter(section_path)
        required_inputs = [str(item) for item in metadata.get("required_inputs", [])]
        required_figures = [str(item) for item in metadata.get("required_figures_tables", [])]
        results.append(
            SectionPreflightResult(
                section_id=str(metadata.get("section_id", section_path.stem)),
                missing_inputs=tuple(item for item in required_inputs if not (root / item).exists()),
                missing_figures=tuple(item for item in required_figures if not (root / item).exists()),
            )
        )
    return results


def assert_manuscript_ready(manuscript_root: str | Path) -> list[SectionPreflightResult]:
    results = validate_manuscript_sections(manuscript_root)
    failures = [result for result in results if not result.ok]
    if failures:
        missing = {
            result.section_id: [*result.missing_inputs, *result.missing_figures]
            for result in failures
        }
        raise FileNotFoundError(f"Manuscript prerequisites missing: {missing}")
    return results


__all__ = ["SectionPreflightResult", "assert_manuscript_ready", "validate_manuscript_sections"]