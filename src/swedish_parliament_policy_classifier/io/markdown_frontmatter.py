"""Utilities for ensuring YAML frontmatter in generated Markdown files."""

from __future__ import annotations

from typing import Any

import yaml


def has_frontmatter(text: str) -> bool:
    if not text.startswith("---\n"):
        return False
    return "\n---\n" in text


def _frontmatter_block(metadata: dict[str, Any]) -> str:
    payload = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=False).strip()
    return f"---\n{payload}\n---\n\n"


def ensure_frontmatter(text: str, metadata: dict[str, Any]) -> str:
    """Prepend YAML frontmatter when absent.

    Existing frontmatter is preserved as-is.
    """
    if has_frontmatter(text):
        return text
    return _frontmatter_block(metadata) + text.lstrip("\n")
