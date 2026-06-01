"""Verified, immutable loader for political_spectrum.yaml.

This module mirrors the repository's top-level `definitions.loader` so that
imports resolving under the packaged `swedish_parliament_policy_classifier`
namespace work regardless of execution context.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Dict

import yaml

from swedish_parliament_policy_classifier.models.models import CategoryDef

_YAML_PATH = Path(__file__).resolve().parent / "political_spectrum.yaml"
_CHECKSUM_PATTERN = re.compile(r'checksum:\s*"([^"]+)"')
_PLACEHOLDER = "PLACEHOLDER"


def _neutralise(content: str) -> str:
    """Replace the checksum value with PLACEHOLDER so the hash is stable."""
    return _CHECKSUM_PATTERN.sub(f'checksum: "{_PLACEHOLDER}"', content)


def _compute_checksum(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    return hashlib.sha256(_neutralise(content).encode("utf-8")).hexdigest()


def _read_stored_checksum(path: Path) -> str | None:
    content = path.read_text(encoding="utf-8")
    m = _CHECKSUM_PATTERN.search(content)
    return m.group(1) if m else None


def verify(path: Path = _YAML_PATH) -> bool:
    """Return True if the file matches its embedded checksum."""
    stored = _read_stored_checksum(path)
    if stored is None or stored == _PLACEHOLDER:
        return False
    return stored == _compute_checksum(path)


def load_verified_definitions(
    yaml_path: Path = _YAML_PATH, *, strict: bool = True
) -> Dict[str, CategoryDef]:
    """Load and validate category definitions.

    Parameters
    ----------
    yaml_path : Path
        Path to political_spectrum.yaml.
    strict : bool
        If True (default), raise ValueError when the checksum does not match
        (file was edited without updating the checksum).  Set to False only in
        tests where you intentionally mutate the definitions.

    Returns
    -------
    dict mapping category name -> CategoryDef
    """
    if strict and not verify(yaml_path):
        stored = _read_stored_checksum(yaml_path)
        computed = _compute_checksum(yaml_path)
        raise ValueError(
            f"Category definitions integrity check failed!\n"
            f"  file   : {yaml_path}\n"
            f"  stored : {stored}\n"
            f"  actual : {computed}\n"
            f"If this edit was intentional, run:\n"
            f"  python -m definitions.loader --recheck"
        )

    with open(yaml_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    categories: Dict[str, CategoryDef] = {}
    for entry in raw.get("categories", []):
        cat = CategoryDef(**{k: v for k, v in entry.items() if k in CategoryDef.model_fields})
        categories[cat.name] = cat

    return categories


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _cmd_verify(args) -> int:
    path = Path(args.file)
    ok = verify(path)
    if ok:
        print(f"OK  checksum verified: {path}")
        return 0
    stored = _read_stored_checksum(path)
    computed = _compute_checksum(path)
    print(f"FAIL  checksum mismatch in {path}")
    print(f"  stored : {stored}")
    print(f"  actual : {computed}")
    return 1


def _cmd_recheck(args) -> int:
    path = Path(args.file)
    new_checksum = _compute_checksum(path)
    content = path.read_text(encoding="utf-8")
    updated = _CHECKSUM_PATTERN.sub(f'checksum: "{new_checksum}"', content)
    path.write_text(updated, encoding="utf-8")
    print(f"Updated checksum in {path}")
    print(f"  new checksum: {new_checksum}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify or update the political_spectrum.yaml integrity checksum."
    )
    parser.add_argument(
        "--file",
        default=str(_YAML_PATH),
        help="Path to political_spectrum.yaml (default: bundled file)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--verify",
        action="store_true",
        help="Verify the embedded checksum and exit 0 if OK, 1 if mismatch.",
    )
    group.add_argument(
        "--recheck",
        action="store_true",
        help="Recompute and overwrite the embedded checksum (use after intentional edits).",
    )
    args = parser.parse_args()

    if args.verify:
        sys.exit(_cmd_verify(args))
    else:
        sys.exit(_cmd_recheck(args))


if __name__ == "__main__":
    main()
