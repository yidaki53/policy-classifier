"""Versioned definitions registry for stable category snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from swedish_parliament_policy_classifier.models.models import CategoryDef
from swedish_parliament_policy_classifier.definitions.loader import load_verified_definitions


@dataclass(frozen=True)
class DefinitionSnapshot:
    version: str
    created_at: str
    categories: Dict[str, CategoryDef]


def snapshot_definitions(version_prefix: str = "defs") -> DefinitionSnapshot:
    cats = load_verified_definitions()
    payload = {k: v.model_dump() for k, v in sorted(cats.items())}
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()[:12]
    version = f"{version_prefix}-{digest}"
    created_at = datetime.now(timezone.utc).isoformat()
    return DefinitionSnapshot(version=version, created_at=created_at, categories=cats)


def write_snapshot_manifest(path: str | Path, snap: DefinitionSnapshot) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": snap.version,
        "created_at": snap.created_at,
        "categories": sorted(list(snap.categories.keys())),
    }
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
