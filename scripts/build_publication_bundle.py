#!/usr/bin/env python3
"""Create an immutable publication bundle from manuscript, figure, and output artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_command(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def _iter_files(root: Path, artifact_roots: list[str]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for rel_root in artifact_roots:
        base = root / rel_root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel_path = path.relative_to(root).as_posix()
            files.append(
                {
                    "path": rel_path,
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return files


def build_publication_bundle(
    *,
    root: str | Path,
    output_dir: str | Path,
    tag: str,
    commit_sha: str | None,
    title: str,
    artifact_roots: list[str],
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    files = _iter_files(root_path, artifact_roots)
    resolved_commit = commit_sha or _git_command(root_path, "rev-parse", "HEAD") or "unknown"
    manifest = {
        "tag": tag,
        "commit_sha": resolved_commit,
        "branch": _git_command(root_path, "rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "title": title,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_roots": artifact_roots,
        "file_count": len(files),
        "files": files,
    }

    manifest_path = output_path / "publication_bundle_manifest.json"
    archive_path = output_path / "publication_bundle.tar.gz"

    manifest_tmp = None
    archive_tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, dir=output_path, encoding="utf-8") as handle:
            manifest_tmp = Path(handle.name)
            handle.write(json.dumps(manifest, indent=2))
        os.replace(manifest_tmp, manifest_path)

        with tempfile.NamedTemporaryFile("wb", delete=False, dir=output_path) as handle:
            archive_tmp = Path(handle.name)
        with tarfile.open(archive_tmp, mode="w:gz") as archive:
            for item in files:
                source = root_path / item["path"]
                archive.add(source, arcname=item["path"])
            archive.add(manifest_path, arcname=manifest_path.name)
        os.replace(archive_tmp, archive_path)
    finally:
        for tmp_path in (manifest_tmp, archive_tmp):
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a publication bundle")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="output/publication_bundle")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit-sha")
    parser.add_argument("--title", default="Publication bundle")
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        help="Artifact root to include (repeatable)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_roots = args.artifact_root or [
        "manuscript/build",
        "figures",
        "output/analysis",
        "output/manuscript",
    ]
    build_publication_bundle(
        root=args.root,
        output_dir=args.output_dir,
        tag=args.tag,
        commit_sha=args.commit_sha,
        title=args.title,
        artifact_roots=artifact_roots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
