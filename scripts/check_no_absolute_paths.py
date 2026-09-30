#!/usr/bin/env python3
"""Detect and scrub absolute filesystem paths from tracked repository files.

Absolute paths leak the author's home directory, username, and institution
onto a public remote. This module finds occurrences of the repository root
(or any user home prefix) in text files and can rewrite them to a
placeholder.

Usage:
    python scripts/check_no_absolute_paths.py --check    # exit 1 if any found
    python scripts/check_no_absolute_paths.py --fix      # rewrite in place
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

LOG = "checked %d files, %d replacement(s)"

# Matches /home/<user>/... and /Users/<user>/... with at least one segment.
# Deliberately broad: any user home prefix is a leak, not just this repo's.
HOME_ROOT = "/home/"
HOME_PATH_RE = re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+(?:/[^\s\"'`,)\]}]*)?")

# Text files only. Binary artifacts (safetensors, .pkl, .bin) are skipped.
BINARY_SUFFIXES = {
    ".bin", ".npy", ".npz", ".onnx", ".pt", ".pth", ".pkl", ".safetensors",
    ".zst", ".zip", ".parquet", ".joblib", ".db", ".sqlite", ".pdf",
}

# Paths matching these are legitimate: they point at the agent's own tooling
# or at system locations, not at the user's home directory.
ALLOWLIST_PREFIXES = ("/home/runner/", "/usr/", "/opt/", "/tmp/")

TEXT_SUFFIXES = {
    ".md", ".json", ".txt", ".yml", ".yaml", ".py", ".sh", ".cfg", ".toml",
    ".jsonl", ".csv", "", ".mdc", ".log",
}

# The checker and its tests necessarily contain example home paths, which is
# the whole point of them. Scanning them would make the check unsatisfiable.
SELF_EXEMPT = {
    "scripts/check_no_absolute_paths.py",
    "tests/test_no_absolute_paths.py",
}


def iter_tracked_files(repo_root: Path) -> list[Path]:
    """Return tracked files that are text and eligible for scanning."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    files = []
    for raw in result.stdout.split("\0"):
        if not raw:
            continue
        if raw.replace("\\", "/") in SELF_EXEMPT:
            continue
        path = repo_root / raw
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if not path.is_file():
            continue
        files.append(path)
    return files


def find_hits(text: str) -> list[re.Match[str]]:
    """Return matches for absolute home paths, excluding allowlisted ones."""
    return [
        m
        for m in HOME_PATH_RE.finditer(text)
        if not m.group(0).startswith(ALLOWLIST_PREFIXES)
    ]


def _min_root_prefix_length(repo_root: str) -> int:
    """Length of `/home/<user>/` — the shortest prefix that identifies a user.

    Truncations shorter than this would also match unrelated users such as
    `/home/runner`, so they are never considered.
    """
    if not repo_root.startswith(HOME_ROOT):
        return len(HOME_ROOT)
    rest = repo_root[len(HOME_ROOT):]
    slash = rest.find("/")
    return len(HOME_ROOT) + (slash + 1 if slash != -1 else len(rest))


def _root_variants(repo_root: str) -> list[str]:
    """Return the repo-root spellings to look for, longest first.

    Covers the plain form, the shell-escaped form (`My\\ Papers`), the
    lower-cased form that graphify labels use, and the truncated forms
    produced by log previews, which clip long paths to a character budget.
    """
    variants = {repo_root, repo_root.replace(" ", "\\ ")}
    floor = _min_root_prefix_length(repo_root)

    def add_prefixes(base: str) -> None:
        for cut in range(len(base) - 1, floor, -1):
            prefix = base[:cut]
            variants.add(prefix)
            variants.add(prefix.replace(" ", "\\ "))

    add_prefixes(repo_root)

    # Lowercased spellings: graphify normalises labels to lower case, so the
    # same path can appear as /home/robin/onedrive/university and such/...
    lower = repo_root.lower()
    if lower != repo_root:
        variants.add(lower)
        variants.add(lower.replace(" ", "\\ "))
        add_prefixes(lower)

    # Longest first so the full path wins over its own truncations.
    return sorted(variants, key=len, reverse=True)


def scrub_text(
    text: str, placeholder: str, repo_root: str | None = None
) -> tuple[str, int]:
    """Replace home paths with a placeholder.

    Paths under the repository root collapse to the placeholder plus their
    repo-relative tail, so the record stays meaningful:
        <root>/manuscript/x.md  ->  <REPO_ROOT>/manuscript/x.md

    The root is matched by literal string replacement (not regex) because
    directory names contain spaces. A regex pass afterwards catches any
    remaining `/home/<user>/...` path that is not under this repository, and
    collapses it to the placeholder entirely.
    """
    count = 0

    if repo_root:
        # Longest first: the full root must win over its own truncations.
        # Only the root span is replaced, so any repo-relative tail that
        # follows it in the text is preserved untouched.
        for variant in _root_variants(repo_root):
            if not variant:
                continue
            occurrences = text.count(variant)
            if not occurrences:
                continue
            count += occurrences
            text = text.replace(variant, placeholder)

    def _replace(match: re.Match[str]) -> str:
        nonlocal count
        if match.group(0).startswith(ALLOWLIST_PREFIXES):
            return match.group(0)
        count += 1
        return placeholder

    return HOME_PATH_RE.sub(_replace, text), count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="report only, exit 1 on findings"
    )
    parser.add_argument("--fix", action="store_true", help="rewrite in place")
    parser.add_argument(
        "--placeholder",
        default="<REPO_ROOT>",
        help="replacement token (default: <REPO_ROOT>)",
    )
    parser.add_argument("--repo-root", default=".", help="repository root")
    args = parser.parse_args(argv)

    if not (args.check or args.fix):
        parser.error("pass exactly one of --check or --fix")

    repo_root = Path(args.repo_root).resolve()
    files = iter_tracked_files(repo_root)

    offenders: list[tuple[Path, int]] = []
    total = 0

    for path in files:
        try:
            # Read as bytes: read_text() would translate the carriage returns
            # in tqdm progress bars into newlines and corrupt the file.
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        if args.fix:
            new_text, count = scrub_text(text, args.placeholder, str(repo_root))
            if count:
                # Write bytes for the same reason; encode() leaves line
                # endings exactly as they were.
                path.write_bytes(new_text.encode("utf-8"))
                total += count
                offenders.append((path, count))
        else:
            count = len(find_hits(text))
            if count:
                total += count
                offenders.append((path, count))

    for path, count in offenders:
        rel = path.relative_to(repo_root)
        print(f"{rel}: {count} occurrence(s)")

    print(LOG % (len(files), total))

    if args.fix:
        return 0
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
