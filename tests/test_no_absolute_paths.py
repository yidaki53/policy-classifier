"""Tests for the absolute-path leak guard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_no_absolute_paths import (  # noqa: E402
    ALLOWLIST_PREFIXES,
    SELF_EXEMPT,
    find_hits,
    scrub_text,
)


@pytest.mark.parametrize(
    "text",
    [
        'cd /home/robin/OneDrive/University and such/project',
        'See "/home/robin/project/manuscript/sections/01_title.md" for details',
        "rendered=/home/robin/project/build/x.md",
        "/home/robin",
        "/Users/alice/project",
    ],
)
def test_find_hits_detects_home_paths(text: str) -> None:
    assert find_hits(text), f"expected a hit in: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "cd /home/runner/work/policy-classifier",
        "installed under /usr/lib/python3",
        "output written to /tmp/build.log",
        "model cached in /opt/models",
        "relative path: manuscript/sections/01_title.md",
        "no paths at all here",
    ],
)
def test_find_hits_ignores_allowlisted_and_relative(text: str) -> None:
    assert not find_hits(text), f"unexpected hit in: {text!r}"


REPO_ROOT = "/home/alice/My Thesis/proj"


def test_scrub_preserves_repo_relative_tail() -> None:
    text = f'source: "{REPO_ROOT}/manuscript/sections/01_title.md"'
    scrubbed, count = scrub_text(text, "<REPO_ROOT>", REPO_ROOT)
    assert count == 1
    assert scrubbed == f'source: "<REPO_ROOT>/manuscript/sections/01_title.md"'
    assert "/home/alice" not in scrubbed


def test_scrub_handles_shell_escaped_root() -> None:
    escaped = REPO_ROOT.replace(" ", "\\ ")
    scrubbed, count = scrub_text(f"cd {escaped}", "<REPO_ROOT>", REPO_ROOT)
    assert count == 1
    assert scrubbed == "cd <REPO_ROOT>"


def test_scrub_handles_truncated_root() -> None:
    """Log previews clip long paths; a clipped root is still a leak."""
    scrubbed, count = scrub_text('x: "/home/alice/My', "<REPO_ROOT>", REPO_ROOT)
    assert count == 1
    assert "/home/alice" not in scrubbed


def test_scrub_handles_lowercased_root() -> None:
    """graphify normalises labels to lower case."""
    lower = REPO_ROOT.lower().replace(" ", "\\ ")
    scrubbed, count = scrub_text(f"cd {lower}", "<REPO_ROOT>", REPO_ROOT)
    assert count == 1
    assert "/home/alice" not in scrubbed


def test_scrub_does_not_mangle_path_separators() -> None:
    """Regression: a short truncation prefix must not match bare slashes."""
    scrubbed, count = scrub_text("rel a/b/c stays", "<REPO_ROOT>", REPO_ROOT)
    assert count == 0
    assert scrubbed == "rel a/b/c stays"


def test_scrub_preserves_allowlisted_paths() -> None:
    text = "CI path /home/runner/work/repo and local /home/alice/secret"
    scrubbed, count = scrub_text(text, "<REPO_ROOT>", REPO_ROOT)
    assert count == 1
    assert "/home/runner/work/repo" in scrubbed
    assert "/home/alice" not in scrubbed


def test_scrub_handles_multiple_paths_on_one_line() -> None:
    text = f"a: {REPO_ROOT}/a.md and b: {REPO_ROOT}/b.md"
    scrubbed, count = scrub_text(text, "<REPO_ROOT>", REPO_ROOT)
    assert count == 2
    assert scrubbed == "a: <REPO_ROOT>/a.md and b: <REPO_ROOT>/b.md"


def test_scrub_reports_zero_when_clean() -> None:
    scrubbed, count = scrub_text("manuscript/build/combined.md", "<REPO_ROOT>")
    assert count == 0
    assert scrubbed == "manuscript/build/combined.md"


def test_check_mode_fails_on_this_repository() -> None:
    """The repository must be free of home-path leaks.

    Guards the public remote: any commit that reintroduces an absolute
    path will fail CI.
    """
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "check_no_absolute_paths.py"),
            "--check",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "absolute path leak detected; run:\n"
        "  uv run python scripts/check_no_absolute_paths.py --fix\n"
        f"{result.stdout}"
    )


def test_allowlist_covers_ci_and_system_prefixes() -> None:
    for prefix in ("/home/runner/", "/usr/", "/opt/", "/tmp/"):
        assert prefix in ALLOWLIST_PREFIXES
