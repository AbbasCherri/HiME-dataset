"""Shared helpers for repo-hygiene tests (gitignore, token and agent-artifact scans).

`git ls-files` is the authoritative tracked-file list, but a git worktree
whose `.git` file points at a gitdir outside the current bind mount can't
resolve it from inside a container. These helpers fall back to a plain
filesystem walk that excludes the directories and files this repo's own
`.gitignore` always excludes, so the scans still work in that case.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_ALWAYS_EXCLUDED_DIRS = {
    ".git",
    "data",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    # Never tracked: kept out of the repo via local git excludes, not
    # .gitignore, so a plain filesystem walk must drop them by hand too.
    "docs",
    ".superpowers",
}
_ALWAYS_EXCLUDED_FILES = {".env"}

_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".yml",
    ".yaml",
    ".cfg",
    ".ini",
    ".txt",
    ".sh",
    "",
    ".json",
    ".csv",
}


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("could not locate repo root (no pyproject.toml found)")


def tracked_files(root: Path | None = None) -> list[Path]:
    """Tracked files as paths relative to the repo root.

    Uses `git ls-files` when it succeeds and returns something; otherwise
    walks the tree, skipping what this repo always ignores.
    """
    root = root or repo_root()
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        result = None
    if result is not None and result.returncode == 0 and result.stdout.strip():
        return [Path(line) for line in result.stdout.splitlines() if line]

    out: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(root)
        if any(part in _ALWAYS_EXCLUDED_DIRS for part in rel.parts):
            continue
        if rel.name in _ALWAYS_EXCLUDED_FILES:
            continue
        out.append(rel)
    return out


def is_probably_text(path: Path) -> bool:
    if path.name in {"Dockerfile", "Makefile"}:
        return True
    return path.suffix.lower() in _TEXT_SUFFIXES
