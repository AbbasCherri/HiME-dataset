import subprocess

from _fsscan import repo_root


def _strip_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith("#"))


def test_uv_lock_check_passes():
    root = repo_root()
    result = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_requirements_lock_matches_a_fresh_export(tmp_path):
    root = repo_root()
    fresh = tmp_path / "requirements.lock"
    result = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--hashes",
            "--no-dev",
            "--no-emit-project",
            "-o",
            str(fresh),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    committed = _strip_comments((root / "requirements.lock").read_text())
    assert committed == _strip_comments(fresh.read_text())
