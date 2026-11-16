import re
import subprocess

from _fsscan import is_probably_text, repo_root, tracked_files

_FORBIDDEN_PATH_RE = re.compile(r"(?i)claude|superpowers|anthropic|\.agent")
_FORBIDDEN_CONTENT_RE = re.compile(r"(?i)claude|anthropic|superpowers|co-authored-by")
_CONTENT_SCAN_EXCLUDE = {
    "uv.lock",
    "requirements.lock",
    # These files necessarily name the patterns/paths they search for or skip.
    "tests/test_no_agent_artifacts.py",
    "tests/_fsscan.py",
    # These reference the `.superpowers/` directory only to exclude it from
    # the Docker build context / assert that exclusion -- not as a mention.
    ".dockerignore",
    "tests/test_gitignore.py",
}


def test_no_agent_tooling_paths_are_tracked():
    root = repo_root()
    offenders = [str(p) for p in tracked_files(root) if _FORBIDDEN_PATH_RE.search(str(p))]
    assert not offenders, f"agent-tooling paths tracked: {offenders}"


def test_no_agent_mentions_in_tracked_file_contents():
    root = repo_root()
    offenders = []
    for rel in tracked_files(root):
        if str(rel) in _CONTENT_SCAN_EXCLUDE or rel.name == ".env":
            continue
        path = root / rel
        if not is_probably_text(path):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if _FORBIDDEN_CONTENT_RE.search(text):
            offenders.append(str(rel))
    assert not offenders, f"agent-tooling mention found in: {offenders}"


def test_no_commit_message_contains_a_coauthor_trailer():
    root = repo_root()
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "log", "--format=%B"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        # git isn't available in this environment (or a worktree's gitdir
        # lives outside this mount); nothing to check from inside.
        return
    if result.returncode != 0:
        return
    offenders = [
        line for line in result.stdout.splitlines() if re.search(r"(?i)co-authored-by", line)
    ]
    assert not offenders, f"commit message contains a co-author trailer: {offenders}"
