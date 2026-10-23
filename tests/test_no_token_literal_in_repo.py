import re

from _fsscan import is_probably_text, repo_root, tracked_files

# Hugging Face user-access tokens look like hf_<30+ alphanumerics>.
_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{30,}")


def test_no_hf_token_literal_in_any_tracked_file():
    root = repo_root()
    offenders = []
    for rel in tracked_files(root):
        if rel.name == ".env":
            # Never opened, even by name: the token must never be read here.
            continue
        if rel.parts and rel.parts[0] == "data":
            continue
        path = root / rel
        if not is_probably_text(path):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if _TOKEN_RE.search(text):
            offenders.append(str(rel))
    assert not offenders, f"token-shaped literal found in: {offenders}"
