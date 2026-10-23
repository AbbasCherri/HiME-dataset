from _fsscan import repo_root


def test_gitignore_excludes_data_and_env():
    lines = {line.strip() for line in (repo_root() / ".gitignore").read_text().splitlines()}
    assert "data/" in lines
    assert ".env" in lines


def test_gitignore_excludes_audit_keys_before_unblinding():
    text = (repo_root() / ".gitignore").read_text().lower()
    assert "audit" in text
    assert "key" in text


def test_dockerignore_excludes_the_same_paths():
    text = (repo_root() / ".dockerignore").read_text()
    assert "data/" in text
    assert ".env" in text
    lowered = text.lower()
    assert "audit" in lowered
    assert "key" in lowered
