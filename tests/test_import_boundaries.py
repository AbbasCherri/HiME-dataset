"""Skeleton hygiene check (full enforcement lands once tiles.py/source.py exist).

Only `tiles.py` may import `requests`; only `source.py` may import
`huggingface_hub` or `datasets`. Today neither file exists, so this passes
vacuously -- the point is that it starts failing the moment some other
module reaches for one of these imports.
"""

import ast
from pathlib import Path

from _fsscan import repo_root

_FORBIDDEN = {
    "requests": {"tiles.py"},
    "huggingface_hub": {"source.py"},
    "datasets": {"source.py"},
}


def _imported_top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_only_designated_modules_import_their_external_dependency():
    src = repo_root() / "src" / "notodata"
    violations = []
    for path in sorted(src.glob("*.py")):
        imported = _imported_top_level_names(path)
        for forbidden_name, allowed_files in _FORBIDDEN.items():
            if forbidden_name in imported and path.name not in allowed_files:
                violations.append(f"{path.name} imports {forbidden_name}")
    assert not violations, violations
