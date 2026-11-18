"""Import boundaries: each external dependency has one designated module.

Only `tiles.py` may import `requests`; only `source.py` may import geopandas or
pyogrio. Raw data never goes through the Hugging Face libraries (rulings P9-P12), so
no module may import `datasets`; only `publish.py` (the upload of the finished dataset)
may import `huggingface_hub`.
"""

import ast
from pathlib import Path

from _fsscan import repo_root

_FORBIDDEN = {
    "requests": {"tiles.py"},
    "geopandas": {"source.py"},
    "pyogrio": {"source.py"},
    "huggingface_hub": {"publish.py"},
    "datasets": set(),
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
