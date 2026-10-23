"""`notodata report`: rewrite the AUTO blocks of DATASET.md from `reports/` and the logs.

Only text between `<!-- AUTO:name -->` and `<!-- /AUTO -->` is ever replaced;
everything else in the document is hand-written and stays byte-identical. A
block whose inputs do not exist yet is left exactly as it was.
"""

from __future__ import annotations

import json
import re
from importlib import metadata
from pathlib import Path

import pandas as pd

from notodata import manifest, paths, source
from notodata.config import CONFIG
from notodata.io import atomic_write
from notodata.layers import LAYERS

_AUTO = re.compile(r"(<!-- AUTO:(?P<name>\w+) -->)(?P<body>.*?)(<!-- /AUTO -->)", re.DOTALL)

# AUTO block -> the CSV in `reports/` it renders
_CSV_BLOCKS = {
    "class_counts_raw": "class_counts_raw.csv",
    "class_counts_mapped": "class_counts_mapped.csv",
    "coverage_by_layer": "coverage_by_layer.csv",
    "audit": "audit.csv",
    "dropped": "dropped_by_post_coverage.csv",
    "coverage_by_pattern": "coverage_by_pattern.csv",
    "moran": "moran.csv",
    "folds": "folds.csv",
    "splits_summary": "splits_summary.csv",
    "clients_summary": "clients_summary.csv",
}


def markdown_table(df: pd.DataFrame) -> str:
    """A pipe table; cells are the text as written, so no number is reformatted."""
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in df.fillna("").astype(str).to_numpy().tolist()]
    return "\n".join([head, rule, *rows])


def _csv_block(path: Path) -> str | None:
    if not path.exists():
        return None
    return markdown_table(pd.read_csv(path, dtype=str, keep_default_na=False))


def _source_files_block() -> str | None:
    record = source.read_source_files()
    if not record:
        return None
    rows = [(name, str(v["size"]), v["sha256"]) for name, v in sorted(record.items())]
    return markdown_table(pd.DataFrame(rows, columns=["file", "size", "sha256"]))


def _layers_block() -> str:
    rows = [
        (
            layer.layer_id,
            layer.role_candidate,
            " to ".join(d.isoformat() for d in layer.capture_dates) or layer.capture_dates_note,
            ", ".join(layer.districts),
            layer.url_template,
            layer.found_in,
        )
        for layer in LAYERS
    ]
    cols = ["id", "role", "capture dates", "districts", "template", "found in"]
    return markdown_table(pd.DataFrame(rows, columns=cols))


def _tile_fingerprints_block() -> str | None:
    rows = []
    for layer in LAYERS:
        if paths.tiles_dir(layer.layer_id).is_dir():
            digest, count = manifest.tile_fingerprint(layer.layer_id)
            rows.append((layer.layer_id, str(count), digest))
    if not rows:
        return None
    return markdown_table(pd.DataFrame(rows, columns=["layer", "tiles", "sha256"]))


def _pip_freeze_block() -> str:
    pins = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    return "```\n" + "\n".join(pins) + "\n```"


def _seed_register_block() -> str:
    seeds = ", ".join(str(s) for s in CONFIG.test_seeds)
    return f"Default seeds: {seeds}. Folds: {CONFIG.folds.n_folds}; audit seed {CONFIG.audit.seed}."


def _verify_block(root: Path) -> str | None:
    path = paths.reports_dir(root) / "verify_result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    head = f"{result['date']}, mode `{result['mode']}`: **{result['status']}**"
    files = pd.DataFrame(sorted(result["files"].items()), columns=["file", "result"])
    return head + ("\n\n" + markdown_table(files) if len(files) else "")


def render_blocks(root: Path | None = None) -> dict[str, str]:
    """Every AUTO block that can be rendered from what exists now."""
    reports = paths.reports_dir(root)
    candidates = {name: _csv_block(reports / csv) for name, csv in _CSV_BLOCKS.items()}
    candidates["source_files"] = _source_files_block()
    candidates["layers"] = _layers_block()
    candidates["tile_fingerprints"] = _tile_fingerprints_block()
    candidates["pip_freeze"] = _pip_freeze_block()
    candidates["seed_register"] = _seed_register_block()
    candidates["verify_result"] = _verify_block(root or Path())
    return {name: body for name, body in candidates.items() if body is not None}


def replace_auto_blocks(text: str, blocks: dict[str, str]) -> str:
    def swap(m: re.Match) -> str:
        body = blocks.get(m.group("name"))
        if body is None:
            return m.group(0)
        return f"{m.group(1)}\n{body}\n{m.group(4)}"

    return _AUTO.sub(swap, text)


def update_dataset_md(path: Path | str, root: Path | None = None) -> bool:
    """Rewrite the AUTO blocks in place; True if the file changed."""
    path = Path(path)
    old = path.read_text()
    new = replace_auto_blocks(old, render_blocks(root))
    if new == old:
        return False
    atomic_write(path, new.encode())
    return True
