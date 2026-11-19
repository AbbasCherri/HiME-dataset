"""SHA-256 manifest of every derived file (D8, Phase 12).

`MANIFEST.sha256` is in `sha256sum` format (`<hash>  <name>`), sorted by name.
Chip PNGs are covered by one tree hash per layer; raw tiles are never listed,
they get a fingerprint instead so a mismatch can be blamed on GSI or on code.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from notodata import chips, paths
from notodata.io import atomic_write, sha256_file

TREE_SUFFIX = "@tree"
"""Manifest name of a layer's chip tree hash: `chips/<layer>/@tree`."""

log = logging.getLogger(__name__)

_SEP = "  "
_TILE_SKIP_SUFFIXES = {".missing", ".tmp"}


def _digest(lines: Iterable[str]) -> str:
    return hashlib.sha256("".join(f"{ln}\n" for ln in lines).encode()).hexdigest()


def chip_tree_hash(layer_id: str, from_pixels: bool = False) -> str:
    """SHA-256 over the sorted `<relative path> <pixel sha256>` lines of the `ok` chips.

    By default the pixel hashes come from the chip index; `from_pixels=True`
    decodes every PNG instead, which catches a repainted file.
    """
    index = pd.read_parquet(paths.chip_index_path(layer_id))
    ok = index[index["status"] == "ok"].sort_values("path", kind="mergesort")
    if from_pixels:
        home = paths.notodata_home()
        hashes = []
        for rel in ok["path"]:
            with Image.open(home / rel) as img:
                hashes.append(chips.pixel_sha256(np.asarray(img.convert("RGB"))))
    else:
        hashes = ok["sha256"].tolist()
    return _digest(f"{p} {h}" for p, h in zip(ok["path"], hashes, strict=True))


def tile_fingerprint(layer_id: str) -> tuple[str, int]:
    """SHA-256 over sorted `<z>/<x>/<y> <file sha256>` lines, and the tile count."""
    root = paths.tiles_dir(layer_id)
    lines = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if not p.is_file() or len(rel.parts) != 3 or p.suffix in _TILE_SKIP_SUFFIXES:
            continue
        z, x = rel.parts[0], rel.parts[1]
        lines.append(f"{z}/{x}/{p.stem} {sha256_file(p)}")
    lines.sort()
    return _digest(lines), len(lines)


def _files(directory: Path, pattern: str) -> list[Path]:
    return sorted(directory.glob(pattern)) if directory.is_dir() else []


def _embedding_files() -> list[Path]:
    """CPU-built embedding files; GPU builds (`deterministic: false`) are verified by cosine."""
    out = []
    for p in _files(paths.notodata_home() / "embeddings", "*/*.npy"):
        vectors = p.with_name(p.name.replace(".ids.npy", ".npy"))  # ids go with their vectors
        side = vectors.with_name(f"{vectors.name}.meta.json")
        if side.exists() and json.loads(side.read_text()).get("deterministic") is False:
            continue
        out.append(p)
    return out


def collect_entries(layer_ids: Iterable[str], root: Path | None = None) -> dict[str, str]:
    """Name to hash for every derived file; names are relative so the manifest is portable.

    `root` is the repo root holding `reports/` (default: the current directory).
    """
    home = paths.notodata_home()
    entries: dict[str, str] = {}

    def add(p: Path, base: Path, prefix: str = "") -> None:
        entries[prefix + p.relative_to(base).as_posix()] = sha256_file(p)

    table = paths.buildings_table_path()
    if table.exists():
        add(table, home)
    for pattern in ("splits/*.parquet", "clients/*.parquet"):
        for p in _files(paths.derived_dir(), pattern):
            add(p, home)
    for layer in layer_ids:
        if not paths.chip_index_path(layer).exists():
            log.warning("layer %s has no chip index; left out of the manifest", layer)
            continue  # reported as missing by `verify_manifest`
        add(paths.chip_index_path(layer), home)
        entries[f"chips/{layer}/{TREE_SUFFIX}"] = chip_tree_hash(layer)
    for p in _embedding_files():
        add(p, home)
    reports = paths.reports_dir(root)
    for p in _files(reports, "*.csv") + _files(reports, "imagery_decision.json"):
        add(p, reports.parent, "")
    return dict(sorted(entries.items()))


def write_manifest(path: Path | str, entries: dict[str, str]) -> None:
    body = "".join(f"{h}{_SEP}{name}\n" for name, h in sorted(entries.items()))
    atomic_write(path, body.encode())


def read_manifest(path: Path | str) -> dict[str, str]:
    entries = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            digest, name = line.split(_SEP, 1)
            entries[name] = digest
    return entries


def verify_manifest(
    path: Path | str, layer_ids: Iterable[str], root: Path | None = None
) -> list[str]:
    """Problems found against the manifest (empty when every line matches).

    Each message names the file: a hash mismatch, a file that is missing, or a
    derived file present but not listed.
    """
    expected = read_manifest(path)
    actual = collect_entries(layer_ids, root=root)
    problems = []
    for name, digest in expected.items():
        if name not in actual:
            problems.append(f"{name}: missing")
        elif actual[name] != digest:
            problems.append(f"{name}: hash mismatch")
    problems += [f"{name}: not in manifest" for name in actual if name not in expected]
    return problems
