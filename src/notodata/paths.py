"""Every filesystem path derived from `NOTODATA_HOME`.

No other module builds a `data/` path by hand; a function here does it, so
the layout only has to change in one place.
"""

from __future__ import annotations

import os
from pathlib import Path


def notodata_home() -> Path:
    """The root of every cache, from `NOTODATA_HOME` (default `./data`)."""
    return Path(os.environ.get("NOTODATA_HOME", "./data"))


def raw_dir() -> Path:
    return notodata_home() / "raw"


def tiles_dir(layer_id: str) -> Path:
    return notodata_home() / "tiles" / layer_id


def tile_path(layer_id: str, z: int, x: int, y: int, ext: str) -> Path:
    return tiles_dir(layer_id) / str(z) / str(x) / f"{y}.{ext}"


def chips_dir(layer_id: str) -> Path:
    return notodata_home() / "chips" / layer_id


def chip_path(layer_id: str, building_id: int) -> Path:
    bucket = building_id // 1000
    return chips_dir(layer_id) / str(bucket) / f"{building_id}.png"


def embeddings_dir(encoder_id: str) -> Path:
    return notodata_home() / "embeddings" / encoder_id


def audit_dir() -> Path:
    return notodata_home() / "audit"


def logs_dir() -> Path:
    return notodata_home() / "logs"


def derived_dir() -> Path:
    return notodata_home() / "derived"


def buildings_table_path() -> Path:
    return derived_dir() / "buildings.parquet"


def splits_dir() -> Path:
    return derived_dir() / "splits"


def split_path(seed: int) -> Path:
    return splits_dir() / f"seed_{seed}.parquet"


def clients_dir() -> Path:
    return derived_dir() / "clients"


def client_path(n: int, seed: int) -> Path:
    return clients_dir() / f"N{n}_seed_{seed}.parquet"


def client_meta_path(n: int, seed: int) -> Path:
    return clients_dir() / f"N{n}_seed_{seed}_meta.parquet"


def reports_dir(root: Path | None = None) -> Path:
    """Reports are committed, so they live at the repo root, not under NOTODATA_HOME."""
    return (root or Path()) / "reports"


def chip_index_path(layer_id: str) -> Path:
    return chips_dir(layer_id) / "index.parquet"
