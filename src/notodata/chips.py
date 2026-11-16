"""D1 chip window, mosaic, crop and status rules.

A chip is the 128 px window centred on a building, cut from the 1, 2 or 4
GSI tiles that cover it. Tiles are never resized, and a chip that cannot be
cut honestly gets a status other than `ok` and no pixels written anywhere.

Bump `PIPELINE_VERSION` in `config.py` whenever this module's maths changes.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from notodata import paths
from notodata.config import CONFIG
from notodata.tiles import (
    TileFetcher,
    TileFetchError,
    TileShapeError,
    covering_tiles,
    global_pixel,
    window_origin,
)

log = logging.getLogger(__name__)

STATUSES = ("ok", "missing", "partial", "blank", "error")

_INDEX_COLUMNS = ["building_id", "status", "path", "sha256"]


@dataclass(frozen=True)
class ChipResult:
    status: str
    pixels: np.ndarray | None
    """RGB uint8 (chip, chip, 3); kept for `ok`, `partial` and `blank`, else `None`."""
    tiles_used: tuple[tuple[int, int], ...]


def pixel_sha256(pixels: np.ndarray) -> str:
    """SHA-256 of the raw pixel bytes in C order, independent of the PNG encoder."""
    return hashlib.sha256(np.ascontiguousarray(pixels, dtype=np.uint8).tobytes()).hexdigest()


def _is_blank(rgb: np.ndarray) -> bool:
    cfg = CONFIG.blank_chip
    if float(rgb.std()) < cfg.std_below:
        return True
    near_black = (rgb <= cfg.black_level).all(axis=-1)
    return float(near_black.mean()) >= cfg.black_share


def _rgba(tile: np.ndarray) -> np.ndarray:
    """Tile as (h, w, 4); a tile without alpha is fully opaque."""
    if tile.ndim != 3 or tile.shape[2] not in (3, 4):
        raise TileShapeError(f"tile has shape {tile.shape}, expected (h, w, 3 or 4)")
    if tile.shape[2] == 4:
        return tile
    alpha = np.full((*tile.shape[:2], 1), 255, dtype=tile.dtype)
    return np.concatenate([tile, alpha], axis=2)


def _window_origin(lat: float, lon: float) -> tuple[int, int]:
    """Top-left global pixel of the window; checks the building sits at chip centre."""
    cfg = CONFIG.tile_chip
    gx, gy = global_pixel(lat, lon, cfg.zoom)
    x0a, y0a = window_origin(gx, gy)
    x0, y0 = int(x0a), int(y0a)
    half = cfg.chip_size_px // 2
    # the building lies at chip pixel (64, 64) to within one pixel
    if not (half <= gx - x0 < half + 1 and half <= gy - y0 < half + 1):
        raise RuntimeError(f"building not at chip centre: ({gx - x0}, {gy - y0})")
    return x0, y0


def _fetch_tiles(fetcher, layer_id, tiles):
    """Tile pixels by `(tx, ty)`, or `None` when any fetch failed."""
    got: dict[tuple[int, int], np.ndarray | None] = {}
    failed = False
    for tx, ty in tiles:
        try:
            got[tx, ty] = fetcher.get(layer_id, CONFIG.tile_chip.zoom, tx, ty)
        except TileFetchError:
            failed = True
    return None if failed else got


def _crop_window(got, tiles, x0: int, y0: int) -> np.ndarray:
    """Paste the tiles into a mosaic and crop the chip window as (h, w, 4)."""
    size, chip = CONFIG.tile_chip.tile_size_px, CONFIG.tile_chip.chip_size_px
    tx_min, ty_min = min(t[0] for t in tiles), min(t[1] for t in tiles)
    nx = max(t[0] for t in tiles) - tx_min + 1
    ny = max(t[1] for t in tiles) - ty_min + 1
    mosaic = np.zeros((ny * size, nx * size, 4), dtype=np.uint8)
    for (tx, ty), tile in got.items():
        if tile.shape[:2] != (size, size):
            raise TileShapeError(f"tile is {tile.shape[1]}x{tile.shape[0]}, expected {size}x{size}")
        r, c = (ty - ty_min) * size, (tx - tx_min) * size
        mosaic[r : r + size, c : c + size] = _rgba(tile)
    cy, cx = y0 - ty_min * size, x0 - tx_min * size
    return mosaic[cy : cy + chip, cx : cx + chip]


def _classify(window: np.ndarray) -> tuple[str, np.ndarray]:
    rgb = np.ascontiguousarray(window[:, :, :3])
    if (window[:, :, 3] == 0).any():
        return "partial", rgb
    return ("blank" if _is_blank(rgb) else "ok"), rgb


def build_chip(fetcher: TileFetcher, layer_id: str, lat: float, lon: float) -> ChipResult:
    """Cut the chip centred on (lat, lon); never resizes, never invents pixels."""
    x0, y0 = _window_origin(lat, lon)
    tiles = tuple(covering_tiles(x0, y0))
    got = _fetch_tiles(fetcher, layer_id, tiles)
    if got is None:
        return ChipResult("error", None, tiles)
    if any(t is None for t in got.values()):
        return ChipResult("missing", None, tiles)
    status, rgb = _classify(_crop_window(got, tiles, x0, y0))
    return ChipResult(status, rgb, tiles)


def write_chip(layer_id: str, building_id: int, result: ChipResult) -> Path:
    """Write an `ok` chip as PNG; anything else is refused, so no placeholder can exist."""
    if result.status != "ok" or result.pixels is None:
        raise ValueError(f"only ok chips are written, not {result.status!r}")
    path = paths.chip_path(layer_id, int(building_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    Image.fromarray(result.pixels, "RGB").save(tmp, format="PNG")
    tmp.replace(path)
    return path


def build_chips(fetcher: TileFetcher, layer_id: str, buildings: pd.DataFrame) -> pd.DataFrame:
    """Chip every building; write the `ok` ones; return the per-layer chip index.

    Columns: building_id, status, path (relative to `NOTODATA_HOME`, empty unless
    `ok`), sha256 (pixel-byte hash, empty unless `ok`).
    """
    home = paths.notodata_home()
    rows = []
    for bid, lat, lon in zip(
        buildings["building_id"], buildings["lat"], buildings["lon"], strict=True
    ):
        res = build_chip(fetcher, layer_id, lat, lon)
        if res.status == "ok":
            rel = write_chip(layer_id, bid, res).relative_to(home).as_posix()
            rows.append((int(bid), "ok", rel, pixel_sha256(res.pixels)))
        else:
            rows.append((int(bid), res.status, "", ""))
    counts = pd.Series([r[1] for r in rows]).value_counts().to_dict()
    log.info(
        "chips %s: %s (blank if std < %s or >= %s near black within %s)",
        layer_id,
        counts,
        CONFIG.blank_chip.std_below,
        CONFIG.blank_chip.black_share,
        CONFIG.blank_chip.black_level,
    )
    return pd.DataFrame(rows, columns=_INDEX_COLUMNS)
