"""Web-Mercator tile maths, the `TileFetcher` protocol and the GSI HTTP fetcher.

This is the only module that touches the network for imagery. Tiles are cached
as the server's raw bytes under `tiles/<layer_id>/<z>/<x>/<y>.<ext>`; a 404 is
cached as `<y>.missing` so it is never requested twice.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Protocol

import numpy as np
import requests
from PIL import Image

from notodata import paths
from notodata.config import CONFIG, FetchConfig

log = logging.getLogger(__name__)

_THROTTLE_STATUSES = frozenset({429, 503})


class TileShapeError(ValueError):
    """A decoded tile is not exactly `tile_size_px` square; tiles are never resized."""


class TemplateMismatchError(RuntimeError):
    """A layer id already has cached tiles from a different URL template."""


class TileFetchError(RuntimeError):
    """A tile could not be fetched after every retry."""


class TileFetcher(Protocol):
    def get(self, layer_id: str, z: int, x: int, y: int) -> np.ndarray | None:
        """Decoded tile pixels, or `None` when the layer has no such tile."""
        ...


# ---- tile maths ------------------------------------------------------------


def global_pixel(lat, lon, z: int, tile_size: int | None = None):
    """Global pixel coordinates (float64) of lat/lon at zoom `z`, as in D1."""
    size = CONFIG.tile_chip.tile_size_px if tile_size is None else tile_size
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    phi = np.radians(lat)
    scale = 2.0**z * size
    gx = (lon + 180.0) / 360.0 * scale
    gy = (1.0 - np.log(np.tan(phi) + 1.0 / np.cos(phi)) / np.pi) / 2.0 * scale
    return gx, gy


def window_origin(gx, gy, chip_size: int | None = None):
    """Top-left global pixel of the chip window centred on (gx, gy)."""
    half = (CONFIG.tile_chip.chip_size_px if chip_size is None else chip_size) // 2
    return np.floor(gx - half).astype(np.int64), np.floor(gy - half).astype(np.int64)


def covering_tiles(
    x0: int, y0: int, chip_size: int | None = None, tile_size: int | None = None
) -> list[tuple[int, int]]:
    """Tiles `(tx, ty)` overlapping the window, row-major; 1, 2 or 4 of them."""
    chip = CONFIG.tile_chip.chip_size_px if chip_size is None else chip_size
    size = CONFIG.tile_chip.tile_size_px if tile_size is None else tile_size
    x0, y0 = int(x0), int(y0)
    xs = range(x0 // size, (x0 + chip - 1) // size + 1)
    ys = range(y0 // size, (y0 + chip - 1) // size + 1)
    return [(tx, ty) for ty in ys for tx in xs]


def parent_tile(z: int, x: int, y: int, parent_z: int) -> tuple[int, int]:
    shift = z - parent_z
    return x >> shift, y >> shift


# ---- rate limiting ---------------------------------------------------------


class TokenBucket:
    """Evenly spaces `acquire()` calls at most `rate` per second, across threads."""

    def __init__(
        self,
        rate: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._interval = 1.0 / rate
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            slot = max(now, self._next)
            self._next = slot + self._interval
        if slot > now:
            self._sleep(slot - now)


# ---- HTTP fetcher ----------------------------------------------------------


def _decode(data: bytes, tile_size: int) -> np.ndarray:
    with Image.open(BytesIO(data)) as img:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert(
                "RGBA" if "A" in img.getbands() or "transparency" in img.info else "RGB"
            )
        arr = np.asarray(img)
    if arr.shape[:2] != (tile_size, tile_size):
        raise TileShapeError(
            f"tile is {arr.shape[1]}x{arr.shape[0]}, expected {tile_size}x{tile_size}"
        )
    return arr


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class HttpTileFetcher:
    """Fetches one layer's tiles with a disk cache, retries and polite limits."""

    def __init__(
        self,
        layer_id: str,
        url_template: str,
        ext: str,
        *,
        settings: FetchConfig = CONFIG.fetch,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.layer_id = layer_id
        self.url_template = url_template
        self.ext = ext
        self._s = settings
        self._tile_size = CONFIG.tile_chip.tile_size_px
        self._sleep = sleep
        self._session = session or requests.Session()
        self._session.headers["User-Agent"] = settings.user_agent
        self._slots = threading.BoundedSemaphore(settings.max_concurrent)
        self._bucket = TokenBucket(settings.max_requests_per_s)
        self._err_lock = threading.Lock()
        self._check_template()

    def _check_template(self) -> None:
        meta = paths.tiles_dir(self.layer_id) / "layer.meta.json"
        if meta.exists():
            recorded = json.loads(meta.read_text()).get("url_template")
            if recorded != self.url_template:
                raise TemplateMismatchError(
                    f"layer {self.layer_id!r} was cached from {recorded!r}, "
                    f"now {self.url_template!r}"
                )
        else:
            _write_atomic(meta, json.dumps({"url_template": self.url_template}).encode())

    def get(self, layer_id: str, z: int, x: int, y: int) -> np.ndarray | None:
        if layer_id != self.layer_id:
            raise ValueError(f"fetcher is for layer {self.layer_id!r}, not {layer_id!r}")
        pz = self._s.precheck_zoom
        if pz > 0 and z > pz:
            px, py = parent_tile(z, x, y, pz)
            if self._fetch_raw(pz, px, py) is None:
                return None
        data = self._fetch_raw(z, x, y)
        return None if data is None else _decode(data, self._tile_size)

    def fetch_many(
        self, tile_ids: Iterable[tuple[str, int, int, int]], workers: int | None = None
    ) -> dict[tuple[str, int, int, int], np.ndarray | None]:
        """Fetch many tiles in parallel; a tile that fails after retries raises."""
        tile_ids = list(tile_ids)
        with ThreadPoolExecutor(workers or self._s.max_concurrent) as pool:
            return dict(zip(tile_ids, pool.map(lambda t: self.get(*t), tile_ids), strict=True))

    def _fetch_raw(self, z: int, x: int, y: int) -> bytes | None:
        """Raw tile bytes from cache or server; `None` for a (cached) 404."""
        path = paths.tile_path(self.layer_id, z, x, y, self.ext)
        missing = path.with_suffix(".missing")
        if path.exists():
            return path.read_bytes()
        if missing.exists():
            return None
        last, throttled = "no attempt", False
        for attempt in range(self._s.max_retries + 1):
            if attempt:
                base = self._s.throttle_backoff_base_s if throttled else self._s.backoff_base_s
                self._sleep(min(base * 2 ** (attempt - 1), self._s.backoff_max_s))
            outcome, body, last = self._request(z, x, y)
            if outcome == "ok":
                _write_atomic(path, body)
                return body
            if outcome == "missing":
                _write_atomic(missing, b"")
                return None
            if outcome == "fatal":
                break  # other 4xx will not improve with retries
            throttled = outcome == "throttle"
        self._record_failure(z, x, y, last)
        raise TileFetchError(f"{self.layer_id}/{z}/{x}/{y}: {last}")

    def _request(self, z: int, x: int, y: int) -> tuple[str, bytes, str]:
        """One rate-limited GET, classified as ok, missing, throttle, retry or fatal."""
        self._bucket.acquire()
        try:
            with self._slots:
                resp = self._session.get(
                    self.url_template.format(z=z, x=x, y=y), timeout=self._s.timeout_s
                )
        except requests.RequestException as exc:
            return "retry", b"", type(exc).__name__
        status = resp.status_code
        if status == 200:
            return "ok", resp.content, "HTTP 200"
        if status == 404:
            return "missing", b"", "HTTP 404"
        if status in _THROTTLE_STATUSES:
            return "throttle", b"", f"HTTP {status}"
        return ("retry" if status >= 500 else "fatal"), b"", f"HTTP {status}"

    def _record_failure(self, z: int, x: int, y: int, error: str) -> None:
        log.error("tile fetch failed: %s/%d/%d/%d: %s", self.layer_id, z, x, y, error)
        out = paths.logs_dir() / "fetch_errors.csv"
        with self._err_lock:
            out.parent.mkdir(parents=True, exist_ok=True)
            new = not out.exists()
            with out.open("a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["layer_id", "z", "x", "y", "error"])
                w.writerow([self.layer_id, z, x, y, error])
