"""Shared test fixtures: deterministic tile stubs and a synthetic dataset.

`StubFetcher`, `FailingFetcher` and `synthetic_buildings` are plain
classes/functions (not pytest fixtures) so a test can parametrise them
directly, exactly as the build plan's fixture table describes. `tmp_home`
is a pytest fixture: a fresh, per-test `NOTODATA_HOME`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class StubFetcher:
    """Deterministic tile pixels, exactly as in the D1 test stub.

    Pixel (u, v) of tile (tx, ty) is
    `((tx*256 + u) % 251, (ty*256 + v) % 241, 0)`.
    """

    def get(self, layer_id: str, z: int, x: int, y: int) -> np.ndarray | None:
        del layer_id, z
        tx, ty = x, y
        u = np.arange(256, dtype=np.int64)
        v = np.arange(256, dtype=np.int64)
        red = (tx * 256 + u) % 251
        green = (ty * 256 + v) % 241
        tile = np.zeros((256, 256, 3), dtype=np.uint8)
        tile[:, :, 0] = red[np.newaxis, :]
        tile[:, :, 1] = green[:, np.newaxis]
        tile[:, :, 2] = 0
        return tile


class FailingFetcher:
    """Returns `None` for the given `(layer_id, z, x, y)` tiles, else delegates to `StubFetcher`."""

    def __init__(self, fail_tiles: set[tuple[str, int, int, int]]):
        self._fail_tiles = set(fail_tiles)
        self._delegate = StubFetcher()

    def get(self, layer_id: str, z: int, x: int, y: int) -> np.ndarray | None:
        if (layer_id, z, x, y) in self._fail_tiles:
            return None
        return self._delegate.get(layer_id, z, x, y)


_LAT_MIN, _LAT_MAX = 37.005, 37.526
_LON_MIN, _LON_MAX = 136.677, 137.356


def synthetic_buildings(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Synthetic buildings for offline tests.

    Points sit in spatial clusters inside the real bounding box; labels
    correlate with cluster, so spatial statistics have something to find.
    All four raw codes (0, 1, 9, 99) appear. A minority of rows carry only
    `label` (not `damage_val`), to exercise the D4 fallback. Seismic columns
    are strictly positive floats.
    """
    rng = np.random.default_rng(seed)

    n_clusters = max(1, n // 200)
    cluster_lat = rng.uniform(_LAT_MIN, _LAT_MAX, n_clusters)
    cluster_lon = rng.uniform(_LON_MIN, _LON_MAX, n_clusters)
    cluster_damage_rate = rng.uniform(0.0, 0.6, n_clusters)

    cluster_id = rng.integers(0, n_clusters, n)
    jitter_deg = 0.01
    lat = np.clip(cluster_lat[cluster_id] + rng.normal(0, jitter_deg, n), _LAT_MIN, _LAT_MAX)
    lon = np.clip(cluster_lon[cluster_id] + rng.normal(0, jitter_deg, n), _LON_MIN, _LON_MAX)

    draw = rng.random(n)
    damage_rate = cluster_damage_rate[cluster_id]
    raw_code = np.where(
        draw < damage_rate * 0.8,
        1,
        np.where(
            draw < damage_rate * 0.8 + 0.05,
            99,
            np.where(draw < damage_rate * 0.8 + 0.1, 9, 0),
        ),
    )

    df = pd.DataFrame(
        {
            "building_id": np.arange(n, dtype=np.int64),
            "lat": lat,
            "lon": lon,
            "damage_val": raw_code.astype(np.float64),
            "label": raw_code,
            "MMI_original": rng.uniform(1.0, 8.0, n),
            "MMI_shape": rng.uniform(0.01, 1.0, n),
            "PGA": rng.uniform(0.01, 2.0, n),
            "PGV": rng.uniform(0.01, 1.5, n),
            "SA_0_3": rng.uniform(0.01, 3.0, n),
            "SA_1_0": rng.uniform(0.01, 2.0, n),
            "SA_3_0": rng.uniform(0.01, 1.0, n),
        }
    )

    # A minority of rows carry only `label`, exercising the D4 fallback.
    label_only = rng.random(n) < 0.05
    df.loc[label_only, "damage_val"] = np.nan

    return df


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    """A temporary `NOTODATA_HOME`, fresh for each test."""
    home = tmp_path / "data"
    home.mkdir()
    monkeypatch.setenv("NOTODATA_HOME", str(home))
    return home
