import hashlib
import math

import numpy as np
import pytest
from PIL import Image

from conftest import FailingFetcher, StubFetcher
from notodata import chips, paths
from notodata.config import CONFIG, EARTH_RADIUS_M
from notodata.tiles import TileFetchError, TileShapeError, covering_tiles, global_pixel

LAYER = "stub"
Z = CONFIG.tile_chip.zoom
STUB = CONFIG.d1_test_stub
T = CONFIG.tile_chip.tile_size_px
CHIP = CONFIG.tile_chip.chip_size_px
HALF = CHIP // 2


def _points(n=STUB.n_points, seed=STUB.seed):
    rng = np.random.default_rng(seed)
    bb = CONFIG.bounding_box
    return rng.uniform(bb.lat_min, bb.lat_max, n), rng.uniform(bb.lon_min, bb.lon_max, n)


def _expected_centre(lat, lon):
    gx, gy = global_pixel(lat, lon, Z)
    x0, y0 = math.floor(gx - HALF), math.floor(gy - HALF)
    return ((x0 + HALF) % STUB.mod_u, (y0 + HALF) % STUB.mod_v, STUB.channel_b)


def test_chip_matches_stub_1000_points():
    lats, lons = _points()
    for lat, lon in zip(lats, lons, strict=True):
        res = chips.build_chip(StubFetcher(), LAYER, lat, lon)
        assert res.status == "ok"
        assert res.pixels.shape == (CHIP, CHIP, 3)
        assert res.pixels.dtype == np.uint8
        assert tuple(int(v) for v in res.pixels[HALF, HALF]) == _expected_centre(lat, lon)


def test_50m_east_west_differs():
    lat, lon = 37.3, 137.0
    dlon = 50.0 / (EARTH_RADIUS_M * math.cos(math.radians(lat)))
    a = chips.build_chip(StubFetcher(), LAYER, lat, lon)
    b = chips.build_chip(StubFetcher(), LAYER, lat, lon + math.degrees(dlon))
    assert not np.array_equal(a.pixels, b.pixels)


def test_tile_corner_point():
    # inverse projection of the exact corner of tile (230000, 103000) at z18
    tx, ty = 230_000, 103_000
    n = 2.0**Z
    lon = tx / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))
    res = chips.build_chip(StubFetcher(), LAYER, lat, lon)
    assert res.status == "ok"
    assert res.pixels.shape == (CHIP, CHIP, 3)
    assert len(res.tiles_used) == 4


def test_building_at_centre():
    lats, lons = _points()
    gx, gy = global_pixel(lats, lons, Z)
    x0 = np.floor(gx - HALF)
    y0 = np.floor(gy - HALF)
    assert ((gx - x0 >= HALF) & (gx - x0 < HALF + 1)).all()
    assert ((gy - y0 >= HALF) & (gy - y0 < HALF + 1)).all()
    # build_chip asserts the same internally; it must not raise for any point
    for lat, lon in zip(lats[:200], lons[:200], strict=True):
        chips.build_chip(StubFetcher(), LAYER, lat, lon)


def test_tiles_used_matches_covering_tiles():
    lat, lon = 37.3, 137.0
    gx, gy = global_pixel(lat, lon, Z)
    x0, y0 = math.floor(gx - HALF), math.floor(gy - HALF)
    res = chips.build_chip(StubFetcher(), LAYER, lat, lon)
    assert list(res.tiles_used) == covering_tiles(x0, y0)


class _BadShape:
    def get(self, layer_id, z, x, y):
        return np.zeros((T, T - 1, 3), dtype=np.uint8)


def test_non_256_tile_raises():
    with pytest.raises(TileShapeError):
        chips.build_chip(_BadShape(), LAYER, 37.3, 137.0)


class _Const:
    """Every tile is one value (RGB or RGBA)."""

    def __init__(self, value, alpha=None):
        ch = len(value) + (alpha is not None)
        self._tile = np.zeros((T, T, ch), dtype=np.uint8)
        self._tile[:, :, : len(value)] = value
        if alpha is not None:
            self._tile[:, :, 3] = alpha

    def get(self, layer_id, z, x, y):
        return self._tile.copy()


class _Raises:
    def get(self, layer_id, z, x, y):
        raise TileFetchError("boom")


def test_status_missing():
    lat, lon = 37.3, 137.0
    gx, gy = global_pixel(lat, lon, Z)
    first = covering_tiles(math.floor(gx - HALF), math.floor(gy - HALF))[0]
    f = FailingFetcher({(LAYER, Z, *first)})
    res = chips.build_chip(f, LAYER, lat, lon)
    assert res.status == "missing"
    assert res.pixels is None


def test_status_error_wins_and_has_no_pixels():
    res = chips.build_chip(_Raises(), LAYER, 37.3, 137.0)
    assert res.status == "error"
    assert res.pixels is None


def test_status_partial_alpha_zero_inside_window():
    lat, lon = 37.3, 137.0
    res = chips.build_chip(_Const((200, 100, 50), alpha=0), LAYER, lat, lon)
    assert res.status == "partial"


def test_alpha_zero_outside_window_is_not_partial():
    class OneCorner:
        def get(self, layer_id, z, x, y):
            t = np.zeros((T, T, 4), dtype=np.uint8)
            t[:, :, 0] = (np.arange(T)[None, :] * 7 + np.arange(T)[:, None] * 3) % 200 + 20
            t[:, :, 1] = 90
            t[:, :, 2] = 150
            t[:, :, 3] = 255
            # transparent only in the far corner of the tile, outside this window
            t[:2, :2, 3] = 0
            return t

    # tile-centre point: window is the tile centre, one tile
    n = 2.0**Z
    tx, ty = 230_000, 103_000
    lon = (tx + 0.5) / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 0.5) / n))))
    res = chips.build_chip(OneCorner(), LAYER, lat, lon)
    assert res.status == "ok"
    assert res.pixels.shape == (CHIP, CHIP, 3)


def test_status_blank_black_and_flat():
    assert chips.build_chip(_Const((0, 0, 0)), LAYER, 37.3, 137.0).status == "blank"
    assert chips.build_chip(_Const((120, 120, 120)), LAYER, 37.3, 137.0).status == "blank"


def _dark_columns_chip(k):
    """Noisy tiles whose first `k` window columns are near black (global columns)."""
    lat, lon = 37.3, 137.0
    gx, _ = global_pixel(lat, lon, Z)
    x0 = math.floor(gx - HALF)

    class Dark:
        def get(self, layer_id, z, x, y):
            rng = np.random.default_rng(1)
            t = rng.integers(40, 255, (T, T, 3), dtype=np.uint8)
            cols = x * T + np.arange(T)
            t[:, (cols >= x0) & (cols < x0 + k)] = 1
            return t

    return chips.build_chip(Dark(), LAYER, lat, lon)


def test_blank_half_near_black():
    # high std, so only the near-black share can decide
    assert _dark_columns_chip(CHIP // 2).status == "blank"
    assert _dark_columns_chip(CHIP // 2 - 1).status == "ok"


def _edge_point(horizontal):
    """A point on a tile edge so the window spans exactly two tiles."""
    n = 2.0**Z
    tx, ty = 230_000, 103_000
    if horizontal:
        lon = (tx + 1) / n * 360.0 - 180.0
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 0.5) / n))))
    else:
        lon = (tx + 0.5) / n * 360.0 - 180.0
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 1) / n))))
    return lat, lon


@pytest.mark.parametrize("horizontal", [True, False])
def test_two_tile_window(horizontal):
    lat, lon = _edge_point(horizontal)
    res = chips.build_chip(StubFetcher(), LAYER, lat, lon)
    assert res.status == "ok"
    assert len(res.tiles_used) == 2
    assert tuple(int(v) for v in res.pixels[HALF, HALF]) == _expected_centre(lat, lon)


def test_just_under_half_near_black_is_ok():
    class Stripe:
        def get(self, layer_id, z, x, y):
            rng = np.random.default_rng(1)
            t = rng.integers(40, 255, (T, T, 3), dtype=np.uint8)
            t[:, :, :] = np.where((np.arange(T) % 4 == 0)[None, :, None], 1, t)
            return t  # a quarter of the pixels near black

    assert chips.build_chip(Stripe(), LAYER, 37.3, 137.0).status == "ok"


def test_near_black_needs_every_channel():
    # bright red channel keeps pixels from counting as near black; std is high enough
    class RedOnly:
        def get(self, layer_id, z, x, y):
            t = np.zeros((T, T, 3), dtype=np.uint8)
            t[:, :, 0] = (np.arange(T)[None, :] * 5) % 256
            t[:, :, 0] += 30
            return t

    assert chips.build_chip(RedOnly(), LAYER, 37.3, 137.0).status == "ok"


def test_pixel_hash_is_sha256_of_c_order_bytes():
    res = chips.build_chip(StubFetcher(), LAYER, 37.3, 137.0)
    assert (
        chips.pixel_sha256(res.pixels)
        == hashlib.sha256(np.ascontiguousarray(res.pixels).tobytes()).hexdigest()
    )


def test_only_ok_chips_written_and_hash_stable(tmp_home):
    import pandas as pd

    lats, lons = _points(20)
    df = pd.DataFrame({"building_id": np.arange(5000, 5020), "lat": lats, "lon": lons})
    first = covering_tiles(*(math.floor(v - HALF) for v in global_pixel(lats[3], lons[3], Z)))[0]
    fetcher = FailingFetcher({(LAYER, Z, *first)})
    index = chips.build_chips(fetcher, LAYER, df)
    assert list(index.columns) == ["building_id", "status", "path", "sha256"]
    assert len(index) == 20
    assert index.loc[index.building_id == 5003, "status"].item() == "missing"
    ok = index[index.status == "ok"]
    bad = index[index.status != "ok"]
    assert len(bad) >= 1
    assert (bad.path == "").all()
    assert (bad.sha256 == "").all()
    for row in ok.itertuples():
        p = paths.chip_path(LAYER, row.building_id)
        assert p.exists()
        assert p.relative_to(paths.notodata_home()).as_posix() == row.path
        arr = np.asarray(Image.open(p))
        assert arr.shape == (CHIP, CHIP, 3)
        assert chips.pixel_sha256(arr) == row.sha256
    written = list(paths.chips_dir(LAYER).rglob("*.png"))
    assert len(written) == len(ok)
    # a second run is byte-identical
    again = chips.build_chips(fetcher, LAYER, df)
    assert again.equals(index)


# --- R3: evidence that the new code does not share the old bug -----------------


def _old_chip_centre_global_pixel(lat, lon):
    """Evidence only, not the old code: the brief's description of `_get_building_chip`.

    Integer tile of the point, a 512 x 512 mosaic whose top-left is that tile,
    a centre 128 x 128 crop. The crop centre is therefore a fixed offset from
    the tile corner, whatever the point's position inside the tile.
    """
    gx, gy = global_pixel(lat, lon, Z)
    tx, ty = int(gx // T), int(gy // T)
    return tx * T + T, ty * T + T, gx, gy


def test_old_bug_disagrees_with_new_code():
    lats, lons = _points()
    inside = 0
    disagree = 0
    for lat, lon in zip(lats, lons, strict=True):
        cx, cy, gx, gy = _old_chip_centre_global_pixel(lat, lon)
        if abs(cx - gx) < HALF and abs(cy - gy) < HALF:
            inside += 1
        res = chips.build_chip(StubFetcher(), LAYER, lat, lon)
        old_centre = ((cx) % STUB.mod_u, (cy) % STUB.mod_v, STUB.channel_b)
        if tuple(int(v) for v in res.pixels[HALF, HALF]) != old_centre:
            disagree += 1
    # old tile maths: a building lies in its chip for only a small share of points
    assert inside / len(lats) < 0.15
    assert disagree / len(lats) > 0.9
