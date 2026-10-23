import numpy as np
import pandas as pd
import pytest

from conftest import FailingFetcher, StubFetcher
from notodata import imagery, paths
from notodata.config import CONFIG
from notodata.tiles import covering_tiles, global_pixel, window_origin

POST, PRE = "post_stub", "pre_stub"
Z = CONFIG.tile_chip.zoom


def _buildings(n=6):
    rng = np.random.default_rng(3)
    bb = CONFIG.bounding_box
    return pd.DataFrame(
        {
            "building_id": np.arange(n, dtype=np.int64) + 2500,
            "lat": rng.uniform(bb.lat_min, bb.lat_max, n),
            "lon": rng.uniform(bb.lon_min, bb.lon_max, n),
        }
    )


def _tiles_of(row):
    gx, gy = global_pixel(row.lat, row.lon, Z)
    x0, y0 = window_origin(gx, gy)
    return set(covering_tiles(int(x0), int(y0)))


def _index(ids, ok):
    rows = [
        (
            int(b),
            "ok" if b in ok else "missing",
            f"chips/x/{b}.png" if b in ok else "",
            "h" if b in ok else "",
        )
        for b in ids
    ]
    return pd.DataFrame(rows, columns=["building_id", "status", "path", "sha256"])


@pytest.mark.parametrize(
    ("post", "pre", "g"), [(1, 1, "IP"), (1, 0, "I"), (0, 1, "none"), (0, 0, "none")]
)
def test_pattern_truth_table(post, pre, g):
    assert imagery.image_pattern(post, pre) == g


def test_pre_without_post_is_none():
    ids = [1, 2, 3]
    flags = imagery.flags_from_indexes(ids, _index(ids, {2}), _index(ids, {1, 2, 3}), True)
    by = flags.set_index("building_id")
    assert by.loc[1, "has_pre"] == 1
    assert by.loc[1, "has_post"] == 0
    assert by.loc[1, "G"] == "none"
    assert by.loc[2, "G"] == "IP"
    assert by.loc[3, "G"] == "none"


def test_pre_unavailable():
    ids = [1, 2]
    flags = imagery.flags_from_indexes(ids, _index(ids, {1, 2}), _index(ids, {1, 2}), False)
    assert (flags["has_pre"] == 0).all()
    assert (flags["pre_chip"] == "").all()
    assert (flags["pre_chip_sha256"] == "").all()
    assert "IP" not in set(flags["G"])
    flags = imagery.flags_from_indexes(ids, _index(ids, {1, 2}), None, True)
    assert (flags["has_pre"] == 0).all()


def test_failing_fetcher_flags(tmp_home):
    b = _buildings()
    victim = b.iloc[2]
    tile = sorted(_tiles_of(victim))[0]
    fetcher = FailingFetcher({(POST, Z, *tile)})
    expect_bad = {int(r.building_id) for r in b.itertuples() if tile in _tiles_of(r)}
    assert int(victim.building_id) in expect_bad
    flags = imagery.build_image_flags(fetcher, b, POST, PRE, True)
    bad = set(flags.loc[flags["has_post"] == 0, "building_id"])
    assert bad == expect_bad
    for r in flags[flags["has_post"] == 0].itertuples():
        assert r.post_chip == ""
        assert r.post_chip_sha256 == ""
        assert not paths.chip_path(POST, r.building_id).exists()
    good = flags[flags["has_post"] == 1]
    assert (good["post_chip"] != "").all()
    assert (good["post_chip_sha256"] != "").all()


def test_no_blank_chip_written(tmp_home):
    class Black:
        def get(self, layer_id, z, x, y):
            return np.zeros((256, 256, 3), dtype=np.uint8)

    b = _buildings(4)
    flags = imagery.build_image_flags(Black(), b, POST, PRE, True)
    assert (flags["has_post"] == 0).all()
    assert (flags["has_pre"] == 0).all()
    assert (flags["G"] == "none").all()
    assert not list(paths.chips_dir(POST).rglob("*.png"))
    assert not list(paths.chips_dir(PRE).rglob("*.png"))


def test_no_pre_fetch_when_unavailable(tmp_home):
    seen = set()

    class Spy(StubFetcher):
        def get(self, layer_id, z, x, y):
            seen.add(layer_id)
            return super().get(layer_id, z, x, y)

    flags = imagery.build_image_flags(Spy(), _buildings(3), POST, PRE, False)
    assert seen == {POST}
    assert (flags["G"] == "I").all()
    assert not paths.chips_dir(PRE).exists()


def test_flags_consistent_with_paths(tmp_home):
    flags = imagery.build_image_flags(StubFetcher(), _buildings(), POST, PRE, True)
    assert (flags["G"] == "IP").all()
    for col in ("post_chip", "pre_chip"):
        for rel in flags[col]:
            assert (paths.notodata_home() / rel).is_file()


def test_coverage_report(tmp_path):
    flags = pd.DataFrame(
        {
            "building_id": [1, 2, 3, 4],
            "G": ["IP", "I", "none", "I"],
            "label_raw": [1, 1, 0, 0],
            "y_collapse": [1.0, 1.0, 0.0, np.nan],
        }
    )
    status = {"L": pd.Series({"ok": 3, "missing": 1})}
    out = imagery.coverage_by_pattern(flags, status)
    overall = out[(out["group"] == "overall")].set_index("pattern")
    assert overall.loc["I", "count"] == 2
    assert overall.loc["I", "share"] == 0.5
    raw1 = out[(out["group"] == "label_raw") & (out["value"] == "1")].set_index("pattern")
    assert raw1.loc["IP", "count"] == 1
    assert raw1.loc["none", "count"] == 0
    y1 = out[(out["group"] == "y_collapse") & (out["value"] == "1")]
    assert y1["count"].sum() == 2
    path = imagery.write_coverage_report(flags, status, tmp_path)
    assert path == tmp_path / "reports" / "coverage_by_pattern.csv"
    text = path.read_text()
    assert "chip_status" in text
    assert "missing" in text
