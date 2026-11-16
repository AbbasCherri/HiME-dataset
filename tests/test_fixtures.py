import os

import numpy as np

from conftest import FailingFetcher, StubFetcher, synthetic_buildings


def test_stub_fetcher_matches_the_d1_formula():
    fetcher = StubFetcher()
    tile = fetcher.get("seamlessphoto", 18, 7, 11)
    assert tile.shape == (256, 256, 3)
    assert tile.dtype == np.uint8
    for u, v in [(0, 0), (5, 9), (255, 255)]:
        pixel = tile[v, u]
        expected = ((7 * 256 + u) % 251, (11 * 256 + v) % 241, 0)
        assert tuple(int(c) for c in pixel) == expected


def test_stub_fetcher_differs_by_tile():
    fetcher = StubFetcher()
    a = fetcher.get("seamlessphoto", 18, 1, 1)
    b = fetcher.get("seamlessphoto", 18, 2, 1)
    assert not np.array_equal(a, b)


def test_failing_fetcher_returns_none_for_chosen_tiles():
    fetcher = FailingFetcher({("seamlessphoto", 18, 3, 4)})
    assert fetcher.get("seamlessphoto", 18, 3, 4) is None


def test_failing_fetcher_delegates_other_tiles_to_the_stub():
    stub = StubFetcher()
    fetcher = FailingFetcher({("seamlessphoto", 18, 3, 4)})
    delegated = fetcher.get("seamlessphoto", 18, 9, 9)
    expected = stub.get("seamlessphoto", 18, 9, 9)
    assert np.array_equal(delegated, expected)


def test_synthetic_buildings_shape_and_columns():
    df = synthetic_buildings(n=500, seed=0)
    assert len(df) == 500
    required = {
        "building_id",
        "lat",
        "lon",
        "damage_val",
        "label",
        "MMI_original",
        "MMI_shape",
        "PGA",
        "PGV",
        "SA_0_3",
        "SA_1_0",
        "SA_3_0",
    }
    assert required.issubset(df.columns)


def test_synthetic_buildings_is_deterministic():
    a = synthetic_buildings(n=200, seed=0)
    b = synthetic_buildings(n=200, seed=0)
    assert a.equals(b)


def test_synthetic_buildings_has_all_four_raw_codes():
    df = synthetic_buildings(n=3000, seed=0)
    codes = set(df["label"].unique().tolist())
    assert {0, 1, 9, 99}.issubset(codes)


def test_synthetic_buildings_stays_in_the_bounding_box():
    df = synthetic_buildings(n=500, seed=0)
    assert df["lat"].between(37.005, 37.526).all()
    assert df["lon"].between(136.677, 137.356).all()


def test_synthetic_buildings_has_label_only_rows():
    df = synthetic_buildings(n=3000, seed=0)
    assert df["damage_val"].isna().any()


def test_synthetic_buildings_seismic_values_are_positive():
    df = synthetic_buildings(n=500, seed=0)
    seismic = ["MMI_original", "MMI_shape", "PGA", "PGV", "SA_0_3", "SA_1_0", "SA_3_0"]
    assert (df[seismic] > 0).all().all()


def test_tmp_home_fixture_sets_env_and_creates_a_directory(tmp_home):
    assert tmp_home.exists()
    assert os.environ["NOTODATA_HOME"] == str(tmp_home)
