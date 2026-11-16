import numpy as np
import pytest

from notodata import geometry
from notodata.config import EARTH_RADIUS_M


def test_projection_matches_stated_formulas():
    lat = np.array([37.1, 37.4, 37.25])
    lon = np.array([136.8, 137.2, 137.0])
    lat0, lon0 = 37.2, 137.0
    x, y = geometry.project(lat, lon, lat0, lon0)
    ex = EARTH_RADIUS_M * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    ey = EARTH_RADIUS_M * np.radians(lat - lat0)
    assert x.dtype == np.float64
    assert y.dtype == np.float64
    np.testing.assert_allclose(x, ex, rtol=0, atol=1e-9)
    np.testing.assert_allclose(y, ey, rtol=0, atol=1e-9)


def test_projection_round_trip():
    rng = np.random.default_rng(0)
    lat = rng.uniform(37.005, 37.526, 500)
    lon = rng.uniform(136.677, 137.356, 500)
    lat0, lon0 = geometry.reference_point(lat, lon)
    x, y = geometry.project(lat, lon, lat0, lon0)
    lat2, lon2 = geometry.unproject(x, y, lat0, lon0)
    np.testing.assert_allclose(lat2, lat, rtol=0, atol=1e-10)
    np.testing.assert_allclose(lon2, lon, rtol=0, atol=1e-10)


def test_reference_point_is_mean_over_all_rows_including_nan_label_rows():
    lat = np.array([37.0, 37.2, 37.4, 37.6])
    lon = np.array([136.8, 137.0, 137.2, 137.4])
    lat0, lon0 = geometry.reference_point(lat, lon)
    assert lat0 == pytest.approx(37.3)
    assert lon0 == pytest.approx(137.1)


def test_reference_point_rejects_empty_and_non_finite():
    with pytest.raises(ValueError, match="non-empty"):
        geometry.reference_point(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="finite"):
        geometry.reference_point(np.array([37.0, np.nan]), np.array([137.0, 137.0]))


def test_min_distance_and_within_against_brute_force():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 5000, (300, 2))
    b = rng.uniform(0, 5000, (200, 2))
    brute = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(axis=1)
    np.testing.assert_allclose(geometry.min_distance(a, b), brute, atol=1e-9)
    for r in (0.0, 50.0, 200.0, 500.0, 2000.0):
        np.testing.assert_array_equal(geometry.within(a, b, r), brute <= r)


def test_within_is_inclusive_at_exact_radius():
    a = np.array([[0.0, 0.0]])
    b = np.array([[3.0, 4.0]])
    assert geometry.within(a, b, 5.0).tolist() == [True]
    assert geometry.within(a, b, 4.999).tolist() == [False]


def test_build_kdtree_accepts_prebuilt_tree():
    b = np.array([[0.0, 0.0], [10.0, 0.0]])
    tree = geometry.build_kdtree(b)
    assert geometry.min_distance(np.array([[9.0, 0.0]]), tree).tolist() == [1.0]


def test_empty_target_raises():
    with pytest.raises(ValueError, match="non-empty"):
        geometry.min_distance(np.zeros((1, 2)), np.zeros((0, 2)))
