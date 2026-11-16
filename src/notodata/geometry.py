"""Equirectangular projection and KD-tree distance helpers.

Every distance in the buffer, fold and Moran stages is in the metres produced
here. The projection is the brief's: `x = R (lon - lon0) cos(lat0)`,
`y = R (lat - lat0)`, angles in radians, float64.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from notodata.config import EARTH_RADIUS_M


def reference_point(lat, lon) -> tuple[float, float]:
    """Mean latitude and longitude over ALL rows (labelled or not)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if lat.size == 0 or lat.shape != lon.shape:
        raise ValueError("lat and lon must be non-empty and the same shape")
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError("lat and lon must be finite")
    return float(lat.mean()), float(lon.mean())


def project(lat, lon, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    x = EARTH_RADIUS_M * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = EARTH_RADIUS_M * np.radians(lat - lat0)
    return x, y


def unproject(x, y, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lat = lat0 + np.degrees(y / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(x / (EARTH_RADIUS_M * np.cos(np.radians(lat0))))
    return lat, lon


def build_kdtree(xy) -> cKDTree:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] == 0:
        raise ValueError("xy must be a non-empty (n, 2) array")
    return cKDTree(xy)


def _as_tree(b_xy) -> cKDTree:
    return b_xy if isinstance(b_xy, cKDTree) else build_kdtree(b_xy)


def min_distance(a_xy, b_xy) -> np.ndarray:
    """Distance in metres from each point of `a_xy` to its nearest in `b_xy`."""
    a_xy = np.asarray(a_xy, dtype=np.float64).reshape(-1, 2)
    dist, _ = _as_tree(b_xy).query(a_xy, k=1)
    return dist


def within(a_xy, b_xy, r: float) -> np.ndarray:
    """True where a point of `a_xy` has some point of `b_xy` at distance <= r."""
    return min_distance(a_xy, b_xy) <= r
