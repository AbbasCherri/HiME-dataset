"""Moran's I by distance bin and the block size L (brief D5.4).

Bins are (lo, hi] in metres with binary weights and i != j. Cumulative
weighted sums from `cKDTree.count_neighbors` give each bin by differencing,
so no pair list is ever built. At r = 0 the tree counts every point with
itself; those self-pairs are removed from the first bin only, which keeps
coincident but distinct buildings in it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from notodata.config import CONFIG
from notodata.geometry import build_kdtree

logger = logging.getLogger("notodata.spatial")


def sample_ids(ids, size: int, seed: int) -> np.ndarray:
    """Sorted ids, then `size` drawn without replacement (all if fewer)."""
    ids = np.sort(np.asarray(ids))
    if ids.size <= size:
        logger.info("fewer labelled buildings (%d) than sample size %d: using all", ids.size, size)
        return ids
    return np.random.default_rng(seed).choice(ids, size, replace=False)


def morans_i_bins(xy, y, edges_m) -> pd.DataFrame:
    """Moran's I per (lo, hi] bin; `xy` in metres, `y` the binary label."""
    xy = np.asarray(xy, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.asarray(edges_m, dtype=np.float64)
    if edges[0] != 0.0:
        raise ValueError("first bin edge must be 0")
    n = len(y)
    z = y - y.mean()
    zz = float((z**2).sum())
    tree = build_kdtree(xy)
    s = tree.count_neighbors(tree, edges, weights=(z, z), cumulative=True)
    c = tree.count_neighbors(tree, edges, cumulative=True)
    s, c = np.asarray(s, dtype=np.float64), np.asarray(c, dtype=np.float64)
    # count_neighbors counts d <= r, so the first bin is S(hi) minus the n
    # self-pairs (sum z_i^2 / n); coincident distinct pairs stay in it.
    num = np.diff(s)
    cnt = np.diff(c)
    num[0] = s[1] - zz
    cnt[0] = c[1] - n
    with np.errstate(divide="ignore", invalid="ignore"):
        i = np.where(cnt > 0, n / cnt * num / zz, np.nan)
    return pd.DataFrame(
        {
            "bin_lo_km": edges[:-1] / 1000.0,
            "bin_hi_km": edges[1:] / 1000.0,
            "n": n,
            "W": cnt.astype(np.int64),
            "I": i,
            "expected": -1.0 / (n - 1),
        }
    )


def block_size_km(table: pd.DataFrame) -> float:
    """Upper edge of the first bin with I < threshold, clamped to the L range."""
    thr = CONFIG.moran.threshold
    lo, hi = CONFIG.moran.l_clamp_km
    logger.info("moran bins are (lo, hi] in metres; first bin keeps coincident distinct buildings")
    below = table[table["I"] < thr]
    if below.empty:
        raw = float(table["bin_hi_km"].iloc[-1])
        logger.info("no bin has I below %s: unclamped L = %s km", thr, raw)
    else:
        raw = float(below["bin_hi_km"].iloc[0])
    return float(min(max(raw, lo), hi))


def run(ids, xy, y) -> tuple[pd.DataFrame, float]:
    """Sample, bin and pick L from labelled buildings (ids, projected xy, y)."""
    ids = np.asarray(ids)
    chosen = sample_ids(ids, CONFIG.moran.sample_size, CONFIG.moran.seed)
    idx = pd.Index(ids).get_indexer(chosen)
    edges_m = np.array(CONFIG.moran.bin_edges_km) * 1000.0
    table = morans_i_bins(np.asarray(xy)[idx], np.asarray(y)[idx], edges_m)
    return table, block_size_km(table)
