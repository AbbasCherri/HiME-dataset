import logging
from itertools import pairwise

import numpy as np
import pandas as pd

from notodata import spatial
from notodata.config import CONFIG

EDGES_M = np.array(CONFIG.moran.bin_edges_km) * 1000.0


def _dense(xy, y, edges_m):
    n = len(y)
    z = y - y.mean()
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    off = ~np.eye(n, dtype=bool)
    out = []
    for k, (lo, hi) in enumerate(pairwise(edges_m)):
        # first bin keeps coincident distinct pairs (d == 0)
        w = off & ((d >= lo) if k == 0 else (d > lo)) & (d <= hi)
        W = w.sum()
        num = (w * np.outer(z, z)).sum()
        out.append((W, n / W * num / (z**2).sum() if W else np.nan))
    return out


def test_morans_i_bruteforce():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 25_000, (1500, 2))
    # coincident distinct points exercise the first-bin self-pair subtraction
    xy[1:6] = xy[0]
    # a point exactly on a bin edge exercises (lo, hi]
    xy[10] = xy[11] + [500.0, 0.0]
    y = (rng.random(1500) < 0.3).astype(float)
    got = spatial.morans_i_bins(xy, y, EDGES_M)
    want = _dense(xy, y, EDGES_M)
    assert len(got) == len(want) == len(EDGES_M) - 1
    np.testing.assert_allclose(got["W"], [w for w, _ in want], rtol=0, atol=1e-9)
    np.testing.assert_allclose(got["I"], [i for _, i in want], rtol=0, atol=1e-9)
    assert (got["n"] == 1500).all()
    np.testing.assert_allclose(got["expected"], -1 / 1499)


def _table(values):
    return pd.DataFrame({"bin_hi_km": CONFIG.moran.bin_edges_km[1:], "I": values})


def test_L_rule(caplog):
    # first bin under 0.05 -> 0.5 km, clamped up to 2
    with caplog.at_level(logging.INFO):
        assert spatial.block_size_km(_table([0.01] + [0.5] * 7)) == 2.0
    # first under at 5 km (index 4)
    assert spatial.block_size_km(_table([0.5, 0.4, 0.3, 0.2, 0.04, 0.0, 0, 0])) == 5.0
    # first under at 12 km -> clamped down to 10
    assert spatial.block_size_km(_table([0.5] * 6 + [0.01, 0.0])) == 10.0
    # none below -> 20 unclamped, 10 clamped, case logged
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert spatial.block_size_km(_table([0.5] * 8)) == 10.0
    assert "no bin" in "\n".join(r.message for r in caplog.records)


def test_threshold_edge_not_below():
    assert spatial.block_size_km(_table([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.049])) == 10.0


def test_sample_deterministic_and_small_population_logged(caplog):
    ids = np.array([5, 3, 9, 1, 7, 2])
    a = spatial.sample_ids(ids, 4, seed=0)
    b = spatial.sample_ids(ids[::-1], 4, seed=0)
    np.testing.assert_array_equal(a, b)
    assert len(set(a)) == 4
    with caplog.at_level(logging.INFO):
        c = spatial.sample_ids(ids, 100, seed=0)
    np.testing.assert_array_equal(np.sort(c), np.sort(ids))
    assert "fewer" in "\n".join(r.message for r in caplog.records)


def test_bin_labels_and_columns():
    rng = np.random.default_rng(2)
    xy = rng.uniform(0, 30_000, (200, 2))
    y = (rng.random(200) < 0.5).astype(float)
    t = spatial.morans_i_bins(xy, y, EDGES_M)
    assert list(t.columns) == ["bin_lo_km", "bin_hi_km", "n", "W", "I", "expected"]
    assert t["bin_hi_km"].tolist() == list(CONFIG.moran.bin_edges_km[1:])
